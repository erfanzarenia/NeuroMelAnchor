#!/usr/bin/env python3
"""
================================================================================
NEXUS NeuroMelAnchor 
================================================================================
NeuroMelAnchor is a neuromelnanin centric preprocessing and segmentation pipeline, 
intended for navigating certain challenges associated with masking the substantia
nigra. 

Pipeline Stages
-------------------------------
Stage 1  — Forward Normalisation
    1. Gunzip NM volumes
    2. SPM Realign (motion correction, register to mean)
    3. ANTs N4BiasFieldCorrection on each realigned volume
    4. Compute mean of bias-corrected volumes
    5. Perform ANTs BrainExtraction on T1
    6. Perform ANTs Registration of T1 to MNI (Rigid + Affine + SyN, brainstem-weighted)
    7. Perform ANTs Registration pf NM-contrast mean to T1 (Rigid)
    8. Combine transforms and apply to bring NM to MNI space
    9. (Optional) SPM Smooth
    10. Generates QC plots and motion metrics
    11. Creates Group average NM template
 
Segment — Manual ROI Segmentation (ITK-SNAP)
    Interactive segmentation of Cerebral Penducle (label 1) and Substantia Nigra (label 2) on the
    group-average NM template.  
    Masks are saved to resources/tpl-MNI152NLin2009cAsym/.
 
Stage 2  — Inverse Normalisation and CNR Seed Extraction
    1. Register subject NM mean to group template (Rigid + Affine)
    2. Warp segmented SN and CP masks to native NM space via inverse transforms
    3. Compute CNR map and NM seed mask
    4. Anchor MNI atlas to SN seed mask using ANTs and Centroid data
    4. Compute QC Metrics: DICE and Hausdorff Distance between data-driven seed and supplied atlas mask
            (optional: assess between manual mask if --manual_dir provided)
 
Output Structure (per subject)
-------------------------------
    <output_root>/
    └── sub-XXX/
        ├── anat/          NM volumes (bias-corrected, mean, MNI-space, smoothed)
        ├── transforms/    ANTs composite warps (.h5)
        ├── metrics/       Motion stats JSON + plots
        └── qc/            Registration overlay PNGs, run-comparison PNG
        └── cnr/           Outputs of CNR Steps
Usage
-------------------------------
    python BrainHack_NM_Nipype_Pipeline.py 
        --project_root /path/to/project 
        --design_dir   /path/to/bids_root 
        --output_root  /path/to/outputs 
        --work_dir     /path/to/work 
        --n_procs 10                        defaults to 10, but can set to whatever matchs system
        --stage all                         defaults to all, but can specifiy section
        --test                              test run on the first subject
        --smooth_fwhm                       defaults to 0, but can smooth at specified mm
        --manual_dir  /path/to/manual_mask  for per-subject manual SN masks in Stage 2 QC

- Additional Fixes
    - Add additional command line parameters
    - Need to make it so that if there is only a single NM scan or multiple NM scan, there can adapt either way, instead of the current hard coded nature
    - Copyright logo beside NEXUS
================================================================================
By NEXUS
================================================================================
"""
__version__ = "2.5.0"

# =================================================================
# Import Statements
# =================================================================
import os
import sys
import shutil
import logging
import argparse
import subprocess
from pathlib import Path 
from datetime import datetime

import numpy as np
import nipype as ni
import nibabel as nib
from templateflow.api import get
from IPython.display import Image
from nilearn.image import smooth_img, mean_img

from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import Realign, Smooth, SPMCommand
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from nipype.interfaces.ants import Registration, BrainExtraction, ApplyTransforms, N4BiasFieldCorrection

# =================================================================
# Argument Parsing
# =================================================================

parser = argparse.ArgumentParser(description="NEXUS NeuroMelAnchor")
parser.add_argument("--project_root", required=True, help="Project Root)", type=Path)
parser.add_argument("--output_root", required=True, help="Directory where all outputs will be written", type=Path)
parser.add_argument("--design_dir", required=True, help="BIDs root containing subject directories", type=Path)
parser.add_argument("--work_dir", required=True, help="NiPyPe working / scratch directory", type=Path)
parser.add_argument("--n_procs", default=10, type=int, help="Number of parallel processes (default: 10)")
parser.add_argument("--stage", choices=['1','segment', '2', 'all'], default='all',
                     help="NeuroMelAnchor stages to run (default: all)")
parser.add_argument("--smooth", default=0, type=int, help="Smoothing FWHM Kernal applied after MNI Warp. Set to 0 to skip smoothing entirely (default: 0)")
parser.add_argument("--manual_dir", default=None, type=Path, help="Directory of per-subject manual SN Segmentations (optional for similarity QC during Stage 2)")
parser.add_argument("--test", action="store_true", help="Test Mode: Runs only on ")
args = parser.parse_args()

# =================================================================
# Paths
# =================================================================
project_root = args.project_root
design_dir = args.design_dir
work_dir = args.work_dir
work_dir.mkdir(parents=True, exist_ok=True)
output_base = args.output_root
output_base.mkdir(parents=True, exist_ok=True)
manual_dir = args.manual_dir
# =================================================================
# Logging
# =================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H")
logs = output_base / "logs"
logs.mkdir(parents=True, exist_ok=True)
log_file = output_base / "NeuroMelAnchor_master.log"
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(str(log_file)), logging.StreamHandler(sys.stdout)])

log = logging.getLogger("NeuroMelAnchor")

log.info("=" * 60)
log.info(f"NeuroMelAnchor v{__version__}")
log.info("-" * 60)
log.info(f"Start Time and Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"project_root        : {project_root}")
log.info(f"design_dir          : {design_dir}")
log.info(f"output_base         : {output_base}")
log.info(f"work_dir            : {work_dir}")
log.info(f"n_procs             : {args.n_procs}")
log.info(f"stage               : {args.stage}")
log.info(f"smooth_fwhm         : {args.smooth}")
log.info(f"manual_dir          : {manual_dir}")
log.info(f"test mode           : {args.test}")
log.info("=" * 60)

# =================================================================
# Validation Helper Function
# =================================================================

def check_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found at :\n {path} \n"
                                f"Please ensure the expected directory structure is in place.")

# =================================================================
# Environment and Thread Control
# =================================================================

n_threads = str(min(args.n_procs, 8))
safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': n_threads, 'OMP_NUM_THREADS': n_threads}

if shutil.which('itksnap') is None:
    raise RuntimeError("ITK-SNAP not found on PATH. \n" \
    "Install from https://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4")

if shutil.which('antsRegistration') is None:
    raise RuntimeError("ANTs not found on path. \n" \
    "Install from https://github.com/antsx/ants")

if shutil.which('matlab') is None:
    raise RuntimeError("Matlab not found on path. \n" \
    "Ensure SPM is on your MATLAB Path")
# =================================================================
# Execution Flags
# =================================================================

run_phase1 = args.stage in ['1', 'all']
run_segment = args.stage in ['segment', 'all']
run_phase2 = args.stage in ['2', 'all']
do_smooth = args.smooth > 0

# =================================================================
# Subject
# =================================================================
subjects = sorted([p.name for p in design_dir.glob("sub-*")])
if len(subjects) == 0:
    raise RuntimeError(f"No sub-* directories found in {design_dir.resolve()}\n"
                       f"Contents: {sorted(str(p.name) for p in design_dir.iterdir())[:10]}\n"
                       f"Check that --design_dir flag points to the BIDS root containing sub-* folders.")

if args.test and len(subjects) > 0:
    log.info(f"\n{'='*60}")
    log.info("Test Mode Enabled: Running on first subject only.")
    log.info(f"\n{'='*60}")
    subjects =[subjects[0]]

log.info(f"Found {len(subjects)} subjects(s): {subjects}")
# =================================================================
# Templates and Masks
# =================================================================

t1_template = get('MNI152NLin2009cAsym', resolution=1, suffix='T1w', desc=None, extension='nii.gz')
if isinstance(t1_template, list):
    t1_template = t1_template[0]
else:
    t1_template = t1_template
log.info(f"Whole-Head T1 Candidates, {t1_template}")
t1_brain_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='nii.gz')
if isinstance(t1_brain_template, list):
    t1_brain_template=t1_brain_template[0]
log.info(f"Brain T1 Candidates, {t1_brain_template}")
brain_prob_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='probseg', extension='nii.gz')
if isinstance(brain_prob_template, list):
    brain_prob_template = brain_prob_template[0]
log.info(f"Prob Brain Candidates, {brain_prob_template}")

log.info(f"{'='*60}")
log.info(f"Whole-head T1 template: {t1_template}")
log.info(f"Brain-extracted template: {t1_brain_template}")
log.info(f"Brain probability mask: {brain_prob_template}")
log.info(f"{'='*60}")

mask_dir = project_root / "resources" / "tpl-MNI152NLin2009cAsym"
combined_mask = str(mask_dir /"MNI_Manual_Masks_combined.nii.gz")
SN_mask = str(mask_dir / "MNI_SNc_Manual.nii.gz")
CP_mask = str(mask_dir /"MNI_CP_Manual.nii.gz")
BrainStem_mask = str(mask_dir / "HarvardOxford" / "Brainstem.nii.gz")
atlas_mask_mni = str(mask_dir / "CIT168" / "SNc.nii.gz")

if manual_dir is not None and not manual_dir.exists():
    raise FileNotFoundError(f"--manual_dir specified but does not exist: {manual_dir}")

check_file(BrainStem_mask, "Brainstem Weighting Mask (Required for T1 to MNI registration)")

if run_phase2 and not run_segment:
    check_file(SN_mask, "SN mask missing... \n Run --stage segment first to create it")
    check_file(CP_mask, "CP mask missing... \n Run --stage segment first to create it")

group_average_path = str(output_base / "Study-Specific_NM_Template.nii.gz")
if run_phase2 and not run_phase1:
    check_file(group_average_path, "Group Average NM Template missing.. \n Run --stage 1 first to generate it")

# =================================================================
# Helper Functions
# =================================================================

def compute_motion_params(realignment_parameters):
    import numpy as np
    import os
    import json
    import matplotlib
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print(f"Computing Motion Parameters and Plotting Translation and Rotation values...")
    print(f"\n{'='*60}")
    parameters = np.loadtxt(realignment_parameters)
    if parameters.ndim ==1:
        parameters = parameters[np.newaxis, :]

    trans_range = parameters[:, :3].max(axis=0) - parameters[:, :3].min(axis=0)
    rot_range = parameters[:, 3:].max(axis=0) - parameters[:,3:].min(axis=0)

    stats = {
        'translation_range_mm': {
            'x': float(trans_range[0]),
            'y': float(trans_range[1]),
            'z': float(trans_range[2]),
            'max': float(trans_range.max()),},
        'rotation_range_deg': {
            'pitch': float(np.degrees(rot_range[0])),
            'roll':  float(np.degrees(rot_range[1])),
            'yaw':   float(np.degrees(rot_range[2])),
            'max':   float(np.degrees(rot_range.max()))},
    }
 
    stats_file = os.path.abspath('motion_params.json')
    with open(stats_file, 'w') as f:
        import json
        json.dump(stats, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    fig.suptitle('Realignment Parameters', fontsize=10)

    for i, (lbl, col) in enumerate(zip(['x (mm)', 'y (mm)', 'z (mm)'], ['b', 'y', 'r'])):
        axes[0, i].plot(parameters[:, i], color=col, lw=0.9)
        axes[0, i].set_title(lbl, fontsize=9)
        axes[0, i].axhline(0, color='k', lw=0.4, ls='--', alpha=0.4)

    for i, (lbl, col) in enumerate(zip(['pitch (deg)', 'roll (deg)', 'yaw (deg)'], ['g', 'm', 'c'])):
        axes[1, i].plot(np.degrees(parameters[:, i + 3]), color=col, lw=0.9)
        axes[1, i].set_title(lbl, fontsize=9)
        axes[1, i].axhline(0, color='k', lw=0.4, ls='--', alpha=0.4)

    fig.tight_layout()
    plot_file = os.path.abspath('Motion_Params_Plot.png')
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return stats_file, plot_file

def check_run_similarity(realigned_files):
    import numpy as np
    import nibabel as nib
    import os
    from nilearn.image import mean_img, math_img
    from nilearn.plotting import plot_anat

    print(f"\n{'='*60}")
    print(f"Checking Run Similarity...")
    print(f"\n{'='*60}")

    out_png = os.path.abspath('Run_Difference_QC.png')

    if len(realigned_files) < 2:
        print("Warning: Fewer than 2 runs found, Skipping run similarity comparison.")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, 'Run comparison skipped: only 1 run detected.',
                ha='center', va='center', fontsize=11, color='gray',
                transform=ax.transAxes)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png
    
    half = len(realigned_files) // 2
    r1_img = mean_img(realigned_files[:half])
    r2_img = mean_img(realigned_files[half:])

    r1 = r1_img.get_fdata()
    r2 = r2_img.get_fdata()

    correlation = np.corrcoef(r1.flatten(), r2.flatten())[0,1]

    diff_img = math_img("img1 - img2", img1=r1_img, img2=r2_img)

    out_png = os.path.abspath('Run_Difference_QC.png')
    display = plot_anat(diff_img, title=f'Run 1 - Run 2 (r = {correlation:.3f})', display_mode='ortho', cmap='RdBu_r', draw_cross=False)
    display.savefig(out_png, dpi=150)
    display.close()

    return out_png
 
def combine_transforms(t1_to_mni_composite, nm_to_t1_composite):
    """ 
    ANTs ApplyTransforms applies transforms in REVERSE order
    """
    print(f"\n{'='*60}")
    print(f"Combing Forward Transforms...")
    print(f"\n{'='*60}")
    return [t1_to_mni_composite, nm_to_t1_composite]

def generate_qc_nm(nm_mni_image, t1_template_image):
    import os
    import matplotlib
    import numpy as np
    import nibabel as nib
    matplotlib.use('Agg')
    from nilearn import plotting
    print(f"\n{'='*60}")
    print(f"Computing Registration NM to MNI QC Plot...")
    print(f"\n{'='*60}")

    out_png = os.path.abspath('Registration_NM_to_MNI_QC.png')
    display = plotting.plot_anat(t1_template_image, cut_coords=[0, -15, -12],
                                            display_mode='ortho', cmap='gray',
                                            title="NM to MNI Alignment QC", draw_cross=False, colorbar=False, dim=-0.3)
    nm_data = nib.load(nm_mni_image).get_fdata()
    nm_nonzero = nm_data[nm_data > 0]
    threshold_val = np.percentile(nm_nonzero, 90) if len(nm_nonzero) > 0 else 0
    display.add_overlay(nm_mni_image, cmap='hot', alpha=0.7, colorbar=True, vmin=threshold_val)
    display.savefig(out_png, dpi=150)
    display.close()
    return out_png

def generate_qc_t1(t1_mni_image, t1_template_image):
    import os
    import matplotlib
    import numpy as np
    import nibabel as nib
    matplotlib.use('Agg')
    from nilearn import plotting
    print(f"\n{'='*60}")
    print(f"Computing Registration between Normalized T1 and MNI QC Plot...")
    print(f"\n{'='*60}")

    out_png = os.path.abspath('Registration_norm-T1_to_MNI_QC.png')
    display = plotting.plot_anat(t1_template_image,
                                            display_mode='ortho', dim=-0.5, cmap='gray',
                                            title="T1-norm to MNI Alignment QC", draw_cross=False, colorbar=False)
    display.add_overlay(t1_mni_image, cmap='Reds', alpha=0.8)
    display.savefig(out_png, dpi=300)
    display.close()
    return out_png

def extract_tractography_seed(nm_image, native_sn_mask, native_cp_mask):
    import nibabel as nib
    import numpy as np
    import os
    from scipy.stats import gaussian_kde
    from nilearn.image import smooth_img
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print(f"Computing CNR and Extracting Tractography Seed...")
    print(f"\n{'='*60}")

    nm_image = nib.load(nm_image)
    nm_data = nm_image.get_fdata()
    sn_data = nib.load(native_sn_mask).get_fdata().astype(bool)
    cp_data = nib.load(native_cp_mask).get_fdata().astype(bool)

    if not sn_data.any() or not cp_data.any():
        raise ValueError(f"Empty mask detected. SN voxels: {sn_data.sum()} | CP voxels: {cp_data.sum()}")

    cp_int = nm_data[cp_data]; cp_int = cp_int[cp_int > 0]
    sn_int = nm_data[sn_data]; sn_int = sn_int[sn_int > 0]
    
    x_range = np.linspace(cp_int.min(), cp_int.max(), 1000)
    cp_mode = x_range[np.argmax(gaussian_kde(cp_int)(x_range))]
    if cp_mode <= 0: cp_mode = float(np.mean(cp_int))

    cnr_map = np.zeros_like(nm_data)
    cnr_map[sn_data] = (nm_data[sn_data] - cp_mode) / cp_mode
    cnr_nii = nib.Nifti1Image(cnr_map, nm_image.affine, nm_image.header)

    smoothed_cnr_nii = smooth_img(cnr_nii, fwhm=1.0)
    smoothed_cnr_data = smoothed_cnr_nii.get_fdata()

    final_seed_mask = np.zeros_like(nm_data)
    final_seed_mask[sn_data & (smoothed_cnr_data >= 0.05)] = 1

    cnr_out = os.path.abspath('Subject_CNR_Map_Smoothed.nii.gz')
    seed_out = os.path.abspath('Native_Tractography_Seed_SNc.nii.gz')

    nib.save(smoothed_cnr_nii, cnr_out)
    nib.save(nib.Nifti1Image(final_seed_mask, nm_image.affine, nm_image.header), seed_out)

    # Histogram
    all_vals = np.concatenate([cp_int, sn_int])
    bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cp_int, bins=bins, alpha=0.55, color='b', label='CP (background)', density=True)
    ax.hist(sn_int, bins=bins, alpha=0.55, color='y', label='SN (signal)', density=True)
    ax.axvline(cp_mode, color='m', lw=1.2, ls='--', label=f'CP mode = {cp_mode:.1f}')
    ax.set_title(f'SN vs CP intensities  |  mean SN CNR = {np.mean(smoothed_cnr_data[sn_data]):.3f}', fontsize=9)
    ax.set_xlabel('Voxel intensity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    hist_file = os.path.abspath('CNR_Histogram.png')
    fig.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return cnr_out, seed_out, hist_file

def mean_of_bias_corrected(in_files):
    from nilearn.image import mean_img
    import os
    out = os.path.abspath('mean_NM_bias_corrected.nii.gz')
    mean_img(in_files).to_filename(out)
    return out

def compute_mask_qc(cnr_mask, atlas_mask, anchored_atlas_mask=None, manual_mask=None):
    import os
    import json
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt, binary_erosion
    def _dice(a, b):
        a, b  = a.astype(bool), b.astype(bool)
        denom = a.sum() + b.sum()
        if denom == 0:
            return float('nan')
        return float(2 * (a & b).sum() / denom)
    def _hd95(a, b, spacing):
        a, b = a.astype(bool), b.astype(bool)
        if not a.any() or not b.any():
            return float('nan')
        surf_a = a & ~binary_erosion(a)
        surf_b = b & ~binary_erosion(b)
        dist_a_to_b = distance_transform_edt(~b, sampling=spacing)[surf_a]
        dist_b_to_a = distance_transform_edt(~a, sampling=spacing)[surf_b]
        return float(np.percentile(np.concatenate([dist_a_to_b, dist_b_to_a]), 95))
 
    def _centroid_mm(img, mask):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        return nib.affines.apply_affine(img.affine, coords.mean(axis=0)).tolist()
 
    def _load(path):
        img = nib.load(path)
        return img, img.get_fdata().astype(bool)
 
    cnr_img,   cnr_data   = _load(cnr_mask)
    atlas_img, atlas_data = _load(atlas_mask)
 
    if cnr_img.shape != atlas_img.shape:
        raise ValueError(
            f"Shape mismatch: CNR {cnr_img.shape} vs atlas {atlas_img.shape}.\n"
            "Both masks must be in the same native NM space.")
    if not np.allclose(cnr_img.affine, atlas_img.affine, atol=1e-3):
        raise ValueError("Affine mismatch: CNR and atlas masks not in same space.")
 
    spacing = tuple(float(z) for z in cnr_img.header.get_zooms()[:3])
 
    centroid_cnr   = _centroid_mm(cnr_img,   cnr_data)
    centroid_atlas = _centroid_mm(atlas_img, atlas_data)
    dist_cnr_atlas = None
    if centroid_cnr and centroid_atlas:
        dist_cnr_atlas = float(np.linalg.norm(
            np.array(centroid_cnr) - np.array(centroid_atlas)))
 
    qc = {
        # CNR vs raw atlas
        "n_voxels_cnr":                          int(cnr_data.sum()),
        "n_voxels_atlas":                        int(atlas_data.sum()),
        "diff_voxels_cnr_vs_atlas":              int(cnr_data.sum()) - int(atlas_data.sum()),
        "dice_cnr_vs_atlas":                     _dice(cnr_data, atlas_data),
        "hd95_cnr_vs_atlas_mm":                  _hd95(cnr_data, atlas_data, spacing),
        "centroid_cnr_mm":                       centroid_cnr,
        "centroid_atlas_mm":                     centroid_atlas,
        "centroid_distance_cnr_vs_atlas_mm":     dist_cnr_atlas,
        # CNR vs anchored atlas
        "n_voxels_anchored_atlas":               None,
        "dice_cnr_vs_anchored_atlas":            None,
        "hd95_cnr_vs_anchored_atlas_mm":         None,
        "centroid_anchored_atlas_mm":            None,
        "centroid_distance_cnr_vs_anchored_mm":  None,
        # Manual comparisons
        "n_voxels_manual":                       None,
        "dice_cnr_vs_manual":                    None,
        "hd95_cnr_vs_manual_mm":                 None,
        "dice_atlas_vs_manual":                  None,
        "hd95_atlas_vs_manual_mm":               None,
        "dice_anchored_vs_manual":               None,
        "hd95_anchored_vs_manual_mm":            None,
        "centroid_manual_mm":                    None,
        "centroid_distance_cnr_vs_manual_mm":    None,
        "centroid_distance_atlas_vs_manual_mm":  None,
        "flags": [],
    }
 
    # CNR vs ANCHORED
    if anchored_atlas_mask is not None and str(anchored_atlas_mask).strip():
        anch_img, anch_data = _load(anchored_atlas_mask)
        if cnr_img.shape != anch_img.shape:
            raise ValueError(
                f"Shape mismatch: CNR {cnr_img.shape} vs anchored {anch_img.shape}.")
        if not np.allclose(cnr_img.affine, anch_img.affine, atol=1e-3):
            raise ValueError("Affine mismatch: CNR and anchored atlas not in same space.")
 
        centroid_anch = _centroid_mm(anch_img, anch_data)
        qc["n_voxels_anchored_atlas"]    = int(anch_data.sum())
        qc["dice_cnr_vs_anchored_atlas"] = _dice(cnr_data, anch_data)
        qc["hd95_cnr_vs_anchored_atlas_mm"] = _hd95(cnr_data, anch_data, spacing)
        qc["centroid_anchored_atlas_mm"] = centroid_anch
        if centroid_cnr and centroid_anch:
            qc["centroid_distance_cnr_vs_anchored_mm"] = float(
                np.linalg.norm(np.array(centroid_cnr) - np.array(centroid_anch)))

        raw_dice  = qc["dice_cnr_vs_atlas"]
        anch_dice = qc["dice_cnr_vs_anchored_atlas"]
        if raw_dice is not None and anch_dice is not None and anch_dice < raw_dice - 0.05:
            qc["flags"].append(
                "Anchoring degraded DICE vs CNR seed — check ROI mask and CNR quality")
 
    # Manual
    if manual_mask is not None and str(manual_mask).strip():
        man_img, man_data = _load(manual_mask)
        if cnr_img.shape != man_img.shape:
            raise ValueError(
                f"Shape mismatch: CNR {cnr_img.shape} vs manual {man_img.shape}.")
        if not np.allclose(cnr_img.affine, man_img.affine, atol=1e-3):
            raise ValueError("Affine mismatch: CNR and manual not in same space.")
 
        centroid_man = _centroid_mm(man_img, man_data)
        qc["n_voxels_manual"]        = int(man_data.sum())
        qc["dice_cnr_vs_manual"]     = _dice(cnr_data, man_data)
        qc["hd95_cnr_vs_manual_mm"]  = _hd95(cnr_data, man_data, spacing)
        qc["dice_atlas_vs_manual"]   = _dice(atlas_data, man_data)
        qc["hd95_atlas_vs_manual_mm"] = _hd95(atlas_data, man_data, spacing)
        qc["centroid_manual_mm"]     = centroid_man
 
        if centroid_cnr and centroid_man:
            qc["centroid_distance_cnr_vs_manual_mm"] = float(
                np.linalg.norm(np.array(centroid_cnr) - np.array(centroid_man)))
        if centroid_atlas and centroid_man:
            qc["centroid_distance_atlas_vs_manual_mm"] = float(
                np.linalg.norm(np.array(centroid_atlas) - np.array(centroid_man)))
 
        # Anchored vs manual
        if anchored_atlas_mask is not None and str(anchored_atlas_mask).strip():
            qc["dice_anchored_vs_manual"]    = _dice(anch_data, man_data)
            qc["hd95_anchored_vs_manual_mm"] = _hd95(anch_data, man_data, spacing)
 
        # Flags
        if qc["dice_cnr_vs_manual"] is not None and qc["dice_cnr_vs_manual"] < 0.5:
            qc["flags"].append(
                "Low DICE CNR vs manual — automated mask diverges from expert")
        if qc["dice_atlas_vs_manual"] is not None and qc["dice_atlas_vs_manual"] < 0.5:
            qc["flags"].append(
                "Low DICE atlas vs manual — registration or atlas fit may be poor")
        if qc["hd95_cnr_vs_manual_mm"] is not None and qc["hd95_cnr_vs_manual_mm"] > 5.0:
            qc["flags"].append("High HD95 CNR vs manual (>5 mm)")
        if (qc["centroid_distance_cnr_vs_manual_mm"] is not None and
                qc["centroid_distance_cnr_vs_manual_mm"] > 3.0):
            qc["flags"].append("Large centroid offset CNR vs manual (>3 mm)")
 
        # Did anchoring improve vs manual?
        a_vs_m   = qc.get("dice_anchored_vs_manual")
        raw_vs_m = qc.get("dice_atlas_vs_manual")
        if a_vs_m is not None and raw_vs_m is not None and a_vs_m < raw_vs_m - 0.05:
            qc["flags"].append(
                "Anchoring reduced DICE vs manual — CNR seed may be poorly defined")
 
    # Flags always applied
    if qc["n_voxels_cnr"] < 20:
        qc["flags"].append(
            "CNR seed very small (<20 voxels) — check CNR map and SN registration")
    if qc["dice_cnr_vs_atlas"] is not None and qc["dice_cnr_vs_atlas"] < 0.4:
        qc["flags"].append("Low DICE CNR vs atlas (<0.4) — possible registration failure")
    if qc["hd95_cnr_vs_atlas_mm"] is not None and qc["hd95_cnr_vs_atlas_mm"] > 5.0:
        qc["flags"].append("High HD95 CNR vs atlas (>5 mm)")
    if dist_cnr_atlas is not None and dist_cnr_atlas > 3.0:
        qc["flags"].append("Large centroid offset CNR vs atlas (>3 mm)")
 
    qc_file = os.path.abspath("SN_Mask_QC.json")
    with open(qc_file, "w") as f:
        json.dump(qc, f, indent=2)
 
    return qc_file
 

def build_sn_roi_mask(cnr_seed, dilation_mm=12.0):
    import os
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import binary_dilation

    img     = nib.load(cnr_seed)
    data    = img.get_fdata().astype(bool)
    spacing = np.array(img.header.get_zooms()[:3])

    radius_vox = int(np.ceil(dilation_mm / spacing.min()))

    dilated = binary_dilation(data, iterations=radius_vox)

    out = os.path.abspath('SN_ROI_Mask.nii.gz')
    nib.save(nib.Nifti1Image(dilated.astype(np.uint8), img.affine, img.header), out)
    return out

# =================================================================
# Phase 1: Forward Normalization
# =================================================================
if run_phase1:

    wf1 = Workflow(name="NeuroMelAnchor_Phase-1")
    wf1.base_dir = str(work_dir)
    wf1.config['execution']['crashdump_dir'] = str(output_base / 'crash_logs')
    wf1.config['execution']['stop_on_first_crash'] = False

    inputnode = Node(IdentityInterface(fields=['subject_id']), name="NM_Subject_IDs")
    inputnode.iterables = [('subject_id', subjects)]

    NM_Data = {'runs':'{subject_id}/anat/{subject_id}_acq-CombEchoNM_run-*_GRE.nii.gz'}
    SelectNM = Node(SelectFiles(NM_Data, base_directory=str(design_dir)), name='SelectNM')

    T1_Data = {'T1': '{subject_id}/anat/{subject_id}_acq-T1w.nii.gz'}
    SelectT1 = Node(SelectFiles(T1_Data, base_directory=str(design_dir)), name='SelectT1')

    # =================================================================
    # Utility Nodes
    # =================================================================

    gunzip = MapNode(Gunzip(), name='Gunzip_NM', iterfield=['in_file'])

    # =================================================================
    # Realign (SPM)
    # =================================================================

    realign = Node(Realign(), name="NM_Realign")
    realign.inputs.register_to_mean = True
    realign.inputs.fwhm = 5
    realign.inputs.quality = 0.9
    realign.inputs.interp = 2

    # =================================================================
    # BiasCorrection
    # =================================================================
    bias = MapNode(N4BiasFieldCorrection(dimension = 3), name='NM_BiasCorr', iterfield=['input_image'])
    bias.inputs.copy_header = True

    # =================================================================
    # BiasCorrection Mean
    # =================================================================

    bias_mean_node = Node(Function(input_names=['in_files'], output_names=['mean_image'], function=mean_of_bias_corrected), name='NM_BiasCorrected_Mean')

    # =================================================================
    # Motion Parameters
    # =================================================================

    motion_node = MapNode(Function(input_names=['realignment_parameters'], 
                            output_names=['stats_file', 'plot_file'],
                            function=compute_motion_params), name='Motion_Params', iterfield=['realignment_parameters'])

    # =================================================================
    # Compare Runs
    # =================================================================
    run_compare = Node(Function(input_names=['realigned_files'], 
                                output_names=['out_png'],
                                function=check_run_similarity), name='Run_Comparison_QC')

    # =================================================================
    # Brain Extraction (ANTS)
    # =================================================================

    brainextraction = Node(BrainExtraction(dimension=3), name='T1_BrainExtraction')
    brainextraction.inputs.brain_template=str(t1_template)
    brainextraction.inputs.brain_probability_mask=str(brain_prob_template)
    brainextraction.inputs.out_prefix = "brain_"
    brainextraction.inputs.environ = safe_env  

    # =================================================================
    # Registration (T1 → MNI)
    # =================================================================

    t1_to_mni = Node(Registration(), name='T1_to_MNI')
    t1_to_mni.inputs.environ = safe_env
    t1_to_mni.inputs.dimension = 3
    t1_to_mni.inputs.interpolation = 'LanczosWindowedSinc'
    t1_to_mni.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    t1_to_mni.inputs.transform_parameters = [(0.05,), (0.08,), (0.1, 3.0, 0.0)]
    t1_to_mni.inputs.fixed_image_masks = ['NULL', 'NULL', str(BrainStem_mask)]
    t1_to_mni.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                            [1000, 500, 250, 100],
                                            [100, 70, 50, 20]]
    t1_to_mni.inputs.metric = ['Mattes', 'Mattes', 'CC']
    t1_to_mni.inputs.metric_weight = [1, 1, 1]
    t1_to_mni.inputs.radius_or_number_of_bins = [64, 64, 4]
    t1_to_mni.inputs.shrink_factors = [[8, 4, 2, 1]]*3
    t1_to_mni.inputs.smoothing_sigmas = [[3, 2, 1, 0]]*3
    t1_to_mni.inputs.sigma_units = ['vox']*3
    t1_to_mni.inputs.sampling_percentage = [0.25, 0.25, 1]
    t1_to_mni.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
    t1_to_mni.inputs.convergence_threshold = [1e-6]*3
    t1_to_mni.inputs.convergence_window_size = [20, 20, 10]
    t1_to_mni.inputs.winsorize_lower_quantile = 0.005
    t1_to_mni.inputs.winsorize_upper_quantile = 0.995
    t1_to_mni.inputs.use_histogram_matching = [False, False, True]

    t1_to_mni.inputs.output_warped_image = True
    t1_to_mni.inputs.write_composite_transform = True
    t1_to_mni.inputs.collapse_output_transforms = True
    t1_to_mni.inputs.fixed_image = str(t1_brain_template)

    # =================================================================
    # Registration (NM → T1) 
    # =================================================================

    nm_to_t1 = Node(Registration(), name='NM_to_T1')
    nm_to_t1.inputs.dimension = 3
    nm_to_t1.inputs.interpolation = 'LanczosWindowedSinc'
    nm_to_t1.inputs.transforms = ['Rigid']
    nm_to_t1.inputs.transform_parameters = [(0.05,)]
    nm_to_t1.inputs.number_of_iterations = [[1000, 500, 250, 100]]
    nm_to_t1.inputs.metric = ['Mattes']
    nm_to_t1.inputs.metric_weight = [1]
    nm_to_t1.inputs.radius_or_number_of_bins = [64]
    nm_to_t1.inputs.shrink_factors = [[8, 4, 2, 1]]
    nm_to_t1.inputs.smoothing_sigmas = [[3, 2, 1, 0]]
    nm_to_t1.inputs.sigma_units = ['vox']
    nm_to_t1.inputs.sampling_percentage = [0.25]
    nm_to_t1.inputs.sampling_strategy = ['Regular']
    nm_to_t1.inputs.convergence_threshold = [1e-6]
    nm_to_t1.inputs.convergence_window_size = [20]
    nm_to_t1.inputs.winsorize_lower_quantile = 0.005
    nm_to_t1.inputs.winsorize_upper_quantile = 0.995
    nm_to_t1.inputs.use_histogram_matching = [False]

    nm_to_t1.inputs.output_warped_image = True
    nm_to_t1.inputs.write_composite_transform = True
    nm_to_t1.inputs.collapse_output_transforms = True
    nm_to_t1.inputs.environ = safe_env  

    # =================================================================
    # Combine Transformations
    # =================================================================

    combine_forward = Node(Function(input_names=['t1_to_mni_composite', 'nm_to_t1_composite'], 
                                output_names=['combined_transforms'], 
                                function=combine_transforms), name='CombineForwardTransforms')

    # =================================================================
    # Apply Transformations
    # =================================================================

    nm_to_mni = Node(ApplyTransforms(), name='NM_to_MNI')
    nm_to_mni.inputs.dimension = 3
    nm_to_mni.inputs.interpolation = 'LanczosWindowedSinc'
    nm_to_mni.inputs.reference_image = str(t1_brain_template)
    nm_to_mni.inputs.output_image = 'NM_MNI.nii'
    nm_to_mni.inputs.environ = safe_env  

    # =================================================================
    # NM-QC Plots
    # =================================================================

    qc_node_nm = Node(Function(input_names=['nm_mni_image', 't1_template_image'], output_names=['out_png'], function=generate_qc_nm), name='Visual_QC_NM')
    qc_node_nm.inputs.t1_template_image = str(t1_brain_template)

    qc_node_t1 = Node(Function(input_names=['t1_mni_image', 't1_template_image'], output_names=['out_png'], function=generate_qc_t1), name='Visual_QC_T1')
    qc_node_t1.inputs.t1_template_image = str(t1_brain_template)

    # =================================================================
    # Smooth
    # =================================================================
    if do_smooth:
        log.info(f"Smoothing enabled: FWHM = {args.smooth}")
        smooth = Node(Smooth(), name='NM_Smooth')
        smooth.inputs.fwhm=[args.smooth] * 3
    else:
        log.info("Smoothing disabled (--smooth 0)")

    # =================================================================
    # Save Final Output
    # -----------------------------------------------------------------
    # Saved outputs and where they end up in the output directory:
    # NM_MotPar/            - SPM realignment parameter files (.txt, one per run)
    # NM_MotionStats/       - Per-run motion stats JSON
    # NM_MotionPlot/        - Per-run motion parameter PNGs
    # QC_RunComparison/     - Run 1 vs Run 2 difference image
    # NM_BiasCorr/          - Bias-corrected NM volumes (post-realignment, per volume)
    # NM_BiasCorr_Mean/     - Mean of bias-corrected volumes (used by Phase 2)
    # NM_Realigned_Mean/    - SPM mean image (pre-bias-correction, QC only)
    # NM_to_MNI/            - Unsmoothed NM bias-corrected mean in MNI space
    # NM_Smooth/            - Smoothed NM mean in MNI space
    # QC_NM_Plot/           - NM-to-MNI overlay PNG
    # QC_T1_Plot/           - T1-to-MNI overlay PNG
    # T1_to_MNI_Composite/  - Composite forward warp (T1 to MNI, .h5)
    # T1_to_MNI_InvComposite/ - Composite inverse warp (MNI to T1, .h5, used in Phase 2)
    # NM_to_T1_Composite/   - Composite forward warp (NM mean to T1, .h5)
    # NM_to_T1_InvComposite/ - Composite inverse warp (T1 to NM, .h5, used in Phase 2)
    # =================================================================

    datasink1 = Node(DataSink(base_directory=str(output_base)), name='PPData')
    datasink1.inputs.parameterization = False

    datasink1.inputs.substitutions = [('_subject_id_', ''),
                                      ('_Motion_Params0/', ''),
                                      ('_NM_BiasCorr0/', ''),
                                      ('_NM_BiasCorr1/', '')]

    # =================================================================
    # Workflow
    # =================================================================

    connections = [
        (inputnode, SelectNM, [('subject_id', 'subject_id')]),
        (inputnode, SelectT1, [('subject_id', 'subject_id')]),
        (inputnode, datasink1, [('subject_id', 'container')]),

        # NM Path: Merges, Unzipped, Realigned, Compute FD, and Perform Bias Correction
        (SelectNM, gunzip, [('runs', 'in_file')]),
        (gunzip, realign, [('out_file', 'in_files')]), 
        (realign, bias, [('realigned_files', 'input_image')]),
        (bias, bias_mean_node, [('output_image', 'in_files')]),
        (realign, motion_node, [('realignment_parameters', 'realignment_parameters')]),
        (realign, run_compare, [('realigned_files', 'realigned_files')]),

    
        # T1 Path: Brain Extraction and T1 to MNI Registration
        (SelectT1, brainextraction, [('T1', 'anatomical_image')]),
        (brainextraction, t1_to_mni, [('BrainExtractionBrain', 'moving_image')]),

        # NM-T1 Path: Fixed to T1, using Brain Mask, Moving image is bias corrected NM
        (SelectT1, nm_to_t1, [('T1', 'fixed_image')]),
        (brainextraction, nm_to_t1, [('BrainExtractionMask', 'fixed_image_masks')]),
        (bias_mean_node, nm_to_t1, [('mean_image', 'moving_image')]), 
        
        (t1_to_mni, combine_forward, [('composite_transform', 't1_to_mni_composite')]), 
        (nm_to_t1, combine_forward, [('composite_transform', 'nm_to_t1_composite')]), 
        (combine_forward, nm_to_mni, [('combined_transforms', 'transforms')]),
        (bias_mean_node, nm_to_mni, [('mean_image', 'input_image')]),

        # QC and Smooth
        (nm_to_mni, qc_node_nm, [('output_image', 'nm_mni_image')]),
        (t1_to_mni, qc_node_t1, [('warped_image', 't1_mni_image')]),

        # Metrics
        (realign, datasink1, [('realignment_parameters', 'metrics.@NM_MotPar')]),
        (motion_node, datasink1, [('stats_file', 'metrics.@NM_MotionStats'),
                                ('plot_file', 'metrics.@NM_MotionPlot')]),

        # QC
        (run_compare, datasink1, [('out_png', 'qc.@QC_RunComparison')]),
        (qc_node_nm, datasink1, [('out_png', 'qc.@QC_NM_Plot')]),
        (qc_node_t1, datasink1, [('out_png', 'qc.@QC_T1_Plot')]),

        # Anat
        (bias, datasink1, [('output_image', 'anat.@NM_BiasCorr')]),
        (bias_mean_node, datasink1, [('mean_image', 'anat.@NM_BiasCorr_Mean')]),
        (realign, datasink1, [('mean_image', 'anat.@NM_Realigned_Mean')]),
        (nm_to_mni, datasink1, [('output_image', 'anat.@NM_to_MNI')]),

        # Transforms
        (t1_to_mni, datasink1, [('composite_transform', 'transforms.@T1_to_MNI_Composite'),
                                ('inverse_composite_transform', 'transforms.@T1_to_MNI_InvComposite')]),
        (nm_to_t1, datasink1, [('composite_transform', 'transforms.@NM_to_T1_Composite'),
                            ('inverse_composite_transform', 'transforms.@NM_to_T1_InvComposite')]),]

    if do_smooth:
        connections += [
            (nm_to_mni, smooth, [('output_image', 'in_files')]),
            (smooth, datasink1, [('smoothed_files', 'anat.@nm_mni_smooth')]),
        ]

    wf1.connect(connections)

    # =================================================================
    # Save DAG and Run Phase 1
    # =================================================================

    log.info(f"{'='*60}")
    log.info("Saving DAG of Job...")
    log.info(f"{'='*60}")

    wf1.write_graph(graph2use='colored', dotfilename=str(output_base / 'NeuroMelAnchor_Phase-1_Dag'), format='png', simple_form=True)

    log.info(f"{'='*60}")
    log.info("Starting Phase 1: Forward Normalization...")
    log.info(f"{'='*60}")

    try:
        wf1.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})
    except RuntimeError as e:
        log.info(f"{'='*60}")
        log.warning(f"Phase 1 finished with errors: {e}")
        log.warning("Proceeding to build the group average using the successful participants")
        log.info(f"{'='*60}")

    log.info(f"{'='*60}")
    log.info("Creating Group Average...")
    log.info(f"{'='*60}")

    if do_smooth:
        mni_files = list(output_base.glob('sub-*/anat/sNM_MNI.nii'))
        if not mni_files:
            raise FileNotFoundError("No smoothed NM_MNI files found under output_base/sub-*/anat. \n " \
            "Check that Phase 1 completed successfully for at least one subject")
    else:
        mni_files = list(output_base.glob('sub-*/anat/NM_MNI.nii'))
        if not mni_files:
            raise FileNotFoundError("No NM_MNI files found under output_base/sub-*/anat. \n " \
            "Check that Phase 1 completed successfully for at least one subject")

    log.info(f" Using {len(mni_files)} subject(s) for group average")
    group_average_path = f"{output_base}/Study-Specific_NM_Template.nii.gz"
    mean_img([str(f) for f in mni_files]).to_filename(group_average_path)
    log.info(f"Group Average saved: {group_average_path}")

# =================================================================
# ITK-SNAP Interactive Segmentation
# =================================================================
if run_segment:
    if not os.path.exists(combined_mask):
        log.info(f"Generating blank placeholder mask at {combined_mask}")
        t1_img = nib.load(str(t1_brain_template))
        placeholder = np.zeros(t1_img.shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(placeholder, t1_img.affine, t1_img.header), combined_mask)
    else:
        log.info(f"Existing mask found at {combined_mask}. Preserving Data")

    template_file = project_root / "workflow" / "workspace" / "NEXUS_NM_Template_ITKsnap.xml"
    with open(template_file, 'r') as f:
        xml_data = f.read()

    xml_data = xml_data.replace("PROJECT_ROOT", str(project_root))
    xml_data = xml_data.replace("TARGET_T1_PATH", str(t1_brain_template))
    xml_data = xml_data.replace("TARGET_NM_PATH", group_average_path)
    xml_data = xml_data.replace("TARGET_MASK_PATH", combined_mask)

    workspace_path = project_root / "workflow" / "workspace" / "Automated_Setup.itksnap"
    with open(workspace_path, 'w') as f:
        f.write(xml_data)

    log.info(f"{'='*60}")
    log.info("Manaul Segmentation Required...")
    log.info("1. Cerebral Penducle (CP) - Background reference Region")
    log.info("2. Substantia Nigra (SN) - NM signal region")
    log.info("1. Jump to Midbrain : Type  98  109   61 in the 'Cursor Position' box (center-left) and hit Enter.")
    log.info("2. Center Cameras   : Click 'zoom to fit' below the image")
    log.info("3. Zoom In          : Below 'Cursor Inspector', click Magnifcation Symbol and zoom into MidBrain.")
    log.info(f"{'-'*60}")
    log.info("Save your segmentation (Ctrl + S), then close the ITK-SNAP window to resume the pipeline.")
    log.info(f"{'='*60}")
    subprocess.run(['itksnap', '-w', workspace_path])

    combo_img = nib.load(combined_mask)
    combo_data = combo_img.get_fdata()
    if np.max(combo_data) == 0:
        raise ValueError("Segmentation Mask is Empty - re-run and save the labels.")

    nib.save(nib.Nifti1Image((combo_data == 1).astype(float), combo_img.affine, combo_img.header), CP_mask)
    nib.save(nib.Nifti1Image((combo_data == 2).astype(float), combo_img.affine, combo_img.header), SN_mask)

    CP_n = int((combo_data == 1).sum())
    SN_n = int((combo_data == 2).sum())
    log.info(f"{'='*60}")
    log.info(f"CP Voxels: {CP_n}  |  SN_voxels: {SN_n}")
    if CP_n < 50 or SN_n < 50:
        raise ValueError("One or both masks have very few voxels. \n You may want to inspect segmentation before continuing.")
    log.info(f"{'='*60}")

# =================================================================
# Phase 2: Inverse Normalization and Seeds
# =================================================================
if run_phase2:

    wf2 = Workflow(name="NeuroMelAnchor_Phase2")
    wf2.base_dir = str(work_dir)

    inputnode2 = Node(IdentityInterface(fields=['subject_id']), name="NM_Subject_IDs")
    inputnode2.iterables = [('subject_id', subjects)]

    # =================================================================
    # Grab Intermediate Files from Phase 1
    # =================================================================

    GrabPhase1 = Node(SelectFiles({
        'bias_nm': '{subject_id}/anat/mean_NM_bias_corrected.nii.gz',
    }, base_directory=str(output_base)), name='GrabPhase1')

    # =================================================================
    # Registration (NM → MNI) 
    # =================================================================

    nm_to_template = Node(Registration(), name='NM_to_Group_Template')
    nm_to_template.inputs.fixed_image = group_average_path
    nm_to_template.inputs.dimension = 3
    nm_to_template.inputs.interpolation = 'LanczosWindowedSinc'
    nm_to_template.inputs.transforms = ['Rigid', 'Affine']
    nm_to_template.inputs.transform_parameters = [(0.1,), (0.1,)]
    nm_to_template.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                            [1000, 500, 250, 100]]
    nm_to_template.inputs.metric = ['Mattes', 'Mattes']
    nm_to_template.inputs.metric_weight = [1, 1]
    nm_to_template.inputs.radius_or_number_of_bins = [64, 64]
    nm_to_template.inputs.shrink_factors = [[8, 4, 2, 1],
                                      [8,4,2,1]]
    nm_to_template.inputs.smoothing_sigmas = [[3, 2, 1, 0],
                                        [3,2,1,0]]
    nm_to_template.inputs.sigma_units = ['vox', 'vox']
    nm_to_template.inputs.sampling_percentage = [0.25, 0.25]
    nm_to_template.inputs.sampling_strategy = ['Regular', 'Regular']
    nm_to_template.inputs.convergence_threshold = [1e-6, 1e-6]
    nm_to_template.inputs.convergence_window_size = [20, 20]
    nm_to_template.inputs.winsorize_lower_quantile = 0.005
    nm_to_template.inputs.winsorize_upper_quantile = 0.995
    nm_to_template.inputs.use_histogram_matching = [True, True]

    nm_to_template.inputs.output_warped_image = True
    nm_to_template.inputs.write_composite_transform = True
    nm_to_template.inputs.collapse_output_transforms = True
    nm_to_template.inputs.initial_moving_transform_com = 1
    nm_to_template.inputs.environ = safe_env  

    # =================================================================
    # Transform Masks to Subject Space
    # =================================================================

    mni_to_nm_sn = Node(ApplyTransforms(), name='MNI_to_NM_SN')
    mni_to_nm_sn.inputs.dimension = 3
    mni_to_nm_sn.inputs.interpolation = 'NearestNeighbor'
    mni_to_nm_sn.inputs.input_image = SN_mask
    mni_to_nm_sn.inputs.environ = safe_env  

    mni_to_nm_cp = Node(ApplyTransforms(), name='MNI_to_NM_CP')
    mni_to_nm_cp.inputs.dimension = 3
    mni_to_nm_cp.inputs.interpolation = 'NearestNeighbor'
    mni_to_nm_cp.inputs.input_image = CP_mask
    mni_to_nm_cp.inputs.environ = safe_env  

    mni_to_nm_atlas = Node(ApplyTransforms(), name="MNI_Atlas_to_Native")
    mni_to_nm_atlas.inputs.dimension=3
    mni_to_nm_atlas.inputs.interpolation = 'NearestNeighbor'
    mni_to_nm_atlas.inputs.input_image=atlas_mask_mni
    mni_to_nm_atlas.inputs.environ = safe_env

    # =================================================================
    # Compute CNR + Tractography Seed
    # =================================================================
    cnr_seed = Node(Function(input_names=['nm_image', 'native_sn_mask', 'native_cp_mask'], 
                            output_names=['cnr_out', 'seed_out', 'histogram'], function=extract_tractography_seed), name='CNR_Seed')
    
    # =================================================================
    # Perform Mask Dilation
    # =================================================================

    roi_mask_node = Node(Function(input_names=['cnr_seed', 'dilation_mm'], output_names=['roi_mask'],
             function=build_sn_roi_mask), name='Build_SN_ROI')
    roi_mask_node.inputs.dilation_mm = 12.0

    # =================================================================
    # Anchor Atlas Mask to SN
    # =================================================================

    anchor_atlas = Node(Registration(), name='Anchor_CIT168_to_NM')
    anchor_atlas.inputs.environ = safe_env
    anchor_atlas.inputs.dimension = 3
    anchor_atlas.inputs.interpolation = 'NearestNeighbor'
    anchor_atlas.inputs.transforms = ['Rigid', 'SyN']
    anchor_atlas.inputs.transform_parameters = [(0.05,), (0.1, 3.0, 0.0)]
    anchor_atlas.inputs.number_of_iterations = [[500, 250, 100],
                                            [50, 25, 10]]
    anchor_atlas.inputs.metric = ['Mattes', 'CC']
    anchor_atlas.inputs.metric_weight = [1, 1]
    anchor_atlas.inputs.radius_or_number_of_bins = [32, 4]
    anchor_atlas.inputs.shrink_factors = [[4, 2, 1]]*2
    anchor_atlas.inputs.smoothing_sigmas = [[2, 1, 0]]*2
    anchor_atlas.inputs.sigma_units = ['vox']*2
    anchor_atlas.inputs.sampling_percentage = [0.5, 1]
    anchor_atlas.inputs.sampling_strategy = ['Regular', 'None']
    anchor_atlas.inputs.convergence_threshold = [1e-6]*2
    anchor_atlas.inputs.convergence_window_size = [20, 10]
    anchor_atlas.inputs.winsorize_lower_quantile = 0.005
    anchor_atlas.inputs.winsorize_upper_quantile = 0.995
    anchor_atlas.inputs.use_histogram_matching = [True, True]
    anchor_atlas.inputs.initial_moving_transform_com = 1

    anchor_atlas.inputs.output_warped_image = True
    anchor_atlas.inputs.write_composite_transform = True
    anchor_atlas.inputs.collapse_output_transforms = True
    anchor_atlas.inputs.fixed_image = str(t1_brain_template)
    
    # =================================================================
    # QC
    # =================================================================

    qc_node = Node(Function(input_names=['cnr_mask', 'atlas_mask', 'anchored_atlas_mask', 'manual_mask'], output_names=['qc_file'], function=compute_mask_qc), name='QC_Metrics')

    if manual_dir is not None:
        ManualData = {'manual_sn': '{subject_id}/anat/{subject_id}_space-NM_label-SN_desc-manual_mask.nii'}
        SelectManual = Node(SelectFiles(ManualData, base_directory=str(manual_dir)), name='SelectManual')
    else:
        qc_node.inputs.manual_mask = None

    # =================================================================
    # Datasink Phase 2
    # =================================================================

    datasink2 = Node(DataSink(base_directory=str(output_base), parameterization=False), name='Sink_Phase2')
    datasink2.inputs.substitutions = [('_subject_id_', '')]

    # =================================================================
    # Workflow Connect Phase 2
    # =================================================================

    wf2_connections = [
        (inputnode2, GrabPhase1, [('subject_id', 'subject_id')]),
        (inputnode2, datasink2, [('subject_id', 'container')]),

        (GrabPhase1, nm_to_template, [('bias_nm', 'moving_image')]),

        (nm_to_template, mni_to_nm_sn, [('inverse_composite_transform', 'transforms')]),
        (GrabPhase1, mni_to_nm_sn, [('bias_nm', 'reference_image')]),
        
        (nm_to_template, mni_to_nm_cp, [('inverse_composite_transform', 'transforms')]),
        (GrabPhase1, mni_to_nm_cp, [('bias_nm', 'reference_image')]),

        (nm_to_template, mni_to_nm_atlas, [('inverse_composite_transform', 'transforms')]),
        (GrabPhase1, mni_to_nm_atlas, [('bias_nm', 'reference_image')]),


        (GrabPhase1, cnr_seed, [('bias_nm', 'nm_image')]),
        (mni_to_nm_sn, cnr_seed, [('output_image', 'native_sn_mask')]),
        (mni_to_nm_cp, cnr_seed, [('output_image', 'native_cp_mask')]),

        (cnr_seed, qc_node, [('seed_out','cnr_mask')]), 
        (mni_to_nm_atlas, qc_node, [('output_image', 'atlas_mask')]),

        (cnr_seed, roi_mask_node, [('seed_out', 'cnr_seed')]),

        (GrabPhase1,      anchor_atlas, [('bias_nm',       'fixed_image')]),
        (mni_to_nm_atlas, anchor_atlas, [('output_image',  'moving_image')]),
        (roi_mask_node,   anchor_atlas, [('roi_mask',       'fixed_image_masks')]),

        (anchor_atlas,    qc_node, [('warped_image',  'anchored_atlas_mask')]),

        (anchor_atlas, datasink2, [('warped_image',                'anat.@CIT168_Anchored'),
                                   ('composite_transform',         'transforms.@anchor_fwd'),
                                 ('inverse_composite_transform', 'transforms.@anchor_inv'),
        ]),
        (roi_mask_node, datasink2, [('roi_mask', 'anat.@SN_ROI_Mask')]),
        
        (mni_to_nm_sn, datasink2, [('output_image', 'anat.@Native_SN_Mask')]),
        (mni_to_nm_cp, datasink2, [('output_image', 'anat.@Native_CP_Mask')]),
        (mni_to_nm_atlas, datasink2, [('output_image', 'anat.@Native_Atlas_Mask')]),
        (cnr_seed, datasink2, [('cnr_out', 'cnr.@CNR_Map'),
                            ('seed_out', 'anat.@NM_Defined_Mask'),
                            ('histogram', 'qc.@CNR_Histogram')]),
        (qc_node, datasink2, [('qc_file', 'cnr.@SN_Mask_Dice_QC')]),
    ]

    if manual_dir is not None:
        wf2_connections += [
            (inputnode2, SelectManual, [('subject_id', 'subject_id')]),
            (SelectManual, qc_node, [('manual_sn', 'manual_mask')])
        ]
    wf2.connect(wf2_connections)

    # =================================================================
    # Run Phase 2
    # =================================================================
    log.info(f"{'='*60}")
    log.info("Starting Phase 2: Inverse Transformation and Seed Extraction...")
    log.info(f"{'='*60}")

    log.info(f"{'='*60}")
    log.info("Saving DAG of Job...")
    log.info(f"{'='*60}")

    wf2.write_graph(graph2use='colored', dotfilename=str(output_base / 'NeuroMelAnchor_Pipeline_Dag'), format='png', simple_form=True)

    wf2.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})

# =================================================================
# Group QC Plotting
# =================================================================
log.info("=" * 60)
log.info("Generating Group QC Plots...")
log.info("=" * 60)

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

output_path = Path(output_base)
qc_files = sorted(output_path.glob('sub-*/cnr/SN_Mask_QC.json'))

if not qc_files:
    log.warning("No QC JSON files found — skipping group QC plots.")
else:
    # Collect Metrics
    records = []
    for f in qc_files:
        with open(f, 'r') as fh:
            d = json.load(fh)
            d['subject'] = f.parts[-3]
            records.append(d)

    def extract(key):
        return [r.get(key) for r in records]

    subjects = extract('subject')
    has_manual = any(v is not None for v in extract('dice_cnr_vs_manual'))
    has_anchored = any(v is not None for v in extract('dice_cnr_vs_anchored_atlas'))

    # DICE PLOT
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    plot_data = [
        ("CNR vs Atlas", extract('dice_cnr_vs_atlas'), 'steelblue')
    ]
    if has_anchored:
        plot_data.append(("CNR vs Anchored", extract('dice_cnr_vs_anchored_atlas'), 'darkorange'))
    if has_manual:
        plot_data.append(("CNR vs Manual", extract('dice_cnr_vs_manual'), 'seagreen'))
    if has_anchored and has_manual:
        plot_data.append(("Anchored vs Manual", extract('dice_anchored_vs_manual'), 'mediumpurple'))
        plot_data.append(("Atlas vs Manual", extract('dice_atlas_vs_manual'), 'indianred'))

    positions = range(1, len(plot_data) + 1)
    clean_data = [[v for v in group[1] if v is not None] for group in plot_data]
    
    parts = ax1.violinplot(clean_data, positions=positions, showmeans=True, showextrema=False)

    for pc, pos, group in zip(parts['bodies'], positions, plot_data):
        pc.set_facecolor(group[2])
        pc.set_alpha(0.4)
        
        jitter = np.random.uniform(-0.05, 0.05, size=len(clean_data[pos-1]))
        ax1.scatter(pos + jitter, clean_data[pos-1], color=group[2], s=25, alpha=0.8, edgecolor='white')

    ax1.axhline(0.5, color='firebrick', ls='--', alpha=0.5)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([group[0] for group in plot_data])
    ax1.set_ylabel('DICE Coefficient')
    ax1.set_title('SN Mask Overlap — DICE', fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)
    
    out1 = output_path / 'Group_QC_DICE_Summary.png'
    fig1.savefig(str(out1), dpi=200, bbox_inches='tight')
    plt.close(fig1)
    log.info(f"DICE plot saved: {out1}")

    #HD95 Plot
    hd95_atlas = extract('hd95_cnr_vs_atlas_mm')
    valid_subs = [(i, subj, val) for i, (subj, val) in enumerate(zip(subjects, hd95_atlas)) if val is not None]
    
    if valid_subs:
        valid_subs.sort(key=lambda x: x[2])
        sorted_indices = [x[0] for x in valid_subs]
        sorted_names = [x[1] for x in valid_subs]
        
        fig2, ax2 = plt.subplots(figsize=(max(8, len(sorted_names) * 0.4), 4))
        x_axis = np.arange(len(sorted_names))

        # Plot Atlas Baseline
        hd_atlas_sorted = [hd95_atlas[i] for i in sorted_indices]
        ax2.scatter(x_axis, hd_atlas_sorted, c='steelblue', s=50, label='CNR vs Atlas', zorder=3)

        # Plot Anchored Data
        if has_anchored:
            hd_anch_sorted = [extract('hd95_cnr_vs_anchored_atlas_mm')[i] for i in sorted_indices]
            ax2.scatter(x_axis, hd_anch_sorted, c='darkorange', s=40, marker='D', label='CNR vs Anchored', zorder=3)

        # Plot Manual Data
        if has_manual:
            hd_man_sorted = [extract('hd95_cnr_vs_manual_mm')[i] for i in sorted_indices]
            ax2.scatter(x_axis, hd_man_sorted, c='seagreen', s=35, marker='s', label='CNR vs Manual', zorder=3)

        ax2.axhline(5.0, color='firebrick', ls='--', alpha=0.6, label='5 mm Threshold')
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('HD95 (mm)')
        ax2.set_title('SN Mask Surface Distance — HD95', fontweight='bold')
        ax2.legend(fontsize=9, loc='upper left')
        ax2.spines[['top', 'right']].set_visible(False)
        
        out2 = output_path / 'Group_QC_HD95_Summary.png'
        fig2.savefig(str(out2), dpi=200, bbox_inches='tight')
        plt.close(fig2)
        log.info(f"HD95 plot saved: {out2}")

    # Anchoring Beneift Plot
    if has_anchored and has_manual:
        deltas, sub_names = [], []
        for i, subj in enumerate(subjects):
            anch_val = extract('dice_anchored_vs_manual')[i]
            raw_val = extract('dice_atlas_vs_manual')[i]
            if anch_val is not None and raw_val is not None:
                deltas.append(anch_val - raw_val)
                sub_names.append(subj)

        if deltas:
            # Sort by highest improvement
            sorted_pairs = sorted(zip(deltas, sub_names))
            deltas_sorted, subs_sorted = zip(*sorted_pairs)

            fig3, ax3 = plt.subplots(figsize=(6, max(4, len(subs_sorted) * 0.3)))
            colors = ['seagreen' if d > 0 else 'firebrick' for d in deltas_sorted]
            
            ax3.barh(range(len(deltas_sorted)), deltas_sorted, color=colors, alpha=0.8)
            ax3.axvline(0, color='black', lw=1)
            
            mean_delta = np.mean(deltas_sorted)
            ax3.axvline(mean_delta, color='gray', ls='--', label=f'Mean Δ = {mean_delta:+.3f}')

            ax3.set_yticks(range(len(subs_sorted)))
            ax3.set_yticklabels(subs_sorted, fontsize=8)
            ax3.set_xlabel('DICE (Anchored - Raw Atlas)')
            ax3.set_title('Does Anchoring Improve Manual Overlap?', fontweight='bold')
            ax3.legend()
            ax3.spines[['top', 'right']].set_visible(False)
            
            out3 = output_path / 'Group_QC_Anchoring_Benefit.png'
            fig3.savefig(str(out3), dpi=200, bbox_inches='tight')
            plt.close(fig3)
            log.info(f"Benefit plot saved: {out3}")

log.info("=" * 60)
log.info("Pipeline complete.")
log.info(f"Log: {log_file}")
log.info("=" * 60)


