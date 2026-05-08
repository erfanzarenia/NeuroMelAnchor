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
parser.add_argument("--seed_cnr_floor", default=0.05, type=float, help="Minimum CNR threshold used to build NM-defined seed (default: 0.05)")
parser.add_argument("--seed_min_cluster_vox", default=15, type=int, help="Minimum connected-component size kept in NM seed (default: 15 voxels)")
parser.add_argument("--seed_keep_components", default=2, type=int, help="Maximum number of largest seed components to keep (default: 2 for bilateral SN)")
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
log.info(f"seed_cnr_floor      : {args.seed_cnr_floor}")
log.info(f"seed_min_cluster    : {args.seed_min_cluster_vox}")
log.info(f"seed_keep_components: {args.seed_keep_components}")
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

if args.stage in ['1', 'all'] and shutil.which('matlab') is None:
    raise RuntimeError("Matlab not found on path. \n" \
    "Please ensure SPM is installed and on your MATLAB Path")
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

def generate_qc_atlas(atlas_mask_mni, t1_mni_image):
    import os
    import matplotlib
    import numpy as np
    import nibabel as nib
    matplotlib.use('Agg')
    from nilearn import plotting
    print(f"\n{'='*60}")
    print(f"Computing Atlas Overlay QC Plot...")
    print(f"\n{'='*60}")

    out_png = os.path.abspath('Registration_Atlas_on_T1Norm_QC.png')
    display = plotting.plot_anat(t1_mni_image,
                                            display_mode='ortho', dim=-0.5, cmap='gray',
                                            title="Atlas on T1-norm QC", draw_cross=False, colorbar=False)
    display.add_overlay(atlas_mask_mni, cmap='Reds', alpha=0.8)
    display.savefig(out_png, dpi=300)
    display.close()
    return out_png

def extract_tractography_seed(
    nm_image,
    native_sn_mask,
    native_cp_mask,
    cnr_floor=0.05,
    min_cluster_vox=15,
    keep_components=2,
):
    import nibabel as nib
    import numpy as np
    import os
    from scipy.stats import gaussian_kde
    import scipy.ndimage as ndi
    from nilearn.image import smooth_img
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print("Computing CNR and Extracting Tractography Seed...")
    print(f"\n{'='*60}")

    nm_image = nib.load(nm_image)
    nm_data = nm_image.get_fdata()
    sn_data = nib.load(native_sn_mask).get_fdata().astype(bool)
    cp_data = nib.load(native_cp_mask).get_fdata().astype(bool)

    if not sn_data.any() or not cp_data.any():
        raise ValueError(f"Empty mask detected. SN voxels: {sn_data.sum()} | CP voxels: {cp_data.sum()}")

    cp_int = nm_data[cp_data]
    cp_int = cp_int[cp_int > 0]
    sn_int = nm_data[sn_data]
    sn_int = sn_int[sn_int > 0]

    if cp_int.size < 10 or sn_int.size < 10:
        raise ValueError(
            f"Insufficient positive intensity samples. CP: {cp_int.size}, SN: {sn_int.size}")

    try:
        x_range = np.linspace(cp_int.min(), cp_int.max(), 1000)
        cp_mode = float(x_range[np.argmax(gaussian_kde(cp_int)(x_range))])
    except Exception:
        cp_mode = float(np.median(cp_int))

    if cp_mode <= 0:
        cp_mode = float(np.mean(cp_int))

    cnr_map = np.zeros_like(nm_data)
    cnr_map[sn_data] = (nm_data[sn_data] - cp_mode) / cp_mode
    cnr_nii = nib.Nifti1Image(cnr_map, nm_image.affine, nm_image.header)

    smoothed_cnr_nii = smooth_img(cnr_nii, fwhm=1.0)
    smoothed_cnr_data = smoothed_cnr_nii.get_fdata()

    sn_cnr_values = smoothed_cnr_data[sn_data]
    p01 = np.percentile(sn_cnr_values, 1)
    p99 = np.percentile(sn_cnr_values, 99)

    valid_voxels = (
        sn_data
        & (smoothed_cnr_data >= max(float(cnr_floor), float(p01)))
        & (smoothed_cnr_data <= p99)
    )

    # Refinement step: remove tiny islands and keep up to N largest components
    refined_mask = ndi.binary_fill_holes(valid_voxels)
    labels, n_labels = ndi.label(refined_mask)
    if n_labels > 0:
        counts = ndi.sum(refined_mask, labels, index=np.arange(1, n_labels + 1))
        keep = [int(i + 1) for i, c in enumerate(counts) if c >= int(min_cluster_vox)]
        if keep and int(keep_components) > 0 and len(keep) > int(keep_components):
            keep = sorted(keep, key=lambda k: counts[k - 1], reverse=True)[: int(keep_components)]
        if keep:
            refined_mask = np.isin(labels, keep)

    final_seed_mask = refined_mask.astype(np.uint8)

    cnr_out = os.path.abspath('Subject_CNR_Map_Smoothed.nii.gz')
    seed_out = os.path.abspath('Native_Tractography_Seed_SNc.nii.gz')

    nib.save(smoothed_cnr_nii, cnr_out)
    nib.save(nib.Nifti1Image(final_seed_mask, nm_image.affine, nm_image.header), seed_out)

    all_vals = np.concatenate([cp_int, sn_int])
    bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cp_int, bins=bins, alpha=0.55, color='b', label='CP (background)', density=True)
    ax.hist(sn_int, bins=bins, alpha=0.55, color='y', label='SN (signal)', density=True)
    ax.axvline(cp_mode, color='m', lw=1.2, ls='--', label=f'CP mode = {cp_mode:.1f}')
    ax.set_title(
        f'SN vs CP intensities | mean SN CNR = {np.mean(smoothed_cnr_data[sn_data]):.3f} | '
        f'seed voxels = {int(final_seed_mask.sum())}',
        fontsize=9,
    )
    ax.set_xlabel('Voxel intensity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
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
        a, b = a.astype(bool), b.astype(bool)
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

    def _pair_metrics(mask_a, mask_b, img, spacing):
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)

        n_a = int(a.sum())
        n_b = int(b.sum())
        inter = int((a & b).sum())
        union = int((a | b).sum())

        precision = float(inter / n_a) if n_a > 0 else float('nan')
        recall = float(inter / n_b) if n_b > 0 else float('nan')
        jaccard = float(inter / union) if union > 0 else float('nan')

        voxel_volume_mm3 = float(np.prod(spacing))
        vol_a_mm3 = float(n_a * voxel_volume_mm3)
        vol_b_mm3 = float(n_b * voxel_volume_mm3)
        vol_delta_mm3 = float(vol_a_mm3 - vol_b_mm3)
        vol_delta_pct = float(((vol_a_mm3 - vol_b_mm3) / vol_b_mm3) * 100.0) if vol_b_mm3 > 0 else float('nan')

        centroid_a = _centroid_mm(img, a)
        centroid_b = _centroid_mm(img, b)
        centroid_delta_xyz = None
        centroid_distance_mm = None
        if centroid_a and centroid_b:
            delta = np.array(centroid_a) - np.array(centroid_b)
            centroid_delta_xyz = [float(delta[0]), float(delta[1]), float(delta[2])]
            centroid_distance_mm = float(np.linalg.norm(delta))

        return {
            'n_voxels_a': n_a,
            'n_voxels_b': n_b,
            'intersection_voxels': inter,
            'fp_voxels_a_vs_b': int(n_a - inter),
            'fn_voxels_a_vs_b': int(n_b - inter),
            'dice': _dice(a, b),
            'jaccard': jaccard,
            'precision_a_vs_b': precision,
            'recall_a_vs_b': recall,
            'hd95_mm': _hd95(a, b, spacing),
            'volume_a_mm3': vol_a_mm3,
            'volume_b_mm3': vol_b_mm3,
            'volume_delta_a_minus_b_mm3': vol_delta_mm3,
            'volume_delta_a_minus_b_percent': vol_delta_pct,
            'centroid_a_mm': centroid_a,
            'centroid_b_mm': centroid_b,
            'centroid_delta_a_minus_b_mm_xyz': centroid_delta_xyz,
            'centroid_distance_mm': centroid_distance_mm,
        }

    cnr_img, cnr_data = _load(cnr_mask)
    atlas_img, atlas_data = _load(atlas_mask)

    if cnr_img.shape != atlas_img.shape:
        raise ValueError(
            f"Shape mismatch: CNR {cnr_img.shape} vs atlas {atlas_img.shape}.\n"
            'Both masks must be in the same native NM space.')
    if not np.allclose(cnr_img.affine, atlas_img.affine, atol=1e-3):
        raise ValueError('Affine mismatch: CNR and atlas masks not in same space.')

    spacing = tuple(float(z) for z in cnr_img.header.get_zooms()[:3])

    pair_cnr_atlas = _pair_metrics(cnr_data, atlas_data, cnr_img, spacing)

    qc = {
        'n_voxels_cnr': int(cnr_data.sum()),
        'n_voxels_atlas': int(atlas_data.sum()),
        'diff_voxels_cnr_vs_atlas': int(cnr_data.sum()) - int(atlas_data.sum()),
        'dice_cnr_vs_atlas': pair_cnr_atlas['dice'],
        'hd95_cnr_vs_atlas_mm': pair_cnr_atlas['hd95_mm'],
        'centroid_cnr_mm': pair_cnr_atlas['centroid_a_mm'],
        'centroid_atlas_mm': pair_cnr_atlas['centroid_b_mm'],
        'centroid_distance_cnr_vs_atlas_mm': pair_cnr_atlas['centroid_distance_mm'],
        'n_voxels_anchored_atlas': None,
        'dice_cnr_vs_anchored_atlas': None,
        'hd95_cnr_vs_anchored_atlas_mm': None,
        'centroid_anchored_atlas_mm': None,
        'centroid_distance_cnr_vs_anchored_mm': None,
        'n_voxels_manual': None,
        'dice_cnr_vs_manual': None,
        'hd95_cnr_vs_manual_mm': None,
        'dice_atlas_vs_manual': None,
        'hd95_atlas_vs_manual_mm': None,
        'dice_anchored_vs_manual': None,
        'hd95_anchored_vs_manual_mm': None,
        'centroid_manual_mm': None,
        'centroid_distance_cnr_vs_manual_mm': None,
        'centroid_distance_atlas_vs_manual_mm': None,
        'pairwise': {
            'cnr_vs_atlas': pair_cnr_atlas,
        },
        'flags': [],
    }

    anch_data = None
    pair_cnr_anch = None
    if anchored_atlas_mask is not None and str(anchored_atlas_mask).strip():
        anch_img, anch_data = _load(anchored_atlas_mask)
        if cnr_img.shape != anch_img.shape:
            raise ValueError(f"Shape mismatch: CNR {cnr_img.shape} vs anchored {anch_img.shape}.")
        if not np.allclose(cnr_img.affine, anch_img.affine, atol=1e-3):
            raise ValueError('Affine mismatch: CNR and anchored atlas not in same space.')

        pair_cnr_anch = _pair_metrics(cnr_data, anch_data, cnr_img, spacing)
        qc['n_voxels_anchored_atlas'] = int(anch_data.sum())
        qc['dice_cnr_vs_anchored_atlas'] = pair_cnr_anch['dice']
        qc['hd95_cnr_vs_anchored_atlas_mm'] = pair_cnr_anch['hd95_mm']
        qc['centroid_anchored_atlas_mm'] = pair_cnr_anch['centroid_b_mm']
        qc['centroid_distance_cnr_vs_anchored_mm'] = pair_cnr_anch['centroid_distance_mm']
        qc['pairwise']['cnr_vs_anchored_atlas'] = pair_cnr_anch

        raw_dice = qc['dice_cnr_vs_atlas']
        anch_dice = qc['dice_cnr_vs_anchored_atlas']
        if raw_dice is not None and anch_dice is not None and anch_dice < raw_dice - 0.05:
            qc['flags'].append('Anchoring degraded DICE vs CNR seed — check ROI mask and CNR quality')

    pair_cnr_manual = None
    pair_atlas_manual = None
    pair_anch_manual = None
    if manual_mask is not None and str(manual_mask).strip():
        man_img, man_data = _load(manual_mask)
        if cnr_img.shape != man_img.shape:
            raise ValueError(f"Shape mismatch: CNR {cnr_img.shape} vs manual {man_img.shape}.")
        if not np.allclose(cnr_img.affine, man_img.affine, atol=1e-3):
            raise ValueError('Affine mismatch: CNR and manual not in same space.')

        pair_cnr_manual = _pair_metrics(cnr_data, man_data, cnr_img, spacing)
        pair_atlas_manual = _pair_metrics(atlas_data, man_data, cnr_img, spacing)

        qc['n_voxels_manual'] = int(man_data.sum())
        qc['dice_cnr_vs_manual'] = pair_cnr_manual['dice']
        qc['hd95_cnr_vs_manual_mm'] = pair_cnr_manual['hd95_mm']
        qc['dice_atlas_vs_manual'] = pair_atlas_manual['dice']
        qc['hd95_atlas_vs_manual_mm'] = pair_atlas_manual['hd95_mm']
        qc['centroid_manual_mm'] = pair_cnr_manual['centroid_b_mm']
        qc['centroid_distance_cnr_vs_manual_mm'] = pair_cnr_manual['centroid_distance_mm']
        qc['centroid_distance_atlas_vs_manual_mm'] = pair_atlas_manual['centroid_distance_mm']

        qc['pairwise']['cnr_vs_manual'] = pair_cnr_manual
        qc['pairwise']['atlas_vs_manual'] = pair_atlas_manual

        if anch_data is not None:
            pair_anch_manual = _pair_metrics(anch_data, man_data, cnr_img, spacing)
            qc['dice_anchored_vs_manual'] = pair_anch_manual['dice']
            qc['hd95_anchored_vs_manual_mm'] = pair_anch_manual['hd95_mm']
            qc['pairwise']['anchored_atlas_vs_manual'] = pair_anch_manual

        if qc['dice_cnr_vs_manual'] is not None and qc['dice_cnr_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE CNR vs manual — automated mask diverges from expert')
        if qc['dice_atlas_vs_manual'] is not None and qc['dice_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE atlas vs manual — registration or atlas fit may be poor')
        if qc['hd95_cnr_vs_manual_mm'] is not None and qc['hd95_cnr_vs_manual_mm'] > 5.0:
            qc['flags'].append('High HD95 CNR vs manual (>5 mm)')
        if qc['centroid_distance_cnr_vs_manual_mm'] is not None and qc['centroid_distance_cnr_vs_manual_mm'] > 3.0:
            qc['flags'].append('Large centroid offset CNR vs manual (>3 mm)')

        a_vs_m = qc.get('dice_anchored_vs_manual')
        raw_vs_m = qc.get('dice_atlas_vs_manual')
        if a_vs_m is not None and raw_vs_m is not None and a_vs_m < raw_vs_m - 0.05:
            qc['flags'].append('Anchoring reduced DICE vs manual — CNR seed may be poorly defined')

    if qc['n_voxels_cnr'] < 20:
        qc['flags'].append('CNR seed very small (<20 voxels) — check CNR map and SN registration')
    if qc['dice_cnr_vs_atlas'] is not None and qc['dice_cnr_vs_atlas'] < 0.4:
        qc['flags'].append('Low DICE CNR vs atlas (<0.4) — possible registration failure')
    if qc['hd95_cnr_vs_atlas_mm'] is not None and qc['hd95_cnr_vs_atlas_mm'] > 5.0:
        qc['flags'].append('High HD95 CNR vs atlas (>5 mm)')
    if qc['centroid_distance_cnr_vs_atlas_mm'] is not None and qc['centroid_distance_cnr_vs_atlas_mm'] > 3.0:
        qc['flags'].append('Large centroid offset CNR vs atlas (>3 mm)')

    qc_file = os.path.abspath('SN_Mask_QC.json')
    with open(qc_file, 'w') as f:
        json.dump(qc, f, indent=2)

    return qc_file
def generate_disagreement_maps_core(cnr_mask, atlas_mask, anchored_atlas_mask):
    import os
    import json
    import numpy as np
    import nibabel as nib

    def _load(path):
        img = nib.load(path)
        return img, img.get_fdata().astype(bool)

    def _assert_same(ref_img, img, label):
        if ref_img.shape != img.shape:
            raise ValueError(f"Shape mismatch for {label}: {ref_img.shape} vs {img.shape}")
        if not np.allclose(ref_img.affine, img.affine, atol=1e-3):
            raise ValueError(f"Affine mismatch for {label}")

    def _build_map(a, b, ref_img, stem):
        out = np.zeros(a.shape, dtype=np.uint8)
        overlap = a & b
        a_only = a & ~b
        b_only = b & ~a

        out[overlap] = 1
        out[a_only] = 2
        out[b_only] = 3

        out_path = os.path.abspath(f"{stem}.nii.gz")
        nib.save(nib.Nifti1Image(out, ref_img.affine, ref_img.header), out_path)

        return out_path, {
            'overlap_voxels': int(overlap.sum()),
            'a_only_voxels': int(a_only.sum()),
            'b_only_voxels': int(b_only.sum()),
            'total_disagreement_voxels': int((a_only | b_only).sum()),
        }

    cnr_img, cnr = _load(cnr_mask)
    atlas_img, atlas = _load(atlas_mask)
    anch_img, anchored = _load(anchored_atlas_mask)

    _assert_same(cnr_img, atlas_img, 'cnr_vs_atlas')
    _assert_same(cnr_img, anch_img, 'cnr_vs_anchored_atlas')

    cnr_vs_atlas_map, s1 = _build_map(cnr, atlas, cnr_img, 'Disagreement_CNR_vs_Atlas')
    cnr_vs_anchored_map, s2 = _build_map(cnr, anchored, cnr_img, 'Disagreement_CNR_vs_AnchoredAtlas')

    summary = {
        'label_meaning': {'0': 'background', '1': 'overlap', '2': 'A_only', '3': 'B_only'},
        'cnr_vs_atlas': s1,
        'cnr_vs_anchored_atlas': s2,
    }
    summary_file = os.path.abspath('Disagreement_Core_Summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return cnr_vs_atlas_map, cnr_vs_anchored_map, summary_file


def generate_disagreement_maps_manual(cnr_mask, atlas_mask, anchored_atlas_mask, manual_mask):
    import os
    import json
    import numpy as np
    import nibabel as nib

    def _load(path):
        img = nib.load(path)
        return img, img.get_fdata().astype(bool)

    def _assert_same(ref_img, img, label):
        if ref_img.shape != img.shape:
            raise ValueError(f"Shape mismatch for {label}: {ref_img.shape} vs {img.shape}")
        if not np.allclose(ref_img.affine, img.affine, atol=1e-3):
            raise ValueError(f"Affine mismatch for {label}")

    def _build_map(a, b, ref_img, stem):
        out = np.zeros(a.shape, dtype=np.uint8)
        overlap = a & b
        a_only = a & ~b
        b_only = b & ~a

        out[overlap] = 1
        out[a_only] = 2
        out[b_only] = 3

        out_path = os.path.abspath(f"{stem}.nii.gz")
        nib.save(nib.Nifti1Image(out, ref_img.affine, ref_img.header), out_path)

        return out_path, {
            'overlap_voxels': int(overlap.sum()),
            'a_only_voxels': int(a_only.sum()),
            'b_only_voxels': int(b_only.sum()),
            'total_disagreement_voxels': int((a_only | b_only).sum()),
        }

    cnr_img, cnr = _load(cnr_mask)
    atlas_img, atlas = _load(atlas_mask)
    anch_img, anchored = _load(anchored_atlas_mask)
    man_img, manual = _load(manual_mask)

    _assert_same(cnr_img, atlas_img, 'cnr_vs_manual')
    _assert_same(cnr_img, anch_img, 'anchored_vs_manual')
    _assert_same(cnr_img, man_img, 'atlas_vs_manual')

    cnr_vs_manual_map, s1 = _build_map(cnr, manual, cnr_img, 'Disagreement_CNR_vs_Manual')
    atlas_vs_manual_map, s2 = _build_map(atlas, manual, cnr_img, 'Disagreement_Atlas_vs_Manual')
    anchored_vs_manual_map, s3 = _build_map(anchored, manual, cnr_img, 'Disagreement_AnchoredAtlas_vs_Manual')

    summary = {
        'label_meaning': {'0': 'background', '1': 'overlap', '2': 'A_only', '3': 'B_only'},
        'cnr_vs_manual': s1,
        'atlas_vs_manual': s2,
        'anchored_atlas_vs_manual': s3,
    }
    summary_file = os.path.abspath('Disagreement_Manual_Summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return cnr_vs_manual_map, atlas_vs_manual_map, anchored_vs_manual_map, summary_file


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

    NM_Data = {'runs':'{subject_id}/anat/{subject_id}_acq-CombEchoNM*_GRE.nii.gz'}
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

    qc_node_atlas = Node(Function(input_names=['atlas_mask_mni', 't1_mni_image'], output_names=['out_png'], function=generate_qc_atlas), name='Visual_QC_Atlas')
    qc_node_atlas.inputs.atlas_mask_mni = str(atlas_mask_mni)

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
        (t1_to_mni, qc_node_atlas, [('warped_image', 't1_mni_image')]),

        # Metrics
        (realign, datasink1, [('realignment_parameters', 'metrics.@NM_MotPar')]),
        (motion_node, datasink1, [('stats_file', 'metrics.@NM_MotionStats'),
                                ('plot_file', 'metrics.@NM_MotionPlot')]),

        # QC
        (run_compare, datasink1, [('out_png', 'qc.@QC_RunComparison')]),
        (qc_node_nm, datasink1, [('out_png', 'qc.@QC_NM_Plot')]),
        (qc_node_t1, datasink1, [('out_png', 'qc.@QC_T1_Plot')]),
        (qc_node_atlas, datasink1, [('out_png', 'qc.@QC_atlas_Plot')]),

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
    nm_to_template.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    nm_to_template.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    nm_to_template.inputs.number_of_iterations = [[1000, 500, 250, 100],
                                                  [1000, 500, 250, 100],
                                                  [100, 70, 50, 20]]
    nm_to_template.inputs.metric = ['Mattes', 'Mattes', 'CC']
    nm_to_template.inputs.metric_weight = [1, 1, 1]
    nm_to_template.inputs.radius_or_number_of_bins = [64, 64, 4]
    
    nm_to_template.inputs.shrink_factors = [[8, 4, 2, 1],
                                            [8, 4, 2, 1],
                                            [8, 4, 2, 1]]
    nm_to_template.inputs.smoothing_sigmas = [[3, 2, 1, 0],
                                              [3, 2, 1, 0],
                                              [3, 2, 1, 0]]
    nm_to_template.inputs.sigma_units = ['vox', 'vox', 'vox']
    nm_to_template.inputs.sampling_percentage = [0.25, 0.25, 1]
    nm_to_template.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
    nm_to_template.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
    nm_to_template.inputs.convergence_window_size = [20, 20, 10]
    nm_to_template.inputs.winsorize_lower_quantile = 0.005
    nm_to_template.inputs.winsorize_upper_quantile = 0.995
    nm_to_template.inputs.use_histogram_matching = [True, True, True]

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
    cnr_seed = Node(Function(input_names=['nm_image', 'native_sn_mask', 'native_cp_mask', 'cnr_floor', 'min_cluster_vox', 'keep_components'], 
                            output_names=['cnr_out', 'seed_out', 'histogram'], function=extract_tractography_seed), name='CNR_Seed')
    cnr_seed.inputs.cnr_floor = float(args.seed_cnr_floor)
    cnr_seed.inputs.min_cluster_vox = int(args.seed_min_cluster_vox)
    cnr_seed.inputs.keep_components = int(args.seed_keep_components)
    
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
    
    # =================================================================
    # QC
    # =================================================================

    qc_node = Node(Function(input_names=['cnr_mask', 'atlas_mask', 'anchored_atlas_mask', 'manual_mask'], output_names=['qc_file'], function=compute_mask_qc), name='QC_Metrics')

    disagree_core = Node(Function(
        input_names=['cnr_mask', 'atlas_mask', 'anchored_atlas_mask'],
        output_names=['cnr_vs_atlas_map', 'cnr_vs_anchored_map', 'summary_file'],
        function=generate_disagreement_maps_core), name='Disagreement_Core')

    if manual_dir is not None:
        ManualData = {'manual_sn': '{subject_id}/anat/{subject_id}_space-NM_label-SN_desc-manual_mask.nii'}
        SelectManual = Node(SelectFiles(ManualData, base_directory=str(manual_dir)), name='SelectManual')

        disagree_manual = Node(Function(
            input_names=['cnr_mask', 'atlas_mask', 'anchored_atlas_mask', 'manual_mask'],
            output_names=['cnr_vs_manual_map', 'atlas_vs_manual_map', 'anchored_vs_manual_map', 'summary_file'],
            function=generate_disagreement_maps_manual), name='Disagreement_Manual')
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
        (inputnode2,        GrabPhase1, [('subject_id', 'subject_id')]),
        (inputnode2,        datasink2, [('subject_id', 'container')]),

        (GrabPhase1,        nm_to_template, [('bias_nm', 'moving_image')]),

        (GrabPhase1,        mni_to_nm_sn, [('bias_nm', 'reference_image')]),
        (nm_to_template,    mni_to_nm_sn, [('inverse_composite_transform', 'transforms')]),
        
        (GrabPhase1,        mni_to_nm_cp, [('bias_nm', 'reference_image')]),
        (nm_to_template,    mni_to_nm_cp, [('inverse_composite_transform', 'transforms')]),
        
        (GrabPhase1,        mni_to_nm_atlas, [('bias_nm', 'reference_image')]),
        (nm_to_template,    mni_to_nm_atlas, [('inverse_composite_transform', 'transforms')]),
    
        (GrabPhase1,        cnr_seed, [('bias_nm', 'nm_image')]),
        (mni_to_nm_sn,      cnr_seed, [('output_image', 'native_sn_mask')]),
        (mni_to_nm_cp,      cnr_seed, [('output_image', 'native_cp_mask')]),

        (cnr_seed,          qc_node, [('seed_out','cnr_mask')]), 
        (mni_to_nm_atlas,   qc_node, [('output_image', 'atlas_mask')]),

        (cnr_seed,          roi_mask_node, [('seed_out', 'cnr_seed')]),

        (GrabPhase1,        anchor_atlas, [('bias_nm', 'fixed_image')]),
        (mni_to_nm_atlas,   anchor_atlas, [('output_image', 'moving_image')]),
        (roi_mask_node,     anchor_atlas, [('roi_mask', 'fixed_image_masks')]),
        

        (anchor_atlas,      qc_node, [('warped_image', 'anchored_atlas_mask')]),

        (cnr_seed,          disagree_core, [('seed_out', 'cnr_mask')]),
        (mni_to_nm_atlas,   disagree_core, [('output_image', 'atlas_mask')]),
        (anchor_atlas,      disagree_core, [('warped_image', 'anchored_atlas_mask')]),

        (anchor_atlas,      datasink2, [('warped_image', 'anat.@CIT168_Anchored'),
                                   ('composite_transform','transforms.@anchor_fwd'),
                                 ('inverse_composite_transform','transforms.@anchor_inv'),
        ]),
        (roi_mask_node,     datasink2, [('roi_mask', 'anat.@SN_ROI_Mask')]),
        
        (mni_to_nm_sn,      datasink2, [('output_image', 'anat.@Native_SN_Mask')]),
        (mni_to_nm_cp,      datasink2, [('output_image', 'anat.@Native_CP_Mask')]),
        (mni_to_nm_atlas,   datasink2, [('output_image', 'anat.@Native_Atlas_Mask')]),
        (cnr_seed,          datasink2, [('cnr_out', 'cnr.@CNR_Map'),
                                        ('seed_out', 'anat.@NM_Defined_Mask'),
                                        ('histogram', 'qc.@CNR_Histogram')]),
        (qc_node,           datasink2, [('qc_file', 'cnr.@SN_Mask_Dice_QC')]),
        (disagree_core,     datasink2, [('cnr_vs_atlas_map', 'qc.@Disagreement_CNR_vs_Atlas'),
                                        ('cnr_vs_anchored_map', 'qc.@Disagreement_CNR_vs_Anchored'),
                                        ('summary_file', 'qc.@Disagreement_Core_Summary')]),
    ]

    if manual_dir is not None:
        wf2_connections += [
            (inputnode2,    SelectManual, [('subject_id', 'subject_id')]),
            (SelectManual,  qc_node, [('manual_sn', 'manual_mask')]),

            (cnr_seed,        disagree_manual, [('seed_out', 'cnr_mask')]),
            (mni_to_nm_atlas, disagree_manual, [('output_image', 'atlas_mask')]),
            (anchor_atlas,    disagree_manual, [('warped_image', 'anchored_atlas_mask')]),
            (SelectManual,    disagree_manual, [('manual_sn', 'manual_mask')]),

            (disagree_manual, datasink2, [('cnr_vs_manual_map', 'qc.@Disagreement_CNR_vs_Manual'),
                                          ('atlas_vs_manual_map', 'qc.@Disagreement_Atlas_vs_Manual'),
                                          ('anchored_vs_manual_map', 'qc.@Disagreement_Anchored_vs_Manual'),
                                          ('summary_file', 'qc.@Disagreement_Manual_Summary')]),
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

log.info(f"{'=' * 60}")
log.info("Generating Group QC Plots...")

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

output_path = Path(output_base)
qc_files = sorted(output_path.glob("sub-*/cnr/SN_Mask_QC.json"))

if not qc_files:
    log.info("No QC JSON files found — skipping group QC plots.")
else:
    subjects = []
    
    dice_atlas, hd95_atlas = [], []
    dice_anchored, hd95_anchored = [], []
    dice_manual, hd95_manual = [], []
    dice_atlas_man, hd95_atlas_man = [], []
    dice_anchored_man, hd95_anchored_man = [], []

    pairwise_rows = []

    for f in qc_files:
        with open(f, "r") as fh:
            data = json.load(fh)
            
        subjects.append(f.parts[-3])
        
        dice_atlas.append(data.get("dice_cnr_vs_atlas"))
        hd95_atlas.append(data.get("hd95_cnr_vs_atlas_mm"))
        
        dice_anchored.append(data.get("dice_cnr_vs_anchored_atlas"))
        hd95_anchored.append(data.get("hd95_cnr_vs_anchored_atlas_mm"))
        
        dice_manual.append(data.get("dice_cnr_vs_manual"))
        hd95_manual.append(data.get("hd95_cnr_vs_manual_mm"))
        
        dice_atlas_man.append(data.get("dice_atlas_vs_manual"))
        hd95_atlas_man.append(data.get("hd95_atlas_vs_manual_mm"))
        
        dice_anchored_man.append(data.get("dice_anchored_vs_manual"))
        hd95_anchored_man.append(data.get("hd95_anchored_vs_manual_mm"))

        pairwise = data.get("pairwise", {})
        for pair_name, metrics in pairwise.items():
            if not isinstance(metrics, dict):
                continue
            pairwise_rows.append({
                "subject_id": f.parts[-3],
                "pair": pair_name,
                "dice": metrics.get("dice"),
                "jaccard": metrics.get("jaccard"),
                "precision": metrics.get("precision_a_vs_b"),
                "recall": metrics.get("recall_a_vs_b"),
                "hd95_mm": metrics.get("hd95_mm"),
                "centroid_distance_mm": metrics.get("centroid_distance_mm"),
                "volume_a_mm3": metrics.get("volume_a_mm3"),
                "volume_b_mm3": metrics.get("volume_b_mm3"),
                "volume_delta_mm3": metrics.get("volume_delta_a_minus_b_mm3"),
                "volume_delta_percent": metrics.get("volume_delta_a_minus_b_percent"),
            })

    if pairwise_rows:
        csv_out = output_path / "Group_QC_Pairwise_Summary.csv"
        fieldnames = [
            "subject_id", "pair", "dice", "jaccard", "precision", "recall",
            "hd95_mm", "centroid_distance_mm",
            "volume_a_mm3", "volume_b_mm3", "volume_delta_mm3", "volume_delta_percent",
        ]

        with open(csv_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in pairwise_rows:
                clean_row = {}
                for key in fieldnames:
                    val = row.get(key)
                    if isinstance(val, float) and not np.isfinite(val):
                        val = ""
                    clean_row[key] = val
                writer.writerow(clean_row)

        log.info(f"Pairwise QC summary saved: {csv_out}")

    has_manual = any(v is not None for v in dice_manual)
    has_anchored = any(v is not None for v in dice_anchored)

    if has_manual:
        paired_dice = [
            (sid, float(nm), float(at))
            for sid, nm, at in zip(subjects, dice_manual, dice_atlas_man)
            if nm is not None and at is not None and np.isfinite(float(nm)) and np.isfinite(float(at))
        ]
        paired_hd95 = [
            (sid, float(nm), float(at))
            for sid, nm, at in zip(subjects, hd95_manual, hd95_atlas_man)
            if nm is not None and at is not None and np.isfinite(float(nm)) and np.isfinite(float(at))
        ]

        comp = {
            'n_subjects_dice_compared': len(paired_dice),
            'n_subjects_hd95_compared': len(paired_hd95),
            'dice_nm_minus_atlas_median': None,
            'dice_nm_better_count': 0,
            'dice_atlas_better_count': 0,
            'hd95_atlas_minus_nm_median_mm': None,
            'hd95_nm_better_count': 0,
            'hd95_atlas_better_count': 0,
            'subjects_nm_better_both_dice_and_hd95': 0,
        }

        if paired_dice:
            dice_deltas = np.array([nm - at for _, nm, at in paired_dice], dtype=float)
            comp['dice_nm_minus_atlas_median'] = float(np.median(dice_deltas))
            comp['dice_nm_better_count'] = int(np.sum(dice_deltas > 0))
            comp['dice_atlas_better_count'] = int(np.sum(dice_deltas < 0))

        if paired_hd95:
            hd95_deltas = np.array([at - nm for _, nm, at in paired_hd95], dtype=float)
            comp['hd95_atlas_minus_nm_median_mm'] = float(np.median(hd95_deltas))
            comp['hd95_nm_better_count'] = int(np.sum(hd95_deltas > 0))
            comp['hd95_atlas_better_count'] = int(np.sum(hd95_deltas < 0))

        dice_map = {sid: (nm, at) for sid, nm, at in paired_dice}
        hd95_map = {sid: (nm, at) for sid, nm, at in paired_hd95}
        shared = sorted(set(dice_map).intersection(hd95_map))
        comp['subjects_nm_better_both_dice_and_hd95'] = int(
            sum((dice_map[s][0] > dice_map[s][1]) and (hd95_map[s][0] < hd95_map[s][1]) for s in shared)
        )

        comp_json = output_path / 'Group_QC_Method_Comparison.json'
        with open(comp_json, 'w') as fh:
            json.dump(comp, fh, indent=2)
        log.info(f"Method comparison summary saved: {comp_json}")

        if paired_dice or paired_hd95:
            fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(10.5, 4.5))

            if paired_dice:
                atlas_d = np.array([at for _, _, at in paired_dice], dtype=float)
                nm_d = np.array([nm for _, nm, _ in paired_dice], dtype=float)
                lim_min = float(min(atlas_d.min(), nm_d.min(), 0.0))
                lim_max = float(max(atlas_d.max(), nm_d.max(), 1.0))
                axes_cmp[0].scatter(atlas_d, nm_d, s=34, alpha=0.85, color='#4C72B0')
                axes_cmp[0].plot([lim_min, lim_max], [lim_min, lim_max], ls='--', lw=1.0, color='gray')
                axes_cmp[0].set_xlim(lim_min, lim_max)
                axes_cmp[0].set_ylim(lim_min, lim_max)
                axes_cmp[0].set_xlabel('Atlas vs Manual DICE')
                axes_cmp[0].set_ylabel('NM vs Manual DICE')
                axes_cmp[0].set_title('DICE: above diagonal favors NM')
                axes_cmp[0].spines[['top', 'right']].set_visible(False)
            else:
                axes_cmp[0].text(0.5, 0.5, 'No paired DICE data', ha='center', va='center', transform=axes_cmp[0].transAxes)
                axes_cmp[0].axis('off')

            if paired_hd95:
                atlas_h = np.array([at for _, _, at in paired_hd95], dtype=float)
                nm_h = np.array([nm for _, nm, _ in paired_hd95], dtype=float)
                lim_min = float(min(atlas_h.min(), nm_h.min(), 0.0))
                lim_max = float(max(atlas_h.max(), nm_h.max()))
                axes_cmp[1].scatter(atlas_h, nm_h, s=34, alpha=0.85, color='#E1812C')
                axes_cmp[1].plot([lim_min, lim_max], [lim_min, lim_max], ls='--', lw=1.0, color='gray')
                axes_cmp[1].set_xlim(lim_min, lim_max)
                axes_cmp[1].set_ylim(lim_min, lim_max)
                axes_cmp[1].set_xlabel('Atlas vs Manual HD95 (mm)')
                axes_cmp[1].set_ylabel('NM vs Manual HD95 (mm)')
                axes_cmp[1].set_title('HD95: below diagonal favors NM')
                axes_cmp[1].spines[['top', 'right']].set_visible(False)
            else:
                axes_cmp[1].text(0.5, 0.5, 'No paired HD95 data', ha='center', va='center', transform=axes_cmp[1].transAxes)
                axes_cmp[1].axis('off')

            fig_cmp.tight_layout()
            cmp_png = output_path / 'Group_QC_Method_Comparison.png'
            fig_cmp.savefig(str(cmp_png), dpi=200, bbox_inches='tight')
            plt.close(fig_cmp)
            log.info(f"Method comparison plot saved: {cmp_png}")

    def raincloud_strip(ax, values, pos, color, label):
        vals = np.array([v for v in values if v is not None], dtype=float)
        if len(vals) == 0: return
        kde = gaussian_kde(vals, bw_method=0.4)
        y_kde = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 200)
        x_kde = (kde(y_kde) / kde(y_kde).max()) * 0.35
        ax.fill_betweenx(y_kde, pos, pos + x_kde, alpha=0.5, color=color, label=label)
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(vals))
        ax.scatter(pos + jitter - 0.15, vals, s=18, color=color, alpha=0.7, zorder=3)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos - 0.18, pos - 0.12], [med, med], color="k", lw=2, zorder=4)
        ax.plot([pos - 0.18, pos - 0.18], [q1, q3], color="k", lw=1.5, zorder=4)

    total_strips = 1 + (1 if has_anchored else 0) + (2 if has_manual else 0) + (1 if has_anchored and has_manual else 0)
    fig1, ax1 = plt.subplots(figsize=(3 + (total_strips * 1.5), 5))
    
    current_pos = 1
    x_labels = []

    raincloud_strip(ax1, dice_atlas, current_pos, "#4C72B0", "CNR vs Atlas")
    x_labels.append("CNR vs Atlas")
    current_pos += 1

    if has_anchored:
        raincloud_strip(ax1, dice_anchored, current_pos, "#E1812C", "CNR vs Anchored")
        x_labels.append("CNR vs Anchored")
        current_pos += 1

    if has_manual:
        raincloud_strip(ax1, dice_manual, current_pos, "#DD8452", "CNR vs Manual")
        x_labels.append("CNR vs Manual")
        current_pos += 1
        
        raincloud_strip(ax1, dice_atlas_man, current_pos, "#55A868", "Atlas vs Manual")
        x_labels.append("Atlas vs Manual")
        current_pos += 1
        
        raincloud_strip(ax1, dice_anchored_man, current_pos, "#151372", "Anchored vs Manual")
        x_labels.append("Anchored vs Manual")
        current_pos += 1

    ax1.axhline(0.5, color="firebrick", lw=0.8, ls="--", alpha=0.6, label="DICE = 0.5")
    ax1.set_xlim(0.5, current_pos - 0.2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(range(1, current_pos))
    ax1.set_xticklabels(x_labels, rotation=15)
    ax1.set_ylabel("DICE coefficient", fontsize=11)
    ax1.set_title("SN Mask Overlap — DICE", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    out1 = output_path / "Group_QC_DICE_Raincloud.png"
    fig1.savefig(str(out1), dpi=200, bbox_inches="tight")
    plt.close(fig1)
    log.info(f"DICE raincloud saved: {out1}")

    valid_hd = [(i, s, v) for i, (s, v) in enumerate(zip(subjects, hd95_atlas)) if v is not None]

    if valid_hd:
        valid_hd.sort(key=lambda x: x[2])
        sorted_indices = [x[0] for x in valid_hd]
        sorted_subs = [x[1] for x in valid_hd]
        sorted_hd_atlas = [x[2] for x in valid_hd]

        fig2, ax2 = plt.subplots(figsize=(max(8, len(sorted_subs) * 0.4), 4))
        x_pos = range(len(sorted_subs))

        colours_atlas = ["firebrick" if v > 5.0 else "#4C72B0" for v in sorted_hd_atlas]
        ax2.scatter(x_pos, sorted_hd_atlas, c=colours_atlas, s=40, marker="o", label="CNR vs Atlas", zorder=3)

        sorted_hd_anch = [hd95_anchored[i] for i in sorted_indices]
        ax2.scatter(x_pos, sorted_hd_anch, c="#E1812C", s=40, marker="D", alpha=0.9, label="CNR vs Anchored", zorder=3)

        if has_manual:
            sorted_hd_man = [hd95_manual[i] for i in sorted_indices]
            sorted_hd_atlas_man = [hd95_atlas_man[i] for i in sorted_indices]

            ax2.scatter(x_pos, sorted_hd_man, c="#DD8452", s=35, marker="s", alpha=0.8, label="CNR vs Manual", zorder=3)
            ax2.scatter(x_pos, sorted_hd_atlas_man, c="#55A868", s=45, marker="^", alpha=0.8, label="Atlas vs Manual", zorder=3)

            sorted_hd_anch_man = [hd95_anchored_man[i] for i in sorted_indices]
            ax2.scatter(x_pos, sorted_hd_anch_man, c="#151372", s=45, marker="v", alpha=0.8, label="Anchored vs Manual", zorder=3)

        for idx, x in enumerate(x_pos):
            y_vals = [sorted_hd_atlas[idx]]
            if has_anchored and sorted_hd_anch[idx] is not None: y_vals.append(sorted_hd_anch[idx])
            if has_manual and sorted_hd_man[idx] is not None: y_vals.append(sorted_hd_man[idx])
            if has_manual and sorted_hd_atlas_man[idx] is not None: y_vals.append(sorted_hd_atlas_man[idx])
            if has_anchored and has_manual and sorted_hd_anch_man[idx] is not None: y_vals.append(sorted_hd_anch_man[idx])
            
            if len(y_vals) > 1:
                ax2.plot([x, x], [min(y_vals), max(y_vals)], color="gray", linestyle=":", lw=0.8, zorder=1)

        ax2.axhline(5.0, color="firebrick", lw=0.8, ls="--", alpha=0.7, label="5mm threshold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sorted_subs, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("HD95 (mm)", fontsize=11)
        ax2.set_title("SN Mask Surface Distance — HD95", fontsize=12)

        handles, labels = ax2.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax2.legend(unique.values(), unique.keys(), fontsize=9)

        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()

        out2 = output_path / "Group_QC_HD95_DotPlot.png"
        fig2.savefig(str(out2), dpi=200, bbox_inches="tight")
        plt.close(fig2)
        log.info(f"HD95 dot plot saved: {out2}")

    if pairwise_rows:
        pair_pretty = {
            'cnr_vs_atlas': 'CNR vs Atlas',
            'cnr_vs_anchored_atlas': 'CNR vs Anchored',
            'cnr_vs_manual': 'CNR vs Manual',
            'atlas_vs_manual': 'Atlas vs Manual',
            'anchored_atlas_vs_manual': 'Anchored vs Manual',
        }
        pair_order = [
            'cnr_vs_atlas',
            'cnr_vs_anchored_atlas',
            'cnr_vs_manual',
            'atlas_vs_manual',
            'anchored_atlas_vs_manual',
        ]

        vol_data, vol_labels = [], []
        cent_data, cent_labels = [], []

        for pair_key in pair_order:
            rows_for_pair = [r for r in pairwise_rows if r.get('pair') == pair_key]

            vol_vals = [
                abs(float(r['volume_delta_percent']))
                for r in rows_for_pair
                if r.get('volume_delta_percent') is not None and np.isfinite(float(r['volume_delta_percent']))
            ]
            if vol_vals:
                vol_data.append(vol_vals)
                vol_labels.append(pair_pretty[pair_key])

            cent_vals = [
                float(r['centroid_distance_mm'])
                for r in rows_for_pair
                if r.get('centroid_distance_mm') is not None and np.isfinite(float(r['centroid_distance_mm']))
            ]
            if cent_vals:
                cent_data.append(cent_vals)
                cent_labels.append(pair_pretty[pair_key])

        if vol_data or cent_data:
            fig3, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            if vol_data:
                axes[0].boxplot(vol_data, tick_labels=vol_labels, showmeans=True)
                axes[0].set_ylabel('|Volume delta| (%)')
                axes[0].set_title('Mask Volume Deviation')
                axes[0].tick_params(axis='x', labelrotation=20)
                axes[0].spines[['top', 'right']].set_visible(False)
            else:
                axes[0].text(0.5, 0.5, 'No volume delta data', ha='center', va='center', transform=axes[0].transAxes)
                axes[0].axis('off')

            if cent_data:
                axes[1].boxplot(cent_data, tick_labels=cent_labels, showmeans=True)
                axes[1].set_ylabel('Centroid distance (mm)')
                axes[1].set_title('Mask Centroid Deviation')
                axes[1].tick_params(axis='x', labelrotation=20)
                axes[1].spines[['top', 'right']].set_visible(False)
            else:
                axes[1].text(0.5, 0.5, 'No centroid distance data', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].axis('off')

            fig3.tight_layout()
            out3 = output_path / 'Group_QC_Deviation_Boxplots.png'
            fig3.savefig(str(out3), dpi=200, bbox_inches='tight')
            plt.close(fig3)
            log.info(f"Deviation boxplots saved: {out3}")


