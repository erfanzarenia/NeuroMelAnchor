#!/usr/bin/env python3
"""
================================================================================
NEXUS NeuroMelAnchor 
================================================================================
NeuroMelAnchor is a neuromelanin-centric preprocessing and segmentation pipeline, 
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
    7. Perform ANTs registration of NM-contrast mean to T1 (Rigid)
    8. Combine transforms and apply to bring NM to MNI space
    9. (Optional) SPM Smooth
    10. Generate registration QC plots, run-similarity QC, and motion metrics
    11. Create group-average NM template
 
Segment — Manual ROI Segmentation (ITK-SNAP)
    Interactive segmentation of cerebral peduncle (label 1) and substantia nigra (label 2) on the
    group-average NM template.  
    Masks are saved to resources/tpl-MNI152NLin2009cAsym/.
 
Stage 2  — Inverse Normalisation and CNR Seed Extraction
    1. Register subject NM mean to group template (Rigid + Affine + SyN)
    2. Warp segmented SN and CP masks to native NM space via inverse transforms
    3. Compute CNR map and NM-defined seed mask
    4. Compute Core-rim NM, probabilistic atlas, NMS atlas, and anchored-atlas comparisons
    5. Compute per-subject QC metrics and dashboards
            (optional: assess against manual masks if --manual_dir is provided)

Group QC — Rebuild dashboards and summary plots from existing Stage 2 outputs
    This stage is intentionally cheap: it does not rerun registration or mask generation.
 
Output Structure (per subject)
-------------------------------
    <output_root>/
    └── sub-XXX/
        ├── anat/          NM volumes (bias-corrected, mean, MNI-space, smoothed)
        ├── transforms/    ANTs composite warps (.h5)
        ├── metrics/       Motion stats JSON
        └── qc/            Registration overlays, run-comparison PNG, subject dashboard

Usage
-------------------------------
    python BrainHack_NM_Nipype_Pipeline.py 
        --project_root /path/to/project 
        --design_dir   /path/to/bids_root 
        --output_root  /path/to/outputs 
        --work_dir     /path/to/work 
        --n_procs 10                        defaults to 10; set lower/higher for your system
        --stage all                         defaults to all; can run 1, segment, 2, or group_qc
        --test                              test run on the first subject
        --smooth 0                          defaults to 0; set a smoothing FWHM in mm if needed
        --manual_dir  /path/to/manual_mask  for per-subject manual SN masks in Stage 2 QC

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
import csv
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path 
from datetime import datetime

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from templateflow.api import get
from nilearn.image import mean_img
from scipy.stats import gaussian_kde

from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import Realign, Smooth
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.ants import Registration, BrainExtraction, ApplyTransforms, N4BiasFieldCorrection

# =================================================================
# Argument Parsing
# =================================================================

parser = argparse.ArgumentParser(description="NEXUS NeuroMelAnchor")
parser.add_argument("--project_root", required=True, help="Project root directory", type=Path)
parser.add_argument("--output_root", required=True, help="Directory where all outputs will be written", type=Path)
parser.add_argument("--design_dir", required=True, help="BIDS root containing subject directories", type=Path)
parser.add_argument("--work_dir", required=True, help="NiPyPe working / scratch directory", type=Path)
parser.add_argument("--n_procs", default=10, type=int, help="Number of parallel processes (default: 10)")
parser.add_argument("--stage", choices=['1', 'segment', '2', 'group_qc', 'all'], default='all',
                     help="NeuroMelAnchor stage to run. Use group_qc to rebuild dashboards without rerunning registration (default: all)")
parser.add_argument("--smooth", default=0, type=int, help="Smoothing FWHM kernel applied after MNI warp. Set to 0 to skip smoothing (default: 0)")
parser.add_argument("--manual_dir", default=None, type=Path, help="Directory of per-subject manual SN Segmentations (optional for similarity QC during Stage 2)")
parser.add_argument("--prob_atlas_mni", default=None, type=Path, help="Optional probabilistic MNI SNc atlas for Stage 2 QC (default: CIT168 probabilistic SNc if present)")
parser.add_argument("--prob_atlas_threshold", default=0.25, type=float, help="Probability threshold used after warping probabilistic SNc atlas to native space (default: 0.25)")
parser.add_argument("--skip_prob_atlas", action="store_true", help="Disable optional probabilistic atlas Stage 2 QC")
parser.add_argument("--nms_atlas_mni", default=None, type=Path, help="Optional probabilistic NMS SNc atlas for Stage 2 QC (default: NMS probabilistic SNc if present)")
parser.add_argument("--nms_atlas_threshold", default=0.25, type=float, help="Probability threshold used after warping NMS SNc atlas to native space (default: 0.25)")
parser.add_argument("--skip_nms_atlas", action="store_true", help="Disable optional NMS atlas Stage 2 QC")
parser.add_argument("--skip_core_rim_nm", action="store_true", help="Disable optional core-rim NM mask generation")
parser.add_argument("--test", action="store_true", help="Test mode: run only the first subject or write TEST group-QC outputs")
args = parser.parse_args()

# =================================================================
# NM Mask Method Settings
# =================================================================

# These are method choices, not routine command-line options.
# Keep them here so the masking assumptions are easy to audit.
seed_cnr_floor = 0.05
seed_cnr_percentile = 55
seed_min_cluster_voxels = 15
seed_components_to_keep = 2

core_rim_erosion_mm = 1.0
core_rim_core_cnr_threshold = 0.05
core_rim_rim_cnr_threshold = 0.10
core_rim_min_cluster_voxels = 25
core_rim_components_to_keep = 2

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
log.info(f"smooth              : {args.smooth}")
log.info(f"manual_dir          : {manual_dir}")
log.info(f"seed_cnr_floor      : {seed_cnr_floor}")
log.info(f"seed_cnr_percentile : {seed_cnr_percentile}")
log.info(f"seed_min_cluster    : {seed_min_cluster_voxels}")
log.info(f"seed_keep_components: {seed_components_to_keep}")
log.info(f"prob_atlas_mni      : {args.prob_atlas_mni}")
log.info(f"prob_atlas_threshold: {args.prob_atlas_threshold}")
log.info(f"skip_prob_atlas     : {args.skip_prob_atlas}")
log.info(f"nms_atlas_mni       : {args.nms_atlas_mni}")
log.info(f"nms_atlas_threshold : {args.nms_atlas_threshold}")
log.info(f"skip_nms_atlas      : {args.skip_nms_atlas}")
log.info(f"core_rim_erosion_mm : {core_rim_erosion_mm}")
log.info(f"core_rim_core_cnr   : {core_rim_core_cnr_threshold}")
log.info(f"core_rim_rim_cnr    : {core_rim_rim_cnr_threshold}")
log.info(f"core_rim_min_cluster: {core_rim_min_cluster_voxels}")
log.info(f"core_rim_keep_comp  : {core_rim_components_to_keep}")
log.info(f"skip_core_rim_nm    : {args.skip_core_rim_nm}")
log.info(f"test mode           : {args.test}")
log.info("=" * 60)

# =================================================================
# Validation Helper Function
# =================================================================

def require_existing_file(path, label):
    """Stop early if a required input file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found at: {path}")


# =================================================================
# Environment and Thread Control
# =================================================================

n_threads = str(min(args.n_procs, 8))
safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': n_threads, 'OMP_NUM_THREADS': n_threads}

needs_itksnap = args.stage in ['segment', 'all']
needs_ants = args.stage in ['1', '2', 'all']
needs_matlab = args.stage in ['1', 'all']

if needs_itksnap and shutil.which('itksnap') is None:
    raise RuntimeError("ITK-SNAP not found on PATH. \n" \
    "Install from https://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4")

if needs_ants and shutil.which('antsRegistration') is None:
    raise RuntimeError("ANTs not found on path. \n" \
    "Install from https://github.com/antsx/ants")

if needs_matlab and shutil.which('matlab') is None:
    raise RuntimeError("Matlab not found on path. \n" \
    "Please ensure SPM is installed and on your MATLAB Path")
# =================================================================
# Execution Flags
# =================================================================

run_phase1 = args.stage in ['1', 'all']
run_segment = args.stage in ['segment', 'all']
run_phase2 = args.stage in ['2', 'all']
run_group_qc = args.stage in ['2', 'group_qc', 'all']
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
default_prob_atlas_mni = mask_dir / "CIT168_prob" / "CIT168toMNI152-2009c_prob_0006.nii.gz"
default_nms_atlas_mni = mask_dir / "NMS_prob" / "NMS_Snc_prob.nii.gz"
if not 0.0 <= float(args.prob_atlas_threshold) <= 1.0:
    raise ValueError("--prob_atlas_threshold must be between 0 and 1.")
if not 0.0 <= float(args.nms_atlas_threshold) <= 1.0:
    raise ValueError("--nms_atlas_threshold must be between 0 and 1.")

if args.skip_prob_atlas:
    prob_atlas_mni = None
elif args.prob_atlas_mni is not None:
    prob_atlas_mni = str(args.prob_atlas_mni)
else:
    prob_atlas_mni = str(default_prob_atlas_mni) if default_prob_atlas_mni.exists() else None

if args.skip_nms_atlas:
    nms_atlas_mni = None
elif args.nms_atlas_mni is not None:
    nms_atlas_mni = str(args.nms_atlas_mni)
else:
    nms_atlas_mni = str(default_nms_atlas_mni) if default_nms_atlas_mni.exists() else None

if manual_dir is not None and not manual_dir.exists():
    raise FileNotFoundError(f"--manual_dir specified but does not exist: {manual_dir}")

if run_phase1:
    require_existing_file(BrainStem_mask, "Brainstem Weighting Mask (Required for T1 to MNI registration)")

if run_phase2:
    if args.prob_atlas_mni is not None:
        require_existing_file(prob_atlas_mni, "User-specified probabilistic SNc atlas")
    elif prob_atlas_mni is None:
        log.info("No probabilistic SNc atlas found or requested; Stage 2 will skip ProbCIT QC.")
    if args.nms_atlas_mni is not None:
        require_existing_file(nms_atlas_mni, "User-specified NMS probabilistic SNc atlas")
    elif nms_atlas_mni is None:
        log.info("No NMS probabilistic SNc atlas found or requested; Stage 2 will skip NMS QC.")

if run_phase2 and not run_segment:
    require_existing_file(SN_mask, "SN mask missing... \n Run --stage segment first to create it")
    require_existing_file(CP_mask, "CP mask missing... \n Run --stage segment first to create it")

group_average_path = str(output_base / "Study-Specific_NM_Template.nii.gz")
if run_phase2 and not run_phase1:
    require_existing_file(group_average_path, "Group Average NM Template missing.. \n Run --stage 1 first to generate it")

# =================================================================
# Helper Functions
# =================================================================

def compute_motion_params(realignment_parameters):
    """Summarize SPM realignment parameters as compact motion QC JSON."""
    import os
    import json
    import numpy as np

    realignment_parameters = np.loadtxt(realignment_parameters)
    if realignment_parameters.ndim == 1:
        realignment_parameters = realignment_parameters[np.newaxis, :]

    translation_range_mm = realignment_parameters[:, :3].max(axis=0) - realignment_parameters[:, :3].min(axis=0)
    rotation_range_rad = realignment_parameters[:, 3:].max(axis=0) - realignment_parameters[:, 3:].min(axis=0)

    motion_summary = {
        'translation_range_mm': {
            'x': float(translation_range_mm[0]),
            'y': float(translation_range_mm[1]),
            'z': float(translation_range_mm[2]),
            'max': float(translation_range_mm.max()),
        },
        'rotation_range_deg': {
            'pitch': float(np.degrees(rotation_range_rad[0])),
            'roll': float(np.degrees(rotation_range_rad[1])),
            'yaw': float(np.degrees(rotation_range_rad[2])),
            'max': float(np.degrees(rotation_range_rad.max())),
        },
    }

    stats_file = os.path.abspath('motion_params.json')
    with open(stats_file, 'w') as f:
        json.dump(motion_summary, f, indent=2)

    return stats_file

def check_run_similarity(realigned_files):
    """Create a simple QC image comparing the first and second halves of NM runs."""
    import os
    import numpy as np
    from nilearn.image import mean_img, math_img
    from nilearn.plotting import plot_anat

    out_png = os.path.abspath('Run_Difference_QC.png')

    if len(realigned_files) < 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            'Run comparison skipped: only 1 run detected.',
            ha='center',
            va='center',
            fontsize=11,
            color='gray',
            transform=ax.transAxes,
        )
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png

    split_index = len(realigned_files) // 2
    first_half_mean_img = mean_img(realigned_files[:split_index])
    second_half_mean_img = mean_img(realigned_files[split_index:])

    first_half_data = first_half_mean_img.get_fdata()
    second_half_data = second_half_mean_img.get_fdata()
    run_correlation = np.corrcoef(first_half_data.ravel(), second_half_data.ravel())[0, 1]

    difference_img = math_img('img1 - img2', img1=first_half_mean_img, img2=second_half_mean_img)
    display = plot_anat(
        difference_img,
        title=f'Run 1 - Run 2 (r = {run_correlation:.3f})',
        display_mode='ortho',
        cmap='RdBu_r',
        draw_cross=False,
    )
    display.savefig(out_png, dpi=150)
    display.close()

    return out_png

def combine_transforms(t1_to_mni_composite, nm_to_t1_composite):
    """Return transforms in the order expected by ANTs ApplyTransforms."""
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

    atlas_img = nib.load(atlas_mask_mni)
    atlas_data = atlas_img.get_fdata() > 0

    # Dynamic cut coordinates: center the orthogonal slices on atlas center-of-mass.
    if np.any(atlas_data):
        vox_coords = np.argwhere(atlas_data)
        center_vox = vox_coords.mean(axis=0)
        cut_coords = nib.affines.apply_affine(atlas_img.affine, center_vox).tolist()
    else:
        cut_coords = [0, -15, -12]

    out_png = os.path.abspath('Registration_Atlas_on_T1Norm_QC.png')
    display = plotting.plot_anat(
        t1_mni_image,
        display_mode='ortho',
        cut_coords=cut_coords,
        dim=-0.5,
        cmap='gray',
        title='Atlas on T1-norm QC',
        draw_cross=False,
        colorbar=False,
    )
    display.add_overlay(atlas_mask_mni, cmap='Reds', alpha=0.8)
    display.savefig(out_png, dpi=300)
    display.close()
    return out_png

def extract_tractography_seed(
    nm_image,
    native_sn_mask,
    native_cp_mask,
    cnr_floor=0.05,
    cnr_percentile=60,
    min_cluster_vox=15,
    keep_components=2,
):
    """
    Create the subject-specific NM seed mask.

    The mask is built from NM contrast relative to the cerebral peduncle (CP).
    The final threshold is subject-adaptive: it keeps voxels above the fixed CNR
    floor and above a chosen percentile of this subject's SN-search CNR values.
    """
    import os
    import nibabel as nib
    import numpy as np
    import scipy.ndimage as ndi
    import matplotlib.pyplot as plt
    from nilearn.image import smooth_img
    from scipy.stats import gaussian_kde

    def keep_largest_components(mask, min_size, max_components):
        """Remove tiny islands and keep the largest connected components."""
        filled_mask = ndi.binary_fill_holes(mask)
        component_labels, n_components = ndi.label(filled_mask)
        if n_components == 0:
            return filled_mask

        component_sizes = ndi.sum(
            filled_mask,
            component_labels,
            index=np.arange(1, n_components + 1),
        )
        components_to_keep = [
            int(i + 1)
            for i, size in enumerate(component_sizes)
            if size >= int(min_size)
        ]
        if components_to_keep and int(max_components) > 0:
            components_to_keep = sorted(
                components_to_keep,
                key=lambda component_id: component_sizes[component_id - 1],
                reverse=True,
            )[: int(max_components)]

        if not components_to_keep:
            return np.zeros_like(filled_mask, dtype=bool)
        return np.isin(component_labels, components_to_keep)

    print("Computing CNR and extracting NM tractography seed...")

    nm_img = nib.load(nm_image)
    nm_data = nm_img.get_fdata()
    sn_mask = nib.load(native_sn_mask).get_fdata().astype(bool)
    cp_mask = nib.load(native_cp_mask).get_fdata().astype(bool)

    if not sn_mask.any() or not cp_mask.any():
        raise ValueError(f"Empty mask detected. SN voxels: {sn_mask.sum()} | CP voxels: {cp_mask.sum()}")

    cp_values = nm_data[cp_mask]
    cp_values = cp_values[np.isfinite(cp_values) & (cp_values > 0)]
    sn_values = nm_data[sn_mask]
    sn_values = sn_values[np.isfinite(sn_values) & (sn_values > 0)]

    if cp_values.size < 10 or sn_values.size < 10:
        raise ValueError(
            f"Insufficient positive intensity samples. CP: {cp_values.size}, SN: {sn_values.size}")

    try:
        cp_intensity_grid = np.linspace(cp_values.min(), cp_values.max(), 1000)
        cp_reference_intensity = float(cp_intensity_grid[np.argmax(gaussian_kde(cp_values)(cp_intensity_grid))])
    except Exception:
        cp_reference_intensity = float(np.median(cp_values))

    if cp_reference_intensity <= 0:
        cp_reference_intensity = float(np.mean(cp_values))

    cnr_data = np.zeros_like(nm_data)
    finite_nm_voxels = np.isfinite(nm_data)
    cnr_data[sn_mask & finite_nm_voxels] = (
        nm_data[sn_mask & finite_nm_voxels] - cp_reference_intensity
    ) / cp_reference_intensity
    cnr_img = nib.Nifti1Image(cnr_data, nm_img.affine, nm_img.header)

    smoothed_cnr_img = smooth_img(cnr_img, fwhm=1.0)
    smoothed_cnr_data = smoothed_cnr_img.get_fdata()

    sn_cnr_values = smoothed_cnr_data[sn_mask]
    sn_cnr_values = sn_cnr_values[np.isfinite(sn_cnr_values)]
    if sn_cnr_values.size == 0:
        raise ValueError("No finite CNR values found inside the native SN search mask.")
    adaptive_cnr_bound = float(np.percentile(sn_cnr_values, float(cnr_percentile)))
    lower_cnr_bound = max(float(cnr_floor), adaptive_cnr_bound)
    upper_cnr_bound = float(np.percentile(sn_cnr_values, 99))

    candidate_seed_mask = (
        sn_mask
        & np.isfinite(smoothed_cnr_data)
        & (smoothed_cnr_data >= lower_cnr_bound)
        & (smoothed_cnr_data <= upper_cnr_bound)
    )
    final_seed_mask = keep_largest_components(
        candidate_seed_mask,
        min_size=min_cluster_vox,
        max_components=keep_components,
    ).astype(np.uint8)

    cnr_out = os.path.abspath('Subject_CNR_Map_Smoothed.nii.gz')
    seed_out = os.path.abspath('Native_Tractography_Seed_SNc.nii.gz')
    nib.save(smoothed_cnr_img, cnr_out)
    nib.save(nib.Nifti1Image(final_seed_mask, nm_img.affine, nm_img.header), seed_out)

    all_values = np.concatenate([cp_values, sn_values])
    histogram_bins = np.linspace(np.percentile(all_values, 1), np.percentile(all_values, 99), 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cp_values, bins=histogram_bins, alpha=0.55, color='b', label='CP (background)', density=True)
    ax.hist(sn_values, bins=histogram_bins, alpha=0.55, color='y', label='SN (signal)', density=True)
    ax.axvline(
        cp_reference_intensity,
        color='m',
        lw=1.2,
        ls='--',
        label=f'CP mode = {cp_reference_intensity:.1f}',
    )
    ax.set_title(
        f'SN vs CP intensities | CNR threshold = {lower_cnr_bound:.3f} | '
        f'mean SN CNR = {np.mean(smoothed_cnr_data[sn_mask]):.3f} | '
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
    """Average the bias-corrected NM runs for a subject."""
    import os
    from nilearn.image import mean_img

    mean_image_path = os.path.abspath('mean_NM_bias_corrected.nii.gz')
    mean_img(in_files).to_filename(mean_image_path)
    return mean_image_path

def threshold_probability_mask(prob_atlas, threshold=0.25, label="ProbCIT_SNc"):
    """Threshold a native-space probabilistic atlas and save probability + binary maps."""
    import os
    import numpy as np
    import nibabel as nib

    probability_img = nib.load(prob_atlas)
    probability_data = probability_img.get_fdata()
    thresholded_mask = probability_data >= float(threshold)

    threshold_label = int(round(float(threshold) * 100))
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
    probability_out = os.path.abspath(f'Native_{safe_label}_Probability.nii.gz')
    mask_out = os.path.abspath(f'Native_{safe_label}_thr{threshold_label:03d}.nii.gz')

    mask_header = probability_img.header.copy()
    mask_header.set_data_dtype(np.uint8)
    nib.save(probability_img, probability_out)
    nib.save(nib.Nifti1Image(thresholded_mask.astype(np.uint8), probability_img.affine, mask_header), mask_out)

    return probability_out, mask_out


def create_core_rim_nm_mask(
    cnr_map,
    native_sn_mask,
    core_erosion_mm=1.0,
    core_cnr_threshold=0.05,
    rim_cnr_threshold=0.10,
    min_cluster_vox=25,
    keep_components=2,
):
    """
    Create a stricter NM mask by treating the SN core and rim differently.

    The core is allowed to use a lower CNR threshold because it is spatially
    more trustworthy. The rim uses a higher CNR threshold because boundary
    voxels are where false positives most often inflate HD95.
    """
    import os
    import json
    import numpy as np
    import nibabel as nib
    import scipy.ndimage as ndi

    cnr_img = nib.load(cnr_map)
    cnr_data = cnr_img.get_fdata()
    sn_img = nib.load(native_sn_mask)
    sn_mask = sn_img.get_fdata().astype(bool)

    if cnr_img.shape != sn_img.shape:
        raise ValueError(f"Shape mismatch: CNR {cnr_img.shape} vs native SN {sn_img.shape}.")
    if not np.allclose(cnr_img.affine, sn_img.affine, atol=1e-3):
        raise ValueError("Affine mismatch: CNR map and native SN mask not in same space.")

    voxel_size_mm = tuple(float(z) for z in cnr_img.header.get_zooms()[:3])
    finite_cnr_voxels = np.isfinite(cnr_data)

    if float(core_erosion_mm) > 0:
        distance_inside_sn_mm = ndi.distance_transform_edt(sn_mask, sampling=voxel_size_mm)
        core_mask = sn_mask & (distance_inside_sn_mm > float(core_erosion_mm))
    else:
        core_mask = sn_mask.copy()

    rim_mask = sn_mask & ~core_mask
    core_candidate_mask = core_mask & finite_cnr_voxels & (cnr_data >= float(core_cnr_threshold))
    rim_candidate_mask = rim_mask & finite_cnr_voxels & (cnr_data >= float(rim_cnr_threshold))
    raw_seed_mask = core_candidate_mask | rim_candidate_mask

    cleaned_mask = ndi.binary_fill_holes(raw_seed_mask)
    component_labels, n_components = ndi.label(cleaned_mask, structure=np.ones((3, 3, 3), dtype=bool))
    component_sizes_sorted = []
    components_to_keep = []

    if n_components > 0:
        component_sizes = ndi.sum(cleaned_mask, component_labels, index=np.arange(1, n_components + 1))
        component_sizes_sorted = sorted([int(size) for size in component_sizes], reverse=True)
        components_to_keep = [
            int(i + 1)
            for i, size in enumerate(component_sizes)
            if size >= int(min_cluster_vox)
        ]
        if components_to_keep and int(keep_components) > 0:
            components_to_keep = sorted(
                components_to_keep,
                key=lambda component_id: component_sizes[component_id - 1],
                reverse=True,
            )[: int(keep_components)]
        cleaned_mask = np.isin(component_labels, components_to_keep) if components_to_keep else np.zeros_like(cleaned_mask, dtype=bool)

    final_mask = cleaned_mask.astype(np.uint8)
    mask_header = cnr_img.header.copy()
    mask_header.set_data_dtype(np.uint8)

    mask_out = os.path.abspath("Native_CoreRim_NM_Mask.nii.gz")
    summary_file = os.path.abspath("CoreRim_NM_Summary.json")

    nib.save(nib.Nifti1Image(final_mask, cnr_img.affine, mask_header), mask_out)

    summary = {
        "method": "core_rim_nm",
        "core_erosion_mm": float(core_erosion_mm),
        "core_cnr_threshold": float(core_cnr_threshold),
        "rim_cnr_threshold": float(rim_cnr_threshold),
        "min_cluster_vox": int(min_cluster_vox),
        "keep_components": int(keep_components),
        "native_sn_voxels": int(sn_mask.sum()),
        "core_voxels": int(core_mask.sum()),
        "rim_voxels": int(rim_mask.sum()),
        "core_candidate_voxels": int(core_candidate_mask.sum()),
        "rim_candidate_voxels": int(rim_candidate_mask.sum()),
        "raw_mask_voxels": int(raw_seed_mask.sum()),
        "component_voxel_counts_before_cleanup_top5": component_sizes_sorted[:5],
        "kept_components": len(components_to_keep),
        "final_voxels": int(final_mask.sum()),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return mask_out, summary_file


def compute_mask_qc(
    cnr_mask,
    atlas_mask,
    anchored_atlas_mask=None,
    prob_atlas_mask=None,
    nms_atlas_mask=None,
    core_rim_nm_mask=None,
    manual_mask=None,
):
    import os
    import json
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt, binary_erosion, label

    def dice_coefficient(a, b):
        a, b = a.astype(bool), b.astype(bool)
        denom = a.sum() + b.sum()
        if denom == 0:
            return float('nan')
        return float(2 * (a & b).sum() / denom)

    def surface_distances_mm(a, b, spacing):
        a, b = a.astype(bool), b.astype(bool)
        if not a.any() or not b.any():
            return np.array([], dtype=float)
        surf_a = a & ~binary_erosion(a)
        surf_b = b & ~binary_erosion(b)
        if not surf_a.any() or not surf_b.any():
            return np.array([], dtype=float)
        dist_a_to_b = distance_transform_edt(~surf_b, sampling=spacing)[surf_a]
        dist_b_to_a = distance_transform_edt(~surf_a, sampling=spacing)[surf_b]
        return np.concatenate([dist_a_to_b, dist_b_to_a])

    def surface_distance_summary(a, b, spacing):
        distances = surface_distances_mm(a, b, spacing)
        if distances.size == 0:
            return {
                'surface_distance_n': 0,
                'surface_distance_mean_mm': float('nan'),
                'surface_distance_median_mm': float('nan'),
                'surface_distance_hd50_mm': float('nan'),
                'surface_distance_hd90_mm': float('nan'),
                'surface_distance_hd95_mm': float('nan'),
                'surface_distance_max_mm': float('nan'),
            }
        return {
            'surface_distance_n': int(distances.size),
            'surface_distance_mean_mm': float(np.mean(distances)),
            'surface_distance_median_mm': float(np.median(distances)),
            'surface_distance_hd50_mm': float(np.percentile(distances, 50)),
            'surface_distance_hd90_mm': float(np.percentile(distances, 90)),
            'surface_distance_hd95_mm': float(np.percentile(distances, 95)),
            'surface_distance_max_mm': float(np.max(distances)),
        }

    def component_summary(mask):
        mask = mask.astype(bool)
        n_voxels = int(mask.sum())
        if n_voxels == 0:
            return {
                'n_components': 0,
                'largest_component_voxels': 0,
                'largest_component_fraction': float('nan'),
                'component_voxel_counts_top5': [],
                'surface_voxels': 0,
                'surface_to_volume_ratio': float('nan'),
            }

        component_structure = np.ones((3, 3, 3), dtype=bool)
        labels, n_components = label(mask, structure=component_structure)
        counts = np.bincount(labels.ravel())[1:]
        counts_sorted = sorted([int(c) for c in counts], reverse=True)
        largest = counts_sorted[0] if counts_sorted else 0
        surface = mask & ~binary_erosion(mask, structure=component_structure)

        return {
            'n_components': int(n_components),
            'largest_component_voxels': int(largest),
            'largest_component_fraction': float(largest / n_voxels) if n_voxels > 0 else float('nan'),
            'component_voxel_counts_top5': counts_sorted[:5],
            'surface_voxels': int(surface.sum()),
            'surface_to_volume_ratio': float(surface.sum() / n_voxels) if n_voxels > 0 else float('nan'),
        }

    def centroid_mm(img, mask):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        return nib.affines.apply_affine(img.affine, coords.mean(axis=0)).tolist()

    def load_mask(path):
        img = nib.load(path)
        return img, img.get_fdata().astype(bool)

    def require_same_space(reference_img, moving_img, reference_label, moving_label):
        if reference_img.shape != moving_img.shape:
            raise ValueError(
                f"Shape mismatch: {reference_label} {reference_img.shape} vs "
                f"{moving_label} {moving_img.shape}."
            )
        if not np.allclose(reference_img.affine, moving_img.affine, atol=1e-3):
            raise ValueError(f"Affine mismatch: {reference_label} and {moving_label} not in same space.")

    def pairwise_mask_metrics(mask_a, mask_b, img, spacing):
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

        centroid_a = centroid_mm(img, a)
        centroid_b = centroid_mm(img, b)
        surface_summary = surface_distance_summary(a, b, spacing)
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
            'dice': dice_coefficient(a, b),
            'jaccard': jaccard,
            'precision_a_vs_b': precision,
            'recall_a_vs_b': recall,
            'hd95_mm': surface_summary['surface_distance_hd95_mm'],
            **surface_summary,
            'volume_a_mm3': vol_a_mm3,
            'volume_b_mm3': vol_b_mm3,
            'volume_delta_a_minus_b_mm3': vol_delta_mm3,
            'volume_delta_a_minus_b_percent': vol_delta_pct,
            'centroid_a_mm': centroid_a,
            'centroid_b_mm': centroid_b,
            'centroid_delta_a_minus_b_mm_xyz': centroid_delta_xyz,
            'centroid_distance_mm': centroid_distance_mm,
        }

    cnr_img, cnr_data = load_mask(cnr_mask)
    atlas_img, atlas_data = load_mask(atlas_mask)

    require_same_space(cnr_img, atlas_img, 'CNR', 'atlas')

    spacing = tuple(float(z) for z in cnr_img.header.get_zooms()[:3])

    pair_cnr_atlas = pairwise_mask_metrics(cnr_data, atlas_data, cnr_img, spacing)

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
        'n_voxels_prob_atlas': None,
        'dice_cnr_vs_prob_atlas': None,
        'hd95_cnr_vs_prob_atlas_mm': None,
        'centroid_prob_atlas_mm': None,
        'centroid_distance_cnr_vs_prob_atlas_mm': None,
        'n_voxels_nms_atlas': None,
        'dice_cnr_vs_nms_atlas': None,
        'hd95_cnr_vs_nms_atlas_mm': None,
        'centroid_nms_atlas_mm': None,
        'centroid_distance_cnr_vs_nms_atlas_mm': None,
        'n_voxels_core_rim_nm': None,
        'dice_cnr_vs_core_rim_nm': None,
        'hd95_cnr_vs_core_rim_nm_mm': None,
        'centroid_core_rim_nm_mm': None,
        'centroid_distance_cnr_vs_core_rim_nm_mm': None,
        'n_voxels_manual': None,
        'dice_cnr_vs_manual': None,
        'hd95_cnr_vs_manual_mm': None,
        'dice_atlas_vs_manual': None,
        'hd95_atlas_vs_manual_mm': None,
        'dice_prob_atlas_vs_manual': None,
        'hd95_prob_atlas_vs_manual_mm': None,
        'dice_nms_atlas_vs_manual': None,
        'hd95_nms_atlas_vs_manual_mm': None,
        'dice_core_rim_nm_vs_manual': None,
        'hd95_core_rim_nm_vs_manual_mm': None,
        'dice_core_rim_nm_vs_nms_atlas': None,
        'hd95_core_rim_nm_vs_nms_atlas_mm': None,
        'dice_anchored_vs_manual': None,
        'hd95_anchored_vs_manual_mm': None,
        'centroid_manual_mm': None,
        'centroid_distance_cnr_vs_manual_mm': None,
        'centroid_distance_atlas_vs_manual_mm': None,
        'centroid_distance_prob_atlas_vs_manual_mm': None,
        'centroid_distance_nms_atlas_vs_manual_mm': None,
        'centroid_distance_core_rim_nm_vs_manual_mm': None,
        'centroid_distance_core_rim_nm_vs_nms_atlas_mm': None,
        'pairwise': {
            'cnr_vs_atlas': pair_cnr_atlas,
        },
        'mask_components': {
            'cnr': component_summary(cnr_data),
            'atlas': component_summary(atlas_data),
            'anchored_atlas': None,
            'prob_atlas': None,
            'nms_atlas': None,
            'core_rim_nm': None,
            'manual': None,
        },
        'flags': [],
    }

    anch_data = None
    pair_cnr_anch = None
    if anchored_atlas_mask is not None and str(anchored_atlas_mask).strip():
        anch_img, anch_data = load_mask(anchored_atlas_mask)
        require_same_space(cnr_img, anch_img, 'CNR', 'anchored atlas')

        pair_cnr_anch = pairwise_mask_metrics(cnr_data, anch_data, cnr_img, spacing)
        qc['n_voxels_anchored_atlas'] = int(anch_data.sum())
        qc['dice_cnr_vs_anchored_atlas'] = pair_cnr_anch['dice']
        qc['hd95_cnr_vs_anchored_atlas_mm'] = pair_cnr_anch['hd95_mm']
        qc['centroid_anchored_atlas_mm'] = pair_cnr_anch['centroid_b_mm']
        qc['centroid_distance_cnr_vs_anchored_mm'] = pair_cnr_anch['centroid_distance_mm']
        qc['pairwise']['cnr_vs_anchored_atlas'] = pair_cnr_anch
        qc['mask_components']['anchored_atlas'] = component_summary(anch_data)

        raw_dice = qc['dice_cnr_vs_atlas']
        anch_dice = qc['dice_cnr_vs_anchored_atlas']
        if raw_dice is not None and anch_dice is not None and anch_dice < raw_dice - 0.05:
            qc['flags'].append('Anchoring degraded DICE vs CNR seed — check ROI mask and CNR quality')

    prob_data = None
    pair_cnr_prob = None
    if prob_atlas_mask is not None and str(prob_atlas_mask).strip():
        prob_img, prob_data = load_mask(prob_atlas_mask)
        require_same_space(cnr_img, prob_img, 'CNR', 'probabilistic atlas')

        pair_cnr_prob = pairwise_mask_metrics(cnr_data, prob_data, cnr_img, spacing)
        qc['n_voxels_prob_atlas'] = int(prob_data.sum())
        qc['dice_cnr_vs_prob_atlas'] = pair_cnr_prob['dice']
        qc['hd95_cnr_vs_prob_atlas_mm'] = pair_cnr_prob['hd95_mm']
        qc['centroid_prob_atlas_mm'] = pair_cnr_prob['centroid_b_mm']
        qc['centroid_distance_cnr_vs_prob_atlas_mm'] = pair_cnr_prob['centroid_distance_mm']
        qc['pairwise']['cnr_vs_prob_atlas'] = pair_cnr_prob
        qc['mask_components']['prob_atlas'] = component_summary(prob_data)

    nms_data = None
    pair_cnr_nms = None
    if nms_atlas_mask is not None and str(nms_atlas_mask).strip():
        nms_img, nms_data = load_mask(nms_atlas_mask)
        require_same_space(cnr_img, nms_img, 'CNR', 'NMS atlas')

        pair_cnr_nms = pairwise_mask_metrics(cnr_data, nms_data, cnr_img, spacing)
        qc['n_voxels_nms_atlas'] = int(nms_data.sum())
        qc['dice_cnr_vs_nms_atlas'] = pair_cnr_nms['dice']
        qc['hd95_cnr_vs_nms_atlas_mm'] = pair_cnr_nms['hd95_mm']
        qc['centroid_nms_atlas_mm'] = pair_cnr_nms['centroid_b_mm']
        qc['centroid_distance_cnr_vs_nms_atlas_mm'] = pair_cnr_nms['centroid_distance_mm']
        qc['pairwise']['cnr_vs_nms_atlas'] = pair_cnr_nms
        qc['mask_components']['nms_atlas'] = component_summary(nms_data)

    core_rim_data = None
    pair_cnr_core_rim = None
    if core_rim_nm_mask is not None and str(core_rim_nm_mask).strip():
        core_rim_img, core_rim_data = load_mask(core_rim_nm_mask)
        require_same_space(cnr_img, core_rim_img, 'CNR', 'core-rim NM mask')

        pair_cnr_core_rim = pairwise_mask_metrics(cnr_data, core_rim_data, cnr_img, spacing)
        qc['n_voxels_core_rim_nm'] = int(core_rim_data.sum())
        qc['dice_cnr_vs_core_rim_nm'] = pair_cnr_core_rim['dice']
        qc['hd95_cnr_vs_core_rim_nm_mm'] = pair_cnr_core_rim['hd95_mm']
        qc['centroid_core_rim_nm_mm'] = pair_cnr_core_rim['centroid_b_mm']
        qc['centroid_distance_cnr_vs_core_rim_nm_mm'] = pair_cnr_core_rim['centroid_distance_mm']
        qc['pairwise']['cnr_vs_core_rim_nm'] = pair_cnr_core_rim
        qc['mask_components']['core_rim_nm'] = component_summary(core_rim_data)

        if nms_data is not None:
            pair_core_rim_nms = pairwise_mask_metrics(core_rim_data, nms_data, cnr_img, spacing)
            qc['dice_core_rim_nm_vs_nms_atlas'] = pair_core_rim_nms['dice']
            qc['hd95_core_rim_nm_vs_nms_atlas_mm'] = pair_core_rim_nms['hd95_mm']
            qc['centroid_distance_core_rim_nm_vs_nms_atlas_mm'] = pair_core_rim_nms['centroid_distance_mm']
            qc['pairwise']['core_rim_nm_vs_nms_atlas'] = pair_core_rim_nms

    pair_cnr_manual = None
    pair_atlas_manual = None
    pair_prob_manual = None
    pair_nms_manual = None
    pair_core_rim_manual = None
    pair_anch_manual = None
    if manual_mask is not None and str(manual_mask).strip():
        man_img, man_data = load_mask(manual_mask)
        require_same_space(cnr_img, man_img, 'CNR', 'manual')

        pair_cnr_manual = pairwise_mask_metrics(cnr_data, man_data, cnr_img, spacing)
        pair_atlas_manual = pairwise_mask_metrics(atlas_data, man_data, cnr_img, spacing)
        if prob_data is not None:
            pair_prob_manual = pairwise_mask_metrics(prob_data, man_data, cnr_img, spacing)
        if nms_data is not None:
            pair_nms_manual = pairwise_mask_metrics(nms_data, man_data, cnr_img, spacing)
        if core_rim_data is not None:
            pair_core_rim_manual = pairwise_mask_metrics(core_rim_data, man_data, cnr_img, spacing)

        qc['n_voxels_manual'] = int(man_data.sum())
        qc['mask_components']['manual'] = component_summary(man_data)
        qc['dice_cnr_vs_manual'] = pair_cnr_manual['dice']
        qc['hd95_cnr_vs_manual_mm'] = pair_cnr_manual['hd95_mm']
        qc['dice_atlas_vs_manual'] = pair_atlas_manual['dice']
        qc['hd95_atlas_vs_manual_mm'] = pair_atlas_manual['hd95_mm']
        qc['centroid_manual_mm'] = pair_cnr_manual['centroid_b_mm']
        qc['centroid_distance_cnr_vs_manual_mm'] = pair_cnr_manual['centroid_distance_mm']
        qc['centroid_distance_atlas_vs_manual_mm'] = pair_atlas_manual['centroid_distance_mm']

        qc['pairwise']['cnr_vs_manual'] = pair_cnr_manual
        qc['pairwise']['atlas_vs_manual'] = pair_atlas_manual

        if pair_prob_manual is not None:
            qc['dice_prob_atlas_vs_manual'] = pair_prob_manual['dice']
            qc['hd95_prob_atlas_vs_manual_mm'] = pair_prob_manual['hd95_mm']
            qc['centroid_distance_prob_atlas_vs_manual_mm'] = pair_prob_manual['centroid_distance_mm']
            qc['pairwise']['prob_atlas_vs_manual'] = pair_prob_manual

        if pair_nms_manual is not None:
            qc['dice_nms_atlas_vs_manual'] = pair_nms_manual['dice']
            qc['hd95_nms_atlas_vs_manual_mm'] = pair_nms_manual['hd95_mm']
            qc['centroid_distance_nms_atlas_vs_manual_mm'] = pair_nms_manual['centroid_distance_mm']
            qc['pairwise']['nms_atlas_vs_manual'] = pair_nms_manual

        if pair_core_rim_manual is not None:
            qc['dice_core_rim_nm_vs_manual'] = pair_core_rim_manual['dice']
            qc['hd95_core_rim_nm_vs_manual_mm'] = pair_core_rim_manual['hd95_mm']
            qc['centroid_distance_core_rim_nm_vs_manual_mm'] = pair_core_rim_manual['centroid_distance_mm']
            qc['pairwise']['core_rim_nm_vs_manual'] = pair_core_rim_manual

        if anch_data is not None:
            pair_anch_manual = pairwise_mask_metrics(anch_data, man_data, cnr_img, spacing)
            qc['dice_anchored_vs_manual'] = pair_anch_manual['dice']
            qc['hd95_anchored_vs_manual_mm'] = pair_anch_manual['hd95_mm']
            qc['pairwise']['anchored_atlas_vs_manual'] = pair_anch_manual

        if qc['dice_cnr_vs_manual'] is not None and qc['dice_cnr_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE CNR vs manual — automated mask diverges from expert')
        if qc['dice_atlas_vs_manual'] is not None and qc['dice_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE atlas vs manual — registration or atlas fit may be poor')
        if qc['dice_prob_atlas_vs_manual'] is not None and qc['dice_prob_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE probabilistic atlas vs manual — threshold or atlas fit may be poor')
        if qc['dice_nms_atlas_vs_manual'] is not None and qc['dice_nms_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE NMS atlas vs manual — threshold or atlas fit may be poor')
        if qc['dice_core_rim_nm_vs_manual'] is not None and qc['dice_core_rim_nm_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE core-rim NM vs manual — boundary thresholds may need adjustment')
        if (
            qc['dice_core_rim_nm_vs_manual'] is not None
            and qc['dice_cnr_vs_manual'] is not None
            and qc['dice_core_rim_nm_vs_manual'] < qc['dice_cnr_vs_manual'] - 0.05
        ):
            qc['flags'].append('Core-rim NM reduced DICE vs original NM — check core/rim thresholds')
        if qc['hd95_cnr_vs_manual_mm'] is not None and qc['hd95_cnr_vs_manual_mm'] > 5.0:
            qc['flags'].append('High HD95 CNR vs manual (>5 mm)')
        if qc['hd95_core_rim_nm_vs_manual_mm'] is not None and qc['hd95_core_rim_nm_vs_manual_mm'] > 5.0:
            qc['flags'].append('High HD95 core-rim NM vs manual (>5 mm)')
        if qc['centroid_distance_cnr_vs_manual_mm'] is not None and qc['centroid_distance_cnr_vs_manual_mm'] > 3.0:
            qc['flags'].append('Large centroid offset CNR vs manual (>3 mm)')

        anchored_dice_vs_manual = qc.get('dice_anchored_vs_manual')
        atlas_dice_vs_manual = qc.get('dice_atlas_vs_manual')
        if (
            anchored_dice_vs_manual is not None
            and atlas_dice_vs_manual is not None
            and anchored_dice_vs_manual < atlas_dice_vs_manual - 0.05
        ):
            qc['flags'].append('Anchoring reduced DICE vs manual — CNR seed may be poorly defined')

    if qc['n_voxels_cnr'] < 20:
        qc['flags'].append('CNR seed very small (<20 voxels) — check CNR map and SN registration')
    cnr_components = qc['mask_components']['cnr']
    if cnr_components['n_components'] > 2:
        qc['flags'].append('CNR seed has >2 connected components — HD95 may be inflated by islands')
    if (
        cnr_components['largest_component_fraction'] is not None
        and np.isfinite(cnr_components['largest_component_fraction'])
        and cnr_components['largest_component_fraction'] < 0.8
    ):
        qc['flags'].append('CNR seed largest component <80% of mask — possible fragmentation')
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


def build_sn_roi_mask(cnr_seed, dilation_mm=6.0):
    """Build the search ROI used by the anchored-atlas registration step."""
    import os
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt

    seed_img = nib.load(cnr_seed)
    seed_mask = seed_img.get_fdata().astype(bool)
    if not seed_mask.any():
        raise ValueError("Cannot build anchor ROI: NM seed mask is empty.")

    voxel_size_mm = tuple(float(z) for z in seed_img.header.get_zooms()[:3])
    distance_from_seed_mm = distance_transform_edt(~seed_mask, sampling=voxel_size_mm)
    roi_mask = distance_from_seed_mm <= float(dilation_mm)

    roi_file = os.path.abspath('SN_ROI_Mask.nii.gz')
    roi_header = seed_img.header.copy()
    roi_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(roi_mask.astype(np.uint8), seed_img.affine, roi_header), roi_file)
    return roi_file


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
                            output_names=['stats_file'],
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
        (motion_node, datasink1, [('stats_file', 'metrics.@NM_MotionStats')]),

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

    if prob_atlas_mni is not None:
        mni_to_nm_prob_atlas = Node(ApplyTransforms(), name="MNI_ProbCIT_SNc_to_Native")
        mni_to_nm_prob_atlas.inputs.dimension = 3
        mni_to_nm_prob_atlas.inputs.interpolation = 'Linear'
        mni_to_nm_prob_atlas.inputs.input_image = prob_atlas_mni
        mni_to_nm_prob_atlas.inputs.environ = safe_env

        prob_atlas_thresh = Node(
            Function(
                input_names=['prob_atlas', 'threshold', 'label'],
                output_names=['prob_out', 'mask_out'],
                function=threshold_probability_mask,
            ),
            name='Threshold_ProbCIT_SNc',
        )
        prob_atlas_thresh.inputs.threshold = float(args.prob_atlas_threshold)
        prob_atlas_thresh.inputs.label = "ProbCIT_SNc"

    if nms_atlas_mni is not None:
        mni_to_nm_nms_atlas = Node(ApplyTransforms(), name="MNI_NMS_SNc_to_Native")
        mni_to_nm_nms_atlas.inputs.dimension = 3
        mni_to_nm_nms_atlas.inputs.interpolation = 'Linear'
        mni_to_nm_nms_atlas.inputs.input_image = nms_atlas_mni
        mni_to_nm_nms_atlas.inputs.environ = safe_env

        nms_atlas_thresh = Node(
            Function(
                input_names=['prob_atlas', 'threshold', 'label'],
                output_names=['prob_out', 'mask_out'],
                function=threshold_probability_mask,
            ),
            name='Threshold_NMS_SNc',
        )
        nms_atlas_thresh.inputs.threshold = float(args.nms_atlas_threshold)
        nms_atlas_thresh.inputs.label = "NMS_SNc"

    # =================================================================
    # Compute CNR + Tractography Seed
    # =================================================================
    cnr_seed = Node(Function(input_names=['nm_image', 'native_sn_mask', 'native_cp_mask', 'cnr_floor', 'cnr_percentile', 'min_cluster_vox', 'keep_components'], 
                            output_names=['cnr_out', 'seed_out', 'histogram'], function=extract_tractography_seed), name='CNR_Seed')
    cnr_seed.inputs.cnr_floor = seed_cnr_floor
    cnr_seed.inputs.cnr_percentile = seed_cnr_percentile
    cnr_seed.inputs.min_cluster_vox = seed_min_cluster_voxels
    cnr_seed.inputs.keep_components = seed_components_to_keep

    if not args.skip_core_rim_nm:
        core_rim_nm = Node(
            Function(
                input_names=[
                    'cnr_map',
                    'native_sn_mask',
                    'core_erosion_mm',
                    'core_cnr_threshold',
                    'rim_cnr_threshold',
                    'min_cluster_vox',
                    'keep_components',
                ],
                output_names=['mask_out', 'summary_file'],
                function=create_core_rim_nm_mask,
            ),
            name='Core_Rim_NM_Mask',
        )
        core_rim_nm.inputs.core_erosion_mm = core_rim_erosion_mm
        core_rim_nm.inputs.core_cnr_threshold = core_rim_core_cnr_threshold
        core_rim_nm.inputs.rim_cnr_threshold = core_rim_rim_cnr_threshold
        core_rim_nm.inputs.min_cluster_vox = core_rim_min_cluster_voxels
        core_rim_nm.inputs.keep_components = core_rim_components_to_keep

    # =================================================================
    # Build Anchoring Search ROI
    # =================================================================

    roi_mask_node = Node(
        Function(
            input_names=['cnr_seed', 'dilation_mm'],
            output_names=['roi_mask'],
            function=build_sn_roi_mask,
        ),
        name='Build_SN_ROI',
    )
    roi_mask_node.inputs.dilation_mm = 6.0

    # =================================================================
    # Anchor Atlas Mask to SN
    # =================================================================

    anchor_atlas = Node(Registration(), name='Anchor_CIT168_to_NM')
    anchor_atlas.inputs.environ = safe_env
    anchor_atlas.inputs.dimension = 3
    anchor_atlas.inputs.interpolation = 'NearestNeighbor'
    # Constrained anchor: align the native atlas to the NM-defined seed mask.
    # This preserves the atlas boundary and avoids SyN overfitting to a small/noisy seed.
    anchor_atlas.inputs.transforms = ['Rigid']
    anchor_atlas.inputs.transform_parameters = [(0.05,)]
    anchor_atlas.inputs.number_of_iterations = [[500, 250, 100]]
    anchor_atlas.inputs.metric = ['MeanSquares']
    anchor_atlas.inputs.metric_weight = [1]
    anchor_atlas.inputs.radius_or_number_of_bins = [0]
    anchor_atlas.inputs.shrink_factors = [[4, 2, 1]]
    anchor_atlas.inputs.smoothing_sigmas = [[1, 0.5, 0]]
    anchor_atlas.inputs.sigma_units = ['vox']
    anchor_atlas.inputs.sampling_percentage = [1]
    anchor_atlas.inputs.sampling_strategy = ['None']
    anchor_atlas.inputs.convergence_threshold = [1e-6]
    anchor_atlas.inputs.convergence_window_size = [10]
    anchor_atlas.inputs.winsorize_lower_quantile = 0.005
    anchor_atlas.inputs.winsorize_upper_quantile = 0.995
    anchor_atlas.inputs.use_histogram_matching = [False]
    anchor_atlas.inputs.initial_moving_transform_com = 1

    anchor_atlas.inputs.output_warped_image = True
    anchor_atlas.inputs.write_composite_transform = True
    anchor_atlas.inputs.collapse_output_transforms = True
    
    # =================================================================
    # QC
    # =================================================================

    qc_metric_inputs = [
        'cnr_mask',
        'atlas_mask',
        'anchored_atlas_mask',
        'prob_atlas_mask',
        'nms_atlas_mask',
        'core_rim_nm_mask',
        'manual_mask',
    ]
    qc_node = Node(
        Function(
            input_names=qc_metric_inputs,
            output_names=['qc_file'],
            function=compute_mask_qc,
        ),
        name='QC_Metrics',
    )
    qc_node.inputs.prob_atlas_mask = None
    qc_node.inputs.nms_atlas_mask = None
    qc_node.inputs.core_rim_nm_mask = None


    if manual_dir is not None:
        ManualData = {'manual_sn': '{subject_id}/anat/{subject_id}_space-NM_label-SN_desc-manual_mask.nii*'}
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

        (cnr_seed,          anchor_atlas, [('seed_out', 'fixed_image')]),
        (mni_to_nm_atlas,   anchor_atlas, [('output_image', 'moving_image')]),
        (roi_mask_node,     anchor_atlas, [('roi_mask', 'fixed_image_masks')]),
        

        (anchor_atlas,      qc_node, [('warped_image', 'anchored_atlas_mask')]),


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
    ]

    if prob_atlas_mni is not None:
        wf2_connections += [
            (GrabPhase1,          mni_to_nm_prob_atlas, [('bias_nm', 'reference_image')]),
            (nm_to_template,      mni_to_nm_prob_atlas, [('inverse_composite_transform', 'transforms')]),
            (mni_to_nm_prob_atlas, prob_atlas_thresh, [('output_image', 'prob_atlas')]),
            (prob_atlas_thresh,   qc_node, [('mask_out', 'prob_atlas_mask')]),
            (prob_atlas_thresh,   datasink2, [('prob_out', 'anat.@ProbCIT_SNc_Probability'),
                                              ('mask_out', 'anat.@ProbCIT_SNc_Mask')]),
        ]

    if nms_atlas_mni is not None:
        wf2_connections += [
            (GrabPhase1,          mni_to_nm_nms_atlas, [('bias_nm', 'reference_image')]),
            (nm_to_template,      mni_to_nm_nms_atlas, [('inverse_composite_transform', 'transforms')]),
            (mni_to_nm_nms_atlas, nms_atlas_thresh, [('output_image', 'prob_atlas')]),
            (nms_atlas_thresh,    qc_node, [('mask_out', 'nms_atlas_mask')]),
            (nms_atlas_thresh,    datasink2, [('prob_out', 'anat.@NMS_SNc_Probability'),
                                              ('mask_out', 'anat.@NMS_SNc_Mask')]),
        ]

    if not args.skip_core_rim_nm:
        wf2_connections += [
            (cnr_seed,     core_rim_nm, [('cnr_out', 'cnr_map')]),
            (mni_to_nm_sn, core_rim_nm, [('output_image', 'native_sn_mask')]),
            (core_rim_nm,  qc_node, [('mask_out', 'core_rim_nm_mask')]),
            (core_rim_nm,  datasink2, [('mask_out', 'anat.@CoreRim_NM_Mask'),
                                       ('summary_file', 'qc.@CoreRim_NM_Summary')]),
        ]

    if manual_dir is not None:
        wf2_connections += [
            (inputnode2,    SelectManual, [('subject_id', 'subject_id')]),
            (SelectManual,  qc_node, [('manual_sn', 'manual_mask')]),

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
# Stage 2 QC Reports
# =================================================================


def build_stage2_output_manifest(subject_id, output_path, manual_dir=None):
    """Write a plain audit file describing this subject's Stage 2 outputs."""
    subject_dir = Path(output_path) / subject_id
    anat_dir = subject_dir / "anat"
    cnr_dir = subject_dir / "cnr"
    qc_dir = subject_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    prob_threshold_label = int(round(float(args.prob_atlas_threshold) * 100))
    nms_threshold_label = int(round(float(args.nms_atlas_threshold) * 100))

    reference_nm_image_path = anat_dir / "mean_NM_bias_corrected.nii.gz"
    if not reference_nm_image_path.exists():
        reference_nm_image_path = anat_dir / "mean_NM_bias_corrected.nii"
    if not reference_nm_image_path.exists():
        reference_nm_image_path = None

    nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii.gz"
    if not nm_seed_mask.exists():
        nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii"
    if not nm_seed_mask.exists():
        nm_seed_mask = None

    core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii.gz"
    if not core_rim_mask.exists():
        core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii"
    if not core_rim_mask.exists():
        core_rim_mask = None

    atlas_mask = anat_dir / "SNc_trans.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc_trans.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = None

    prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii"
    if not prob_cit_mask.exists():
        prob_matches = sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii.gz"))
        prob_matches += sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii"))
        prob_cit_mask = prob_matches[0] if prob_matches else None

    nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii"
    if not nms_mask.exists():
        nms_matches = sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii.gz"))
        nms_matches += sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii"))
        nms_mask = nms_matches[0] if nms_matches else None

    anchored_mask = anat_dir / "transform_Warped.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "transform_Warped.nii"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "CIT168_Anchored.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = None

    manual_mask = None
    if manual_dir is not None:
        manual_anat_dir = Path(manual_dir) / subject_id / "anat"
        manual_matches = sorted(manual_anat_dir.glob(f"{subject_id}_space-NM_label-SN_desc-manual_mask.nii*"))
        manual_mask = manual_matches[0] if manual_matches else None

    mask_paths = {
        "nm_seed": nm_seed_mask,
        "core_rim_nm": core_rim_mask,
        "atlas": atlas_mask,
        "prob_cit": prob_cit_mask,
        "nms": nms_mask,
        "anchored_atlas": anchored_mask,
        "manual": manual_mask,
    }

    manifest_warnings = []
    reference_shape = None
    reference_affine = None

    if reference_nm_image_path is None:
        manifest_warnings.append("Missing native NM reference image: mean_NM_bias_corrected.nii[.gz]")
    else:
        try:
            reference_img = nib.load(str(reference_nm_image_path))
            reference_shape = tuple(int(v) for v in reference_img.shape)
            reference_affine = reference_img.affine
        except Exception as exc:
            manifest_warnings.append(f"Could not load native NM reference image: {exc}")

    mask_records = {}
    for mask_name, mask_path in mask_paths.items():
        mask_exists = mask_path is not None and Path(mask_path).exists()
        mask_shape = None
        mask_voxel_count = None
        shape_matches_reference = None
        affine_matches_reference = None
        load_error = None

        if mask_exists:
            try:
                mask_img = nib.load(str(mask_path))
                mask_data = mask_img.get_fdata()
                mask_shape = [int(v) for v in mask_img.shape]
                mask_voxel_count = int(np.count_nonzero(mask_data))

                if reference_shape is not None:
                    shape_matches_reference = bool(tuple(mask_img.shape) == reference_shape)
                    if not shape_matches_reference:
                        manifest_warnings.append(f"{mask_name} shape does not match native NM reference")

                if reference_affine is not None:
                    affine_matches_reference = bool(np.allclose(mask_img.affine, reference_affine, atol=1e-3))
                    if not affine_matches_reference:
                        manifest_warnings.append(f"{mask_name} affine does not match native NM reference")
            except Exception as exc:
                load_error = str(exc)
                manifest_warnings.append(f"Could not inspect {mask_name}: {exc}")

        mask_records[mask_name] = {
            "path": str(mask_path) if mask_path is not None else None,
            "exists": mask_exists,
            "shape": mask_shape,
            "voxel_count": mask_voxel_count,
            "shape_matches_reference": shape_matches_reference,
            "affine_matches_reference": affine_matches_reference,
            "error": load_error,
        }

    required_mask_names = ["nm_seed", "atlas"]
    missing_required_masks = [name for name in required_mask_names if not mask_records[name]["exists"]]
    if manual_dir is not None and not mask_records["manual"]["exists"]:
        manifest_warnings.append("Manual mask directory was provided, but this subject has no matching manual SN mask")

    manifest = {}
    manifest["subject_id"] = subject_id
    manifest["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest["status"] = "complete" if not missing_required_masks else "incomplete"
    manifest["native_nm_reference"] = {
        "path": str(reference_nm_image_path) if reference_nm_image_path is not None else None,
        "exists": reference_nm_image_path is not None and reference_nm_image_path.exists(),
        "shape": list(reference_shape) if reference_shape is not None else None,
    }
    manifest["stage2_qc_json"] = {
        "path": str(cnr_dir / "SN_Mask_QC.json"),
        "exists": (cnr_dir / "SN_Mask_QC.json").exists(),
    }
    manifest["masks"] = mask_records
    manifest["missing_required_masks"] = missing_required_masks
    manifest["warnings"] = manifest_warnings

    manifest_path = qc_dir / "Stage2_Output_Manifest.json"
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    return manifest_path


def build_subject_qc_dashboard(subject_id, output_path, manual_dir=None):
    """Build the per-subject dashboard from existing Stage 2 files."""
    subject_dir = Path(output_path) / subject_id
    anat_dir = subject_dir / "anat"
    cnr_dir = subject_dir / "cnr"
    qc_dir = subject_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    prob_threshold_label = int(round(float(args.prob_atlas_threshold) * 100))
    nms_threshold_label = int(round(float(args.nms_atlas_threshold) * 100))

    base_img_path = anat_dir / "mean_NM_bias_corrected.nii.gz"
    if not base_img_path.exists():
        base_img_path = anat_dir / "mean_NM_bias_corrected.nii"
    if not base_img_path.exists():
        base_img_path = anat_dir / "NM_MNI.nii"

    qc_json = cnr_dir / "SN_Mask_QC.json"
    if not base_img_path.exists() or not qc_json.exists():
        return None

    manual_mask = None
    if manual_dir is not None:
        manual_anat_dir = Path(manual_dir) / subject_id / "anat"
        manual_matches = sorted(manual_anat_dir.glob(f"{subject_id}_space-NM_label-SN_desc-manual_mask.nii*"))
        manual_mask = manual_matches[0] if manual_matches else None

    nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii.gz"
    if not nm_seed_mask.exists():
        nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii"
    if not nm_seed_mask.exists():
        nm_seed_mask = None

    core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii.gz"
    if not core_rim_mask.exists():
        core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii"
    if not core_rim_mask.exists():
        core_rim_mask = None

    atlas_mask = anat_dir / "SNc_trans.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc_trans.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = None

    prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii"
    if not prob_cit_mask.exists():
        prob_matches = sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii.gz"))
        prob_matches += sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii"))
        prob_cit_mask = prob_matches[0] if prob_matches else None

    nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii"
    if not nms_mask.exists():
        nms_matches = sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii.gz"))
        nms_matches += sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii"))
        nms_mask = nms_matches[0] if nms_matches else None

    anchored_mask = anat_dir / "transform_Warped.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "transform_Warped.nii"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "CIT168_Anchored.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = None

    display_mask_paths = {
        "Manual": manual_mask,
        "NM": nm_seed_mask,
        "Core-rim": core_rim_mask,
        "Atlas": atlas_mask,
        "ProbCIT": prob_cit_mask,
        "NMS": nms_mask,
        "Anchored": anchored_mask,
    }

    base_img = nib.load(str(base_img_path))
    base_data = base_img.get_fdata()
    reference_shape = base_data.shape

    masks = {}
    for label, mask_path in display_mask_paths.items():
        masks[label] = None
        if mask_path is None:
            continue

        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata().astype(bool)
        if mask_data.shape == reference_shape:
            masks[label] = mask_data

    valid_masks = [mask for mask in masks.values() if mask is not None and mask.any()]
    if valid_masks:
        mask_union = np.logical_or.reduce(valid_masks)
        z_values = np.argwhere(mask_union)[:, 2]
        display_slice = int(np.clip(np.median(z_values), 0, reference_shape[2] - 1))
    else:
        display_slice = reference_shape[2] // 2

    with open(qc_json, "r") as qc_file:
        qc_data = json.load(qc_file)

    color_map = {
        "Manual": "#FFD166",
        "NM": "#EF476F",
        "Core-rim": "#2EC4B6",
        "Atlas": "#118AB2",
        "ProbCIT": "#F28E2B",
        "NMS": "#8E6C8A",
        "Anchored": "#06D6A0",
    }

    finite_base = base_data[np.isfinite(base_data)]
    if finite_base.size:
        vmin, vmax = np.percentile(finite_base, [2, 98])
    else:
        vmin, vmax = 0, 1

    def draw_mask_panel(ax, panel_title, panel_labels):
        ax.imshow(np.rot90(base_data[:, :, display_slice]), cmap="gray", vmin=vmin, vmax=vmax)
        for label in panel_labels:
            mask = masks.get(label)
            if mask is None or not mask[:, :, display_slice].any():
                continue
            ax.contour(
                np.rot90(mask[:, :, display_slice]),
                levels=[0.5],
                colors=[color_map[label]],
                linewidths=2.0 if label == "Manual" else 1.5,
                linestyles="--" if label == "Manual" else "-",
            )
        ax.set_title(f"{panel_title} | z={display_slice}", fontsize=10)
        ax.axis("off")

    comparison_panels = [
        ("Manual vs NM", ["NM", "Manual"]),
        ("Manual vs Core-rim", ["Core-rim", "Manual"]),
        ("Manual vs Atlas", ["Atlas", "Manual"]),
        ("Manual vs ProbCIT", ["ProbCIT", "Manual"]),
        ("Manual vs NMS", ["NMS", "Manual"]),
        ("Manual vs Anchored", ["Anchored", "Manual"]),
        ("All Masks Overview", ["Atlas", "ProbCIT", "NMS", "Anchored", "NM", "Core-rim", "Manual"]),
    ]

    manual_dice_metrics = [
        ("NM", "dice_cnr_vs_manual"),
        ("Core-rim", "dice_core_rim_nm_vs_manual"),
        ("Atlas", "dice_atlas_vs_manual"),
        ("ProbCIT", "dice_prob_atlas_vs_manual"),
        ("NMS", "dice_nms_atlas_vs_manual"),
        ("Anchored", "dice_anchored_vs_manual"),
    ]
    fallback_dice_metrics = [
        ("NM vs Atlas", "dice_cnr_vs_atlas"),
        ("NM vs Core-rim", "dice_cnr_vs_core_rim_nm"),
        ("NM vs ProbCIT", "dice_cnr_vs_prob_atlas"),
        ("NM vs NMS", "dice_cnr_vs_nms_atlas"),
        ("NM vs Anchored", "dice_cnr_vs_anchored_atlas"),
    ]

    manual_hd95_metrics = [
        ("NM", "hd95_cnr_vs_manual_mm"),
        ("Core-rim", "hd95_core_rim_nm_vs_manual_mm"),
        ("Atlas", "hd95_atlas_vs_manual_mm"),
        ("ProbCIT", "hd95_prob_atlas_vs_manual_mm"),
        ("NMS", "hd95_nms_atlas_vs_manual_mm"),
        ("Anchored", "hd95_anchored_vs_manual_mm"),
    ]
    fallback_hd95_metrics = [
        ("NM vs Atlas", "hd95_cnr_vs_atlas_mm"),
        ("NM vs Core-rim", "hd95_cnr_vs_core_rim_nm_mm"),
        ("NM vs ProbCIT", "hd95_cnr_vs_prob_atlas_mm"),
        ("NM vs NMS", "hd95_cnr_vs_nms_atlas_mm"),
        ("NM vs Anchored", "hd95_cnr_vs_anchored_atlas_mm"),
    ]

    dice_metrics = manual_dice_metrics
    if all(qc_data.get(metric_key) is None for _, metric_key in manual_dice_metrics):
        dice_metrics = fallback_dice_metrics

    hd95_metrics = manual_hd95_metrics
    if all(qc_data.get(metric_key) is None for _, metric_key in manual_hd95_metrics):
        hd95_metrics = fallback_hd95_metrics

    dice_labels = [label for label, _ in dice_metrics]
    dice_values = []
    for _, metric_key in dice_metrics:
        try:
            value = float(qc_data.get(metric_key))
        except (TypeError, ValueError):
            value = np.nan
        dice_values.append(value if np.isfinite(value) else np.nan)

    hd95_labels = [label for label, _ in hd95_metrics]
    hd95_values = []
    for _, metric_key in hd95_metrics:
        try:
            value = float(qc_data.get(metric_key))
        except (TypeError, ValueError):
            value = np.nan
        hd95_values.append(value if np.isfinite(value) else np.nan)

    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 7, height_ratios=[1.4, 1.0])

    for panel_idx, (panel_title, panel_labels) in enumerate(comparison_panels):
        draw_mask_panel(fig.add_subplot(grid[0, panel_idx]), panel_title, panel_labels)

    ax_dice = fig.add_subplot(grid[1, 0:3])
    dice_colors = [color_map.get(label, "#888888") for label in dice_labels]
    ax_dice.bar(dice_labels, dice_values, color=dice_colors)
    ax_dice.set_ylim(0, 1)
    ax_dice.set_ylabel("DICE")
    ax_dice.set_title("Overlap with manual segmentation" if dice_labels[0] == "NM" else "Overlap QC")
    ax_dice.axhline(0.5, color="firebrick", lw=0.8, ls="--", alpha=0.6)
    ax_dice.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(dice_values):
        if np.isfinite(value):
            ax_dice.text(idx, value + 0.03, f"{value:.2f}", ha="center", fontsize=9)

    ax_hd95 = fig.add_subplot(grid[1, 3:7])
    hd95_colors = [color_map.get(label, "#888888") for label in hd95_labels]
    ax_hd95.bar(hd95_labels, hd95_values, color=hd95_colors)
    ax_hd95.set_ylabel("HD95 (mm)")
    ax_hd95.set_title("Boundary distance with manual segmentation" if hd95_labels[0] == "NM" else "Boundary QC")
    ax_hd95.axhline(5.0, color="firebrick", lw=0.8, ls="--", alpha=0.6)
    ax_hd95.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(hd95_values):
        if np.isfinite(value):
            ax_hd95.text(idx, value + 0.2, f"{value:.1f}", ha="center", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], color=color_map["Manual"], lw=2, ls="--", label="Manual"),
        plt.Line2D([0], [0], color=color_map["NM"], lw=2, ls="-", label="NM"),
        plt.Line2D([0], [0], color=color_map["Core-rim"], lw=2, ls="-", label="Core-rim"),
        plt.Line2D([0], [0], color=color_map["Atlas"], lw=2, ls="-", label="Atlas"),
        plt.Line2D([0], [0], color=color_map["ProbCIT"], lw=2, ls="-", label="ProbCIT"),
        plt.Line2D([0], [0], color=color_map["NMS"], lw=2, ls="-", label="NMS"),
        plt.Line2D([0], [0], color=color_map["Anchored"], lw=2, ls="-", label="Anchored"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7, frameon=False)
    fig.suptitle(f"{subject_id} Subject QC Dashboard", fontsize=14)

    out_png = qc_dir / "Subject_QC_Dashboard.png"
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_png


GROUP_QC_COMPARISONS = [
    {
        "pair": "cnr_vs_atlas",
        "label": "CNR vs Atlas",
        "dice": "dice_cnr_vs_atlas",
        "hd95": "hd95_cnr_vs_atlas_mm",
        "color": "#4C72B0",
        "marker": "o",
    },
    {
        "pair": "cnr_vs_core_rim_nm",
        "label": "NM vs Core-rim",
        "dice": "dice_cnr_vs_core_rim_nm",
        "hd95": "hd95_cnr_vs_core_rim_nm_mm",
        "color": "#2EC4B6",
        "marker": "*",
    },
    {
        "pair": "cnr_vs_prob_atlas",
        "label": "CNR vs ProbCIT",
        "dice": "dice_cnr_vs_prob_atlas",
        "hd95": "hd95_cnr_vs_prob_atlas_mm",
        "color": "#F28E2B",
        "marker": "P",
    },
    {
        "pair": "cnr_vs_nms_atlas",
        "label": "CNR vs NMS",
        "dice": "dice_cnr_vs_nms_atlas",
        "hd95": "hd95_cnr_vs_nms_atlas_mm",
        "color": "#8E6C8A",
        "marker": "h",
    },
    {
        "pair": "cnr_vs_anchored_atlas",
        "label": "CNR vs Anchored",
        "dice": "dice_cnr_vs_anchored_atlas",
        "hd95": "hd95_cnr_vs_anchored_atlas_mm",
        "color": "#E1812C",
        "marker": "D",
    },
]


GROUP_QC_MANUAL_COMPARISONS = [
    {
        "pair": "cnr_vs_manual",
        "label": "CNR vs Manual",
        "dice": "dice_cnr_vs_manual",
        "hd95": "hd95_cnr_vs_manual_mm",
        "color": "#DD8452",
        "marker": "s",
    },
    {
        "pair": "core_rim_nm_vs_manual",
        "label": "Core-rim vs Manual",
        "dice": "dice_core_rim_nm_vs_manual",
        "hd95": "hd95_core_rim_nm_vs_manual_mm",
        "color": "#2EC4B6",
        "marker": "*",
    },
    {
        "pair": "atlas_vs_manual",
        "label": "Atlas vs Manual",
        "dice": "dice_atlas_vs_manual",
        "hd95": "hd95_atlas_vs_manual_mm",
        "color": "#55A868",
        "marker": "^",
    },
    {
        "pair": "prob_atlas_vs_manual",
        "label": "ProbCIT vs Manual",
        "dice": "dice_prob_atlas_vs_manual",
        "hd95": "hd95_prob_atlas_vs_manual_mm",
        "color": "#F28E2B",
        "marker": "X",
    },
    {
        "pair": "nms_atlas_vs_manual",
        "label": "NMS vs Manual",
        "dice": "dice_nms_atlas_vs_manual",
        "hd95": "hd95_nms_atlas_vs_manual_mm",
        "color": "#8E6C8A",
        "marker": "*",
    },
    {
        "pair": "anchored_atlas_vs_manual",
        "label": "Anchored vs Manual",
        "dice": "dice_anchored_vs_manual",
        "hd95": "hd95_anchored_vs_manual_mm",
        "color": "#151372",
        "marker": "v",
    },
]


GROUP_QC_CORE_RIM_NMS_COMPARISON = {
    "pair": "core_rim_nm_vs_nms_atlas",
    "label": "Core-rim vs NMS",
}


GROUP_QC_COMPONENT_LABELS = {
    "cnr": "NM",
    "core_rim_nm": "Core-rim NM",
    "atlas": "Atlas",
    "prob_atlas": "ProbCIT",
    "nms_atlas": "NMS",
    "anchored_atlas": "Anchored",
    "manual": "Manual",
}


GROUP_QC_PAIR_ORDER = [
    "cnr_vs_atlas",
    "cnr_vs_core_rim_nm",
    "cnr_vs_prob_atlas",
    "cnr_vs_nms_atlas",
    "cnr_vs_anchored_atlas",
    "cnr_vs_manual",
    "core_rim_nm_vs_manual",
    "core_rim_nm_vs_nms_atlas",
    "atlas_vs_manual",
    "prob_atlas_vs_manual",
    "nms_atlas_vs_manual",
    "anchored_atlas_vs_manual",
]


GROUP_QC_PAIRWISE_FIELDS = [
    "subject_id", "pair", "dice", "jaccard", "precision", "recall",
    "hd95_mm", "surface_distance_mean_mm", "surface_distance_median_mm",
    "surface_distance_hd90_mm", "surface_distance_hd95_mm",
    "surface_distance_max_mm", "centroid_distance_mm",
    "volume_a_mm3", "volume_b_mm3", "volume_delta_mm3", "volume_delta_percent",
]


GROUP_QC_COMPONENT_FIELDS = [
    "subject_id", "mask", "n_components", "largest_component_voxels",
    "largest_component_fraction", "surface_voxels", "surface_to_volume_ratio",
    "component_voxel_counts_top5",
]


# =================================================================
# Group QC Plotting
# =================================================================

def run_group_qc_stage(output_path, manual_dir, selected_subjects, test_mode=False):
    """Generate subject dashboards, audit manifests, and group-level QC plots.

    This function intentionally works only from existing Stage 2 outputs. It should
    never trigger registration or mask generation, which makes it safe to rerun while
    inspecting and refining plots.
    """
    output_path = Path(output_path)
    log.info(f"{'=' * 60}")
    log.info("Generating Group QC Plots...")

    selected_subject_ids = set(selected_subjects)
    qc_files = sorted(output_path.glob("sub-*/cnr/SN_Mask_QC.json"))
    qc_files = [qc_file for qc_file in qc_files if qc_file.parts[-3] in selected_subject_ids]

    def group_qc_output_path(filename):
        """Return output paths for group QC, using _TEST names in test mode."""
        requested_path = Path(filename)
        if test_mode:
            return output_path / f"{requested_path.stem}_TEST{requested_path.suffix}"
        return output_path / requested_path.name

    if not qc_files:
        log.info("No QC JSON files found — skipping group QC plots.")
        return

    subject_records = []
    pairwise_rows = []
    component_rows = []

    def finite_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def metric_values(metric_key):
        return [subject_qc.get(metric_key) for _, subject_qc in subject_records]

    def metric_has_values(metric_key):
        return any(finite_float(value) is not None for value in metric_values(metric_key))

    def paired_metric_rows(metric_a, metric_b):
        rows = []
        for subject_id, subject_qc in subject_records:
            value_a = finite_float(subject_qc.get(metric_a))
            value_b = finite_float(subject_qc.get(metric_b))
            if value_a is not None and value_b is not None:
                rows.append((subject_id, value_a, value_b))
        return rows

    def clean_csv_value(value):
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        return value

    def write_csv(path, fieldnames, rows):
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: clean_csv_value(row.get(key)) for key in fieldnames})

    def add_component_rows(subject_id, subject_qc):
        for component_key, component_metrics in subject_qc.get("mask_components", {}).items():
            if not isinstance(component_metrics, dict):
                continue
            component_rows.append({
                "subject_id": subject_id,
                "mask": GROUP_QC_COMPONENT_LABELS.get(component_key, component_key),
                "n_components": component_metrics.get("n_components"),
                "largest_component_voxels": component_metrics.get("largest_component_voxels"),
                "largest_component_fraction": component_metrics.get("largest_component_fraction"),
                "surface_voxels": component_metrics.get("surface_voxels"),
                "surface_to_volume_ratio": component_metrics.get("surface_to_volume_ratio"),
                "component_voxel_counts_top5": ";".join(
                    str(v) for v in component_metrics.get("component_voxel_counts_top5", [])
                ),
            })

    def add_pairwise_rows(subject_id, subject_qc):
        for pair_name, metrics in subject_qc.get("pairwise", {}).items():
            if not isinstance(metrics, dict):
                continue
            pairwise_rows.append({
                "subject_id": subject_id,
                "pair": pair_name,
                "dice": metrics.get("dice"),
                "jaccard": metrics.get("jaccard"),
                "precision": metrics.get("precision_a_vs_b"),
                "recall": metrics.get("recall_a_vs_b"),
                "hd95_mm": metrics.get("hd95_mm"),
                "surface_distance_mean_mm": metrics.get("surface_distance_mean_mm"),
                "surface_distance_median_mm": metrics.get("surface_distance_median_mm"),
                "surface_distance_hd90_mm": metrics.get("surface_distance_hd90_mm"),
                "surface_distance_hd95_mm": metrics.get("surface_distance_hd95_mm"),
                "surface_distance_max_mm": metrics.get("surface_distance_max_mm"),
                "centroid_distance_mm": metrics.get("centroid_distance_mm"),
                "volume_a_mm3": metrics.get("volume_a_mm3"),
                "volume_b_mm3": metrics.get("volume_b_mm3"),
                "volume_delta_mm3": metrics.get("volume_delta_a_minus_b_mm3"),
                "volume_delta_percent": metrics.get("volume_delta_a_minus_b_percent"),
            })

    for qc_json_path in qc_files:
        with open(qc_json_path, "r") as qc_json_file:
            subject_qc = json.load(qc_json_file)

        subject_id = qc_json_path.parts[-3]
        subject_records.append((subject_id, subject_qc))

        manifest_json = build_stage2_output_manifest(subject_id, output_path, manual_dir)
        if manifest_json is not None:
            log.info(f"Stage 2 manifest saved: {manifest_json}")

        dashboard_png = build_subject_qc_dashboard(subject_id, output_path, manual_dir)
        if dashboard_png is not None:
            log.info(f"Subject dashboard saved: {dashboard_png}")

        add_component_rows(subject_id, subject_qc)
        add_pairwise_rows(subject_id, subject_qc)

    if pairwise_rows:
        csv_out = group_qc_output_path("Group_QC_Pairwise_Summary.csv")
        write_csv(csv_out, GROUP_QC_PAIRWISE_FIELDS, pairwise_rows)
        log.info(f"Pairwise QC summary saved: {csv_out}")

    if component_rows:
        comp_csv = group_qc_output_path("Group_QC_Component_Summary.csv")
        write_csv(comp_csv, GROUP_QC_COMPONENT_FIELDS, component_rows)
        log.info(f"Component QC summary saved: {comp_csv}")

    has_manual = metric_has_values("dice_cnr_vs_manual")

    if has_manual:
        paired_dice = paired_metric_rows("dice_cnr_vs_manual", "dice_atlas_vs_manual")
        paired_hd95 = paired_metric_rows("hd95_cnr_vs_manual_mm", "hd95_atlas_vs_manual_mm")

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
            dice_deltas = np.array([nm - atlas for _, nm, atlas in paired_dice], dtype=float)
            comp['dice_nm_minus_atlas_median'] = float(np.median(dice_deltas))
            comp['dice_nm_better_count'] = int(np.sum(dice_deltas > 0))
            comp['dice_atlas_better_count'] = int(np.sum(dice_deltas < 0))

        if paired_hd95:
            hd95_deltas = np.array([atlas - nm for _, nm, atlas in paired_hd95], dtype=float)
            comp['hd95_atlas_minus_nm_median_mm'] = float(np.median(hd95_deltas))
            comp['hd95_nm_better_count'] = int(np.sum(hd95_deltas > 0))
            comp['hd95_atlas_better_count'] = int(np.sum(hd95_deltas < 0))

        dice_map = {sid: (nm, atlas) for sid, nm, atlas in paired_dice}
        hd95_map = {sid: (nm, atlas) for sid, nm, atlas in paired_hd95}
        shared = sorted(set(dice_map).intersection(hd95_map))
        comp['subjects_nm_better_both_dice_and_hd95'] = int(
            sum((dice_map[s][0] > dice_map[s][1]) and (hd95_map[s][0] < hd95_map[s][1]) for s in shared)
        )

        comp_json = group_qc_output_path('Group_QC_Method_Comparison.json')
        with open(comp_json, 'w') as fh:
            json.dump(comp, fh, indent=2)
        log.info(f"Method comparison summary saved: {comp_json}")

        if paired_dice or paired_hd95:
            fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(10.5, 4.5))

            if paired_dice:
                atlas_d = np.array([atlas for _, _, atlas in paired_dice], dtype=float)
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
                atlas_h = np.array([atlas for _, _, atlas in paired_hd95], dtype=float)
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
            cmp_png = group_qc_output_path('Group_QC_Method_Comparison.png')
            fig_cmp.savefig(str(cmp_png), dpi=200, bbox_inches='tight')
            plt.close(fig_cmp)
            log.info(f"Method comparison plot saved: {cmp_png}")

    def raincloud_strip(ax, values, pos, color, label):
        vals = [finite_float(v) for v in values]
        vals = np.array([v for v in vals if v is not None], dtype=float)
        if len(vals) == 0:
            return

        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(vals))
        ax.scatter(pos + jitter - 0.15, vals, s=18, color=color, alpha=0.7, zorder=3, label=label)

        if len(vals) > 1 and not np.allclose(vals, vals[0]):
            kde = gaussian_kde(vals, bw_method=0.4)
            y_kde = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 200)
            x_kde = (kde(y_kde) / kde(y_kde).max()) * 0.35
            ax.fill_betweenx(y_kde, pos, pos + x_kde, alpha=0.5, color=color)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos - 0.18, pos - 0.12], [med, med], color="k", lw=2, zorder=4)
        ax.plot([pos - 0.18, pos - 0.18], [q1, q3], color="k", lw=1.5, zorder=4)

    dice_plot_specs = [
        spec for spec in GROUP_QC_COMPARISONS
        if spec["pair"] == "cnr_vs_atlas" or metric_has_values(spec["dice"])
    ]
    if has_manual:
        dice_plot_specs.extend([
            spec for spec in GROUP_QC_MANUAL_COMPARISONS
            if spec["pair"] in ["cnr_vs_manual", "atlas_vs_manual"] or metric_has_values(spec["dice"])
        ])

    fig1, ax1 = plt.subplots(figsize=(3 + (len(dice_plot_specs) * 1.5), 5))
    current_pos = 1
    x_labels = []

    for spec in dice_plot_specs:
        raincloud_strip(ax1, metric_values(spec["dice"]), current_pos, spec["color"], spec["label"])
        x_labels.append(spec["label"])
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

    out1 = group_qc_output_path("Group_QC_DICE_Raincloud.png")
    fig1.savefig(str(out1), dpi=200, bbox_inches="tight")
    plt.close(fig1)
    log.info(f"DICE raincloud saved: {out1}")

    valid_hd = []
    for index, (subject_id, subject_qc) in enumerate(subject_records):
        hd95_value = finite_float(subject_qc.get("hd95_cnr_vs_atlas_mm"))
        if hd95_value is not None:
            valid_hd.append((index, subject_id, hd95_value))

    if valid_hd:
        valid_hd.sort(key=lambda x: x[2])
        sorted_indices = [x[0] for x in valid_hd]
        sorted_subs = [x[1] for x in valid_hd]
        sorted_subject_qc = [subject_records[index][1] for index in sorted_indices]

        fig2, ax2 = plt.subplots(figsize=(max(8, len(sorted_subs) * 0.4), 4))
        x_pos = list(range(len(sorted_subs)))

        hd95_plot_specs = [
            spec for spec in GROUP_QC_COMPARISONS
            if spec["pair"] == "cnr_vs_atlas" or metric_has_values(spec["hd95"])
        ]
        if has_manual:
            hd95_plot_specs.extend([
                spec for spec in GROUP_QC_MANUAL_COMPARISONS
                if spec["pair"] in ["cnr_vs_manual", "atlas_vs_manual"] or metric_has_values(spec["hd95"])
            ])

        for spec in hd95_plot_specs:
            y_values = [
                finite_float(subject_qc.get(spec["hd95"]))
                for subject_qc in sorted_subject_qc
            ]
            y_values = [np.nan if value is None else value for value in y_values]
            if not np.isfinite(y_values).any():
                continue

            if spec["pair"] == "cnr_vs_atlas":
                point_colors = ["firebrick" if value > 5.0 else spec["color"] for value in y_values]
            else:
                point_colors = spec["color"]

            ax2.scatter(
                x_pos,
                y_values,
                c=point_colors,
                s=40,
                marker=spec["marker"],
                alpha=0.9,
                label=spec["label"],
                zorder=3,
            )

        for idx, x in enumerate(x_pos):
            subject_qc = sorted_subject_qc[idx]
            y_vals = [
                finite_float(subject_qc.get(spec["hd95"]))
                for spec in hd95_plot_specs
            ]
            y_vals = [value for value in y_vals if value is not None]
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

        out2 = group_qc_output_path("Group_QC_HD95_DotPlot.png")
        fig2.savefig(str(out2), dpi=200, bbox_inches="tight")
        plt.close(fig2)
        log.info(f"HD95 dot plot saved: {out2}")

    if pairwise_rows:
        all_pair_specs = (
            GROUP_QC_COMPARISONS
            + GROUP_QC_MANUAL_COMPARISONS
            + [GROUP_QC_CORE_RIM_NMS_COMPARISON]
        )
        pair_pretty = {spec["pair"]: spec["label"] for spec in all_pair_specs}

        vol_data, vol_labels = [], []
        cent_data, cent_labels = [], []

        for pair_key in GROUP_QC_PAIR_ORDER:
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
            out3 = group_qc_output_path('Group_QC_Deviation_Boxplots.png')
            fig3.savefig(str(out3), dpi=200, bbox_inches='tight')
            plt.close(fig3)
            log.info(f"Deviation boxplots saved: {out3}")


if run_group_qc:
    run_group_qc_stage(Path(output_base), manual_dir, subjects, test_mode=args.test)
else:
    log.info("Skipping Group QC Plots for this stage.")
