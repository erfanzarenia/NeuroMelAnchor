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
Stage 1: Forward Normalisation
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
 
Segment: Manual ROI Segmentation (ITK-SNAP)
    Interactive segmentation of cerebral peduncle (label 1) and substantia nigra (label 2) on the
    group-average NM template.  
    Masks are saved to resources/tpl-MNI152NLin2009cAsym/.
 
Stage 2: Inverse Normalisation and CNR Seed Extraction
    1. Register subject NM mean to group template (Rigid + Affine + SyN)
    2. Warp segmented SN and CP masks to native NM space via inverse transforms
    3. Compute CNR map and NM-defined seed mask
    4. Compute Core-rim NM, probabilistic atlas, NMS atlas, and anchored-atlas comparisons
    5. Compute per-subject QC metrics and dashboards
            (optional: assess against manual masks if --manual_dir is provided)

Group QC: Rebuild dashboards and summary plots from existing Stage 2 outputs
    This stage is intentionally cheap: it does not rerun registration or mask generation.
    The code lives in workflow/qc/group_qc.py; it runs after Stage 2 unless --skip_group_qc,
    and its group outputs go to <output_root>/group_qc/.
 
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
    python neuromelanchor_pipeline.py 
        --project_root /path/to/project 
        --design_dir   /path/to/bids_root 
        --output_root  /path/to/outputs 
        --work_dir     /path/to/work 
        --n_procs 10                        defaults to 10; set lower/higher for your system
        --stage all                         defaults to all; can run 1, segment, 2, or group_qc
        --test                              test run on the first subject
        --smooth 0                          defaults to 0; set a smoothing FWHM in mm if needed
        --manual_dir  /path/to/manual_mask  for per-subject manual SN masks in Stage 2 QC
        --search_dilation_mm 0              grow the propagated SN search region by this many mm

Limitations
-------------------------------
Currently the floor is not effectively introduced.
The threshold is a simple within-subject percentile.
The top 45% of SN voxels are labelled as SN; the remaining 55% are labelled as noise and removed.
This means the weaker subjects likely don't look as good and need to be reflected upon.
The floor is supposed to help stop this, but it is certainly too low to ever work.

A contrast ratio is actually used here instead of a contrast-to-noise ratio.
================================================================================
By NEXUS
================================================================================
"""
__version__ = "3.0"

# =================================================================
# Import Statements
# =================================================================
import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import SimpleITK as sitk

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
from scipy.ndimage import distance_transform_edt, binary_erosion

from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.spm import Realign, Smooth
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.ants import Registration, BrainExtraction, ApplyTransforms, N4BiasFieldCorrection

# =================================================================
# Container Runtime Detection
# =================================================================

def get_container_cmd(project_root):
    """Return the command prefix that runs Python in the FireANTs container, or None if no runtime is found."""
    sif = Path(project_root) / "fireants.sif"
    if sif.exists() and shutil.which("singularity"):
        return ["singularity", "exec", "--nv", f"--bind={project_root}", str(sif), "python"]
    if sif.exists() and shutil.which("apptainer"):
        return ["apptainer", "exec", "--nv", f"--bind={project_root}", str(sif), "python"]
    if shutil.which("docker"):
        return ["docker", "run", "--gpus", "all", "--rm",
                "-v", f"{project_root}:{project_root}",
                "--entrypoint", "python", "fireants"]
    return None

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
parser.add_argument("--nm_image_strategy", choices=["mean", "first"], default="mean", help="Strategy to create the subject NM image. mean = realign and average all acquistion; first = only use first acquisition (default: mean)")
parser.add_argument("--nm_image_n", default=None, type=int, help="Optional: in mean mode, average only the first N ComboEchoNM acquistions")
parser.add_argument('--hires', action='store_true', help='Also output NM_MNI_res-05.nii at 0.5mm by re-applying saved transforms to a finer grid.')
parser.add_argument('--skip_template_builder', action='store_true', help='Skip FireANTs iterative NM template build; use the naive group mean for ITK-SNAP instead')
parser.add_argument('--drawing_template', choices=['v4', 'naive'], default='v4', help="Template the SN/CP masks are drawn on in --stage segment: v4 = FireANTs V4 midbrain template (default), naive = plain group mean")
parser.add_argument('--rebuild_template', action='store_true', help="Rebuild and re-align the V4 drawing template")
parser.add_argument('--search_dilation_mm', default=0.0, type=float, help="Grow the propagated SN search region by this many mm before NM refinement (default: 0 = current behaviour)")
parser.add_argument("--dataset", choices=["biogen", "snit"], default="biogen",help="Input file naming: biogen = sub-X_run-1_echo-N_part-mag_MEGRE + sub-X_acq-T1w; "
                         "snit = sub-X_acq-NM_rec-DIS2D_run-*_echo-N_part-mag_GRE + sub-X_acq-MP2RAGEpostproc_run-01_T1w (default: biogen)")
parser.add_argument('--seed_cnr_floor', default=0.05, type=float, help="Absolute minimum CNR for a voxel to enter the NM seed (default: 0.05)")
parser.add_argument('--seed_cnr_percentile', default=55, type=float, help="Subject-adaptive percentile of search-region CNR; the seed keeps voxels above it. Set low (0-5) to let --seed_cnr_floor define the mask absolutely (default: 55)")
parser.add_argument('--seed_min_cluster_voxels', default=15, type=int, help="Drop connected clusters smaller than this many voxels (default: 15)")
parser.add_argument('--seed_components_to_keep', default=2, type=int, help="Keep only the N largest connected components; SN is bilateral so 2 is correct (default: 2)")
parser.add_argument("--nm_input", choices=["combo", "echo"], default="combo", help="NM source: 'combo' = vendor-combined multi-echo GRE (default, higher SNR)'echo' = single MEGRE echo, selected by --nm_echo")
parser.add_argument("--nm_echo", default=3, type=int, help="Echo number to use when --nm_input echo (default: 2)")
parser.add_argument('--seed_cnr_fwhm', default=1.0, type=float, help="FWHM (mm) used to smooth the CNR map before thresholding the seed (default: 1.0; 1.5 recommended)")
parser.add_argument('--seed_slice_fill', action='store_true', help="Fill holes within each slice of the final seed (holes open in 3D survive the 3D fill)")
parser.add_argument('--skip_group_qc', action='store_true', help="Do not build the QC reports after Stage 2 (they can be rebuilt later with --stage group_qc or workflow/qc/group_qc.py)")
parser.add_argument('--skip_t1_outputs', action='store_true', help="Stage 2: do not write T1w-space copies of the seed. By default the seed is moved to T1w with the stage-1 rigid NM->T1 transform (linear, >= 0.5, largest pieces kept)")
parser.add_argument('--seed_image', choices=['n4', 'raw'], default='n4', help="Stage 2: image the seed is built from. n4 = stage-1 bias-corrected image (default); raw = the same NM run without bias correction")
parser.add_argument('--trace_route', choices=['template', 't1'], default='template', help="Stage 2: how the SN/CP traces reach NM space. template = stage-2 NM->template registration (default); t1 = stage-1 transforms (NM->T1, then T1->MNI)")
parser.add_argument('--t1_interp', choices=['linear', 'nearest'], default='linear', help="Stage 2: how the seed is moved from NM to T1w space. linear = linear, then keep >= 0.5 (default); nearest = nearest-neighbour")
parser.add_argument('--t1_seed_res', type=float, default=None,
                    help="T1 seed voxel size in mm; omitted = original T1 grid")


args = parser.parse_args()
if args.skip_template_builder:
    args.drawing_template = 'naive'
if args.t1_seed_res is not None:
    if not np.isfinite(args.t1_seed_res) or args.t1_seed_res <= 0:
        parser.error("--t1_seed_res must be positive and finite")

# =================================================================
# NM Mask Method Settings
# =================================================================

# This is minimum allowed cnr threshold for NM-defined seed (Accept voxels must be at least 5% brighter than background)
seed_cnr_floor = args.seed_cnr_floor
# Subject-adpative threshold inside the SNsearch mask. Keep's voxels above a specified percentile
seed_cnr_percentile = args.seed_cnr_percentile
# Removes tiny, disconnected islands (Connected clusters must be 15 voxels or more. Somewhat arbitrary but rectified by keeping only 2 components)
seed_min_cluster_voxels = args.seed_min_cluster_voxels
# Keeps only the largest 2 connected components
seed_components_to_keep = args.seed_components_to_keep

seed_cnr_fwhm = args.seed_cnr_fwhm
seed_slice_fill = args.seed_slice_fill

# Removes approximalty 1 mm from the outside boundary of the mask (assumes uncertain boundaries)
core_rim_erosion_mm = 0.5
# CNR threshold for the core part of the NM mask (more liberal)
core_rim_core_cnr_threshold = 0.05
# CNR threshold for the rim part of the NM mask (more strict)
core_rim_rim_cnr_threshold = 0.10
# Minimum cluster size for the rim
core_rim_min_cluster_voxels = 25
# Keeps only the largest 2 connected components
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
log.info(f"skip_group_qc       : {args.skip_group_qc}")
log.info(f"nm_image_stategy    : {args.nm_image_strategy}")
log.info(f"nm_image_n          : {args.nm_image_n}")
log.info(f"nm_input            : {args.nm_input}")
log.info(f"nm_echo             : {args.nm_echo if args.nm_input == 'echo' else 'n/a'}")
log.info(f"hires               : {args.hires}")
log.info(f"drawing_template    : {args.drawing_template}")
log.info(f"rebuild_template    : {args.rebuild_template}")
log.info(f"search_dilation_mm  : {args.search_dilation_mm}")
log.info(f"script              : {Path(__file__).resolve()}")
log.info(f"command             : {' '.join(sys.argv)}")
log.info(f"dataset             : {args.dataset}")
log.info(f"seed_cnr_fwhm       : {seed_cnr_fwhm}")
log.info(f"seed_slice_fill     : {seed_slice_fill}")
log.info(f"seed_image          : {args.seed_image}")
log.info(f"trace_route         : {args.trace_route}")
log.info(f"t1_interp           : {args.t1_interp}")
log.info(f"t1_seed_res         : {args.t1_seed_res}")
log.info("=" * 60)

# =================================================================
# Validation Helper Function
# =================================================================

def require_existing_file(path, label):
    """Stop early if a required input file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found at: {path}")

# =================================================================
# V4 Drawing Template Helpers
# =================================================================
# These functions let --stage segment build the V4 template, line it up with
# the group average, and keep track of which template a tracing was drawn on.




def find_subjects_missing_hires(output_base):
    """
    List subjects that are missing the 0.5 mm NM image V4 is built from.
    Stage 1 always writes a 1 mm image (NM_MNI.nii). With --hires it also writes
    a 0.5 mm image (NM_MNI_res-05.nii). V4 should be built from 0.5 mm images only.
    """
    missing_subjects = []
    for nm_mni_path in sorted(Path(output_base).glob('sub-*/anat/NM_MNI.nii')):
        hires_path = nm_mni_path.parent / 'NM_MNI_res-05.nii'
        if not hires_path.exists():
            subject_id = nm_mni_path.parent.parent.name
            missing_subjects.append(subject_id)
    return missing_subjects


def build_v4_template(project_root, output_base, v4_template):
    """
    Build the V4 midbrain template with template_builder.py in the FireANTs container.

    Settings match the echo-1 build: 12 iterations, rigid + SyN, shape averaging
    with a 0.25 step, no Laplacian sharpening. Takes about 35 minutes on the GPU.
    """
    if not Path(output_base).resolve().is_relative_to(Path(project_root).resolve()):
        raise ValueError("--output_root must sit inside --project_root so the container can see it.")

    container_cmd = get_container_cmd(project_root)
    if container_cmd is None:
        log.warning("No container runtime found - running the V4 builder with the local Python")
        container_cmd = [sys.executable]

    builder_cmd = container_cmd + [
        str(project_root / "workflow" / "scripts" / "template_builder.py"),
        "--output-root", str(output_base),
        "--work-dir", str(Path(v4_template).parent / "build"),
        "--output-template", str(v4_template),
        "--mni-prior", str(mask_dir / "HarvardOxford" / "Midbrain.nii.gz"),
        "--iterations", "12",
        "--rigid",
        "--shape-avg",
        "--shape-avg-step", "0.25",
        "--laplacian", "0",
        "--save-iterations",
    ]

    log.info(f"Building the V4 drawing template: {' '.join(builder_cmd)}")
    subprocess.run(builder_cmd, check=True)
    require_existing_file(v4_template, "V4 drawing template")

def align_v4_to_group_average(v4_template, group_average, aligned_v4):
    """
    Shift V4 about 1.3 mm so it sits exactly where the group average does.

    Stage 2 registers every subject to the group average, so a mask drawn on V4 only
    lands correctly if the two templates agree on where the midbrain is. The shift is
    rigid (no stretching) and the result keeps V4's own 0.5 mm grid for drawing.
    """

    # ANTs runs from inside this folder, so all paths must be absolute.
    v4_template = Path(v4_template).resolve()
    group_average = Path(group_average).resolve()
    aligned_v4 = Path(aligned_v4).resolve()
    drawing_dir = aligned_v4.parent

    # V4 is a midbrain crop surrounded by zeros. Without a mask those zeros pull the
    # alignment off (tested: 0.7 mm instead of 1.3 mm), so mark the part V4 covers.
    v4_img = nib.load(str(v4_template))
    v4_mask = drawing_dir / "V4_Mask.nii.gz"
    nib.save(nib.Nifti1Image((v4_img.get_fdata() > 0).astype(np.uint8), v4_img.affine), str(v4_mask))

    # V4 is the fixed image because its small crop keeps this to about 15 seconds.
    # No centre-of-mass start: the two templates are already within a couple of millimetres.
    group_to_v4 = Registration()
    group_to_v4.inputs.fixed_image = str(v4_template)
    group_to_v4.inputs.moving_image = str(group_average)
    group_to_v4.inputs.fixed_image_masks = [str(v4_mask)]
    group_to_v4.inputs.transforms = ['Rigid']
    group_to_v4.inputs.transform_parameters = [(0.1,)]
    group_to_v4.inputs.metric = ['CC']
    group_to_v4.inputs.metric_weight = [1]
    group_to_v4.inputs.radius_or_number_of_bins = [2]
    group_to_v4.inputs.sampling_strategy = [None]
    group_to_v4.inputs.sampling_percentage = [None]
    group_to_v4.inputs.number_of_iterations = [[60, 30]]
    group_to_v4.inputs.shrink_factors = [[4, 2]]
    group_to_v4.inputs.smoothing_sigmas = [[2, 1]]
    group_to_v4.inputs.sigma_units = ['vox']
    group_to_v4.inputs.convergence_threshold = [1e-7]
    group_to_v4.inputs.convergence_window_size = [10]
    group_to_v4.inputs.output_transform_prefix = str(drawing_dir / "GroupAverage_to_V4_")
    group_to_v4.inputs.environ = safe_env
    rigid_transform = group_to_v4.run(cwd=str(drawing_dir)).outputs.forward_transforms[0]

    # The transform points from V4 to the group average, so it is applied inverted.
    align_v4 = ApplyTransforms()
    align_v4.inputs.dimension = 3
    align_v4.inputs.input_image = str(v4_template)
    align_v4.inputs.reference_image = str(v4_template)
    align_v4.inputs.transforms = [rigid_transform]
    align_v4.inputs.invert_transform_flags = [True]
    align_v4.inputs.interpolation = 'LanczosWindowedSinc'
    align_v4.inputs.output_image = str(aligned_v4)
    align_v4.inputs.environ = safe_env
    align_v4.run(cwd=str(drawing_dir))

    # Report how far V4 moved, so the log shows it was a small shift and not a failure.
    v4_sitk = sitk.ReadImage(str(v4_template))
    v4_centre = v4_sitk.TransformContinuousIndexToPhysicalPoint([size / 2.0 for size in v4_sitk.GetSize()])
    moved_centre = sitk.ReadTransform(str(rigid_transform)).TransformPoint(v4_centre)
    log.info(f"V4 aligned to the group average: shifted {np.linalg.norm(np.subtract(moved_centre, v4_centre)):.2f} mm")


def create_blank_v4_mask(v4_template_path, mask_path):
    """
    Create an empty mask (all zeros) on exactly the same grid as the V4 template.

    ITK-SNAP draws into this file, so your labels are saved at V4's full 0.5 mm resolution.
    """
    v4_img = nib.load(str(v4_template_path))
    mask_header = v4_img.header.copy()
    mask_header.set_data_dtype(np.uint8)
    blank_mask = np.zeros(v4_img.shape[:3], dtype=np.uint8)
    nib.save(nib.Nifti1Image(blank_mask, v4_img.affine, mask_header), str(mask_path))


def images_share_grid(first_image_path, second_image_path):
    """Return True if two images have the same size and sit in the same place in space."""
    first_img = nib.load(str(first_image_path))
    second_img = nib.load(str(second_image_path))
    same_size = first_img.shape[:3] == second_img.shape[:3]
    same_position = np.allclose(first_img.affine, second_img.affine, atol=1e-4)
    return same_size and same_position


def write_itksnap_label_file(label_file_path):
    """Write the label names and colours ITK-SNAP shows while you draw (1 = CP in cyan, 2 = SN in yellow)."""
    label_lines = [
        "################################################",
        "# ITK-SnAP Label Description File",
        "# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL",
        "################################################",
        '    0     0    0    0        0  0  0    "Clear Label"',
        '    1   255    0    0        1  1  1    "Cerebral Penducle (Background)"',
        '    2     0  255    0        1  1  1    "Substantia Nigra (Target)"',
    ]
    with open(label_file_path, 'w') as f:
        f.write("\n".join(label_lines) + "\n")

# =================================================================
# Environment and Thread Control
# =================================================================

n_threads = str(min(args.n_procs, 8))
safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': n_threads, 'OMP_NUM_THREADS': n_threads}

needs_itksnap = args.stage in ['segment', 'all']
needs_ants = args.stage in ['1', '2', 'segment','all']
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
# NM file naming differs between cohorts. Stage 1 reads these files; stage 2 reads them again for --seed_image raw.
if args.dataset == "snit":
    echo_pattern = f'{{subject_id}}/anat/{{subject_id}}_acq-NM_rec-DIS2D_run-*_echo-{args.nm_echo}_part-mag_GRE.nii.gz'
else:
    echo_pattern = f'{{subject_id}}/anat/{{subject_id}}_run-1_echo-{args.nm_echo}_part-mag_MEGRE.nii.gz'

if args.nm_input == "combo":
    NM_Data = {'runs': '{subject_id}/anat/{subject_id}_acq-CombEchoNM*_GRE.nii.gz'}
else:
    NM_Data = {'runs': echo_pattern}

if args.seed_image == 'raw' and args.nm_image_strategy != 'first':
    raise ValueError("--seed_image raw needs --nm_image_strategy first (the seed must come from the run stage 1 used).")

mask_dir = project_root / "resources" / "tpl-MNI152NLin2009cAsym"
use_v4_drawing = args.drawing_template == 'v4'
trace_dir = output_base / ("manual_trace_v4" if use_v4_drawing else "manual_trace")
trace_dir.mkdir(parents=True, exist_ok=True)
drawing_dir = output_base / "drawing_template"
v4_template = drawing_dir / "SN_Template_V4.nii.gz"
v4_aligned_template = drawing_dir / "SN_Template_V4_aligned.nii.gz"

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
hires_ref_path = mask_dir / 'MNI152NLin2009cAsym_res-05_desc-brain_T1w.nii.gz'
if args.hires and not hires_ref_path.exists():
    raise FileNotFoundError(f"0.5mm MNI reference not found at {hires_ref_path}. Generate it first.")


log.info(f"{'='*60}")
log.info(f"Whole-head T1 template: {t1_template}")
log.info(f"Brain-extracted template: {t1_brain_template}")
log.info(f"Brain probability mask: {brain_prob_template}")
log.info(f"{'='*60}")
combined_mask = str(trace_dir / "MNI_Manual_Masks_combined.nii.gz")
SN_mask       = str(trace_dir / "MNI_SNc_Manual.nii.gz")
CP_mask       = str(trace_dir / "MNI_CP_Manual.nii.gz")
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
    """Create a QC image overlaying realigned NM runs to assess run alignment."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_anat

    out_png = os.path.abspath('Run_Alignment_Overlay_QC.png')

    if isinstance(realigned_files, str):
        files = [realigned_files]
    else:
        files = sorted(list(realigned_files))

    if len(files) == 0:
        raise ValueError("No realigned NM files were provided for run alignment QC.")

    if len(files) < 2:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, 'Run alignment QC skipped: only 1 run detected.', ha='center', va='center',
            fontsize=11, color='gray', transform=ax.transAxes,)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return out_png

    display = plot_anat(files[0], title=f'Realigned NM run overlay ({len(files)} runs)',
        display_mode='ortho', draw_cross=False)

    edge_colors = ['red', 'lime', 'cyan', 'magenta', 'yellow', 'orange', 'blue']

    for run_index, run_file in enumerate(files[1:], start=2):
        color = edge_colors[(run_index - 2) % len(edge_colors)]
        display.add_edges(run_file, color=color)

    display.savefig(out_png, dpi=150)
    display.close()

    return out_png

def combine_transforms(t1_to_mni_composite, nm_to_t1_composite):
    """Return transforms in the order expected by ANTs ApplyTransforms."""
    return [t1_to_mni_composite, nm_to_t1_composite]

def combine_prealign_transforms(nm_to_t1_inverse, t1_to_mni_inverse):
    return [nm_to_t1_inverse, t1_to_mni_inverse]

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
    display = plotting.plot_anat(t1_template_image, display_mode='ortho', dim=-0.5, cmap='gray', 
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
    display = plotting.plot_anat(t1_mni_image, display_mode='ortho', cut_coords=cut_coords, dim=-0.5, cmap='gray', title='Atlas on T1-norm QC',
        draw_cross=False, colorbar=False,)
    display.add_overlay(atlas_mask_mni, cmap='Reds', alpha=0.8)
    display.savefig(out_png, dpi=300)
    display.close()
    return out_png

def extract_tractography_seed(nm_image, native_sn_mask, native_cp_mask, cnr_floor=0.05, cnr_percentile=60, min_cluster_vox=15,
    keep_components=2, cnr_fwhm=1.0, slice_fill=False):
    """
    Create the subject-specific NM seed mask.

    The mask is built from NM contrast relative to the cerebral peduncle (CP).
    The final threshold is subject-adaptive: it keeps voxels above the fixed CNR
    floor and above a chosen percentile of this subject's SN-search CNR values.
    """
    import os
    import json
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

        component_sizes = ndi.sum(filled_mask, component_labels,index=np.arange(1, n_components + 1),)

        components_to_keep = [int(i + 1) for i, size in enumerate(component_sizes)
            if size >= int(min_size)
        ]
        if components_to_keep and int(max_components) > 0:
            components_to_keep = sorted(components_to_keep,key=lambda component_id: component_sizes[component_id - 1],
            reverse=True,)[: int(max_components)]

        if not components_to_keep:
            return np.zeros_like(filled_mask, dtype=bool)
        return np.isin(component_labels, components_to_keep)

    print("Computing CNR and extracting NM tractography seed...")

    nm_img = nib.load(nm_image)
    nm_data = nm_img.get_fdata()
    sn_mask = nib.load(native_sn_mask).get_fdata().astype(bool)
    cp_mask = nib.load(native_cp_mask).get_fdata().astype(bool)
    if nm_data.shape[:3] != sn_mask.shape or not np.allclose(nm_img.affine, nib.load(native_sn_mask).affine, atol=1e-4):
        raise ValueError("The seed image and the warped SN trace are not on the same voxel grid.")
    
    cp_mask_eroded = ndi.binary_erosion(cp_mask, iterations=1)
    if cp_mask_eroded.sum() < 20:
        print("CP erosion left too few voxels; using uneroded mask.")
        cp_mask_eroded = cp_mask

    if not sn_mask.any() or not cp_mask.any():
        raise ValueError(f"Empty mask detected. SN voxels: {sn_mask.sum()} | CP voxels: {cp_mask.sum()}")

    cp_values = nm_data[cp_mask_eroded]
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
        nm_data[sn_mask & finite_nm_voxels] - cp_reference_intensity) / cp_reference_intensity
    cnr_img = nib.Nifti1Image(cnr_data, nm_img.affine, nm_img.header)

    smoothed_cnr_img = smooth_img(cnr_img, fwhm=float(cnr_fwhm))
    smoothed_cnr_data = smoothed_cnr_img.get_fdata()

    sn_cnr_values = smoothed_cnr_data[sn_mask]
    sn_cnr_values = sn_cnr_values[np.isfinite(sn_cnr_values)]
    if sn_cnr_values.size == 0:
        raise ValueError("No finite CNR values found inside the native SN search mask.")
    adaptive_cnr_bound = float(np.percentile(sn_cnr_values, float(cnr_percentile)))
    lower_cnr_bound = max(float(cnr_floor), adaptive_cnr_bound)
    # The upper bound drops the brightest 1% of search-region voxels.
    upper_cnr_bound = float(np.percentile(sn_cnr_values, 99))

    candidate_seed_mask = (
        sn_mask
        & np.isfinite(smoothed_cnr_data)
        & (smoothed_cnr_data >= lower_cnr_bound)
        & (smoothed_cnr_data <= upper_cnr_bound)
    )
    final_seed_mask = keep_largest_components(candidate_seed_mask, min_size=min_cluster_vox, max_components=keep_components,
    )

    # Close small gaps in the mask without letting it grow outside the search region.
    final_seed_mask = (ndi.binary_closing(final_seed_mask, structure=np.ones((3, 3, 3), dtype=bool))
                       & sn_mask).astype(np.uint8)
    if slice_fill:
        for z in range(final_seed_mask.shape[2]):
            final_seed_mask[:, :, z] = ndi.binary_fill_holes(final_seed_mask[:, :, z])
        final_seed_mask = (final_seed_mask.astype(bool) & sn_mask).astype(np.uint8)

    cnr_out = os.path.abspath('Subject_CNR_Map_Smoothed.nii.gz')
    seed_out = os.path.abspath('Native_Tractography_Seed_SNc.nii.gz')
    nib.save(smoothed_cnr_img, cnr_out)
    # Save the seed with the search region's header (written by ANTs), not the seed image's own header.
    # A raw file can carry two different position records (BioGen sub-059 does), which would shift the seed in ANTs.
    grid_img = nib.load(native_sn_mask)
    seed_header = grid_img.header.copy()
    seed_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(final_seed_mask, grid_img.affine, seed_header), seed_out)


    all_values = np.concatenate([cp_values, sn_values])
    histogram_bins = np.linspace(np.percentile(all_values, 1), np.percentile(all_values, 99), 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cp_values, bins=histogram_bins, alpha=0.55, color='b', label='CP (background)', density=True)
    ax.hist(sn_values, bins=histogram_bins, alpha=0.55, color='y', label='SN (signal)', density=True)
    ax.axvline(cp_reference_intensity, color='m', lw=1.2, ls='--', label=f'CP mode = {cp_reference_intensity:.1f}',)
    ax.set_title(
        f'SN vs CP intensities | CNR threshold = {lower_cnr_bound:.3f} | '
        f'mean SN CNR = {np.mean(smoothed_cnr_data[sn_mask]):.3f} | '
        f'seed voxels = {int(final_seed_mask.sum())}',
        fontsize=9,)
    ax.set_xlabel('Voxel intensity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()

    hist_file = os.path.abspath('CNR_Histogram.png')
    fig.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    seed_diagnostics = {
        'seed_voxels': int(final_seed_mask.sum()),
        'search_roi_voxels': int(sn_mask.sum()),
        'mean_search_region_cr': float(np.mean(sn_cnr_values)),
    }
    seed_diag_file = os.path.abspath('Seed_Diagnostics.json')
    with open(seed_diag_file, 'w') as f:
        json.dump(seed_diagnostics, f, indent=2)

    return cnr_out, seed_out, hist_file, seed_diag_file


def make_bias_corrected_reference(in_files):
    """Create the subject NM reference image from bias-corrected NM file(s)."""
    import os
    from nilearn.image import mean_img

    if isinstance(in_files, str):
        files = [in_files]
    else:
        files = list(in_files)

    files = sorted(files)

    if len(files) == 0:
        raise ValueError("No bias-corrected NM files were provided.")

    reference_image_path = os.path.abspath("mean_NM_bias_corrected.nii.gz")
    mean_img(files).to_filename(reference_image_path)
    return reference_image_path

def threshold_t1_seed(seed_fraction, keep_components=2):
    """Binarise the linearly warped T1w seed at 0.5 and keep the largest pieces (resampling can shed specks)."""
    import os
    import numpy as np
    import nibabel as nib
    from scipy import ndimage as ndi
    img = nib.load(seed_fraction)
    mask = np.asarray(img.dataobj) >= 0.5
    labels, n = ndi.label(mask)
    if n > keep_components:
        sizes = np.bincount(labels.ravel())[1:]
        mask = np.isin(labels, np.argsort(sizes)[::-1][:keep_components] + 1)
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    out = os.path.abspath('T1w_Tractography_Seed_SNc.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, header), out)
    return out

def make_t1_seed_reference(t1_file, resolution):
    """Build a grid with the T1 orientation and outer corner."""
    import os
    import numpy as np
    import nibabel as nib

    img = nib.load(t1_file)
    spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
    direction = img.affine[:3, :3] / spacing

    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("T1 affine contains shear; check grid construction.")

    extent = np.array(img.shape[:3]) * spacing
    shape = np.maximum(1, np.rint(extent / resolution)).astype(int)
    corner = nib.affines.apply_affine(img.affine, [-0.5, -0.5, -0.5])

    affine = np.eye(4)
    affine[:3, :3] = direction * resolution
    affine[:3, 3] = corner + direction @ np.full(3, resolution / 2)

    out = os.path.abspath("T1_seed_reference.nii.gz")
    reference = nib.Nifti1Image(np.zeros(tuple(shape), dtype=np.uint8), affine)
    nib.save(reference, out)
    return out

def first_nm_file(in_files):
    """Return the file stage 1 uses with --nm_image_strategy first (the first one in sorted order)."""
    files = [in_files] if isinstance(in_files, str) else sorted(in_files)
    if len(files) == 0:
        raise ValueError("No NM files were found.")
    return files[0]


def select_nm_image_files(in_files, strategy="mean", n=None):
    """Choose which CombEchoNM files enter the NM image path."""
    if isinstance(in_files, str):
        files = [in_files]
    else:
        files = list(in_files)

    files = sorted(files)

    if len(files) == 0:
        raise ValueError("No CombEchoNM files were found.")

    if strategy == "first":
        return files[:1]

    if strategy == "mean":
        if n is not None:
            if int(n) < 1:
                raise ValueError("--nm_image_n must be >= 1.")
            return files[:int(n)]
        return files

    raise ValueError(f"Unknown NM image strategy: {strategy}")


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


def create_core_rim_nm_mask(cnr_map, native_sn_mask, core_erosion_mm=1.0, core_cnr_threshold=0.05,
    rim_cnr_threshold=0.10, min_cluster_vox=25, keep_components=2,
):
    """
    Create a stricter NM mask by treating the SN core and rim differently.

    The core is allowed to use a lower CNR threshold because it is spatially
    more trustworthy. The rim uses a higher CNR threshold because boundary
    voxels are where false positives most often inflate HD95. One possible concern is the arbitrary 
    choice of 1 mm for the core erosion
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
    cnr_map=None,
):
    """Score the NM, core-rim and atlas masks against each other and any manual mask, and write SN_Mask_QC.json with QC flags."""
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

    def require_same_grid(reference_img, moving_img, reference_label, moving_label):
        # Manual masks are drawn on the raw NM voxel grid, so only the grid must match.
        # The world affine can differ when the source NM header was edited after the
        # mask was drawn (e.g. sub-059 echo-1 carries a later sform_code=2 sform).
        if reference_img.shape != moving_img.shape:
            raise ValueError(
                f"Shape mismatch: {reference_label} {reference_img.shape} vs "
                f"{moving_label} {moving_img.shape}."
            )
        if not np.allclose(reference_img.header.get_zooms()[:3], moving_img.header.get_zooms()[:3], atol=1e-3):
            raise ValueError(f"Voxel size mismatch: {reference_label} and {moving_label}.")
        return float(np.abs(reference_img.affine - moving_img.affine).max())

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
        'manual_affine_max_abs_diff': None,
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
        'core_median_cr': None,
        'boundary_median_cr': None,
        'boundary_over_core_median_cr_ratio': None,
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
            qc['flags'].append('Anchoring degraded DICE vs CNR seed; check ROI mask and CNR quality')

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
        manual_affine_diff = require_same_grid(cnr_img, man_img, 'CNR', 'manual')
        if manual_affine_diff > 1e-3:
            print(
                f"WARNING: manual mask affine differs from CNR (max abs diff {manual_affine_diff:.3f}); "
                "comparing on the shared voxel grid."
            )
        qc['manual_affine_max_abs_diff'] = manual_affine_diff
        if cnr_map is not None and str(cnr_map).strip():
            cnr_map_img = nib.load(cnr_map)
            cnr_map_data = cnr_map_img.get_fdata()
            require_same_space(cnr_img, cnr_map_img, 'CNR mask', 'CNR map')

            core_erosion_mm = 1.0
            boundary_band_mm = 1.0
            dist_inside_mm = distance_transform_edt(man_data, sampling=spacing)
            dist_outside_mm = distance_transform_edt(~man_data, sampling=spacing)
            core_band = man_data & (dist_inside_mm > core_erosion_mm)
            boundary_band = (
                (man_data & (dist_inside_mm <= boundary_band_mm))
                | (~man_data & (dist_outside_mm <= boundary_band_mm))
            )
            finite_cr = np.isfinite(cnr_map_data)
            core_values = cnr_map_data[core_band & finite_cr]
            boundary_values = cnr_map_data[boundary_band & finite_cr]

            if core_values.size > 0 and boundary_values.size > 0:
                qc['core_median_cr'] = float(np.median(core_values))
                qc['boundary_median_cr'] = float(np.median(boundary_values))
                qc['boundary_over_core_median_cr_ratio'] = (
                    qc['boundary_median_cr'] / qc['core_median_cr']
                    if qc['core_median_cr'] != 0 else None
                )

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
            qc['flags'].append('Low DICE CNR vs manual; automated mask diverges from expert')
        if qc['dice_atlas_vs_manual'] is not None and qc['dice_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE atlas vs manual; registration or atlas fit may be poor')
        if qc['dice_prob_atlas_vs_manual'] is not None and qc['dice_prob_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE probabilistic atlas vs manual; threshold or atlas fit may be poor')
        if qc['dice_nms_atlas_vs_manual'] is not None and qc['dice_nms_atlas_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE NMS atlas vs manual; threshold or atlas fit may be poor')
        if qc['dice_core_rim_nm_vs_manual'] is not None and qc['dice_core_rim_nm_vs_manual'] < 0.5:
            qc['flags'].append('Low DICE core-rim NM vs manual; boundary thresholds may need adjustment')
        if (
            qc['dice_core_rim_nm_vs_manual'] is not None
            and qc['dice_cnr_vs_manual'] is not None
            and qc['dice_core_rim_nm_vs_manual'] < qc['dice_cnr_vs_manual'] - 0.05
        ):
            qc['flags'].append('Core-rim NM reduced DICE vs original NM; check core/rim thresholds')
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
            qc['flags'].append('Anchoring reduced DICE vs manual; CNR seed may be poorly defined')

    if qc['n_voxels_cnr'] < 20:
        qc['flags'].append('CNR seed very small (<20 voxels); check CNR map and SN registration')
    cnr_components = qc['mask_components']['cnr']
    if cnr_components['n_components'] > 2:
        qc['flags'].append('CNR seed has >2 connected components; HD95 may be inflated by islands')
    if (
        cnr_components['largest_component_fraction'] is not None
        and np.isfinite(cnr_components['largest_component_fraction'])
        and cnr_components['largest_component_fraction'] < 0.8
    ):
        qc['flags'].append('CNR seed largest component <80% of mask; possible fragmentation')
    if qc['dice_cnr_vs_atlas'] is not None and qc['dice_cnr_vs_atlas'] < 0.4:
        qc['flags'].append('Low DICE CNR vs atlas (<0.4); possible registration failure')
    if qc['hd95_cnr_vs_atlas_mm'] is not None and qc['hd95_cnr_vs_atlas_mm'] > 5.0:
        qc['flags'].append('High HD95 CNR vs atlas (>5 mm)')
    if qc['centroid_distance_cnr_vs_atlas_mm'] is not None and qc['centroid_distance_cnr_vs_atlas_mm'] > 3.0:
        qc['flags'].append('Large centroid offset CNR vs atlas (>3 mm)')

    qc_file = os.path.abspath('SN_Mask_QC.json')
    with open(qc_file, 'w') as f:
        json.dump(qc, f, indent=2)

    return qc_file

def dilate_search_region(native_sn_mask, dilation_mm=0.0):
    """
    Grow the propagated SN search region by a few millimetres.

    The NM refinement can only keep voxels inside this region, so a tight drawing
    caps how much of the SN it can ever find. Growing it slightly lets the subject's
    own NM contrast decide the boundary instead of the drawn outline.
    """
    import os
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt


    search_img = nib.load(native_sn_mask)
    search_mask = search_img.get_fdata() > 0

    if float(dilation_mm) > 0:
        voxel_size_mm = tuple(float(z) for z in search_img.header.get_zooms()[:3])
        distance_from_mask_mm = distance_transform_edt(~search_mask, sampling=voxel_size_mm)
        search_mask = distance_from_mask_mm <= float(dilation_mm)

    search_header = search_img.header.copy()
    search_header.set_data_dtype(np.uint8)
    search_file = os.path.abspath('Native_SN_Search_Region.nii.gz')
    nib.save(nib.Nifti1Image(search_mask.astype(np.uint8), search_img.affine, search_header), search_file)
    return search_file


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


    if args.dataset == "snit":
        T1_Data = {'T1': '{subject_id}/anat/{subject_id}_acq-MP2RAGEpostproc_run-01_T1w.nii.gz'}
    else:
        T1_Data = {'T1': '{subject_id}/anat/{subject_id}_acq-T1w.nii.gz'}


    SelectNM = Node(SelectFiles(NM_Data, base_directory=str(design_dir)), name='SelectNM')

    select_nm_image = Node(Function(input_names=["in_files", "strategy", "n"],
                                        output_names=["selected_files"], function=select_nm_image_files),
                                        name="Select_NM_image_Files")

    select_nm_image.inputs.strategy = args.nm_image_strategy
    select_nm_image.inputs.n = args.nm_image_n


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

    bias_mean_node = Node(Function(input_names=['in_files'], output_names=['mean_image'], function=make_bias_corrected_reference), name='NM_BiasCorrected_Mean')

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
    # High-Resolution Solution
    # =================================================================

    if args.hires:
        nm_to_mni_hires = Node(ApplyTransforms(), name='NM_to_MNI_HiRes')
        nm_to_mni_hires.inputs.dimension = 3
        nm_to_mni_hires.inputs.interpolation = 'LanczosWindowedSinc'
        nm_to_mni_hires.inputs.reference_image = str(hires_ref_path)
        nm_to_mni_hires.inputs.output_image = 'NM_MNI_res-05.nii'
        nm_to_mni_hires.inputs.environ = safe_env

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
    # QC_RunComparison/     - Oerlay of selected realigned NM runs
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
    (SelectNM, select_nm_image, [('runs', 'in_files')]),
    (select_nm_image, gunzip, [('selected_files', 'in_file')]),

    # T1 Path: Brain Extraction and T1 to MNI Registration
    (SelectT1, brainextraction, [('T1', 'anatomical_image')]),
    (brainextraction, t1_to_mni, [('BrainExtractionBrain', 'moving_image')]),

    # NM-T1 Path: Fixed to T1, using Brain Mask, Moving image is bias corrected NM
    (SelectT1, nm_to_t1, [('T1', 'fixed_image')]),
    (brainextraction, nm_to_t1, [('BrainExtractionMask', 'fixed_image_masks')]),
    (bias_mean_node, nm_to_t1, [('mean_image', 'moving_image')]),

    (bias, bias_mean_node, [('output_image', 'in_files')]),

    (t1_to_mni, combine_forward, [('composite_transform', 't1_to_mni_composite')]),
    (nm_to_t1, combine_forward, [('composite_transform', 'nm_to_t1_composite')]),
    (combine_forward, nm_to_mni, [('combined_transforms', 'transforms')]),
    (bias_mean_node, nm_to_mni, [('mean_image', 'input_image')]),

    # QC and Smooth
    (nm_to_mni, qc_node_nm, [('output_image', 'nm_mni_image')]),
    (t1_to_mni, qc_node_t1, [('warped_image', 't1_mni_image')]),
    (t1_to_mni, qc_node_atlas, [('warped_image', 't1_mni_image')]),
    
    # QC
    (qc_node_nm, datasink1, [('out_png', 'qc.@QC_NM_Plot')]),
    (qc_node_t1, datasink1, [('out_png', 'qc.@QC_T1_Plot')]),
    (qc_node_atlas, datasink1, [('out_png', 'qc.@QC_atlas_Plot')]),

    # Anat
    (bias, datasink1, [('output_image', 'anat.@NM_BiasCorr')]),
    (bias_mean_node, datasink1, [('mean_image', 'anat.@NM_BiasCorr_Mean')]),
    (nm_to_mni, datasink1, [('output_image', 'anat.@NM_to_MNI')]),

    # Transforms
    (t1_to_mni, datasink1, [('composite_transform', 'transforms.T1_to_MNI_Composite'),
                            ('inverse_composite_transform', 'transforms.T1_to_MNI_InvComposite')]),
    (nm_to_t1, datasink1, [('composite_transform', 'transforms.NM_to_T1_Composite'),
                           ('inverse_composite_transform', 'transforms.NM_to_T1_InvComposite')]),]   
    if args.hires:
        connections += [
            (combine_forward, nm_to_mni_hires, [('combined_transforms', 'transforms')]),
            (bias_mean_node,  nm_to_mni_hires, [('mean_image', 'input_image')]),
            (nm_to_mni_hires, datasink1,       [('output_image', 'anat.@NM_to_MNI_HiRes')]),
        ]
    
    if do_smooth:
        connections += [
            (nm_to_mni, smooth, [('output_image', 'in_files')]),
            (smooth, datasink1, [('smoothed_files', 'anat.@nm_mni_smooth')]),]
        
    if args.nm_image_strategy == "first":
        log.info("NM image strategy: first only; skipping SPM realign, motion QC, and run-comparison QC.")
        connections += [(gunzip, bias, [('out_file', 'input_image')]),]
    else:
        log.info("NM image strategy: mean; running SPM realign, motion QC, and run-comparison QC.")
        connections += [
            (gunzip, realign, [('out_file', 'in_files')]),
            (realign, bias, [('realigned_files', 'input_image')]),
            (realign, motion_node, [('realignment_parameters', 'realignment_parameters')]),
            (realign, run_compare, [('realigned_files', 'realigned_files')]),

            (realign, datasink1, [('realignment_parameters', 'metrics.@NM_MotPar')]),
            (motion_node, datasink1, [('stats_file', 'metrics.@NM_MotionStats')]),
            (run_compare, datasink1, [('out_png', 'qc.@QC_RunComparison')]),
            (realign, datasink1, [('mean_image', 'anat.@NM_Realigned_Mean')]),
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
# ITK-SNAP Interactive Segmentation on the V4 Template
# =================================================================
if run_segment and use_v4_drawing:
    require_existing_file(group_average_path, "Group average NM template (run --stage 1 first)")
    require_existing_file(hires_ref_path, "0.5 mm MNI T1 (shown behind V4 in ITK-SNAP)")

    missing_hires = find_subjects_missing_hires(output_base)
    if missing_hires:
        raise FileNotFoundError("V4 needs a 0.5 mm NM_MNI_res-05.nii image for every subject. "
                                f"Missing for: {missing_hires}. Re-run --stage 1 with --hires.")

    drawing_dir.mkdir(parents=True, exist_ok=True)

    # A rebuilt template would not match an existing tracing, so ask for it to be moved first.
    if args.rebuild_template and os.path.exists(combined_mask):
        raise ValueError(f"Move your existing tracing out of {trace_dir} before rebuilding the template.")

    # Build V4 once (~35 min on the GPU), then shift it onto the group average (~15 s).
    if args.rebuild_template or not v4_template.exists():
        build_v4_template(project_root, output_base, v4_template)
    else:
        log.info(f"Using the existing V4 template: {v4_template}")

    if args.rebuild_template or not v4_aligned_template.exists():
        align_v4_to_group_average(v4_template, group_average_path, v4_aligned_template)
    else:
        log.info("V4 is already aligned to the group average.")

    # Draw into a mask on the V4 grid: empty the first time, your saved tracing after that.
    if not os.path.exists(combined_mask):
        log.info(f"Creating an empty V4 mask at {combined_mask}")
        create_blank_v4_mask(v4_aligned_template, combined_mask)
    elif not images_share_grid(combined_mask, v4_aligned_template):
        raise ValueError(f"{combined_mask} is not on the V4 drawing grid. Move it aside, or use --rebuild_template.")
    else:
        log.info(f"Existing V4 tracing found at {combined_mask}. Preserving Data")

    label_file = trace_dir / "itksnap_labels.txt"
    write_itksnap_label_file(label_file)

    log.info(f"{'='*60}")
    log.info("Manaul Segmentation Required (V4 template)...")
    log.info("1. Cerebral Penducle (CP) - Background reference Region  -> label 1")
    log.info("2. Substantia Nigra (SN) - NM signal region              -> label 2")
    log.info("V4 opens cropped to the midbrain, so no navigation is needed.")
    log.info(f"{'-'*60}")
    log.info("Save your segmentation (Ctrl + S), then close the ITK-SNAP window to resume the pipeline.")
    log.info(f"{'='*60}")
    subprocess.run(['itksnap',
                    '-g', str(v4_aligned_template),
                    '-o', str(hires_ref_path),
                    '-s', combined_mask,
                    '-l', str(label_file)])

# =================================================================
# ITK-SNAP Interactive Segmentation
# =================================================================
if run_segment and not use_v4_drawing:
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

    # Written under the output root: it is a generated file and embeds run-specific paths.
    workspace_path = output_base / "Automated_Setup.itksnap"
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

if run_segment:
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
    if use_v4_drawing and not images_share_grid(SN_mask, v4_aligned_template):
        raise ValueError(f"{SN_mask} is not on the V4 drawing grid. Re-run --stage segment.")
    log.info(f"Stage 2 drawing masks: SN={SN_mask} | CP={CP_mask}")


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

    # Which transforms carry template-space masks (SN/CP traces, QC atlases) to NM space
    if args.trace_route == 't1':
        GrabT1Route = Node(SelectFiles({
            'nm_to_t1_inv': '{subject_id}/transforms/NM_to_T1_InvComposite/transformInverseComposite.h5',
            't1_to_mni_inv': '{subject_id}/transforms/T1_to_MNI_InvComposite/transformInverseComposite.h5',
        }, base_directory=str(output_base)), name='GrabT1Route')
        t1_route = Node(Function(input_names=['nm_to_t1_inverse', 't1_to_mni_inverse'], output_names=['transforms'],
                                 function=combine_prealign_transforms), name='T1_Route_Transforms')
        to_native, to_native_field = t1_route, 'transforms'
    else:
        to_native, to_native_field = nm_to_template, 'inverse_composite_transform'

    # Which image the seed is built from
    if args.seed_image == 'raw':
        GrabRawNM = Node(SelectFiles(NM_Data, base_directory=str(design_dir)), name='GrabRawNM')
        first_raw_nm = Node(Function(input_names=['in_files'], output_names=['nm_file'], function=first_nm_file), name='First_Raw_NM')

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
    cnr_seed = Node(Function(input_names=['nm_image', 'native_sn_mask', 'native_cp_mask', 'cnr_floor', 'cnr_percentile', 'min_cluster_vox', 'keep_components', 'cnr_fwhm', 'slice_fill'], 
                            output_names=['cnr_out', 'seed_out', 'histogram', 'seed_diagnostics'], function=extract_tractography_seed), name='CNR_Seed')
    cnr_seed.inputs.cnr_floor = seed_cnr_floor
    cnr_seed.inputs.cnr_percentile = seed_cnr_percentile
    cnr_seed.inputs.min_cluster_vox = seed_min_cluster_voxels
    cnr_seed.inputs.keep_components = seed_components_to_keep
    cnr_seed.inputs.cnr_fwhm = seed_cnr_fwhm
    cnr_seed.inputs.slice_fill = seed_slice_fill

    # =================================================================
    # Search Region (propagated SN mask, optionally grown)
    # =================================================================
    search_region = Node(Function(input_names=['native_sn_mask', 'dilation_mm'],
                                  output_names=['search_file'],
                                  function=dilate_search_region), name='Search_Region')
    search_region.inputs.dilation_mm = args.search_dilation_mm

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
        'cnr_map',
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
    # Seed to T1w (stage-1 rigid NM -> T1)
    # =================================================================
    if not args.skip_t1_outputs:
        t1_file_pattern = ('{subject_id}/anat/{subject_id}_acq-MP2RAGEpostproc_run-01_T1w.nii.gz' if args.dataset == 'snit'
                           else '{subject_id}/anat/{subject_id}_acq-T1w.nii.gz')
        GrabT1 = Node(SelectFiles({'t1': t1_file_pattern}, base_directory=str(design_dir)), name='GrabT1')

        GrabNMtoT1 = Node(SelectFiles({'nm_to_t1': '{subject_id}/transforms/NM_to_T1_Composite/transformComposite.h5'},
                                      base_directory=str(output_base)), name='GrabNMtoT1')
        if args.t1_seed_res is not None:
            t1_seed_reference = Node(
                Function(input_names=['t1_file', 'resolution'],
                        output_names=['reference_image'],
                        function=make_t1_seed_reference),
                name='T1_Seed_Reference',
            )
            t1_seed_reference.inputs.resolution = args.t1_seed_res

        seed_to_t1 = Node(ApplyTransforms(), name='Seed_to_T1')
        seed_to_t1.inputs.dimension = 3
        seed_to_t1.inputs.interpolation = 'NearestNeighbor' if args.t1_interp == 'nearest' else 'Linear'
        seed_to_t1.inputs.output_image = ('T1w_Tractography_Seed_SNc_probabilistic.nii.gz' if args.t1_interp == 'linear'
                                          else 'T1w_Tractography_Seed_SNc_nearest-neighbor.nii.gz')

        seed_to_t1.inputs.environ = safe_env

        seed_t1_binary = Node(Function(input_names=['seed_fraction', 'keep_components'], output_names=['seed_t1'],
                                       function=threshold_t1_seed), name='Seed_T1_Binary')
        seed_t1_binary.inputs.keep_components = seed_components_to_keep

        nm_to_t1_qc = Node(ApplyTransforms(), name='NM_to_T1_QC')
        nm_to_t1_qc.inputs.dimension = 3
        nm_to_t1_qc.inputs.interpolation = 'LanczosWindowedSinc'
        nm_to_t1_qc.inputs.output_image = 'T1w_NM.nii.gz'
        nm_to_t1_qc.inputs.environ = safe_env

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
        (search_region,     datasink2, [('search_file', 'anat.@Search_Region')]),

        (GrabPhase1,        mni_to_nm_sn, [('bias_nm', 'reference_image')]),
        (to_native,         mni_to_nm_sn, [(to_native_field, 'transforms')]),
        
        (GrabPhase1,        mni_to_nm_cp, [('bias_nm', 'reference_image')]),
        (to_native,         mni_to_nm_cp, [(to_native_field, 'transforms')]),
        
        (GrabPhase1,        mni_to_nm_atlas, [('bias_nm', 'reference_image')]),
        (to_native,         mni_to_nm_atlas, [(to_native_field, 'transforms')]),
    
        (mni_to_nm_sn,      search_region, [('output_image', 'native_sn_mask')]),
        (search_region,     cnr_seed, [('search_file', 'native_sn_mask')]),

        (mni_to_nm_cp,      cnr_seed, [('output_image', 'native_cp_mask')]),

        (cnr_seed,          qc_node, [('seed_out','cnr_mask'), ('cnr_out', 'cnr_map')]), 
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
                                        ('histogram', 'qc.@CNR_Histogram'),
                                        ('seed_diagnostics', 'cnr.@Seed_Diagnostics')]),
        (qc_node,           datasink2, [('qc_file', 'cnr.@SN_Mask_Dice_QC')]),
    ]
    if args.trace_route == 't1':
        wf2_connections += [
            (inputnode2,    GrabT1Route, [('subject_id', 'subject_id')]),
            (GrabT1Route,   t1_route, [('nm_to_t1_inv', 'nm_to_t1_inverse'), ('t1_to_mni_inv', 't1_to_mni_inverse')]),
        ]
    else:
        wf2_connections += [(GrabPhase1, nm_to_template, [('bias_nm', 'moving_image')])]

    if args.seed_image == 'raw':
        wf2_connections += [
            (inputnode2,    GrabRawNM, [('subject_id', 'subject_id')]),
            (GrabRawNM,     first_raw_nm, [('runs', 'in_files')]),
            (first_raw_nm,  cnr_seed, [('nm_file', 'nm_image')]),
        ]
    else:
        wf2_connections += [(GrabPhase1, cnr_seed, [('bias_nm', 'nm_image')])]
    if prob_atlas_mni is not None:
        wf2_connections += [
            (GrabPhase1,          mni_to_nm_prob_atlas, [('bias_nm', 'reference_image')]),
            (to_native,           mni_to_nm_prob_atlas, [(to_native_field, 'transforms')]),
            (mni_to_nm_prob_atlas, prob_atlas_thresh, [('output_image', 'prob_atlas')]),
            (prob_atlas_thresh,   qc_node, [('mask_out', 'prob_atlas_mask')]),
            (prob_atlas_thresh,   datasink2, [('prob_out', 'anat.@ProbCIT_SNc_Probability'),
                                              ('mask_out', 'anat.@ProbCIT_SNc_Mask')]),
        ]

    if nms_atlas_mni is not None:
        wf2_connections += [
            (GrabPhase1,          mni_to_nm_nms_atlas, [('bias_nm', 'reference_image')]),
            (to_native,           mni_to_nm_nms_atlas, [(to_native_field, 'transforms')]),
            (mni_to_nm_nms_atlas, nms_atlas_thresh, [('output_image', 'prob_atlas')]),
            (nms_atlas_thresh,    qc_node, [('mask_out', 'nms_atlas_mask')]),
            (nms_atlas_thresh,    datasink2, [('prob_out', 'anat.@NMS_SNc_Probability'),
                                              ('mask_out', 'anat.@NMS_SNc_Mask')]),
        ]

    if not args.skip_core_rim_nm:
        wf2_connections += [
            (cnr_seed,     core_rim_nm, [('cnr_out', 'cnr_map')]),
            (search_region, core_rim_nm, [('search_file', 'native_sn_mask')]),
            (core_rim_nm,  qc_node, [('mask_out', 'core_rim_nm_mask')]),
            (core_rim_nm,  datasink2, [('mask_out', 'anat.@CoreRim_NM_Mask'),
                                       ('summary_file', 'qc.@CoreRim_NM_Summary')]),
        ]

    if manual_dir is not None:
        wf2_connections += [
            (inputnode2,    SelectManual, [('subject_id', 'subject_id')]),
            (SelectManual,  qc_node, [('manual_sn', 'manual_mask')]),

        ]

    if not args.skip_t1_outputs:
        wf2_connections += [
            (inputnode2,     GrabT1, [('subject_id', 'subject_id')]),
            (inputnode2,     GrabNMtoT1, [('subject_id', 'subject_id')]),
            (cnr_seed,       seed_to_t1, [('seed_out', 'input_image')]),
            (GrabNMtoT1,     seed_to_t1, [('nm_to_t1', 'transforms')]),
            (seed_to_t1,     seed_t1_binary, [('output_image', 'seed_fraction')]),
            (GrabPhase1,     nm_to_t1_qc, [('bias_nm', 'input_image')]),
            (GrabT1,         nm_to_t1_qc, [('t1', 'reference_image')]),
            (GrabNMtoT1,     nm_to_t1_qc, [('nm_to_t1', 'transforms')]),
            (seed_t1_binary, datasink2, [('seed_t1', 'anat_T1w.@Seed')]),
            (seed_to_t1,     datasink2, [('output_image', 'anat_T1w.@SeedFraction')]),
            (nm_to_t1_qc,    datasink2, [('output_image', 'anat_T1w.@NM')]),
        ]
        if args.t1_seed_res is None:
            wf2_connections += [
                (GrabT1, seed_to_t1, [('t1', 'reference_image')]),
            ]
        else:
            wf2_connections += [
                (GrabT1, t1_seed_reference, [('t1', 't1_file')]),
                (t1_seed_reference, seed_to_t1,
                [('reference_image', 'reference_image')]),
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
# Group QC
# =================================================================
if run_group_qc and not args.skip_group_qc:
    # Group QC lives in workflow/qc/group_qc.py so it can also be rerun on its own.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qc"))
    from group_qc import run_group_qc_stage
    run_group_qc_stage(Path(output_base), manual_dir, subjects, test_mode=args.test,
                       seed_cnr_percentile=seed_cnr_percentile,
                       prob_atlas_threshold=args.prob_atlas_threshold,
                       nms_atlas_threshold=args.nms_atlas_threshold)
else:
    log.info("Skipping Group QC Plots for this stage.")
