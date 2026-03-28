#!/usr/bin/env python3
"""
Nipype Pipeline for Preprocessing NM_MRI Data in Accordance with Salzman et al and Cassidy et al. 

- Current Steps
    - Load Echo-1 GRE and T1
    - Load MNI Template
    - Call SPM to Realign and Compute Average of Realigned Images
    - Call ANTs to perform N4BiasCorrection on the NM data
    - Call ANTs for Brain Extraction from T1
    - Call ANTs for RegistrationSyN of T1 to MNI Template, informed by a BrainStem mask
    - Call ANTs for Regristartion of NM to T1
    - Combine Transformations to move NM to MNI
    - Call SPM to smooth

- Additional Fixes
    - Add additional command line parameters
    - Need to make it so that if there is only a single NM scan or multiple NM scan, there can adapt either way, instead of the current hard coded nature
    - Copyright logo beside NEXUS

By NEXUS 

"""

# =================================================================
# Import Statements
# =================================================================
import os

import argparse
import subprocess
from pathlib import Path 

import nipype as ni
from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from nipype.interfaces.spm import Realign, Smooth, SPMCommand
from nipype.interfaces.ants import Registration, BrainExtraction, ApplyTransforms, N4BiasFieldCorrection
from nipype.algorithms.misc import Gunzip

from nilearn.image import smooth_img, mean_img
import nibabel as nib
import numpy as np

from templateflow.api import get

# =================================================================
# Inputs
# =================================================================
safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '8', 'OMP_NUM_THREADS': '8'}

parser = argparse.ArgumentParser(description="Run NM_QC at the Subject Level")
parser.add_argument("--project_root", required=True, help="Root Data Folder (Location of Project Folder)", type=Path)
parser.add_argument("--output_root", required=True, help="Output Folder", type=Path)
parser.add_argument("--design_dir", required=True, help="Data Folder (Location of Subject Data)", type=Path)
parser.add_argument("--work_dir", required=True, help="Location of Work Directory", type=Path)
parser.add_argument("--n_procs", default=10, type=int)
parser.add_argument("--stage", choices=['1','segment', '2', 'all'], default='all',
                     help="Pipeline Stage to run. Defaults to 'all")
parser.add_argument("--manual_dir", default=None, type=Path, help="Directory of per-subject manual SN Segmentations (optional for similarity QC)")
args = parser.parse_args()

project_root = args.project_root
design_dir = args.design_dir
work_dir = args.work_dir
output_base = args.output_root
output_base.mkdir(parents=True, exist_ok=True)
manual_dir = args.manual_dir

subjects = sorted([p.name for p in design_dir.glob("sub-*")])
print(f"Found {len(subjects)} subjects: {subjects}")

# =================================================================
# Execution Flags
# =================================================================

run_phase1 = args.stage in ['1', 'all']
run_segment = args.stage in ['segment', 'all']
run_phase2 = args.stage in ['2', 'all']

# =================================================================
# Templates and Masks
# =================================================================

t1_template = get('MNI152NLin2009cAsym', resolution=1, suffix='T1w', extension='nii.gz')[1]
print("Whole-Head T1 Candidates", t1_template)
t1_brain_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='nii.gz')
print("Brain T1 Candidates", t1_brain_template)
brain_prob_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='probseg', extension='nii.gz')
print("Prob Brain Candidates", brain_prob_template)

combined_mask = f"{project_root}/Masks/MNI/MNI_Manual_Masks_combined.nii.gz"
SN_mask = f"{project_root}/Masks/MNI/MNI_SNc_Manual.nii.gz"
CP_mask = f"{project_root}/Masks/MNI/MNI_CP_Manual.nii.gz"
BrainStem_mask = f"{project_root}/Masks/MNI/MNI_Brainstem_Weight_Mask.nii.gz"

group_average_path = str(output_base / "Study-Specific_NM_Template.nii.gz")

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
    matplotlib.use('Agg')
    from nilearn import plotting
    print(f"\n{'='*60}")
    print(f"Computing Registration NM to MNI QC Plot...")
    print(f"\n{'='*60}")

    out_png = os.path.abspath('Registration_NM_to_MNI_QC.png')
    display = plotting.plot_stat_map(t1_template_image, cut_coords=[-20, -10, 0, 10, 20],
                                            display_mode='z', cmap='gray',
                                            title="NM to MNI Alignment QC", draw_cross=False, colorbar=False)
    display.add_overlay(nm_mni_image, cmap='hot', alpha=0.6, colorbar=True)
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
    display = plotting.plot_stat_map(t1_template_image, cut_coords=[-20, -10, 0, 10, 20],
                                            display_mode='z', cmap='gray',
                                            title="T1-norm to MNI Alignment QC", draw_cross=False, colorbar=False)
    display.add_overlay(t1_mni_image, cmap='hot', alpha=0.6, colorbar=True)
    display.savefig(out_png, dpi=150)
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

def compute_dice(cnr_mask, atlas_mask, manual_mask=None):
    import nibabel as nib
    import numpy as np
    import os 
    import json

    def dice(a,b):
        a = a.astype(bool)
        b = b.astype(bool)
        total = a.sum() + b.sum()
        intersection = a & b

        if total == 0:
            return float('nan')
        
        return float(2*intersection.sum() / total)
    
    def hausdroff(a,b):
        a = a.astype(bool)
        b = b.astype(bool)
    
    cnr_data = nib.load(cnr_mask).get_fdata().astype(bool)
    atlas_data = nib.load(atlas_mask).get_fdata().astype(bool)

    flags = []

    qc = {'n_voxels_cnr_mask': int(cnr_data.sum()), 'n_voxels_atlas_mask': int(atlas_data.sum()), 'diff_in_voxels': int(cnr_data.sum()) - int(atlas_data.sum()), 'dice_cnr_vs_atlas': dice(cnr_data, atlas_data)}

    if manual_mask is not None:
        manual_data = nib.load(manual_mask).get_fdata().astype(bool)
        qc['n_voxels_manual'] = int(manual_data.sum())
        qc['dice_cnr_vs_manual'] = dice(cnr_data, manual_data)
        qc['dice_atlas_vs_manual'] = dice(atlas_data, manual_data)

    if qc['dice_cnr_vs_manual'] < 0.5:
        flags.append('Low DICE Between CNR and Manual - Automated Mask Diverges Significantly from Expert Segmentation')

    if qc['dice_atlas_vs_manual'] < 0.5:
        flags.append('Low DICE Between Atlas and Manual - Registration quality may be insufficient or Atlas poorly matches anatomy')

    if qc['n_voxels_cnr_mask'] < 20:
        flags.append('CNR Produced a Small Mask - Check CNR Map and SN Registration')

    if flags:
        qc['flags'] = flags

    qc_file = os.path.abspath('SN_Mask_DICE_QC.json')

    with open(qc_file, 'w') as f:
        json.dump(qc, f, indent=2)
    
    return qc_file

# =================================================================
# Phase 1: Forward Normalization
# =================================================================
if run_phase1:

    wf1 = Workflow(name="NM_Preprocessing_Phase-1")
    wf1.base_dir = str(work_dir)

    inputnode = Node(IdentityInterface(fields=['subject_id']), name="NM_Subject_IDs")
    inputnode.iterables = [('subject_id', subjects)]

    NM_Data = {'runs':'{subject_id}/anat/{subject_id}_acq-CombEchoNM_run-*_GRE.nii.gz'}
    SelectNM = Node(SelectFiles(NM_Data, base_directory=design_dir), name='SelectNM')

    T1_Data = {'T1': '{subject_id}/anat/{subject_id}_acq-MP2RAGEpostproc_run-01_T1w.nii.gz'}
    SelectT1 = Node(SelectFiles(T1_Data, base_directory=design_dir), name='SelectT1')

    # =================================================================
    # Utility Nodes
    # =================================================================

    gunzip = MapNode(Gunzip(), name='Gunzip_NM', iterfield=['in_file'])

    # =================================================================
    # BiasCorrection
    # =================================================================
    bias = MapNode(N4BiasFieldCorrection(dimension = 3), name='NM_BiasCorr', iterfield=['input_image'])
    bias.inputs.copy_header = True
    bias.inputs.output_image = 'NM_bias_corrected.nii.gz'

    # =================================================================
    # Realign (SPM)
    # =================================================================

    realign = Node(Realign(), name="NM_Realign")
    realign.inputs.register_to_mean = True
    realign.inputs.fwhm = 5
    realign.inputs.quality = 0.9
    realign.inputs.interp = 2

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
    t1_to_mni.inputs.use_histogram_matching = [True]*3

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
    smooth = Node(Smooth(), name='NM_Smooth')
    smooth.inputs.fwhm=[1,1,1]

    # =================================================================
    # Save Final Output
    # -----------------------------------------------------------------
    # Saved outputs and where they end up in the WORKDIR
    # Should be removed before publishing
    # NM_MotParStats/ - SPM realignment paramete rfiles (.txt, one per run)
    # NM_MotParPlot/ - per-run parameters PNG
    # QC_RunComparison - run 1 vs run 2 difference Image
    # NM_BiasCorr/ - Bias-corrected NM in native space (used by Phase 2)
    # NM_to_MNI/ - Unsoothed NM in MNI Space
    # NM_Smooth/ - Smoothed NM in MNI Space
    # QC_Plot/   - PNG Overlay
    # T1_to_MNI_Composite/  - Composite Forward Ward(T1->MNI, .h5)
    # T1_to_MNI_InvComposite/ - Composite Inverse Warp (MNI->T1, .h5, used in Phase 2)
    # NM_to_T1_Composite/   - Composite Forward Warp (NM->T1, .h5)
    # NM_to_T1_InvComposite  - Composite Inverse Warp (T1->NM, .h5, used in Phase 2)
    # =================================================================

    datasink1 = Node(DataSink(base_directory=str(output_base)), name='PPData')

    datasink1.inputs.substitutions = [('_subject_id_', ''),]


    # =================================================================
    # Workflow
    # =================================================================

    wf1.connect([
        (inputnode, SelectNM, [('subject_id', 'subject_id')]),
        (inputnode, SelectT1, [('subject_id', 'subject_id')]),

        # NM Path: Merges, Unzipped, Realigned, Compute FD, and Perform Bias Correction
        (SelectNM, gunzip, [('runs', 'in_file')]),
        (gunzip, bias, [('out_file', 'input_images')]), 
        (bias, realign, [('output_images', 'in_files')])
        (realign, motion_node, [('realignment_parameters', 'realignment_parameters')]),
        (realign, run_compare, [('realigned_files', 'realigned_files')]),

    
        # T1 Path: Brain Extraction and T1 to MNI Registration
        (SelectT1, brainextraction, [('T1', 'anatomical_image')]),
        (brainextraction, t1_to_mni, [('BrainExtractionBrain', 'moving_image')]),

        # NM-T1 Path: Fixed to T1, using Brain Mask, Moving image is bias corrected NM
        (SelectT1, nm_to_t1, [('T1', 'fixed_image')]),
        (brainextraction, nm_to_t1, [('BrainExtractionMask', 'fixed_image_masks')]),
        (bias, nm_to_t1, [('output_image', 'moving_image')]), 
        
        (t1_to_mni, combine_forward, [('composite_transform', 't1_to_mni_composite')]), 
        (nm_to_t1, combine_forward, [('composite_transform', 'nm_to_t1_composite')]), 
        (combine_forward, nm_to_mni, [('combined_transforms', 'transforms')]),
        (bias, nm_to_mni, [('output_image', 'input_image')]),

        # QC and Smooth
        (nm_to_mni, qc_node_nm, [('output_image', 'nm_mni_image')]),
        (t1_to_mni, qc_node_t1, [('output_image', 't1_mni_image')])
        (nm_to_mni, smooth, [('output_image', 'in_files')]),

        # Data Saving 
        (realign, datasink1, [('realignment_parameters', 'NM_MotPar')]),
        (motion_node, datasink1, [('stats_file', 'NM_MotionStats'),
                                ('plot_file', 'NM_MotionPlot')]),
        (run_compare, datasink1, [('out_png', 'QC_RunComparison')]),
        (nm_to_mni, datasink1, [('output_image', 'NM_to_MNI')]), 
        (smooth, datasink1, [('smoothed_files', 'NM_Smooth')]),
        (qc_node_nm, datasink1, [('out_png', 'QC_NM_Plot')]),
        (qc_node_t1, datasink1, [('out_png', 'QC_T1_Plot')])
        (bias, datasink1, [('output_image', 'NM_BiasCorr')]),
        (realign, datasink1, [('mean_image', 'NM_BiasCorr_Mean')])
        (t1_to_mni, datasink1, [('composite_transform', 'T1_to_MNI_Composite'),
                                ('inverse_composite_transform', 'T1_to_MNI_InvComposite')]),
        (nm_to_t1, datasink1, [('composite_transform', 'NM_to_T1_Composite'),
                            ('inverse_composite_transform', 'NM_to_T1_InvComposite')]),
    ])

    # =================================================================
    # Save DAG and Run Phase 1
    # =================================================================

    print(f"{'='*60}")
    print("Saving DAG of Job...")
    print(f"{'='*60}")

    wf1.write_graph(graph2use='colored', dotfilename=str(output_base / 'NM_PP_Phase-1_Pipeline_Dag'), simple_form=True)

    print(f"{'='*60}")
    print("Starting Phase 1: Forward Normalization...")
    print(f"{'='*60}")

    wf1.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})

    print(f"{'='*60}")
    print("Creating Group Average...")
    print(f"{'='*60}")

    mni_files = list(output_base.glob('NM_to_MNI/sub-*/NM_MNI.nii*'))
    if not mni_files:
        raise FileNotFoundError("No NM_MNI files found - Check DataSink or WorkDir")

    group_average_path = f"{output_base}/Study-Specific_NM_Template.nii.gz"
    mean_img(mni_files).to_filename(group_average_path)

# =================================================================
# ITK-SNAP Interactive Segmentation
# =================================================================
if run_segment:

    t1_img = nib.load(str(t1_brain_template))
    placeholder = np.zeros(t1_img.shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(placeholder, t1_img.affine, t1_img.header), combined_mask)

    template_file = project_root / "Scripts" / "NEXUS_NM_Template_ITKsnap.xml"
    with open(template_file, 'r') as f:
        xml_data = f.read()

    xml_data = xml_data.replace("PROJECT_ROOT", str(project_root))
    xml_data = xml_data.replace("TARGET_T1_PATH", str(t1_brain_template))
    xml_data = xml_data.replace("TARGET_NM_PATH", group_average_path)
    xml_data = xml_data.replace("TARGET_MASK_PATH", combined_mask)

    workspace_path = f"{output_base}/Automated_Setup.itksnap"
    with open(workspace_path, 'w') as f:
        f.write(xml_data)

    print(f"{'='*60}", flush=True)
    print("Manaul Segmentation Required...", flush=True)
    print("1. Cerebral Penducle (CP) - Background reference Region", flush=True)
    print("2. Substantia Nigra (SN) - NM signal region", flush=True)
    print("1. Jump to Midbrain : Type  98  109   61 in the 'Cursor Position' box (center-left) and hit Enter.", flush=True)
    print("2. Center Cameras   : Click 'zoom to fit' below the image", flush=True)
    print("3. Zoom In          : Below 'Cursor Inspector', click Magnifcation Symbol and zoom into MidBrain .", flush=True)
    print(f"{'-'*60}", flush=True)
    print("Save your segmentation (Ctrl + S), then close the ITK-SNAP window to resume the pipeline.", flush=True)
    print(f"{'='*60}", flush=True)
    subprocess.run(['itksnap', '-w', workspace_path], check=True)

    combo_img = nib.load(combined_mask)
    combo_data = combo_img.get_fdata()
    if np.max(combo_data) == 0:
        raise ValueError("Segmentation Mask is Empty - re-run and save the labels.")

    nib.save(nib.Nifti1Image((combo_data == 1).astype(float), combo_img.affine, combo_img.header), CP_mask)
    nib.save(nib.Nifti1Image((combo_data == 2).astype(float), combo_img.affine, combo_img.header), SN_mask)

    CP_n = int((combo_data == 1).sum())
    SN_n = int((combo_data == 2).sum())
    print(f"{'='*60}")
    print(f"CP Voxels: {CP_n}  |  SN_voxels: {SN_n}")
    if CP_n < 50 or SN_n < 50:
        print("Warning: one or both masks have very few voxels. \n You may want to inspect segmentation before continuing.")
    print(f"{'='*60}")

# =================================================================
# Phase 2: Inverse Normalization and Seeds
# =================================================================
if run_phase2:

    wf2 = Workflow(name="NM_Phase2")
    wf2.base_dir = str(work_dir)

    inputnode2 = Node(IdentityInterface(fields=['subject_id']), name="NM_Subject_IDs")
    inputnode2.iterables = [('subject_id', subjects)]

    # =================================================================
    # Grab Intermediate Files from Phase 1
    # =================================================================

    GrabPhase1 = Node(SelectFiles({
        'bias_nm': 'NM_BiasCorr_Mean/{subject_id}/mean_NM_bias_corrected.nii.gz',
        't1_inv_composite': 'T1_to_MNI_InvComposite/{subject_id}/*InverseComposite.h5',
        'nm_inv_composite': 'NM_to_T1_InvComposite/{subject_id}/*InverseComposite.h5'
    }, base_directory=str(output_base)), name='GrabPhase1')

    # =================================================================
    # Registration (NM → MNI) 
    # =================================================================

    nm_to_template = Node(Registration(), name='NM_to_Group_Template')
    nm_to_template.inputs.dimension = 3
    nm_to_template.inputs.interpolation = 'LanczosWindowedSinc'
    nm_to_template.inputs.transforms = ['Rigid', 'Affine']
    nm_to_template.inputs.transform_parameters = [(0.05,), (0.08)]
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

    # =================================================================
    # Compute CNR + Tractography Seed
    # =================================================================
    cnr_seed = Node(Function(input_names=['nm_image', 'native_sn_mask', 'native_cp_mask'], 
                            output_names=['cnr_out', 'seed_out', 'histogram'], function=extract_tractography_seed), name='CNR_Seed')
    
    # =================================================================
    # Register Seed to Atlas
    # =================================================================
    
    # =================================================================
    # QC
    # =================================================================

    qc_node = Node(Function(input_names=['cnr_mask', 'atlas_mask', 'manual_mask'], output_names=['qc_file'], function=compute_dice), name='Dice_QC')

    if manual_dir is not None:
        ManualData = {'manual_sn': '{subject_id}/SN_manual.nii.gz'}
        SelectManual = Node(SelectFiles(ManualData, base_directory=str(manual_dir)), name='SelectManual')
    else:
        qc_node.inputs.manual_mask = None

    # =================================================================
    # Datasink Phase 2
    # =================================================================
    datasink2 = Node(DataSink(base_directory=str(output_base)), name='Sink_Phase2')
    datasink2.inputs.substitutions = [('_subject_id_', '')]

    # =================================================================
    # Workflow Connect Phase 2
    # =================================================================

    wf2.connect([
        (inputnode2, GrabPhase1, [('subject_id', 'subject_id')]),

        (GrabPhase1, nm_to_template, [('bias_nm', 'moving_image')])

        (nm_to_template, mni_to_nm_sn, [('combined_inverse_transforms', 'transforms')]),
        (GrabPhase1, mni_to_nm_sn, [('bias_nm', 'reference_image')]),
        
        (nm_to_template, mni_to_nm_cp, [('combined_inverse_transforms', 'transforms')]),
        (GrabPhase1, mni_to_nm_cp, [('bias_nm', 'reference_image')]),

        (GrabPhase1, cnr_seed, [('bias_nm', 'nm_image')]),
        (mni_to_nm_sn, cnr_seed, [('output_image', 'native_sn_mask')]),
        (mni_to_nm_cp, cnr_seed, [('output_image', 'native_cp_mask')]),

        (cnr_seed, dice_node, [('seed_out','cnr_mask')]), 
        (mni_to_nm_sn, dice_node, [('output_image', 'atlas_mask'))]

        (mni_to_nm_sn, datasink2, [('output_image', 'Native_SN_Mask')]),
        (mni_to_nm_cp, datasink2, [('output_image', 'Native_CP_Mask')]),
        (cnr_seed, datasink2, [('cnr_out', 'CNR_Map'),
                            ('seed_out', 'Tractography_Seeds'),
                            ('histogram', 'CNR_Histogram')]),
        (dice_node, datasink2, [('qc_file', 'SN_Mask_Dice_QC')])
    ])

    if manual_mask is not None:
        wf2.connect([
            (inputnode2, SelectManual, [('subject_id', 'subject_id')]),
            (SelectManual, dice_node, [('manual_sn', 'manual_mask')])
        ])
    # =================================================================
    # Run Phase 2
    # =================================================================
    print(f"{'='*60}")
    print("Starting Phase 2: Inverse Transformation and Seed Extraction...")
    print(f"{'='*60}")

    print(f"{'='*60}")
    print("Saving DAG of Job...")
    print(f"{'='*60}")

    wf2.write_graph(graph2use='colored', dotfilename=str(output_base / 'NM_PP_Phase-2_Pipeline_Dag'), simple_form=True)

    wf2.run(plugin='MultiProc', plugin_args={'n_procs': args.n_procs})


