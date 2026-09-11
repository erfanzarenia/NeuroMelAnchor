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
    - Save intermediate steps for use in cnr pipeline

"""

# =================================================================
# Import Statements
# =================================================================
import os

import argparse
from pathlib import Path 

import nipype as ni
from nipype import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface, Merge, Function
from nipype.interfaces.spm import Realign, Smooth, SPMCommand
from nipype.interfaces.ants import Registration, BrainExtraction, ApplyTransforms, N4BiasFieldCorrection
from nipype.algorithms.misc import Gunzip

from templateflow.api import get

# =================================================================
# Inputs
# =================================================================
safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '8', 'OMP_NUM_THREADS': '8'}

parser = argparse.ArgumentParser(description="Run NM_QC at the Subject Level")
parser.add_argument("--project_root", required=True, help="Data Folder (Location of Project Folder)", type=Path)
parser.add_argument("--output_root", required=True, help="Output Folder", type=Path)
args = parser.parse_args()

project_root = args.project_root
design_dir = project_root / "NM_Adults"
work_dir = project_root / "NM_PP" / "workdir"
output_base = args.output_root
output_base.mkdir(parents=True, exist_ok=True)

subjects = sorted([p.name for p in design_dir.glob("sub-*")])
print(f"Found {len(subjects)} subjects: {subjects}")

# =================================================================
# Templates
# =================================================================

t1_template = get('MNI152NLin2009cAsym', resolution=1, suffix='T1w', extension='nii.gz')[1]
print("Whole-Head T1 Candidates", t1_template)
t1_brain_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='T1w', extension='nii.gz')
print("Brain T1 Candidates", t1_brain_template)
brain_prob_template = get('MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='probseg', extension='nii.gz')
print("Prob Brain Candidates", brain_prob_template)

# =================================================================
# Helper Functions
# =================================================================

def compute_framewise_displacement(realignment_parameters):
    import numpy as np
    import os

    parameters = np.loadtxt(realignment_parameters)
    if parameters.ndim ==1:
        parameters = parameters[np.newaxis, :]

    head_radius = 50
    parameters_mm = parameters.copy()
    parameters_mm[:,3:] = parameters_mm[:,3:] * head_radius
    diff = np.abs(np.diff(parameters_mm, axis=0))
    fd = diff.sum(axis=1)
    fd = np.insert(fd, 0, 0)

    fd_file = os.path.join(os.getcwd(), 'framewise_displacement.txt')
    np.savetxt(fd_file, fd, fmt='%.6f')

    # mean_fd = np.mean(fd)
    # mean_fd_file = os.path.abspath('mean_fd.txt')
    # with open(mean_fd_file, "w") as f:
    #     f.write(f"{mean_fd:.6f}")
    return fd_file

def combine_transforms(t1_to_mni_transforms, nm_to_t1_transforms):
    import os
    list_t1 = t1_to_mni_transforms if isinstance(t1_to_mni_transforms, str) else t1_to_mni_transforms
    list_nm = nm_to_t1_transforms if isinstance(nm_to_t1_transforms, str) else nm_to_t1_transforms

    combined = list_t1 + list_nm
    for t in combined:
        if not os.path.exists(t):
            raise FileNotFoundError(f"Transform file not found: {t}")
    print(f"Combined transform order ready for ANTS: {combined}\n")
    return combined

def generate_qc_png(nm_mni_image, t1_template_image):
    import os
    import matplotlib
    matplotlib.use('Agg')
    from nilearn import plotting

    out_png = os.path.abspath('Registration_QC.png')
    display = plotting.plot_stat_map(t1_template_image, cut_coords=[-16, -14, -12, -10, -8],
                                            display_mode='z', cmap='gray',
                                            title="NM to MNI Alignment QC")
    display.add_overlay(nm_mni_image, cmap='gray', alpha=0.6)
    display.savefig(out_png)
    display.close()
    return out_png


# =================================================================
# Construct Workflow
# =================================================================

wf = Workflow(name="NM_Preprocessing")
wf.base_dir = str(work_dir)

inputnode = Node(IdentityInterface(fields=['subject_id']), name="NM_Subject_IDs")
inputnode.iterables = [('subject_id', subjects)]

NM_Data = {'run1':'{subject_id}/anat/{subject_id}_acq-CombEchoNM_run-01_GRE.nii.gz', 
           'run2':'{subject_id}/anat/{subject_id}_acq-CombEchoNM_run-02_GRE.nii.gz'}
SelectNM = Node(SelectFiles(NM_Data, base_directory=design_dir), name='SelectNM')

T1_Data = {'T1': '{subject_id}/anat/{subject_id}_acq-MP2RAGEpostproc_run-01_T1w.nii.gz'}
SelectT1 = Node(SelectFiles(T1_Data, base_directory=design_dir), name='SelectT1')

# =================================================================
# Utility Nodes
# =================================================================

merge_runs = Node(Merge(2), name='Merge_Runs')

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
# Framewise Displacement
# =================================================================

fd_node = MapNode(Function(input_names=['realignment_parameters'], 
                        output_names=['fd_file'],
                        function=compute_framewise_displacement), name='ComputeFD', iterfield='realignment_parameters')

# =================================================================
# BiasCorrection
# =================================================================
bias = Node(N4BiasFieldCorrection(dimension = 3), name='NM_BiasCorr')
bias.inputs.copy_header = True

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
t1_to_mni.inputs.fixed_image_masks = ['NULL', 'NULL', '/scratch/Projects/BrainHack_Nexus/Masks/MNI/MNI_Brainstem_Weight_Mask.nii.gz']
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
t1_to_mni.inputs.write_composite_transform = False
t1_to_mni.inputs.collapse_output_transforms = False
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
nm_to_t1.inputs.write_composite_transform = False
nm_to_t1.inputs.collapse_output_transforms = False
nm_to_t1.inputs.environ = safe_env  

# =================================================================
# Combine Transformations
# =================================================================

combine_node = Node(Function(input_names=['t1_to_mni_transforms', 'nm_to_t1_transforms'], 
                             output_names=['combined_transforms'], 
                             function=combine_transforms), name='CombinedTransforms')

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

qc_node = Node(Function(input_names=['nm_mni_image', 't1_template_image'], output_names=['out_png'], function=generate_qc_png), name='Visual_QC')
qc_node.inputs.t1_template_image = str(t1_brain_template)

# =================================================================
# Smooth
# =================================================================
smooth = Node(Smooth(), name='NM_Smooth')
smooth.inputs.fwhm=[1,1,1]

# =================================================================
# Save Final Output
# =================================================================

datasink = Node(DataSink(base_directory=str(output_base)), name='PPData')

datasink.inputs.substitutions = [('_subject_id_', ''),
                                 ('NM_to_MNI', ''), 
                                 ('NM_Smooth', '')]


# =================================================================
# Workflow
# =================================================================

wf.connect([
    (inputnode, SelectNM, [('subject_id', 'subject_id')]),
    (inputnode, SelectT1, [('subject_id', 'subject_id')]),

    (SelectNM, merge_runs, [('run1', 'in1'), ('run2', 'in2')]),
    (merge_runs, gunzip, [('out', 'in_file')]),
    (gunzip, realign, [('out_file', 'in_files')]), 
    (realign, fd_node, [('realignment_parameters', 'realignment_parameters')]),

    (realign, bias, [('mean_image', 'input_image')]),

 
    (SelectT1, brainextraction, [('T1', 'anatomical_image')]),

    (brainextraction, t1_to_mni, [('BrainExtractionBrain', 'moving_image')]),

    (SelectT1, nm_to_t1, [('T1', 'fixed_image')]),
    (brainextraction, nm_to_t1, [('BrainExtractionMask', 'fixed_image_masks')]),
    (bias, nm_to_t1, [('output_image', 'moving_image')]), 
    

    (t1_to_mni, combine_node, [('forward_transforms', 't1_to_mni_transforms')]), 
    (nm_to_t1, combine_node, [('forward_transforms', 'nm_to_t1_transforms')]), 
    (combine_node, nm_to_mni, [('combined_transforms', 'transforms')]),
    (bias, nm_to_mni, [('output_image', 'input_image')]),

    (nm_to_mni, qc_node, [('output_image', 'nm_mni_image')]),
    (nm_to_mni, smooth, [('output_image', 'in_files')]),


    (realign, datasink, [('realignment_parameters', 'NM_MotPam')]),
    (fd_node, datasink, [('fd_file', 'NM_FD')]),
    (nm_to_mni, datasink, [('output_image', 'NM_to_MNI')]), 
    (smooth, datasink, [('smoothed_files', 'NM_Smooth')]),
    (qc_node, datasink, [('out_png', 'QC_Plot')])
])


# =================================================================
# Save DAG for Verification
# =================================================================

wf.write_graph(graph2use='colored', dotfilename='/scratch/Projects/BrainHack_Nexus/NM_PP/NM_PP_Pipeline_Dag', simple_form=True)

# =================================================================
# Run Script
# =================================================================

wf.run(plugin='MultiProc', plugin_args={'n_procs': 10})

