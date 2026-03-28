import os

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

from functions import extract_tractography_seed, compute_dice


def run_phase2(output_base, work_dir, subjects, group_average_path,
               SN_mask, CP_mask, manual_dir, n_procs):
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

    safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '8', 'OMP_NUM_THREADS': '8'}
    nm_to_template = Node(Registration(), name='NM_to_Group_Template')
    nm_to_template.inputs.fixed_image = group_average_path
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

        (GrabPhase1, nm_to_template, [('bias_nm', 'moving_image')]),

        (nm_to_template, mni_to_nm_sn, [('combined_inverse_transforms', 'transforms')]),
        (GrabPhase1, mni_to_nm_sn, [('bias_nm', 'reference_image')]),
        
        (nm_to_template, mni_to_nm_cp, [('combined_inverse_transforms', 'transforms')]),
        (GrabPhase1, mni_to_nm_cp, [('bias_nm', 'reference_image')]),

        (GrabPhase1, cnr_seed, [('bias_nm', 'nm_image')]),
        (mni_to_nm_sn, cnr_seed, [('output_image', 'native_sn_mask')]),
        (mni_to_nm_cp, cnr_seed, [('output_image', 'native_cp_mask')]),

        (cnr_seed, qc_node, [('seed_out','cnr_mask')]), 
        (mni_to_nm_sn, qc_node, [('output_image', 'atlas_mask')]),

        (mni_to_nm_sn, datasink2, [('output_image', 'Native_SN_Mask')]),
        (mni_to_nm_cp, datasink2, [('output_image', 'Native_CP_Mask')]),
        (cnr_seed, datasink2, [('cnr_out', 'CNR_Map'),
                            ('seed_out', 'Tractography_Seeds'),
                            ('histogram', 'CNR_Histogram')]),
        (qc_node, datasink2, [('qc_file', 'SN_Mask_Dice_QC')]),
    ])

    # if manual_mask is not None:
    #     wf2.connect([
    #         (inputnode2, SelectManual, [('subject_id', 'subject_id')]),
    #         (SelectManual, dice_node, [('manual_sn', 'manual_mask')])
    #     ])
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

    wf2.run(plugin='MultiProc', plugin_args={'n_procs': n_procs})
