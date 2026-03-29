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


def compute_motion_params(realignment_parameters):
    import numpy as np
    import os
    import json
    import matplotlib
    import matplotlib.pyplot as plt

    print(f"\n{'=' * 60}")
    print(f"Computing Motion Parameters and Plotting Translation and Rotation values...")
    print(f"\n{'=' * 60}")
    parameters = np.loadtxt(realignment_parameters)
    if parameters.ndim == 1:
        parameters = parameters[np.newaxis, :]

    trans_range = parameters[:, :3].max(axis=0) - parameters[:, :3].min(axis=0)
    rot_range = parameters[:, 3:].max(axis=0) - parameters[:, 3:].min(axis=0)

    stats = {
        'translation_range_mm': {
            'x': float(trans_range[0]),
            'y': float(trans_range[1]),
            'z': float(trans_range[2]),
            'max': float(trans_range.max()), },
        'rotation_range_deg': {
            'pitch': float(np.degrees(rot_range[0])),
            'roll': float(np.degrees(rot_range[1])),
            'yaw': float(np.degrees(rot_range[2])),
            'max': float(np.degrees(rot_range.max()))},
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

    print(f"\n{'=' * 60}")
    print(f"Checking Run Similarity...")
    print(f"\n{'=' * 60}")

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

    correlation = np.corrcoef(r1.flatten(), r2.flatten())[0, 1]

    diff_img = math_img("img1 - img2", img1=r1_img, img2=r2_img)

    out_png = os.path.abspath('Run_Difference_QC.png')
    display = plot_anat(diff_img, title=f'Run 1 - Run 2 (r = {correlation:.3f})', display_mode='ortho', cmap='RdBu_r',
                        draw_cross=False)
    display.savefig(out_png, dpi=150)
    display.close()

    return out_png


def combine_transforms(t1_to_mni_composite, nm_to_t1_composite):
    """
    ANTs ApplyTransforms applies transforms in REVERSE order
    """
    print(f"\n{'=' * 60}")
    print(f"Combing Forward Transforms...")
    print(f"\n{'=' * 60}")
    return [t1_to_mni_composite, nm_to_t1_composite]


def generate_qc_nm(nm_mni_image, t1_template_image):
    import os
    import matplotlib
    import numpy as np
    import nibabel as nib
    matplotlib.use('Agg')
    from nilearn import plotting
    print(f"\n{'=' * 60}")
    print(f"Computing Registration NM to MNI QC Plot...")
    print(f"\n{'=' * 60}")

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
    print(f"\n{'=' * 60}")
    print(f"Computing Registration between Normalized T1 and MNI QC Plot...")
    print(f"\n{'=' * 60}")

    out_png = os.path.abspath('Registration_norm-T1_to_MNI_QC.png')
    display = plotting.plot_anat(t1_template_image,
                                 display_mode='ortho', dim=-0.5, cmap='gray',
                                 title="T1-norm to MNI Alignment QC", draw_cross=False, colorbar=False)
    display.add_overlay(t1_mni_image, color='r')
    display.savefig(out_png, dpi=150)
    display.close()
    return out_png


def mean_of_bias_corrected(in_files):
    from nilearn.image import mean_img
    import os
    out = os.path.abspath('mean_NM_bias_corrected.nii.gz')
    mean_img(in_files).to_filename(out)
    return out


def run_phase1(design_dir, work_dir, output_base, subjects,
               t1_template, t1_brain_template, brain_prob_template,
               BrainStem_mask, group_average_path, n_procs):
    wf1 = Workflow(name="NM_Preprocessing_Phase-1")
    wf1.base_dir = str(work_dir)

    inputnode = Node(IdentityInterface(fields=['subject_id']), name="NM_Subject_IDs")
    inputnode.iterables = [('subject_id', subjects)]

    NM_Data = {'runs': '{subject_id}/anat/{subject_id}_acq-CombEchoNM_run-*_GRE.nii.gz'}
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
    bias = MapNode(N4BiasFieldCorrection(dimension=3), name='NM_BiasCorr', iterfield=['input_image'])
    bias.inputs.copy_header = True
    bias.inputs.output_image = 'NM_bias_corrected.nii.gz'

    # =================================================================
    # BiasCorrection Mean
    # =================================================================

    bias_mean_node = Node(
        Function(input_names=['in_files'], output_names=['mean_image'], function=mean_of_bias_corrected),
        name='NM_BiasCorrected_Mean')

    # =================================================================
    # Motion Parameters
    # =================================================================

    motion_node = MapNode(Function(input_names=['realignment_parameters'],
                                   output_names=['stats_file', 'plot_file'],
                                   function=compute_motion_params), name='Motion_Params',
                          iterfield=['realignment_parameters'])

    # =================================================================
    # Compare Runs
    # =================================================================
    run_compare = Node(Function(input_names=['realigned_files'],
                                output_names=['out_png'],
                                function=check_run_similarity), name='Run_Comparison_QC')

    # =================================================================
    # Brain Extraction (ANTS)
    # =================================================================

    safe_env = {'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '8', 'OMP_NUM_THREADS': '8'}
    brainextraction = Node(BrainExtraction(dimension=3), name='T1_BrainExtraction')
    brainextraction.inputs.brain_template = str(t1_template)
    brainextraction.inputs.brain_probability_mask = str(brain_prob_template)
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
    t1_to_mni.inputs.shrink_factors = [[8, 4, 2, 1]] * 3
    t1_to_mni.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
    t1_to_mni.inputs.sigma_units = ['vox'] * 3
    t1_to_mni.inputs.sampling_percentage = [0.25, 0.25, 1]
    t1_to_mni.inputs.sampling_strategy = ['Regular', 'Regular', 'None']
    t1_to_mni.inputs.convergence_threshold = [1e-6] * 3
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

    qc_node_nm = Node(
        Function(input_names=['nm_mni_image', 't1_template_image'], output_names=['out_png'], function=generate_qc_nm),
        name='Visual_QC_NM')
    qc_node_nm.inputs.t1_template_image = str(t1_brain_template)

    qc_node_t1 = Node(
        Function(input_names=['t1_mni_image', 't1_template_image'], output_names=['out_png'], function=generate_qc_t1),
        name='Visual_QC_T1')
    qc_node_t1.inputs.t1_template_image = str(t1_brain_template)

    # =================================================================
    # Smooth
    # =================================================================
    smooth = Node(Smooth(), name='NM_Smooth')
    smooth.inputs.fwhm = [1, 1, 1]
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
    # T1_to_MNI_Composite/  - Composite forward warp (T1→MNI, .h5)
    # T1_to_MNI_InvComposite/ - Composite inverse warp (MNI→T1, .h5, used in Phase 2)
    # NM_to_T1_Composite/   - Composite forward warp (NM mean→T1, .h5)
    # NM_to_T1_InvComposite/ - Composite inverse warp (T1→NM, .h5, used in Phase 2)
    # =================================================================

    datasink1 = Node(DataSink(base_directory=str(output_base)), name='PPData')

    datasink1.inputs.substitutions = [('_subject_id_', ''),
                                      ('_Motion_Params0/', ''),
                                      ('_NM_BiasCorr0/', ''),
                                      ('_NM_BiasCorr1/', '')]

    # =================================================================
    # Workflow
    # =================================================================

    wf1.connect([
        (inputnode, SelectNM, [('subject_id', 'subject_id')]),
        (inputnode, SelectT1, [('subject_id', 'subject_id')]),

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
        (nm_to_mni, smooth, [('output_image', 'in_files')]),

        # Data Saving 
        (realign, datasink1, [('realignment_parameters', 'NM_MotPar')]),
        (motion_node, datasink1, [('stats_file', 'NM_MotionStats'),
                                  ('plot_file', 'NM_MotionPlot')]),
        (run_compare, datasink1, [('out_png', 'QC_RunComparison')]),
        (nm_to_mni, datasink1, [('output_image', 'NM_to_MNI')]),
        (smooth, datasink1, [('smoothed_files', 'NM_Smooth')]),
        (qc_node_nm, datasink1, [('out_png', 'QC_NM_Plot')]),
        (qc_node_t1, datasink1, [('out_png', 'QC_T1_Plot')]),
        (bias, datasink1, [('output_image', 'NM_BiasCorr')]),
        (bias_mean_node, datasink1, [('mean_image', 'NM_BiasCorr_Mean')]),
        (realign, datasink1, [('mean_image', 'NM_Realigned_Mean')]),
        (t1_to_mni, datasink1, [('composite_transform', 'T1_to_MNI_Composite'),
                                ('inverse_composite_transform', 'T1_to_MNI_InvComposite')]),
        (nm_to_t1, datasink1, [('composite_transform', 'NM_to_T1_Composite'),
                               ('inverse_composite_transform', 'NM_to_T1_InvComposite')]),
    ])

    # =================================================================
    # Save DAG and Run Phase 1
    # =================================================================

    print(f"{'=' * 60}")
    print("Saving DAG of Job...")
    print(f"{'=' * 60}")

    wf1.write_graph(graph2use='colored', dotfilename=str(output_base / 'NM_PP_Phase-1_Pipeline_Dag'), simple_form=True)

    print(f"{'=' * 60}")
    print("Starting Phase 1: Forward Normalization...")
    print(f"{'=' * 60}")

    wf1.run(plugin='MultiProc', plugin_args={'n_procs': n_procs})

    print(f"{'=' * 60}")
    print("Creating Group Average...")
    print(f"{'=' * 60}")

    mni_files = list(output_base.glob('NM_to_MNI/sub-*/NM_MNI.nii*'))
    if not mni_files:
        raise FileNotFoundError("No NM_MNI files found - Check DataSink or WorkDir")

    group_average_path = f"{output_base}/Study-Specific_NM_Template.nii.gz"
    mean_img(mni_files).to_filename(group_average_path)
