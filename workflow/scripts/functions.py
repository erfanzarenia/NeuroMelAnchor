"""
Helper functions moved from the original script with no internal logic changes.
"""

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
    display.add_overlay(t1_mni_image, color='r')
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

def mean_of_bias_corrected(in_files):
    from nilearn.image import mean_img
    import os
    out = os.path.abspath('mean_NM_bias_corrected.nii.gz')
    mean_img(in_files).to_filename(out)
    return out

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
    
    # def hausdroff(a,b):
    #     a = a.astype(bool)
    #     b = b.astype(bool)
    
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
