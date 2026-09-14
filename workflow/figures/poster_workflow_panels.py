#!/usr/bin/env python3
"""
Make the workflow panels for the poster.

Workflow_1a_Input_NM.png     the subject's NM scan (input)
Workflow_1b_Input_T1.png     the T1 reference (input, no NM contrast)
Workflow_2_Template.png      the study template
Workflow_3_Tracing.png       the template with the SN and CP tracings
Workflow_4_SearchRegion.png  the propagated search region in native space
Workflow_5_SubjectMask.png   the final NM-defined mask, same subject and slice

"""
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================================================================
# Settings
# =================================================================
OUTPUT_ROOT = Path("/scratch/Projects/BrainHack_Nexus/NM_BioGen_PP/echo-1/results_dil-0.5")
DRAWING_TEMPLATE = OUTPUT_ROOT / "drawing_template" / "SN_Template_V4_aligned.nii.gz"
TRACE_DIR = OUTPUT_ROOT / "manual_trace_v4"
T1_REFERENCE = Path("/scratch/Projects/BrainHack_Nexus/resources/tpl-MNI152NLin2009cAsym/MNI152NLin2009cAsym_res-05_desc-brain_T1w.nii.gz")
PANEL_DIR = OUTPUT_ROOT.parent / "poster_figures"

SUBJECT = "sub-059"                # the subject shown in the native-space panels
SUBJECT_SLICE = 30                 # None = middle of the mask

FOV_MM = (48.0, 36.0)              # field of view for every panel, in mm
T1_FOV_MM = (48.0, 36.0)           # widen for context, e.g. (160.0, 118.0) keeps the same aspect
MM_PER_INCH = 8.0                  # figure scale; all panels share it

SN_COLOUR = "#2ECC71"              # green, as drawn in ITK-SNAP
CP_COLOUR = "#FF4136"              # red, as drawn in ITK-SNAP
REGION_COLOUR = "#FFD400"          # the propagated search region
MASK_COLOUR = "#FF3B30"            # the final mask, matching the results figures


# =================================================================
# Helpers
# =================================================================
def middle_slice(mask):
    """The middle of the slices where this mask actually has substance."""
    slice_areas = mask.sum(axis=(0, 1))
    solid_slices = np.flatnonzero(slice_areas >= 0.10 * slice_areas.max())
    return int(np.percentile(solid_slices, 50))

def save_poster_figure(fig, output_file, dpi=400):
    """Save a poster figure as PNG (300 dpi) plus SVG and PDF, which stay sharp at any print size."""
    output_file = Path(output_file)
    fig.savefig(str(output_file), dpi=dpi, bbox_inches='tight')
    fig.savefig(str(output_file.with_suffix('.svg')), bbox_inches='tight')
    fig.savefig(str(output_file.with_suffix('.png')), dpi=dpi, bbox_inches='tight')

def centre_of(mask):
    """Centre of a mask, in voxels."""
    return np.argwhere(mask).mean(axis=0)


def crop_at(centre_voxel, zooms, shape, fov_mm):
    """A fixed field of view in mm around a point, returned as row and column slices."""
    half_widths = [int(round(fov_mm[axis] / 2.0 / zooms[axis])) for axis in (0, 1)]
    starts = []
    for axis in (0, 1):
        start = int(round(centre_voxel[axis])) - half_widths[axis]
        starts.append(min(max(start, 0), shape[axis] - 2 * half_widths[axis]))
    return (slice(starts[0], starts[0] + 2 * half_widths[0]),
            slice(starts[1], starts[1] + 2 * half_widths[1]))


def save_panel(background, overlays, output_file, fov_mm=FOV_MM):
    """One greyscale slice with filled and/or outlined overlays, filling the whole figure."""
    fig = plt.figure(figsize=(fov_mm[0] / MM_PER_INCH, fov_mm[1] / MM_PER_INCH))
    axis = fig.add_axes([0, 0, 1, 1])

    shown = background[background > 0]
    vmin, vmax = np.percentile(shown, [2, 99]) if shown.size else (0, 1)
    axis.imshow(np.rot90(background), cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")

    for plane, colour, style in overlays:
        rotated = np.rot90(plane)
        if not rotated.any():
            continue
        if style in ("fill", "both"):
            axis.contourf(rotated, levels=[0.5, 1.5], colors=[colour], alpha=0.45)
        if style in ("outline", "both"):
            axis.contour(rotated, levels=[0.5], colors=colour, linewidths=2.5)

    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    save_poster_figure(fig, output_file)
    plt.close(fig)
    print(f"saved: {output_file.name}")


# =================================================================
# Panels
# =================================================================
def main():
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    # Template space: the template, the tracing, and the T1 behind them
    template_img = nib.load(str(DRAWING_TEMPLATE))
    template = template_img.get_fdata()
    template_zooms = template_img.header.get_zooms()[:3]
    sn_trace = nib.load(str(TRACE_DIR / "MNI_SNc_Manual.nii.gz")).get_fdata() > 0
    cp_trace = nib.load(str(TRACE_DIR / "MNI_CP_Manual.nii.gz")).get_fdata() > 0

    template_z = middle_slice(sn_trace)
    template_centre = centre_of(sn_trace | cp_trace)
    rows, cols = crop_at(template_centre, template_zooms, template.shape, FOV_MM)
    print(f"template panels: slice {template_z}")

    save_panel(template[rows, cols, template_z], [], PANEL_DIR / "Workflow_2_Template.png")
    save_panel(template[rows, cols, template_z],
               [(cp_trace[rows, cols, template_z], CP_COLOUR, "both"),
                (sn_trace[rows, cols, template_z], SN_COLOUR, "both")],
               PANEL_DIR / "Workflow_3_Tracing.png")

    # the T1 sits in the same physical space, so match the crop through world coordinates
    t1_img = nib.load(str(T1_REFERENCE))
    t1_data = t1_img.get_fdata()
    world_point = nib.affines.apply_affine(template_img.affine,
                                           [template_centre[0], template_centre[1], template_z])
    t1_voxel = nib.affines.apply_affine(np.linalg.inv(t1_img.affine), world_point)
    t1_rows, t1_cols = crop_at(t1_voxel, t1_img.header.get_zooms()[:3], t1_data.shape, T1_FOV_MM)
    save_panel(t1_data[t1_rows, t1_cols, int(round(t1_voxel[2]))], [],
               PANEL_DIR / "Workflow_1b_Input_T1.png", fov_mm=T1_FOV_MM)

    # Native space: the input NM, the search region, and the final mask
    anat_dir = OUTPUT_ROOT / SUBJECT / "anat"
    native_img = nib.load(str(anat_dir / "mean_NM_bias_corrected.nii.gz"))
    native_nm = native_img.get_fdata()
    native_zooms = native_img.header.get_zooms()[:3]

    region_file = anat_dir / "Native_SN_Search_Region.nii.gz"
    if not region_file.exists():
        region_file = anat_dir / "MNI_SNc_Manual_trans.nii.gz"
    search_region = nib.load(str(region_file)).get_fdata() > 0
    subject_mask = nib.load(str(anat_dir / "Native_Tractography_Seed_SNc.nii.gz")).get_fdata() > 0

    subject_z = SUBJECT_SLICE if SUBJECT_SLICE is not None else middle_slice(subject_mask)
    subject_rows, subject_cols = crop_at(centre_of(search_region), native_zooms, native_nm.shape, FOV_MM)
    print(f"subject panels: {SUBJECT}, slice {subject_z}, region from {region_file.name}")

    save_panel(native_nm[subject_rows, subject_cols, subject_z], [],
               PANEL_DIR / "Workflow_1a_Input_NM.png")
    save_panel(native_nm[subject_rows, subject_cols, subject_z],
               [(search_region[subject_rows, subject_cols, subject_z], REGION_COLOUR, "both")],
               PANEL_DIR / "Workflow_4_SearchRegion.png")
    save_panel(native_nm[subject_rows, subject_cols, subject_z],
               [(subject_mask[subject_rows, subject_cols, subject_z], MASK_COLOUR, "both")],
               PANEL_DIR / "Workflow_5_SubjectMask.png")


if __name__ == "__main__":
    main()
