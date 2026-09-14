#!/usr/bin/env python3
"""
Make the workflow panels for the poster.

Workflow_1a_Input_NM.png     the subject's NM scan (input)
Workflow_1b_Input_T1.png     the T1 reference (input, no NM contrast)
Workflow_2_Template.png      the study template
Workflow_3_Tracing.png       the template with the SN and CP tracings
Workflow_4_SearchRegion.png  the propagated search region in native space
Workflow_5_SubjectMask.png   the final NM-defined mask, same subject and slice

Every panel is also saved as SVG. The script reads one finished pipeline output
root: the V4 drawing template and its tracing, plus one subject's native-space NM
image, search region and NM-defined mask.

The Brainhack Western 2026 poster panels were made with:

    python workflow/figures/poster_workflow_panels.py \
        --output-root /path/to/NM_BioGen_PP/echo-1/results_dil-0.5 \
        --subject sub-059 --slice 30

The T1 panel needs the 0.5 mm MNI reference; if it is missing, generate it with
workflow/tools/resample_mni_reference.py.

A post-hoc figure script; it is not part of the pipeline.
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================================================================
# Settings
# =================================================================
# Resolved from this file's location so the default works from any checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T1_REFERENCE = (REPO_ROOT / "resources" / "tpl-MNI152NLin2009cAsym"
                        / "MNI152NLin2009cAsym_res-05_desc-brain_T1w.nii.gz")

DEFAULT_FOV_MM = (48.0, 36.0)      # field of view for every panel, in mm
DEFAULT_MM_PER_INCH = 8.0          # figure scale; all panels share it

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
    """Save an edge-to-edge panel as PNG and SVG, with no white border.

    The panel axes already fill the whole figure, so bbox_inches='tight' is not used:
    it adds 0.1 in of white padding around the image.
    """
    output_file = Path(output_file)
    fig.savefig(str(output_file.with_suffix('.png')), dpi=dpi)
    fig.savefig(str(output_file.with_suffix('.svg')))


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


def save_panel(background, overlays, output_file, fov_mm=DEFAULT_FOV_MM, mm_per_inch=DEFAULT_MM_PER_INCH):
    """One greyscale slice with filled and/or outlined overlays, filling the whole figure."""
    fig = plt.figure(figsize=(fov_mm[0] / mm_per_inch, fov_mm[1] / mm_per_inch))
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
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Pipeline output root containing drawing_template/, manual_trace_v4/ and sub-*/anat/.")
    parser.add_argument("--subject", required=True,
                        help="Subject shown in the native-space panels, e.g. sub-059.")
    parser.add_argument("--slice", type=int, default=None, dest="subject_slice",
                        help="Axial slice for the native-space panels (default: middle of the subject's mask).")
    parser.add_argument("--drawing-template", type=Path, default=None,
                        help="Default: <output-root>/drawing_template/SN_Template_V4_aligned.nii.gz.")
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Directory with MNI_SNc_Manual and MNI_CP_Manual (default: <output-root>/manual_trace_v4).")
    parser.add_argument("--t1-reference", type=Path, default=DEFAULT_T1_REFERENCE,
                        help="T1 shown in the input panel (default: the 0.5 mm MNI reference in resources/).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the panels (default: poster_figures/ beside --output-root).")
    parser.add_argument("--fov-mm", type=float, nargs=2, default=list(DEFAULT_FOV_MM),
                        metavar=("WIDTH", "HEIGHT"), help="Field of view of every panel in mm (default: 48 36).")
    parser.add_argument("--t1-fov-mm", type=float, nargs=2, default=list(DEFAULT_FOV_MM),
                        metavar=("WIDTH", "HEIGHT"),
                        help="Field of view of the T1 panel; widen it for context, e.g. 160 118 (default: 48 36).")
    parser.add_argument("--mm-per-inch", type=float, default=DEFAULT_MM_PER_INCH,
                        help="Figure scale shared by every panel (default: 8).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root
    drawing_template = args.drawing_template or output_root / "drawing_template" / "SN_Template_V4_aligned.nii.gz"
    trace_dir = args.trace_dir or output_root / "manual_trace_v4"
    panel_dir = args.out_dir or output_root.parent / "poster_figures"
    fov_mm, t1_fov_mm = tuple(args.fov_mm), tuple(args.t1_fov_mm)
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Template space: the template, the tracing, and the T1 behind them
    template_img = nib.load(str(drawing_template))
    template = template_img.get_fdata()
    template_zooms = template_img.header.get_zooms()[:3]
    sn_trace = nib.load(str(trace_dir / "MNI_SNc_Manual.nii.gz")).get_fdata() > 0
    cp_trace = nib.load(str(trace_dir / "MNI_CP_Manual.nii.gz")).get_fdata() > 0

    template_z = middle_slice(sn_trace)
    template_centre = centre_of(sn_trace | cp_trace)
    rows, cols = crop_at(template_centre, template_zooms, template.shape, fov_mm)
    print(f"template panels: slice {template_z}")

    save_panel(template[rows, cols, template_z], [], panel_dir / "Workflow_2_Template.png",
               fov_mm=fov_mm, mm_per_inch=args.mm_per_inch)
    save_panel(template[rows, cols, template_z],
               [(cp_trace[rows, cols, template_z], CP_COLOUR, "both"),
                (sn_trace[rows, cols, template_z], SN_COLOUR, "both")],
               panel_dir / "Workflow_3_Tracing.png", fov_mm=fov_mm, mm_per_inch=args.mm_per_inch)

    # the T1 sits in the same physical space, so match the crop through world coordinates
    t1_img = nib.load(str(args.t1_reference))
    t1_data = t1_img.get_fdata()
    world_point = nib.affines.apply_affine(template_img.affine,
                                           [template_centre[0], template_centre[1], template_z])
    t1_voxel = nib.affines.apply_affine(np.linalg.inv(t1_img.affine), world_point)
    t1_rows, t1_cols = crop_at(t1_voxel, t1_img.header.get_zooms()[:3], t1_data.shape, t1_fov_mm)
    save_panel(t1_data[t1_rows, t1_cols, int(round(t1_voxel[2]))], [],
               panel_dir / "Workflow_1b_Input_T1.png", fov_mm=t1_fov_mm, mm_per_inch=args.mm_per_inch)

    # Native space: the input NM, the search region, and the final mask
    anat_dir = output_root / args.subject / "anat"
    native_img = nib.load(str(anat_dir / "mean_NM_bias_corrected.nii.gz"))
    native_nm = native_img.get_fdata()
    native_zooms = native_img.header.get_zooms()[:3]

    region_file = anat_dir / "Native_SN_Search_Region.nii.gz"
    if not region_file.exists():
        # Outputs from before v3.0 have no search region file; the propagated drawing is the same region.
        region_file = anat_dir / "MNI_SNc_Manual_trans.nii.gz"
    search_region = nib.load(str(region_file)).get_fdata() > 0
    subject_mask = nib.load(str(anat_dir / "Native_Tractography_Seed_SNc.nii.gz")).get_fdata() > 0

    subject_z = args.subject_slice if args.subject_slice is not None else middle_slice(subject_mask)
    subject_rows, subject_cols = crop_at(centre_of(search_region), native_zooms, native_nm.shape, fov_mm)
    print(f"subject panels: {args.subject}, slice {subject_z}, region from {region_file.name}")

    save_panel(native_nm[subject_rows, subject_cols, subject_z], [],
               panel_dir / "Workflow_1a_Input_NM.png", fov_mm=fov_mm, mm_per_inch=args.mm_per_inch)
    save_panel(native_nm[subject_rows, subject_cols, subject_z],
               [(search_region[subject_rows, subject_cols, subject_z], REGION_COLOUR, "both")],
               panel_dir / "Workflow_4_SearchRegion.png", fov_mm=fov_mm, mm_per_inch=args.mm_per_inch)
    save_panel(native_nm[subject_rows, subject_cols, subject_z],
               [(subject_mask[subject_rows, subject_cols, subject_z], MASK_COLOUR, "both")],
               panel_dir / "Workflow_5_SubjectMask.png", fov_mm=fov_mm, mm_per_inch=args.mm_per_inch)


if __name__ == "__main__":
    main()
