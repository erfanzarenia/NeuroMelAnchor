#!/usr/bin/env python3
"""
Prepare images for a native-reference NM template.

Each subject is mapped directly onto one selected reference subject's
native NM grid:

    subject NM -> subject T1 -> MNI -> reference T1 -> reference NM

Outputs:
    <output_root>/reference/reference_NM.nii.gz
    <output_root>/masks/template_mask.nii.gz
    <output_root>/masks/registration_mask.nii.gz
    <output_root>/masks/left_search_mask.nii.gz
    <output_root>/masks/right_search_mask.nii.gz
    <output_root>/sub-*/anat/NM_space-studyRef_desc-prealigned.nii.gz
"""
# =============================================================================
# Import
# =============================================================================

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# =============================================================================
# Configs
# =============================================================================

# Same MNI midbrain bounds used by Template_Builder_V4.py.
midbrain_x_bounds_ras = (-20.0, 20.0)
midbrain_y_bounds_ras = (-38.0, -8.0)
midbrain_z_bounds_ras = (-28.0, 2.0)

template_dilation_mm = 8.0
search_dilation_mm = 3.0
midline_exclusion_mm = 5.0

native_nm_name = "mean_NM_bias_corrected.nii.gz"
prepared_nm_name = "NM_space-studyRef_desc-prealigned.nii.gz"

# =============================================================================
# Helper Functions
# =============================================================================

# Resolved from this file's location so the script works from any checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MNI_PRIOR = REPO_ROOT / "resources" / "tpl-MNI152NLin2009cAsym" / "HarvardOxford" / "Midbrain.nii.gz"

def trim_prior_to_midbrain(mni_prior):
    """Restrict the MNI prior to the fixed SN/midbrain region."""
    prior_array = sitk.GetArrayFromImage(mni_prior).astype(np.float32)
    trimmed_array = np.zeros_like(prior_array, dtype=np.uint8)

    x_low, x_high = sorted(midbrain_x_bounds_ras)
    y_low, y_high = sorted(midbrain_y_bounds_ras)
    z_low, z_high = sorted(midbrain_z_bounds_ras)

    for z_index, y_index, x_index in np.argwhere(prior_array > 0):
        point_lps = mni_prior.TransformIndexToPhysicalPoint(
            (int(x_index), int(y_index), int(z_index))
        )
        point_ras = (-point_lps[0], -point_lps[1], point_lps[2])

        if (
            x_low <= point_ras[0] <= x_high
            and y_low <= point_ras[1] <= y_high
            and z_low <= point_ras[2] <= z_high
        ):
            trimmed_array[z_index, y_index, x_index] = 1

    if not trimmed_array.any():
        raise ValueError("The midbrain-trimmed prior is empty.")

    trimmed_prior = sitk.GetImageFromArray(trimmed_array)
    trimmed_prior.CopyInformation(mni_prior)
    return trimmed_prior

def split_mni_prior(mni_prior):
    """Split the MNI prior into anatomical left and right masks."""
    prior_array = sitk.GetArrayFromImage(mni_prior).astype(np.uint8)
    left_array = np.zeros_like(prior_array, dtype=np.uint8)
    right_array = np.zeros_like(prior_array, dtype=np.uint8)

    for z_index, y_index, x_index in np.argwhere(prior_array > 0):
        point_lps = mni_prior.TransformIndexToPhysicalPoint(
            (int(x_index), int(y_index), int(z_index))
        )

        # In the SimpleITK LPS coordinate system, positive x is left.
        if point_lps[0] >= 0.0:
            left_array[z_index, y_index, x_index] = 1
        else:
            right_array[z_index, y_index, x_index] = 1

    if not left_array.any() or not right_array.any():
        raise ValueError("Could not split the MNI prior into left and right masks.")

    left_prior = sitk.GetImageFromArray(left_array)
    left_prior.CopyInformation(mni_prior)

    right_prior = sitk.GetImageFromArray(right_array)
    right_prior.CopyInformation(mni_prior)

    return left_prior, right_prior


def exclude_midline_band(prior_mask, half_width_mm):
    """Remove mask voxels within a specified distance of the MNI midline."""
    mask_array = sitk.GetArrayFromImage(prior_mask).astype(np.uint8)
    output_array = mask_array.copy()

    for z_index, y_index, x_index in np.argwhere(mask_array > 0):
        point_lps = prior_mask.TransformIndexToPhysicalPoint(
            (int(x_index), int(y_index), int(z_index))
        )

        if abs(point_lps[0]) < half_width_mm:
            output_array[z_index, y_index, x_index] = 0

    output_mask = sitk.GetImageFromArray(output_array)
    output_mask.CopyInformation(prior_mask)
    return output_mask


def dilate_prior(prior, dilation_mm):
    """Dilate a binary mask by approximately the requested distance in mm."""
    radii = [
        max(1, int(round(dilation_mm / spacing)))
        for spacing in prior.GetSpacing()
    ]
    return sitk.BinaryDilate(prior, radii)

def run_ants_apply_transforms(
    input_image,
    reference_image,
    output_image,
    transforms,
    interpolation,
):
    """Run one ANTs resampling operation."""
    command = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(input_image),
        "-r", str(reference_image),
        "-o", str(output_image),
        "-n", interpolation,
    ]

    # ANTs applies the final transform first.
    for transform in transforms:
        command.extend(["-t", str(transform)])

    print(f"  Writing: {output_image}")
    subprocess.run(command, check=True)

# =============================================================================
# Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare native-reference inputs for NM template building.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-subject", required=True)
    parser.add_argument("--mni-prior", type=Path, default=DEFAULT_MNI_PRIOR,
                        help=f"MNI midbrain prior (default: {DEFAULT_MNI_PRIOR}).")
    args = parser.parse_args()

    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.source_root}")

    if not args.mni_prior.exists():
        raise FileNotFoundError(f"MNI prior not found: {args.mni_prior}")

    if args.output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {args.output_root}\n"
            "Choose a new output directory so this run cannot mix files "
            "from different reference subjects."
        )

    subject_dirs = sorted(
        path for path in args.source_root.glob("sub-*") if path.is_dir()
    )

    if not subject_dirs:
        raise FileNotFoundError(f"No sub-* directories found in {args.source_root}")

    reference_dir = args.source_root / args.reference_subject
    reference_nm = reference_dir / "anat" / native_nm_name

    if not reference_nm.exists():
        raise FileNotFoundError(
            f"Reference NM image not found: {reference_nm}"
        )

    reference_nm_to_t1_inverse = (
        reference_dir
        / "transforms"
        / "NM_to_T1_InvComposite"
        / "transformInverseComposite.h5"
    )
    reference_t1_to_mni_inverse = (
        reference_dir
        / "transforms"
        / "T1_to_MNI_InvComposite"
        / "transformInverseComposite.h5"
    )

    for transform in [
        reference_nm_to_t1_inverse,
        reference_t1_to_mni_inverse,
    ]:
        if not transform.exists():
            raise FileNotFoundError(f"Reference transform not found: {transform}")

    args.output_root.mkdir()
    reference_output = args.output_root / "reference" / "reference_NM.nii.gz"
    reference_output.parent.mkdir(parents=True)

    # Do not resample the reference subject.
    shutil.copy2(reference_nm, reference_output)

    print(f"Found {len(subject_dirs)} subject(s).")
    print(f"Reference subject: {args.reference_subject}")
    print("\nPreparing subject images...")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        subject_nm = subject_dir / "anat" / native_nm_name

        output_image = (
            args.output_root
            / subject_id
            / "anat"
            / prepared_nm_name
        )
        output_image.parent.mkdir(parents=True)

        if not subject_nm.exists():
            raise FileNotFoundError(f"Native NM image not found: {subject_nm}")

        if subject_id == args.reference_subject:
            shutil.copy2(subject_nm, output_image)
            continue

        subject_nm_to_t1 = (
            subject_dir
            / "transforms"
            / "NM_to_T1_Composite"
            / "transformComposite.h5"
        )
        subject_t1_to_mni = (
            subject_dir
            / "transforms"
            / "T1_to_MNI_Composite"
            / "transformComposite.h5"
        )

        for transform in [subject_nm_to_t1, subject_t1_to_mni]:
            if not transform.exists():
                raise FileNotFoundError(f"Subject transform not found: {transform}")

        # Listed in ANTs order; the last transform is applied first.
        run_ants_apply_transforms(
            input_image=subject_nm,
            reference_image=reference_output,
            output_image=output_image,
            transforms=[
                reference_nm_to_t1_inverse,
                reference_t1_to_mni_inverse,
                subject_t1_to_mni,
                subject_nm_to_t1,
            ],
            interpolation="LanczosWindowedSinc",
        )

    print("\nPreparing reference-space masks...")

    mni_prior = sitk.ReadImage(str(args.mni_prior))
    trimmed_prior = trim_prior_to_midbrain(mni_prior)

    # Construct the masks while x=0 still represents the MNI midline.
    template_mask_mni = dilate_prior(
        trimmed_prior,
        template_dilation_mm,
    )

    registration_mask_mni = exclude_midline_band(
        template_mask_mni,
        midline_exclusion_mm,
    )

    left_prior_mni, right_prior_mni = split_mni_prior(trimmed_prior)

    left_search_mask_mni = exclude_midline_band(
        dilate_prior(left_prior_mni, search_dilation_mm),
        midline_exclusion_mm,
    )

    right_search_mask_mni = exclude_midline_band(
        dilate_prior(right_prior_mni, search_dilation_mm),
        midline_exclusion_mm,
    )

    mni_masks = {
        "template_mask": template_mask_mni,
        "registration_mask": registration_mask_mni,
        "left_search_mask": left_search_mask_mni,
        "right_search_mask": right_search_mask_mni,
    }

    masks_dir = args.output_root / "masks"
    masks_dir.mkdir(parents=True)

    # ANTs requires file inputs. These MNI files exist only temporarily.
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)

        for mask_name, mni_mask in mni_masks.items():
            temporary_mask = temporary_dir / f"{mask_name}_mni.nii.gz"
            output_mask = masks_dir / f"{mask_name}.nii.gz"

            sitk.WriteImage(mni_mask, str(temporary_mask))

            # MNI -> reference T1 -> reference NM
            run_ants_apply_transforms(
                input_image=temporary_mask,
                reference_image=reference_output,
                output_image=output_mask,
                transforms=[
                    reference_nm_to_t1_inverse,
                    reference_t1_to_mni_inverse,
                ],
                interpolation="NearestNeighbor",
            )

    print("\nPreparation complete.")
    print(f"Input root:        {args.output_root}")
    print(f"Reference image:   {reference_output}")
    print(f"Template mask:     {masks_dir / 'template_mask.nii.gz'}")
    print(f"Registration mask: {masks_dir / 'registration_mask.nii.gz'}")
    print(f"Left search mask:  {masks_dir / 'left_search_mask.nii.gz'}")
    print(f"Right search mask: {masks_dir / 'right_search_mask.nii.gz'}")

if __name__ == "__main__":
    main()