#!/usr/bin/env python3
"""
Build a substantia-nigra-centred neuromelanin template with FireANTs.

Assumes Phase 1 has placed each subject's native mean NM image at:
    <output_root>/sub-*/anat/mean_NM_bias_corrected.nii.gz

Workflow:
    1. Trim an external MNI brainstem prior to a fixed midbrain subregion.
    2. Split the trimmed prior into bilateral L/R halves.
    3. Find each subject's L/R SN centroid in MNI from top-intensity voxels.
    4. Pre-align each subject to the cohort-average centroid via translation.
    5. Mask each subject to the midbrain region.
    6. Initial template = correlation-weighted mean of pre-aligned, masked subjects.
    7. Iteratively refine with FireANTs SyN; output in MNI grid.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import json

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from fireants.io.image import BatchedImages, Image
from fireants.registration.rigid import RigidRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.syn import SyNRegistration
from fireants.utils.warputils import shape_averaging_invwarp
from fireants.utils.imageutils import LaplacianFilter
from fireants.interpolator import fireants_interpolator


# =============================================================================
# Config
# =============================================================================

# Resolved from this file's location so the builder works from any checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
default_mni_prior = REPO_ROOT / "resources" / "tpl-MNI152NLin2009cAsym" / "HarvardOxford" / "Midbrain.nii.gz"

target_spacing = (0.5, 0.5, 0.5)
midbrain_x_bounds_ras = (-20.0, 20.0)
midbrain_y_bounds_ras = (-38.0, -8.0)
midbrain_z_bounds_ras = (-28.0, 2.0)
search_dilation_mm = 3.0
top_intensity_percentile = 85.0
minimum_centroid_voxels = 12

scales = [4.0, 2.0, 1.0]
syn_iterations = [80, 40, 40]
syn_lr = 0.2
cc_kernel_size = 7            # was 5: at 0.5mm a 5-voxel CC window is only 2.5mm (noisy); 7 is roughly 3.5mm
rigid_iterations = [100, 50, 25]
affine_iterations = [100, 50, 25]

# SyN deformation regularization (in voxels). FireANTs defaults are grad=1.0 / warp=0.5.
syn_smooth_grad_sigma = 1.5   # >default: smoother update field to less noise-driven warping
syn_smooth_warp_sigma = 0.5

lower_pct = 1.0
upper_pct = 99.0

shape_avg_step = 0.25   # damping on the averaged warp before inversion (see build_fireants_template).
                        # ANTs' antsMultivariateTemplateConstruction2.sh always damps its shape update
                        # with a "gradientstep" (default ~0.25, verified: AverageImages to MultiplyImages
                        # by gradientstep) rather than applying the full averaged correction each round.


midline_exclusion_mm = 5.0  # exclude voxels within this distance of the midline


# =============================================================================
# Utility Functions
# =============================================================================

def hard_gc():
    """Release Python and CUDA memory between registrations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def robust_normalise(image, lower, upper):
    """Normalizes each image by clipping intensity to a percentile range and scaling to [0, 1]."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    values = arr[np.isfinite(arr)]
    values = values[np.abs(values) > 1e-6]
    if values.size == 0:
        raise ValueError("Cannot normalise an image with no finite non-zero voxels.")

    lo, hi = np.percentile(values, [lower, upper])
    arr = (arr - lo) / (hi - lo + 1e-8)
    arr = np.maximum(arr, 0.0).astype(np.float32) 

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out

def average_sitk(images, reference):
    """Computes the average of a list of SimpleITK images."""
    if not images:
        raise ValueError("No images were provided for averaging.")

    arr = np.stack([
        sitk.GetArrayFromImage(image).astype(np.float32)
        for image in images
    ]).mean(axis=0)

    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.CopyInformation(reference)
    return out


def collect_subject_inputs(output_root):
    """Collect MNI-space NM images per subject."""
    subject_dirs = sorted(path for path in output_root.glob("sub-*") if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found under {output_root}")

    subject_records = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        anat_dir = subject_dir / "anat"
        mni_nm = anat_dir / "NM_MNI_res-05.nii"
        if not mni_nm.exists():
            mni_nm = anat_dir / "NM_MNI.nii"
        if not mni_nm.exists():
            mni_nm = anat_dir / "NM_MNI.nii.gz"
        if not mni_nm.exists():
            print(f"  Skipping {subject_id}: no NM_MNI image in {anat_dir}", flush=True)
            continue

        subject_records.append({
            "subject_id": subject_id,
            "mni_nm": mni_nm,
        })

    return subject_records


# =============================================================================
# QC Functions
# =============================================================================

def crop_qc_records(crops, ids, template):
    """Per-subject QC records comparing each crop with the current template."""
    template_arr = sitk.GetArrayFromImage(template).astype(np.float32)
    records = []

    for index, (crop, subject_id) in enumerate(zip(crops, ids)):
        arr = sitk.GetArrayFromImage(crop).astype(np.float32)
        mask = np.isfinite(arr) & np.isfinite(template_arr)
        mask &= (np.abs(arr) > 1e-6) | (np.abs(template_arr) > 1e-6)

        corr = None
        mad = None

        if mask.sum() > 10:
            subject_values = arr[mask]
            template_values = template_arr[mask]

            mad = float(np.mean(np.abs(subject_values - template_values)))

            if subject_values.std() > 1e-8 and template_values.std() > 1e-8:
                corr = float(np.corrcoef(subject_values, template_values)[0, 1])

        records.append({
            "index": index,
            "subject_id": subject_id,
            "corr": corr,
            "mad": mad,
            "score": None,
            "iqr_outlier": False,
        })

    corrs = np.array([np.nan if r["corr"] is None else r["corr"] for r in records], dtype=float)
    mads = np.array([np.nan if r["mad"] is None else r["mad"] for r in records], dtype=float)

    def robust_zscores(values):
        finite = np.isfinite(values)
        zscores = np.zeros(values.shape, dtype=float)

        if finite.sum() < 2:
            return zscores

        finite_values = values[finite]
        median = np.median(finite_values)
        mad = np.median(np.abs(finite_values - median))
        scale = 1.4826 * mad

        if scale < 1e-8:
            scale = np.std(finite_values)
        if scale < 1e-8:
            return zscores

        zscores[finite] = (values[finite] - median) / scale
        return np.clip(zscores, -5.0, 5.0)

    corr_z = robust_zscores(corrs)
    mad_z = robust_zscores(mads)

    scores = -corr_z + mad_z

    for index, record in enumerate(records):
        if record["corr"] is None:
            scores[index] += 3.0
        if record["mad"] is None:
            scores[index] += 3.0
        record["score"] = float(scores[index])

    finite_scores = np.asarray([record["score"] for record in records], dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size >= 4:
        q1, q3 = np.percentile(finite_scores, [25.0, 75.0])
        upper_fence = q3 + (1.5 * (q3 - q1))

        for record in records:
            record["iqr_outlier"] = record["score"] > upper_fence

    records.sort(key=lambda record: record["score"], reverse=True)

    for rank, record in enumerate(records, start=1):
        record["rank"] = rank

    return records

def save_contact_sheet(crops, ids, template, path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.set_loglevel("warning")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"  Contact sheet skipped: matplotlib is not available: {exc}", flush=True)
        return

    records = crop_qc_records(crops, ids, template)
    n_rows = len(records)
    fig_height = max(3.0, 1.15 * n_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, fig_height), squeeze=False)

    column_titles = ["Sagittal", "Coronal", "Axial"]
    for row, record in enumerate(records):
        arr = sitk.GetArrayFromImage(crops[record["index"]]).astype(np.float32)
        z_mid, y_mid, x_mid = [dim // 2 for dim in arr.shape]
        slices = [
            arr[:, :, x_mid],
            arr[:, y_mid, :],
            arr[z_mid, :, :],
        ]

        for col, slice_arr in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(
                np.rot90(slice_arr),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(column_titles[col], fontsize=9)

        score_text = "NA" if record["score"] is None else f"{record['score']:.2f}"
        corr_text = "NA" if record["corr"] is None else f"{record['corr']:.2f}"
        mad_text = "NA" if record["mad"] is None else f"{record['mad']:.3f}"
        label_lines = [
            (f"{record['rank']}. {record['subject_id']}", "black", 0.63),
            (
                f"score={score_text}",
                "red" if record["iqr_outlier"] else "black",
                0.50,
            ),
            (
                f"corr={corr_text} mad={mad_text}",
                "black",
                0.37,
            ),
        ]

        for text, color, y_position in label_lines:
            axes[row, 0].text(-0.08, y_position, text, transform=axes[row, 0].transAxes, ha="right",
                va="center", fontsize=6, color=color,)

    top_margin = 0.84 if n_rows <= 3 else 0.92 if n_rows <= 8 else 0.97
    fig.suptitle("SN Crop QC: highest corr/mad score first", fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.24, right=0.99, top=top_margin, bottom=0.01, wspace=0.03, hspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Crop QC contact sheet: {path}", flush=True)


# =============================================================================
# Crop Functions
# =============================================================================
def trim_prior_to_midbrain(mni_prior):
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
        raise ValueError(
            "Midbrain-trimmed prior is empty. "
            f"x={midbrain_x_bounds_ras}, y={midbrain_y_bounds_ras}, z={midbrain_z_bounds_ras}"
        )

    trimmed_prior = sitk.GetImageFromArray(trimmed_array.astype(np.uint8))
    trimmed_prior.CopyInformation(mni_prior)
    return trimmed_prior

def build_template_mask(trimmed_prior_mni, reference_image_mni, dilation_mm=4.0):
    """Resample the trimmed midbrain prior onto the reference grid and dilate."""
    identity = sitk.Transform(3, sitk.sitkIdentity)
    resampled = sitk.Resample(
        trimmed_prior_mni,
        reference_image_mni,
        identity,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return dilate_prior(resampled, dilation_mm)


def split_mni_prior(mni_prior):
    """Split a prior at the midline into left (LPS x >= 0) and right halves."""
    prior_array = sitk.GetArrayFromImage(mni_prior).astype(np.float32)
    left_array = np.zeros_like(prior_array, dtype=np.uint8)
    right_array = np.zeros_like(prior_array, dtype=np.uint8)

    for z_index, y_index, x_index in np.argwhere(prior_array > 0):
        point = mni_prior.TransformIndexToPhysicalPoint(
            (int(x_index), int(y_index), int(z_index))
        )
        if point[0] >= 0.0:
            left_array[z_index, y_index, x_index] = 1
        else:
            right_array[z_index, y_index, x_index] = 1

    if not left_array.any() or not right_array.any():
        raise ValueError(f"Could not split the MNI prior into bilateral halves: {default_mni_prior}")

    left_prior = sitk.GetImageFromArray(left_array.astype(np.uint8))
    left_prior.CopyInformation(mni_prior)
    right_prior = sitk.GetImageFromArray(right_array.astype(np.uint8))
    right_prior.CopyInformation(mni_prior)
    return left_prior, right_prior

def exclude_midline_band(prior_mask, half_width_mm):
    arr = sitk.GetArrayFromImage(prior_mask).astype(np.uint8)
    out = arr.copy()
    for z_index, y_index, x_index in np.argwhere(arr > 0):
        point = prior_mask.TransformIndexToPhysicalPoint(
            (int(x_index), int(y_index), int(z_index))
        )
        if abs(point[0]) < half_width_mm:
            out[z_index, y_index, x_index] = 0
    result = sitk.GetImageFromArray(out)
    result.CopyInformation(prior_mask)
    return result



def weighted_centroid(image, search_mask):
    """Intensity-weighted centroid, in physical coordinates, of the brightest voxels in a search mask."""
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    search_array = sitk.GetArrayFromImage(search_mask).astype(bool)
    finite_mask = np.isfinite(image_array)
    candidate_values = image_array[search_array & finite_mask]
    if candidate_values.size == 0:
        raise ValueError("Search mask contains no finite intensities.")

    threshold = np.percentile(candidate_values, float(top_intensity_percentile))
    candidate_mask = search_array & finite_mask & (image_array >= threshold)
    if int(candidate_mask.sum()) < int(minimum_centroid_voxels):
        fallback_threshold = np.percentile(candidate_values, 75.0)
        candidate_mask = search_array & finite_mask & (image_array >= fallback_threshold)
    if int(candidate_mask.sum()) == 0:
        candidate_mask = search_array & finite_mask

    weights = image_array[candidate_mask].astype(np.float64)
    weights -= weights.min()
    if np.all(weights <= 1e-8):
        weights = np.ones_like(weights, dtype=np.float64)
    else:
        weights += 1e-6

    coordinates_zyx = np.argwhere(candidate_mask)
    centroid_xyz = np.average(
        coordinates_zyx[:, ::-1].astype(np.float64),
        axis=0,
        weights=weights,
    )
    return image.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in centroid_xyz))

def find_subject_centroids(mni_image, left_prior_dilated, right_prior_dilated):
    """Return (left_xyz_mm, right_xyz_mm) MNI physical coords of intensity-weighted centroids."""
    left_xyz = weighted_centroid(mni_image, left_prior_dilated)
    right_xyz = weighted_centroid(mni_image, right_prior_dilated)
    return left_xyz, right_xyz


def dilate_prior(prior, dilation_mm):
    """Isotropic mm-spec'd dilation of a binary mask."""
    spacing = prior.GetSpacing()
    radii = [max(1, int(round(dilation_mm / s))) for s in spacing]
    return sitk.BinaryDilate(prior, radii)

def cohort_average_centroids(per_subject_centroids):
    """per_subject_centroids: list of (left_xyz, right_xyz) tuples."""
    lefts = np.array([np.asarray(c[0]) for c in per_subject_centroids])
    rights = np.array([np.asarray(c[1]) for c in per_subject_centroids])
    return tuple(lefts.mean(axis=0).tolist()), tuple(rights.mean(axis=0).tolist())


def translation_to_target(subject_left, subject_right, target_left, target_right):
    """SimpleITK translation that brings subject midpoint to target midpoint.
    Used as the transform argument to sitk.Resample (which maps output→input)."""
    subject_mid = (np.asarray(subject_left) + np.asarray(subject_right)) / 2.0
    target_mid = (np.asarray(target_left) + np.asarray(target_right)) / 2.0
    # Resample's transform maps output→input. To put subject's anatomy AT the target position
    # in the output, we sample from (output_point + (subject_mid - target_mid)) in the input.
    offset = (subject_mid - target_mid).tolist()
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(tuple(float(v) for v in offset))
    return transform

def resample_to_spacing(image, new_spacing, interpolator=sitk.sitkBSpline):
    old_spacing = np.array(image.GetSpacing())
    old_size = np.array(image.GetSize())
    new_size = [int(round(s * o / n)) for s, o, n in zip(old_size, old_spacing, new_spacing)]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        image.GetOrigin(),
        tuple(float(v) for v in new_spacing),
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )

def spatial_crop(image, z_lo, z_hi, y_lo, y_hi, x_lo, x_hi):
    arr = sitk.GetArrayFromImage(image)[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    new_origin = image.TransformIndexToPhysicalPoint((int(x_lo), int(y_lo), int(z_lo)))
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(image.GetSpacing())
    out.SetDirection(image.GetDirection())
    out.SetOrigin(new_origin)
    return out

def prepare_subject_crops(subject_records, work_dir, mni_prior_path, save_subject_crops):
    """Centroid-align, mask, crop and normalize every subject, then build the initial template."""
    if not subject_records:
        raise FileNotFoundError("No subject records provided.")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Build the L/R midbrain priors in MNI 
    mni_prior = sitk.ReadImage(str(mni_prior_path), sitk.sitkUInt8)
    trimmed_prior = trim_prior_to_midbrain(mni_prior)
    left_prior, right_prior = split_mni_prior(trimmed_prior)
    left_prior_dilated = dilate_prior(left_prior, search_dilation_mm)
    right_prior_dilated = dilate_prior(right_prior, search_dilation_mm)
    left_prior_dilated = exclude_midline_band(left_prior_dilated, midline_exclusion_mm)
    right_prior_dilated = exclude_midline_band(right_prior_dilated, midline_exclusion_mm)


    # Read each subject in MNI grid, locate per-side centroids
    first = sitk.ReadImage(str(subject_records[0]["mni_nm"]), sitk.sitkFloat32)
    first = resample_to_spacing(first, target_spacing)
    identity = sitk.Transform(3, sitk.sitkIdentity)

    # Resample priors to the first subject's MNI grid (the reference grid for the whole build)
    left_on_ref = sitk.Resample(left_prior_dilated, first, identity, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    right_on_ref = sitk.Resample(right_prior_dilated, first, identity, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    raw_images = []   # MNI-grid float images, in the same order as subject_records
    centroids = []    # list of (left_xyz, right_xyz)

    for record in subject_records:
        subject_id = record["subject_id"]
        print(f"  Locating centroids for {subject_id}", flush=True)
        image = sitk.ReadImage(str(record["mni_nm"]), sitk.sitkFloat32)

        # Pass 1:Geometry-align to the shared MNI grid if needed
        same_geometry = (
            tuple(image.GetSize()) == tuple(first.GetSize())
            and np.allclose(image.GetSpacing(), first.GetSpacing(), atol=1e-6)
            and np.allclose(image.GetOrigin(), first.GetOrigin(), atol=1e-6)
            and np.allclose(image.GetDirection(), first.GetDirection(), atol=1e-6)
        )
        if not same_geometry:
            image = sitk.Resample(image, first, identity, sitk.sitkBSpline, 0.0, sitk.sitkFloat32)

        left_xyz, right_xyz = find_subject_centroids(image, left_on_ref, right_on_ref)
        raw_images.append(image)
        centroids.append((left_xyz, right_xyz))
        print(f"    L = {tuple(round(v, 2) for v in left_xyz)}, R = {tuple(round(v, 2) for v in right_xyz)}", flush=True)

    # Compute cohort-average centroids
    target_left, target_right = cohort_average_centroids(centroids)
    print(f"  Cohort average L = {tuple(round(v, 2) for v in target_left)}", flush=True)
    print(f"  Cohort average R = {tuple(round(v, 2) for v in target_right)}", flush=True)

    # Build the template mask once (on the shared MNI grid)
    template_mask = build_template_mask(trimmed_prior, first, dilation_mm=8.0)
    registration_mask = exclude_midline_band(template_mask, midline_exclusion_mm)
    mask_array = sitk.GetArrayFromImage(template_mask).astype(np.float32)


    # Compute bounding box of the mask and crop everything to it
    nz_vox = np.argwhere(mask_array > 0)
    pad_vox = max(10, int(round(10.0 / min(first.GetSpacing()))))
    z_lo = max(0, int(nz_vox[:, 0].min()) - pad_vox)
    z_hi = min(mask_array.shape[0], int(nz_vox[:, 0].max()) + pad_vox + 1)
    y_lo = max(0, int(nz_vox[:, 1].min()) - pad_vox)
    y_hi = min(mask_array.shape[1], int(nz_vox[:, 1].max()) + pad_vox + 1)
    x_lo = max(0, int(nz_vox[:, 2].min()) - pad_vox)
    x_hi = min(mask_array.shape[2], int(nz_vox[:, 2].max()) + pad_vox + 1)

    mask_array = mask_array[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    reference_cropped = spatial_crop(first, z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)


    # Pass 2: apply translation, mask, normalize each subject
    crops = []
    ids = []
    for record, image, (left_xyz, right_xyz) in zip(subject_records, raw_images, centroids):
        subject_id = record["subject_id"]
        translation = translation_to_target(left_xyz, right_xyz, target_left, target_right)
        aligned = sitk.Resample(image, first, translation, sitk.sitkBSpline, 0.0, sitk.sitkFloat32)

        aligned_arr = sitk.GetArrayFromImage(aligned).astype(np.float32)
        cropped_arr = aligned_arr[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        cropped = sitk.GetImageFromArray(cropped_arr)
        cropped.SetSpacing(aligned.GetSpacing())
        cropped.SetDirection(aligned.GetDirection())
        cropped.SetOrigin(aligned.TransformIndexToPhysicalPoint((int(x_lo), int(y_lo), int(z_lo))))
        normalized = robust_normalise(cropped, lower_pct, upper_pct)

        crops.append(normalized)
        ids.append(subject_id)

        if save_subject_crops:
            out_path = work_dir / "subject_crops" / f"{subject_id}_space-MNI_desc-centroidAligned_NM.nii.gz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(normalized, str(out_path))

    # Initial mean and QC
    initial = robust_normalise(average_sitk(crops, reference_cropped), lower_pct, upper_pct)
    initial_path = work_dir / "SN_template_pre_fireants_mean.nii.gz"
    qc_path = work_dir / "SN_crop_QC.png"
    centroid_log_path = work_dir / "subject_centroids.json"

    sitk.WriteImage(initial, str(initial_path))
    sitk.WriteImage(template_mask, str(work_dir / "SN_template_mask.nii.gz"))
    sitk.WriteImage(registration_mask, str(work_dir / "SN_registration_mask.nii.gz"))

    save_contact_sheet(crops, ids, initial, qc_path)

    with open(centroid_log_path, "w") as fh:
        json.dump({
            "target_left_mm": list(target_left),
            "target_right_mm": list(target_right),
            "subjects": [
                {
                    "subject_id": rec["subject_id"],
                    "left_mm": list(c[0]),
                    "right_mm": list(c[1]),
                }
                for rec, c in zip(subject_records, centroids)
            ],
        }, fh, indent=2)

    return crops, ids, reference_cropped, initial, qc_path

def template_sharpness_metrics(template_sitk, mask_bool):
    """Objective, SN-mask-restricted metrics (numpy-only): Laplacian variance
    (sharpness) and mean gradient magnitude (edge energy). No manual masks."""
    arr = sitk.GetArrayFromImage(template_sitk).astype(np.float32)
    m = mask_bool & np.isfinite(arr)
    if int(m.sum()) < 100:
        return {"lap_var": float("nan"), "grad_energy": float("nan")}
    gz, gy, gx = np.gradient(arr)
    lap = np.gradient(gz, axis=0) + np.gradient(gy, axis=1) + np.gradient(gx, axis=2)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    return {"lap_var": float(np.var(lap[m])), "grad_energy": float(np.mean(grad_mag[m]))}

def shape_update_diagnostics(disp_field, mask_bool):
    """disp_field: torch tensor [1,H,W,D,3], grid_sample displacement units (align_corners=True),
    channel order (x,y,z) per PyTorch's 3D grid_sample convention (channel k pairs with spatial
    axis 2-k). Returns the fraction of folding (non-positive Jacobian determinant) voxels,
    log|det J| mean/std, and mean displacement magnitude, all restricted to mask_bool.

    Jacobian is computed in voxel-index space (isotropic 0.5mm grid here, so sign and relative
    magnitude are meaningful for iteration-to-iteration comparison; this is a stability/folding
    diagnostic, not a calibrated physical Jacobian map)."""
    d = disp_field[0].detach().cpu().float().numpy()   # [H,W,D,3]
    H, W, D, _ = d.shape
    sx, sy, sz = (D - 1) / 2.0, (W - 1) / 2.0, (H - 1) / 2.0
    ux, uy, uz = d[..., 0], d[..., 1], d[..., 2]

    dux0, dux1, dux2 = np.gradient(ux)
    duy0, duy1, duy2 = np.gradient(uy)
    duz0, duz1, duz2 = np.gradient(uz)

    J = np.empty(d.shape[:3] + (3, 3), dtype=np.float32)
    J[..., 0, 0] = 1.0 + dux2 * sx; J[..., 0, 1] = dux1 * sx; J[..., 0, 2] = dux0 * sx
    J[..., 1, 0] = duy2 * sy;       J[..., 1, 1] = 1.0 + duy1 * sy; J[..., 1, 2] = duy0 * sy
    J[..., 2, 0] = duz2 * sz;       J[..., 2, 1] = duz1 * sz;       J[..., 2, 2] = 1.0 + duz0 * sz

    det = np.linalg.det(J)
    mag = np.sqrt((ux * sx) ** 2 + (uy * sy) ** 2 + (uz * sz) ** 2)
    m = mask_bool & np.isfinite(det)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_absdet = np.log(np.abs(det[m]) + 1e-8)
    return {
        "frac_folding": float((det[m] <= 0).mean()),
        "log_absdet_mean": float(np.mean(log_absdet)),
        "log_absdet_std": float(np.std(log_absdet)),
        "mean_disp_mag": float(np.mean(mag[m])),
    }

def append_qc_row(work_dir, row):
    import csv
    csv_path = work_dir / "qc_metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# =============================================================================
# FireANTs Template Build
# =============================================================================

def register_subject_to_template(template, moving, reg_mask, device, use_rigid):
    mask_img = Image(reg_mask, device=device, dtype=torch.float32)
    fixed_img = Image(template, device=device).concatenate(mask_img)
    moving_img = Image(moving, device=device).concatenate(Image(reg_mask, device=device, dtype=torch.float32))
    fixed_batch = BatchedImages([fixed_img])
    moving_batch = BatchedImages([moving_img])

    init_rigid = None
    if use_rigid:
        rigid = RigidRegistration(
            scales=scales, iterations=rigid_iterations,
            fixed_images=fixed_batch, moving_images=moving_batch,
            loss_type="masked_cc", cc_kernel_size=cc_kernel_size,
            optimizer="Adam", optimizer_lr=3e-2,
        )
        rigid.optimize()
        init_rigid = rigid.get_rigid_matrix()
        del rigid
        hard_gc()

    affine = AffineRegistration(
        scales=scales, iterations=affine_iterations,
        fixed_images=fixed_batch, moving_images=moving_batch,
        loss_type="masked_cc", cc_kernel_size=cc_kernel_size,
        optimizer="Adam", optimizer_lr=3e-2, init_rigid=init_rigid,
    )
    affine.optimize()
    init_affine = affine.get_affine_matrix()
    del affine
    hard_gc()

    syn = SyNRegistration(
        scales=scales, iterations=syn_iterations,
        fixed_images=fixed_batch, moving_images=moving_batch,
        loss_type="masked_cc", cc_kernel_size=cc_kernel_size,
        optimizer="Adam", optimizer_lr=syn_lr,
        smooth_grad_sigma=syn_smooth_grad_sigma,
        smooth_warp_sigma=syn_smooth_warp_sigma,
        init_affine=init_affine,
    )
    syn.optimize()

    with torch.no_grad():
        warped = syn.evaluate(fixed_batch, moving_batch)[:, :1].detach()
        warp_coords = syn.get_warped_coordinates(fixed_batch, moving_batch).detach()

    warped_cpu = warped.cpu()
    del fixed_batch, moving_batch, syn, warped
    hard_gc()
    return warped_cpu, warp_coords



def build_fireants_template(
        crops, ids, reference, initial, output_template, work_dir,
        outer_iterations, use_rigid, device, save_iterations,
        shape_avg=False, laplacian_passes=0, conv_tol=0.0, legacy_sharpen=False,
        shape_avg_step=0.25):
    """Iteratively register every subject to the template and rebuild it as a weighted average."""
    template = initial

    # SN/midbrain prior mask on the reference grid, used for QC metrics AND to
    # restrict the correlation weights to the structure we care about.
    full_mask = sitk.ReadImage(str(work_dir / "SN_template_mask.nii.gz"), sitk.sitkUInt8)
    mask_on_ref = sitk.Resample(full_mask, reference, sitk.Transform(3, sitk.sitkIdentity),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sn_mask_arr = sitk.GetArrayFromImage(mask_on_ref).astype(bool)

    # SN registration/analysis mask (midline-excluded) feeds registration loss, weighting, QC.
    full_reg_mask = sitk.ReadImage(str(work_dir / "SN_registration_mask.nii.gz"), sitk.sitkUInt8)
    reg_mask_on_ref = sitk.Resample(full_reg_mask, reference, sitk.Transform(3, sitk.sitkIdentity),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sn_reg_mask_arr = sitk.GetArrayFromImage(reg_mask_on_ref).astype(bool)

    lap_filter = LaplacianFilter(dims=3, device=device, itk_scale=True, learning_rate=0.5) if (laplacian_passes > 0 or shape_avg) else None


    prev_arr = None
    for outer in range(outer_iterations):
        print(f"\n{'=' * 68}", flush=True)
        print(f"FireANTs template iteration {outer + 1}/{outer_iterations}", flush=True)
        print(f"{'=' * 68}", flush=True)

        warped_tensors = []
        warp_coords_list = []
        for subject_id, moving in zip(ids, crops):
            print(f"  Registering {subject_id} to current SN template.", flush=True)
            warped, warp_coords = register_subject_to_template(
                template=template, moving=moving, reg_mask=reg_mask_on_ref, device=device, use_rigid=use_rigid)
            warped_tensors.append(warped)
            if shape_avg:
                warp_coords_list.append(warp_coords)
            else:
                del warp_coords

        # --- correlation weights, restricted to the SN mask, with outlier exclusion ---
        template_arr = sitk.GetArrayFromImage(template).astype(np.float32)
        weights = []
        for warped in warped_tensors:
            warped_arr = warped.squeeze().cpu().float().numpy()
            m = sn_reg_mask_arr & np.isfinite(warped_arr) & np.isfinite(template_arr)
            if m.sum() < 100 or warped_arr[m].std() < 1e-8 or template_arr[m].std() < 1e-8:
                weights.append(0.0)
            else:
                corr = float(np.corrcoef(warped_arr[m], template_arr[m])[0, 1])
                weights.append(max(corr, 0.0))
        weights = np.asarray(weights, dtype=np.float64)

        dropped = []
        # drop subjects whose SN fit is a robust-z low outlier (>2.5 MAD below median)
        pos = weights > 0
        if pos.sum() >= 4:
            med = np.median(weights[pos])
            mad = 1.4826 * np.median(np.abs(weights[pos] - med))
            if mad > 1e-8:
                z = (weights - med) / mad
                dropped = [ids[i] for i in np.where(z < -2.5)[0]]
                weights[z < -2.5] = 0.0
                if dropped:
                    print(f"  Outlier subjects down-weighted this iter: {dropped}", flush=True)

        if weights.sum() < 1e-8:
            weights = np.ones_like(weights)
        weights = (weights / weights.sum()).astype(np.float32)

        # --- weighted intensity average (on device for shape-avg / laplacian) ---
        weights_t = torch.from_numpy(weights).to(device)
        stacked = torch.stack(warped_tensors, dim=0).to(device)          # [N,1,1,H,W,D]
        avg_tensor = (stacked * weights_t.view(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)

        with torch.no_grad():
            vox_sd = stacked.squeeze(2).squeeze(1).std(dim=0).cpu().numpy()   # cross-subject SD
        del stacked

        # --- unbiased shape-averaging (toggle: --shape-avg) ---
        if shape_avg:
            template_batch = BatchedImages([Image(template, device=device)])
            avg_warp = torch.zeros_like(warp_coords_list[0])
            for w_i, coords_i in zip(weights, warp_coords_list):
                avg_warp = avg_warp + float(w_i) * coords_i
            del warp_coords_list

            # Damped shape update (ANTs-style gradient step): only apply a FRACTION of the
            # averaged correction each iteration, instead of the full inversion. Undamped,
            # this oscillates (confirmed in SN_TemplateBuild_v3_03_core).
            Nb, Hh, Ww, Dd, _ = avg_warp.shape
            theta = torch.eye(3, 4, device=avg_warp.device, dtype=avg_warp.dtype)[None].expand(Nb, -1, -1)
            identity_grid = F.affine_grid(theta, (Nb, 1, Hh, Ww, Dd), align_corners=True)
            avg_warp = identity_grid + shape_avg_step * (avg_warp - identity_grid)
            del identity_grid, theta

            inv_avg_warp = shape_averaging_invwarp(template_batch, avg_warp)

            fold_stats = shape_update_diagnostics(inv_avg_warp, sn_mask_arr)
            print(f"  [warp iter {outer + 1}] frac_folding={fold_stats['frac_folding']:.4g} "
                  f"log|detJ| mean/std={fold_stats['log_absdet_mean']:.4g}/{fold_stats['log_absdet_std']:.4g} "
                  f"mean_disp_mag={fold_stats['mean_disp_mag']:.4g}", flush=True)

            if lap_filter is not None:
                avg_tensor = lap_filter(avg_tensor, itk_scale=True, learning_rate=1.0)
            avg_tensor = fireants_interpolator(input=avg_tensor, affine=None,
                                               grid=inv_avg_warp, mode='bilinear', align_corners=True)
            del avg_warp, inv_avg_warp, template_batch
            hard_gc()
        else:
            fold_stats = {"frac_folding": None, "log_absdet_mean": None, "log_absdet_std": None, "mean_disp_mag": None}


        # --- principled Laplacian sharpening (toggle: --laplacian N) ---
        for _ in range(laplacian_passes):
            avg_tensor = lap_filter(avg_tensor)

        avg_array = avg_tensor.squeeze().detach().cpu().float().numpy().astype(np.float32)
        del avg_tensor
        hard_gc()
        template = sitk.GetImageFromArray(avg_array)
        template.CopyInformation(reference)
        template = robust_normalise(template, lower_pct, upper_pct)

        # --- objective QC (SN-mask-restricted; no manual masks) ---
        metrics = template_sharpness_metrics(template, sn_reg_mask_arr)
        cross_sd = float(np.mean(vox_sd[sn_reg_mask_arr]))
        print(f"  [QC iter {outer + 1}] lap_var={metrics['lap_var']:.5g} "
              f"grad_energy={metrics['grad_energy']:.5g} cross_subj_SD={cross_sd:.5g}", flush=True)

        append_qc_row(work_dir, {
            "iteration": outer + 1,
            "lap_var": metrics["lap_var"],
            "grad_energy": metrics["grad_energy"],
            "cross_subj_SD": cross_sd,
            "mean_weight": float(weights.mean()),
            "weight_std": float(weights.std()),
            "n_dropped_outliers": len(dropped),
            "frac_folding": fold_stats["frac_folding"],
            "log_absdet_mean": fold_stats["log_absdet_mean"],
            "log_absdet_std": fold_stats["log_absdet_std"],
            "mean_disp_mag": fold_stats["mean_disp_mag"],
        })

        if save_iterations:
            iter_path = work_dir / f"SN_FireANTs_Template_iter-{outer + 1:02d}.nii.gz"
            sitk.WriteImage(template, str(iter_path))
            print(f"  Saved iteration template: {iter_path}", flush=True)

        # --- convergence early-stop (only if --conv-tol > 0) ---
        new_arr = sitk.GetArrayFromImage(template).astype(np.float32)
        if prev_arr is not None and conv_tol > 0.0:
            change = float(np.mean(np.abs(new_arr[sn_mask_arr] - prev_arr[sn_mask_arr])))
            print(f"  [conv iter {outer + 1}] mean|delta| in SN mask = {change:.6f}", flush=True)
            if change < conv_tol:
                print(f"  Converged (delta<{conv_tol}); stopping early.", flush=True)
                break
        prev_arr = new_arr

    # --- finish: optional legacy unsharp mask (default OFF), then mask + normalize ---
    output_template.parent.mkdir(parents=True, exist_ok=True)
    if legacy_sharpen:
        blurred = sitk.SmoothingRecursiveGaussian(template, sigma=0.8)
        diff = sitk.Subtract(template, blurred)
        sharpened = sitk.Add(template, sitk.Multiply(diff, 1.5))
        template = sitk.Clamp(sharpened, sitk.sitkFloat32, 0.0, 3.0)

    dilated_mask = sitk.BinaryDilate(mask_on_ref, [7, 7, 7])
    mask_arr = sitk.GetArrayFromImage(dilated_mask).astype(np.float32)
    template_arr = sitk.GetArrayFromImage(template).astype(np.float32) * mask_arr
    template = sitk.GetImageFromArray(template_arr)
    template.CopyInformation(reference)
    template = robust_normalise(template, lower_pct, upper_pct)

    sitk.WriteImage(template, str(output_template))
    print(f"\nFinal SN FireANTs template saved to: {output_template}", flush=True)
    return template


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build a 0.5 mm substantia-nigra-centred NM template from native NM crops with FireANTs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory containing sub-*/anat/mean_NM_bias_corrected.nii.gz files.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-template", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rigid", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--exclude-subjects", default="", help="Comma-separated subject IDs to exclude")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--save-subject-crops", action="store_true")
    parser.add_argument("--save-iterations", action="store_true")
    parser.add_argument("--shape-avg", action="store_true",
                        help="Unbiased shape-averaging: average the warps and apply the inverse to the mean.")
    parser.add_argument("--laplacian", type=int, default=0,
                        help="Laplacian sharpening passes per iteration (0=off).")
    parser.add_argument("--conv-tol", type=float, default=0.0,
                        help="Early-stop when mean|delta| in SN mask < this value (0=off).")
    parser.add_argument("--legacy-sharpen", action="store_true",
                        help="Apply the old unsharp-mask finish (default off).")
    parser.add_argument("--shape-avg-step", type=float, default=0.25,
                    help="Damping fraction (0-1) applied to the averaged warp before inversion "
                            "when --shape-avg is set. Defaults to the module constant shape_avg_step "
                            "(0.25, matching ANTs' gradient-step convention).")


    parser.add_argument("--mni-prior", type=Path, default=None,
                    help=f"Path to MNI midbrain/brainstem prior. Defaults to {default_mni_prior}.")
    args = parser.parse_args()

    work_dir = args.work_dir or args.output_root / "SN_FireANTs_TemplateBuild"
    output_template = args.output_template or args.output_root / "SN_FireANTs_Template_MNI.nii.gz"
    mni_prior_path = args.mni_prior or default_mni_prior

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not mni_prior_path.exists():
        raise FileNotFoundError(f"MNI midbrain prior not found: {mni_prior_path}")

    subject_records = collect_subject_inputs(args.output_root)
    excluded_subjects = {part.strip() for part in args.exclude_subjects.split(",") if part.strip()}
    if excluded_subjects:
        found_subjects = {record["subject_id"] for record in subject_records}
        subject_records = [
            record for record in subject_records
            if record["subject_id"] not in excluded_subjects
        ]
        excluded_found = sorted(excluded_subjects & found_subjects)
        excluded_missing = sorted(excluded_subjects - found_subjects)
        print(f"Excluded subject(s): {', '.join(excluded_found) if excluded_found else 'none matched'}")
        if excluded_missing:
            print(f"Requested exclusions not found: {', '.join(excluded_missing)}")

    if not subject_records:
        raise FileNotFoundError(f"No NM_MNI inputs found under {args.output_root}")

    print(f"\nFound {len(subject_records)} subject(s).")
    print(f"Using device: {device}")
    print(f"MNI prior: {mni_prior_path}")
    print(f"Midbrain trim (RAS mm): x={midbrain_x_bounds_ras}, y={midbrain_y_bounds_ras}, z={midbrain_z_bounds_ras}")
    print(f"Work directory: {work_dir}")
    print(f"Registration stack: {'rigid -> SyN' if args.rigid else 'SyN'}")

    crops, ids, reference, initial, qc_path = prepare_subject_crops(
        subject_records=subject_records,
        work_dir=work_dir,
        mni_prior_path=mni_prior_path,
        save_subject_crops=args.save_subject_crops,
    )

    if args.prepare_only:
        print("\nPrepare-only mode complete.")
        print(f" Initial template mean: {work_dir / 'SN_template_pre_fireants_mean.nii.gz'}")
        print(f" QC contact sheet: {qc_path}")
        return
    
    resolved_shape_avg_step = args.shape_avg_step if args.shape_avg_step is not None else shape_avg_step


    build_fireants_template(
        crops=crops,
        ids=ids,
        reference=reference,
        initial=initial,
        output_template=output_template,
        work_dir=work_dir,
        outer_iterations=args.iterations,
        use_rigid=args.rigid,
        device=device,
        save_iterations=args.save_iterations,
        shape_avg=args.shape_avg,
        laplacian_passes=args.laplacian,
        conv_tol=args.conv_tol,
        legacy_sharpen=args.legacy_sharpen,
        shape_avg_step=resolved_shape_avg_step
    )

if __name__ == "__main__":
    main()
