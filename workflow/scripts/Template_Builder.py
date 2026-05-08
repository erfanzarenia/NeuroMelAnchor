#!/usr/bin/env python3
"""
Build a substantia-nigra-centred neuromelanin template with FireANTs.

Assumes Phase 1 has placed each subject's NM image in MNI space at:
    <output_root>/sub-*/anat/NM_MNI.nii

Workflow:
    1. Prepare fixed MNI-space crops from NM_MNI images.
    2. Resample crops to a shared 0.5 mm isotropic grid.
    3. Robust-normalise each crop.
    4. Save a QC contact sheet of all subject crops vs the initial mean.
    5. Iteratively register crops to the running mean with FireANTs.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from fireants.io.image import BatchedImages, Image
from fireants.registration.rigid import RigidRegistration
from fireants.registration.syn import SyNRegistration


# =============================================================================
# Config
# =============================================================================

default_output_root = Path("/scratch/Projects/BrainHack_Nexus/NM_BioGen_PP/Output")

x_bounds = (-28.0, 28.0)
y_bounds = (-44.0, 2.0)
z_bounds = (-34.0, 8.0)
target_spacing = (0.5, 0.5, 0.5)

scales = [4.0, 2.0, 1.0]
syn_iterations = [80, 40, 20]
syn_lr = 0.2
cc_kernel_size = 5
rigid_iterations = [100, 50, 25]

lower_pct = 1.0
upper_pct = 99.0


# =============================================================================
# Utility Functions
# =============================================================================

def hard_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def robust_normalise(image, lower, upper):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    values = arr[np.isfinite(arr)]
    values = values[np.abs(values) > 1e-6]
    if values.size == 0:
        raise ValueError("Cannot normalise an image with no finite non-zero voxels.")

    lo, hi = np.percentile(values, [lower, upper])
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def average_sitk(images, reference):
    if not images:
        raise ValueError("No images were provided for averaging.")

    arr = np.stack([
        sitk.GetArrayFromImage(image).astype(np.float32)
        for image in images
    ]).mean(axis=0)

    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.CopyInformation(reference)
    return out


# =============================================================================
# QC Functions
# =============================================================================

def crop_qc_records(crops, ids, template):
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

    corrs = np.array([np.nan if record["corr"] is None else record["corr"] for record in records], dtype=float)
    mads = np.array([np.nan if record["mad"] is None else record["mad"] for record in records], dtype=float)

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

    scores = -robust_zscores(corrs) + robust_zscores(mads)

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
            (f"score={score_text}", "red" if record["iqr_outlier"] else "black", 0.50),
            (f"corr={corr_text} mad={mad_text}", "black", 0.37),
        ]

        for text, color, y_position in label_lines:
            axes[row, 0].text(
                -0.08,
                y_position,
                text,
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=6,
                color=color,
            )

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

def prepare_subject_crops(image_paths, work_dir, save_subject_crops):
    if not image_paths:
        raise FileNotFoundError("No NM_MNI images were found.")

    work_dir.mkdir(parents=True, exist_ok=True)

    first = sitk.ReadImage(str(image_paths[0]), sitk.sitkFloat32)

    corners_lps = [
        (-float(x_value), -float(y_value), float(z_value))
        for x_value in sorted(x_bounds)
        for y_value in sorted(y_bounds)
        for z_value in sorted(z_bounds)
    ]
    continuous_indices = np.asarray([
        first.TransformPhysicalPointToContinuousIndex(corner)
        for corner in corners_lps
    ], dtype=float)

    start = np.floor(continuous_indices.min(axis=0)).astype(int)
    stop = np.ceil(continuous_indices.max(axis=0)).astype(int)
    image_size = np.asarray(first.GetSize(), dtype=int)
    start = np.maximum(start, 0)
    stop = np.minimum(stop, image_size - 1)
    size = stop - start + 1
    if np.any(size <= 0):
        raise ValueError(
            "SN crop box does not overlap the image. "
            f"x={x_bounds}, y={y_bounds}, z={z_bounds}"
        )

    crop_reference = sitk.RegionOfInterest(first, size=[int(v) for v in size], index=[int(v) for v in start])
    old_size = np.asarray(crop_reference.GetSize(), dtype=float)
    old_spacing = np.asarray(crop_reference.GetSpacing(), dtype=float)
    new_spacing = np.asarray(target_spacing, dtype=float)
    new_size = np.round((old_size - 1.0) * old_spacing / new_spacing).astype(int) + 1
    new_size = np.maximum(new_size, 1)

    highres_ref = sitk.Image([int(v) for v in new_size], sitk.sitkFloat32)
    highres_ref.SetSpacing(tuple(float(v) for v in new_spacing))
    highres_ref.SetOrigin(crop_reference.GetOrigin())
    highres_ref.SetDirection(crop_reference.GetDirection())

    identity = sitk.Transform(first.GetDimension(), sitk.sitkIdentity)
    crops = []
    ids = []

    for path in image_paths:
        subject_id = path.parents[1].name
        print(f"  Preparing {subject_id}: {path}", flush=True)

        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
        same_geometry = (
            tuple(image.GetSize()) == tuple(first.GetSize())
            and np.allclose(image.GetSpacing(), first.GetSpacing(), atol=1e-6)
            and np.allclose(image.GetOrigin(), first.GetOrigin(), atol=1e-6)
            and np.allclose(image.GetDirection(), first.GetDirection(), atol=1e-6)
        )
        if not same_geometry:
            print(f"    {subject_id}: resampling to first subject's MNI grid.", flush=True)
            image = sitk.Resample(
                image,
                first,
                identity,
                sitk.sitkBSpline,
                0.0,
                sitk.sitkFloat32,
            )

        crop = sitk.RegionOfInterest(
            image,
            size=[int(v) for v in size],
            index=[int(v) for v in start],
        )
        crop = sitk.Resample(
            crop,
            highres_ref,
            identity,
            sitk.sitkBSpline,
            0.0,
            sitk.sitkFloat32,
        )
        crop = robust_normalise(crop, lower_pct, upper_pct)

        crops.append(crop)
        ids.append(subject_id)

        if save_subject_crops:
            crop_path = work_dir / "subject_crops" / f"{subject_id}_space-MNI_desc-SNcrop0p5_NM.nii.gz"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(crop, str(crop_path))

    initial = robust_normalise(average_sitk(crops, highres_ref), lower_pct, upper_pct)
    initial_path = work_dir / "SN_crop_pre_fireants_mean.nii.gz"
    qc_path = work_dir / "SN_crop_QC.png"

    sitk.WriteImage(initial, str(initial_path))
    save_contact_sheet(crops, ids, initial, qc_path)

    return crops, ids, highres_ref, initial, qc_path


# =============================================================================
# FireANTs Template Build
# =============================================================================

def register_subject_to_template(template, moving, device, use_rigid):
    fixed_batch = BatchedImages([Image(template, device=device)])
    moving_batch = BatchedImages([Image(moving, device=device)])

    init_affine = None
    if use_rigid:
        rigid = RigidRegistration(
            scales=scales,
            iterations=rigid_iterations,
            fixed_images=fixed_batch,
            moving_images=moving_batch,
            loss_type="cc",
            cc_kernel_size=cc_kernel_size,
            optimizer="Adam",
            optimizer_lr=3e-2,
        )
        rigid.optimize()
        init_affine = rigid.get_rigid_matrix()
        hard_gc()

    syn = SyNRegistration(
        scales=scales,
        iterations=syn_iterations,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type="cc",
        cc_kernel_size=cc_kernel_size,
        optimizer="Adam",
        optimizer_lr=syn_lr,
        init_affine=init_affine,
    )
    syn.optimize()

    with torch.no_grad():
        warped = syn.evaluate(fixed_batch, moving_batch).detach().cpu()

    del fixed_batch, moving_batch, syn
    hard_gc()
    return warped


def build_fireants_template(crops, ids, reference, initial, output_template, work_dir, outer_iterations, use_rigid, device, save_iterations):
    template = initial

    for outer in range(outer_iterations):
        print(f"\n{'=' * 68}", flush=True)
        print(f"FireANTs template iteration {outer + 1}/{outer_iterations}", flush=True)
        print(f"{'=' * 68}", flush=True)

        warped_tensors = []
        for subject_id, moving in zip(ids, crops):
            print(f"  Registering {subject_id} to current SN template.", flush=True)
            warped = register_subject_to_template(
                template=template,
                moving=moving,
                device=device,
                use_rigid=use_rigid,
            )
            warped_tensors.append(warped)

        avg_tensor = torch.stack(warped_tensors, dim=0).mean(dim=0)
        avg_array = avg_tensor.squeeze().detach().cpu().float().numpy().astype(np.float32)
        template = sitk.GetImageFromArray(avg_array)
        template.CopyInformation(reference)
        template = robust_normalise(template, lower_pct, upper_pct)

        if save_iterations:
            iter_path = work_dir / f"SN_FireANTs_Template_iter-{outer + 1:02d}.nii.gz"
            sitk.WriteImage(template, str(iter_path))
            print(f"  Saved iteration template: {iter_path}", flush=True)

    output_template.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(template, str(output_template))
    print(f"\nFinal SN FireANTs template saved to: {output_template}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build a 0.5 mm substantia-nigra-centred NM template from MNI-space NM crops with FireANTs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Directory containing sub-*/anat/NM_MNI.nii files.",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-template", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--rigid", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--exclude-subjects", default="", help="Comma-separated subject IDs to exclude")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--save-subject-crops", action="store_true")
    parser.add_argument("--save-iterations", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir or (args.output_root / "SN_FireANTs_TemplateBuild")
    output_template = args.output_template or (args.output_root / "SN_FireANTs_Template_0p5mm.nii.gz")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    excluded_subjects = {part.strip() for part in args.exclude_subjects.split(",") if part.strip()}

    image_paths = sorted(args.output_root.glob("sub-*/anat/NM_MNI.nii"))
    if not image_paths:
        image_paths = sorted(args.output_root.glob("sub-*/anat/NM_MNI.nii.gz"))

    if excluded_subjects:
        found_subjects = {path.parents[1].name for path in image_paths}
        image_paths = [path for path in image_paths if path.parents[1].name not in excluded_subjects]

        excluded_found = sorted(excluded_subjects & found_subjects)
        excluded_missing = sorted(excluded_subjects - found_subjects)
        print(f"Excluded subject(s): {', '.join(excluded_found) if excluded_found else 'none matched'}")
        if excluded_missing:
            print(f"Requested exclusions not found: {', '.join(excluded_missing)}")

    if not image_paths:
        raise FileNotFoundError(f"No NM_MNI.nii files found under {args.output_root}")

    print(f"\nFound {len(image_paths)} NM_MNI image(s).")

    print(f"Using device: {device}")
    print(f"MNI RAS crop bounds: x={x_bounds}, y={y_bounds}, z={z_bounds}")
    print(f"Target spacing: {target_spacing}")
    print("Averaging method: mean")
    use_rigid = args.rigid
    print(f"Registration stack: {'rigid -> SyN' if use_rigid else 'SyN'}")
    print(f"Work directory: {work_dir}")

    crops, ids, reference, initial, qc_path = prepare_subject_crops(
        image_paths=image_paths,
        work_dir=work_dir,
        save_subject_crops=args.save_subject_crops,
    )

    if args.prepare_only:
        print("\nPrepare-only mode complete.")
        print(f"  Initial crop mean: {work_dir / 'SN_crop_pre_fireants_mean.nii.gz'}")
        print(f"  QC contact sheet: {qc_path}")
        return

    build_fireants_template(
        crops=crops,
        ids=ids,
        reference=reference,
        initial=initial,
        output_template=output_template,
        work_dir=work_dir,
        outer_iterations=args.iterations,
        use_rigid=use_rigid,
        device=device,
        save_iterations=args.save_iterations,
    )


if __name__ == "__main__":
    main()
