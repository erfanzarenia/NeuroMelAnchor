#!/usr/bin/env python
# =================================================================
# Mask Comparison Figures
# =================================================================
# Standalone QC visualisation for NeuroMelAnchor Stage 2 outputs.
#
# Reads finished Stage 2 products only - it never triggers registration
# or mask generation, so it is safe to rerun while refining plots.
#
# Produces:
#   1. Per-subject comparison figure - the NM seed against CIT168, NMS and
#      ProbCIT, cropped to the midbrain, three slices through the mask.
#   2. Group contact sheet - every subject on a few pages, sorted so the
#      cohort outliers come first.
#
# Example:
#   python workflow/figures/mask_comparison_figures.py \
#       --output_root /path/to/stage2_output_root
# =================================================================

# =================================================================
# Import Statements
# =================================================================
import csv
import json
import logging
import argparse

from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, binary_erosion, label, center_of_mass

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("MaskComparison")

# =================================================================
# Method Definitions
# =================================================================
# Colours are fixed per method so a contour means the same thing in every
# panel of every figure. Core-rim and Anchored are carried for the overlay
# panel only - core-rim is derived from the same CNR map as the NM seed, so
# it is not independent evidence and does not earn a pairwise panel.

METHOD_COLORS = {
    "NM":       "#EF476F",
    "CIT168":   "#118AB2",
    "ProbCIT":  "#F28E2B",
    "NMS":      "#8E6C8A",
    "Core-rim": "#2EC4B6",
    "Anchored": "#06D6A0",
}

PAIR_PANELS = [
    ("NM", "CIT168"),
    ("NM", "NMS"),
    ("NM", "ProbCIT"),
    ("CIT168", "NMS"),
]

OVERLAY_METHODS = ["NM", "CIT168", "ProbCIT", "NMS", "Core-rim"]
CONTACT_METHODS = ["NM", "CIT168", "NMS"]

# =================================================================
# Path Resolution
# =================================================================

def first_existing(*candidates):
    """Return the first path that exists, or None."""
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return None


def glob_first(directory, pattern):
    """Return the first match for a glob, or None."""
    matches = sorted(Path(directory).glob(pattern))
    return matches[0] if matches else None


def resolve_subject_paths(subject_dir, background="nm"):
    """Map a subject directory onto the Stage 2 files this script needs."""
    anat = Path(subject_dir) / "anat"
    cnr = Path(subject_dir) / "cnr"

    nm_image = first_existing(anat / "mean_NM_bias_corrected.nii.gz")
    cnr_map = first_existing(cnr / "Subject_CNR_Map_Smoothed.nii.gz")
    base = cnr_map if background == "cnr" else nm_image

    return {
        "base": base,
        "qc_json": first_existing(cnr / "SN_Mask_QC.json"),
        "masks": {
            "NM":       first_existing(anat / "Native_Tractography_Seed_SNc.nii.gz"),
            "CIT168":   first_existing(anat / "SNc_trans.nii.gz",
                                       anat / "Native_Atlas_Mask.nii.gz"),
            "ProbCIT":  glob_first(anat, "Native_ProbCIT_SNc_thr*.nii.gz"),
            "NMS":      glob_first(anat, "Native_NMS_SNc_thr*.nii.gz"),
            "Core-rim": first_existing(anat / "Native_CoreRim_NM_Mask.nii.gz"),
            "Anchored": first_existing(anat / "transform_Warped.nii.gz",
                                       anat / "CIT168_Anchored.nii.gz"),
        },
    }

# =================================================================
# Mask Loading and Cropping
# =================================================================

def load_subject(subject_dir, background="nm"):
    """Load the base image and every mask that shares its grid."""
    paths = resolve_subject_paths(subject_dir, background)
    if paths["base"] is None:
        return None

    base_img = nib.load(str(paths["base"]))
    base_data = base_img.get_fdata()
    reference_shape = base_data.shape[:3]
    if base_data.ndim > 3:
        base_data = base_data[..., 0]

    masks = {}
    for method, mask_path in paths["masks"].items():
        if mask_path is None:
            continue
        mask_data = nib.load(str(mask_path)).get_fdata()
        if mask_data.ndim > 3:
            mask_data = mask_data[..., 0]
        # A mask on a different grid cannot be drawn on this background.
        if mask_data.shape[:3] != reference_shape:
            log.warning(f"{Path(subject_dir).name}: {method} grid "
                        f"{mask_data.shape[:3]} != base {reference_shape}; skipped")
            continue
        masks[method] = mask_data > 0

    qc_data = {}
    if paths["qc_json"] is not None:
        with open(paths["qc_json"], "r") as qc_file:
            qc_data = json.load(qc_file)

    return {
        "subject_id": Path(subject_dir).name,
        "base": base_data,
        "affine": base_img.affine,
        "spacing": tuple(float(z) for z in base_img.header.get_zooms()[:3]),
        "masks": masks,
        "qc": qc_data,
    }


def crop_bounds(masks, shape, margin=20):
    """Bounding box of the mask union, padded, so the midbrain fills the panel."""
    present = [m for m in masks.values() if m is not None and m.any()]
    if not present:
        return (0, shape[0], 0, shape[1])

    union = np.logical_or.reduce(present)
    idx = np.argwhere(union)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    x0 = int(max(lo[0] - margin, 0))
    x1 = int(min(hi[0] + margin, shape[0]))
    y0 = int(max(lo[1] - margin, 0))
    y1 = int(min(hi[1] + margin, shape[1]))
    return (x0, x1, y0, y1)


def display_slices(masks, shape, n_slices=3):
    """Slices at even percentiles through the z-extent the masks actually occupy."""
    present = [m for m in masks.values() if m is not None and m.any()]
    if not present:
        return [shape[2] // 2] * n_slices

    union = np.logical_or.reduce(present)
    z_values = np.argwhere(union)[:, 2]
    percentiles = np.linspace(25, 75, n_slices)
    slices = [int(np.clip(np.percentile(z_values, p), 0, shape[2] - 1)) for p in percentiles]
    # Prefer distinct slices when the mask is thick enough to allow it.
    if len(set(slices)) < n_slices:
        lo, hi = int(z_values.min()), int(z_values.max())
        if hi > lo:
            slices = [int(round(v)) for v in np.linspace(lo, hi, n_slices)]
    return slices

# =================================================================
# Metrics
# =================================================================
# Computed on the cropped sub-volume: identical results to the full field of
# view, but the distance transforms run on ~1% of the voxels.

def dice_score(mask_a, mask_b):
    total = int(mask_a.sum()) + int(mask_b.sum())
    if total == 0:
        return float("nan")
    return float(2.0 * np.logical_and(mask_a, mask_b).sum() / total)


def hd95_mm(mask_a, mask_b, spacing):
    if not mask_a.any() or not mask_b.any():
        return float("nan")
    surface_a = mask_a & ~binary_erosion(mask_a)
    surface_b = mask_b & ~binary_erosion(mask_b)
    if not surface_a.any() or not surface_b.any():
        return float("nan")
    distances = np.concatenate([
        distance_transform_edt(~surface_b, sampling=spacing)[surface_a],
        distance_transform_edt(~surface_a, sampling=spacing)[surface_b],
    ])
    return float(np.percentile(distances, 95))


def lobe_stats(mask, affine):
    """Left/right component sizes for the SN, which is bilateral by anatomy.

    A healthy seed is two clusters of comparable size straddling the midline.
    The ratio is the informative asymmetry check - a single connected-component
    threshold flags every correct bilateral mask.
    """
    if mask is None or not mask.any():
        return None
    labelled, n_components = label(mask)
    if n_components == 0:
        return None

    sizes = np.array([int((labelled == i).sum()) for i in range(1, n_components + 1)])
    order = np.argsort(sizes)[::-1][:2]
    centroids = center_of_mass(mask, labelled, [int(i) + 1 for i in order])
    world_x = [float(nib.affines.apply_affine(affine, c)[0]) for c in np.atleast_2d(centroids)]

    kept = sizes[order]
    # Report the lower-x lobe first so left/right ordering is stable.
    if len(kept) == 2 and world_x[0] > world_x[1]:
        kept = kept[::-1]
        world_x = world_x[::-1]

    ratio = float(kept.max() / kept.min()) if len(kept) == 2 and kept.min() > 0 else float("nan")
    return {
        "n_components": int(n_components),
        "sizes": [int(v) for v in kept],
        "centroid_x_mm": [round(v, 1) for v in world_x],
        "asymmetry_ratio": ratio,
    }

# =================================================================
# Drawing Helpers
# =================================================================

def draw_panel(axis, base_data, masks, bounds, z, methods, spacing, title):
    """One cropped slice with the requested method contours drawn over it."""
    x0, x1, y0, y1 = bounds
    tile = base_data[x0:x1, y0:y1, z]

    finite = tile[np.isfinite(tile)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [2, 98])
        if vmin == vmax:
            vmin, vmax = float(finite.min()), float(finite.max()) or 1.0
    else:
        vmin, vmax = 0.0, 1.0

    axis.imshow(np.rot90(tile), cmap="gray", vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="equal")

    for method in methods:
        mask = masks.get(method)
        if mask is None:
            continue
        mask_tile = mask[x0:x1, y0:y1, z]
        if not mask_tile.any():
            continue
        axis.contour(np.rot90(mask_tile), levels=[0.5],
                     colors=[METHOD_COLORS[method]], linewidths=1.6)

    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def legend_handles(methods):
    return [Line2D([0], [0], color=METHOD_COLORS[m], lw=2.2, label=m) for m in methods]

# =================================================================
# Per-Subject Figure
# =================================================================

def build_subject_figure(subject, out_file, n_slices=3, margin=20, dpi=170):
    """NM seed against CIT168, NMS and ProbCIT - cropped, multi-slice, annotated."""
    masks = subject["masks"]
    base_data = subject["base"]
    spacing = subject["spacing"]
    shape = base_data.shape[:3]

    panel_masks = {m: masks.get(m) for m in OVERLAY_METHODS}
    bounds = crop_bounds(panel_masks, shape, margin)
    slices = display_slices(panel_masks, shape, n_slices)

    x0, x1, y0, y1 = bounds
    cropped = {m: (v[x0:x1, y0:y1, :] if v is not None else None)
               for m, v in masks.items()}

    # Metrics on the crop - same numbers, a fraction of the work.
    pair_metrics = {}
    for method_a, method_b in PAIR_PANELS:
        mask_a, mask_b = cropped.get(method_a), cropped.get(method_b)
        if mask_a is None or mask_b is None:
            pair_metrics[(method_a, method_b)] = None
            continue
        pair_metrics[(method_a, method_b)] = {
            "dice": dice_score(mask_a, mask_b),
            "hd95": hd95_mm(mask_a, mask_b, spacing),
        }

    n_cols = len(PAIR_PANELS) + 1
    fig, axes = plt.subplots(n_slices, n_cols,
                             figsize=(2.5 * n_cols, 2.7 * n_slices),
                             squeeze=False)

    for row, z in enumerate(slices):
        for col, (method_a, method_b) in enumerate(PAIR_PANELS):
            metrics = pair_metrics[(method_a, method_b)]
            if row == 0:
                if metrics is None:
                    title = f"{method_a} vs {method_b}\n(unavailable)\nz={z}"
                else:
                    title = (f"{method_a} vs {method_b}\n"
                             f"DICE {metrics['dice']:.2f} | HD95 {metrics['hd95']:.1f} mm\n"
                             f"z={z}")
            else:
                title = f"z={z}"
            draw_panel(axes[row][col], base_data, masks, bounds, z,
                       [method_a, method_b], spacing, title)

        overlay_title = f"All masks\nz={z}" if row == 0 else f"z={z}"
        draw_panel(axes[row][n_cols - 1], base_data, masks, bounds, z,
                   OVERLAY_METHODS, spacing, overlay_title)

    lobes = lobe_stats(masks.get("NM"), subject["affine"])
    if lobes and len(lobes["sizes"]) == 2:
        lobe_text = (f"NM seed lobes {lobes['sizes'][0]} / {lobes['sizes'][1]} vox "
                     f"(ratio {lobes['asymmetry_ratio']:.2f}, "
                     f"x = {lobes['centroid_x_mm'][0]} / {lobes['centroid_x_mm'][1]} mm)")
    elif lobes:
        lobe_text = f"NM seed: {lobes['n_components']} component(s), {lobes['sizes']} vox"
    else:
        lobe_text = "NM seed: empty"

    fig.suptitle(f"{subject['subject_id']} - SN mask comparison\n{lobe_text}", fontsize=11)
    fig.legend(handles=legend_handles([m for m in OVERLAY_METHODS if masks.get(m) is not None]),
               loc="lower center", ncol=len(OVERLAY_METHODS), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file, pair_metrics, lobes

# =================================================================
# Cohort Ranking
# =================================================================

def robust_z(values):
    """Median/MAD z-score - a few bad subjects cannot inflate the spread."""
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale == 0:
        scale = float(finite.std()) or 1.0
    return {"median": median, "scale": scale}


def rank_subjects(records):
    """Order subjects worst-first on agreement with CIT168, then asymmetry."""
    dice_values = [r["summary"].get("dice_nm_vs_cit") for r in records]
    stats = robust_z(dice_values)

    for record in records:
        dice_value = record["summary"].get("dice_nm_vs_cit")
        if stats and dice_value is not None and np.isfinite(dice_value):
            record["summary"]["dice_z"] = (dice_value - stats["median"]) / stats["scale"]
        else:
            record["summary"]["dice_z"] = float("nan")

    def sort_key(record):
        z = record["summary"].get("dice_z")
        return (0, z) if np.isfinite(z) else (1, 0.0)

    return sorted(records, key=sort_key)

# =================================================================
# Contact Sheet
# =================================================================

def build_contact_sheet(records, out_dir, rows_per_page=8, n_slices=3,
                        margin=20, background="nm", dpi=150):
    """Every subject on a few pages, worst-first, so triage is one scroll."""
    out_files = []
    pages = [records[i:i + rows_per_page] for i in range(0, len(records), rows_per_page)]

    for page_index, page in enumerate(pages, start=1):
        n_cols = n_slices + 1
        fig, axes = plt.subplots(len(page), n_cols,
                                 figsize=(2.3 * n_cols, 2.5 * len(page)),
                                 squeeze=False)

        for row, record in enumerate(page):
            subject = load_subject(record["dir"], background)
            if subject is None:
                for col in range(n_cols):
                    axes[row][col].axis("off")
                continue

            masks = subject["masks"]
            shape = subject["base"].shape[:3]
            panel_masks = {m: masks.get(m) for m in CONTACT_METHODS}
            bounds = crop_bounds(panel_masks, shape, margin)
            slices = display_slices(panel_masks, shape, n_slices)

            for col, z in enumerate(slices):
                title = f"{subject['subject_id']}  z={z}" if col == 0 else f"z={z}"
                draw_panel(axes[row][col], subject["base"], masks, bounds, z,
                           CONTACT_METHODS, subject["spacing"], title)

            summary = record["summary"]
            axis = axes[row][n_cols - 1]
            axis.axis("off")
            z_score = summary.get("dice_z", float("nan"))
            flag = "  <-- OUTLIER" if np.isfinite(z_score) and abs(z_score) > 3 else ""
            ratio = summary.get("asymmetry_ratio")
            ratio_text = f"{ratio:.2f}" if ratio is not None and np.isfinite(ratio) else "n/a"
            lines = [
                f"{subject['subject_id']}{flag}",
                "",
                f"DICE NM/CIT168 : {summary.get('dice_nm_vs_cit', float('nan')):.3f}",
                f"   robust z    : {z_score:+.1f}" if np.isfinite(z_score) else "   robust z    : n/a",
                f"DICE NM/NMS    : {summary.get('dice_nm_vs_nms', float('nan')):.3f}",
                f"NM voxels      : {summary.get('nm_voxels', 0)}",
                f"lobe ratio     : {ratio_text}",
            ]
            axis.text(0.0, 0.5, "\n".join(lines), fontsize=8.5, family="monospace",
                      va="center", ha="left", transform=axis.transAxes)

        fig.suptitle(f"SN mask contact sheet - page {page_index} of {len(pages)} "
                     f"(worst agreement first)", fontsize=12)
        fig.legend(handles=legend_handles(CONTACT_METHODS), loc="lower center",
                   ncol=len(CONTACT_METHODS), fontsize=9, frameon=False)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        out_file = Path(out_dir) / f"Contact_Sheet_page-{page_index:02d}.png"
        fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out_files.append(out_file)
        log.info(f"Contact sheet page saved: {out_file}")

    return out_files

# =================================================================
# Main
# =================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-subject and cohort SN mask comparison figures from Stage 2 outputs")
    parser.add_argument("--output_root", required=True, type=Path,
                        help="Stage 2 output directory containing sub-* folders")
    parser.add_argument("--out_dir", default=None, type=Path,
                        help="Where figures are written (default: <output_root>/mask_comparison)")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Limit to these subject IDs (default: all sub-* found)")
    parser.add_argument("--background", choices=["nm", "cnr"], default="nm",
                        help="Panel background: nm = bias-corrected NM (default), cnr = CNR map")
    parser.add_argument("--n_slices", default=3, type=int,
                        help="Slices through the mask per subject (default: 3)")
    parser.add_argument("--margin", default=20, type=int,
                        help="Voxels of padding around the mask bounding box (default: 20)")
    parser.add_argument("--rows_per_page", default=8, type=int,
                        help="Subjects per contact sheet page (default: 8)")
    parser.add_argument("--skip_subject_figures", action="store_true",
                        help="Build only the contact sheet and the CSV")
    parser.add_argument("--skip_contact_sheet", action="store_true",
                        help="Build only the per-subject figures")
    parser.add_argument("--dpi", default=170, type=int, help="Figure DPI (default: 170)")
    args = parser.parse_args()

    output_root = args.output_root
    if not output_root.exists():
        raise FileNotFoundError(f"--output_root does not exist: {output_root}")

    out_dir = args.out_dir or (output_root / "mask_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(p for p in output_root.glob("sub-*") if p.is_dir())
    if args.subjects:
        wanted = set(args.subjects)
        subject_dirs = [p for p in subject_dirs if p.name in wanted]
    if not subject_dirs:
        raise RuntimeError(f"No sub-* directories found in {output_root}")

    log.info(f"{'=' * 60}")
    log.info(f"Subjects        : {len(subject_dirs)}")
    log.info(f"Background      : {args.background}")
    log.info(f"Output directory: {out_dir}")
    log.info(f"{'=' * 60}")

    records = []
    for subject_dir in subject_dirs:
        subject = load_subject(subject_dir, args.background)
        if subject is None:
            log.warning(f"{subject_dir.name}: no usable background image; skipped")
            continue
        if "NM" not in subject["masks"]:
            log.warning(f"{subject_dir.name}: no NM seed mask; skipped")
            continue

        summary = {"nm_voxels": int(subject["masks"]["NM"].sum())}

        if args.skip_subject_figures:
            # Still need the pair metrics for ranking, so compute on the crop.
            shape = subject["base"].shape[:3]
            bounds = crop_bounds({m: subject["masks"].get(m) for m in OVERLAY_METHODS},
                                 shape, args.margin)
            x0, x1, y0, y1 = bounds
            cropped = {m: (v[x0:x1, y0:y1, :] if v is not None else None)
                       for m, v in subject["masks"].items()}
            pair_metrics = {}
            for pair in PAIR_PANELS:
                mask_a, mask_b = cropped.get(pair[0]), cropped.get(pair[1])
                pair_metrics[pair] = (None if mask_a is None or mask_b is None
                                      else {"dice": dice_score(mask_a, mask_b),
                                            "hd95": hd95_mm(mask_a, mask_b, subject["spacing"])})
            lobes = lobe_stats(subject["masks"].get("NM"), subject["affine"])
        else:
            figure_path = out_dir / "subjects" / f"{subject['subject_id']}_Mask_Comparison.png"
            figure_path, pair_metrics, lobes = build_subject_figure(
                subject, figure_path, args.n_slices, args.margin, args.dpi)
            log.info(f"Subject figure saved: {figure_path}")

        for (method_a, method_b), metrics in pair_metrics.items():
            if metrics is None:
                continue
            key = f"{method_a}_vs_{method_b}".lower().replace("168", "")
            summary[f"dice_{key}"] = metrics["dice"]
            summary[f"hd95_{key}_mm"] = metrics["hd95"]

        summary["dice_nm_vs_cit"] = summary.get("dice_nm_vs_cit")
        summary["dice_nm_vs_nms"] = summary.get("dice_nm_vs_nms")
        summary["asymmetry_ratio"] = lobes["asymmetry_ratio"] if lobes else float("nan")
        summary["n_components"] = lobes["n_components"] if lobes else 0

        records.append({"subject_id": subject["subject_id"], "dir": subject_dir,
                        "summary": summary})

    if not records:
        raise RuntimeError("No subjects produced usable masks.")

    records = rank_subjects(records)

    csv_path = out_dir / "Mask_Comparison_Summary.csv"
    fieldnames = ["subject_id"] + sorted({k for r in records for k in r["summary"]})
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"subject_id": record["subject_id"]}
            for key, value in record["summary"].items():
                row[key] = "" if isinstance(value, float) and not np.isfinite(value) else value
            writer.writerow(row)
    log.info(f"Summary CSV saved: {csv_path}")

    if not args.skip_contact_sheet:
        build_contact_sheet(records, out_dir, args.rows_per_page, args.n_slices,
                            args.margin, args.background, dpi=max(120, args.dpi - 30))

    log.info(f"{'=' * 60}")
    log.info("Cohort outliers (|robust z| > 3 on DICE NM vs CIT168):")
    outliers = [r for r in records if np.isfinite(r["summary"].get("dice_z", float("nan")))
                and abs(r["summary"]["dice_z"]) > 3]
    if outliers:
        for record in outliers:
            log.info(f"   {record['subject_id']:<12} dice "
                     f"{record['summary']['dice_nm_vs_cit']:.3f}  "
                     f"z={record['summary']['dice_z']:+.1f}")
    else:
        log.info("   none")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
