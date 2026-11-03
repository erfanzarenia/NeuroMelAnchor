"""
NeuroMelAnchor QC reports, built only from existing Stage 2 outputs.

Writes, per subject, sub-*/qc/Stage2_Output_Manifest.json and a QC dashboard, and, for the
group, <output_root>/group_qc/: pairwise and component summaries, method comparisons, and
with manual masks the accuracy figure, contours, the NM-blind control and a statistics table.
It never registers images or builds masks, so it is safe to rerun.

The pipeline calls this after Stage 2 by default (skip with --skip_group_qc), and
--stage group_qc runs only this. It can also be run on its own:

    python workflow/qc/group_qc.py --output_root /path/to/output \
        --manual_dir /path/to/manual --seed_cnr_percentile 55

Pass the --seed_cnr_percentile and atlas thresholds the run used, since the NM-blind
control and the output names depend on them.
"""
import csv
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wilcoxon
from scipy.ndimage import distance_transform_edt, binary_erosion
from matplotlib.patheffects import withStroke

log = logging.getLogger("NeuroMelAnchor")


# =================================================================
# Stage 2 QC Reports
# =================================================================


def build_stage2_output_manifest(subject_id, output_path, manual_dir=None,
                                 prob_atlas_threshold=0.25, nms_atlas_threshold=0.25):
    """Write a plain audit file describing this subject's Stage 2 outputs."""
    subject_dir = Path(output_path) / subject_id
    anat_dir = subject_dir / "anat"
    cnr_dir = subject_dir / "cnr"
    qc_dir = subject_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    prob_threshold_label = int(round(float(prob_atlas_threshold) * 100))
    nms_threshold_label = int(round(float(nms_atlas_threshold) * 100))

    reference_nm_image_path = anat_dir / "mean_NM_bias_corrected.nii.gz"
    if not reference_nm_image_path.exists():
        reference_nm_image_path = anat_dir / "mean_NM_bias_corrected.nii"
    if not reference_nm_image_path.exists():
        reference_nm_image_path = None

    nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii.gz"
    if not nm_seed_mask.exists():
        nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii"
    if not nm_seed_mask.exists():
        nm_seed_mask = None

    core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii.gz"
    if not core_rim_mask.exists():
        core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii"
    if not core_rim_mask.exists():
        core_rim_mask = None

    atlas_mask = anat_dir / "SNc_trans.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc_trans.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = None

    prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii"
    if not prob_cit_mask.exists():
        prob_matches = sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii.gz"))
        prob_matches += sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii"))
        prob_cit_mask = prob_matches[0] if prob_matches else None

    nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii"
    if not nms_mask.exists():
        nms_matches = sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii.gz"))
        nms_matches += sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii"))
        nms_mask = nms_matches[0] if nms_matches else None

    # DataSink keeps the ANTs default name for the anchored atlas; CIT168_Anchored is only a fallback.

    anchored_mask = anat_dir / "transform_Warped.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "transform_Warped.nii"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "CIT168_Anchored.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = None

    manual_mask = None
    if manual_dir is not None:
        manual_anat_dir = Path(manual_dir) / subject_id / "anat"
        manual_matches = sorted(manual_anat_dir.glob(f"{subject_id}_space-NM_label-SN_desc-manual_mask.nii*"))
        manual_mask = manual_matches[0] if manual_matches else None

    mask_paths = {
        "nm_seed": nm_seed_mask,
        "core_rim_nm": core_rim_mask,
        "atlas": atlas_mask,
        "prob_cit": prob_cit_mask,
        "nms": nms_mask,
        "anchored_atlas": anchored_mask,
        "manual": manual_mask,
    }

    manifest_warnings = []
    reference_shape = None
    reference_affine = None

    if reference_nm_image_path is None:
        manifest_warnings.append("Missing native NM reference image: mean_NM_bias_corrected.nii[.gz]")
    else:
        try:
            reference_img = nib.load(str(reference_nm_image_path))
            reference_shape = tuple(int(v) for v in reference_img.shape)
            reference_affine = reference_img.affine
        except Exception as exc:
            manifest_warnings.append(f"Could not load native NM reference image: {exc}")

    mask_records = {}
    for mask_name, mask_path in mask_paths.items():
        mask_exists = mask_path is not None and Path(mask_path).exists()
        mask_shape = None
        mask_voxel_count = None
        shape_matches_reference = None
        affine_matches_reference = None
        load_error = None

        if mask_exists:
            try:
                mask_img = nib.load(str(mask_path))
                mask_data = mask_img.get_fdata()
                mask_shape = [int(v) for v in mask_img.shape]
                mask_voxel_count = int(np.count_nonzero(mask_data))

                if reference_shape is not None:
                    shape_matches_reference = bool(tuple(mask_img.shape) == reference_shape)
                    if not shape_matches_reference:
                        manifest_warnings.append(f"{mask_name} shape does not match native NM reference")

                if reference_affine is not None:
                    affine_matches_reference = bool(np.allclose(mask_img.affine, reference_affine, atol=1e-3))
                    if not affine_matches_reference:
                        manifest_warnings.append(f"{mask_name} affine does not match native NM reference")
            except Exception as exc:
                load_error = str(exc)
                manifest_warnings.append(f"Could not inspect {mask_name}: {exc}")

        mask_records[mask_name] = {
            "path": str(mask_path) if mask_path is not None else None,
            "exists": mask_exists,
            "shape": mask_shape,
            "voxel_count": mask_voxel_count,
            "shape_matches_reference": shape_matches_reference,
            "affine_matches_reference": affine_matches_reference,
            "error": load_error,
        }

    required_mask_names = ["nm_seed", "atlas"]
    missing_required_masks = [name for name in required_mask_names if not mask_records[name]["exists"]]
    if manual_dir is not None and not mask_records["manual"]["exists"]:
        manifest_warnings.append("Manual mask directory was provided, but this subject has no matching manual SN mask")

    manifest = {}
    manifest["subject_id"] = subject_id
    manifest["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest["status"] = "complete" if not missing_required_masks else "incomplete"
    manifest["native_nm_reference"] = {
        "path": str(reference_nm_image_path) if reference_nm_image_path is not None else None,
        "exists": reference_nm_image_path is not None and reference_nm_image_path.exists(),
        "shape": list(reference_shape) if reference_shape is not None else None,
    }
    manifest["stage2_qc_json"] = {
        "path": str(cnr_dir / "SN_Mask_QC.json"),
        "exists": (cnr_dir / "SN_Mask_QC.json").exists(),
    }
    manifest["masks"] = mask_records
    manifest["missing_required_masks"] = missing_required_masks
    manifest["warnings"] = manifest_warnings

    manifest_path = qc_dir / "Stage2_Output_Manifest.json"
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    return manifest_path


def build_subject_qc_dashboard(subject_id, output_path, manual_dir=None,
                               prob_atlas_threshold=0.25, nms_atlas_threshold=0.25):
    """Build the per-subject dashboard from existing Stage 2 files."""
    subject_dir = Path(output_path) / subject_id
    anat_dir = subject_dir / "anat"
    cnr_dir = subject_dir / "cnr"
    qc_dir = subject_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    prob_threshold_label = int(round(float(prob_atlas_threshold) * 100))
    nms_threshold_label = int(round(float(nms_atlas_threshold) * 100))

    base_img_path = anat_dir / "mean_NM_bias_corrected.nii.gz"
    if not base_img_path.exists():
        base_img_path = anat_dir / "mean_NM_bias_corrected.nii"
    if not base_img_path.exists():
        base_img_path = anat_dir / "NM_MNI.nii"

    qc_json = cnr_dir / "SN_Mask_QC.json"
    if not base_img_path.exists() or not qc_json.exists():
        return None

    manual_mask = None
    if manual_dir is not None:
        manual_anat_dir = Path(manual_dir) / subject_id / "anat"
        manual_matches = sorted(manual_anat_dir.glob(f"{subject_id}_space-NM_label-SN_desc-manual_mask.nii*"))
        manual_mask = manual_matches[0] if manual_matches else None

    nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii.gz"
    if not nm_seed_mask.exists():
        nm_seed_mask = anat_dir / "Native_Tractography_Seed_SNc.nii"
    if not nm_seed_mask.exists():
        nm_seed_mask = None

    core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii.gz"
    if not core_rim_mask.exists():
        core_rim_mask = anat_dir / "Native_CoreRim_NM_Mask.nii"
    if not core_rim_mask.exists():
        core_rim_mask = None

    atlas_mask = anat_dir / "SNc_trans.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc_trans.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "Native_Atlas_Mask.nii"
    if not atlas_mask.exists():
        atlas_mask = anat_dir / "SNc.nii.gz"
    if not atlas_mask.exists():
        atlas_mask = None

    prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / f"Native_ProbCIT_SNc_thr{prob_threshold_label:03d}.nii"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii.gz"
    if not prob_cit_mask.exists():
        prob_cit_mask = anat_dir / "ProbCIT_SNc_Mask.nii"
    if not prob_cit_mask.exists():
        prob_matches = sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii.gz"))
        prob_matches += sorted(anat_dir.glob("Native_ProbCIT_SNc_thr*.nii"))
        prob_cit_mask = prob_matches[0] if prob_matches else None

    nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / f"Native_NMS_SNc_thr{nms_threshold_label:03d}.nii"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii.gz"
    if not nms_mask.exists():
        nms_mask = anat_dir / "NMS_SNc_Mask.nii"
    if not nms_mask.exists():
        nms_matches = sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii.gz"))
        nms_matches += sorted(anat_dir.glob("Native_NMS_SNc_thr*.nii"))
        nms_mask = nms_matches[0] if nms_matches else None

    # DataSink keeps the ANTs default name for the anchored atlas; CIT168_Anchored is only a fallback.

    anchored_mask = anat_dir / "transform_Warped.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "transform_Warped.nii"
    if not anchored_mask.exists():
        anchored_mask = anat_dir / "CIT168_Anchored.nii.gz"
    if not anchored_mask.exists():
        anchored_mask = None

    display_mask_paths = {
        "Manual": manual_mask,
        "NM": nm_seed_mask,
        "Core-rim": core_rim_mask,
        "Atlas": atlas_mask,
        "ProbCIT": prob_cit_mask,
        "NMS": nms_mask,
        "Anchored": anchored_mask,
    }

    base_img = nib.load(str(base_img_path))
    base_data = base_img.get_fdata()
    reference_shape = base_data.shape

    masks = {}
    for label, mask_path in display_mask_paths.items():
        masks[label] = None
        if mask_path is None:
            continue

        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata().astype(bool)
        if mask_data.shape == reference_shape:
            masks[label] = mask_data

    valid_masks = [mask for mask in masks.values() if mask is not None and mask.any()]
    if valid_masks:
        mask_union = np.logical_or.reduce(valid_masks)
        z_values = np.argwhere(mask_union)[:, 2]
        display_slice = int(np.clip(np.median(z_values), 0, reference_shape[2] - 1))
    else:
        display_slice = reference_shape[2] // 2

    with open(qc_json, "r") as qc_file:
        qc_data = json.load(qc_file)

    color_map = {
        "Manual": "#FFD166",
        "NM": "#EF476F",
        "Core-rim": "#2EC4B6",
        "Atlas": "#118AB2",
        "ProbCIT": "#F28E2B",
        "NMS": "#8E6C8A",
        "Anchored": "#06D6A0",
    }

    finite_base = base_data[np.isfinite(base_data)]
    if finite_base.size:
        vmin, vmax = np.percentile(finite_base, [2, 98])
    else:
        vmin, vmax = 0, 1

    def draw_mask_panel(ax, panel_title, panel_labels):
        ax.imshow(np.rot90(base_data[:, :, display_slice]), cmap="gray", vmin=vmin, vmax=vmax)
        for label in panel_labels:
            mask = masks.get(label)
            if mask is None or not mask[:, :, display_slice].any():
                continue
            ax.contour(
                np.rot90(mask[:, :, display_slice]),
                levels=[0.5],
                colors=[color_map[label]],
                linewidths=2.0 if label == "Manual" else 1.5,
                linestyles="--" if label == "Manual" else "-",
            )
        ax.set_title(f"{panel_title} | z={display_slice}", fontsize=10)
        ax.axis("off")

    comparison_panels = [
        ("Manual vs NM", ["NM", "Manual"]),
        ("Manual vs Core-rim", ["Core-rim", "Manual"]),
        ("Manual vs Atlas", ["Atlas", "Manual"]),
        ("Manual vs ProbCIT", ["ProbCIT", "Manual"]),
        ("Manual vs NMS", ["NMS", "Manual"]),
        ("Manual vs Anchored", ["Anchored", "Manual"]),
        ("All Masks Overview", ["Atlas", "ProbCIT", "NMS", "Anchored", "NM", "Core-rim", "Manual"]),
    ]

    manual_dice_metrics = [
        ("NM", "dice_cnr_vs_manual"),
        ("Core-rim", "dice_core_rim_nm_vs_manual"),
        ("Atlas", "dice_atlas_vs_manual"),
        ("ProbCIT", "dice_prob_atlas_vs_manual"),
        ("NMS", "dice_nms_atlas_vs_manual"),
        ("Anchored", "dice_anchored_vs_manual"),
    ]
    fallback_dice_metrics = [
        ("NM vs Atlas", "dice_cnr_vs_atlas"),
        ("NM vs Core-rim", "dice_cnr_vs_core_rim_nm"),
        ("NM vs ProbCIT", "dice_cnr_vs_prob_atlas"),
        ("NM vs NMS", "dice_cnr_vs_nms_atlas"),
        ("NM vs Anchored", "dice_cnr_vs_anchored_atlas"),
    ]

    manual_hd95_metrics = [
        ("NM", "hd95_cnr_vs_manual_mm"),
        ("Core-rim", "hd95_core_rim_nm_vs_manual_mm"),
        ("Atlas", "hd95_atlas_vs_manual_mm"),
        ("ProbCIT", "hd95_prob_atlas_vs_manual_mm"),
        ("NMS", "hd95_nms_atlas_vs_manual_mm"),
        ("Anchored", "hd95_anchored_vs_manual_mm"),
    ]
    fallback_hd95_metrics = [
        ("NM vs Atlas", "hd95_cnr_vs_atlas_mm"),
        ("NM vs Core-rim", "hd95_cnr_vs_core_rim_nm_mm"),
        ("NM vs ProbCIT", "hd95_cnr_vs_prob_atlas_mm"),
        ("NM vs NMS", "hd95_cnr_vs_nms_atlas_mm"),
        ("NM vs Anchored", "hd95_cnr_vs_anchored_atlas_mm"),
    ]

    dice_metrics = manual_dice_metrics
    if all(qc_data.get(metric_key) is None for _, metric_key in manual_dice_metrics):
        dice_metrics = fallback_dice_metrics

    hd95_metrics = manual_hd95_metrics
    if all(qc_data.get(metric_key) is None for _, metric_key in manual_hd95_metrics):
        hd95_metrics = fallback_hd95_metrics

    dice_labels = [label for label, _ in dice_metrics]
    dice_values = []
    for _, metric_key in dice_metrics:
        try:
            value = float(qc_data.get(metric_key))
        except (TypeError, ValueError):
            value = np.nan
        dice_values.append(value if np.isfinite(value) else np.nan)

    hd95_labels = [label for label, _ in hd95_metrics]
    hd95_values = []
    for _, metric_key in hd95_metrics:
        try:
            value = float(qc_data.get(metric_key))
        except (TypeError, ValueError):
            value = np.nan
        hd95_values.append(value if np.isfinite(value) else np.nan)

    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 7, height_ratios=[1.4, 1.0])

    for panel_idx, (panel_title, panel_labels) in enumerate(comparison_panels):
        draw_mask_panel(fig.add_subplot(grid[0, panel_idx]), panel_title, panel_labels)

    ax_dice = fig.add_subplot(grid[1, 0:3])
    dice_colors = [color_map.get(label, "#888888") for label in dice_labels]
    ax_dice.bar(dice_labels, dice_values, color=dice_colors)
    ax_dice.set_ylim(0, 1)
    ax_dice.set_ylabel("DICE")
    ax_dice.set_title("Overlap with manual segmentation" if dice_labels[0] == "NM" else "Overlap QC")
    ax_dice.axhline(0.5, color="firebrick", lw=0.8, ls="--", alpha=0.6)
    ax_dice.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(dice_values):
        if np.isfinite(value):
            ax_dice.text(idx, value + 0.03, f"{value:.2f}", ha="center", fontsize=9)

    ax_hd95 = fig.add_subplot(grid[1, 3:7])
    hd95_colors = [color_map.get(label, "#888888") for label in hd95_labels]
    ax_hd95.bar(hd95_labels, hd95_values, color=hd95_colors)
    ax_hd95.set_ylabel("HD95 (mm)")
    ax_hd95.set_title("Boundary distance with manual segmentation" if hd95_labels[0] == "NM" else "Boundary QC")
    ax_hd95.axhline(5.0, color="firebrick", lw=0.8, ls="--", alpha=0.6)
    ax_hd95.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(hd95_values):
        if np.isfinite(value):
            ax_hd95.text(idx, value + 0.2, f"{value:.1f}", ha="center", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], color=color_map["Manual"], lw=2, ls="--", label="Manual"),
        plt.Line2D([0], [0], color=color_map["NM"], lw=2, ls="-", label="NM"),
        plt.Line2D([0], [0], color=color_map["Core-rim"], lw=2, ls="-", label="Core-rim"),
        plt.Line2D([0], [0], color=color_map["Atlas"], lw=2, ls="-", label="Atlas"),
        plt.Line2D([0], [0], color=color_map["ProbCIT"], lw=2, ls="-", label="ProbCIT"),
        plt.Line2D([0], [0], color=color_map["NMS"], lw=2, ls="-", label="NMS"),
        plt.Line2D([0], [0], color=color_map["Anchored"], lw=2, ls="-", label="Anchored"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=7, frameon=False)
    fig.suptitle(f"{subject_id} Subject QC Dashboard", fontsize=14)

    out_png = qc_dir / "Subject_QC_Dashboard.png"
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_png


GROUP_QC_COMPARISONS = [
    {
        "pair": "cnr_vs_atlas",
        "label": "CNR vs Atlas",
        "dice": "dice_cnr_vs_atlas",
        "hd95": "hd95_cnr_vs_atlas_mm",
        "color": "#4C72B0",
        "marker": "o",
    },
    {
        "pair": "cnr_vs_core_rim_nm",
        "label": "NM vs Core-rim",
        "dice": "dice_cnr_vs_core_rim_nm",
        "hd95": "hd95_cnr_vs_core_rim_nm_mm",
        "color": "#2EC4B6",
        "marker": "*",
    },
    {
        "pair": "cnr_vs_prob_atlas",
        "label": "CNR vs ProbCIT",
        "dice": "dice_cnr_vs_prob_atlas",
        "hd95": "hd95_cnr_vs_prob_atlas_mm",
        "color": "#F28E2B",
        "marker": "P",
    },
    {
        "pair": "cnr_vs_nms_atlas",
        "label": "CNR vs NMS",
        "dice": "dice_cnr_vs_nms_atlas",
        "hd95": "hd95_cnr_vs_nms_atlas_mm",
        "color": "#8E6C8A",
        "marker": "h",
    },
    {
        "pair": "cnr_vs_anchored_atlas",
        "label": "CNR vs Anchored",
        "dice": "dice_cnr_vs_anchored_atlas",
        "hd95": "hd95_cnr_vs_anchored_atlas_mm",
        "color": "#E1812C",
        "marker": "D",
    },
]


GROUP_QC_MANUAL_COMPARISONS = [
    {
        "pair": "cnr_vs_manual",
        "label": "CNR vs Manual",
        "dice": "dice_cnr_vs_manual",
        "hd95": "hd95_cnr_vs_manual_mm",
        "color": "#DD8452",
        "marker": "s",
    },
    {
        "pair": "core_rim_nm_vs_manual",
        "label": "Core-rim vs Manual",
        "dice": "dice_core_rim_nm_vs_manual",
        "hd95": "hd95_core_rim_nm_vs_manual_mm",
        "color": "#2EC4B6",
        "marker": "*",
    },
    {
        "pair": "atlas_vs_manual",
        "label": "Atlas vs Manual",
        "dice": "dice_atlas_vs_manual",
        "hd95": "hd95_atlas_vs_manual_mm",
        "color": "#55A868",
        "marker": "^",
    },
    {
        "pair": "prob_atlas_vs_manual",
        "label": "ProbCIT vs Manual",
        "dice": "dice_prob_atlas_vs_manual",
        "hd95": "hd95_prob_atlas_vs_manual_mm",
        "color": "#F28E2B",
        "marker": "X",
    },
    {
        "pair": "nms_atlas_vs_manual",
        "label": "NMS vs Manual",
        "dice": "dice_nms_atlas_vs_manual",
        "hd95": "hd95_nms_atlas_vs_manual_mm",
        "color": "#8E6C8A",
        "marker": "*",
    },
    {
        "pair": "anchored_atlas_vs_manual",
        "label": "Anchored vs Manual",
        "dice": "dice_anchored_vs_manual",
        "hd95": "hd95_anchored_vs_manual_mm",
        "color": "#151372",
        "marker": "v",
    },
]


GROUP_QC_CORE_RIM_NMS_COMPARISON = {
    "pair": "core_rim_nm_vs_nms_atlas",
    "label": "Core-rim vs NMS",
}


GROUP_QC_COMPONENT_LABELS = {
    "cnr": "NM",
    "core_rim_nm": "Core-rim NM",
    "atlas": "Atlas",
    "prob_atlas": "ProbCIT",
    "nms_atlas": "NMS",
    "anchored_atlas": "Anchored",
    "manual": "Manual",
}


GROUP_QC_PAIR_ORDER = [
    "cnr_vs_atlas",
    "cnr_vs_core_rim_nm",
    "cnr_vs_prob_atlas",
    "cnr_vs_nms_atlas",
    "cnr_vs_anchored_atlas",
    "cnr_vs_manual",
    "core_rim_nm_vs_manual",
    "core_rim_nm_vs_nms_atlas",
    "atlas_vs_manual",
    "prob_atlas_vs_manual",
    "nms_atlas_vs_manual",
    "anchored_atlas_vs_manual",
]


GROUP_QC_PAIRWISE_FIELDS = [
    "subject_id", "pair", "dice", "jaccard", "precision", "recall",
    "hd95_mm", "surface_distance_mean_mm", "surface_distance_median_mm",
    "surface_distance_hd90_mm", "surface_distance_hd95_mm",
    "surface_distance_max_mm", "centroid_distance_mm",
    "volume_a_mm3", "volume_b_mm3", "volume_delta_mm3", "volume_delta_percent",
]


GROUP_QC_COMPONENT_FIELDS = [
    "subject_id", "mask", "n_components", "largest_component_voxels",
    "largest_component_fraction", "surface_voxels", "surface_to_volume_ratio",
    "component_voxel_counts_top5",
]



# =================================================================
# Evaluation Figures
# =================================================================

def save_qc_figure(fig, output_file, dpi=200):
    """Save a group QC figure as a PNG, like the other group plots."""
    fig.savefig(str(Path(output_file).with_suffix('.png')), dpi=dpi, bbox_inches='tight')

ACCURACY_METHODS = [
    ('CNR mask', 'dice_cnr_vs_manual', 'hd95_cnr_vs_manual_mm', '#C44E52'),
    ('CIT168', 'dice_atlas_vs_manual', 'hd95_atlas_vs_manual_mm', '#4C72B0'),
    ('CIT168 prob', 'dice_prob_atlas_vs_manual', 'hd95_prob_atlas_vs_manual_mm', '#55A868'),
    ('NMS prob', 'dice_nms_atlas_vs_manual', 'hd95_nms_atlas_vs_manual_mm', '#8172B2'),
]


def build_accuracy_figure(subject_records, output_file):
    """DICE and HD95 for the CNR mask against the three atlases, with significance marks."""
    from scipy.stats import wilcoxon

    def holm_adjust(p_values):
        """Holm correction, because the CNR mask is tested against three atlases."""
        order = np.argsort(p_values)
        adjusted = np.ones(len(p_values))
        running_max = 0.0
        for rank, index in enumerate(order):
            value = min(1.0, p_values[index] * (len(p_values) - rank))
            running_max = max(running_max, value)
            adjusted[index] = running_max
        return adjusted

    def stars(p_value):
        if p_value < 0.001:
            return '***'
        if p_value < 0.01:
            return '**'
        if p_value < 0.05:
            return '*'
        return 'n.s.'

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric_index, metric_name, better in [(axes[0], 1, 'DICE vs expert mask', 'higher'),
                                                    (axes[1], 2, 'HD95 vs expert mask (mm)', 'lower')]:
        columns = []
        for label, dice_key, hd95_key, colour in ACCURACY_METHODS:
            key = dice_key if metric_index == 1 else hd95_key
            values = {subject_id: qc.get(key) for subject_id, qc in subject_records
                      if isinstance(qc.get(key), (int, float)) and np.isfinite(qc.get(key))}
            columns.append((label, colour, values))

        shared = sorted(set.intersection(*[set(values) for _, _, values in columns]))
        if not shared:
            continue

        for position, (label, colour, values) in enumerate(columns):
            points = np.array([values[subject_id] for subject_id in shared])
            axis.bar(position, np.median(points), width=0.62, color=colour, alpha=0.35)
            jitter = (np.random.RandomState(0).rand(len(points)) - 0.5) * 0.22
            axis.plot(position + jitter, points, 'o', ms=3.5, color=colour, alpha=0.9)
            axis.text(position, np.median(points), f'{np.median(points):.2f}',
                      ha='center', va='bottom', fontsize=9, fontweight='bold')

        # The CNR mask (first column) against each atlas, paired across subjects.
        cnr_points = np.array([columns[0][2][subject_id] for subject_id in shared])
        raw_p = []
        for _, _, values in columns[1:]:
            atlas_points = np.array([values[subject_id] for subject_id in shared])
            # wilcoxon raises when every paired difference is zero, which tied values can cause in small groups.
            raw_p.append(wilcoxon(cnr_points, atlas_points).pvalue
                         if np.any(cnr_points != atlas_points) else float('nan'))
        adjusted_p = holm_adjust(np.array(raw_p))

        span = max(np.max([value for _, _, values in columns for value in values.values()]), 1e-6)
        for offset, (position, p_value) in enumerate(zip([1, 2, 3], adjusted_p)):
            height = span * (1.08 + 0.10 * offset)
            axis.plot([0, 0, position, position],
                      [height - span * 0.02, height, height, height - span * 0.02], lw=1.0, color='0.3')
            axis.text(position / 2.0, height, stars(p_value), ha='center', va='bottom', fontsize=10)

        axis.set_xticks(range(len(columns)))
        axis.set_xticklabels([label for label, _, _ in columns], fontsize=10)
        axis.set_ylabel(metric_name)
        axis.set_title(f'{metric_name}\n({better} is better)', fontsize=11)
        axis.set_ylim(0, span * 1.45)
        axis.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    save_qc_figure(fig, output_file)
    plt.close(fig)
    return output_file


CONTOUR_METHODS = [
    ('CNR mask', 'Native_Tractography_Seed_SNc.nii.gz', 'dice_cnr_vs_manual', '#FF3B30'),
    ('CIT168', 'SNc_trans.nii.gz', 'dice_atlas_vs_manual', '#4C9AFF'),
    ('NMS prob', 'Native_NMS_SNc_thr025.nii.gz', 'dice_nms_atlas_vs_manual', '#B08BFF'),
]


def build_contour_figure(subject_id, subject_qc, output_path, manual_dir, output_file, rank_label=None):
    """
    One row per method, three axial slices per row.
    The expert mask is filled in gold and the method is drawn on top as an outline. 
    Each row is labelled with that method's DICE for this subject.
    """
    anat_dir = Path(output_path) / subject_id / "anat"
    nm_data = nib.load(str(anat_dir / "mean_NM_bias_corrected.nii.gz")).get_fdata()

    manual_matches = sorted(Path(manual_dir).glob(
        f"{subject_id}/anat/{subject_id}_space-NM_label-SN_desc-manual_mask.nii*")) if manual_dir else []
    if not manual_matches:
        return None
    expert_mask = nib.load(str(manual_matches[0])).get_fdata() > 0

    # Choose slices from where the expert mask actually has substance, otherwise
    # a few stray voxels far from the SN would otherwise place slices in empty space.
    slice_areas = expert_mask.sum(axis=(0, 1))
    solid_slices = np.flatnonzero(slice_areas >= 0.10 * slice_areas.max())
    slices = np.percentile(solid_slices, [25, 50, 75]).astype(int)

    margin = 14
    in_plane = np.argwhere(np.any(expert_mask, axis=2))
    rows = slice(max(in_plane[:, 0].min() - margin, 0), in_plane[:, 0].max() + margin + 1)
    cols = slice(max(in_plane[:, 1].min() - margin, 0), in_plane[:, 1].max() + margin + 1)

    fig, axes = plt.subplots(len(CONTOUR_METHODS), len(slices),
                             figsize=(3.5 * len(slices), 3.2 * len(CONTOUR_METHODS)))
    for row, (label, filename, dice_key, colour) in enumerate(CONTOUR_METHODS):
        mask_path = anat_dir / filename
        method_mask = nib.load(str(mask_path)).get_fdata() > 0 if mask_path.exists() else None
        dice_value = subject_qc.get(dice_key)

        for col, z in enumerate(slices):
            axis = axes[row, col]
            background = nm_data[rows, cols, z]
            vmin, vmax = np.percentile(background[background > 0], [2, 99]) if np.any(background > 0) else (0, 1)
            axis.imshow(np.rot90(background), cmap='gray', vmin=vmin, vmax=vmax)

            expert_plane = np.rot90(expert_mask[rows, cols, z])
            if expert_plane.any():
                axis.contourf(expert_plane, levels=[0.5, 1.5], colors=['#FFD400'], alpha=0.30)
                axis.contour(expert_plane, levels=[0.5], colors='#FFD400', linewidths=1.0)
            if method_mask is not None and method_mask[rows, cols, z].any():
                axis.contour(np.rot90(method_mask[rows, cols, z]), levels=[0.5], colors=colour, linewidths=2.0)

            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f'axial slice {z}', fontsize=11)
            if col == 0:
                dice_text = f'\nDICE {dice_value:.2f}' if isinstance(dice_value, (int, float)) else ''
                axis.set_ylabel(f'{label}{dice_text}', fontsize=12, color=colour, fontweight='bold')

    handles = [plt.Line2D([0], [0], color='#FFD400', lw=6, alpha=0.5, label='Expert mask (filled)'),
               plt.Line2D([0], [0], color='0.2', lw=2, label='Method outline')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=11)
    heading = f'{rank_label.capitalize()} case ({subject_id})' if rank_label else subject_id
    fig.suptitle(f'{heading}: contours of each method against the expert mask', fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_qc_figure(fig, output_file)
    plt.close(fig)
    return output_file

def subject_intensities(subject_id, output_path):
    """NM intensities inside the CP reference region and the SN search region."""
    anat_dir = Path(output_path) / subject_id / "anat"
    search_file = anat_dir / "Native_SN_Search_Region.nii.gz"
    if not search_file.exists():
        search_file = anat_dir / "MNI_SNc_Manual_trans.nii.gz"

    nm_data = nib.load(str(anat_dir / "mean_NM_bias_corrected.nii.gz")).get_fdata()
    cp_mask = nib.load(str(anat_dir / "MNI_CP_Manual_trans.nii.gz")).get_fdata() > 0
    sn_mask = nib.load(str(search_file)).get_fdata() > 0

    cp_values = nm_data[cp_mask & np.isfinite(nm_data) & (nm_data > 0)]
    sn_values = nm_data[sn_mask & np.isfinite(nm_data) & (nm_data > 0)]

    # The pipeline's CP reference (the KDE mode), but from the uneroded CP mask, so it can differ slightly.
    intensity_grid = np.linspace(cp_values.min(), cp_values.max(), 1000)
    cp_reference = float(intensity_grid[np.argmax(gaussian_kde(cp_values)(intensity_grid))])
    return cp_values, sn_values, cp_reference


def build_contrast_histogram_figure(subject_ids, output_path, output_file):
    """Every subject's distributions, each scaled by its own CP reference so they can be pooled."""
    cp_ratios = []
    sn_ratios = []
    for subject_id in subject_ids:
        try:
            cp_values, sn_values, cp_reference = subject_intensities(subject_id, output_path)
        except Exception:
            continue
        cp_ratios.append(cp_values / cp_reference)
        sn_ratios.append(sn_values / cp_reference)
    if not cp_ratios:
        return None

    bins = np.linspace(0.6, 1.8, 80)
    bin_centres = 0.5 * (bins[1:] + bins[:-1])
    fig, axis = plt.subplots(figsize=(7, 4.2))
    for ratios, colour in [(cp_ratios, '#4C72B0'), (sn_ratios, '#C44E52')]:
        curves = np.array([np.histogram(values, bins=bins, density=True)[0] for values in ratios])
        axis.plot(bin_centres, curves.mean(axis=0), color=colour, lw=2)
        axis.fill_between(bin_centres, np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0),
                          color=colour, alpha=0.25)
    axis.axvline(1.0, color='#2B4570', ls='--', lw=1.2)
    axis.set_xlabel('NM intensity / CP reference')
    axis.set_ylabel('Density')
    axis.set_title(f'NM contrast across the cohort (n = {len(cp_ratios)}; line = mean, band = IQR)', fontsize=11)
    axis.legend(handles=[plt.Line2D([0], [0], color='#4C72B0', lw=2, label='Cerebral peduncle'),
                         plt.Line2D([0], [0], color='#C44E52', lw=2, label='SN search region')], fontsize=9)
    axis.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save_qc_figure(fig, output_file)
    plt.close(fig)
    return output_file

def nm_blind_control_rows(subject_ids, output_path, manual_dir, seed_cnr_percentile=55):
    """
    Compare three masks per subject against the expert mask.
    drawing: the propagated SN drawing (registration only).
    nm_blind: the same share of the search region the seed keeps (top 45% at percentile 55),
               but chosen by depth inside the region instead of by NM contrast.
    nm_seed: the NM contrast seed the pipeline produces.
    """
    def dice(a, b):
        return float(2 * (a & b).sum() / (a.sum() + b.sum()))

    def hd95(a, b, spacing):
        surface_a = a & ~binary_erosion(a)
        surface_b = b & ~binary_erosion(b)
        distances = np.concatenate([distance_transform_edt(~surface_b, sampling=spacing)[surface_a],
                                    distance_transform_edt(~surface_a, sampling=spacing)[surface_b]])
        return float(np.percentile(distances, 95))

    rows = []
    for subject_id in subject_ids:
        anat_dir = Path(output_path) / subject_id / "anat"
        seed_file = anat_dir / "Native_Tractography_Seed_SNc.nii.gz"
        search_file = anat_dir / "Native_SN_Search_Region.nii.gz"
        drawing_file = anat_dir / "MNI_SNc_Manual_trans.nii.gz"
        manual_matches = sorted(Path(manual_dir).glob(
            f"{subject_id}/anat/{subject_id}_space-NM_label-SN_desc-manual_mask.nii*"))
        if not (seed_file.exists() and search_file.exists() and drawing_file.exists() and manual_matches):
            continue

        seed_img = nib.load(str(seed_file))
        spacing = tuple(float(z) for z in seed_img.header.get_zooms()[:3])
        seed = seed_img.get_fdata() > 0
        search = nib.load(str(search_file)).get_fdata() > 0
        drawing = nib.load(str(drawing_file)).get_fdata() > 0
        expert = nib.load(str(manual_matches[0])).get_fdata() > 0
        if not (seed.any() and search.any() and drawing.any() and expert.any()):
            continue

        # Work in a box around the SN so the distance transforms stay fast.
        corners = np.argwhere(search | drawing | expert)
        box = tuple(slice(max(low - 12, 0), high + 13) for low, high in zip(corners.min(0), corners.max(0)))
        seed, search, drawing, expert = seed[box], search[box], drawing[box], expert[box]

        # NM-blind control: keep the same share of the search region as the seed threshold does,
        # but take the voxels furthest from the region's edge instead of the brightest ones.
        n_keep = max(1, int(round((1 - seed_cnr_percentile / 100) * search.sum())))
        depth = distance_transform_edt(search, sampling=spacing)
        deepest_first = np.argsort(-depth[search], kind="stable")[:n_keep]
        nm_blind = np.zeros_like(search)
        nm_blind[tuple(np.argwhere(search)[deepest_first].T)] = True

        rows.append({
            "subject_id": subject_id,
            "search_region_voxels": int(search.sum()),
            "nm_blind_voxels": int(nm_blind.sum()),
            "nm_seed_voxels": int(seed.sum()),
            "dice_drawing": dice(drawing, expert),
            "dice_nm_blind": dice(nm_blind, expert),
            "dice_nm_seed": dice(seed, expert),
            "hd95_drawing": hd95(drawing, expert, spacing),
            "hd95_nm_blind": hd95(nm_blind, expert, spacing),
            "hd95_nm_seed": hd95(seed, expert, spacing),
        })
    return rows


def build_nm_blind_control_figure(rows, output_file):
    """Per subject: registration only to same share of the region without NM -> NM contrast seed."""

    def holm_adjust(p_values):
        """Holm correction, because the NM seed is tested against two controls in each panel."""
        order = np.argsort(p_values)
        adjusted = np.ones(len(p_values))
        running_max = 0.0
        for rank, index in enumerate(order):
            running_max = max(running_max, min(1.0, p_values[index] * (len(p_values) - rank)))
            adjusted[index] = running_max
        return adjusted

    def stars(p_value):
        if p_value < 0.001:
            return '***'
        if p_value < 0.01:
            return '**'
        if p_value < 0.05:
            return '*'
        return 'n.s.'

    columns = [("drawing", "Registration only", "#4C72B0"),
               ("nm_blind", "45% of search region,\nno NM contrast", "#8C8C8C"),
               ("nm_seed", "NM contrast\nseed", "#C44E52")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for axis, metric, label, better in [(axes[0], "dice", "DICE vs expert mask", "higher"),
                                        (axes[1], "hd95", "HD95 vs expert mask (mm)", "lower")]:
        values = [np.array([row[f"{metric}_{key}"] for row in rows]) for key, _, _ in columns]

        for subject_values in zip(*values):
            axis.plot([0, 1, 2], subject_values, color="0.7", lw=0.8, alpha=0.6)
        for position, ((_, _, colour), column_values) in enumerate(zip(columns, values)):
            axis.plot(np.full(len(column_values), position), column_values, "o", color=colour, ms=5)
            axis.annotate(f"{np.median(column_values):.2f}",
                          xy=(position, np.median(column_values)), xytext=(11, 0),
                          textcoords="offset points", fontsize=10, fontweight="bold",
                          ha="left", va="center",
                          path_effects=[withStroke(linewidth=3, foreground="white")])
        axis.plot([0, 1, 2], [np.median(column_values) for column_values in values],
                  color="k", lw=2.5, marker="s", ms=8)

        blind, seed = values[1], values[2]
        improved = int((seed > blind).sum()) if better == "higher" else int((seed < blind).sum())
        p_value = wilcoxon(seed, blind).pvalue if np.any(seed != blind) else float("nan")
        #axis.set_title(f"{label} ({better} is better)\nNM seed vs no-NM control: better in "
                       #f"{improved}/{len(seed)}, p = {p_value:.2g}", fontsize=10)

        # The NM seed against each control, paired across subjects.
        drawing = values[0]
        raw_p = [p_value,
                 wilcoxon(seed, drawing).pvalue if np.any(seed != drawing) else float("nan")]
        adjusted_p = holm_adjust(np.array(raw_p))

        low = min(column_values.min() for column_values in values)
        high = max(column_values.max() for column_values in values)
        span = high - low
        for offset, (position, adjusted) in enumerate(zip([1, 0], adjusted_p)):
            height = high + span * (0.08 + 0.11 * offset)
            axis.plot([position, position, 2, 2],
                      [height - span * 0.02, height, height, height - span * 0.02], lw=1.0, color='0.3')
            axis.text((position + 2) / 2.0, height, stars(adjusted), ha='center', va='bottom', fontsize=10)
        axis.set_ylim(low - span * 0.08, high + span * 0.32)

        axis.set_ylabel(label)
        axis.set_xticks([0, 1, 2])
        axis.set_xticklabels([column_label for _, column_label, _ in columns])
        axis.set_xlim(-0.4, 2.5)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Registration alone, position alone, and NM contrast", fontsize=12)
    fig.tight_layout()
    save_qc_figure(fig, output_file)
    plt.close(fig)
    return output_file

def build_statistics_table(subject_records, control_rows, output_file):
    """
    Save every paired comparison of the NM seed as one CSV.

    One row per comparison: the NM seed against each atlas, the propagated drawing
    (registration only) and the NM-blind control, for DICE and HD95. Differences are
    signed so that a positive value always means the NM seed is better.
    """
    from scipy.stats import wilcoxon, binomtest

    def holm(p_values):
        order = np.argsort(p_values)
        adjusted = np.ones(len(p_values))
        running_max = 0.0
        for rank, index in enumerate(order):
            running_max = max(running_max, min(1.0, p_values[index] * (len(p_values) - rank)))
            adjusted[index] = running_max
        return adjusted

    # Per-subject values for each method: (label, family, {subject: (dice, hd95)})
    qc_lookup = dict(subject_records)
    methods = [
        ("CIT168", "atlas", "dice_atlas_vs_manual", "hd95_atlas_vs_manual_mm"),
        ("CIT168 prob", "atlas", "dice_prob_atlas_vs_manual", "hd95_prob_atlas_vs_manual_mm"),
        ("NMS-SNc", "atlas", "dice_nms_atlas_vs_manual", "hd95_nms_atlas_vs_manual_mm"),
    ]
    seed_values = {subject_id: (qc.get("dice_cnr_vs_manual"), qc.get("hd95_cnr_vs_manual_mm"))
                   for subject_id, qc in qc_lookup.items()}
    comparisons = [(label, family, {subject_id: (qc.get(dice_key), qc.get(hd95_key))
                                    for subject_id, qc in qc_lookup.items()})
                   for label, family, dice_key, hd95_key in methods]
    if control_rows:
        comparisons.append(("Propagated drawing (registration only)", "control",
                            {row["subject_id"]: (row["dice_drawing"], row["hd95_drawing"]) for row in control_rows}))
        comparisons.append(("NM-blind control (45% of region, no NM)", "control",
                            {row["subject_id"]: (row["dice_nm_blind"], row["hd95_nm_blind"]) for row in control_rows}))

    rng = np.random.default_rng(0)
    rows = []
    for metric_index, metric, better in [(0, "DICE", "higher"), (1, "HD95 (mm)", "lower")]:
        for label, family, values in comparisons:
            shared = sorted(subject_id for subject_id in values
                            if subject_id in seed_values
                            and isinstance(values[subject_id][metric_index], (int, float))
                            and isinstance(seed_values[subject_id][metric_index], (int, float))
                            and np.isfinite(values[subject_id][metric_index])
                            and np.isfinite(seed_values[subject_id][metric_index]))
            if not shared:
                continue
            seed = np.array([seed_values[s][metric_index] for s in shared], dtype=float)
            other = np.array([values[s][metric_index] for s in shared], dtype=float)
            difference = seed - other if better == "higher" else other - seed   # positive = NM seed better

            seed_better = int((difference > 0).sum())
            other_better = int((difference < 0).sum())
            wilcoxon_p = float(wilcoxon(seed, other).pvalue) if np.any(difference != 0) else float("nan")
            sign_p = float(binomtest(seed_better, seed_better + other_better, 0.5).pvalue) if seed_better + other_better else float("nan")
            bootstrap = np.median(rng.choice(difference, size=(10000, len(difference))), axis=1)

            rows.append({
                "metric": metric,
                "better_is": better,
                "comparison": f"NM seed vs {label}",
                "family": family,
                "n_subjects": len(shared),
                "median_nm_seed": float(np.median(seed)),
                "median_other": float(np.median(other)),
                "median_difference_nm_better_positive": float(np.median(difference)),
                "ci95_low": float(np.percentile(bootstrap, 2.5)),
                "ci95_high": float(np.percentile(bootstrap, 97.5)),
                "nm_seed_better": seed_better,
                "other_better": other_better,
                "ties": int((difference == 0).sum()),
                "wilcoxon_p": wilcoxon_p,
                "sign_test_p": sign_p,
            })

    # Holm within each metric's three atlas tests (as on Group_QC_Accuracy), and across every test in the table
    for metric in ["DICE", "HD95 (mm)"]:
        atlas_rows = [row for row in rows if row["metric"] == metric and row["family"] == "atlas"]
        for row, adjusted in zip(atlas_rows, holm(np.array([row["wilcoxon_p"] for row in atlas_rows]))):
            row["holm_p_atlas_tests_this_metric"] = float(adjusted)
    for row, adjusted in zip(rows, holm(np.array([row["wilcoxon_p"] for row in rows]))):
        row["holm_p_all_tests"] = float(adjusted)

    fieldnames = ["metric", "better_is", "comparison", "family", "n_subjects", "median_nm_seed", "median_other",
                  "median_difference_nm_better_positive", "ci95_low", "ci95_high", "nm_seed_better",
                  "other_better", "ties", "wilcoxon_p", "holm_p_atlas_tests_this_metric", "holm_p_all_tests",
                  "sign_test_p"]
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_file


# =================================================================
# Group QC Plotting
# =================================================================

def run_group_qc_stage(output_path, manual_dir, selected_subjects, test_mode=False,
                       seed_cnr_percentile=55, prob_atlas_threshold=0.25, nms_atlas_threshold=0.25):
    """Generate subject dashboards, audit manifests, and group-level QC plots.

    This function intentionally works only from existing Stage 2 outputs. It should
    never trigger registration or mask generation, which makes it safe to rerun while
    inspecting and refining plots.
    """
    output_path = Path(output_path)
    log.info(f"{'=' * 60}")
    log.info("Generating Group QC Plots...")

    selected_subject_ids = set(selected_subjects)
    qc_files = sorted(output_path.glob("sub-*/cnr/SN_Mask_QC.json"))
    qc_files = [qc_file for qc_file in qc_files if qc_file.parts[-3] in selected_subject_ids]

    group_dir = output_path / "group_qc"

    def group_qc_output_path(filename):
        """Return output paths in <output_root>/group_qc, using _TEST names in test mode."""
        group_dir.mkdir(parents=True, exist_ok=True)
        requested_path = Path(filename)
        if test_mode:
            return group_dir / f"{requested_path.stem}_TEST{requested_path.suffix}"
        return group_dir / requested_path.name

    if not qc_files:
        log.info("No QC JSON files found; skipping group QC plots.")
        return

    subject_records = []
    pairwise_rows = []
    component_rows = []

    def finite_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def metric_values(metric_key):
        return [subject_qc.get(metric_key) for _, subject_qc in subject_records]

    def metric_has_values(metric_key):
        return any(finite_float(value) is not None for value in metric_values(metric_key))

    def paired_metric_rows(metric_a, metric_b):
        rows = []
        for subject_id, subject_qc in subject_records:
            value_a = finite_float(subject_qc.get(metric_a))
            value_b = finite_float(subject_qc.get(metric_b))
            if value_a is not None and value_b is not None:
                rows.append((subject_id, value_a, value_b))
        return rows

    def clean_csv_value(value):
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        return value

    def write_csv(path, fieldnames, rows):
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: clean_csv_value(row.get(key)) for key in fieldnames})

    def add_component_rows(subject_id, subject_qc):
        for component_key, component_metrics in subject_qc.get("mask_components", {}).items():
            if not isinstance(component_metrics, dict):
                continue
            component_rows.append({
                "subject_id": subject_id,
                "mask": GROUP_QC_COMPONENT_LABELS.get(component_key, component_key),
                "n_components": component_metrics.get("n_components"),
                "largest_component_voxels": component_metrics.get("largest_component_voxels"),
                "largest_component_fraction": component_metrics.get("largest_component_fraction"),
                "surface_voxels": component_metrics.get("surface_voxels"),
                "surface_to_volume_ratio": component_metrics.get("surface_to_volume_ratio"),
                "component_voxel_counts_top5": ";".join(
                    str(v) for v in component_metrics.get("component_voxel_counts_top5", [])
                ),
            })

    def add_pairwise_rows(subject_id, subject_qc):
        for pair_name, metrics in subject_qc.get("pairwise", {}).items():
            if not isinstance(metrics, dict):
                continue
            pairwise_rows.append({
                "subject_id": subject_id,
                "pair": pair_name,
                "dice": metrics.get("dice"),
                "jaccard": metrics.get("jaccard"),
                "precision": metrics.get("precision_a_vs_b"),
                "recall": metrics.get("recall_a_vs_b"),
                "hd95_mm": metrics.get("hd95_mm"),
                "surface_distance_mean_mm": metrics.get("surface_distance_mean_mm"),
                "surface_distance_median_mm": metrics.get("surface_distance_median_mm"),
                "surface_distance_hd90_mm": metrics.get("surface_distance_hd90_mm"),
                "surface_distance_hd95_mm": metrics.get("surface_distance_hd95_mm"),
                "surface_distance_max_mm": metrics.get("surface_distance_max_mm"),
                "centroid_distance_mm": metrics.get("centroid_distance_mm"),
                "volume_a_mm3": metrics.get("volume_a_mm3"),
                "volume_b_mm3": metrics.get("volume_b_mm3"),
                "volume_delta_mm3": metrics.get("volume_delta_a_minus_b_mm3"),
                "volume_delta_percent": metrics.get("volume_delta_a_minus_b_percent"),
            })

    for qc_json_path in qc_files:
        with open(qc_json_path, "r") as qc_json_file:
            subject_qc = json.load(qc_json_file)

        subject_id = qc_json_path.parts[-3]

        seed_diag_path = qc_json_path.parent / "Seed_Diagnostics.json"
        if seed_diag_path.exists():
            with open(seed_diag_path, "r") as seed_diag_file:
                subject_qc.update(json.load(seed_diag_file))

        subject_records.append((subject_id, subject_qc))


        manifest_json = build_stage2_output_manifest(subject_id, output_path, manual_dir,
                                                     prob_atlas_threshold, nms_atlas_threshold)
        if manifest_json is not None:
            log.info(f"Stage 2 manifest saved: {manifest_json}")

        dashboard_png = build_subject_qc_dashboard(subject_id, output_path, manual_dir,
                                                   prob_atlas_threshold, nms_atlas_threshold)
        if dashboard_png is not None:
            log.info(f"Subject dashboard saved: {dashboard_png}")

        add_component_rows(subject_id, subject_qc)
        add_pairwise_rows(subject_id, subject_qc)

    if pairwise_rows:
        csv_out = group_qc_output_path("Group_QC_Pairwise_Summary.csv")
        write_csv(csv_out, GROUP_QC_PAIRWISE_FIELDS, pairwise_rows)
        log.info(f"Pairwise QC summary saved: {csv_out}")

    if component_rows:
        comp_csv = group_qc_output_path("Group_QC_Component_Summary.csv")
        write_csv(comp_csv, GROUP_QC_COMPONENT_FIELDS, component_rows)
        log.info(f"Component QC summary saved: {comp_csv}")

    has_manual = metric_has_values("dice_cnr_vs_manual")

    if has_manual:
        paired_dice = paired_metric_rows("dice_cnr_vs_manual", "dice_atlas_vs_manual")
        paired_hd95 = paired_metric_rows("hd95_cnr_vs_manual_mm", "hd95_atlas_vs_manual_mm")

        comp = {
            'n_subjects_dice_compared': len(paired_dice),
            'n_subjects_hd95_compared': len(paired_hd95),
            'dice_nm_minus_atlas_median': None,
            'dice_nm_better_count': 0,
            'dice_atlas_better_count': 0,
            'hd95_atlas_minus_nm_median_mm': None,
            'hd95_nm_better_count': 0,
            'hd95_atlas_better_count': 0,
            'subjects_nm_better_both_dice_and_hd95': 0,
        }

        if paired_dice:
            dice_deltas = np.array([nm - atlas for _, nm, atlas in paired_dice], dtype=float)
            comp['dice_nm_minus_atlas_median'] = float(np.median(dice_deltas))
            comp['dice_nm_better_count'] = int(np.sum(dice_deltas > 0))
            comp['dice_atlas_better_count'] = int(np.sum(dice_deltas < 0))

        if paired_hd95:
            hd95_deltas = np.array([atlas - nm for _, nm, atlas in paired_hd95], dtype=float)
            comp['hd95_atlas_minus_nm_median_mm'] = float(np.median(hd95_deltas))
            comp['hd95_nm_better_count'] = int(np.sum(hd95_deltas > 0))
            comp['hd95_atlas_better_count'] = int(np.sum(hd95_deltas < 0))

        dice_map = {sid: (nm, atlas) for sid, nm, atlas in paired_dice}
        hd95_map = {sid: (nm, atlas) for sid, nm, atlas in paired_hd95}
        shared = sorted(set(dice_map).intersection(hd95_map))
        comp['subjects_nm_better_both_dice_and_hd95'] = int(
            sum((dice_map[s][0] > dice_map[s][1]) and (hd95_map[s][0] < hd95_map[s][1]) for s in shared)
        )

        comp_json = group_qc_output_path('Group_QC_Method_Comparison.json')
        with open(comp_json, 'w') as fh:
            json.dump(comp, fh, indent=2)
        log.info(f"Method comparison summary saved: {comp_json}")

        if paired_dice or paired_hd95:
            fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(10.5, 4.5))

            if paired_dice:
                atlas_d = np.array([atlas for _, _, atlas in paired_dice], dtype=float)
                nm_d = np.array([nm for _, nm, _ in paired_dice], dtype=float)
                lim_min = float(min(atlas_d.min(), nm_d.min(), 0.0))
                lim_max = float(max(atlas_d.max(), nm_d.max(), 1.0))
                axes_cmp[0].scatter(atlas_d, nm_d, s=34, alpha=0.85, color='#4C72B0')
                axes_cmp[0].plot([lim_min, lim_max], [lim_min, lim_max], ls='--', lw=1.0, color='gray')
                axes_cmp[0].set_xlim(lim_min, lim_max)
                axes_cmp[0].set_ylim(lim_min, lim_max)
                axes_cmp[0].set_xlabel('Atlas vs Manual DICE')
                axes_cmp[0].set_ylabel('NM vs Manual DICE')
                axes_cmp[0].set_title('DICE: above diagonal favors NM')
                axes_cmp[0].spines[['top', 'right']].set_visible(False)
            else:
                axes_cmp[0].text(0.5, 0.5, 'No paired DICE data', ha='center', va='center', transform=axes_cmp[0].transAxes)
                axes_cmp[0].axis('off')

            if paired_hd95:
                atlas_h = np.array([atlas for _, _, atlas in paired_hd95], dtype=float)
                nm_h = np.array([nm for _, nm, _ in paired_hd95], dtype=float)
                lim_min = float(min(atlas_h.min(), nm_h.min(), 0.0))
                lim_max = float(max(atlas_h.max(), nm_h.max()))
                axes_cmp[1].scatter(atlas_h, nm_h, s=34, alpha=0.85, color='#E1812C')
                axes_cmp[1].plot([lim_min, lim_max], [lim_min, lim_max], ls='--', lw=1.0, color='gray')
                axes_cmp[1].set_xlim(lim_min, lim_max)
                axes_cmp[1].set_ylim(lim_min, lim_max)
                axes_cmp[1].set_xlabel('Atlas vs Manual HD95 (mm)')
                axes_cmp[1].set_ylabel('NM vs Manual HD95 (mm)')
                axes_cmp[1].set_title('HD95: below diagonal favors NM')
                axes_cmp[1].spines[['top', 'right']].set_visible(False)
            else:
                axes_cmp[1].text(0.5, 0.5, 'No paired HD95 data', ha='center', va='center', transform=axes_cmp[1].transAxes)
                axes_cmp[1].axis('off')

            fig_cmp.tight_layout()
            cmp_png = group_qc_output_path('Group_QC_Method_Comparison.png')
            fig_cmp.savefig(str(cmp_png), dpi=200, bbox_inches='tight')
            plt.close(fig_cmp)
            log.info(f"Method comparison plot saved: {cmp_png}")

    def raincloud_strip(ax, values, pos, color, label):
        vals = [finite_float(v) for v in values]
        vals = np.array([v for v in vals if v is not None], dtype=float)
        if len(vals) == 0:
            return

        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(vals))
        ax.scatter(pos + jitter - 0.15, vals, s=18, color=color, alpha=0.7, zorder=3, label=label)

        if len(vals) > 1 and not np.allclose(vals, vals[0]):
            kde = gaussian_kde(vals, bw_method=0.4)
            y_kde = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 200)
            x_kde = (kde(y_kde) / kde(y_kde).max()) * 0.35
            ax.fill_betweenx(y_kde, pos, pos + x_kde, alpha=0.5, color=color)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos - 0.18, pos - 0.12], [med, med], color="k", lw=2, zorder=4)
        ax.plot([pos - 0.18, pos - 0.18], [q1, q3], color="k", lw=1.5, zorder=4)

    dice_plot_specs = [
        spec for spec in GROUP_QC_COMPARISONS
        if spec["pair"] == "cnr_vs_atlas" or metric_has_values(spec["dice"])
    ]
    if has_manual:
        dice_plot_specs.extend([
            spec for spec in GROUP_QC_MANUAL_COMPARISONS
            if spec["pair"] in ["cnr_vs_manual", "atlas_vs_manual"] or metric_has_values(spec["dice"])
        ])

    fig1, ax1 = plt.subplots(figsize=(3 + (len(dice_plot_specs) * 1.5), 5))
    current_pos = 1
    x_labels = []

    for spec in dice_plot_specs:
        raincloud_strip(ax1, metric_values(spec["dice"]), current_pos, spec["color"], spec["label"])
        x_labels.append(spec["label"])
        current_pos += 1

    ax1.axhline(0.5, color="firebrick", lw=0.8, ls="--", alpha=0.6, label="DICE = 0.5")
    ax1.set_xlim(0.5, current_pos - 0.2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(range(1, current_pos))
    ax1.set_xticklabels(x_labels, rotation=15)
    ax1.set_ylabel("DICE coefficient", fontsize=11)
    ax1.set_title("SN Mask Overlap: DICE", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout()

    out1 = group_qc_output_path("Group_QC_DICE_Raincloud.png")
    fig1.savefig(str(out1), dpi=200, bbox_inches="tight")
    plt.close(fig1)
    log.info(f"DICE raincloud saved: {out1}")

    valid_hd = []
    for index, (subject_id, subject_qc) in enumerate(subject_records):
        hd95_value = finite_float(subject_qc.get("hd95_cnr_vs_atlas_mm"))
        if hd95_value is not None:
            valid_hd.append((index, subject_id, hd95_value))

    if valid_hd:
        valid_hd.sort(key=lambda x: x[2])
        sorted_indices = [x[0] for x in valid_hd]
        sorted_subs = [x[1] for x in valid_hd]
        sorted_subject_qc = [subject_records[index][1] for index in sorted_indices]

        fig2, ax2 = plt.subplots(figsize=(max(8, len(sorted_subs) * 0.4), 4))
        x_pos = list(range(len(sorted_subs)))

        hd95_plot_specs = [
            spec for spec in GROUP_QC_COMPARISONS
            if spec["pair"] == "cnr_vs_atlas" or metric_has_values(spec["hd95"])
        ]
        if has_manual:
            hd95_plot_specs.extend([
                spec for spec in GROUP_QC_MANUAL_COMPARISONS
                if spec["pair"] in ["cnr_vs_manual", "atlas_vs_manual"] or metric_has_values(spec["hd95"])
            ])

        for spec in hd95_plot_specs:
            y_values = [
                finite_float(subject_qc.get(spec["hd95"]))
                for subject_qc in sorted_subject_qc
            ]
            y_values = [np.nan if value is None else value for value in y_values]
            if not np.isfinite(y_values).any():
                continue

            if spec["pair"] == "cnr_vs_atlas":
                point_colors = ["firebrick" if value > 5.0 else spec["color"] for value in y_values]
            else:
                point_colors = spec["color"]

            ax2.scatter(
                x_pos,
                y_values,
                c=point_colors,
                s=40,
                marker=spec["marker"],
                alpha=0.9,
                label=spec["label"],
                zorder=3,
            )

        for idx, x in enumerate(x_pos):
            subject_qc = sorted_subject_qc[idx]
            y_vals = [
                finite_float(subject_qc.get(spec["hd95"]))
                for spec in hd95_plot_specs
            ]
            y_vals = [value for value in y_vals if value is not None]
            if len(y_vals) > 1:
                ax2.plot([x, x], [min(y_vals), max(y_vals)], color="gray", linestyle=":", lw=0.8, zorder=1)

        ax2.axhline(5.0, color="firebrick", lw=0.8, ls="--", alpha=0.7, label="5mm threshold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sorted_subs, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("HD95 (mm)", fontsize=11)
        ax2.set_title("SN Mask Surface Distance: HD95", fontsize=12)

        handles, labels = ax2.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax2.legend(unique.values(), unique.keys(), fontsize=9)

        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()

        out2 = group_qc_output_path("Group_QC_HD95_DotPlot.png")
        fig2.savefig(str(out2), dpi=200, bbox_inches="tight")
        plt.close(fig2)
        log.info(f"HD95 dot plot saved: {out2}")

    if pairwise_rows:
        all_pair_specs = (
            GROUP_QC_COMPARISONS
            + GROUP_QC_MANUAL_COMPARISONS
            + [GROUP_QC_CORE_RIM_NMS_COMPARISON]
        )
        pair_pretty = {spec["pair"]: spec["label"] for spec in all_pair_specs}

        vol_data, vol_labels = [], []
        cent_data, cent_labels = [], []

        for pair_key in GROUP_QC_PAIR_ORDER:
            rows_for_pair = [r for r in pairwise_rows if r.get('pair') == pair_key]

            vol_vals = [
                abs(float(r['volume_delta_percent']))
                for r in rows_for_pair
                if r.get('volume_delta_percent') is not None and np.isfinite(float(r['volume_delta_percent']))
            ]
            if vol_vals:
                vol_data.append(vol_vals)
                vol_labels.append(pair_pretty[pair_key])

            cent_vals = [
                float(r['centroid_distance_mm'])
                for r in rows_for_pair
                if r.get('centroid_distance_mm') is not None and np.isfinite(float(r['centroid_distance_mm']))
            ]
            if cent_vals:
                cent_data.append(cent_vals)
                cent_labels.append(pair_pretty[pair_key])

        if vol_data or cent_data:
            fig3, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            if vol_data:
                axes[0].boxplot(vol_data, tick_labels=vol_labels, showmeans=True)
                axes[0].set_ylabel('|Volume delta| (%)')
                axes[0].set_title('Mask Volume Deviation')
                axes[0].tick_params(axis='x', labelrotation=20)
                axes[0].spines[['top', 'right']].set_visible(False)
            else:
                axes[0].text(0.5, 0.5, 'No volume delta data', ha='center', va='center', transform=axes[0].transAxes)
                axes[0].axis('off')

            if cent_data:
                axes[1].boxplot(cent_data, tick_labels=cent_labels, showmeans=True)
                axes[1].set_ylabel('Centroid distance (mm)')
                axes[1].set_title('Mask Centroid Deviation')
                axes[1].tick_params(axis='x', labelrotation=20)
                axes[1].spines[['top', 'right']].set_visible(False)
            else:
                axes[1].text(0.5, 0.5, 'No centroid distance data', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].axis('off')

            fig3.tight_layout()
            out3 = group_qc_output_path('Group_QC_Deviation_Boxplots.png')
            fig3.savefig(str(out3), dpi=200, bbox_inches='tight')
            plt.close(fig3)
            log.info(f"Deviation boxplots saved: {out3}")

    coupling_pairs = [
        (
            sid,
            finite_float(subject_qc.get('seed_voxels')),
            finite_float(subject_qc.get('search_roi_voxels')),
            finite_float(subject_qc.get('mean_search_region_cr')),
        )
        for sid, subject_qc in subject_records
    ]
    coupling_pairs = [
        (sid, seed, roi, signal) for sid, seed, roi, signal in coupling_pairs
        if seed is not None and roi is not None and signal is not None
    ]

    if len(coupling_pairs) > 2:
        seed_arr = np.array([seed for _, seed, _, _ in coupling_pairs])
        roi_arr = np.array([roi for _, _, roi, _ in coupling_pairs])
        signal_arr = np.array([signal for _, _, _, signal in coupling_pairs])

        r_roi = float(np.corrcoef(seed_arr, roi_arr)[0, 1])
        r_signal = float(np.corrcoef(seed_arr, signal_arr)[0, 1])

        fig5, axes5 = plt.subplots(1, 2, figsize=(10.5, 4.5))

        axes5[0].scatter(roi_arr, seed_arr, s=34, alpha=0.85, color='#4C72B0')
        axes5[0].set_xlabel('Search ROI size (voxels, = warped atlas)')
        axes5[0].set_ylabel('Seed size (voxels)')
        axes5[0].set_title(f'Seed size vs atlas ROI size\nr = {r_roi:.2f}')
        axes5[0].spines[['top', 'right']].set_visible(False)

        axes5[1].scatter(signal_arr, seed_arr, s=34, alpha=0.85, color='#E1812C')
        axes5[1].set_xlabel('Mean NM contrast in search region')
        axes5[1].set_ylabel('Seed size (voxels)')
        axes5[1].set_title(f'Seed size vs NM signal strength\nr = {r_signal:.2f}')
        axes5[1].spines[['top', 'right']].set_visible(False)

        fig5.suptitle('Seed size tracks the atlas ROI far more than it tracks NM signal', fontsize=11)
        fig5.tight_layout()

        out5 = group_qc_output_path('Group_QC_Seed_Atlas_Coupling.png')
        fig5.savefig(str(out5), dpi=200, bbox_inches='tight')
        plt.close(fig5)
        log.info(f"Seed/atlas coupling plot saved: {out5}")

    boundary_pairs = paired_metric_rows('core_median_cr', 'boundary_median_cr')
    if boundary_pairs:
        core_vals = np.array([core for _, core, _ in boundary_pairs])
        boundary_vals = np.array([boundary for _, _, boundary in boundary_pairs])
        lim_min = float(min(core_vals.min(), boundary_vals.min(), 0.0))
        lim_max = float(max(core_vals.max(), boundary_vals.max()))

        fig6, ax6 = plt.subplots(figsize=(5.5, 5))
        ax6.scatter(core_vals, boundary_vals, s=34, alpha=0.85, color='#4C72B0')
        ax6.plot([lim_min, lim_max], [lim_min, lim_max], ls='--', lw=1.0, color='gray',
                 label='Boundary CR = Core CR')
        ax6.set_xlim(lim_min, lim_max)
        ax6.set_ylim(lim_min, lim_max)
        ax6.set_xlabel('Median CR, mask interior (>1mm from manual edge)')
        ax6.set_ylabel('Median CR, manual boundary band (±1mm)')
        ax6.set_title('If the boundary were an NM-intensity feature,\npoints would fall well below the diagonal')
        ax6.legend(fontsize=9, loc='lower right')
        ax6.spines[['top', 'right']].set_visible(False)
        fig6.tight_layout()

        out6 = group_qc_output_path('Group_QC_Boundary_vs_Core_CR.png')
        fig6.savefig(str(out6), dpi=200, bbox_inches='tight')
        plt.close(fig6)
        log.info(f"Boundary-vs-core CR plot saved: {out6}")

    # =================================================================
    # NM-Blind Control and Statistics
    # =================================================================
    if manual_dir is not None:
        control_rows = nm_blind_control_rows([subject_id for subject_id, _ in subject_records],
                                                output_path, manual_dir, seed_cnr_percentile)
        if control_rows:
            control_csv = group_qc_output_path('Group_QC_NM_Blind_Control.csv')
            write_csv(control_csv, list(control_rows[0].keys()), control_rows)
            log.info(f"NM-blind control table saved: {control_csv}")

            control_png = group_qc_output_path('Group_QC_NM_Blind_Control.png')
            build_nm_blind_control_figure(control_rows, control_png)
            log.info(f"NM-blind control figure saved: {control_png}")
        statistics_csv = group_qc_output_path('Group_QC_Statistics.csv')
        build_statistics_table(subject_records, control_rows, statistics_csv)
        log.info(f"Statistics table saved: {statistics_csv}")


    # =================================================================
    # Evaluation Figures
    # =================================================================
    ranked_records = sorted([(subject_id, subject_qc) for subject_id, subject_qc in subject_records
                            if finite_float(subject_qc.get('dice_cnr_vs_manual')) is not None],
                            key=lambda record: record[1]['dice_cnr_vs_manual'])
    if ranked_records:
        accuracy_png = group_qc_output_path('Group_QC_Accuracy.png')
        build_accuracy_figure(subject_records, accuracy_png)
        log.info(f"Accuracy figure saved: {accuracy_png}")

        ranked_subjects = {'worst': ranked_records[0][0],
                        'median': ranked_records[len(ranked_records) // 2][0],
                        'best': ranked_records[-1][0]}
        record_lookup = dict(ranked_records)
        for rank_label, subject_id in ranked_subjects.items():
            contour_png = group_qc_output_path(f'Group_QC_Contours_{rank_label}_{subject_id}.png')
            if build_contour_figure(subject_id, record_lookup[subject_id], output_path, manual_dir, contour_png, rank_label):
                log.info(f"Contour figure saved: {contour_png}")

        group_histogram_png = group_qc_output_path('Group_QC_CNR_Histogram.png')
        build_contrast_histogram_figure([subject_id for subject_id, _ in subject_records],
                                        output_path, group_histogram_png)
        log.info(f"Group contrast histogram saved: {group_histogram_png}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild NeuroMelAnchor QC reports from existing Stage 2 outputs.")
    parser.add_argument("--output_root", type=Path, required=True, help="Pipeline output root holding the sub-* folders")
    parser.add_argument("--manual_dir", type=Path, default=None,
                        help="Root with sub-*/anat/*_desc-manual_mask.nii.gz; enables the accuracy outputs")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subjects to include (default: every sub-* folder)")
    parser.add_argument("--seed_cnr_percentile", type=float, default=55, help="The value the run used (default: 55)")
    parser.add_argument("--prob_atlas_threshold", type=float, default=0.25, help="The value the run used (default: 0.25)")
    parser.add_argument("--nms_atlas_threshold", type=float, default=0.25, help="The value the run used (default: 0.25)")
    parser.add_argument("--test", action="store_true", help="Add a _TEST suffix to the group outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    subjects = args.subjects or sorted(p.name for p in args.output_root.glob("sub-*") if p.is_dir())
    run_group_qc_stage(args.output_root, args.manual_dir, subjects, test_mode=args.test,
                       seed_cnr_percentile=args.seed_cnr_percentile,
                       prob_atlas_threshold=args.prob_atlas_threshold,
                       nms_atlas_threshold=args.nms_atlas_threshold)


if __name__ == "__main__":
    main()
