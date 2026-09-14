#!/usr/bin/env python3
"""
Plot how the refined mask changes as the propagated search region is grown.
"""
import csv
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================================================================
# Settings
# =================================================================
RUNS_ROOT = Path("/scratch/Projects/BrainHack_Nexus/NM_BioGen_PP/echo-1")
RUN_PATTERN = "results_dil-*"          
MANUAL_DIR = Path("/scratch/Projects/BrainHack_Nexus/NM_BioGen/7T")
PANEL_DIR = RUNS_ROOT / "poster_figures"
CHOSEN_MM = 0.5                        # the dilation to mark on each plot
INCLUDE_COVERAGE = True                # adds the coverage line to the DICE plot (slower: reads masks)

MASK_PAIR = "cnr_vs_manual"            # the NM-defined mask against the manual reference
DICE_COLOUR = "#FF3B30"
PRECISION_COLOUR = "#C44E52"
RECALL_COLOUR = "#4C72B0"
HD95_COLOUR = "#55A868"
COVERAGE_COLOUR = "#4C72B0"


# =================================================================
# Reading the runs
# =================================================================
def dilation_from_name(run_dir):
    """Pull the dilation in mm out of a folder name like results_dil-1.5."""
    match = re.search(r"dil-([0-9]*\.?[0-9]+)", Path(run_dir).name)
    return float(match.group(1)) if match else None


def read_metric(run_dir, column):
    """Per-subject value of one column for the NM-defined mask vs the manual reference."""
    values = {}
    with open(Path(run_dir) / "Group_QC_Pairwise_Summary.csv") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["pair"] == MASK_PAIR and row.get(column):
                values[row["subject_id"]] = float(row[column])
    return values


def read_region_coverage(run_dir):
    """Share of each subject's manual reference that falls inside the search region."""
    coverage = {}
    for region_file in sorted(Path(run_dir).glob("sub-*/anat/Native_SN_Search_Region.nii.gz")):
        subject_id = region_file.parts[-3]
        manual_matches = sorted(MANUAL_DIR.glob(
            f"{subject_id}/anat/{subject_id}_space-NM_label-SN_desc-manual_mask.nii*"))
        if not manual_matches:
            continue
        region = nib.load(str(region_file)).get_fdata() > 0
        manual = nib.load(str(manual_matches[0])).get_fdata() > 0
        if manual.any():
            coverage[subject_id] = float((region & manual).sum() / manual.sum())
    return coverage


def collect_runs():
    """Every archived run that has a group QC summary, in order of dilation."""
    runs = []
    for run_dir in sorted(RUNS_ROOT.glob(RUN_PATTERN)):
        dilation = dilation_from_name(run_dir)
        if dilation is not None and (run_dir / "Group_QC_Pairwise_Summary.csv").exists():
            runs.append((dilation, run_dir))
    runs.sort()
    return runs


# =================================================================
# Plotting
# =================================================================
def metric_matrix(runs, subjects, column):
    """Rows are dilations, columns are subjects."""
    return np.array([[read_metric(run_dir, column).get(subject, np.nan) for subject in subjects]
                     for _, run_dir in runs])

def save_poster_figure(fig, output_file, dpi=400):
    """Save a poster figure as PNG (300 dpi) plus SVG and PNG, which stay sharp at any print size."""
    output_file = Path(output_file)
    fig.savefig(str(output_file), dpi=dpi, bbox_inches='tight')
    fig.savefig(str(output_file.with_suffix('.svg')), bbox_inches='tight')
    fig.savefig(str(output_file.with_suffix('.png')), dpi=dpi, bbox_inches='tight')


def plot_sweep(dilations, series, subjects, y_label, title, output_file,
               y_limits=None, second_line=None, second_label=None, second_limits=None):
    """One panel: faint per-subject lines, a bold median, and an optional second line."""
    values, colour, label = series
    fig, axis = plt.subplots(figsize=(8.6, 5.6))

    for column in range(values.shape[1]):
        axis.plot(dilations, values[:, column], color=colour, alpha=0.16, lw=1.0)
    median_values = np.nanmedian(values, axis=1)
    axis.plot(dilations, median_values, color=colour, lw=3.5, marker="o", ms=9, label=label)

    if CHOSEN_MM in dilations:
        chosen_index = dilations.index(CHOSEN_MM)
        axis.axvline(CHOSEN_MM, color="0.45", ls="--", lw=1.4)
        axis.annotate(f"{median_values[chosen_index]:.2f}",
                      xy=(CHOSEN_MM, median_values[chosen_index]), xytext=(6, 10),
                      textcoords="offset points", fontsize=19, fontweight="bold", color=colour)

    axis.set_xlabel("Search region grown by (mm)", fontsize=15)
    axis.set_ylabel(y_label, fontsize=15, color=colour)
    axis.set_xticks(dilations)
    axis.tick_params(labelsize=13)
    if y_limits:
        axis.set_ylim(*y_limits)
    axis.spines[["top"]].set_visible(False)

    handles = [plt.Line2D([0], [0], color=colour, lw=3.5, marker="o", label=label)]

    if second_line is not None:
        second_values, second_colour = second_line
        if second_limits:                       # a different scale, so use the right-hand axis
            second_axis = axis.twinx()
            second_axis.plot(dilations, np.nanmedian(second_values, axis=1), color=second_colour,
                             lw=2.5, ls=":", marker="s", ms=8)
            second_axis.set_ylabel(second_label, fontsize=15, color=second_colour)
            second_axis.set_ylim(*second_limits)
            second_axis.tick_params(labelsize=13)
            second_axis.spines[["top"]].set_visible(False)
        else:                                   # same scale, so plot it alongside
            for column in range(second_values.shape[1]):
                axis.plot(dilations, second_values[:, column], color=second_colour, alpha=0.13, lw=1.0)
            axis.plot(dilations, np.nanmedian(second_values, axis=1), color=second_colour,
                      lw=3.0, ls="--", marker="s", ms=8)
        handles.append(plt.Line2D([0], [0], color=second_colour, lw=2.5,
                                  ls=":" if second_limits else "--", marker="s", label=second_label))

    axis.legend(handles=handles, fontsize=12.5, loc="best", framealpha=0.95)
    axis.set_title(f"{title}\nn = {len(subjects)} subjects", fontsize=16)
    fig.tight_layout()
    save_poster_figure(fig, output_file)
    plt.close(fig)
    print(f"saved: {output_file}")


def main():
    runs = collect_runs()
    if not runs:
        raise SystemExit(f"No runs matching {RUN_PATTERN} under {RUNS_ROOT}")
    dilations = [dilation for dilation, _ in runs]
    print(f"found {len(runs)} runs: {dilations}")

    dice_tables = [read_metric(run_dir, "dice") for _, run_dir in runs]
    subjects = sorted(set.intersection(*[set(table) for table in dice_tables]))
    if not subjects:
        raise SystemExit("No subjects are shared by all runs.")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    dice = metric_matrix(runs, subjects, "dice")
    precision = metric_matrix(runs, subjects, "precision")
    recall = metric_matrix(runs, subjects, "recall")
    hd95 = metric_matrix(runs, subjects, "hd95_mm")

    coverage = None
    if INCLUDE_COVERAGE:
        coverage = np.array([[read_region_coverage(run_dir).get(subject, np.nan) for subject in subjects]
                             for _, run_dir in runs]) * 100

    plot_sweep(dilations, (dice, DICE_COLOUR, "DICE"), subjects,
               "DICE vs expert mask", "How far to grow the search region",
               PANEL_DIR / "Poster_Sweep_Dice.png", y_limits=(0.1, 0.9),
               second_line=(coverage, COVERAGE_COLOUR) if coverage is not None else None,
               second_label="Expert mask covered by the region (%)",
               second_limits=(60, 104) if coverage is not None else None)

    plot_sweep(dilations, (precision, PRECISION_COLOUR, "% of our mask inside the expert mask"), subjects,
               "Share of Voxels", "Growing the region finds more SN but includes more non-SN",
               PANEL_DIR / "Poster_Sweep_Precision.png", y_limits=(0.2, 1.02),
               second_line=(recall, RECALL_COLOUR), second_label="% of the expert mask our mask covers")


    plot_sweep(dilations, (hd95, HD95_COLOUR, "HD95"), subjects,
               "HD95 vs expert Mask (mm)", "Boundary error",
               PANEL_DIR / "Poster_Sweep_HD95.png")

    print(f"\n{'dilation':>9s}{'dice':>8s}{'precision':>11s}{'recall':>9s}{'hd95':>8s}")
    for index, dilation in enumerate(dilations):
        print(f"{dilation:9.1f}{np.nanmedian(dice[index]):8.3f}{np.nanmedian(precision[index]):11.3f}"
              f"{np.nanmedian(recall[index]):9.3f}{np.nanmedian(hd95[index]):8.2f}")


if __name__ == "__main__":
    main()
