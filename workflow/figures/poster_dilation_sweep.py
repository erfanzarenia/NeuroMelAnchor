#!/usr/bin/env python3
"""
Plot how the NM-defined mask changes as the propagated search region is grown.

Reads one Stage 2 output root per --search_dilation_mm value, in directories named
like results_dil-0.5, and draws one two-panel figure against the dilation: precision,
recall and DICE of the NM-defined mask against the expert (manual) masks on one
axis, and HD95 on the other.

    python workflow/figures/poster_dilation_sweep.py --runs-root /path/to/NM_BioGen_PP/echo-1

Each run directory needs a Group_QC_Pairwise_Summary.csv, written by --stage 2 or
--stage group_qc. The figure is saved as a transparent PNG.

A post-hoc figure script written for the Brainhack Western 2026 poster; it is not
part of the pipeline.
"""
import argparse
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
DICE_COLOUR = "0.15"
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


def read_metric(run_dir, column, mask_pair):
    """Per-subject value of one summary column for one mask pair."""
    values = {}
    with open(Path(run_dir) / "Group_QC_Pairwise_Summary.csv") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["pair"] == mask_pair and row.get(column):
                values[row["subject_id"]] = float(row[column])
    return values


def collect_runs(runs_root, run_pattern):
    """Every archived run that has a group QC summary, in order of dilation."""
    runs = []
    for run_dir in sorted(Path(runs_root).glob(run_pattern)):
        dilation = dilation_from_name(run_dir)
        # The summary check also skips archives such as results_dil-2.5.tar.gz that match the glob.
        if dilation is not None and (run_dir / "Group_QC_Pairwise_Summary.csv").exists():
            runs.append((dilation, run_dir))
    runs.sort()
    return runs


# =================================================================
# Plotting
# =================================================================
def metric_matrix(runs, subjects, column, mask_pair):
    """Rows are dilations, columns are subjects."""
    tables = [read_metric(run_dir, column, mask_pair) for _, run_dir in runs]
    return np.array([[table.get(subject, np.nan) for subject in subjects] for table in tables])

def save_poster_figure(fig, output_file, dpi=400):
    """Save a poster figure as a 400 dpi PNG."""
    fig.savefig(str(output_file), dpi=dpi, bbox_inches='tight', transparent=True)


def plot_sweep(axis, dilations, series, subjects, y_label, title,
               y_limits=None, second_line=None, second_label=None, second_limits=None, chosen_mm=None):
    """Draw one panel onto axis: faint per-subject lines, a bold median, and an optional second line."""
    values, colour, label = series

    for column in range(values.shape[1]):
        axis.plot(dilations, values[:, column], color=colour, alpha=0.16, lw=1.0)
    median_values = np.nanmedian(values, axis=1)
    axis.plot(dilations, median_values, color=colour, lw=2.0, marker="o", ms=5, label=label)

    if chosen_mm is not None and chosen_mm in dilations:
        chosen_index = dilations.index(chosen_mm)
        axis.axvline(chosen_mm, color="0.45", ls="--", lw=1.0)
        axis.annotate(f"{median_values[chosen_index]:.2f}",
                      xy=(chosen_mm, median_values[chosen_index]), xytext=(4, 6),
                      textcoords="offset points", fontsize=9, fontweight="bold", color=colour)

    axis.set_xlabel("Search region grown by (mm)", fontsize=10)
    axis.set_ylabel(y_label, fontsize=10)
    axis.set_xticks(dilations)
    axis.tick_params(labelsize=9)
    if y_limits:
        axis.set_ylim(*y_limits)
    axis.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([0], [0], color=colour, lw=2.0, marker="o", label=label)]

    if second_line is not None:
        second_values, second_colour = second_line
        if second_limits:                       # a different scale, so use the right-hand axis
            second_axis = axis.twinx()
            second_axis.plot(dilations, np.nanmedian(second_values, axis=1), color=second_colour,
                             lw=1.5, ls=":", marker="s", ms=4.5)
            second_axis.set_ylabel(second_label, fontsize=10, color=second_colour)
            second_axis.set_ylim(*second_limits)
            second_axis.tick_params(labelsize=9)
            second_axis.spines[["top"]].set_visible(False)
        else:                                   # same scale, so plot it alongside
            for column in range(second_values.shape[1]):
                axis.plot(dilations, second_values[:, column], color=second_colour, alpha=0.13, lw=1.0)
            axis.plot(dilations, np.nanmedian(second_values, axis=1), color=second_colour,
                      lw=1.8, ls="--", marker="s", ms=4.5)
        handles.append(plt.Line2D([0], [0], color=second_colour, lw=1.5,
                                  ls=":" if second_limits else "--", marker="s", label=second_label))
    axis.set_title(title, fontsize=11)
    axis.legend(handles=handles, fontsize=8, loc="best", framealpha=0.95)


def plot_overlap(axis, dilations, precision, recall, dice, subjects, chosen_mm=None):
    """Precision, recall and DICE on one 0-1 axis, so the tradeoff behind DICE is visible."""
    # (values, colour, line style, marker, label, label offset at the chosen dilation)
    series = [
        (precision, PRECISION_COLOUR, "-", "o",
         "Proportion of pipeline mask inside expert segmentation", (5, 5)),
        (recall, RECALL_COLOUR, "-", "s",
         "Proportion of expert segmentation captured by pipeline mask", (5, -14)),
        (dice, DICE_COLOUR, "--", "D",
         "DICE (balance of the two)", (-26, 4)),
    ]
    chosen_index = dilations.index(chosen_mm) if chosen_mm is not None and chosen_mm in dilations else None

    for values, colour, line_style, marker, label, label_offset in series:
        for column in range(values.shape[1]):
            axis.plot(dilations, values[:, column], color=colour, alpha=0.10, lw=1.0)
        median_values = np.nanmedian(values, axis=1)
        axis.plot(dilations, median_values, color=colour, lw=2.0, ls=line_style,
                  marker=marker, ms=5, label=label)
        if chosen_index is not None:
            axis.annotate(f"{median_values[chosen_index]:.2f}",
                          xy=(chosen_mm, median_values[chosen_index]), xytext=label_offset,
                          textcoords="offset points", fontsize=9, fontweight="bold", color=colour)

    if chosen_index is not None:
        axis.axvline(chosen_mm, color="0.45", ls="--", lw=1.0)

    axis.set_xlabel("Search region grown by (mm)", fontsize=10)
    axis.set_ylabel("Overlap with expert segmentation", fontsize=10)
    axis.set_xticks(dilations)
    axis.set_ylim(0.15, 1.02)
    axis.tick_params(labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8, loc="lower left", framealpha=0.95)
    axis.set_title("Growing the search region finds more SN but adds non-SN voxels", fontsize=11)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Directory holding one Stage 2 output root per dilation.")
    parser.add_argument("--run-pattern", default="results_dil-*",
                        help="Glob selecting the run directories under --runs-root (default: results_dil-*).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the figure (default: <runs-root>/poster_figures).")
    parser.add_argument("--chosen-mm", type=float, default=0.5,
                        help="Dilation to mark on each panel (default: 0.5).")
    parser.add_argument("--mask-pair", default="cnr_vs_manual",
                        help="Pair from Group_QC_Pairwise_Summary.csv to plot. The labels assume a "
                             "*_vs_manual pair (default: cnr_vs_manual, the NM-defined mask).")
    return parser.parse_args()


def main():
    args = parse_args()
    runs = collect_runs(args.runs_root, args.run_pattern)
    if not runs:
        raise SystemExit(f"No runs matching {args.run_pattern} under {args.runs_root}")
    dilations = [dilation for dilation, _ in runs]
    print(f"found {len(runs)} runs: {dilations}")

    dice_tables = [read_metric(run_dir, "dice", args.mask_pair) for _, run_dir in runs]
    subjects = sorted(set.intersection(*[set(table) for table in dice_tables]))
    if not subjects:
        raise SystemExit("No subjects are shared by all runs.")

    panel_dir = args.out_dir or args.runs_root / "poster_figures"
    panel_dir.mkdir(parents=True, exist_ok=True)
    dice = metric_matrix(runs, subjects, "dice", args.mask_pair)
    precision = metric_matrix(runs, subjects, "precision", args.mask_pair)
    recall = metric_matrix(runs, subjects, "recall", args.mask_pair)
    hd95 = metric_matrix(runs, subjects, "hd95_mm", args.mask_pair)

    # One 1 x 2 poster figure: overlap (precision, recall, DICE) and boundary error (HD95)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    plot_overlap(axes[0], dilations, precision, recall, dice, subjects, chosen_mm=args.chosen_mm)

    plot_sweep(axes[1], dilations, (hd95, HD95_COLOUR, "HD95"), subjects,
               "HD95 distance to expert segmentation (mm)",
               "Boundary error: 95th-percentile distance to expert segmentation",
               chosen_mm=args.chosen_mm)

    fig.tight_layout(w_pad=2.0)
    sweep_file = panel_dir / "Poster_Sweep_1x2.png"
    save_poster_figure(fig, sweep_file)
    plt.close(fig)
    print(f"saved: {sweep_file}")

    print(f"\n{'dilation':>9s}{'dice':>8s}{'precision':>11s}{'recall':>9s}{'hd95':>8s}")
    for index, dilation in enumerate(dilations):
        print(f"{dilation:9.1f}{np.nanmedian(dice[index]):8.3f}{np.nanmedian(precision[index]):11.3f}"
              f"{np.nanmedian(recall[index]):9.3f}{np.nanmedian(hd95[index]):8.2f}")


if __name__ == "__main__":
    main()
