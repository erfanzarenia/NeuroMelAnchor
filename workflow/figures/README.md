# Figure script

A post-hoc figure script that reads finished pipeline outputs. It is not part of the
pipeline and the pipeline never calls it. It takes its paths on the command line; run it
with `--help`.

| Script | Reads | Writes |
|---|---|---|
| `mask_comparison_figures.py` | One Stage 2 output root | A figure per subject comparing the NM mask with CIT168, NMS and ProbCIT, a contact sheet with the cohort outliers first, and `Mask_Comparison_Summary.csv` |

The group QC figures are not made here; see `workflow/qc/group_qc.py` and
`docs/EVALUATION.md`.
