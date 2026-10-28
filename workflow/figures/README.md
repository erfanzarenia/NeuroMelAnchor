# Figure scripts

Post-hoc figure scripts that read finished pipeline outputs. They are not part of
the pipeline and the pipeline never calls them. Each takes its paths on the
command line; run either with `--help`.

| Script | Reads | Writes |
|---|---|---|
| `poster_workflow_panels.py` | One output root: the V4 drawing template and tracing, plus one subject's NM image, search region and NM-defined mask | `Workflow_1a` to `Workflow_5` panels illustrating the method; panel 1c (the subject's own T1) only with `--raw-t1-dir` |
| `poster_dilation_sweep.py` | Several output roots, one per `--search_dilation_mm` value, named `results_dil-<mm>` | `Poster_Sweep_1x2.png`: precision, recall and DICE on one panel, HD95 on the other |
| `mask_comparison_figures.py` | One Stage 2 output root | A figure per subject comparing the NM mask with CIT168, NMS and ProbCIT, a contact sheet with the cohort outliers first, and `Mask_Comparison_Summary.csv` |

The panels are written as PNG and SVG, the sweep figure as a transparent PNG.

Both scripts were written for the Brainhack Western 2026 poster.

The poster figures were made with:

```bash
python workflow/figures/poster_workflow_panels.py \
    --output-root /path/to/NM_BioGen_PP/echo-1/results_dil-0.5 \
    --subject sub-059 --slice 30 --raw-t1-dir /path/to/NM_BioGen/7T

python workflow/figures/poster_dilation_sweep.py \
    --runs-root /path/to/NM_BioGen_PP/echo-1 --chosen-mm 0.5
```

The group-level poster figures (`Poster_Accuracy`, `Poster_Contours_*`,
`Poster_Strip_*`, `Poster_CNR_Histogram_*` and `Poster_NM_Blind_Control`) are not
made here. The pipeline writes those itself during `--stage group_qc` when manual
masks are available; see `docs/EVALUATION.md`.
