# Checks

Fast checks that the repository still holds together.

```bash
pip install pytest
cd tests && python -m pytest
```

They need the Python packages in `requirements.txt` plus pytest, and nothing else:
Nipype, ANTs, SPM, ITK-SNAP, FireANTs and a GPU are **not** required, and no imaging
data is read. Every numeric test builds its own small synthetic volume.

## What is covered

| File | Covers |
|---|---|
| `test_repo_layout.py` | Expected scripts, tools, figure scripts and documents exist; every atlas file the pipeline hard-codes is present; no site-specific absolute paths; no compiled files are tracked; there is a single requirements file |
| `test_scripts_parse.py` | Every Python file parses and every shell script passes `bash -n`; the pipeline's options and stage names are intact; the bodies of Nipype `Function` nodes are self-contained; every standalone script's `--help` works and none runs without arguments; the figure scripts read only file names and CSV columns that the pipeline still writes |
| `test_mask_helpers.py` | `select_nm_image_files`, `threshold_probability_mask`, `create_core_rim_nm_mask`, `build_sn_roi_mask`, `compute_motion_params`, `dilate_search_region`, `first_nm_file`, `threshold_t1_seed`, `make_t1_seed_reference`, and the closing and slice fill in `extract_tractography_seed` |
| `test_group_qc_helpers.py` | The NM-blind control in `nm_blind_control_rows`, and `build_poster_accuracy_figure` with tied metrics |
| `test_figures.py` | Run discovery, metric reading and panel cropping in the figure scripts |

## How the pipeline is tested without importing it

`neuromelanchor_pipeline.py` parses its arguments at import time, so it cannot be
imported. `conftest.py` instead lifts individual functions, and module constants where
needed, out of it with `ast` and executes just those definitions.

This works because functions handed to Nipype's `Function` interface must import
everything they use inside their own bodies: Nipype re-serializes them from source and
runs them in a bare worker process. `test_nipype_function_nodes_are_self_contained`
checks that property directly, since a node body that relied on a module-level import
would otherwise fail with a `NameError` inside a worker, hours into a run.

Group QC helpers run in the main process and do use module-level imports, so their tests
supply those names explicitly.

## What is not covered

Registration, template building, segmentation and the dashboards are not exercised,
because they need the external software and real data. Treat these as smoke checks and
unit tests, not a validation of scientific output. See `docs/STATUS.md`.
