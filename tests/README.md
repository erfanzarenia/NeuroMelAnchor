# Checks

Fast, dependency-light checks that the cleaned repository still holds together.

```bash
pip install pytest
cd tests && python -m pytest
```

They need only `numpy`, `nibabel`, `scipy`, `matplotlib` and `pytest`. Nipype,
ANTs, SPM, ITK-SNAP, FireANTs and a GPU are **not** required, and no imaging data
is read — every numeric test builds its own synthetic volume.

## What is covered

| File | Covers |
|---|---|
| `test_repo_layout.py` | Expected scripts, tools and docs exist; every atlas file the pipeline hard-codes is present; no site-specific absolute paths; no generated or compiled files committed |
| `test_scripts_parse.py` | Every Python file parses and every shell script passes `bash -n`; the pipeline's command-line options and stage names are intact; each tool's `--help` works and none falls back to a built-in path; **Nipype `Function` node bodies are self-contained** |
| `test_mask_helpers.py` | Numeric behaviour of `select_nm_image_files`, `threshold_probability_mask`, `create_core_rim_nm_mask`, `build_sn_roi_mask`, `binarize_mask`, `compute_motion_params` and `resolve_input_patterns` |

## How the pipeline is tested without importing it

`neuromelanchor_pipeline.py` parses its arguments at import time, so it cannot be
imported. `conftest.py` instead lifts individual functions out of it with `ast`
and executes just those definitions.

This works because functions handed to Nipype's `Function` interface must import
everything they use inside their own body — Nipype re-serialises them by source
and re-executes them in a bare worker process. `test_nipype_function_nodes_are_
self_contained` checks that property directly: a module-level import that a node
body relies on would otherwise surface as a `NameError` inside a worker, hours
into a run.

## What is not covered

Registration, template building, segmentation and the QC dashboards are not
exercised — they need the external toolchain and real data. Treat these as smoke
checks, not a validation of scientific output. See `docs/STATUS.md`.
