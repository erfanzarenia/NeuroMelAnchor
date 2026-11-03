# Changelog

## September 2026

Brings the repository up to date with the pipeline as it is actually run, and reorganizes it.
The previous state of `main` dated from 2026-05-08.

### Pipeline

`workflow/scripts/neuromelanchor_pipeline.py`, previously
`workflow/scripts/BrainHack_NM_Nipype_Pipeline.py`, moves from v2.5.0 to v3.0:

- FireANTs container detection, the V4 midbrain drawing template, and `--rebuild_template`
- `--hires` 0.5 mm re-warping
- `--search_dilation_mm`, which grows the propagated SN tracing before thresholding and writes
  `anat/Native_SN_Search_Region.nii.gz`
- the NM mask is morphologically closed and intersected with the search region
- the NM mask's thresholds, smoothing and component filter are command-line options
  (`--seed_cnr_floor`, `--seed_cnr_percentile`, `--seed_min_cluster_voxels`,
  `--seed_components_to_keep`, `--seed_cnr_fwhm`, `--seed_slice_fill`)
- `--seed_image raw` builds the mask from the run without bias correction, and `--trace_route t1`
  brings the traces to NM space through the stage 1 transforms instead of a stage 2 registration
- the NM mask is also written in the subject's T1 space (`anat_T1w/`), controlled by
  `--t1_interp`, `--t1_seed_res` and `--skip_t1_outputs`
- core-rim masking, and comparisons against the CIT168 probabilistic and NMS probabilistic
  atlases
- a `group_qc` stage with per-subject dashboards, an output manifest, group plots, accuracy
  and contour figures, an NM-blind control and a statistics table
- `--nm_input combo|echo`, `--drawing_template v4|naive`, and `--dataset` for datasets whose
  file names differ
- the run log records the settings, the script path and the full command line

**Results from v3.0 are not directly comparable with earlier ones**, because the closing step
changes the NM mask even at default settings.

### Other changes in behaviour

- `Automated_Setup.itksnap` is written to `<output_root>/` instead of into the repository
- two unused helpers were removed from the pipeline (`combine_downstream_inverse` and
  `binarize_mask`)
- group QC moved out of the pipeline into `workflow/qc/group_qc.py`, which the pipeline still
  runs after Stage 2 (skip with `--skip_group_qc`) and which also runs on its own; group files
  now go to `<output_root>/group_qc/`
- poster-only outputs were dropped (the single-slice contour strip, the single-subject
  histogram, and the transparent PNG and SVG copies); the evaluation figures and tables were
  renamed from `Poster_*` to `Group_QC_*` and are saved as PNG like the other group plots
- the accuracy figure no longer stops group QC when a paired test sees only tied values
- the wording of some QC flag messages and figure titles changed

### Structure

| Before | Now |
|---|---|
| `workflow/scripts/BrainHack_NM_Nipype_Pipeline.py` | `workflow/scripts/neuromelanchor_pipeline.py` |
| `Template_Builder.py`, and its v2, V3 and V4 revisions | `workflow/scripts/template_builder.py` |
| `Template_Builder_Native_V1.py`, `Native_Template_Builder_Preparation.py` | `workflow/scripts/template_builder_native.py` and `native_template_prep.py`, both experimental |
| loose scripts in `Extra_Functions/` and `Scripts/`, including the BioGen 3T preparation | `workflow/tools/`, each with a command line |
| mask comparison figure script | `workflow/figures/` |
| poster figure scripts | dropped |
| two `requirements.txt` files | one at the repository root |

- Superseded pipeline and builder revisions were dropped. The original March pipeline and the
  modular refactor are kept in `legacy/`.
- Site-specific absolute paths are gone: resource defaults resolve from each script's own
  location, and output paths are arguments.
- Added the midbrain, NMS probabilistic SNc and MNI references under `resources/`, and moved
  retired tracings to `resources/retired_traces/`.
- Added `tests/`, `config/example_cohort.yaml`, and a rewritten `docs/`.
- `SimpleITK` and `scikit-image` are now pinned; both were imported but missing from the
  requirements.
