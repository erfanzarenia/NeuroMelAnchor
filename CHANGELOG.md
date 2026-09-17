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
- core-rim masking, and comparisons against the CIT168 probabilistic and NMS probabilistic
  atlases
- a `group_qc` stage with per-subject dashboards, an output manifest, group plots, poster
  figures, an NM-blind control and a statistics table
- `--nm_input combo|echo`, `--drawing_template v4|naive`, and `--dataset` for datasets whose
  file names differ
- the run log records the settings, the script path and the full command line

**Results from v3.0 are not directly comparable with earlier ones**, because the closing step
changes the NM mask even at default settings.

### Other changes in behaviour

- `Automated_Setup.itksnap` is written to `<output_root>/` instead of into the repository
- three unused helpers were removed from the pipeline (`combine_prealign_transforms`,
  `combine_downstream_inverse` and `binarize_mask`)
- the poster accuracy figure no longer stops group QC when a paired test sees only tied values
- the wording of some QC flag messages and figure titles changed

### Structure

| Before | Now |
|---|---|
| `workflow/scripts/BrainHack_NM_Nipype_Pipeline.py` | `workflow/scripts/neuromelanchor_pipeline.py` |
| `Template_Builder.py`, and its v2, V3 and V4 revisions | `workflow/scripts/template_builder.py` |
| `Template_Builder_Native_V1.py`, `Native_Template_Builder_Preparation.py` | `workflow/scripts/template_builder_native.py` and `native_template_prep.py`, both experimental |
| loose scripts in `Extra_Functions/` and `Scripts/` | `workflow/tools/`, each with a command line |
| poster figure scripts | `workflow/figures/` |
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
