# Repository Structure

```
NeuroMelAnchor/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── config/
│   └── example_cohort.yaml
├── docs/
├── workflow/
│   ├── scripts/      the pipeline and the template builders
│   ├── tools/        standalone utilities
│   ├── figures/      post-hoc figure scripts
│   └── workspace/    ITK-SNAP workspace template
├── resources/
│   ├── tpl-MNI152NLin2009cAsym/
│   └── retired_traces/
├── legacy/
└── tests/
```

---

## `workflow/scripts/`: the pipeline

| File | Status | Purpose |
|---|---|---|
| `neuromelanchor_pipeline.py` | **canonical** | The whole pipeline, about 3,900 lines: all four stages, selected with `--stage` |
| `template_builder.py` | **canonical** | FireANTs iterative template construction in MNI space. Called by `--stage segment`; can also be run directly |
| `template_builder_native.py` | experimental | The same approach on a chosen subject's native grid. Not called by the pipeline |
| `native_template_prep.py` | experimental | Prepares per-subject inputs for the native builder |

The pipeline is one long script rather than a package. Splitting it is possible but
needs care: functions passed to Nipype `Function` nodes are re-serialized from their
source and run in bare worker processes, so they must import everything they use
inside their own bodies. See [STATUS.md](STATUS.md).

## `workflow/tools/`: standalone utilities

The pipeline never calls these. Each has its own `--help`; see
[../workflow/tools/README.md](../workflow/tools/README.md).

| File | Purpose |
|---|---|
| `resample_mni_reference.py` | Writes the 1 mm and 0.5 mm MNI brain references into `resources/` |
| `make_brainstem_mask.py` | Rebuilds the Stage 1 brainstem weighting mask |
| `make_midbrain_prior.py` | Rebuilds the midbrain crop prior from BrainstemNavigator |
| `bids_rename.py` | Renames scanner exports to BIDS-style names; a dry run unless `--apply` is given |
| `mp2rage_uniden.py` | Third-party MP2RAGE UNI denoising, kept verbatim |
| `mp2rage_uniden_batch.sh` | Runs the above over a BIDS directory |
| `fit_r2star.py` | R2* and T2* fitting from multi-echo GRE; unrelated to the pipeline |

## `workflow/figures/`: figure scripts

Post-hoc scripts that read finished outputs; see
[../workflow/figures/README.md](../workflow/figures/README.md).

| File | Purpose |
|---|---|
| `poster_workflow_panels.py` | The method panels of the Brainhack Western 2026 poster |
| `poster_dilation_sweep.py` | Metrics of the NM mask against `--search_dilation_mm`, across a sweep |

The pipeline makes the group-level poster figures itself during group QC.

## `workflow/workspace/`

`NEXUS_NM_Template_ITKsnap.xml` is the ITK-SNAP workspace template used by
`--drawing_template naive`. Its placeholders (`PROJECT_ROOT`, `TARGET_T1_PATH`,
`TARGET_NM_PATH`, `TARGET_MASK_PATH`) are filled in at run time and the result is
written to `<output_root>/Automated_Setup.itksnap`, not back into the repository.

## `resources/`

Atlas priors and MNI references, resolved from `--project_root`. The full table,
including the one file you must generate after cloning, is in
[../resources/README.md](../resources/README.md).

Earlier MNI-space manual tracings are kept in `retired_traces/`, one directory per tracing
date. Nothing in the current workflow reads them.

## `config/`

`example_cohort.yaml` writes the per-cohort settings down in one readable file.
**The pipeline does not load it**; it is driven entirely by command-line options.

## `legacy/`

Superseded implementations kept for provenance and not maintained: the original March
2026 pipeline and the `modular_branch` refactor. See
[../legacy/README.md](../legacy/README.md).

## `tests/`

Smoke checks and unit tests that need no external software and no imaging data. See
[../tests/README.md](../tests/README.md).

## `docs/`

| File | Contents |
|---|---|
| `INSTALL.md` | Environment and external software |
| `DATA_SPECIFICATION.md` | Inputs and every output produced |
| `USAGE.md` | Running each stage, the option reference, and sweeps |
| `WORKFLOW.md` | What each step does and why |
| `MANUAL_SEGMENTATION.md` | The ITK-SNAP tracing protocol |
| `EVALUATION.md` | QC metrics, flags and poster figures |
| `REPOSITORY_STRUCTURE.md` | This file |
| `STATUS.md` | Implemented and unfinished work, and known limitations |

---

## What is deliberately absent

Imaging data, derivatives, Nipype work directories, the FireANTs container image and
virtual environments are **not** tracked. `.gitignore` excludes `*.nii` and `*.nii.gz`
everywhere except under `resources/`, the one place where volumes are tracked on
purpose.

The project's working data lives at `/scratch/Projects/BrainHack_Nexus` and is not part
of the repository.
