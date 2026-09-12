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
│   ├── scripts/
│   ├── tools/
│   └── workspace/
├── resources/
│   ├── tpl-MNI152NLin2009cAsym/
│   └── traces_superseded_2026-09-10/
├── legacy/
└── tests/
```

---

## `workflow/scripts/` — the pipeline

| File | Lines | Status | Purpose |
|---|---:|---|---|
| `neuromelanchor_pipeline.py` | 3429 | **canonical** | The whole pipeline. All four stages, driven by `--stage` |
| `template_builder.py` | 1004 | **canonical** | FireANTs iterative template construction in MNI space. Invoked by `--stage segment`; runnable directly |
| `template_builder_native.py` | 1164 | experimental | Same idea on a chosen subject's native grid. Not called by the pipeline |
| `native_template_prep.py` | 354 | experimental | Prepares per-subject inputs for the native builder |

The pipeline is one long script rather than a package. That is a deliberate
consequence of how Nipype works — functions passed to `Function` nodes are
re-serialised by source and re-executed in bare worker processes, so they must
import everything they use inside their own body. See
[STATUS.md](STATUS.md) for the modularization question.

## `workflow/tools/` — standalone utilities

Nothing here is called by the pipeline. Each has its own `--help`; see
[../workflow/tools/README.md](../workflow/tools/README.md).

| File | Purpose |
|---|---|
| `resample_mni_reference.py` | Writes the 1 mm and 0.5 mm MNI brain references into `resources/` |
| `make_brainstem_mask.py` | Rebuilds the Stage 1 brainstem weighting mask |
| `make_midbrain_prior.py` | Rebuilds the midbrain crop prior from BrainstemNavigator |
| `bids_rename.py` | Renames scanner exports to BIDS-style names. Dry run by default |
| `mp2rage_uniden.py` | Third-party MP2RAGE UNI denoising, kept verbatim |
| `mp2rage_uniden_batch.sh` | Runs the above over a BIDS directory |
| `fit_r2star.py` | R2*/T2* fitting from multi-echo GRE. Unrelated to the pipeline |

## `workflow/workspace/`

`NEXUS_NM_Template_ITKsnap.xml` — the ITK-SNAP workspace template used by
`--drawing_template naive`. Placeholders (`PROJECT_ROOT`, `TARGET_T1_PATH`,
`TARGET_NM_PATH`, `TARGET_MASK_PATH`) are substituted at run time and the result
is written to `<output_root>/Automated_Setup.itksnap`, not back into the
repository.

## `resources/`

Atlas priors and MNI references, resolved from `--project_root`. Full table in
[../resources/README.md](../resources/README.md), including the one file you must
generate after cloning.

`traces_superseded_2026-09-10/` holds the MNI-space manual tracings used up to
May 2026, retired when tracing moved onto the V4 template. Nothing reads them.

## `config/`

`example_cohort.yaml` documents the per-cohort settings as a single readable
file. **It is reference material, not a configuration file the pipeline loads** —
the pipeline is driven entirely by command-line options. See
[STATUS.md](STATUS.md).

## `legacy/`

Superseded implementations, kept for provenance and not maintained:
the original March pipeline, and the `modular_branch` refactor.
See [../legacy/README.md](../legacy/README.md) and [PROVENANCE.md](PROVENANCE.md).

## `tests/`

Smoke checks and unit tests that need no external software and no data. See
[../tests/README.md](../tests/README.md).

## `docs/`

| File | Contents |
|---|---|
| `INSTALL.md` | Environment and external software |
| `DATA_SPECIFICATION.md` | Inputs and every output produced |
| `USAGE.md` | Running each stage; full option reference |
| `WORKFLOW.md` | What each step does and why |
| `MANUAL_SEGMENTATION.md` | ITK-SNAP tracing protocol |
| `EVALUATION.md` | QC metrics and how to read them |
| `REPOSITORY_STRUCTURE.md` | This file |
| `STATUS.md` | Implemented vs unfinished, known limitations |
| `PROVENANCE.md` | Authorship and version decisions |
| `MIGRATION_AUDIT.md` | The audit behind the September 2026 reorganization |

---

## What is deliberately absent

Imaging data, derivatives, Nipype work directories, the FireANTs container image
and virtual environments are **not** tracked. `.gitignore` excludes `*.nii`
and `*.nii.gz` everywhere except under `resources/`, which is the one place
volumes are tracked on purpose.

The working data for this project lives at `/scratch/Projects/BrainHack_Nexus`
and is not part of the repository.
