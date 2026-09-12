# Changelog

## September 2026 — repository reorganization

Merged roughly four months of pipeline development from the working directory at
`/scratch/Projects/BrainHack_Nexus` into the repository, consolidated the
scattered scripts, and rewrote the documentation.

The working directory was treated as strictly read-only throughout. The audit
behind these changes is in [docs/MIGRATION_AUDIT.md](docs/MIGRATION_AUDIT.md);
version decisions are in [docs/PROVENANCE.md](docs/PROVENANCE.md).

### Merged from the working directory

- **Pipeline** advanced from the May revision (2929 lines) to the July-15
  revision (3387 lines), renamed `workflow/scripts/neuromelanchor_pipeline.py`.
  New since the committed version: container runtime detection for FireANTs
  (Singularity / Apptainer / Docker), the V4 midbrain drawing template workflow,
  `--hires` 0.5 mm re-warping, NMS probabilistic atlas QC, core/rim NM masking,
  `--nm_input combo|echo`, `--drawing_template v4|naive`, `--rebuild_template`,
  a `group_qc` stage, per-subject QC dashboards and a Stage 2 output manifest.
- **Template builders**: `template_builder.py` (was `Template_Builder_V4.py`),
  plus the experimental `template_builder_native.py` and `native_template_prep.py`.
- **Resources**: `HarvardOxford/Midbrain.nii.gz`,
  `HarvardOxford/Brainstem_Midbrain.nii.gz`, `NMS_prob/NMS_Snc_prob.nii.gz` with
  its upstream LICENSE, and the 1 mm MNI T1 reference.
- **Seven standalone utilities** previously spread across `Extra_Functions/`,
  `Scripts/` and `workflow/scripts/`.

### Cleaned

- Six superseded pipeline revisions and three superseded template builders
  removed; the March original and the `modular_branch` refactor preserved under
  `legacy/` with their provenance.
- The two near-identical `NM_BioGen_ReName.py` copies merged into
  `workflow/tools/bids_rename.py`, with the glob and output name as arguments.
  It is a dry run by default. Two naming bugs in its dead commented code were not
  carried forward.
- `Rename.py` dropped — its subject loop had an empty body.
- Site-specific `/scratch/Projects/BrainHack_Nexus/...` paths removed from nine
  scripts. Resource defaults now resolve from the script's own location; output
  paths are required arguments. A test enforces this.
- Cohort filename patterns moved out of commented-out code blocks into a
  `COHORT_INPUT_PATTERNS` table, reachable via `--cohort`, `--nm_pattern` and
  `--t1_pattern`. The BioGen and SNIT patterns are preserved verbatim.
- `Automated_Setup.itksnap` untracked — it is generated per run and embeds
  run-specific absolute paths. It is now written to `<output_root>/` instead of
  back into the repository.
- Duplicate `requirements.txt` consolidated to one at the repository root, with
  `SimpleITK==2.5.5` and `scikit-image==0.26.0` added — both were imported but
  unpinned.
- Superseded MNI manual tracings moved to
  `resources/traces_superseded_2026-09-10/` with a note explaining what replaced
  them.
- `.gitignore` added: excludes bytecode, environments, Nipype work directories
  and NIfTI volumes, with `resources/` as the deliberate exception.
- `resample_mni_reference.py` imports TemplateFlow lazily so `--help` works
  without it.

### Added

- `tests/` — 127 checks needing only numpy, nibabel, scipy, matplotlib and
  pytest. No ANTs, SPM, ITK-SNAP, GPU or imaging data. Covers repository layout
  and the atlas files the pipeline hard-codes, parsing of every script, the
  pipeline's CLI contract, each tool's `--help`, and the numeric behaviour of
  seven mask and QC helpers. Includes a check that functions passed to Nipype
  `Function` nodes are self-contained — a module-level import a node body relies
  on would otherwise fail inside a worker, hours into a run.
- `config/example_cohort.yaml` — cohort settings in one readable place, with the
  command they describe. Reference only; nothing reads it.
- `docs/REPOSITORY_STRUCTURE.md`, `docs/STATUS.md`, `docs/PROVENANCE.md`,
  `docs/MIGRATION_AUDIT.md`, `resources/README.md`, `workflow/tools/README.md`,
  `legacy/README.md`, `tests/README.md`.

### Documentation

All six existing documents were rewritten. They had described the March/May
pipeline: three stages where the code has four, entrypoints (`pipeline.py`,
`run_pipeline.py`) that do not exist, and illustrative filenames rather than the
ones actually produced. Output listings are now taken from real run output.

`docs/STATUS.md` is new and records what is implemented, what is experimental,
what is unfinished, and fifteen known limitations — including that no output has
been validated against an independent ground truth.

### Unresolved

Output directories contain `Native_Bilateral_NM_Mask.nii.gz` and
`Native_Refined_NM_Hysteresis.nii.gz` from April 2026, with parameter JSONs
describing a `bilateral_nm` method. No surviving source produces them — not in
the working directory, not on any branch, not in Git history. Two masking methods
were implemented, run, and overwritten. The outputs and their parameters survive
and would support reconstruction. Nothing in the current pipeline depends on them.

---

## Earlier history

See the Git log. The repository was created in March 2026 during Brainhack
Western; `main` last advanced on 2026-05-08.
