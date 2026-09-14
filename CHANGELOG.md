# Changelog

## 2026-09-14: sync with the Nexus working directory (pipeline v3.0)

Between 2026-09-12 and 2026-09-14 the Nexus copy of the pipeline was edited and two
poster scripts were added. Everything else in Nexus that the repository draws on was
unchanged: migrated files were compared by content and the rest by size and timestamp.
Nexus was again treated as read-only.

### Merged

- **Pipeline v3.0**, from `NeuroMel_Anchor_Pipeline_July-15th.py` (which keeps its July
  name in Nexus):
  - `--search_dilation_mm` grows the propagated SN tracing by a set distance before the
    NM threshold is applied. The grown region is written as
    `anat/Native_SN_Search_Region.nii.gz` and is used by both the NM mask and the
    core-rim mask. The default of 0 keeps the tracing as drawn.
  - The NM mask is now morphologically closed (3 x 3 x 3) and intersected with the search
    region. **This changes NM masks at default settings**, so v3.0 and v2.5.0 results are
    not directly comparable.
  - Group QC writes poster figures (`Poster_Accuracy`, `Poster_Contours_*`,
    `Poster_Strip_*` and `Poster_CNR_Histogram_*`) and an NM-blind control
    (`Poster_NM_Blind_Control`) when manual masks are available.
  - The run log also records `--nm_input`, `--nm_echo`, `--hires`, `--drawing_template`,
    `--rebuild_template`, `--search_dilation_mm`, the script path and the full command
    line.
  - The module docstring gained the authors' note on limitations: in practice the
    percentile, not the floor, sets the threshold, and the measure is a contrast ratio
    rather than a contrast-to-noise ratio.
- **Figure scripts** `workflow/figures/poster_workflow_panels.py` and
  `workflow/figures/poster_dilation_sweep.py`, from `BrainHack_Poster_Workflow_Methods.py`
  and `Posters_Trajectory_Plots.py`.

### Cleaned

- The six transforms from the 2026-09-11 reorganization were re-applied to the new
  pipeline file, so the commit that brought v3.0 in contains only the upstream change.
- The poster scripts' hard-coded paths and settings became command-line options. Run on
  the same data with outputs redirected, the originals and the converted scripts produce
  pixel-identical figures and identical printed tables.
- Poster figures were being written to PNG twice; each is now written once as PNG and
  once as SVG, the same files as before.
- Three unused pipeline helpers were removed: `combine_prealign_transforms`,
  `combine_downstream_inverse` and `binarize_mask`. They remain in the Nexus copy.
- `Poster_Accuracy` no longer aborts group QC when a paired test sees only zero
  differences, which tied HD95 values can cause in a small group such as a `--test` run.
  Such comparisons are reported as not significant.
- One-line docstrings and short comments were added where the code was unclear, the
  stale script name in the pipeline's usage text was fixed, and typos in the authors'
  limitations note were corrected.
- Dashes in several QC flag messages, figure titles and log messages were replaced with
  colons, semicolons or commas.

### Corrected

- `resources/traces_superseded_2026-09-10/` held the repository's March tracings rather
  than the July tracings actually retired on 2026-09-10. It now holds the July files,
  byte-identical to Nexus, and the March files are in
  `resources/traces_github_2026-03-29/`.
- `docs/DATA_SPECIFICATION.md` and `docs/EVALUATION.md` described outputs of an older run.
  Disagreement maps, `Motion_Params_Plot.png`, `Run_Difference_QC.png` and
  `CIT168_Anchored.nii.gz` are not written by the current code, and
  `transform_Warped.nii.gz`, which is the anchored atlas, was described as a T1 image.
  Both documents were rebuilt from the code's DataSink routes and a complete v3.0 output
  directory.
- `docs/WORKFLOW.md` said the percentile threshold protects low-contrast subjects from
  under-segmentation; the authors' note says the opposite.
- `docs/PROVENANCE.md` misattributed the development since Brainhack. Most of it is the
  work of Sam Marshall (@smarsh127), with input from Dr. Bruce Morton.

### Added

- Tests for the search region, the closing step, the NM-blind control, the tie guard in
  `Poster_Accuracy`, the figure scripts' command lines and helpers, and the file names and
  CSV columns the figure scripts rely on.
- `workflow/figures/README.md`, and a sweep recipe in `docs/USAGE.md`.

---

## 2026-09-11: repository reorganization

Merged roughly four months of pipeline development from the Nexus working directory
(`/scratch/Projects/BrainHack_Nexus`) into the repository, consolidated the scattered
scripts, and rewrote the documentation. Nexus was treated as strictly read-only. The
audit behind these changes is in [docs/MIGRATION_AUDIT.md](docs/MIGRATION_AUDIT.md);
version decisions are in [docs/PROVENANCE.md](docs/PROVENANCE.md).

### Merged from Nexus

- **Pipeline** advanced from the May revision (2929 lines) to the July-15 revision (3387
  lines, v2.5.0), renamed `workflow/scripts/neuromelanchor_pipeline.py`. New since the
  committed version: container detection for FireANTs (Singularity, Apptainer or
  Docker), the V4 midbrain drawing template, `--hires` 0.5 mm re-warping, NMS
  probabilistic atlas QC, core-rim NM masking, `--nm_input combo|echo`,
  `--drawing_template v4|naive`, `--rebuild_template`, a `group_qc` stage, per-subject QC
  dashboards and a Stage 2 output manifest.
- **Template builders:** `template_builder.py` (was `Template_Builder_V4.py`), plus the
  experimental `template_builder_native.py` and `native_template_prep.py`.
- **Resources:** `HarvardOxford/Midbrain.nii.gz`, `HarvardOxford/Brainstem_Midbrain.nii.gz`,
  `NMS_prob/NMS_Snc_prob.nii.gz` with its upstream licence, and the 1 mm MNI T1 reference.
- **Seven standalone utilities** previously spread across `Extra_Functions/`, `Scripts/`
  and `workflow/scripts/`.

### Cleaned

- Six superseded pipeline revisions and three superseded template builders were
  removed; the March original and the `modular_branch` refactor are preserved under
  `legacy/`.
- The two near-identical `NM_BioGen_ReName.py` copies were merged into
  `workflow/tools/bids_rename.py`, with the glob and output name as arguments. It is a dry
  run by default. Two naming bugs in its dead commented-out code were not carried forward.
- `Rename.py` was dropped: its subject loop had an empty body.
- Site-specific `/scratch/Projects/BrainHack_Nexus/...` paths were removed from nine
  scripts. Resource defaults now resolve from each script's own location and output paths
  are required arguments; a test enforces this.
- Cohort filename patterns moved out of commented-out code blocks into a
  `COHORT_INPUT_PATTERNS` table, reachable with `--cohort`, `--nm_pattern` and
  `--t1_pattern`. The BioGen and SNIT patterns are unchanged.
- `Automated_Setup.itksnap` is no longer tracked, because it is generated on every run and
  embeds run-specific absolute paths. It is now written to `<output_root>/` instead of
  back into the repository.
- The duplicate `requirements.txt` files became one at the repository root, with
  `SimpleITK==2.5.5` and `scikit-image==0.26.0` added, since both were imported but not
  pinned.
- Retired MNI manual tracings moved into their own `resources/` directory with a note on
  what replaced them (corrected on 2026-09-14; see above).
- A `.gitignore` excludes bytecode, environments, Nipype work directories and NIfTI
  volumes, with `resources/` as the deliberate exception.
- `resample_mni_reference.py` imports TemplateFlow only when needed, so `--help` works
  without it.

### Added

- `tests/`: 127 checks at the time, needing no external neuroimaging software and no
  imaging data.
- `config/example_cohort.yaml`: cohort settings in one readable place. Reference only;
  nothing reads it.
- `docs/REPOSITORY_STRUCTURE.md`, `docs/STATUS.md`, `docs/PROVENANCE.md`,
  `docs/MIGRATION_AUDIT.md`, and README files for `resources/`, `workflow/tools/`,
  `legacy/` and `tests/`.

### Documentation

All six existing documents were rewritten. They described the March and May pipeline:
three stages where the code has four, entry points (`pipeline.py`, `run_pipeline.py`)
that do not exist, and illustrative rather than real file names.

### Unresolved

Output directories in Nexus contain `Native_Bilateral_NM_Mask.nii.gz` and
`Native_Refined_NM_Hysteresis.nii.gz` from April 2026, with parameter files describing a
`bilateral_nm` method. No surviving source produces them: not in Nexus, not on any
branch, not in Git history. See
[docs/STATUS.md](docs/STATUS.md#two-masking-methods-whose-source-is-lost).

---

## Earlier history

NeuroMelAnchor was started at Brainhack Western 2026, where it won the Excellence in
Innovation award, and has since been developed substantially, mostly by Sam Marshall
(@smarsh127), with input from Dr. Bruce Morton. Before the reorganization, the
repository's `main` branch last advanced on 2026-05-08; see the Git log for earlier
changes.
