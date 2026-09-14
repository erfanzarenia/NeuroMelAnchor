# Provenance

Project history, who contributed what, and which version of each competing
implementation was kept and why. The comparisons behind these decisions are in
[MIGRATION_AUDIT.md](MIGRATION_AUDIT.md).

---

## History

NeuroMelAnchor was started at Brainhack Western 2026, where it won the Excellence in
Innovation award. It has since been developed substantially, mostly by Sam Marshall
(@smarsh127), with input from Dr. Bruce Morton.

After Brainhack, development moved to a shared working directory at
`/scratch/Projects/BrainHack_Nexus` (called Nexus below), while the GitHub repository
last advanced on 2026-05-08. The September 2026 reorganization brought the two back
together in two passes: an audit and merge on 2026-09-11, and a sync on 2026-09-14
that brought in pipeline v3.0.

## Contributors

| Contributor | Contribution |
|---|---|
| **Sam Marshall** (@smarsh127) | The original Nipype pipeline at Brainhack, and most of the pipeline's development since then |
| **Dr. Bruce Morton** | Input throughout the project's development |
| **Dylan Dinesh** | The first documentation set (March 2026), the modular refactor on `modular_branch`, and repository maintenance, including the pull request merged into `main` |
| **rizviwazeen** | Pinning the original Python environment, and early uploads |
| **Erfan Zarenia** | Repository owner; the September 2026 reorganization and sync |

The Git history on `main` and the other branches records the original authorship. The
reorganization did not rewrite it.

---

## Version decisions

Six generations of the main pipeline and four of the MNI template builder existed in
Nexus. Lineage was established by diffing consecutive revisions, and in every case the
chain was strictly linear, each version extending its predecessor.

### Main pipeline: the July-15 revision

| Version | Date | Lines | Decision |
|---|---|---:|---|
| `Extra_Functions/NM_Nipype_Pipeline.py` | 2026-03-22 | ~400 | kept in `legacy/` |
| `BrainHack_NM_Nipype_Pipeline (copy).py` | 2026-03-29 | 1389 | dropped |
| `BrainHack_NM_NiPyPe_Pipeline_Backup.py` | 2026-04-15 | 1867 | dropped |
| `BrainHack_NM_Nipype_Pipeline.py` | 2026-05-08 | 2929 | dropped (was the version on `main`) |
| `Neuromel_Anchor_Pipeline.py` | 2026-07-09 | 3114 | dropped |
| `NeuroMel_Anchor_Pipeline_July-10th.py` | 2026-07-13 | 3049 | dropped |
| `NeuroMel_Anchor_Pipeline_July-15th.py` | 2026-09-11, then 2026-09-14 | 3387, then 3910 | **kept** as `neuromelanchor_pipeline.py` |

Consecutive diffs were 281, 233 and 460 changed lines, each adding capability. The
intermediate revisions carry no unique work and remain available in Git history and in
Nexus. The March original is kept in `legacy/` because it is the clearest statement of
the method before the QC and template machinery was added.

### MNI template builder: V4

| Version | Date | Lines | Decision |
|---|---|---:|---|
| `Template_Builder.py` | 2026-06-10 | 687 | dropped (was the version on `main`) |
| `Template_Builder_v2.py` | 2026-07-14 | 766 | dropped |
| `Template_Builder_V3.py` | 2026-07-15 | 990 | dropped |
| `Template_Builder_V4.py` | 2026-07-22 | 1003 | **kept** as `template_builder.py` |

V3 to V4 is a 41-line change, so V4 is a safe successor rather than a rewrite.

### Native-space builder: kept, marked experimental

`Template_Builder_Native_V1.py` (2026-07-23) differs from V4 by 394 lines and takes a
different approach, building on a chosen subject's native grid rather than in MNI
space. It is **not** a successor to V4: the two are parallel approaches that have never
been compared on the same cohort. Both are kept, and the native builder and its
preparation script are marked experimental in [STATUS.md](STATUS.md).

### BIDS renamer: merged

`Extra_Functions/NM_BioGen_ReName.py` and `Scripts/NM_BioGen_ReName.py` differed by one
glob pattern and one whitespace line, and both were mostly commented-out variants of
three renaming modes. They were merged into `workflow/tools/bids_rename.py`, with the
pattern and output name as arguments, so all three modes are reachable without editing
the file; the original modes appear as examples in its `--help`.

Two naming bugs in the dead code were **not** carried forward: a mode labelled "NM
contrasts" that wrote `_T1w` file names, and an inversion-image mode whose `{:2d}`
format would have produced `acq-INV 1` with a space.

### Poster figure scripts: converted, output unchanged

`BrainHack_Poster_Workflow_Methods.py` and `Posters_Trajectory_Plots.py` appeared in
Nexus between the two passes. They were committed unchanged, then converted to
command-line tools as `workflow/figures/poster_workflow_panels.py` and
`workflow/figures/poster_dilation_sweep.py`. Run on the same data with their outputs
redirected, the originals and the converted scripts produce pixel-identical figures
and identical printed tables.

---

## The September 14 sync

Between the two passes, the Nexus pipeline was edited from 2026-09-12 to 2026-09-14
(still named `NeuroMel_Anchor_Pipeline_July-15th.py`, but now v3.0) and the two poster
scripts appeared beside it. Nothing else the repository draws on had changed: migrated
files were compared by content, and the rest by size and timestamp.

The files were still being saved during the sync, so they were first copied to a
snapshot whose checksums were checked against the live files. The six transforms the
September 11 reorganization applies to the pipeline (the cohort pattern table and its
options and log lines, the generated workspace path, and the renamed template builder)
were re-applied to the new file, so the commit that brought v3.0 in contains exactly
the upstream change: 529 lines added and 6 removed. Cleanup followed in separate
commits.

One file changed again while the sync was under way: at 16:51,
`BrainHack_Poster_Workflow_Methods.py` stopped padding its panels with a white border.
That edit was ported into `workflow/figures/poster_workflow_panels.py` in a follow-up
commit and checked the same way, pixel for pixel against the edited original.

---

## The modular refactor

`legacy/modular/` is Dylan Dinesh's `modular_branch` (March 2026): a deliberate,
behaviour-preserving split of the then-current monolith into `run_pipeline.py`,
`functions.py`, `phase1.py`, `segment.py` and `phase2.py`, with a `CHANGE_LOG.md` that
records exactly what moved and what was deliberately not added.

It was **not** merged forward because it was written against the March monolith. Since
then the pipeline has gained container-based template building, the V4 drawing
template, hi-res re-warping, two more atlas comparisons, core-rim masking, a fourth
stage, per-subject dashboards, search-region dilation and the poster figures, more
than 2,000 lines with no counterpart in the split. Rebasing it would mean redoing the
refactor, not merging two versions. It is preserved in full because the split is sound
and is the obvious reference for any future modularization.

---

## Resources

Three generations of MNI-space manual tracings exist:

| Set | Where | Status |
|---|---|---|
| Dated 2026-03-18 | Nexus `Masks/MNI/` | Never in the repository; not migrated |
| Uploaded 2026-03-29 by Sam Marshall | `resources/traces_github_2026-03-29/` | Tracked on `main` until September 2026 |
| Dated 2026-07-09 | `resources/traces_superseded_2026-09-10/` | Read by the pipeline until retired on 2026-09-10 |

The September 11 reorganization mistakenly put the March upload into
`traces_superseded_2026-09-10/`. The sync on 2026-09-14 corrected this: that directory
now holds the July files, byte-identical to Nexus, and the March files have their own
directory. Nothing in the current pipeline reads either set; the current tracing lives
in each output root's `manual_trace_v4/`.

The NMS probabilistic SNc atlas is redistributed from the upstream `NMS-SNc-atlas`
release with its licence and README alongside it.

---

## What could not be reconciled

Two masking methods, `bilateral_nm` and a hysteresis refinement, produced outputs in
April 2026 that survive on disk, but **no surviving source produces them** in any
script, branch or commit. Details and the surviving evidence are in
[STATUS.md](STATUS.md#two-masking-methods-whose-source-is-lost).
