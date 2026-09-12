# Provenance

Who wrote what, which version of each competing implementation was kept, and why.

The full comparison this is based on is in [MIGRATION_AUDIT.md](MIGRATION_AUDIT.md).

---

## Contributors

| Contributor | Contribution |
|---|---|
| **Samuel Marshall** | Original Nipype pipeline (March 2026) — the ancestor of everything in `workflow/scripts/`. Repository setup, LICENSE, initial README |
| **Dylan Dinesh** | The `docs/` set (March 2026); the modular refactor on `modular_branch`; repository maintenance and the PR merged into `main` |
| **rizviwazeen** | Requirements pinning and initial uploads |
| **Erfan Zarenia** | Pipeline development March–September 2026: container detection, the FireANTs template builders, the V4 drawing template, hi-res re-warping, the NMS and probabilistic atlas comparisons, core/rim masking, the group QC stage, per-subject dashboards, and the standalone tools |

Git history on `main` and the branches records the original authorship; this
reorganization did not rewrite it.

---

## Version decisions

Six generations of the main pipeline and four of the MNI template builder existed
in the working directory. Lineage was established by diffing consecutive
revisions; in every case the chain was strictly linear, with each version a
superset of its predecessor.

### Main pipeline — kept the July-15 revision

| Version | Date | Lines | Decision |
|---|---|---:|---|
| `Extra_Functions/NM_Nipype_Pipeline.py` | 2026-03-22 | ~400 | → `legacy/` |
| `BrainHack_NM_Nipype_Pipeline (copy).py` | 2026-03-29 | 1389 | dropped |
| `BrainHack_NM_NiPyPe_Pipeline_Backup.py` | 2026-04-15 | 1867 | dropped |
| `BrainHack_NM_Nipype_Pipeline.py` | 2026-05-08 | 2929 | dropped (was the version on `main`) |
| `Neuromel_Anchor_Pipeline.py` | 2026-07-09 | 3114 | dropped |
| `NeuroMel_Anchor_Pipeline_July-10th.py` | 2026-07-13 | 3049 | dropped |
| `NeuroMel_Anchor_Pipeline_July-15th.py` | 2026-09-11 | 3387 | **kept** → `neuromelanchor_pipeline.py` |

Consecutive diffs of 281, 233 and 460 changed lines, each strictly additive in
capability. The intermediate revisions carry no unique work and remain available
in Git history and in the working directory.

The March original is kept in `legacy/` because it is the clearest statement of
the method before the QC and template machinery was added — useful when reading
the current script.

### MNI template builder — kept V4

| Version | Date | Lines | Decision |
|---|---|---:|---|
| `Template_Builder.py` | 2026-06-10 | 687 | dropped (was the version on `main`) |
| `Template_Builder_v2.py` | 2026-07-14 | 766 | dropped |
| `Template_Builder_V3.py` | 2026-07-15 | 990 | dropped |
| `Template_Builder_V4.py` | 2026-07-22 | 1003 | **kept** → `template_builder.py` |

V3 → V4 is a 41-line delta, so V4 is a safe successor rather than a rewrite.

### Native-space builder — kept, marked experimental

`Template_Builder_Native_V1.py` (2026-07-23) differs from V4 by 394 lines and
takes a different approach: it builds on a chosen subject's native grid rather
than in MNI space. It is **not** a successor to V4 — the two are parallel
approaches to the same problem, and neither has been compared against the other
on the same cohort.

Both are kept. The native builder and its preparation script are marked
experimental in [STATUS.md](STATUS.md) rather than being deleted or promoted.

### BIDS renamer — merged

`Extra_Functions/NM_BioGen_ReName.py` and `Scripts/NM_BioGen_ReName.py` differed
by a single glob pattern (`*MP2RAGE_UNI_Images*.nii` vs `*LH.nii`) and one
whitespace line. Both were mostly commented-out variants of three renaming modes.

They were merged into `workflow/tools/bids_rename.py`, with the pattern and
output template as arguments so all three modes — and any future one — are
reachable without editing the file. The three original modes appear as worked
examples in its `--help`.

Two naming bugs in the dead commented code were **not** carried forward: a mode
labelled "NM contrasts" that emitted `_T1w` filenames, and an inversion-image
mode whose `{:2d}` format would have produced `acq-INV 1` with a space. The
generic template makes both unreachable.

---

## The modular refactor

`legacy/modular/` is Dylan Dinesh's `modular_branch` (March 2026): a deliberate,
behaviour-preserving split of the then-current monolith into `run_pipeline.py`,
`functions.py`, `phase1.py`, `segment.py` and `phase2.py`, with a `CHANGE_LOG.md`
documenting exactly what moved and what was deliberately not added.

It was **not** merged forward, and the reason is worth recording. It was written
against the March monolith. Since then the pipeline has gained container-based
template building, the V4 drawing template, hi-res re-warping, two additional
atlas comparisons, core/rim masking, a fourth stage and per-subject dashboards —
roughly 2000 lines with no counterpart in the split. Rebasing it would mean
redoing the refactor from scratch, not merging two versions.

It is preserved in full, with its change log, because the split is sound and is
the obvious reference if the current script is modularised. Deleting it would
have discarded real design work.

---

## Resources

`resources/traces_superseded_2026-09-10/` holds the three MNI-space manual
tracings that the pipeline used up to May 2026. They were retired in the working
directory on 2026-09-10, when tracing moved onto the V4 template and the
tracing's home became `<output_root>/manual_trace_v4/`.

They were moved, not deleted, so results produced before that date remain
reproducible. Nothing in the current pipeline reads them.

The NMS probabilistic SNc atlas is redistributed from the upstream `NMS-SNc-atlas`
release with its LICENSE and README alongside it.

---

## What could not be reconciled

Two masking methods — `bilateral_nm` and a hysteresis refinement — produced
outputs in April 2026 that survive on disk, but **no surviving source produces
them**, in any script, branch or commit. They were implemented, run, and
overwritten. Details and the surviving evidence are in
[STATUS.md](STATUS.md#two-masking-methods-whose-source-is-lost).
