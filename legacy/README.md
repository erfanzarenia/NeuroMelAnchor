# Legacy implementations

Preserved for provenance. **Nothing here is maintained or run.** The current
pipeline is `workflow/scripts/neuromelanchor_pipeline.py`.

## `NM_Nipype_Pipeline.py`

Sam Marshall (@smarsh127), March 2026. The original Nipype pipeline every later version
descends from: realign, N4 bias correct, brain extract, register T1→MNI with a
brainstem weighting mask, register NM→T1, compose the transforms, smooth.

It is the ancestor of the current script, which still carries the same Stage 1
node layout under different names. Kept because it is the clearest statement of
the original method, before the QC, template-building and atlas-comparison
machinery was added.

## `modular/`

Dylan Dinesh, March 2026, from the `modular_branch` branch on GitHub.

A deliberate, behaviour-preserving split of the then-current monolith into
`run_pipeline.py` (entrypoint and CLI), `functions.py` (helpers), `phase1.py`,
`segment.py` and `phase2.py` (stages). `CHANGE_LOG.md` documents exactly what was
moved and what was deliberately *not* added.

**Why it was not merged forward.** It was written against the March monolith.
The pipeline has since gained container-based template building, the V4 drawing
template, hi-res re-warping, two additional atlas comparison methods, core/rim
masking, a fourth stage and per-subject QC dashboards, more than 2,000 lines that
have no counterpart here. Rebasing the split onto the current script would mean
redoing it from scratch, not merging.

It is kept because the split itself is sound and is the obvious reference design
if and when the current script is modularised. See `docs/STATUS.md`.

## Intermediate revisions not kept

Six further intermediate versions existed in the Nexus working directory
(`BrainHack_NM_Nipype_Pipeline*` ×3, `Template_Builder{,_v2,_V3}.py`). They form a
strictly linear chain to the versions now in `workflow/scripts/`, carry no unique
work, and are recoverable from Git history and from
`/scratch/Projects/BrainHack_Nexus/workflow/scripts/`. See
`docs/MIGRATION_AUDIT.md` for the diff sizes that established the lineage.
