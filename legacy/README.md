# Legacy implementations

Kept for reference. **Nothing here is maintained or run.** The current pipeline is
`workflow/scripts/neuromelanchor_pipeline.py`.

## `NM_Nipype_Pipeline.py`

The original Nipype pipeline (Sam Marshall, March 2026): realign, bias correct, brain extract,
register T1 to MNI with a brainstem weighting mask, register NM to T1, compose the transforms,
smooth. Every later version descends from it, and the current script still has the same Stage 1
node layout under different names.

## `modular/`

A behaviour-preserving split of the March pipeline into `run_pipeline.py`, `functions.py`,
`phase1.py`, `segment.py` and `phase2.py` (Dylan Dinesh, March 2026), with a change log of what
moved. It was written against the March version and cannot be merged into the current script,
but the split itself is sound and is the obvious starting point if the pipeline is ever
modularised. See [../docs/STATUS.md](../docs/STATUS.md).
