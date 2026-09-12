# Status

What works, what does not, and what is known to be uncertain.
**Read this before relying on any output.**

Current as of the September 2026 reorganization.

---

## Implemented and in routine use

- **Stage 1 — forward normalization.** Realignment, N4 bias correction,
  averaging, brain extraction, brainstem-weighted T1→MNI and rigid NM→T1
  registration, transform composition, optional smoothing, registration and
  motion QC, group-average template. Run on several cohorts.
- **Stage segment.** FireANTs V4 midbrain drawing template, alignment to the
  group average, ITK-SNAP launch, label splitting and validation. Both the `v4`
  and `naive` drawing paths work.
- **Stage 2 — inverse normalization and CNR seeding.** Subject-to-template
  registration, inverse warping of SN/CP, CNR map, NM seed, core/rim variant,
  probabilistic atlas warping and thresholding, atlas anchoring, per-subject QC
  and dashboards.
- **Group QC.** Raincloud, dot, deviation, method comparison, coupling and
  boundary/core plots, plus the two summary CSVs.
- **`--hires`** 0.5 mm re-warping.
- **`--test`** single-subject mode with `_TEST`-suffixed group outputs.
- **Standalone tools** in `workflow/tools/`, all with working CLIs.

## Implemented but lightly exercised

- **`--cohort snit`.** The SNIT filename patterns are carried over from
  commented-out code that was switched in by hand. The patterns are preserved
  verbatim but have not been run since being made configurable.
- **`--nm_input echo`.** Single-echo input works; `combo` is the better-tested
  path and has higher SNR.
- **`--nm_image_n`.** Averaging a subset of acquisitions is implemented and
  validated by unit tests, but has not been used in a full analysis.
- **`--smooth`.** Plumbed through SPM, but every recent run used `--smooth 0`.

## Experimental — not part of the pipeline

- **`template_builder_native.py` and `native_template_prep.py`.** A parallel
  approach that builds the template on a chosen subject's native grid instead of
  in MNI space, avoiding one resampling of the subject data. Runnable, but never
  wired into the pipeline and never compared against the MNI-space builder on the
  same cohort. Whether it is better is an open question, not a settled one.

## Not implemented

- **No configuration file.** `config/example_cohort.yaml` documents cohort
  settings but nothing reads it. All settings are command-line options or
  module-level constants.
- **No VTA segmentation.** The project's stated goal includes the ventral
  tegmental area; only SN is implemented. The CIT168 probabilistic resources for
  VTA are present but unused.
- **No packaging.** No `setup.py`/`pyproject.toml`, no console entry points.
  Scripts are run by path.
- **No CI.** The checks in `tests/` are not run automatically.
- **No containerization of the pipeline itself.** Only the FireANTs builder runs
  in a container; the pipeline needs ANTs, SPM/MATLAB and ITK-SNAP on the host.

---

## Known issues and limitations

### Scientific

1. **Nothing is validated against an independent ground truth.** Every metric is
   agreement between methods, or against manual tracings by the same group.
   Agreement is not accuracy.
2. **Masking constants are not CLI options.** `seed_cnr_floor` (0.05),
   `seed_cnr_percentile` (55), `seed_min_cluster_voxels` (15),
   `seed_components_to_keep` (2), `core_rim_erosion_mm` (0.5),
   `core_rim_core_cnr_threshold` (0.05), `core_rim_rim_cnr_threshold` (0.10),
   `core_rim_min_cluster_voxels` (25) and `core_rim_components_to_keep` (2) are
   module-level constants. They **are** echoed into the run log, so a result can
   always be traced back to the values used, but a sweep means editing the script.
3. **Several constants are acknowledged as arbitrary in the source.** The core
   erosion distance and the minimum cluster size carry comments saying as much.
   Sensitivity to them has not been characterised.
4. **The CNR percentile is subject-adaptive, the floor is not.** This is
   deliberate — it prevents systematic under-segmentation of low-contrast
   subjects — but it means the effective threshold differs between subjects, which
   matters when comparing mask volumes across groups.
5. **`seed_components_to_keep = 2` assumes bilateral SN.** A subject whose two
   sides merge into one component, or where one side is missed, is silently
   handled by the component filter. The component counts in
   `Group_QC_Component_Summary.csv` are the check for this.
6. **The tracing is a single rater's, drawn once.** No inter-rater or
   intra-rater reliability has been assessed. Every subject-level result inherits
   the shape of that one tracing.
7. **The V4→group-average alignment is a rigid shift of about 1.3 mm.** It
   assumes the two templates differ only by translation. This holds for the
   cohorts run so far; it has not been verified to hold generally.

### Technical

8. **The pipeline is a 3400-line script.** It parses arguments at import time, so
   it cannot be imported or unit-tested conventionally. The tests work around
   this by lifting functions out with `ast`.
9. **`--output_root` must sit inside `--project_root`** when using the V4 drawing
   template, because the container binds only `--project_root`.
10. **Stage `segment` is interactive** and cannot run unattended.
11. **`--rebuild_template` and an existing tracing are mutually exclusive.** By
    design — a rebuilt template invalidates the tracing — but it means rebuilding
    is a manual, careful operation.
12. **Subject discovery is a bare `sub-*` glob.** No BIDS validation; a stray
    directory matching the prefix is treated as a subject.
13. **No resume-from-failure beyond Nipype's cache.** A failed subject is
    re-run from its last cached node, which is usually adequate but depends on an
    intact `--work_dir`.

### Reproducibility

14. **`requirements.txt` is a pin of one working environment**, not a tested
    minimum. `SimpleITK` and `scikit-image` were missing from it until this
    reorganization and were added at the versions in use.
15. **External tool versions are unrecorded.** ANTs, SPM12 and ITK-SNAP versions
    are not captured in the run log, so results are not fully reproducible from
    the repository alone. Worth fixing.

---

## Unresolved from the September 2026 reorganization

### Two masking methods whose source is lost

Output directories contain `Native_Bilateral_NM_Mask.nii.gz` (April 19) and
`Native_Refined_NM_Hysteresis.nii.gz` (April 16), with matching
`Bilateral_NM_Summary.json` and `Refined_NM_Hysteresis_Summary.json`. The JSON
records a `"method": "bilateral_nm"` with parameters (`split_x_mm`, per-side
component eligibility) that no surviving script produces.

**No code in the Nexus working directory, in any repository branch, or anywhere
in the Git history produces these files.** They are outputs of an April-era
pipeline revision that was overwritten. Two masking approaches — a bilateral
split with per-side component selection, and a hysteresis-based refinement —
were implemented, run, and lost.

The outputs and their parameter JSONs survive and are enough to reconstruct the
methods if either turns out to be worth having. Nothing in the current pipeline
depends on them.

### The modular refactor

`legacy/modular/` holds a clean, behaviour-preserving split of the March monolith
into stages. It was never carried forward, and roughly 2000 lines of pipeline
development have happened since, so it cannot be merged — only redone. It is kept
because the split itself is sound and is the obvious starting point if the
current script is ever modularised. See [PROVENANCE.md](PROVENANCE.md).

---

## Suggested next steps

1. Record ANTs, SPM and ITK-SNAP versions in the run log (item 15) — cheapest
   real improvement to reproducibility.
2. Promote the masking constants to CLI options with the current values as
   defaults (item 2). Backward-compatible, and makes sensitivity analysis possible.
3. Decide the fate of the native-space builder: compare it against the MNI-space
   builder on one cohort, then either wire it in or mark it abandoned.
4. Decide whether `bilateral_nm` or the hysteresis refinement is worth
   reconstructing from the surviving outputs.
5. Characterise sensitivity to the CNR floor and percentile.
6. Assess intra-rater reliability of the template tracing (item 6).
