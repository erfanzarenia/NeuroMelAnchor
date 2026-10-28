# Status

What works, what does not, and what is known to be uncertain. **Read this before
relying on any output.**

Current as of pipeline v3.0 (2026-09-14).

---

## Implemented and in routine use

- **Stage 1: forward normalization.** Realignment, N4 bias correction, averaging,
  brain extraction, brainstem-weighted T1 to MNI and rigid NM to T1 registration,
  transform composition, optional smoothing, registration and motion QC, and the
  group-average template. Run on several cohorts.
- **Stage segment.** The FireANTs V4 midbrain drawing template, its alignment to the
  group average, the ITK-SNAP launch, and label splitting and validation. Both the
  `v4` and `naive` drawing paths work.
- **Stage 2: inverse normalization and NM masks.** Subject-to-template registration,
  inverse warping of SN and CP, the search region, the contrast map, the NM mask with
  closing, the core-rim variant, probabilistic atlas warping and thresholding, atlas
  anchoring, and pairwise scoring.
- **Group QC.** Per-subject dashboards and output manifests, the group plots and the
  two summary tables. With manual masks, also the poster figures and the NM-blind
  control.
- **`--hires`** 0.5 mm re-warping.
- **`--test`** single-subject mode with `_TEST`-suffixed group outputs.
- **Standalone tools** in `workflow/tools/` and **figure scripts** in
  `workflow/figures/`, all with working command lines.

## Implemented but lightly exercised

- **`--trace_route t1`, `--seed_image raw` and the T1w-space mask** (October 2026). Used for
  the October runs but not yet compared systematically against the defaults.
- **`--search_dilation_mm`** (new in v3.0). Evaluated on one cohort: BioGen 7T, echo
  1, 25 subjects, from 0 to 3 mm in 0.5 mm steps. The poster used 0.5 mm. The default
  stays 0, which keeps the propagated tracing as drawn.
- **`--dataset snit`.** The SNIT file naming is implemented but has not been run recently.
- **`--nm_input echo`.** Works, and the poster sweep used echo 1; `combo` has the
  higher SNR.
- **`--nm_image_n`.** Implemented and unit-tested, but not used in a full analysis.
- **`--smooth`.** Wired through SPM, but every recent run used `--smooth 0`.

## Experimental and not part of the pipeline

- **`template_builder_native.py` and `native_template_prep.py`.** A parallel approach
  that builds the template on a chosen subject's native grid instead of in MNI space,
  avoiding one resampling of the subject data. Runnable, but never wired into the
  pipeline and never compared with the MNI-space builder on the same cohort. Whether
  it is better is an open question.

## Not implemented

- **No configuration file.** `config/example_cohort.yaml` documents cohort settings,
  but nothing reads it. All settings are command-line options or module-level
  constants.
- **No VTA segmentation.** The project's original aims included the ventral tegmental
  area; only the SN is implemented. The CIT168 VTA resources are present but unused.
- **No packaging.** No `pyproject.toml` and no console entry points; scripts are run
  by path.
- **No CI.** The checks in `tests/` are not run automatically.
- **No container for the pipeline itself.** Only the FireANTs builder runs in a
  container; the pipeline needs ANTs, SPM with MATLAB, and ITK-SNAP on the host.

---

## Known issues and limitations

### Scientific

1. **Nothing is validated against an independent ground truth.** Every metric is
   agreement between methods, or with manual tracings made by the same group.
   Agreement is not accuracy.
2. **In practice the percentile sets the threshold, not the floor.** This is the
   authors' own note in the pipeline's docstring. The NM mask keeps voxels above the
   55th percentile of the search region's contrast values, so roughly the top 45% of
   the region survives thresholding in every subject, however strong or weak its NM
   contrast. The 0.05 floor is too low ever to change that. Both values are now options, so this
   can be tested with `--seed_cnr_percentile` and `--seed_cnr_floor`. Subjects with weak
   contrast therefore still get a mask of similar relative size and are likely to be
   segmented less accurately. The authors flag this for review.
3. **"CNR" is a contrast ratio.** The map is (NM - CP) / CP with no noise term. File
   names and QC keys keep "CNR" so existing outputs stay readable.
4. **v3.0 changes NM masks at default settings.** The mask is now morphologically
   closed (3 x 3 x 3) and intersected with the search region, so v3.0 and v2.5.0
   results are not directly comparable, even with `--search_dilation_mm 0`.
5. **The core-rim constants are not command-line options.** The NM mask's settings are, but
   `core_rim_erosion_mm` (0.5),
   `core_rim_core_cnr_threshold` (0.05), `core_rim_rim_cnr_threshold` (0.10),
   `core_rim_min_cluster_voxels` (25) and `core_rim_components_to_keep` (2) are
   module-level constants. They are echoed into the run log, so a result can always be
   traced back to its values, but a sweep means editing the script.
6. **Several constants are acknowledged as arbitrary in the source**, including the
   core erosion distance and the minimum cluster size. Sensitivity to them has not
   been measured.
7. **Keeping two components assumes a bilateral SN.** If the two sides merge into one
   component, or one side is missed, the component filter handles it silently. The
   counts in `Group_QC_Component_Summary.csv` are the check.
8. **The tracing is one rater's, drawn once.** Neither inter-rater nor intra-rater
   reliability has been assessed, and every subject-level result inherits the shape of
   that tracing.
9. **The V4 alignment is a rigid shift of about 1.3 mm.** It assumes the V4 template
   and the group average differ only by translation. This held for the cohorts run so
   far but has not been shown in general.
10. **The threshold line in `Poster_CNR_Histogram_*` is approximate.** The threshold
    ratio matches the pipeline's, but the figure maps it onto raw intensities using a
    CP reference from the uneroded CP mask. The pipeline instead thresholds the
    smoothed ratio map, uses an eroded CP, and also caps the mask at the 99th
    percentile. Treat the line as illustrative.
11. **The NM-blind control is not volume-matched.** It keeps exactly the top-45% share
    of the search region by depth, while the final NM mask is further changed by the
    99th-percentile cap, the component filter and the closing. Both voxel counts are in
    `Poster_NM_Blind_Control.csv`; part of any DICE difference may reflect mask size.
12. **Poster statistics are descriptive.** `Poster_Accuracy` uses paired Wilcoxon
    signed-rank tests with a Holm correction across its three atlas comparisons, and
    `Poster_Statistics.csv` repeats every comparison with sign tests, bootstrap
    intervals and Holm adjustment both within and across the tests. None of this is a
    validation against an independent reference.

### Technical

13. **The pipeline is a single script of about 4,200 lines.** It parses arguments at
    import time, so it cannot be imported or unit-tested in the usual way. The tests
    work around this by lifting functions out with `ast`.
14. **`--output_root` must sit inside `--project_root`** when the V4 template is
    built, because the container mounts only `--project_root`.
15. **Stage `segment` is interactive** and cannot run unattended.
16. **`--rebuild_template` and an existing tracing are mutually exclusive.** This is
    by design, since a rebuilt template invalidates the tracing, but it makes
    rebuilding a careful manual operation.
17. **Subject discovery is a bare `sub-*` glob.** There is no BIDS validation, so a
    stray directory matching the prefix is treated as a subject.
18. **Resuming relies on Nipype's cache.** A failed subject restarts from its last
    cached step, which depends on an intact `--work_dir`.
19. **The anchored atlas is saved as `anat/transform_Warped.nii.gz`,** the ANTs default
    name, rather than the `CIT168_Anchored` its DataSink key suggests. Code that reads
    it back already looks for that name first.
20. **`--seed_image raw` fails when a scan's qform and sform disagree.** ANTs writes the
    warped trace on the qform grid while nibabel reads the raw image's sform, so the grid
    check rejects the subject (seen for BioGen `sub-059`).

### Reproducibility

21. **`requirements.txt` pins one working environment**, not a tested minimum.
22. **Registrations are not bit-reproducible.** ANTs runs multithreaded without a fixed
    seed, so two runs of the same code on the same data give slightly different transforms
    and masks (seed Dice about 0.98 between repeat runs on BioGen `sub-002`).
23. **External tool versions are not recorded.** The run log now records the command
    line and script path, but not the ANTs, SPM or ITK-SNAP versions, so results cannot
    be fully reproduced from the repository alone.

---

## Unresolved

### Two masking methods whose source is lost

Output directories written by earlier runs contain `Native_Bilateral_NM_Mask.nii.gz` (2026-04-19) and
`Native_Refined_NM_Hysteresis.nii.gz` (2026-04-16), with matching
`Bilateral_NM_Summary.json` and `Refined_NM_Hysteresis_Summary.json`. The JSON records
a `"method": "bilateral_nm"` with parameters (`split_x_mm` and per-side component
eligibility) that no surviving script produces.

**No code in the working directory, on any branch, or anywhere in the Git history
produces these files.** They are outputs of an April 2026 pipeline revision that was later
overwritten. Two masking approaches, a bilateral split with per-side component
selection and a hysteresis-based refinement, were implemented, run, and lost.

The outputs and their parameter files survive and would support reconstructing either
method. Nothing in the current pipeline depends on them.

### The modular refactor

`legacy/modular/` holds a clean, behaviour-preserving split of the March monolith into
stages. It was never carried forward, and more than 2,000 lines of pipeline development
have happened since, so it cannot be merged, only redone. It is kept as the starting
point for any future modularization. See [../legacy/README.md](../legacy/README.md).

---

## Suggested next steps

1. Revisit the threshold rule (items 2 and 3): a floor that can act, or a rule tied to
   each subject's contrast distribution, as the authors' note suggests.
2. Record ANTs, SPM and ITK-SNAP versions in the run log (item 23), the cheapest real
   improvement to reproducibility.
3. Promote the core-rim constants to command-line options (item 5), keeping current values as
   defaults.
4. Volume-match the NM-blind control (item 11) before using it as evidence.
5. Decide the fate of the native-space builder by comparing it with the MNI-space
   builder on one cohort.
6. Decide whether `bilateral_nm` or the hysteresis refinement is worth reconstructing
   from the surviving outputs.
7. Assess intra-rater reliability of the template tracing (item 8).
