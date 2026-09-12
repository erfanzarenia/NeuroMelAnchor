# Evaluation

How masks are compared, what is written, and how to read the results.

Evaluation never alters a segmentation — it only characterises agreement.

---

## What is compared with what

Stage 2 produces or warps six masks per subject, all in **native NM space**:

| Label | Mask | How it is derived |
|---|---|---|
| `NM` | `Native_Tractography_Seed_SNc.nii.gz` | CNR threshold inside the template-derived SN search region |
| `Core-rim` | `Native_CoreRim_NM_Mask.nii.gz` | As above, but core and rim use different CNR thresholds |
| `Atlas` | `SNc_trans.nii.gz` | CIT168 binary SNc, warped |
| `ProbCIT` | `Native_ProbCIT_SNc_thr*.nii.gz` | CIT168 probabilistic SNc, warped then thresholded |
| `NMS` | `Native_NMS_SNc_thr*.nii.gz` | NMS probabilistic SNc, warped then thresholded |
| `Anchored` | `CIT168_Anchored.nii.gz` | CIT168 rigidly re-registered onto the subject's own NM seed |
| `Manual` | from `--manual_dir` | Per-subject manual tracing, when supplied |

Every pair is scored. There is no single "correct" reference: with
`--manual_dir` the manual tracing is the natural comparator, and without it the
atlas-derived masks serve as the reference. The pairs the dashboard highlights
are `Manual vs NM`, `Manual vs Core-rim`, `Manual vs Atlas`, `Manual vs ProbCIT`,
`Manual vs NMS` and `Manual vs Anchored`.

All comparisons are in native space. Nothing is compared in MNI space, because
resampling a few-voxel structure through a warp before scoring it would measure
the interpolation as much as the segmentation.

---

## Metrics

For every pair:

| Metric | Meaning |
|---|---|
| **Dice** | Overlap, 0–1. Higher is better. The headline number, but saturates: two masks can both score well while disagreeing systematically at the boundary |
| **Jaccard** | Intersection over union. Same information as Dice, harsher on partial overlap |
| **HD95 (mm)** | 95th percentile of surface distances. Boundary agreement, robust to single stray voxels in a way the raw Hausdorff distance is not. **The metric to watch for this structure** — for a mask a few voxels across, an HD95 of 2 mm is a large disagreement even at high Dice |
| **Mean surface distance (mm)** | Average boundary separation. Less sensitive than HD95 to localised excursions |
| **Centroid distance (mm)** | Distance between mask centres of mass, with the x/y/z breakdown. Separates *displacement* from *shape* disagreement: high Dice with a large centroid offset means one mask is systematically shifted |
| **Voxel counts** | Per mask, plus intersection and union. A Dice drop from a size mismatch reads differently from one caused by displacement |
| **Component summary** | Connected components per mask and their sizes. The SN should give **two** components (left and right); one or three is a red flag regardless of Dice |

Empty or single-voxel masks yield `NaN` for the distance metrics rather than
raising, so one bad subject does not abort a group run — check the manifest.

---

## Per-subject outputs

| File | Contents |
|---|---|
| `cnr/SN_Mask_QC.json` | Every pairwise comparison, with all metrics above |
| `qc/Subject_QC_Dashboard.png` | Overlay contours per method on the subject's NM image, plus the metric table |
| `qc/Disagreement_*.nii.gz` | Voxelwise maps of where two methods disagree — loadable in ITK-SNAP, the fastest way to see *why* a number is low |
| `qc/Disagreement_*_Summary.json` | Voxel counts behind those maps |
| `qc/CNR_Histogram.png` | CNR distribution within the search region. A distribution with no right tail means little NM contrast, and every downstream number should be distrusted |
| `qc/CoreRim_NM_Summary.json` | Core and rim voxel counts, candidates and kept components |
| `qc/Stage2_Output_Manifest.json` | What was produced and any warnings — **check this first when a subject looks wrong** |
| `metrics/motion_params.json` | Translation and rotation ranges from SPM realignment |
| `qc/Run_Difference_QC.png` | Between-run similarity, before averaging |

## Group outputs

| File | Contents |
|---|---|
| `Group_QC_DICE_Raincloud.png` | Dice distribution per method — spread matters more than the mean |
| `Group_QC_HD95_DotPlot.png` | HD95 per subject per method; outlier subjects are visible individually |
| `Group_QC_Deviation_Boxplots.png` | Per-subject deviation from the group |
| `Group_QC_Method_Comparison.{png,json}` | Methods scored side by side |
| `Group_QC_Seed_Atlas_Coupling.png` | How far the NM seed moves away from the atlas — how much subject-specific information is actually being added |
| `Group_QC_Boundary_vs_Core_CR.png` | Boundary versus core contrast, the motivation for the core/rim variant |
| `Group_QC_Pairwise_Summary.csv` | All metrics flattened, one row per subject and pair — **the file to load for group statistics** |
| `Group_QC_Component_Summary.csv` | Connected-component counts per subject and method |

With `--test`, every group file gains a `_TEST` suffix so a test run cannot
overwrite a real one.

---

## Reading the results

A few patterns worth recognising:

- **High Dice, high HD95** — the bulk overlaps but a boundary excursion is
  present. Usually a stray component or a rim voxel cluster; look at the
  disagreement map. This is what the core/rim variant exists to reduce.
- **High Dice, large centroid offset** — systematic displacement, not shape
  disagreement. Suspect the registration, not the threshold.
- **NM mask far smaller than the atlas** — either genuinely low NM contrast
  (check `CNR_Histogram.png`) or a search region clipped by the tracing.
- **Component count ≠ 2** — one component usually means the two sides merged
  across the midline; more than two means the cluster filter let noise through.
- **Anchored close to Atlas** — the anchoring step found little to move, i.e. the
  subject's NM signal was not informative enough to shift the atlas.

Group plots are descriptive. No statistical testing is performed, and none of
these metrics has been validated against an independent ground truth for this
dataset. See [STATUS.md](STATUS.md).
