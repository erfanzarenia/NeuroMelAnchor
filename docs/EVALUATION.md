# Evaluation

How masks are compared, what is written, and how to read it. Evaluation never changes a
segmentation; it only describes agreement.

---

## The masks

Stage 2 produces or warps up to seven masks per subject, all in **native NM space**:

| Label | File in `anat/` | How it is made |
|---|---|---|
| `NM` (`cnr` in QC keys) | `Native_Tractography_Seed_SNc.nii.gz` | Contrast threshold inside the search region |
| `Core-rim` | `Native_CoreRim_NM_Mask.nii.gz` | As above, with separate thresholds for the eroded core and the rim |
| `Atlas` | `SNc_trans.nii.gz` | CIT168 binary SNc, warped |
| `ProbCIT` | `Native_ProbCIT_SNc_thr025.nii.gz` | CIT168 probabilistic SNc, warped then thresholded |
| `NMS` | `Native_NMS_SNc_thr025.nii.gz` | NMS probabilistic SNc, warped then thresholded |
| `Anchored` | `transform_Warped.nii.gz` | CIT168 SNc rigidly re-registered onto the NM mask |
| `Manual` | from `--manual_dir` | The subject's manual tracing, when supplied |

Everything is compared in native space. Nothing is scored in MNI space, because resampling a
structure a few voxels across through a warp before scoring it would measure the interpolation
as much as the segmentation.

## The pairs

Twelve pairs are scored when every mask is available; pairs involving a missing mask are
skipped. In each pair name the first mask is A and the second is B.

| Against the NM mask | Against the manual mask | Other |
|---|---|---|
| `cnr_vs_atlas` | `cnr_vs_manual` | `core_rim_nm_vs_nms_atlas` |
| `cnr_vs_anchored_atlas` | `atlas_vs_manual` | |
| `cnr_vs_prob_atlas` | `prob_atlas_vs_manual` | |
| `cnr_vs_nms_atlas` | `nms_atlas_vs_manual` | |
| `cnr_vs_core_rim_nm` | `core_rim_nm_vs_manual` | |
| | `anchored_atlas_vs_manual` | |

---

## Metrics

| Metric | Meaning |
|---|---|
| **DICE** | Overlap, from 0 to 1; higher is better. The headline number, but it saturates: two masks can score well while disagreeing systematically at the boundary |
| **Jaccard** | Intersection over union. The same information as DICE, harsher on partial overlap |
| **Precision** | The share of A inside B. For `cnr_vs_manual`, the share of the NM mask that lies inside the expert mask |
| **Recall** | The share of B covered by A. For `cnr_vs_manual`, the share of the expert mask that the NM mask covers |
| **HD95 (mm)** | The 95th percentile of surface distances between the masks: boundary agreement that ignores a few stray voxels. **The metric to watch for this structure**, since for a mask a few voxels across an HD95 of 2 mm is a large disagreement even at high DICE |
| **Surface distances (mm)** | Mean, median, HD50, HD90 and maximum, alongside HD95 |
| **Centroid distance (mm)** | Distance between the masks' centres of mass, with the x, y and z offsets. It separates displacement from shape: high DICE with a large offset means one mask is shifted |
| **Sizes** | Voxel counts, intersection, false-positive and false-negative voxels of A against B, and volumes in mm³ with their difference in mm³ and percent |

Empty masks give `NaN` for the distance metrics instead of raising, so one bad subject does not
abort a group run.

`SN_Mask_QC.json` also summarizes each mask's connected components. The SN should give **two**
(left and right); one or three is a warning sign whatever the DICE.

### QC flags

`SN_Mask_QC.json` lists automatic flags:

| Flag | Raised when |
|---|---|
| Low DICE against the manual mask | DICE below 0.5 for the NM, atlas, ProbCIT, NMS or core-rim mask |
| Core-rim worse than NM | Core-rim DICE against the manual mask is more than 0.05 below the NM mask's |
| High HD95 against the manual mask | HD95 above 5 mm for the NM or core-rim mask |
| Large centroid offset against the manual mask | The NM mask's centroid is more than 3 mm from the manual mask's |
| Anchoring reduced agreement with the manual mask | The anchored atlas's DICE is more than 0.05 below the unanchored atlas's |
| Anchoring degraded agreement with the NM mask | Against the NM mask, the anchored atlas's DICE is more than 0.05 below the unanchored atlas's |
| Very small NM mask | Fewer than 20 voxels |
| Fragmented NM mask | More than two components, or the largest holds less than 80% of the mask |
| Disagreement with the atlas | NM mask DICE against the atlas below 0.4, HD95 above 5 mm, or a centroid offset above 3 mm |

---

## Per-subject outputs

| File | Contents |
|---|---|
| `cnr/SN_Mask_QC.json` | All pairwise metrics, component summaries and flags |
| `cnr/Seed_Diagnostics.json` | Sizes of the NM mask and the search region, and the region's mean contrast ratio |
| `qc/Subject_QC_Dashboard.png` | Contours of every mask on the subject's NM image; the quickest way to see why a number is low |
| `qc/CNR_Histogram.png` | CP and SN intensity distributions. A distribution with no bright tail means little NM contrast, and every number downstream should be distrusted |
| `qc/CoreRim_NM_Summary.json` | Core and rim sizes, candidates and kept components |
| `qc/Stage2_Output_Manifest.json` | What was produced and any warnings. **Check this first when a subject looks wrong** |
| `metrics/motion_params.json` | Translation and rotation ranges from realignment |
| `qc/Run_Alignment_Overlay_QC.png` | The realigned acquisitions overlaid, before averaging |

## Group outputs

| File | Contents |
|---|---|
| `Group_QC_Pairwise_Summary.csv` | One row per subject and pair with `subject_id`, `pair`, `dice`, `jaccard`, `precision`, `recall`, `hd95_mm`, the mean, median, HD90, HD95 and maximum surface distances, `centroid_distance_mm`, and the volumes of A and B with their difference. **The file to load for group statistics** |
| `Group_QC_Component_Summary.csv` | Connected-component summaries per subject and mask |
| `Group_QC_DICE_Raincloud.png` | DICE distribution per method; the spread matters more than the mean |
| `Group_QC_HD95_DotPlot.png` | HD95 per subject and method, so outlier subjects are visible individually |
| `Group_QC_Deviation_Boxplots.png` | Deviation box plots by method |
| `Group_QC_Method_Comparison.json` and `.png` | Methods scored side by side |
| `Group_QC_Seed_Atlas_Coupling.png` | How far the NM mask departs from the atlas, which is how much subject-specific information it adds |
| `Group_QC_Boundary_vs_Core_CR.png` | Contrast ratio at the manual boundary against the interior; the motivation for the core-rim variant |

With `--test`, every group and poster file gains a `_TEST` suffix.

---

## Poster figures

Group QC writes these when manual masks are available, each as PNG and SVG. They were made for
the Brainhack Western 2026 poster.

| File | Contents |
|---|---|
| `Poster_Accuracy` | Median DICE and HD95 against the manual masks, with every subject shown, for the NM mask, CIT168, CIT168 probabilistic and NMS probabilistic. The NM mask is compared with each atlas by a paired Wilcoxon signed-rank test, Holm-corrected across the three comparisons; a comparison whose paired differences are all zero is reported as not significant |
| `Poster_Contours_{worst,median,best}_<sub>` | For the subjects with the lowest, median and highest NM-mask DICE: the expert mask filled, with the NM mask, CIT168 and NMS outlined, on three axial slices (one row per method) |
| `Poster_Strip_<sub>` | The median subject on the slice where the expert SN is largest, one panel per method |
| `Poster_CNR_Histogram_<sub>` | The median subject's CP and search-region intensities, with the CP reference and an approximate threshold line |
| `Poster_CNR_Histogram_Group` | Every subject's intensities divided by their CP reference, shown as a mean with an interquartile band |
| `Poster_NM_Blind_Control` (`.csv` and figure) | See below |

The threshold line in `Poster_CNR_Histogram_<sub>` is illustrative: the pipeline thresholds the
smoothed contrast map with an eroded CP mask, which a line on raw intensities cannot show
exactly.

### The NM-blind control

Asks whether NM contrast adds anything beyond registration and position. For each subject with
a manual mask it scores three masks against the expert mask:

1. **Propagated drawing:** the warped SN tracing, from registration alone.
2. **NM-blind mask:** the same share of the search region that the percentile threshold keeps
   (the top 45% at a `seed_cnr_percentile` of 55), chosen by depth, meaning the voxels furthest
   from the region's edge, instead of by brightness.
3. **NM mask:** the pipeline's contrast-based mask.

`Poster_NM_Blind_Control.csv` holds, per subject, the voxel counts of the search region and both
masks, and the DICE and HD95 of all three. The figure links each subject across the three and
reports how many subjects the NM mask beats the NM-blind mask in, with a paired Wilcoxon
signed-rank test per metric.

The control is not volume-matched: the NM mask also passes through the 99th-percentile cap, the
component filter and the closing, so it rarely keeps exactly 45% of the region. Compare the
voxel counts before attributing a DICE difference to contrast.

---

## Reading the results

- **High DICE, high HD95:** the bulk overlaps but the boundary strays somewhere, usually a stray
  component or a cluster of rim voxels. Look at the dashboard. The core-rim variant exists to
  reduce this.
- **High DICE, large centroid offset:** systematic displacement rather than shape disagreement.
  Suspect the registration, not the threshold.
- **NM mask much smaller than the atlas:** either little NM contrast (check `CNR_Histogram.png`)
  or a search region clipped by the tracing, which `--search_dilation_mm` addresses.
- **Component count other than 2:** one component usually means the two sides merged across the
  midline; more than two means the cluster filter let noise through.
- **Anchored close to Atlas:** the anchoring found little to move, so the subject's NM signal was
  not informative enough to shift the atlas.

The plots and tests are descriptive, and none of these metrics has been validated against an
independent ground truth. See [STATUS.md](STATUS.md).
