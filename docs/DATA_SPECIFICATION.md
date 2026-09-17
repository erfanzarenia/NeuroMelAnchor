# Data Specification

The inputs the pipeline expects and the outputs it writes. File names below are the ones
pipeline v3.0 actually produces, checked against its DataSink routes and a complete output
directory.

---

## Inputs

### Directory layout

BIDS-like: one `sub-*` directory per subject under `--design_dir`, each with an `anat/`
subdirectory. Subjects are found by globbing `sub-*`, so any subject label works as long as
the prefix matches.

```
<design_dir>/
├── sub-001/anat/...
├── sub-002/anat/...
└── ...
```

### NM-sensitive images

Neuromelanin-sensitive GRE, 3D NIfTI, in native acquisition space. Subjects can have one or
more acquisitions: with `--nm_image_strategy mean` (the default) they are realigned and
averaged, and with `first` only the first is used.

| `--cohort` | `--nm_input combo` | `--nm_input echo` |
|---|---|---|
| `biogen` | `{sub}_acq-CombEchoNM*_GRE.nii.gz` | `{sub}_run-1_echo-{N}_part-mag_MEGRE.nii.gz` |
| `snit` | `{sub}_acq-CombEchoNM*_GRE.nii.gz` | `{sub}_acq-NM_rec-DIS2D_run-*_echo-{N}_part-mag_GRE.nii.gz` |

`combo` is the vendor-combined multi-echo image and has the higher SNR; `echo` selects one echo
with `--nm_echo`.

### T1-weighted image

| `--cohort` | Pattern |
|---|---|
| `biogen` | `{sub}_acq-T1w.nii.gz` |
| `snit` | `{sub}_acq-MP2RAGEpostproc_run-01_T1w.nii.gz` |

Used for brain extraction, as the NM registration target, and as the bridge to MNI space. For
7T MP2RAGE data, denoise the UNI image first with `workflow/tools/mp2rage_uniden_batch.sh`.

`--nm_pattern` and `--t1_pattern` override either pattern.

### Manual masks (optional)

Passed with `--manual_dir`; binary, in native NM space:

```
<manual_dir>/sub-001/anat/sub-001_space-NM_label-SN_desc-manual_mask.nii[.gz]
```

Subjects without a matching file are processed normally, and the gap is recorded in their
`Stage2_Output_Manifest.json`. Manual masks are used only for evaluation and the poster figures;
they never feed back into segmentation.

### Atlas resources

Resolved from `--project_root/resources/tpl-MNI152NLin2009cAsym/`. See
[../resources/README.md](../resources/README.md) for the full table and for the one file that
must be generated after cloning.

---

## Outputs per subject

`<input>` stands for the input NM file name without its extension, for example
`sub-002_run-1_echo-1_part-mag_MEGRE`.

### `anat/`

| File | Stage | Contents |
|---|---|---|
| `r<input>_corrected.nii` | 1 | Realigned and N4 bias-corrected NM, one per acquisition |
| `mean<input>.nii` | 1 | SPM realignment mean |
| `mean_NM_bias_corrected.nii.gz` | 1 | The subject's NM reference: the mean of the bias-corrected acquisitions |
| `NM_MNI.nii` | 1 | NM in MNI space, 1 mm |
| `NM_MNI_res-05.nii` | 1 | NM in MNI space, 0.5 mm; only with `--hires` |
| `sNM_MNI.nii` | 1 | Smoothed NM in MNI space; only with `--smooth` above 0 |
| `MNI_SNc_Manual_trans.nii.gz` | 2 | The traced SN, warped to native NM space |
| `MNI_CP_Manual_trans.nii.gz` | 2 | The traced CP, warped to native NM space |
| `Native_SN_Search_Region.nii.gz` | 2 | The search region: the warped SN, grown by `--search_dilation_mm` |
| `Native_Tractography_Seed_SNc.nii.gz` | 2 | **The NM-defined SN mask** |
| `Native_CoreRim_NM_Mask.nii.gz` | 2 | The core-rim variant; absent with `--skip_core_rim_nm` |
| `SNc_trans.nii.gz` | 2 | CIT168 binary SNc, warped to native NM space |
| `SN_ROI_Mask.nii.gz` | 2 | The NM mask grown by 6 mm, which limits the anchoring registration |
| `transform_Warped.nii.gz` | 2 | **The anchored CIT168 SNc**, under the ANTs default output name |
| `Native_ProbCIT_SNc_Probability.nii.gz` | 2 | CIT168 probabilistic SNc, warped to native space |
| `Native_ProbCIT_SNc_thr025.nii.gz` | 2 | The same, thresholded at `--prob_atlas_threshold` |
| `Native_NMS_SNc_Probability.nii.gz` | 2 | NMS probabilistic SNc, warped to native space |
| `Native_NMS_SNc_thr025.nii.gz` | 2 | The same, thresholded at `--nms_atlas_threshold` |

The number in `thr025` follows the threshold option: `--prob_atlas_threshold 0.4` writes
`Native_ProbCIT_SNc_thr040.nii.gz`.

### `transforms/`

| File | Contents |
|---|---|
| `T1_to_MNI_Composite/transformComposite.h5` | T1 to MNI, forward |
| `T1_to_MNI_InvComposite/transformInverseComposite.h5` | T1 to MNI, inverse |
| `NM_to_T1_Composite/transformComposite.h5` | NM to T1, forward |
| `NM_to_T1_InvComposite/transformInverseComposite.h5` | NM to T1, inverse |
| `transformComposite.h5` | The anchoring registration (atlas onto NM mask), forward |
| `transformInverseComposite.h5` | The anchoring registration, inverse |

The Stage 2 registration of each subject to the group template is cached in `--work_dir` but
not copied to the output root.

### `cnr/`

| File | Contents |
|---|---|
| `Subject_CNR_Map_Smoothed.nii.gz` | The contrast ratio map, (NM - CP) / CP, computed inside the search region and smoothed |
| `Seed_Diagnostics.json` | Voxel counts of the NM mask and the search region, and the region's mean contrast ratio |
| `SN_Mask_QC.json` | Every pairwise comparison, per-mask component summaries, and QC flags |

### `metrics/`

| File | Contents |
|---|---|
| `rp_<input>.txt` | SPM realignment parameters |
| `motion_params.json` | Translation (mm) and rotation (degrees) ranges |

### `qc/`

| File | Written by | Contents |
|---|---|---|
| `Registration_NM_to_MNI_QC.png` | Stage 1 | NM in MNI space over the MNI T1 |
| `Registration_norm-T1_to_MNI_QC.png` | Stage 1 | The normalized T1 over the MNI T1 |
| `Registration_Atlas_on_T1Norm_QC.png` | Stage 1 | CIT168 SNc over the normalized T1 |
| `Run_Alignment_Overlay_QC.png` | Stage 1 | The realigned acquisitions overlaid |
| `CNR_Histogram.png` | Stage 2 | CP and SN intensity distributions, with the CP reference marked and the threshold in the title |
| `CoreRim_NM_Summary.json` | Stage 2 | Core and rim voxel counts, candidates and kept components |
| `Subject_QC_Dashboard.png` | Group QC | Contours of every mask on the subject's NM image |
| `Stage2_Output_Manifest.json` | Group QC | Which expected masks exist, with their shapes, voxel counts and grid checks, and any warnings |

---

## Group-level outputs

```
<output_root>/
├── Study-Specific_NM_Template.nii.gz       group-average NM template (Stage 1)
├── drawing_template/
│   ├── SN_Template_V4.nii.gz               FireANTs midbrain template, 0.5 mm
│   ├── SN_Template_V4_aligned.nii.gz       V4 shifted onto the group average; the tracing's grid
│   ├── V4_Mask.nii.gz                      where V4 has data; restricts the V4 alignment
│   ├── GroupAverage_to_V4_0GenericAffine.mat
│   └── build/                              builder work directory, when the pipeline builds V4
├── manual_trace_v4/
│   ├── MNI_Manual_Masks_combined.nii.gz    the tracing: 1 = CP, 2 = SN
│   ├── MNI_SNc_Manual.nii.gz               SN, split from the tracing
│   ├── MNI_CP_Manual.nii.gz                CP, split from the tracing
│   └── itksnap_labels.txt
├── manual_trace/                           the same, for --drawing_template naive
├── Automated_Setup.itksnap                 ITK-SNAP workspace, for --drawing_template naive
├── NeuroMelAnchor_master.log               every run's settings and command line
├── NeuroMelAnchor_Phase-1_Dag              Nipype workflow graphs
├── NeuroMelAnchor_Pipeline_Dag(.png)
├── Group_QC_*                              group plots and summary tables
└── Poster_*                                poster figures and the NM-blind control
```

| File | Contents |
|---|---|
| `Group_QC_Pairwise_Summary.csv` | One row per subject and mask pair, with every metric |
| `Group_QC_Component_Summary.csv` | Connected-component summaries per subject and mask |
| `Group_QC_Method_Comparison.json` and `.png` | Methods scored side by side |
| `Group_QC_DICE_Raincloud.png` | DICE distributions per method |
| `Group_QC_HD95_DotPlot.png` | HD95 per subject and method |
| `Group_QC_Deviation_Boxplots.png` | Deviation box plots by method |
| `Group_QC_Seed_Atlas_Coupling.png` | How far the NM mask departs from the atlas |
| `Group_QC_Boundary_vs_Core_CR.png` | Contrast ratio at the manual boundary against the interior |
| `Poster_Accuracy` | DICE and HD95 against the manual masks for the NM mask and three atlases |
| `Poster_Contours_{worst,median,best}_<sub>` | Contours for the subjects with the lowest, median and highest NM-mask DICE |
| `Poster_Strip_<sub>` | One slice per method for the median subject |
| `Poster_CNR_Histogram_<sub>` and `Poster_CNR_Histogram_Group` | CP and search-region intensities, for the median subject and pooled |
| `Poster_NM_Blind_Control.csv` and figure | The NM mask against the propagated tracing and an NM-blind control |

Poster files are written only when manual masks are available, each figure as PNG and SVG. With
`--test`, every group and poster file gains a `_TEST` suffix so a test run cannot overwrite a
real one. The metrics and figures are explained in [EVALUATION.md](EVALUATION.md).

`--work_dir` holds Nipype's cache. It is large, disposable, and safe to delete between
analyses, but deleting it forces a full recomputation.

## Files from older revisions

Output directories written by earlier revisions can contain files the current code does not
write, such as `Disagreement_*` maps, `Motion_Params_Plot.png`, `Run_Difference_QC.png`,
`Native_Bilateral_NM_Mask.nii.gz` and `Native_Refined_NM_Hysteresis.nii.gz`. See [STATUS.md](STATUS.md).
