# Data Specification

Inputs the pipeline expects and outputs it produces. Filenames below are the ones
actually written, not illustrative examples.

---

## Inputs

### Directory layout

BIDS-like: one `sub-*` directory per subject under `--design_dir`, each with an
`anat/` subdirectory. Subjects are discovered by globbing `sub-*`, so any subject
label works as long as the prefix matches.

```
<design_dir>/
├── sub-001/anat/…
├── sub-002/anat/…
└── …
```

### NM-sensitive images

Neuromelanin-sensitive GRE, 3D, NIfTI, native acquisition space. One or more
acquisitions per subject; with `--nm_image_strategy mean` (default) they are
realigned and averaged, with `first` only the first is used.

| `--cohort` | `--nm_input combo` | `--nm_input echo` |
|---|---|---|
| `biogen` | `{sub}_acq-CombEchoNM*_GRE.nii.gz` | `{sub}_run-1_echo-{N}_part-mag_MEGRE.nii.gz` |
| `snit` | `{sub}_acq-CombEchoNM*_GRE.nii.gz` | `{sub}_acq-NM_rec-DIS2D_run-*_echo-{N}_part-mag_GRE.nii.gz` |

`combo` is the vendor-combined multi-echo image and has the higher SNR; `echo`
selects a single echo with `--nm_echo`.

### T1-weighted image

| `--cohort` | Pattern |
|---|---|
| `biogen` | `{sub}_acq-T1w.nii.gz` |
| `snit` | `{sub}_acq-MP2RAGEpostproc_run-01_T1w.nii.gz` |

Used for brain extraction, as the NM registration target, and as the bridge to
MNI space. For 7T MP2RAGE data, denoise the UNI image first with
`workflow/tools/mp2rage_uniden_batch.sh`.

Override either pattern with `--nm_pattern` / `--t1_pattern`.

### Manual masks (optional)

Passed with `--manual_dir`, in native NM space, binary:

```
<manual_dir>/sub-001/anat/sub-001_space-NM_label-SN_desc-manual_mask.nii[.gz]
```

Subjects without a matching file are processed normally; the omission is recorded
in that subject's `Stage2_Output_Manifest.json`.

### Atlas resources

Resolved from `--project_root/resources/tpl-MNI152NLin2009cAsym/`. See
[../resources/README.md](../resources/README.md) for the full table and for the
one file that must be generated after cloning.

---

## Outputs

### Per subject

```
<output_root>/sub-XXX/
├── anat/
│   ├── r<sub>_acq-CombEchoNM_run-0N_GRE_corrected.nii   realigned + N4 bias corrected
│   ├── mean<sub>_…_GRE.nii                              SPM realignment mean
│   ├── mean_NM_bias_corrected.nii.gz                    subject NM reference image
│   ├── NM_MNI.nii                                       NM in MNI space (1 mm)
│   ├── NM_MNI_res-05.nii                                0.5 mm version (--hires only)
│   ├── transform_Warped.nii.gz                          T1 warped to MNI
│   ├── MNI_SNc_Manual_trans.nii.gz                      traced SN warped to native NM
│   ├── MNI_CP_Manual_trans.nii.gz                       traced CP warped to native NM
│   ├── SNc_trans.nii.gz                                 CIT168 atlas SNc in native NM
│   ├── SN_ROI_Mask.nii.gz                               dilated search ROI for anchoring
│   ├── CIT168_Anchored.nii.gz                           atlas re-registered onto the NM seed
│   ├── Native_Tractography_Seed_SNc.nii.gz              the NM-derived SN mask
│   ├── Native_CoreRim_NM_Mask.nii.gz                    core/rim variant
│   ├── Native_ProbCIT_SNc_Probability.nii.gz            warped probabilistic atlas
│   ├── Native_ProbCIT_SNc_thr025.nii.gz                 thresholded at --prob_atlas_threshold
│   ├── Native_NMS_SNc_Probability.nii.gz                warped NMS atlas
│   └── Native_NMS_SNc_thr025.nii.gz                     thresholded at --nms_atlas_threshold
├── transforms/
│   ├── transformComposite.h5                            forward warp
│   └── transformInverseComposite.h5                     inverse warp
├── cnr/
│   ├── Subject_CNR_Map_Smoothed.nii.gz                  CNR relative to the CP mode
│   └── SN_Mask_QC.json                                  all pairwise mask metrics
├── metrics/
│   ├── rp_<sub>_….txt                                   SPM realignment parameters
│   ├── motion_params.json                               translation and rotation ranges
│   └── Motion_Params_Plot.png
└── qc/
    ├── Registration_NM_to_MNI_QC.png
    ├── Registration_norm-T1_to_MNI_QC.png
    ├── Registration_Atlas_on_T1Norm_QC.png
    ├── Run_Difference_QC.png                            between-run similarity
    ├── CNR_Histogram.png
    ├── CoreRim_NM_Summary.json
    ├── Disagreement_*.nii.gz                            voxelwise method disagreement maps
    ├── Disagreement_*_Summary.json
    ├── Subject_QC_Dashboard.png
    └── Stage2_Output_Manifest.json                      what was produced, and any warnings
```

The threshold in `Native_*_thr025.nii.gz` tracks the threshold option, so
`--prob_atlas_threshold 0.4` yields `Native_ProbCIT_SNc_thr040.nii.gz`.

### Group level

```
<output_root>/
├── Study-Specific_NM_Template.nii.gz     group-average NM template (Stage 1)
├── manual_trace_v4/                      the tracing, plus itksnap_labels.txt
│   ├── MNI_Manual_Masks_combined.nii.gz  labels 1 = CP, 2 = SN
│   ├── MNI_SNc_Manual.nii.gz             split out from the above
│   └── MNI_CP_Manual.nii.gz
├── drawing_template/
│   ├── SN_Template_V4.nii.gz             FireANTs midbrain template
│   ├── SN_Template_V4_aligned.nii.gz     shifted onto the group average
│   └── build/                            builder work directory and QC
├── Group_QC_DICE_Raincloud.png
├── Group_QC_HD95_DotPlot.png
├── Group_QC_Deviation_Boxplots.png
├── Group_QC_Method_Comparison.{png,json}
├── Group_QC_Seed_Atlas_Coupling.png
├── Group_QC_Boundary_vs_Core_CR.png
├── Group_QC_Pairwise_Summary.csv
├── Group_QC_Component_Summary.csv
├── NeuroMelAnchor_master.log
└── Automated_Setup.itksnap               generated by --drawing_template naive
```

With `--test`, group QC files gain a `_TEST` suffix so a test run cannot
overwrite a real one.

`--work_dir` holds Nipype's cache. It is large, disposable, and safe to delete
between analyses — but deleting it forces a full recomputation.

---

## Metrics files

`cnr/SN_Mask_QC.json` holds every pairwise comparison for the subject. Each pair
carries Dice, Jaccard, HD95 in mm, mean surface distance, voxel counts, connected
component summaries, and the centroid offset in mm with its x/y/z breakdown.

`Group_QC_Pairwise_Summary.csv` is the same data flattened across subjects, one
row per subject and mask pair — the file to load for group statistics.

See [EVALUATION.md](EVALUATION.md).
