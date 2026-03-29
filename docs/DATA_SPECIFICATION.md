# Data Specification

---

## Purpose

Defines expected inputs, intermediate representations, outputs, and conceptual organization of data used in the NeuroMelAnchor pipeline. Focus is on data structure and representation rather than algorithmic implementation.

---

## Input Data

### Neuromelanin-Sensitive MRI

- **Imaging modality:** neuromelanin-sensitive MRI
- **Dimensionality:** 3D volume
- **Runs per subject:** maximum 2 runs
- **File format:** NIfTI (`.nii` or `.nii.gz`)
- **Role:** provides signal contrast related to neuromelanin concentration supporting localization of substantia nigra
- **Space before preprocessing:** native acquisition space

Conceptual examples:
```
sub-01_acq-CombEchoNM_run-01_GRE.nii.gz
sub-01_acq-CombEchoNM_run-02_GRE.nii.gz
```

---

### T1-Weighted Anatomical MRI

- **Imaging modality:** T1-weighted structural MRI
- **Dimensionality:** 3D volume
- **File format:** NIfTI (`.nii` or `.nii.gz`)
- **Role:**
  - Brain extraction
  - Registration target for neuromelanin images
  - Transformation to MNI space

Conceptual example:
```
sub-01_acq-T1w.nii.gz
```

---

### Manual Segmentation (Optional)

- **Representation type:** binary mask
- **Dimensionality:** 3D volume
- **File format:** NIfTI (`.nii` or `.nii.gz`)
- **Expected space:** preprocessed MNI space
- **Anatomical structure:** substantia nigra
- **Role:** reference representation for evaluation when available

Conceptual example:
```
sub-01_SN_manual_mask.nii.gz
```

---

## Intermediate Data

### Bias-Corrected Neuromelanin-Sensitive Images

N4 bias field correction applied to each neuromelanin-sensitive MRI run.
```
sub-01_nm_run-01_biascorr.nii.gz
sub-01_nm_run-02_biascorr.nii.gz
```

---

### Within-Subject Averaged Neuromelanin-Sensitive Image

Alignment of neuromelanin runs using SPM, then averaged across runs to produce a single within-subject representation.
```
sub-01_nm_mean.nii.gz
```

---

### Brain-Extracted T1 Image

Brain extraction applied to the T1-weighted anatomical image.
```
sub-01_T1_brain.nii.gz
```

---

### Neuromelanin Image Registered to T1

Spatial alignment of neuromelanin-sensitive image to anatomical T1 image.
```
sub-01_nm_in_T1_space.nii.gz
```

---

### Neuromelanin Image Transformed to MNI Space

Neuromelanin image transformed to MNI space using composed registration transforms.
```
sub-01_nm_MNI.nii.gz
```

---

### Group-Level Neuromelanin-Sensitive Reference Image

Aggregate neuromelanin-sensitive representation across subjects in MNI space. Used to support study-specific anatomical localization.
```
group_nm_mean_MNI.nii.gz
```

---

### Study-Specific Substantia Nigra Reference Region

Anatomical region defined using ITK-SNAP on the group-level neuromelanin image.
```
SN_group_reference_mask.nii.gz
```

---

### Atlas Prior Mask

Anatomical constraint defining the plausible substantia nigra region.
```
SN_atlas_prior_mask.nii.gz
```

---

### Refined Subject-Specific Region

Mask obtained after neuromelanin-guided refinement.
```
sub-01_SN_refined_mask.nii.gz
```

---

## Output Data

### Segmentation Representation

Subject-specific representation of the substantia nigra.

- **Representation type:** binary mask or probabilistic map (final form depends on implementation)
- **Dimensionality:** 3D volume
- **File format:** NIfTI (`.nii` or `.nii.gz`)
```
sub-01_SN_segmentation.nii.gz
```

---

### Evaluation Outputs

Quantitative similarity metrics computed relative to the reference representation.

- **Reference hierarchy:**
  - Manual segmentation when available
  - Atlas-derived mask otherwise
- **Metrics:**
  - Dice similarity coefficient
  - Hausdorff distance
- **Structure:** single file per subject
- **File format:** structured text (e.g., `.json`, `.csv`)
```
sub-01_SN_evaluation_metrics.json
```

---

## Conceptual Data Organization
```
inputs
├── neuromelanin-sensitive MRI runs
├── T1-weighted MRI
└── manual segmentation (if available)

intermediate representations
├── bias-corrected NM images
├── within-subject averaged NM image
├── brain-extracted T1
├── NM image registered to T1
├── NM image transformed to MNI space
├── group-level NM average
├── study-specific reference region
├── atlas prior mask
└── refined subject-specific mask

outputs
└── segmentation representation

evaluation
└── similarity metrics per subject
```
