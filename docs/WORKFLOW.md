# Workflow

---

## Overview

The workflow begins with the preparation of neuromelanin-sensitive (NM)-MRI images. Next, a group-level reference representation is constructed, followed by atlas-based initialization. The process integrates anatomical priors from atlas data with subject-level NM-MRI signal contrast to generate individualized region maps. The pipeline begins with preprocessing and normalization of NM-MRI images (Stage 1), continues with manual ROI definition on the group-average NM template using ITK-SNAP, and concludes with subject-level refinement and evaluation (Stage 2). Evaluation metrics provide quantitative measures of how well generated segmentations correspond to reference representations (manual or atlas-based).

---

## Conceptual Pipeline

The workflow proceeds through a clear sequence of stages combining normalization, manual definition, refinement, and quantitative evaluation.

1. Provide NM-MRI and T1-weighted MRI images
2. **Stage 1 — Forward Normalization**
   - Preprocess NM images: motion correction, bias correction, intensity normalization
   - Align NM → T1 and T1 → MNI space
   - Build a group-average NM template for the study
3. **Segment — Manual ROI Definition (ITK-SNAP)**
   - Define cerebral peduncle (CP) and substantia nigra (SN) masks on the group NM template
   - Save results as the study-level reference representation
4. **Stage 2 — Inverse Normalization & CNR Seed Extraction**
   - Warp SN/CP masks from template back to native NM space
   - Compute CNR map and NM-based seed mask
   - Anchor MNI atlas to NM-derived regions
   - Evaluate segmentation using Dice and Hausdorff distance

---

## Stage 1 — Forward Normalization

This stage standardizes NM-MRI data across subjects to establish a common spatial reference and produces the group-level NM template.

---

### Step 1. Input Image

The pipeline uses neuromelanin-sensitive MRI scans and a T1-weighted image per subject. NM-MRI provides contrast related to neuromelanin concentration, crucial for visualizing the SN.

> **Output:** NM-MRI image and T1 suitable for preprocessing.

---

### Step 2. Preprocessing of NM-MRI

Preprocessing ensures intensity and spatial consistency across runs and subjects. Procedures include:

1. Gunzip NM volumes (if compressed)
2. SPM Realign — motion correction and registration to mean
3. ANTs `N4BiasFieldCorrection` on each realigned volume
4. Compute mean of bias-corrected NM volumes
5. ANTs `BrainExtraction` on T1
6. ANTs Registration (T1 → MNI) — Rigid + Affine + SyN (brainstem-weighted)
7. ANTs Registration (NM → T1) — Rigid
8. Combine transforms and apply to map NM → MNI space
9. *(Optional)* SPM Smooth
10. Generate QC plots and motion metrics

These operations ensure spatial comparability across subjects, creating standardized input for localization.

> **Output:** Preprocessed NM-MRI images in MNI space.

---

### Step 3. Group-Level Average in MNI Space

All normalized NM images are combined to construct a group-level NM-MRI template, capturing common neuromelanin signal patterns across subjects.

> **Output:** `Study_Specific_NM_Template.nii.gz`

This template becomes the visual and spatial basis for defining the SN reference region in the next stage.

---

## Segment — Manual ROI Definition (ITK-SNAP)

Manual segmentation defines a study-level anatomical reference that anchors later subject-specific processing.

---

### Step 4. ITK-SNAP Initialization for Group-Level Segmentation

- Open `Study_Specific_NM_Template` in ITK-SNAP
- Label two regions:
  - **Label 1:** Cerebral Peduncle (CP)
  - **Label 2:** Substantia Nigra (SN)
- Save the masks to `resources/tpl-MNI152NLin2009cAsym/`

The resulting binary masks serve as study-level reference regions guiding all subject-level analyses.

> **Output:** CP and SN reference masks in MNI space.

---

## Stage 2 — Inverse Normalization and CNR Seed Extraction

Stage 2 uses inverse transformations to project reference regions into each subject's native space, refines alignment using neuromelanin signal, and evaluates segmentation accuracy.

---

### Step 5. Atlas Initialization

An atlas-based prior (from FSL brainstem atlas) provides an initial estimate of SN location in each subject, constraining the search space to plausible anatomy.

> **Output:** Initial subject-specific region estimate.

---

### Step 6. Neuromelanin-Guided Refinement

Subject-specific NM-contrast is used to adjust and refine the atlas-derived region. Neuromelanin intensity reflects dopaminergic neuron density, enabling fine-grained localization.

> **Output:** Refined subject-specific mask aligned to NM signal.

---

### Step 7. Segmentation Map Generation

The refined region is converted into a segmentation representation — either:

- **Binary mask:** hard boundary of the SN
- **Probabilistic map:** voxel-wise likelihood of membership, preserving uncertainty

> **Output:** Segmentation representation (`*_SN_mask.nii.gz`).

---

### Step 8. Comparison with Reference Representation

Generated segmentation maps are compared to available references:

- Manual SN mask (if provided)
- Atlas-derived reference (when manual not available)

> **Output:** Paired representations ready for quantitative evaluation.

---

### Step 9. Metric Computation

Quantitative metrics characterize agreement between segmented and reference masks:

- **Dice Similarity Coefficient:** spatial overlap
- **Hausdorff Distance (HD95):** boundary difference

These metrics provide complementary views of spatial accuracy and boundary agreement.

> **Output:** Evaluation scores (per-subject JSON), aggregate QC plots (Dice & HD distributions).

---

## Workflow Diagram
```
NM-MRI → preprocessing → group-level reference construction →
study-specific reference definition → atlas prior initialization →
neuromelanin-guided refinement → segmentation representation
(binary or probabilistic) → evaluation relative to reference representation
```
