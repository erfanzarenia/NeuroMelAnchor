# Manual Segmentation

---

## Purpose

This document defines the role of manual segmentation within the NeuroMelAnchor workflow. Manual segmentation provides a reference representation of the substantia nigra used for quantitative evaluation of automated segmentation outputs.

---

## What Manual Segmentation Represents

Manual segmentation refers to a binary mask of the substantia nigra created through expert-guided delineation. The mask encodes the spatial extent of the structure based on anatomical judgment applied to neuromelanin-sensitive MRI data.

The representation:

- Is a voxel-wise binary mask
- Indicates presence or absence of the substantia nigra
- Is defined in the same space as the preprocessed neuromelanin-sensitive MRI (MNI space)

---

## Why Manual Segmentation Is Used

Manual segmentation serves as a reference representation for evaluating the accuracy of the automated segmentation produced by NeuroMelAnchor. It enables:

- Quantitative assessment of segmentation quality
- Validation of subject-specific localization
- Comparison across subjects and datasets

---

## Assumptions

- Manual segmentation is created on preprocessed data
- Manual segmentation is aligned with the neuromelanin-sensitive MRI in MNI space
- Manual segmentation is treated as a reference representation for evaluation
- Variability in manual delineation may exist across raters or datasets

---

## Role in the Pipeline

Manual segmentation is incorporated during the evaluation stage of the workflow. Its role includes:

- Acting as the reference representation when available
- Enabling computation of similarity metrics
- Supporting interpretation of segmentation quality

**Reference hierarchy within the pipeline:**

- Manual segmentation when available
- Atlas-derived representation otherwise

---

## How Manual Segmentation Is Used

Manual segmentation enters the pipeline as an input mask and is used during comparison with the generated segmentation. Processing steps:

1. Input manual segmentation mask
2. Align with generated segmentation representation (same space assumed)
3. Perform voxel-wise comparison
4. Compute similarity metrics

---

## Metric Computation

Manual segmentation is used to compute quantitative evaluation metrics describing agreement with the automated segmentation.

**Metrics:**
- **Dice similarity coefficient** — overlap-based measure
- **Hausdorff distance** — boundary-based measure

> **Output:** numerical values describing agreement between automated and manual segmentation, produced at the subject level.

---

## Summary

Manual segmentation functions as a reference representation within NeuroMelAnchor. It supports quantitative evaluation of segmentation outputs through comparison and metric computation, enabling assessment of localization accuracy at the subject level.
