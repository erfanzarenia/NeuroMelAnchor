# Evaluation

---

## Purpose

This document describes the evaluation framework used to quantify agreement between the generated substantia nigra segmentation and a reference representation. Evaluation provides quantitative measures of spatial correspondence between segmentation outputs and available anatomical references. Evaluation is part of the standard NeuroMelAnchor workflow.

---

## Reference Representations

Evaluation metrics are computed relative to an available reference representation of the substantia nigra. Reference hierarchy:

- Manual segmentation when available
- Atlas-derived reference when manual segmentation is not available

Manual segmentation is expected to be defined on preprocessed data in MNI space. Atlas-derived reference corresponds to the anatomical prior used during initialization.

---

## Segmentation Representations

Evaluation is compatible with segmentation representations in either of the following forms:

- Binary mask
- Probabilistic map

When probabilistic representations are used, evaluation may involve thresholding or direct comparison depending on implementation decisions. The final segmentation representation format is not yet fixed and both forms are supported conceptually.

---

## Evaluation Metrics

### Dice Similarity Coefficient

Measures spatial overlap between segmentation and reference representation.

**Conceptual interpretation:**
- Value range: 0 to 1
- Higher values indicate greater spatial overlap
- A value of 1 indicates identical spatial extent

**Sensitive to:**
- Volumetric overlap
- Agreement in voxel classification

**Less sensitive to:**
- Boundary distance differences when overlap is high

---

### Hausdorff Distance

Measures distance between the boundaries of segmentation and reference representation.

**Conceptual interpretation:**
- Lower values indicate greater spatial agreement
- Reflects boundary correspondence between regions
- Sensitive to local boundary deviations

**Captures:**
- Spatial discrepancy between region edges
- Worst-case boundary disagreement

---

## Interpretation of Evaluation Outputs

Dice similarity coefficient and Hausdorff distance provide complementary information:

- **Dice similarity coefficient** reflects overlap agreement
- **Hausdorff distance** reflects boundary agreement

Together, these metrics characterize spatial correspondence between segmentation and reference representation. Evaluation results are produced per subject. Output format is not constrained and may include structured text representations such as `.json` or `.csv`.

---

## Relationship to Workflow

Evaluation occurs after generation of the segmentation representation. It compares the segmentation representation with the reference representation. Evaluation does not alter segmentation outputs — it provides quantitative characterization of segmentation agreement only.

---

## Summary

- Compatible with binary or probabilistic segmentation representations
- Supports manual segmentation when available
- Supports atlas-derived reference when manual segmentation is not available
- Produces quantitative metrics describing spatial agreement
- Applied per subject
- Integrated as part of the NeuroMelAnchor workflow
