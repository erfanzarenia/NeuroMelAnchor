# NeuroMelAnchor
**Subject-Specific Neuromelanin-Guided Localization of SN and VTA**

## Overview
Small midbrain nuclei such as the **substantia nigra (SN) and ventral tegmental area (VTA)** are difficult to localize reliably using template-based atlases alone, particularly across individuals. **Neuromelanin-sensitive MRI (NM-MRI)** provides subject-specific contrast that may help improve localization of these dopaminergic structures.

In this project, we aim to develop an open and modular workflow that incorporates NM-MRI as a **subject-informed prior** to refine SN and VTA region-of-interest masks in native space. These masks could then be used for applications such as diffusion tractography, connectivity analyses, seed-based fMRI, or quality control of midbrain segmentations.
## Western Brainhack 2026
This repository hosts code, experiments, and documentation developed during Brainhack Western.

## Goals
- Integrate NM-MRI as a subject-specific localization prior
- Generate refined ROI masks
- Develop QC metrics for midbrain segmentation
