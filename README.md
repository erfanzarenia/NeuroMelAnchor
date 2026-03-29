# NeuroMelAnchor

### Subject-Specific Neuromelanin-Guided Localization of Substantia Nigra

---

## Overview

NeuroMelAnchor is a neuroimaging pipeline for subject-specific localization and segmentation of the substantia nigra using neuromelanin-sensitive MRI. The method combines an atlas-based prior with subject-specific neuromelanin signal to improve anatomical specificity when defining this small midbrain structure.

Accurate delineation of the substantia nigra is important for research involving dopaminergic systems, including studies of learning, motivation, aging, and mental health. Standard atlas-based approaches provide convenient region definitions but may be limited by inter-individual anatomical variability, particularly for small subcortical nuclei.

Neuromelanin-sensitive MRI provides signal contrast related to dopaminergic neurons, allowing localization to be guided by signal properties present in each individual brain (Berman et al., 2022; Coulombe et al., 2021; Nakamura et al., 2018).

NeuroMelAnchor combines complementary data in a hybrid workflow: scans are first preprocessed, then atlas-based initialization is refined using neuromelanin-sensitive contrast to generate a probabilistic substantia nigra segmentation (Cassidy et al., 2019; Wengler et al., 2020). The workflow includes a built-in evaluation stage that quantifies agreement between the resulting segmentation and an available reference representation.

The pipeline is designed as a reusable workflow that can be applied to neuromelanin-sensitive MRI datasets. Evaluation is performed relative to manual segmentation when available, or relative to the atlas-derived reference when manual segmentation is not provided.

This project is being developed as part of Brainhack with the goal of producing a transparent and extensible open neuroimaging workflow.

---

## Key Features

- Subject-specific segmentation of substantia nigra from neuromelanin-sensitive MRI
- Hybrid atlas + neuromelanin-guided localization strategy
- Probabilistic segmentation map output
- Evaluation framework supporting comparison with manual segmentation or atlas-derived reference
- Quantitative evaluation using Dice similarity coefficient and Hausdorff distance
- Modular Python-based workflow
- Reproducible and extensible pipeline structure

---

## Method Summary

NeuroMelAnchor initializes the substantia nigra region using an atlas-based prior (FSL) and refines localization using subject-specific neuromelanin-sensitive signal (Smith et al., 2004). This hybrid approach integrates anatomical priors with image-derived contrast to improve segmentation specificity for small midbrain structures. The resulting output is a probabilistic segmentation map representing the estimated location of the substantia nigra in individual subjects. Agreement between the segmentation map and a reference representation is quantified as part of the workflow.

---

## Inputs and Outputs

### Input Data

Expected structure (BIDS-like):
```
sub-01/
└── anat/
    ├── sub-01_acq-CombEchoNM_run-01_GRE.nii.gz
    ├── sub-01_acq-CombEchoNM_run-02_GRE.nii.gz
    └── sub-01_acq-T1w.nii.gz
```

### Outputs

Main outputs include:

- `NM_BiasCorr` — bias-corrected NM volumes
- `NM_BiasCorr_Mean` — mean NM image used for registration
- `NM_to_MNI` — NM image in template space
- `Native_SN_Mask` — subject-specific SN mask
- `CNR_Map` — contrast-to-noise ratio map
- `Tractography_Seeds` — final SN seed mask
- `QC outputs` — registration overlays
- `Motion plots` — Dice and HD95 metrics

---

## Quality Control

Mask quality is evaluated using:

- **Dice coefficient** — overlap between masks
- **HD95** — boundary distance in millimeters
- **Centroid distance** — spatial shift between masks
- **Visual overlays** — for manual inspection

These metrics allow comparison between atlas-derived masks, NM-derived masks, and manual segmentations (if available).

Evaluation metrics are computed relative to the available reference representation. When manual segmentation is provided, comparison is performed against it. When manual segmentation is not available, comparison is performed relative to the atlas-derived reference.

---

## High-Level Workflow

1. Provide neuromelanin-sensitive MRI image
2. Preprocess neuromelanin-sensitive MRI data
3. Construct group-level reference representation in standard space
4. Initialize substantia nigra region using atlas prior
5. Refine localization using subject-specific neuromelanin-sensitive signal
6. Generate probabilistic segmentation map
7. Evaluate segmentation relative to reference representation
8. Compute similarity metrics describing agreement with reference

---

## Pipeline Summary

1. **Preprocess NM-MRI** — realignment, bias field correction, and averaging
2. **Register images** — NM → T1; T1 → template
3. **Build a study-specific NM template** — improves consistency across subjects
4. **Define SN and background regions in template space** — using manual segmentation
5. **Project masks back to native space** — using inverse transformations
6. **Compute contrast-to-noise ratio (CNR)** — using SN vs background intensities
7. **Generate subject-specific SN masks** — thresholded and refined using NM signal
8. **Perform quality control** — Dice, HD95, centroid distance, and visual overlays

---

## Quick Start

1. Install required software dependencies (see `INSTALL.md`)
2. Prepare input data according to `DATA_SPECIFICATION.md`
3. Run the segmentation workflow as described in `USAGE.md`
4. Evaluation metrics are computed as part of the workflow

---

## Repository Structure
```
neuromelanchor/
├── workflows/            # Nipype workflows
├── scripts/              # helper scripts
├── config/               # example configs
├── docs/                 # documentation
├── assets/               # figures and diagrams
├── examples/             # demo configs and expected outputs
```

Documentation files:

- `README.md` — overview of the project
- `INSTALL.md` — environment setup instructions
- `USAGE.md` — instructions for running the pipeline
- `WORKFLOW.md` — conceptual description of processing steps
- `DATA_SPECIFICATION.md` — input and output definitions
- `EVALUATION.md` — description of evaluation metrics
- `requirements.txt` — Python dependencies

---

## Status

This project is under active development as part of Brainhack. The goal is to produce a reproducible and extensible neuroimaging workflow that can be applied to neuromelanin-sensitive MRI datasets.

---

## Reproducibility

This pipeline is designed to be reproducible:

- All parameters are configurable
- Dependencies are explicitly defined
- Workflow is modular and deterministic

Containerization support (Docker/Singularity) is recommended for full reproducibility.

---

## References

Berman, S., Drori, E., & Mezer, A. A. (2022). Spatial profiles provide sensitive MRI measures of the midbrain micro- and macrostructure. *NeuroImage*, 264, 119660. https://doi.org/10.1016/j.neuroimage.2022.119660

Cassidy, C. M., Zucca, F. A., Girgis, R. R., Baker, S. C., Weinstein, J. J., Sharp, M. E., Bellei, C., Valmadre, A., Vanegas, N., Kegeles, L. S., Brucato, G., Kang, U. J., Sulzer, D., Zecca, L., Abi-Dargham, A., & Horga, G. (2019). Neuromelanin-sensitive MRI as a noninvasive proxy measure of dopamine function in the human brain. *Proceedings of the National Academy of Sciences*, 116(11), 5108–5117. https://doi.org/10.1073/pnas.1807983116

Coulombe, V., Saikali, S., Goetz, L., Takech, M. A., Philippe, É., Parent, A., & Parent, M. (2021). A topographic atlas of the human brainstem in the ponto-mesencephalic junction plane. *Frontiers in Neuroanatomy*, 15, 627656. https://doi.org/10.3389/fnana.2021.627656

Nakamura, Y., Okada, N., Kunimatsu, A., Kasai, K., & Koike, S. (2018). Anatomical templates of the midbrain ventral tegmental area and substantia nigra for Asian populations. *Frontiers in Psychiatry*, 9. https://doi.org/10.3389/fpsyt.2018.00383

Smith, S. M., Jenkinson, M., Woolrich, M. W., Beckmann, C. F., Behrens, T. E. J., Johansen-Berg, H., Bannister, P. R., De Luca, M., Drobnjak, I., Flitney, D. E., Niazy, R. K., Saunders, J., Vickers, J., Zhang, Y., De Stefano, N., Brady, J. M., & Matthews, P. M. (2004). Advances in functional and structural MR image analysis and implementation as FSL. *NeuroImage*, 23, S208–S219. https://doi.org/10.1016/j.neuroimage.2004.07.051

Wengler, K., He, X., Abi-Dargham, A., & Horga, G. (2020). Reproducibility assessment of neuromelanin-sensitive magnetic resonance imaging protocols for region-of-interest and voxelwise analyses. *NeuroImage*, 208, 116457. https://doi.org/10.1016/j.neuroimage.2019.116457
