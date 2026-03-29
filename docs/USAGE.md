# Usage

---

## NeuroMelAnchor Usage Guide

This document describes how to run the NeuroMelAnchor pipeline after installation and data preparation.

- See `INSTALL.md` for environment setup
- See `DATA_SPECIFICATION.md` for required data structure

---

## Overview

NeuroMelAnchor is executed via a command-line interface and supports modular execution of different pipeline stages:

- **Phase 1** — preprocessing and normalization
- **Segment** — manual reference definition (ITK-SNAP)
- **Phase 2** — segmentation and evaluation

---

## Basic Command
```bash
python pipeline.py \
  --project_root /path/to/project \
  --design_dir /path/to/data \
  --output_root /path/to/output \
  --work_dir /path/to/work \
  --stage all \
  --n_procs 10
```

---

## Required Arguments

- `--project_root` → project directory (masks, scripts)
- `--design_dir` → input dataset (BIDS-like structure)
- `--output_root` → output directory
- `--work_dir` → Nipype working directory

---

## Optional Arguments

- `--stage` → pipeline stage to run
  - `1` → preprocessing only
  - `segment` → manual segmentation step
  - `2` → segmentation + evaluation
  - `all` → full pipeline (default)
- `--manual_dir` → directory containing manual SN masks (optional)
- `--n_procs` → number of parallel processes

---

## Workflow Execution

### 1. Preprocessing (Phase 1)
```
--stage 1
```

Performs:

- NM-MRI bias correction
- Run alignment and averaging
- T1 preprocessing and normalization
- NM → MNI transformation

Outputs:

- Normalized NM images
- Group-level NM template
- QC plots

---

### 2. Manual Reference Definition
```
--stage segment
```

- Launches ITK-SNAP workspace
- User defines:
  - Substantia Nigra (SN)
  - Cerebral Peduncle (CP)

This step is performed **once per dataset**.

---

### 3. Segmentation and Evaluation (Phase 2)
```
--stage 2
```

Performs:

- Atlas-based initialization
- Neuromelanin-guided refinement
- Segmentation map generation
- Evaluation (Dice, Hausdorff Distance)

Outputs:

- Subject-specific SN segmentation
- QC metrics per subject
- QC plots (Dice and HD distributions)

---

### Full Pipeline
```
--stage all
```

---

## Typical Workflow

1. Prepare dataset (`DATA_SPECIFICATION.md`)
2. Run preprocessing
3. Define SN reference (ITK-SNAP)
4. Run segmentation
5. Review outputs and QC metrics

---

## Quick Start
```bash
python run_pipeline.py \
  --project_root /path/to/project \
  --design_dir /path/to/data \
  --output_root /path/to/output \
  --work_dir /path/to/work \
  --stage all
```

You can also run individual stages:
```bash
--stage 1        # preprocessing and normalization
--stage segment  # manual segmentation (ITK-SNAP)
--stage 2        # seed generation and QC
```

---

## Notes

- Manual segmentation is optional but recommended for evaluation
- Pipeline supports up to two NM runs per subject
- Parallel execution is supported via Nipype

---

## Troubleshooting

- **No outputs** → check input paths
- **Registration issues** → inspect QC plots
- **Empty segmentation** → verify SN reference and CNR thresholds
