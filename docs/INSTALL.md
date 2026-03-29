# Install

---

## NeuroMelAnchor Installation Guide

This document describes how to install and configure all dependencies required to run the NeuroMelAnchor pipeline.

---

## Overview

NeuroMelAnchor depends on a combination of Python libraries and neuroimaging software:

- Python (≥ 3.9)
- `requirements.txt`
- ANTs (Advanced Normalization Tools)
- SPM12 (via MATLAB)
- ITK-SNAP

---

## System Requirements

- **OS:** Linux or macOS (recommended)
- **RAM:** ≥ 16 GB (32 GB recommended)
- **CPU:** ≥ 8 cores recommended
- **Storage:** ≥ 50 GB

---

## 1. Create Python Environment

We recommend using **conda**:
```bash
conda create -n neuromelanchor python=3.9
conda activate neuromelanchor
```

---

## 2. Install Python Dependencies

From the repository root:
```bash
pip install -r requirements.txt
```

---

## 3. Install ANTs

Download ANTs from: https://github.com/ANTsX/ANTs

After installation, set environment variables:
```bash
export ANTSPATH=/path/to/ants/bin
export PATH=$ANTSPATH:$PATH
```

Verify installation:
```bash
antsRegistration --version
```

---

## 4. Install MATLAB + SPM12

NeuroMelAnchor uses SPM for motion correction and smoothing.

---

## 5. Troubleshooting

- **ANTs not found** → Check `ANTSPATH` and `PATH`
- **SPM not detected** → Verify MATLAB path and `MATLAB_CMD`
- **TemplateFlow download issues** → Set `TEMPLATEFLOW_HOME`
- **Memory errors** → Reduce `--n_procs` in pipeline execution

---

## 6. Next Steps

After installation:

- ➡️ See `DATA_SPECIFICATION.md` to prepare input data
- ➡️ See `USAGE.md` to understand the pipeline
