# NeuroMelAnchor

### Subject-Specific Neuromelanin-Guided Localization of the Substantia Nigra

---

## Overview

NeuroMelAnchor is a Nipype pipeline for subject-specific localization and
segmentation of the substantia nigra (SN) using neuromelanin-sensitive MRI
(NM-MRI). It combines an atlas-based prior with each subject's own neuromelanin
contrast to improve anatomical specificity when defining this small midbrain
structure.

Accurate delineation of the SN matters for research on dopaminergic systems —
learning, motivation, aging and mental health. Atlas-based approaches provide
convenient region definitions but are limited by inter-individual anatomical
variability, which is precisely the problem for small subcortical nuclei.
NM-MRI provides contrast related to dopaminergic neurons, allowing localization
to be guided by signal present in each individual brain (Berman et al., 2022;
Coulombe et al., 2021; Nakamura et al., 2018).

The workflow preprocesses and normalizes NM scans, builds a study-specific
midbrain template, takes a single manual tracing of SN and cerebral peduncle (CP)
on that template, projects it back into each subject's native NM space, and
refines it there with a contrast-to-noise (CNR) criterion (Cassidy et al., 2019;
Wengler et al., 2020). A built-in evaluation stage quantifies agreement between
the resulting masks and the available reference — manual tracings when provided,
atlas-derived masks otherwise.

Developed at Brainhack Western 2026 and extended since.

---

## Method summary

The key idea is that the *shape* prior and the *intensity* evidence come from
different places. The prior comes from a group-level template on which SN and CP
are traced once; the evidence comes from each subject's own NM signal. CNR is
computed against the cerebral peduncle — white matter adjacent to SN with no
neuromelanin — so that the threshold means the same thing across subjects even
when absolute NM intensity does not.

Six masks are produced or warped per subject and compared against each other:

| Method | Origin |
|---|---|
| `NM` | CNR threshold inside the template-derived SN search region |
| `Core-rim` | As above, but the eroded core and the boundary rim use different CNR thresholds |
| `Atlas` | CIT168 binary SNc warped to native space |
| `ProbCIT` | CIT168 probabilistic SNc, thresholded in native space |
| `NMS` | Neuromelanin-sensitive probabilistic SNc atlas, thresholded in native space |
| `Anchored` | CIT168 atlas rigidly re-registered onto the subject's own NM seed |

When `--manual_dir` is given, each is also compared against the per-subject
manual tracing.

---

## Pipeline stages

| Stage | Flag | What it does |
|---|---|---|
| **1 — Forward normalization** | `--stage 1` | Realign NM runs (SPM), N4 bias correct, average; brain extract T1; register T1→MNI (brainstem-weighted SyN) and NM→T1 (rigid); compose transforms; write QC overlays and motion metrics; build the group-average NM template |
| **Segment** | `--stage segment` | Build the FireANTs V4 midbrain drawing template, align it to the group average, and open ITK-SNAP to trace CP (label 1) and SN (label 2) — **once per dataset** |
| **2 — Inverse normalization** | `--stage 2` | Register each subject to the group template, warp SN/CP back to native NM space, compute the CNR map and seed mask, derive the comparison masks, and write per-subject QC |
| **Group QC** | `--stage group_qc` | Rebuild dashboards and group plots from existing Stage 2 outputs. Cheap: reruns no registration |

`--stage all` runs 1, segment, 2 and group QC in order.

---

## Quick start

```bash
pip install -r requirements.txt
python workflow/tools/resample_mni_reference.py --out-dir resources/tpl-MNI152NLin2009cAsym

python workflow/scripts/neuromelanchor_pipeline.py \
  --project_root /path/to/this/repo \
  --design_dir   /path/to/bids_root \
  --output_root  /path/to/outputs \
  --work_dir     /path/to/work \
  --stage all --n_procs 10 --hires
```

`--project_root` must point at a checkout of this repository — the atlas priors
under `resources/` are resolved relative to it. See [docs/USAGE.md](docs/USAGE.md)
for every option, and [docs/INSTALL.md](docs/INSTALL.md) for the external
software (ANTs, SPM12/MATLAB, ITK-SNAP, and a container for FireANTs).

Try a single subject first with `--test`.

---

## Inputs

BIDS-like, one directory per subject (BioGen naming shown; `--cohort snit`
selects the other convention, and `--nm_pattern` / `--t1_pattern` override both):

```
sub-01/
└── anat/
    ├── sub-01_acq-CombEchoNM_run-01_GRE.nii.gz
    ├── sub-01_acq-CombEchoNM_run-02_GRE.nii.gz
    └── sub-01_acq-T1w.nii.gz
```

## Outputs

```
<output_root>/
├── sub-XXX/
│   ├── anat/         bias-corrected and mean NM, NM in MNI, native-space masks
│   ├── transforms/   composite and inverse ANTs warps (.h5)
│   ├── cnr/          CNR map and SN_Mask_QC.json
│   ├── metrics/      realignment parameters, motion stats and plot
│   └── qc/           registration overlays, disagreement maps, dashboard, manifest
├── Study-Specific_NM_Template.nii.gz
├── manual_trace_v4/  the SN/CP tracing, drawn once
├── drawing_template/ the V4 template and its build directory
└── Group_QC_*.{png,csv,json}
```

Full listing in [docs/DATA_SPECIFICATION.md](docs/DATA_SPECIFICATION.md).

---

## Repository structure

```
workflow/scripts/   the pipeline and the template builders
workflow/tools/     standalone utilities (resource generation, renaming, R2*)
workflow/workspace/ ITK-SNAP workspace template
resources/          atlas priors and MNI references
config/             example cohort configuration
docs/               documentation
legacy/             superseded implementations, kept for provenance
tests/              smoke checks and unit tests
```

See [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md).

---

## Documentation

| Document | Contents |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | Python environment and external software |
| [DATA_SPECIFICATION.md](docs/DATA_SPECIFICATION.md) | Expected inputs and every output produced |
| [USAGE.md](docs/USAGE.md) | Running each stage; full option reference |
| [WORKFLOW.md](docs/WORKFLOW.md) | What each processing step does and why |
| [MANUAL_SEGMENTATION.md](docs/MANUAL_SEGMENTATION.md) | The ITK-SNAP tracing protocol |
| [EVALUATION.md](docs/EVALUATION.md) | QC metrics and how to read the dashboards |
| [REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) | Where everything lives |
| [STATUS.md](docs/STATUS.md) | **What is implemented, what is unfinished, known limitations** |
| [PROVENANCE.md](docs/PROVENANCE.md) | Who wrote what, and why each version was chosen |
| [MIGRATION_AUDIT.md](docs/MIGRATION_AUDIT.md) | The audit behind the September 2026 reorganization |

New contributors should read `STATUS.md` before `USAGE.md` — several documented
behaviours are experimental.

---

## Checks

```bash
cd tests && python -m pytest
```

Needs only numpy, nibabel, scipy, matplotlib and pytest — no ANTs, SPM or data.
These are smoke checks, not validation of scientific output; see
[tests/README.md](tests/README.md).

---

## References

Berman, S., Drori, E., & Mezer, A. A. (2022). Spatial profiles provide sensitive MRI measures of the midbrain micro- and macrostructure. *NeuroImage*, 264, 119660. https://doi.org/10.1016/j.neuroimage.2022.119660

Cassidy, C. M., Zucca, F. A., Girgis, R. R., Baker, S. C., Weinstein, J. J., Sharp, M. E., Bellei, C., Valmadre, A., Vanegas, N., Kegeles, L. S., Brucato, G., Kang, U. J., Sulzer, D., Zecca, L., Abi-Dargham, A., & Horga, G. (2019). Neuromelanin-sensitive MRI as a noninvasive proxy measure of dopamine function in the human brain. *PNAS*, 116(11), 5108–5117. https://doi.org/10.1073/pnas.1807983116

Coulombe, V., Saikali, S., Goetz, L., Takech, M. A., Philippe, É., Parent, A., & Parent, M. (2021). A topographic atlas of the human brainstem in the ponto-mesencephalic junction plane. *Frontiers in Neuroanatomy*, 15, 627656. https://doi.org/10.3389/fnana.2021.627656

Nakamura, Y., Okada, N., Kunimatsu, A., Kasai, K., & Koike, S. (2018). Anatomical templates of the midbrain ventral tegmental area and substantia nigra for Asian populations. *Frontiers in Psychiatry*, 9. https://doi.org/10.3389/fpsyt.2018.00383

Pauli, W. M., Nili, A. N., & Tyszka, J. M. (2018). A high-resolution probabilistic in vivo atlas of human subcortical brain nuclei. *Scientific Data*, 5, 180063. https://doi.org/10.1038/sdata.2018.63

Smith, S. M., et al. (2004). Advances in functional and structural MR image analysis and implementation as FSL. *NeuroImage*, 23, S208–S219. https://doi.org/10.1016/j.neuroimage.2004.07.051

Wengler, K., He, X., Abi-Dargham, A., & Horga, G. (2020). Reproducibility assessment of neuromelanin-sensitive magnetic resonance imaging protocols for region-of-interest and voxelwise analyses. *NeuroImage*, 208, 116457. https://doi.org/10.1016/j.neuroimage.2019.116457

---

## License

MIT — see [LICENSE](LICENSE). Redistributed atlases keep their own licences; see
[resources/README.md](resources/README.md).
