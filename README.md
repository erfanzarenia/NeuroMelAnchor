# NeuroMelAnchor

### Subject-specific, neuromelanin-guided localization of the substantia nigra

---

## Overview

NeuroMelAnchor is a Nipype pipeline for subject-specific localization and
segmentation of the substantia nigra (SN) using neuromelanin-sensitive MRI
(NM-MRI). It combines an atlas-based prior with each subject's own neuromelanin
contrast to improve anatomical specificity when defining this small midbrain
structure.

Accurate delineation of the SN matters for research on dopaminergic systems,
including learning, motivation, aging and mental health. Atlas-based approaches
provide convenient region definitions, but inter-individual anatomical variability
limits them, and the limitation is worst for small subcortical nuclei. NM-MRI
provides contrast related to dopaminergic neurons, so localization can be guided by
signal present in each individual brain (Berman et al., 2022; Coulombe et al., 2021;
Nakamura et al., 2018).

The workflow preprocesses and normalizes NM scans, builds a study-specific midbrain
template, takes a single manual tracing of SN and cerebral peduncle (CP) on that
template, projects the tracing back into each subject's native NM space, and refines
it there using contrast against the CP (Cassidy et al., 2019; Wengler et al., 2020).
A built-in evaluation stage measures agreement between the resulting masks and the
available references: manual tracings when provided, atlas-derived masks otherwise.

## Background

NeuroMelAnchor was started at Brainhack Western 2026, where it won the Excellence in
Innovation award. It has since been developed substantially, mostly by Sam Marshall
(@smarsh127), with input from Dr. Bruce Morton. The full history is in
[docs/PROVENANCE.md](docs/PROVENANCE.md).

---

## Method summary

The shape prior and the intensity evidence come from different places. The prior
comes from a group-level template on which SN and CP are traced once. The evidence
comes from each subject's own NM signal, expressed relative to the cerebral peduncle,
which is white matter next to the SN with no neuromelanin. Dividing by the CP
reference removes scanner and subject scaling, so one threshold rule can be applied
to every subject.

The measure is a contrast ratio, (NM - CP) / CP. File names and QC keys call it
"CNR" for historical reasons, but there is no noise term.

Six masks are produced or warped for each subject and compared with each other:

| Method | Origin |
|---|---|
| `NM` | Contrast threshold inside the search region: the warped SN tracing, optionally grown by `--search_dilation_mm` |
| `Core-rim` | As above, but the eroded core and the boundary rim use different thresholds |
| `Atlas` | CIT168 binary SNc warped to native space |
| `ProbCIT` | CIT168 probabilistic SNc, thresholded in native space |
| `NMS` | Neuromelanin-sensitive probabilistic SNc atlas, thresholded in native space |
| `Anchored` | CIT168 SNc rigidly re-registered onto the subject's own NM mask |

When `--manual_dir` is given, each is also compared with the subject's manual tracing.

---

## Pipeline stages

| Stage | Flag | What it does |
|---|---|---|
| **1: Forward normalization** | `--stage 1` | Realign the NM acquisitions (SPM), bias correct (N4) and average them; brain-extract the T1; register T1 to MNI (brainstem-weighted SyN) and NM to T1 (rigid); compose the transforms; write registration and motion QC; build the group-average NM template |
| **Segment** | `--stage segment` | Build the FireANTs V4 midbrain drawing template, align it to the group average, and open ITK-SNAP to trace CP (label 1) and SN (label 2). Done **once per dataset** |
| **2: Inverse normalization** | `--stage 2` | Register each subject to the group template, warp SN and CP back to native NM space, build the search region, compute the contrast map and NM mask, derive the comparison masks, and score them |
| **Group QC** | `--stage group_qc` | From existing Stage 2 outputs, build per-subject dashboards, output manifests and group plots, plus the poster figures and an NM-blind control when manual masks are available. Runs no registration |

`--stage all` runs 1, segment, 2 and group QC in order. `--stage 2` also runs group QC.

---

## Quick start

```bash
pip install -r requirements.txt
python workflow/tools/resample_mni_reference.py --out-dir resources/tpl-MNI152NLin2009cAsym

python workflow/scripts/neuromelanchor_pipeline.py \
  --project_root /path/to/NeuroMelAnchor \
  --design_dir   /path/to/bids_root \
  --output_root  /path/to/NeuroMelAnchor/outputs \
  --work_dir     /path/to/work \
  --stage all --n_procs 10 --hires
```

`--project_root` must point at a checkout of this repository, because the atlas priors
under `resources/` are resolved relative to it. Try a single subject first with
`--test`. See [docs/USAGE.md](docs/USAGE.md) for every option and
[docs/INSTALL.md](docs/INSTALL.md) for the external software (ANTs, SPM12 with MATLAB,
ITK-SNAP, and a container for FireANTs).

---

## Inputs

BIDS-like, one directory per subject. BioGen naming is shown; `--cohort snit` selects
the other convention, and `--nm_pattern` and `--t1_pattern` override both.

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
│   ├── anat/         NM images, the search region, the NM mask and every comparison mask
│   ├── transforms/   Stage 1 and anchoring transforms (.h5)
│   ├── cnr/          contrast map, seed diagnostics and all pairwise metrics
│   ├── metrics/      realignment parameters and motion ranges
│   └── qc/           registration overlays, histograms, dashboard and manifest
├── Study-Specific_NM_Template.nii.gz
├── drawing_template/ the V4 template
├── manual_trace_v4/  the SN and CP tracing, drawn once
├── Group_QC_*        group plots and summary tables
├── Poster_*          poster figures and the NM-blind control (with manual masks)
└── NeuroMelAnchor_master.log
```

Every file is listed in [docs/DATA_SPECIFICATION.md](docs/DATA_SPECIFICATION.md).

---

## Repository structure

```
workflow/scripts/   the pipeline and the template builders
workflow/tools/     standalone utilities: resource generation, renaming, R2* fitting
workflow/figures/   post-hoc figure scripts made for the Brainhack Western 2026 poster
workflow/workspace/ ITK-SNAP workspace template
resources/          atlas priors, MNI references and retired tracings
config/             example cohort configuration (reference only)
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
| [USAGE.md](docs/USAGE.md) | Running each stage, the full option reference, and search-region sweeps |
| [WORKFLOW.md](docs/WORKFLOW.md) | What each processing step does and why |
| [MANUAL_SEGMENTATION.md](docs/MANUAL_SEGMENTATION.md) | The ITK-SNAP tracing protocol |
| [EVALUATION.md](docs/EVALUATION.md) | QC metrics, flags, poster figures, and how to read them |
| [REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) | Where everything lives |
| [STATUS.md](docs/STATUS.md) | **What is implemented, what is unfinished, and known limitations** |
| [PROVENANCE.md](docs/PROVENANCE.md) | Project history, contributors, and why each version was chosen |
| [MIGRATION_AUDIT.md](docs/MIGRATION_AUDIT.md) | The audit behind the September 2026 reorganization |
| [CHANGELOG.md](CHANGELOG.md) | What changed, and when |

New contributors should read `STATUS.md` before `USAGE.md`: several behaviours are
experimental, and the authors' own notes on the thresholding are summarized there.

---

## Checks

```bash
cd tests && python -m pytest
```

These need only the Python packages in `requirements.txt` plus pytest: no ANTs, SPM,
ITK-SNAP, GPU or imaging data. They are smoke checks and unit tests, not a validation
of scientific output; see [tests/README.md](tests/README.md).

---

## Credits

NeuroMelAnchor began as a Brainhack Western 2026 project and won the event's
Excellence in Innovation award. Most of its development since then is the work of Sam
Marshall (@smarsh127), with input from Dr. Bruce Morton. Other contributors are listed
in [docs/PROVENANCE.md](docs/PROVENANCE.md).

---

## References

Berman, S., Drori, E., & Mezer, A. A. (2022). Spatial profiles provide sensitive MRI measures of the midbrain micro- and macrostructure. *NeuroImage*, 264, 119660. https://doi.org/10.1016/j.neuroimage.2022.119660

Cassidy, C. M., Zucca, F. A., Girgis, R. R., Baker, S. C., Weinstein, J. J., Sharp, M. E., Bellei, C., Valmadre, A., Vanegas, N., Kegeles, L. S., Brucato, G., Kang, U. J., Sulzer, D., Zecca, L., Abi-Dargham, A., & Horga, G. (2019). Neuromelanin-sensitive MRI as a noninvasive proxy measure of dopamine function in the human brain. *PNAS*, 116(11), 5108-5117. https://doi.org/10.1073/pnas.1807983116

Coulombe, V., Saikali, S., Goetz, L., Takech, M. A., Philippe, É., Parent, A., & Parent, M. (2021). A topographic atlas of the human brainstem in the ponto-mesencephalic junction plane. *Frontiers in Neuroanatomy*, 15, 627656. https://doi.org/10.3389/fnana.2021.627656

Nakamura, Y., Okada, N., Kunimatsu, A., Kasai, K., & Koike, S. (2018). Anatomical templates of the midbrain ventral tegmental area and substantia nigra for Asian populations. *Frontiers in Psychiatry*, 9. https://doi.org/10.3389/fpsyt.2018.00383

Pauli, W. M., Nili, A. N., & Tyszka, J. M. (2018). A high-resolution probabilistic in vivo atlas of human subcortical brain nuclei. *Scientific Data*, 5, 180063. https://doi.org/10.1038/sdata.2018.63

Smith, S. M., et al. (2004). Advances in functional and structural MR image analysis and implementation as FSL. *NeuroImage*, 23, S208-S219. https://doi.org/10.1016/j.neuroimage.2004.07.051

Wengler, K., He, X., Abi-Dargham, A., & Horga, G. (2020). Reproducibility assessment of neuromelanin-sensitive magnetic resonance imaging protocols for region-of-interest and voxelwise analyses. *NeuroImage*, 208, 116457. https://doi.org/10.1016/j.neuroimage.2019.116457

---

## License

See [LICENSE](LICENSE). Redistributed atlases keep their own licences; see
[resources/README.md](resources/README.md).
