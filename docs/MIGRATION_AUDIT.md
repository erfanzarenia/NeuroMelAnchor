# Migration Audit: NeuroMelAnchor

Audit performed before restructuring, comparing the live working project at
`/scratch/Projects/BrainHack_Nexus` (referred to below as **Nexus**) against the
GitHub repository `erfanzarenia/NeuroMelAnchor`.

Nexus was treated as strictly read-only; all work happens in a separate clone at
`/scratch/ezarenia/NeuroMelAnchor_cleanup/NeuroMelAnchor`.

---

## 1. Current Nexus state

Nexus is a ~650 GB analysis working directory. Only a thin slice of it is source
code; the rest is imaging data, derivatives and Nipype scratch.

### Code and configuration actually present

| Path | Lines | Modified | Role |
|---|---|---|---|
| `workflow/scripts/NeuroMel_Anchor_Pipeline_July-15th.py` | 3387 | 2026-09-11 | **Newest pipeline.** v2.5.0, FireANTs V4 drawing template, hi-res 0.5 mm outputs, NMS atlas QC |
| `workflow/scripts/NeuroMel_Anchor_Pipeline_July-10th.py` | 3049 | 2026-07-13 | Prior revision |
| `workflow/scripts/Neuromel_Anchor_Pipeline.py` | 3114 | 2026-07-09 | Prior revision |
| `workflow/scripts/BrainHack_NM_Nipype_Pipeline.py` | 2929 | 2026-05-08 | Revision tracked on GitHub `main` (locally further edited) |
| `workflow/scripts/BrainHack_NM_NiPyPe_Pipeline_Backup.py` | 1867 | 2026-04-15 | Snapshot backup |
| `workflow/scripts/BrainHack_NM_Nipype_Pipeline (copy).py` | 1389 | 2026-03-29 | Editor copy |
| `workflow/scripts/Template_Builder_V4.py` | 1003 | 2026-07-22 | **Newest MNI-space FireANTs template builder** |
| `workflow/scripts/Template_Builder_V3.py` | 990 | 2026-07-15 | Prior revision |
| `workflow/scripts/Template_Builder_v2.py` | 766 | 2026-07-14 | Prior revision |
| `workflow/scripts/Template_Builder.py` | 687 | 2026-06-10 | Revision tracked on GitHub `main` (locally further edited) |
| `workflow/scripts/Template_Builder_Native_V1.py` | 1163 | 2026-07-23 | **Separate native-space fork**, not a linear successor of V4 |
| `workflow/scripts/Native_Template_Builder_Preparation.py` | 348 | 2026-07-25 | Prior-mask prep for the native-space builder |
| `workflow/scripts/R2_Star_generation.py` | 167 | 2026-07-12 | Standalone R2* fitting utility |
| `workflow/scripts/Resample_T1.py` | 23 | 2026-05-25 | Generates the 0.5 mm MNI reference |
| `workflow/scripts/Rename.py` | 25 | 2026-07-20 | **Unfinished**: loop body is empty |
| `Extra_Functions/NM_Nipype_Pipeline.py` | ~400 | 2026-03-22 | Original ancestor pipeline (S. Marshall) |
| `Extra_Functions/BrainStem_Mask_Gen.py` | 20 | 2026-03-22 | Harvard-Oxford brainstem mask generator |
| `Extra_Functions/mp2rage_genUniDen.py` | 160 | 2026-03-28 | Third-party MP2RAGE denoising (O'Brien et al.) |
| `Extra_Functions/BioGen_Denoising.sh` | 20 | 2026-06-04 | Batch driver for the above |
| `Extra_Functions/NM_BioGen_ReName.py` | 60 | 2026-03-26 | BIDS renamer |
| `Scripts/NM_BioGen_ReName.py` | 60 | 2026-04-10 | Near-duplicate of the above (differs by one glob) |
| `Scripts/Generate_Mask_From_Brain-Navigator.py` | 45 | 2026-06-07 | Builds the midbrain prior from BrainstemNavigator |
| `workflow/workspace/NEXUS_NM_Template_ITKsnap.xml` | 260 | n/a | ITK-SNAP workspace template |
| `workflow/workspace/Automated_Setup.itksnap` | 260 | n/a | **Generated at runtime**, but committed |
| `workflow/requirements.txt` | 76 | 2026-03-29 | Pinned Python environment |

### Resources present in Nexus

`resources/tpl-MNI152NLin2009cAsym/` holds the atlas priors the pipeline reads:
CIT168 (`SNc.nii.gz`), CIT168 probabilistic (17 volumes + labels), Harvard-Oxford
`Brainstem.nii.gz`, a derived `Midbrain.nii.gz` and `Brainstem_Midbrain.nii.gz`,
`NMS_prob/NMS_Snc_prob.nii.gz`, and 1 mm / 0.5 mm MNI T1 references.
Also present: `NMS-SNc-atlas-master/` (upstream NMS atlas with its own LICENSE).

The three MNI manual traces (`MNI_SNc_Manual`, `MNI_CP_Manual`,
`MNI_Manual_Masks_combined`) were **moved on 2026-09-10** into
`resources/traces_superseded_2026-09-10/`. The current pipeline no longer reads
them from `resources/`; it reads and writes traces under
`<output_root>/manual_trace_v4/`.

### Not code (excluded from the repository)

`NM_Adults/`, `NM_Adolescents/`, `NM_PP/`, `NM_NotPP/`, `NM_BioGen/`,
`NM_BioGen_PP/`, `NM_SNIT_PP/`, `Normalized_Data/`, `to_be_organized/`,
`Backup/`, `Masks/`, `fireants/`, `fireants.sif` (3.7 GB), `workflow/.venv/`.
`Backup/Apr-15-2026/Output/` holds a complete group-QC result set from
2026-03-29, useful as a reference output example but far too large to commit.

---

## 2. Current GitHub state

`main` is at `98b5c76` (2026-05-08, Dylan). Tracked content:

```
LICENSE  README.md
docs/{INSTALL,USAGE,WORKFLOW,DATA_SPECIFICATION,EVALUATION,MANUAL_SEGMENTATION}.md
docs/requirements.txt
resources/tpl-MNI152NLin2009cAsym/{CIT168,CIT168_prob,HarvardOxford,MNI_*_Manual}
workflow/requirements.txt
workflow/scripts/{BrainHack_NM_Nipype_Pipeline.py, Template_Builder.py}
workflow/workspace/{Automated_Setup.itksnap, NEXUS_NM_Template_ITKsnap.xml}
```

Four branches exist:

- `main`: as above.
- `update-workflow-files`: merged into `main` via PR #1.
- `Post-BrainHack-Updates`: 2026-05-04, has `Template_Builder.py` at repo root and
  a duplicated `BrainHack_NM_Nipype_Pipeline_Updated.py`. Superseded by `main`.
- `modular_branch`: 2026-03-28, a **competing modular refactor** by Dylan:
  `run_pipeline.py`, `functions.py`, `phase1.py`, `phase2.py`, `segment.py`,
  plus `CHANGE_LOG.md` documenting a deliberate behaviour-preserving split.

Contributors in history: Samuel Marshall, Dylan Dinesh, rizviwazeen, Erfan Zarenia.

---

## 3. Major differences (Nexus vs GitHub)

**Nexus is ~4 months ahead on code; GitHub is ahead on documentation.**

Work that exists only in Nexus:

1. Three further pipeline generations (May → July), ending at v2.5.0. Added since
   the committed version: container runtime detection (FireANTs via
   Singularity/Apptainer/Docker), the FireANTs **V4 midbrain drawing template**
   workflow, `--hires` 0.5 mm re-warping, NMS probabilistic atlas QC, core/rim NM
   masking, `--nm_input combo|echo` selection, `--drawing_template v4|naive`,
   `--rebuild_template`, a `group_qc` stage, per-subject QC dashboards and a
   Stage-2 output manifest.
2. The entire FireANTs template-builder family (V1 → V4 plus the native-space
   fork and its preparation script). GitHub only has the June `Template_Builder.py`.
3. Standalone utilities: `R2_Star_generation.py`, `Resample_T1.py`,
   `Generate_Mask_From_Brain-Navigator.py`, `BrainStem_Mask_Gen.py`,
   `mp2rage_genUniDen.py`, `BioGen_Denoising.sh`, the two renamers.
4. Resource additions: `HarvardOxford/Midbrain.nii.gz`,
   `HarvardOxford/Brainstem_Midbrain.nii.gz`, `NMS_prob/NMS_Snc_prob.nii.gz`,
   the 1 mm and 0.5 mm MNI T1 references, `MNI/` hi-res ICBM T1.

Work that exists only on GitHub:

5. All six `docs/*.md` files. They describe the **March/May** pipeline and are now
   materially out of date: they document three stages (`1`/`segment`/`2`) where the
   code has four, name entrypoints that do not exist (`pipeline.py`,
   `run_pipeline.py`), and omit every option added since May.
6. The `modular_branch` refactor, which was never carried forward.

Divergence on shared files: the local copies of `BrainHack_NM_Nipype_Pipeline.py`
(156 changed lines) and `Template_Builder.py` (414 changed lines) are edited beyond
their committed state, and both are themselves superseded by newer Nexus files.

---

## 4. Duplicated / conflicting implementations

| Cluster | Members | Resolution |
|---|---|---|
| **Main pipeline** | 6 files, 1389 to 3387 lines | Strictly linear lineage confirmed by diff (281 / 233 / 460 changed lines between consecutive revisions). Keep **July-15th** as canonical. Archive the original ancestor; drop intermediate revisions (recoverable from Nexus and Git history). |
| **MNI template builder** | `Template_Builder.py` → `v2` → `V3` → `V4` | Linear (359 / 326 / 41 changed lines). Keep **V4**. V3→V4 is a 41-line delta, so V4 is a safe successor. |
| **Native template builder** | `Template_Builder_Native_V1.py` + `Native_Template_Builder_Preparation.py` | **Not** a successor to V4: 394 changed lines, different input model (native-space subject crops, per-subject centroid alignment). Genuinely parallel work. **Keep both**, marked experimental. |
| **BIDS renamer** | `Extra_Functions/NM_BioGen_ReName.py` vs `Scripts/NM_BioGen_ReName.py` | Differ by one glob pattern (`*MP2RAGE_UNI_Images*.nii` vs `*LH.nii`) and one whitespace line. Merge into one script with the pattern as a CLI argument. |
| **Modular refactor** | `modular_branch` (`phase1/phase2/functions/segment/run_pipeline`) | A faithful split of the **March** monolith; ~9 months of pipeline development have happened since, and it shares no code with the current script. Cannot be merged forward without redoing the split. **Preserve under `legacy/` with its CHANGE_LOG**, documented as the reference design for a future modularization. |
| **Generated ITK-SNAP workspace** | `Automated_Setup.itksnap` tracked *and* regenerated at runtime | Stop tracking; regenerate under the output root. |
| **requirements.txt** | `workflow/requirements.txt` and `docs/requirements.txt` | Identical purpose. Consolidate to one at the repository root. |

---

## 5. Known problems found during the audit

- Hard-coded absolute `/scratch/Projects/BrainHack_Nexus/...` paths in
  `Template_Builder_V4.py`, `Template_Builder_Native_V1.py`,
  `Native_Template_Builder_Preparation.py`, `R2_Star_generation.py`,
  `Rename.py`, `Resample_T1.py`, `Generate_Mask_From_Brain-Navigator.py`,
  `BrainStem_Mask_Gen.py`, `BioGen_Denoising.sh`. The main pipeline itself is clean.
- `Rename.py` is dead: the subject loop assigns one unused f-string and does nothing.
- Cohort-specific `SelectFiles` patterns are switched by commenting/uncommenting
  blocks (pipeline lines ~1496 to 1527) rather than by configuration.
- `BioGen_Denoising.sh` iterates a glob hard-wired to a single subject (`sub-782`).
- The `segment` stage writes its generated workspace back into the repository tree.
- No tests of any kind exist.

---

## 6. Proposed final structure

```
NeuroMelAnchor/
├── README.md                  project overview, quick start, status
├── CHANGELOG.md               what this cleanup merged
├── LICENSE
├── requirements.txt           single pinned environment
├── config/
│   └── example_cohort.yaml    documented example of a cohort configuration
├── docs/
│   ├── INSTALL.md             environment and external software
│   ├── DATA_SPECIFICATION.md  expected inputs and produced outputs
│   ├── USAGE.md               how to run each stage
│   ├── WORKFLOW.md            conceptual description of the method
│   ├── MANUAL_SEGMENTATION.md ITK-SNAP tracing protocol
│   ├── EVALUATION.md          QC metrics
│   ├── REPOSITORY_STRUCTURE.md
│   ├── STATUS.md              implemented vs unfinished, known limitations
│   ├── PROVENANCE.md          who wrote what, and why versions were chosen
│   └── MIGRATION_AUDIT.md     this document
├── workflow/
│   ├── scripts/
│   │   ├── neuromelanchor_pipeline.py      canonical pipeline (was July-15th)
│   │   ├── template_builder.py             canonical MNI builder (was V4)
│   │   ├── template_builder_native.py      experimental native builder
│   │   └── native_template_prep.py         prior preparation for the above
│   ├── tools/                              standalone, single-purpose utilities
│   │   ├── make_midbrain_prior.py
│   │   ├── make_brainstem_mask.py
│   │   ├── resample_mni_reference.py
│   │   ├── fit_r2star.py
│   │   ├── mp2rage_uniden.py
│   │   ├── mp2rage_uniden_batch.sh
│   │   └── bids_rename.py
│   └── workspace/
│       └── NEXUS_NM_Template_ITKsnap.xml   template only; no generated files
├── resources/
│   └── tpl-MNI152NLin2009cAsym/…           atlas priors and MNI references
├── legacy/                                 preserved, not maintained
│   ├── README.md
│   ├── NM_Nipype_Pipeline.py               original ancestor
│   └── modular/                            modular_branch refactor + CHANGE_LOG
└── tests/
    ├── README.md
    ├── conftest.py                         loads pipeline helpers without Nipype
    ├── test_repo_layout.py                 structure + resource manifest checks
    ├── test_scripts_parse.py               every script compiles; CLI contracts
    └── test_mask_helpers.py                numeric tests of mask/QC helpers
```

Deliberately **not** migrated: imaging data, derivatives, Nipype work directories,
`fireants.sif`, the `.venv`, and `Backup/`.

---

## Addendum: 2026-09-14 sync

A second comparison, three days after the audit above, found these changes in Nexus:

| Path | Change |
|---|---|
| `workflow/scripts/NeuroMel_Anchor_Pipeline_July-15th.py` | Edited from 2026-09-12 to 2026-09-14 and now v3.0 (153,753 to 181,346 bytes). Adds `--search_dilation_mm`, closing of the NM mask, poster figures and an NM-blind control in group QC, fuller run logging, and the authors' note on limitations |
| `workflow/scripts/Posters_Trajectory_Plots.py` | New: plots of the NM mask's metrics across a search-dilation sweep |
| `workflow/scripts/BrainHack_Poster_Workflow_Methods.py` | New: method panels for the poster (edited again at 16:51 to drop the white border) |
| `NM_BioGen_PP/echo-1/` | New outputs from the dilation sweep (`results_dil-*`), plus earlier `results_preclose_*` runs from before the closing step |
| `NM_SNIT_PP/` | One subdirectory (`SNIT_adult_combo-mean`) was moved or renamed; data only |

Nothing else had changed. The Nexus Git repository had no new commits or refs, and
GitHub was unchanged.

The sync also found three errors in the September 11 work, all corrected on
2026-09-14:

- The superseded tracings copied into the repository were the March upload, not the
  July files retired on 2026-09-10.
- Several documented outputs came from an April run rather than from the current code.
- The contributor history misattributed the development since Brainhack.

Details are in [PROVENANCE.md](PROVENANCE.md) and [../CHANGELOG.md](../CHANGELOG.md).
