# Usage

Running the pipeline. See [INSTALL.md](INSTALL.md) for setup and
[DATA_SPECIFICATION.md](DATA_SPECIFICATION.md) for input layout.

---

## The four stages

```bash
python workflow/scripts/neuromelanchor_pipeline.py \
  --project_root /path/to/NeuroMelAnchor \
  --design_dir   /path/to/bids_root \
  --output_root  /path/to/outputs \
  --work_dir     /path/to/work \
  --stage all
```

`--stage` selects `1`, `segment`, `2`, `group_qc` or `all` (default).

The four paths are all required:

| Option | Meaning |
|---|---|
| `--project_root` | A checkout of **this repository**. Atlas priors under `resources/` are resolved from it, and the V4 builder is launched from `workflow/scripts/` under it |
| `--design_dir` | BIDS root containing `sub-*` directories |
| `--output_root` | Where all outputs are written. Must sit **inside** `--project_root` if you use the V4 drawing template, so the container can see it |
| `--work_dir` | Nipype working directory. Large and disposable |

Start with one subject:

```bash
--stage 1 --test
```

---

## Stage 1 — forward normalization

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage 1 --n_procs 10 --hires
```

Realigns and bias-corrects the NM runs, brain-extracts the T1, registers T1→MNI
and NM→T1, composes the transforms, writes QC and motion metrics, and builds the
group-average NM template.

Requires ANTs and MATLAB/SPM12 on `PATH`. Pass `--hires` if you intend to use the
V4 drawing template — it writes the 0.5 mm `NM_MNI_res-05.nii` that Stage
`segment` needs from **every** subject.

## Stage segment — manual tracing

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage segment
```

Builds the FireANTs V4 midbrain template (~35 min on a GPU), aligns it to the
group average, and opens ITK-SNAP. Trace **CP as label 1** and **SN as label 2**,
save with Ctrl+S, and close ITK-SNAP to let the pipeline continue. Done **once
per dataset**; the tracing is reused by every later run.

See [MANUAL_SEGMENTATION.md](MANUAL_SEGMENTATION.md) for the protocol.

The tracing lives in `<output_root>/manual_trace_v4/`. Use
`--drawing_template naive` (or `--skip_template_builder`) to trace on the plain
group mean instead, which writes to `<output_root>/manual_trace/` and needs no
GPU or container.

## Stage 2 — inverse normalization and seed extraction

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage 2 \
  --manual_dir /path/to/manual_masks
```

Registers each subject to the group template, warps SN and CP back to native NM
space, computes the CNR map and seed mask, derives the comparison masks, and
writes per-subject QC. `--manual_dir` is optional; without it, evaluation is
relative to the atlas.

## Stage group_qc — rebuild summaries

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage group_qc
```

Rebuilds the dashboards and group plots from Stage 2 outputs already on disk.
Reruns no registration, so it is the right stage to use after changing a plot.

---

## Option reference

### Execution

| Option | Default | Purpose |
|---|---|---|
| `--n_procs` | `10` | Parallel Nipype processes. ITK/OMP thread count is capped at `min(n_procs, 8)` |
| `--stage` | `all` | `1`, `segment`, `2`, `group_qc`, `all` |
| `--test` | off | First subject only; group QC outputs get a `_TEST` suffix |
| `--smooth` | `0` | Gaussian FWHM in mm applied after the MNI warp. `0` skips smoothing |

### Input selection

| Option | Default | Purpose |
|---|---|---|
| `--cohort` | `biogen` | Filename convention: `biogen` or `snit` |
| `--nm_input` | `combo` | `combo` = vendor-combined multi-echo GRE (higher SNR); `echo` = a single MEGRE echo |
| `--nm_echo` | `3` | Echo number used when `--nm_input echo` |
| `--nm_pattern` | — | Override the NM glob. `{subject_id}` and `{echo}` are substituted |
| `--t1_pattern` | — | Override the T1 glob. `{subject_id}` is substituted |
| `--nm_image_strategy` | `mean` | `mean` averages all realigned acquisitions; `first` uses only the first |
| `--nm_image_n` | all | In `mean` mode, average only the first N acquisitions |

To add a cohort, add an entry to `COHORT_INPUT_PATTERNS` in the pipeline rather
than editing the workflow.

### Templates and resolution

| Option | Default | Purpose |
|---|---|---|
| `--hires` | off | Also write `NM_MNI_res-05.nii` at 0.5 mm by reapplying the saved transforms |
| `--drawing_template` | `v4` | `v4` = FireANTs midbrain template; `naive` = plain group mean |
| `--skip_template_builder` | off | Implies `--drawing_template naive` |
| `--rebuild_template` | off | Force a rebuild and realignment of V4. Refuses to run if a tracing already exists — move it aside first |

### Evaluation

| Option | Default | Purpose |
|---|---|---|
| `--manual_dir` | — | Per-subject manual SN masks, as `<sub>/anat/<sub>_space-NM_label-SN_desc-manual_mask.nii*` |
| `--prob_atlas_mni` | CIT168 prob SNc | Probabilistic SNc atlas for the `ProbCIT` comparison |
| `--prob_atlas_threshold` | `0.25` | Threshold applied after warping it to native space |
| `--skip_prob_atlas` | off | Disable the `ProbCIT` comparison |
| `--nms_atlas_mni` | NMS prob SNc | Probabilistic NMS SNc atlas for the `NMS` comparison |
| `--nms_atlas_threshold` | `0.25` | Threshold applied after warping it to native space |
| `--skip_nms_atlas` | off | Disable the `NMS` comparison |
| `--skip_core_rim_nm` | off | Disable core/rim mask generation |

Masking constants (`seed_cnr_floor`, `seed_cnr_percentile`,
`core_rim_erosion_mm`, ...) are module-level in the pipeline, not CLI options.
They are echoed into the run log so any result can be traced back to the values
used. See [STATUS.md](STATUS.md).

---

## Template builders

Normally invoked by `--stage segment`. To run one directly, inside the FireANTs
container:

```bash
singularity exec --nv --bind /path/to/project fireants.sif python \
  workflow/scripts/template_builder.py \
    --output-root  /path/to/outputs \
    --work-dir     /path/to/outputs/drawing_template/build \
    --output-template /path/to/outputs/drawing_template/SN_Template_V4.nii.gz \
    --iterations 12 --rigid --shape-avg --shape-avg-step 0.25 --laplacian 0
```

`--mni-prior` defaults to `resources/.../HarvardOxford/Midbrain.nii.gz`, resolved
from the script's own location.

`template_builder_native.py` and `native_template_prep.py` build a template on a
chosen subject's native grid instead of in MNI space. They are experimental and
are not called by the pipeline — see [STATUS.md](STATUS.md).

---

## Standalone tools

See [../workflow/tools/README.md](../workflow/tools/README.md). Every tool takes
its paths on the command line; run any with `--help`.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No sub-* directories found` | `--design_dir` is not the BIDS root |
| `ANTs not found on path` | `ANTSPATH` not exported, or ANTs not installed |
| `Matlab not found on path` | Stage 1 needs MATLAB with SPM12 on the MATLAB path |
| `ITK-SNAP not found on PATH` | Needed by `--stage segment` |
| `0.5 mm MNI reference not found` | Run `workflow/tools/resample_mni_reference.py` |
| `V4 needs a 0.5 mm NM_MNI_res-05.nii for every subject` | Re-run Stage 1 with `--hires` for the listed subjects |
| `--output_root must sit inside --project_root` | The container binds `--project_root` only |
| `Move your existing tracing out of ...` | `--rebuild_template` would invalidate the tracing |
| `... is not on the V4 drawing grid` | The tracing was drawn on a different template; move it aside or rebuild |
| `Segmentation Mask is Empty` | ITK-SNAP was closed without saving |
| `One or both masks have very few voxels` | Under 50 voxels in CP or SN; inspect before continuing |
| `SN mask missing` / `Group Average NM Template missing` | Run the earlier stage first |
| Out of memory | Lower `--n_procs` |
