# Usage

How to run the pipeline. See [INSTALL.md](INSTALL.md) for setup and
[DATA_SPECIFICATION.md](DATA_SPECIFICATION.md) for the input layout.

---

## Running the pipeline

```bash
python workflow/scripts/neuromelanchor_pipeline.py \
  --project_root /path/to/NeuroMelAnchor \
  --design_dir   /path/to/bids_root \
  --output_root  /path/to/NeuroMelAnchor/outputs \
  --work_dir     /path/to/work \
  --stage all
```

`--stage` selects `1`, `segment`, `2`, `group_qc` or `all` (the default).

All four paths are required:

| Option | Meaning |
|---|---|
| `--project_root` | A checkout of **this repository**. Atlas priors under `resources/` are resolved from it, and the V4 builder is launched from `workflow/scripts/` inside it |
| `--design_dir` | BIDS root containing the `sub-*` directories |
| `--output_root` | Where all outputs are written. It must sit **inside** `--project_root` when the V4 drawing template is built, because the container mounts only `--project_root` |
| `--work_dir` | Nipype working directory: large, disposable, and the cache that lets reruns skip finished steps |

Try a single subject first by adding `--test`.

Every run appends to `<output_root>/NeuroMelAnchor_master.log`, which records the
options, the masking constants, the script path and the full command line.

---

## Stage 1: forward normalization

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage 1 --n_procs 10 --hires
```

Realigns and bias-corrects the NM acquisitions, brain-extracts the T1, registers T1
to MNI and NM to T1, composes the transforms, writes registration and motion QC, and
builds the group-average NM template.

Needs ANTs, and MATLAB with SPM12, on `PATH`. Pass `--hires` if you plan to use the V4
drawing template: it writes the 0.5 mm `NM_MNI_res-05.nii` that `--stage segment`
needs for **every** subject.

## Stage segment: manual tracing

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage segment
```

Builds the FireANTs V4 midbrain template (about 35 minutes on a GPU), aligns it to the
group average, and opens ITK-SNAP. Trace **CP as label 1** and **SN as label 2**, save
with Ctrl+S, and close ITK-SNAP to let the pipeline continue. This is done **once per
dataset**; every later run reuses the tracing. See
[MANUAL_SEGMENTATION.md](MANUAL_SEGMENTATION.md) for the protocol.

The tracing lives in `<output_root>/manual_trace_v4/`. Use `--drawing_template naive`
(or `--skip_template_builder`) to trace on the plain group mean instead; it writes to
`<output_root>/manual_trace/` and needs neither a GPU nor a container.

## Stage 2: inverse normalization and NM masks

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage 2 \
  --manual_dir /path/to/manual_masks
```

Registers each subject to the group template, warps SN and CP back to native NM
space, builds the search region, computes the contrast map and the NM mask, derives
the comparison masks, and scores them. `--manual_dir` is optional; without it the
masks are compared only with each other and with the atlases.

`--search_dilation_mm` grows the propagated SN tracing before the NM threshold is
applied. The default of 0 uses the tracing as drawn. See [WORKFLOW.md](WORKFLOW.md)
and the sweep section below.

The October 2026 runs used:

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage 2 \
  --dataset biogen --nm_input echo --nm_echo 1 --nm_image_strategy first \
  --search_dilation_mm 0.5 --seed_cnr_fwhm 1.5 --seed_slice_fill --seed_cnr_floor 0 \
  --seed_image raw --trace_route template
```

`--stage 2` runs group QC when it finishes.

## Stage group_qc: dashboards, group plots and poster figures

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage group_qc \
  --manual_dir /path/to/manual_masks
```

Builds the per-subject dashboards, output manifests and group plots from Stage 2
outputs already on disk. With `--manual_dir` it also writes the poster figures and
the NM-blind control. It reruns no registration, so it is the stage to use after
changing a plot. See [EVALUATION.md](EVALUATION.md).

---

## Option reference

### Execution

| Option | Default | Purpose |
|---|---|---|
| `--n_procs` | `10` | Parallel Nipype processes. ITK and OpenMP threads are capped at `min(n_procs, 8)` |
| `--stage` | `all` | `1`, `segment`, `2`, `group_qc` or `all` |
| `--test` | off | First subject only; group QC outputs get a `_TEST` suffix |
| `--smooth` | `0` | Gaussian FWHM in mm applied after the MNI warp. `0` skips smoothing |

### Input selection

| Option | Default | Purpose |
|---|---|---|
| `--dataset` | `biogen` | File naming convention for the NM echo and T1 images: `biogen` or `snit` |
| `--nm_input` | `combo` | `combo` is the vendor-combined multi-echo GRE (higher SNR); `echo` is a single MEGRE echo |
| `--nm_echo` | `3` | Echo number used when `--nm_input echo` |
| `--nm_image_strategy` | `mean` | `mean` averages all realigned acquisitions; `first` uses only the first |
| `--nm_image_n` | all | In `mean` mode, average only the first N acquisitions |

Another dataset means another branch where `--dataset` is handled at the start of the
Stage 1 workflow.

### Templates and resolution

| Option | Default | Purpose |
|---|---|---|
| `--hires` | off | Also write `NM_MNI_res-05.nii` at 0.5 mm by reapplying the saved transforms |
| `--drawing_template` | `v4` | `v4` is the FireANTs midbrain template; `naive` is the plain group mean |
| `--skip_template_builder` | off | Implies `--drawing_template naive` |
| `--rebuild_template` | off | Force a rebuild and realignment of V4. Refuses to run while a tracing exists; move the tracing aside first |

### NM mask

| Option | Default | Purpose |
|---|---|---|
| `--search_dilation_mm` | `0` | Grow the propagated SN search region by this many mm before the NM threshold. `0` keeps the tracing as drawn |
| `--seed_cnr_floor` | `0.05` | Absolute minimum contrast ratio for a voxel to enter the NM mask |
| `--seed_cnr_percentile` | `55` | Subject-adaptive percentile of the search region's contrast; voxels above it are kept. Set it low (0 to 5) to let the floor define the mask |
| `--seed_min_cluster_voxels` | `15` | Drop connected clusters smaller than this |
| `--seed_components_to_keep` | `2` | Keep only the largest N components; 2 for a bilateral SN |
| `--seed_cnr_fwhm` | `1.0` | Smoothing of the contrast map before thresholding, in mm FWHM |
| `--seed_slice_fill` | off | Also fill holes within each axial slice of the final mask |
| `--seed_image` | `n4` | Build the mask from the stage 1 bias-corrected image (`n4`) or from the same run without bias correction (`raw`, which needs `--nm_image_strategy first`) |
| `--trace_route` | `template` | How the traces reach NM space: through the stage 2 registration to the group template (`template`), or through the stage 1 NM to T1 and T1 to MNI transforms (`t1`) |
| `--skip_core_rim_nm` | off | Disable the core-rim mask |

The core-rim thresholds (`core_rim_erosion_mm` and the others) are module-level constants
in the pipeline rather than options. Every value is echoed into the run log, so any result can
be traced back to the settings used.

### T1w-space outputs

| Option | Default | Purpose |
|---|---|---|
| `--skip_t1_outputs` | off | Do not write the T1w-space copy of the NM mask |
| `--t1_interp` | `linear` | How the mask is moved to T1 space: `linear`, then kept at 0.5 or above, or `nearest` |
| `--t1_seed_res` | T1 grid | Voxel size in mm for the T1w-space mask, on a grid covering the T1's field of view |

### Evaluation

| Option | Default | Purpose |
|---|---|---|
| `--manual_dir` | none | Per-subject manual SN masks, as `<sub>/anat/<sub>_space-NM_label-SN_desc-manual_mask.nii*` |
| `--prob_atlas_mni` | CIT168 probabilistic SNc | Probabilistic SNc atlas for the `ProbCIT` comparison |
| `--prob_atlas_threshold` | `0.25` | Threshold applied after warping it to native space |
| `--skip_prob_atlas` | off | Disable the `ProbCIT` comparison |
| `--nms_atlas_mni` | NMS probabilistic SNc | Probabilistic atlas for the `NMS` comparison |
| `--nms_atlas_threshold` | `0.25` | Threshold applied after warping it to native space |
| `--skip_nms_atlas` | off | Disable the `NMS` comparison |

---

## Running a search-region sweep

To compare `--search_dilation_mm` values, run Stage 2 once per value. Stage 2 reads
the Stage 1 outputs from `--output_root`, so every run uses the same output root and
work directory, and each run's outputs are archived before the next one starts.
Nipype reuses the cached registrations, so only the search region and the steps that
depend on it are recomputed.

For the Brainhack Western 2026 poster this was done at 0, 0.5, 1.0, 1.5, 2.0, 2.5 and
3.0 mm on the BioGen 7T echo-1 data (25 subjects). Each run was archived to a
`results_dil-<mm>` directory, and `--stage group_qc` was then run on each archive:

```bash
for mm in 0 0.5 1.0 1.5 2.0 2.5 3.0; do
  python workflow/scripts/neuromelanchor_pipeline.py ... --stage 2 --search_dilation_mm "$mm"
  # archive this run's outputs to /path/to/runs/results_dil-$mm
done

python workflow/figures/poster_dilation_sweep.py --runs-root /path/to/runs
```

The poster used 0.5 mm.

---

## Template builders

Normally run by `--stage segment`. To run one directly inside the FireANTs container:

```bash
singularity exec --nv --bind /path/to/project fireants.sif python \
  workflow/scripts/template_builder.py \
    --output-root  /path/to/outputs \
    --work-dir     /path/to/outputs/drawing_template/build \
    --output-template /path/to/outputs/drawing_template/SN_Template_V4.nii.gz \
    --iterations 12 --rigid --shape-avg --shape-avg-step 0.25 --laplacian 0
```

`--mni-prior` defaults to `resources/tpl-MNI152NLin2009cAsym/HarvardOxford/Midbrain.nii.gz`,
resolved from the script's own location.

`template_builder_native.py` and `native_template_prep.py` build a template on a
chosen subject's native grid instead of in MNI space. They are experimental and are
not called by the pipeline; see [STATUS.md](STATUS.md).

---

## Figure scripts

`workflow/figures/` holds the post-hoc scripts that made the poster's method panels
and dilation-sweep plots. They read finished outputs and never run the pipeline. See
[../workflow/figures/README.md](../workflow/figures/README.md).

## Standalone tools

`workflow/tools/` holds utilities for preparing inputs and rebuilding resources. Every
tool takes its paths on the command line; run any with `--help`. See
[../workflow/tools/README.md](../workflow/tools/README.md).

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No sub-* directories found` | `--design_dir` is not the BIDS root |
| `ANTs not found on path` | `ANTSPATH` not exported, or ANTs not installed |
| `Matlab not found on path` | Stage 1 needs MATLAB with SPM12 on the MATLAB path |
| `ITK-SNAP not found on PATH` | Needed by `--stage segment` |
| `0.5mm MNI reference not found` | Run `workflow/tools/resample_mni_reference.py` |
| `V4 needs a 0.5 mm NM_MNI_res-05.nii image for every subject` | Rerun Stage 1 with `--hires` for the listed subjects |
| `--output_root must sit inside --project_root` | The container mounts only `--project_root` |
| `Move your existing tracing out of ...` | `--rebuild_template` would invalidate the tracing |
| `... is not on the V4 drawing grid` | The tracing was drawn on a different template; move it aside or rebuild |
| `Segmentation Mask is Empty` | ITK-SNAP was closed without saving |
| `One or both masks have very few voxels` | Fewer than 50 voxels in CP or SN; inspect the tracing before continuing |
| `SN mask missing` or `Group Average NM Template missing` | Run the earlier stage first |
| No poster figures after group QC | Poster figures need `--manual_dir` and manual masks for the subjects |
| Out of memory | Lower `--n_procs` |
