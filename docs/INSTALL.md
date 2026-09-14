# Installation

The pipeline drives external neuroimaging programs through Nipype. The Python packages
are the easy part; ANTs, SPM12 and ITK-SNAP must be installed separately and be on
`PATH`.

---

## Requirements

| Component | Needed for | Notes |
|---|---|---|
| Python 3.9 or newer | everything | Developed on 3.12 |
| **ANTs** | stages 1, segment and 2 | Registration, brain extraction and N4 bias correction |
| **MATLAB and SPM12** | stage 1 | Realignment and optional smoothing |
| **ITK-SNAP** | stage segment | Manual tracing |
| FireANTs and a CUDA GPU | stage segment with `--drawing_template v4` | Normally supplied by a container; see step 5 |
| FSL and MRtrix3 | `workflow/tools/fit_r2star.py` only | Not needed by the pipeline |

Hardware: at least 16 GB of RAM (32 GB recommended), 8 or more cores, and plenty of
disk, since the Nipype work directory is far larger than the outputs. A CUDA GPU is
needed only for the V4 drawing template, and `--drawing_template naive` avoids it.

The figure scripts in `workflow/figures/` and the checks in `tests/` need only the
Python packages.

---

## 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or with conda:

```bash
conda create -n neuromelanchor python=3.12 && conda activate neuromelanchor
pip install -r requirements.txt
```

`requirements.txt` is pinned from the environment used for the September 2026 runs.
It deliberately leaves out `fireants` and `torch`; see step 5.

## 2. ANTs

Install from https://github.com/ANTsX/ANTs, then:

```bash
export ANTSPATH=/path/to/ants/bin
export PATH=$ANTSPATH:$PATH
antsRegistration --version
```

The pipeline checks for `antsRegistration` at startup and refuses to run without it,
rather than failing hours in.

## 3. MATLAB and SPM12

Install SPM12 and add it to the MATLAB path. The pipeline checks for `matlab` on
`PATH` before Stage 1. If Nipype does not find SPM, set:

```bash
export MATLAB_CMD="matlab -nodesktop -nosplash"
```

## 4. ITK-SNAP

Install from https://www.itksnap.org and make sure `itksnap` is on `PATH`; the
pipeline launches it directly for `--stage segment`.

## 5. FireANTs (optional, for the V4 drawing template)

The template builders need `fireants` and a CUDA-capable `torch`. Rather than
installing these into the pipeline environment, the pipeline looks for a container and
runs the builder inside it. At startup it tries, in order:

1. `fireants.sif` in `--project_root` with `singularity`
2. `fireants.sif` in `--project_root` with `apptainer`
3. a local `fireants` Docker image

If none is found it warns and falls back to the current Python interpreter. To run the
builders without a container:

```bash
pip install fireants==1.5.0 torch==2.12.0
```

To skip this machinery entirely, use `--drawing_template naive`.

## 6. TemplateFlow

MNI references are fetched through TemplateFlow on first use. If the default cache is
not writable, set:

```bash
export TEMPLATEFLOW_HOME=/path/to/templateflow
```

## 7. Generate the 0.5 mm MNI reference

Needed by `--hires`, by the V4 drawing template, and by the T1 panel of
`workflow/figures/poster_workflow_panels.py`. It is derived, so it is not committed:

```bash
python workflow/tools/resample_mni_reference.py \
    --out-dir resources/tpl-MNI152NLin2009cAsym
```

---

## Verify the installation

```bash
pip install pytest
cd tests && python -m pytest
```

These checks need none of the external software. They confirm that the repository is
intact, the scripts parse, the resources are in place and the numeric helpers behave.
Passing them does **not** show that ANTs, SPM and ITK-SNAP are installed correctly;
for that, run one subject:

```bash
python workflow/scripts/neuromelanchor_pipeline.py \
  --project_root . --design_dir /path/to/bids --output_root ./outputs \
  --work_dir /path/to/work --stage 1 --test
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ANTs not found on path` | Export `ANTSPATH` and add it to `PATH` |
| `Matlab not found on path` | Put MATLAB on `PATH` and check that SPM12 is on the MATLAB path |
| `ITK-SNAP not found on PATH` | Install ITK-SNAP, or run a stage other than `segment` |
| TemplateFlow download failures | Set `TEMPLATEFLOW_HOME` to a writable path |
| `0.5mm MNI reference not found` | Run step 7 |
| `No module named fireants` | Provide a container, install fireants and torch, or use `--drawing_template naive` |
| Out of memory during registration | Lower `--n_procs`; ITK threads are capped at `min(n_procs, 8)` |

Next: [DATA_SPECIFICATION.md](DATA_SPECIFICATION.md), then [USAGE.md](USAGE.md).
