# Manual Segmentation

The one interactive step in the pipeline: tracing SN and CP on the group
template. Done **once per dataset**, not per subject.

---

## Why two regions

**Substantia nigra (label 2)** is the target — but it also defines the search
region within which each subject's CNR is evaluated. Trace it generously enough
that no subject's true SN falls outside it; the CNR threshold does the
tightening. A tracing that clips the SN cannot be recovered downstream.

**Cerebral peduncle (label 1)** is the background reference. It is white matter
immediately adjacent to SN with no neuromelanin, so it provides the intensity
baseline the CNR is measured against. Every subject-level threshold depends on
it, so place it in clean, homogeneous peduncle — away from the SN border, away
from partial-volume voxels at the peduncle edge, and away from any visible
artefact.

---

## Running it

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage segment
```

With the default `--drawing_template v4` the pipeline builds the V4 midbrain
template (~35 minutes on a GPU), aligns it to the group average, and opens
ITK-SNAP with:

- **Main image** — the V4 template at 0.5 mm, already cropped to the midbrain, so
  no navigation is needed
- **Overlay** — the 0.5 mm MNI T1, for anatomical orientation
- **Segmentation** — an empty mask on the V4 grid, or your existing tracing
- **Label file** — `itksnap_labels.txt`, with label 1 and 2 named for you

With `--drawing_template naive` the plain group mean opens through an ITK-SNAP
workspace instead. That view is not pre-cropped, so jump to the midbrain by
typing `98 109 61` into the Cursor Position box, click *zoom to fit*, and zoom in.

---

## Protocol

1. Work in the axial view; use coronal and sagittal to confirm.
2. Move through slices until the SN is visible as the bright band
   posterolateral to the cerebral peduncle.
3. Trace **SN as label 2**, bilaterally, following the bright band. Prefer
   slightly generous over slightly tight.
4. Trace **CP as label 1**, bilaterally, in the body of the peduncle — not at its
   border with SN.
5. Check all three planes for stray voxels.
6. **Ctrl+S** to save, then close ITK-SNAP. The pipeline resumes automatically.

---

## What the pipeline checks

On resuming it splits the combined label volume into the SN and CP masks and
refuses to continue if:

- the saved mask is empty — ITK-SNAP was closed without saving
- either region has **fewer than 50 voxels** — almost always a partial tracing

Both errors are recoverable: re-run `--stage segment` and the existing tracing is
reopened rather than discarded.

---

## Reruns and rebuilds

The tracing is written to `<output_root>/manual_trace_v4/` (or `manual_trace/`
for `naive`) and reused by every later run. Two guards protect it:

- `--rebuild_template` refuses to run while a tracing exists, because a rebuilt
  template would no longer match it. Move the tracing aside first.
- If the saved mask is not on the current V4 grid, the pipeline stops rather than
  silently resampling a tracing onto a template it was not drawn on.

---

## Per-subject manual masks

Distinct from this step. Optional per-subject SN tracings in **native NM space**,
passed with `--manual_dir`, are used only for evaluation — they never feed back
into segmentation. See [DATA_SPECIFICATION.md](DATA_SPECIFICATION.md).
