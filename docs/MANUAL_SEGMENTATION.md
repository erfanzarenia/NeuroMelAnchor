# Manual Segmentation

The one interactive step in the pipeline: tracing SN and CP on the group template. It
is done **once per dataset**, not once per subject.

---

## Why two regions

**Substantia nigra (label 2)** is the target, and it also defines the search region
inside which each subject's NM contrast is evaluated. Trace it generously enough that
no subject's true SN falls outside it: the contrast threshold does the tightening, and
a tracing that clips the SN cannot be recovered downstream. `--search_dilation_mm` can
grow the propagated tracing afterwards, but it grows it evenly in every direction, so
it is no substitute for a tracing that follows the anatomy.

**Cerebral peduncle (label 1)** is the background reference. It is white matter
immediately next to the SN with no neuromelanin, so it provides the intensity baseline
that the contrast is measured against, and every subject-level threshold depends on
it. Place it in clean, homogeneous peduncle, away from the SN border, away from
partial-volume voxels at the peduncle edge, and away from any visible artefact.

---

## Running it

```bash
python workflow/scripts/neuromelanchor_pipeline.py ... --stage segment
```

With the default `--drawing_template v4` the pipeline builds the V4 midbrain template
(about 35 minutes on a GPU), aligns it to the group average, and opens ITK-SNAP with:

- **Main image:** the V4 template at 0.5 mm, already cropped to the midbrain, so no
  navigation is needed
- **Overlay:** the 0.5 mm MNI T1, for anatomical orientation
- **Segmentation:** an empty mask on the V4 grid, or your existing tracing
- **Label file:** `itksnap_labels.txt`, with labels 1 and 2 already named

With `--drawing_template naive` the plain group mean opens through an ITK-SNAP
workspace instead. That view is not cropped, so go to the midbrain by typing
`98 109 61` into the Cursor Position box, click *zoom to fit*, and zoom in.

---

## Protocol

1. Work in the axial view and confirm in the coronal and sagittal views.
2. Move through the slices until the SN is visible as the bright band posterolateral
   to the cerebral peduncle.
3. Trace **SN as label 2** on both sides, following the bright band. Err towards
   slightly generous rather than slightly tight.
4. Trace **CP as label 1** on both sides, in the body of the peduncle rather than at
   its border with the SN.
5. Check all three planes for stray voxels.
6. Press **Ctrl+S** to save, then close ITK-SNAP. The pipeline resumes by itself.

---

## What the pipeline checks

When it resumes, the pipeline splits the combined label volume into SN and CP masks
and stops if:

- the saved mask is empty, which means ITK-SNAP was closed without saving
- either region has **fewer than 50 voxels**, which almost always means an unfinished
  tracing

Both are recoverable: rerun `--stage segment` and the existing tracing is reopened
rather than discarded.

---

## Reruns and rebuilds

The tracing is written to `<output_root>/manual_trace_v4/` (or `manual_trace/` for
`naive`) and reused by every later run. Two guards protect it:

- `--rebuild_template` refuses to run while a tracing exists, because a rebuilt
  template would no longer match it. Move the tracing aside first.
- If the saved mask is not on the current V4 grid, the pipeline stops rather than
  silently resampling a tracing onto a template it was not drawn on.

Earlier MNI-space tracings from before the V4 template are kept, unused, under
`resources/`; see [../resources/README.md](../resources/README.md).

---

## Per-subject manual masks

These are separate from this step. Optional per-subject SN tracings in **native NM
space**, passed with `--manual_dir`, are used only for evaluation and never feed back
into segmentation. See [DATA_SPECIFICATION.md](DATA_SPECIFICATION.md).
