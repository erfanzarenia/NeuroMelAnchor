# Workflow

What each processing step does and why. For how to run them, see
[USAGE.md](USAGE.md).

---

## The problem

The substantia nigra is small, has no reliable boundary on a T1-weighted image,
and varies in position and shape between people. An atlas mask warped into a
subject's space inherits every error of that warp, and the errors are large
relative to the structure.

NM-MRI gives a subject-specific intensity cue, but thresholding it directly is
also unreliable: absolute NM intensity varies with coil loading, sequence and
subject, so a threshold tuned on one scan means something different on the next.

NeuroMelAnchor separates the two sources of information. The **shape prior**
comes from a group-level template on which SN and CP are traced once. The
**intensity evidence** comes from each subject's own NM signal, expressed as
contrast against the cerebral peduncle — adjacent white matter with no
neuromelanin. Taking the ratio removes the scanner-to-subject scaling, so a CNR
threshold means the same thing everywhere.

---

## Stage 1 — forward normalization

Goal: get every subject into a common space, and build the group template the
tracing will be drawn on.

1. **Gunzip** — SPM cannot read compressed NIfTI.
2. **Realign (SPM)** — motion-correct the NM acquisitions to their mean. Also
   yields the realignment parameters used for motion QC.
3. **N4 bias field correction (ANTs)** — per realigned volume. NM images have
   strong receive-coil shading, which would otherwise dominate any intensity
   comparison between SN and CP.
4. **Average** — mean of the bias-corrected volumes becomes the subject NM
   reference (`--nm_image_strategy first` keeps only the first instead).
5. **Brain extraction (ANTs)** — on the T1.
6. **T1 → MNI (ANTs, rigid + affine + SyN)** — *brainstem-weighted*: the cost is
   weighted by `HarvardOxford/Brainstem.nii.gz` so the warp is driven by the
   midbrain rather than by cortex, which is what a whole-brain registration would
   optimize at the midbrain's expense.
7. **NM → T1 (ANTs, rigid)** — rigid only. NM and T1 are the same brain in the
   same session; anything more would let the registration absorb real anatomy.
8. **Compose transforms** — NM→T1 and T1→MNI are concatenated so NM reaches MNI
   in a single resampling, avoiding a second interpolation.
9. **Smooth (SPM, optional)** — off by default (`--smooth 0`).
10. **QC** — registration overlays, between-run similarity, motion statistics.
11. **Group-average NM template** — mean of the normalized NM images.

With `--hires`, the saved transforms are reapplied to a 0.5 mm grid to give
`NM_MNI_res-05.nii`. At 1 mm the SN is only a few voxels across, so the finer
grid matters for the tracing.

---

## Stage segment — the drawing template and the tracing

Goal: one careful tracing, on the sharpest possible image.

The plain group average is blurred by residual misregistration — it averages
subjects that are only approximately aligned. So by default the pipeline builds a
**V4 midbrain template** with FireANTs: an iterative template construction
restricted to the midbrain, at 0.5 mm, with shape averaging damped by a gradient
step of 0.25 (the same damping `antsMultivariateTemplateConstruction2.sh` uses,
rather than applying the full averaged correction each round). Subjects are
cropped to the midbrain using the BrainstemNavigator-derived prior and aligned by
their bilateral SN centroids before the iterations start.

The template is then rigidly shifted (about 1.3 mm) onto the group average, so a
mask drawn on it lands correctly when Stage 2 registers subjects to that average.
The shift introduces no stretching and keeps V4's 0.5 mm grid.

ITK-SNAP then opens with V4 in the foreground and the 0.5 mm MNI T1 behind it.
**CP is label 1, SN is label 2.** See
[MANUAL_SEGMENTATION.md](MANUAL_SEGMENTATION.md).

`--drawing_template naive` skips all of this and traces on the plain group mean —
no GPU or container needed, at the cost of a blurrier target.

This stage runs **once per dataset**. The tracing is preserved across reruns, and
`--rebuild_template` refuses to proceed while a tracing exists, since a rebuilt
template would no longer match it.

---

## Stage 2 — inverse normalization and CNR seeding

Goal: turn the group tracing plus each subject's NM signal into a subject-specific
mask, then measure how much that differs from the alternatives.

1. **Subject → group template (ANTs, rigid + affine + SyN)** — a direct NM-to-NM
   registration, not reused from Stage 1. Both images are the same contrast,
   which is a far better-posed problem than the NM-to-T1-to-MNI chain.
2. **Warp SN and CP to native NM space** — using the inverse transform, so the
   subject's data is never resampled.
3. **Compute the CNR map** — the CP intensity mode is estimated with a kernel
   density estimate (the mode, not the mean, because the CP distribution is
   skewed by partial-volume voxels at its edges), then
   `CNR = (NM − CP_mode) / CP_mode` within the SN search region. The map is
   lightly smoothed before thresholding.
4. **Extract the NM seed** — voxels above both an absolute CNR floor (5%, so a
   voxel must be genuinely brighter than white matter) and a subject-adaptive
   percentile within the search region (so subjects with globally weaker NM
   contrast are not systematically under-segmented). Components smaller than 15
   voxels are dropped and only the largest two are kept — left and right SN.
5. **Core-rim variant** — the same idea with the eroded core and the boundary rim
   thresholded separately. The core is spatially more trustworthy, so it uses the
   liberal threshold; the rim uses a stricter one, because boundary false
   positives are what inflate HD95.
6. **Anchored atlas** — the CIT168 atlas is rigidly re-registered onto the
   subject's own NM seed within a dilated ROI. This asks a different question
   from the others: keep the atlas *shape*, but let the subject's NM signal say
   where it belongs.
7. **Probabilistic atlases** — CIT168 and NMS probabilistic SNc are warped to
   native space and thresholded, giving two further comparators.
8. **QC** — all pairwise metrics, voxelwise disagreement maps and a per-subject
   dashboard.

---

## Group QC

Aggregates the per-subject metrics into raincloud plots of Dice, dot plots of
HD95, deviation box plots, a method comparison, seed/atlas coupling and a
boundary-versus-core contrast plot, plus flat CSVs for group statistics.

It reads only existing Stage 2 outputs, so `--stage group_qc` is the cheap way to
regenerate summaries after a plotting change.

See [EVALUATION.md](EVALUATION.md) for how to read them.
