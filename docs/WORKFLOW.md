# Workflow

What each processing step does and why. For how to run them, see [USAGE.md](USAGE.md).

---

## The problem

The substantia nigra is small, has no reliable boundary on a T1-weighted image, and varies in
position and shape between people. An atlas mask warped into a subject's space inherits every
error of that warp, and the errors are large relative to the structure.

NM-MRI gives a subject-specific intensity cue, but thresholding it directly is unreliable too:
absolute NM intensity varies with coil loading, sequence and subject, so a fixed intensity
threshold means something different on every scan.

NeuroMelAnchor separates the two sources of information. The **shape prior** comes from a
group-level template on which SN and CP are traced once. The **intensity evidence** comes from
each subject's own NM signal, expressed relative to the cerebral peduncle, which is white matter
next to the SN with no neuromelanin. Dividing by the CP reference removes scanner and subject
scaling.

---

## Stage 1: forward normalization

Goal: bring every subject into a common space and build the group template the tracing is drawn
on.

1. **Gunzip.** SPM cannot read compressed NIfTI.
2. **Realign (SPM).** Motion-correct the NM acquisitions to their mean. This also yields the
   realignment parameters used for motion QC.
3. **N4 bias field correction (ANTs)**, per realigned volume. NM images have strong receive-coil
   shading, which would otherwise dominate any intensity comparison between SN and CP.
4. **Average.** The mean of the bias-corrected volumes becomes the subject's NM reference image
   (`--nm_image_strategy first` keeps only the first acquisition instead).
5. **Brain extraction (ANTs)** on the T1.
6. **T1 to MNI (ANTs: rigid, affine and SyN).** The registration is weighted by
   `HarvardOxford/Brainstem.nii.gz`, so it is driven by the brainstem rather than by cortex,
   which a whole-brain registration would favour at the midbrain's expense.
7. **NM to T1 (ANTs, rigid).** Rigid only: NM and T1 are the same brain in the same session,
   and anything more flexible could absorb real anatomy.
8. **Compose the transforms**, so NM reaches MNI in a single resampling.
9. **Smooth (SPM, optional).** Off by default (`--smooth 0`).
10. **QC:** registration overlays, the run alignment overlay and motion ranges.
11. **Group-average NM template:** the mean of the normalized NM images.

With `--hires`, the saved transforms are reapplied onto a 0.5 mm grid to give
`NM_MNI_res-05.nii`. At 1 mm the SN is only a few voxels across, so the finer grid matters for
tracing.

---

## Stage segment: the drawing template and the tracing

Goal: one careful tracing on the sharpest image available.

The plain group average is blurred by residual misregistration, because it averages subjects
that are only approximately aligned. By default the pipeline therefore builds the **V4 midbrain
template** with FireANTs (`workflow/scripts/template_builder.py`) from each subject's 0.5 mm NM
image in MNI space:

1. The midbrain prior (built from BrainstemNavigator nuclei) is trimmed to a fixed midbrain box
   and split into left and right halves.
2. Each subject's left and right SN centroids are found from its brightest voxels, and every
   subject is translated so its centroids match the cohort average.
3. Subjects are masked and cropped to the midbrain, and the initial template is a
   correlation-weighted mean.
4. The template is refined over 12 iterations. Each subject is registered to it (rigid, then
   SyN), subjects whose fit is a low outlier are left out of the average, and the shape update
   is damped by a gradient step of 0.25, as `antsMultivariateTemplateConstruction2.sh` does,
   rather than applying the full averaged correction each round.

V4 is then shifted rigidly, by about 1.3 mm, onto the group average. The registration is
restricted to the region V4 covers, because the zeros around the crop would otherwise pull it
off. This makes a tracing drawn on V4 land correctly when Stage 2 registers subjects to the
group average; the shift introduces no stretching and keeps V4's 0.5 mm grid.

ITK-SNAP opens with V4 as the main image and the 0.5 mm MNI T1 as an overlay. **CP is label 1
and SN is label 2.** See [MANUAL_SEGMENTATION.md](MANUAL_SEGMENTATION.md).

`--drawing_template naive` skips all of this and uses the plain group mean, which needs no GPU
or container but gives a blurrier target.

This stage runs **once per dataset**. The tracing is kept across reruns, and
`--rebuild_template` refuses to run while a tracing exists, since a rebuilt template would no
longer match it.

---

## Stage 2: inverse normalization and the NM mask

Goal: turn the group tracing and each subject's NM signal into a subject-specific mask, and
measure how it differs from the alternatives.

1. **Subject to group template (ANTs: rigid, affine and SyN).** A direct NM-to-NM registration
   rather than a reuse of Stage 1: both images have the same contrast, which is a much
   better-posed problem than going through the T1 and MNI.
2. **Warp SN, CP and the CIT168 SNc to native NM space** with the inverse transform and
   nearest-neighbour interpolation, so the subject's own data is never resampled.
3. **Search region.** The warped SN tracing, grown by `--search_dilation_mm`. The distance is
   measured in millimetres, so anisotropic voxels are handled correctly. The NM mask can only
   keep voxels inside this region, so a tight tracing caps how much of the SN it can find;
   growing the region lets the subject's contrast decide the boundary instead. The default of 0
   uses the tracing as drawn; the Brainhack Western 2026 poster used 0.5 mm.
4. **Contrast map.** The CP mask is eroded by one voxel, or left uneroded if fewer than 20
   voxels would remain. The CP reference is the mode of the CP intensities, estimated with a
   kernel density, or their median if that fails. The map is (NM - CP reference) / CP reference
   inside the search region, smoothed with a 1 mm FWHM Gaussian. File names call it "CNR", but
   it is a contrast ratio with no noise term.
5. **NM mask.** Search-region voxels whose smoothed ratio is at or above the larger of the floor
   (0.05) and the region's 55th percentile, and at or below its 99th percentile. Holes are
   filled, components under 15 voxels are dropped, and the largest two (left and right SN) are
   kept. The result is closed with a 3 x 3 x 3 element and intersected with the search region,
   so the closing cannot grow the mask outside it.
6. **Core-rim variant.** The search region's core, more than 0.5 mm from its edge, and its rim
   are thresholded separately, at 0.05 and 0.10. The core is spatially more trustworthy, so it
   gets the more liberal threshold, while boundary false positives are what inflate HD95, so the
   rim gets the stricter one. Components under 25 voxels are dropped and the largest two kept.
7. **Anchored atlas.** The NM mask is grown by 6 mm to form an ROI, and the warped CIT168 SNc is
   rigidly registered onto the NM mask within it. This asks a different question from the other
   masks: keep the atlas's shape, but let the subject's NM signal decide where it sits.
8. **Probabilistic atlases.** The CIT168 and NMS probabilistic SNc maps are warped to native
   space with linear interpolation and thresholded, at 0.25 by default.
9. **Scoring.** The masks are compared in twelve pairs; see [EVALUATION.md](EVALUATION.md).

### What the thresholding does in practice

The authors' note in the pipeline is explicit about the limits of step 5. The floor is too low
ever to take effect, so the percentile alone sets the threshold, and roughly the top 45% of the
search region survives it in every subject, however strong or weak that subject's contrast.
Subjects with weak NM contrast therefore still receive a mask of similar relative size and are
likely to be segmented less accurately. This is flagged for review in [STATUS.md](STATUS.md).

---

## Group QC

Builds each subject's dashboard and output manifest, then aggregates the metrics into raincloud
plots of DICE, dot plots of HD95, deviation box plots, a method comparison, the seed-atlas
coupling and a boundary-versus-core contrast plot, plus two CSV summaries for group statistics.

When manual masks are available it also writes the poster figures and the NM-blind control. The
control asks whether NM contrast adds anything beyond registration and position, by comparing
the NM mask with the propagated tracing and with a mask of the same share of the search region
chosen by depth rather than brightness.

Group QC reads only existing Stage 2 outputs, so `--stage group_qc` is the cheap way to
regenerate summaries. See [EVALUATION.md](EVALUATION.md).
