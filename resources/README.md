# Resources

Atlas priors and template references read by the pipeline. Paths below are
relative to `resources/tpl-MNI152NLin2009cAsym/` and are resolved from
`--project_root`, so `--project_root` must point at a checkout of this repository.

| File | Used by | Purpose |
|---|---|---|
| `HarvardOxford/Brainstem.nii.gz` | Stage 1 (required) | Weighting mask that focuses the T1→MNI SyN registration on the brainstem |
| `HarvardOxford/Midbrain.nii.gz` | template builders | Midbrain prior used to crop subjects before template construction |
| `HarvardOxford/Brainstem_Midbrain.nii.gz` | — | Union of the two above; produced alongside `Midbrain.nii.gz`, currently unused |
| `CIT168/SNc.nii.gz` | Stage 2 | Binary atlas SNc, warped to native space as the atlas comparison mask and as the anchoring source |
| `CIT168_prob/CIT168toMNI152-2009c_prob_0006.nii.gz` | Stage 2 | Default probabilistic SNc atlas (`ProbCIT` QC method) |
| `NMS_prob/NMS_Snc_prob.nii.gz` | Stage 2 | Neuromelanin-sensitive probabilistic SNc atlas (`NMS` QC method) |
| `MNI152NLin2009cAsym_res-1_desc-brain_T1w.nii.gz` | reference | 1 mm MNI brain, mirrored from TemplateFlow for convenience |

The remaining `CIT168_prob/` volumes are the full CIT168 probabilistic atlas as
distributed upstream; only volume 0006 (SNc) is read by the pipeline.

## Files that are generated, not stored

`MNI152NLin2009cAsym_res-05_desc-brain_T1w.nii.gz` (30 MB) is required by
`--hires` and by the V4 drawing template, but is **not** committed because it is
derived. Generate it into this directory with:

```bash
python workflow/tools/resample_mni_reference.py --out-dir resources/tpl-MNI152NLin2009cAsym
```

`HarvardOxford/Midbrain.nii.gz` is likewise reproducible from BrainstemNavigator
with `workflow/tools/make_midbrain_prior.py`; it is committed because the
BrainstemNavigator distribution is not freely redistributable.

## Third-party licensing

The NMS probabilistic SNc atlas is redistributed from the `NMS-SNc-atlas` release
(Cassidy et al.); see `NMS_prob/LICENSE`. CIT168 and Harvard-Oxford retain their
upstream licences.
