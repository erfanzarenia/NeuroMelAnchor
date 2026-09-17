# Standalone tools

Single-purpose utilities that sit beside the pipeline. None of them is invoked by
`neuromelanchor_pipeline.py`; they prepare inputs, build resources, or derive
extra maps. Each takes its paths on the command line; run any of them with
`--help`.

| Tool | Purpose |
|---|---|
| `resample_mni_reference.py` | Writes the 1 mm and 0.5 mm MNI brain references into `resources/`. Run this once after cloning if you plan to use `--hires`. |
| `make_brainstem_mask.py` | Rebuilds `HarvardOxford/Brainstem.nii.gz`, the weighting mask for the Stage 1 T1→MNI registration. |
| `make_midbrain_prior.py` | Rebuilds `HarvardOxford/Midbrain.nii.gz` from BrainstemNavigator; this is the crop prior used by the template builders. |
| `bids_rename.py` | Renames scanner exports into BIDS-style names. Dry run by default. |
| `mp2rage_uniden.py` | Third-party MP2RAGE UNI denoising (see file header for attribution). Kept verbatim. |
| `mp2rage_uniden_batch.sh` | Runs the above across a BIDS directory. |
| `fit_r2star.py` | Fits R2*/T2* maps from multi-echo GRE. Independent of the pipeline; needs FSL and MRtrix3. |
