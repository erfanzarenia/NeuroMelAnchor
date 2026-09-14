# Earlier manual traces (repository upload, 2026-03-29)

The MNI-space SN and CP tracings uploaded to the repository by Sam Marshall on
2026-03-29 (commit `ba0ca10`). They were tracked at
`resources/tpl-MNI152NLin2009cAsym/` on `main` until the September 2026
reorganization.

The grid and file names match the set retired on 2026-09-10, but this is a
different, larger tracing:

| File | Non-zero voxels | July set | DICE with July set |
|---|---:|---:|---:|
| `MNI_SNc_Manual.nii.gz` | 1451 | 869 | 0.57 |
| `MNI_CP_Manual.nii.gz` | 814 | 510 | 0.26 |
| `MNI_Manual_Masks_combined.nii.gz` | 2265 | 1379 | 0.46 |

There is no record of when the pipeline switched from these to the July tracings;
the July files are dated 2026-07-09. Nothing in the current workflow reads either
set. They are kept so results produced from the repository before July stay
reproducible.

A third set, dated 2026-03-18, is in `Masks/MNI/` in the Nexus working directory
and differs from both. It was never in the repository and has not been migrated.
