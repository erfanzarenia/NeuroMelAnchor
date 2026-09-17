# Retired manual traces

MNI-space tracings of the substantia nigra (`MNI_SNc_Manual`), the cerebral peduncle
(`MNI_CP_Manual`) and the combined label volume (`MNI_Manual_Masks_combined`, where 1 is CP and
2 is SN), on the 1 mm MNI152NLin2009cAsym grid. Each directory is named for the date its tracing
was drawn.

| Directory | Used until | SN voxels | CP voxels |
|---|---|---:|---:|
| `2026-03-29/` | July 2026 | 1451 | 814 |
| `2026-07-09/` | 2026-09-10 | 869 | 510 |

Tracing now happens per dataset on the FireANTs V4 midbrain template, and each dataset's tracing
lives in `<output_root>/manual_trace_v4/`. Nothing in the current workflow reads these files;
they are kept so earlier results stay reproducible. See
[../../docs/MANUAL_SEGMENTATION.md](../../docs/MANUAL_SEGMENTATION.md).
