# Superseded manual traces (retired 2026-09-10)

MNI-space tracings of the substantia nigra (`MNI_SNc_Manual`), the cerebral
peduncle (`MNI_CP_Manual`) and the combined label volume
(`MNI_Manual_Masks_combined`: 1 = CP, 2 = SN), on the 1 mm MNI152NLin2009cAsym
grid (193 × 229 × 193).

These files are dated 2026-07-09. They were the tracings the pipeline read from
`resources/tpl-MNI152NLin2009cAsym/` until 2026-09-10, when tracing moved onto
the FireANTs V4 midbrain template and the tracing's home became
`<output_root>/manual_trace_v4/`. Nothing in the current workflow reads them.

| File | Non-zero voxels |
|---|---:|
| `MNI_SNc_Manual.nii.gz` | 869 |
| `MNI_CP_Manual.nii.gz` | 510 |
| `MNI_Manual_Masks_combined.nii.gz` | 1379 |

Byte-identical to `resources/traces_superseded_2026-09-10/` in the Nexus working
directory.

## Correction, 2026-09-14

The September 11 reorganization put the repository's older March tracings in
this directory by mistake. Those now live in
[`../traces_github_2026-03-29/`](../traces_github_2026-03-29/).
