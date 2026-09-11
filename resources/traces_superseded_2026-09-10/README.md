# Superseded manual traces

These three MNI-space manual tracings (`MNI_SNc_Manual`, `MNI_CP_Manual` and the
combined label volume) were the reference segmentations used by the pipeline up to
and including the May 2026 revision, when they lived in
`resources/tpl-MNI152NLin2009cAsym/`.

They were retired on 2026-09-10. The current pipeline draws SN and CP on the
FireANTs V4 midbrain template instead, and reads and writes the tracing under
`<output_root>/manual_trace_v4/` rather than from `resources/`.

They are kept here for provenance and for reproducing older results. Nothing in
the current workflow reads them.
