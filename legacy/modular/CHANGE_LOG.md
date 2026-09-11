# Minimal-change modularization log

Goal: split the original script into separate files without changing the pipeline logic.

## Files created
- `run_pipeline.py`
- `functions.py`
- `phase1.py`
- `segment.py`
- `phase2.py`

## Changes made
1. **Moved helper functions** from the original script into `functions.py`.
   - Behavior impact: none intended.
   - Type: pure move.

2. **Wrapped Phase 1 block** into `run_phase1(...)` inside `phase1.py`.
   - Behavior impact: none intended.
   - Type: wrapper-only change.

3. **Wrapped segmentation block** into `run_segment(...)` inside `segment.py`.
   - Behavior impact: none intended.
   - Type: wrapper-only change.

4. **Wrapped Phase 2 block** into `run_phase2(...)` inside `phase2.py`.
   - Behavior impact: none intended.
   - Type: wrapper-only change.

5. **Added imports across files** so moved code can still access helper functions and Nipype classes.
   - Behavior impact: none intended.
   - Type: wiring change.

6. **Passed former top-level variables as function arguments** from `run_pipeline.py` into the phase scripts.
   - Behavior impact: none intended.
   - Type: wiring change.

## Deliberately not added
- no dataclass/config system
- no BIDS validator
- no new template loader helper
- no new pre-flight checks
- no new runtime logic
- no parameter changes
- no changed file naming patterns

## Important note
This is not literally zero edits, because Python needs imports, function wrappers, and argument passing for modularization.
But the intent was to preserve original behavior as closely as possible.
