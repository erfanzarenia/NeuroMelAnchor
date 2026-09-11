#!/usr/bin/env python3
"""
Rename scanner-exported NIfTI files into BIDS-style names, in place.

Files are grouped by their parent ``sub-*`` directory, ordered, and numbered
``run-01``, ``run-02``, ... within each subject. The output name is a format
string so one tool covers every contrast that needs renaming.

Runs as a dry run by default; pass --apply to actually rename.

Available format fields: ``{sub}`` (subject label without the ``sub-`` prefix)
and ``{run}`` (1-based index within the subject).

Examples, as used for the BioGen 7T export:

    # MP2RAGE UNI images exported as *LH.nii
    python workflow/tools/bids_rename.py --root NM_BioGen/7T \
        --pattern '*LH.nii' \
        --template 'sub-{sub}_acq-MP2RAGEUNI_run-{run:02d}_T1w.nii' --apply

    # MP2RAGE UNI images exported with the series name
    python workflow/tools/bids_rename.py --root NM_BioGen/7T \
        --pattern '*MP2RAGE_UNI_Images*.nii' \
        --template 'sub-{sub}_acq-MP2RAGEUNI_run-{run:02d}_T1w.nii' --apply

    # Inversion images
    python workflow/tools/bids_rename.py --root NM_BioGen/7T \
        --pattern '*MP2RAGE_INV*.nii' \
        --template 'sub-{sub}_acq-INV{run}.nii' --apply

    # Multi-series NM acquisitions, ordered by the trailing _s<N> series number
    python workflow/tools/bids_rename.py --root NM_BioGen/7T \
        --pattern '*.nii' --sort-by-series \
        --template 'sub-{sub}_acq-CombEchoNM_run-{run:02d}_GRE.nii' --apply
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

SERIES_PATTERN = re.compile(r'_s(\d+)\.nii$')


def group_by_subject(root, pattern):
    """Map subject label -> list of matching files, keyed by the parent sub-* directory."""
    subjects = defaultdict(list)
    for path in sorted(root.rglob(pattern)):
        parent = path.parent.name
        if parent.startswith('sub'):
            subjects[parent.replace('sub-', '')].append(path)
    return subjects


def series_number(path):
    """Trailing ``_s<N>`` series number, or None when the name has no series suffix."""
    match = SERIES_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def plan_renames(subjects, template, sort_by_series):
    """
    Return [(source, destination)] for every file, numbered per subject.

    Sorting is by series number when --sort-by-series is set (scanner export
    order), otherwise by filename.
    """
    planned = []
    for sub_id, files in sorted(subjects.items()):
        if sort_by_series:
            numbered = [(series_number(p), p) for p in files]
            missing = [p.name for number, p in numbered if number is None]
            if missing:
                raise ValueError(
                    f"sub-{sub_id}: --sort-by-series needs a trailing _s<N> on every "
                    f"match, missing on: {missing}")
            files = [p for _, p in sorted(numbered)]
        else:
            files = sorted(files)

        for run_index, path in enumerate(files, start=1):
            planned.append((path, path.with_name(template.format(sub=sub_id, run=run_index))))
    return planned


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory searched recursively for sub-*/ folders.")
    parser.add_argument("--pattern", required=True,
                        help="Glob matched against filenames, e.g. '*LH.nii'.")
    parser.add_argument("--template", required=True,
                        help="Output name format string using {sub} and {run}.")
    parser.add_argument("--sort-by-series", action="store_true",
                        help="Order files by the trailing _s<N> series number instead of by name.")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the renames. Without it, only the plan is printed.")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"--root is not a directory: {args.root}")

    subjects = group_by_subject(args.root, args.pattern)
    if not subjects:
        raise SystemExit(f"No files under {args.root} matching {args.pattern!r} "
                         f"inside a sub-* directory.")

    planned = plan_renames(subjects, args.template, args.sort_by_series)

    collisions = [dst for _, dst in planned if dst.exists() and dst not in {s for s, _ in planned}]
    if collisions:
        raise SystemExit(f"Refusing to run: these targets already exist: "
                         f"{[str(c) for c in collisions[:5]]}")

    for source, destination in planned:
        print(f"{'renaming' if args.apply else 'would rename'}: "
              f"{source.name} -> {destination.name}")
        if args.apply:
            source.rename(destination)

    print(f"\n{len(planned)} file(s) across {len(subjects)} subject(s)"
          f"{'' if args.apply else '; re-run with --apply to perform the renames'}")


if __name__ == "__main__":
    main()
