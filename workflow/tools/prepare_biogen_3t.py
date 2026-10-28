#!/usr/bin/env python3
"""
================================================================================
Prepare BioGen 3T NM data for NeuroMelAnchor
================================================================================
NM_BioGen/3T holds raw dicm2nii output (uncompressed .nii, four naming schemes,
folders named like 02F_3T). NeuroMelAnchor expects the BIDS-style layout used for
NM_BioGen/7T. This script writes that layout so the pipeline runs unchanged with
--dataset biogen --nm_input echo.

    python workflow/tools/prepare_biogen_3t.py --source /path/to/Data_Archive/3T \
        --out /path/to/NM_BioGen/3T_design --dry_run

For the BioGen 3T archive, 78F_3T/ANAT holds a copy of 78Fb's MPRAGE (2020-11-11,
identical md5) while its NM is from 2021-02-19; the MPRAGE from its own session is in
the Dresden copy, so that run also needs
    --t1_override 78F=/path/to/3T_Dresden/78F_3T/MPRAGE_2021_02_19_NM_MRI_3T_4D4D_78F.nii
================================================================================
"""
import re
import csv
import gzip
import shutil
import argparse
from pathlib import Path

import nibabel as nib

# =================================================================
# Config
# =================================================================

session_to_subject = {
    "02F": "sub-002", "02Fb": "sub-022", "48M": "sub-048", "54F": "sub-054", "56M": "sub-056",
    "57F": "sub-057", "59M": "sub-059", "63F": "sub-063", "64M": "sub-064", "67M": "sub-067",
    "68F": "sub-682", "68M": "sub-068", "69F": "sub-069", "71M": "sub-071", "72M": "sub-072",
    "74F": "sub-074", "76F": "sub-076", "78F": "sub-078", "78Fb": "sub-782", "80F": "sub-802",
    "80M": "sub-080", "81M": "sub-081", "84F": "sub-084", "85M": "sub-085", "92M": "sub-092",
    "94F": "sub-094", "95F": "sub-095", "97F": "sub-097",
}


# =================================================================
# Helpers
# =================================================================

def header_field(path, key):
    """Read one field (TE, time, ...) from the dicm2nii description string in the NIfTI header."""
    description = nib.load(str(path)).header['descrip'].tobytes().decode(errors='replace')
    match = re.search(rf"{key}=([^;]+);", description)
    return match.group(1) if match else None


def find_nm_runs(session_dir):
    """Group the distortion-corrected magnitude NM files into runs, keyed by acquisition time."""
    runs = {}
    for nm_file in sorted(list(session_dir.glob("NM/NM_GRE*.nii")) + list(session_dir.glob("OTHER/NM_GRE*.nii"))):
        name = nm_file.name
        if "phase" in name or "_ND" in name:
            continue
        echo = int(re.search(r"_0000(\d)\.nii$", name).group(1))
        acquisition_time = header_field(nm_file, "time")
        runs.setdefault(acquisition_time, {})[echo] = nm_file
    return runs


def find_t1(session_dir, session, t1_overrides):
    """Return the session's distortion-corrected MPRAGE (or its override)."""
    if session in t1_overrides:
        return t1_overrides[session], "T1 from override"
    candidates = [f for f in session_dir.glob("ANAT/MPRAGE*.nii") if "_ND" not in f.name]
    if len(candidates) != 1:
        return None, f"expected 1 MPRAGE, found {len(candidates)}"
    return candidates[0], ""


def gzip_copy(source, destination):
    """Copy a .nii to .nii.gz byte for byte."""
    temporary = destination.with_name(destination.name + ".part")
    with open(source, 'rb') as f_in, gzip.open(temporary, 'wb', compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    temporary.rename(destination)

# =================================================================
# Main
# =================================================================

parser = argparse.ArgumentParser(description="BIDS-style copy of BioGen 3T NM data for NeuroMelAnchor")
parser.add_argument("--source", required=True, type=Path, help="NM_BioGen/3T folder")
parser.add_argument("--out", required=True, type=Path, help="New design folder to write (passed to the pipeline as --design_dir)")
parser.add_argument("--dry_run", action="store_true", help="Print what would be copied without writing anything")
parser.add_argument("--t1_override", action="append", default=[], metavar="SESSION=PATH",
                    help="Use this MPRAGE for a session instead of the one in its ANAT folder (repeatable)")
args = parser.parse_args()
t1_overrides = {}
for item in args.t1_override:
    session, _, path = item.partition("=")
    if not path:
        parser.error(f"--t1_override expects SESSION=PATH, got {item!r}")
    t1_overrides[session] = Path(path)

manifest = []
skipped = []

for session_dir in sorted(args.source.glob("*_3T")):
    session = session_dir.name[:-len("_3T")]
    subject = session_to_subject.get(session)
    row = {"subject": subject, "session": session_dir.name, "nm_acq_time": "", "n_dc_runs": 0,
           "nm_echo1": "", "t1": "", "status": "", "notes": ""}

    if subject is None:
        row["status"] = "skipped"; row["notes"] = "no ID in the name-matching sheet"
        manifest.append(row); skipped.append(row); continue

    # NM: earliest distortion-corrected run, all 3 echoes
    runs = find_nm_runs(session_dir)
    row["n_dc_runs"] = len(runs)
    if not runs:
        row["status"] = "skipped"; row["notes"] = "no distortion-corrected NM run (only _ND)"
        manifest.append(row); skipped.append(row); continue
    first_time = min(runs)
    echoes = runs[first_time]
    row["nm_acq_time"] = first_time
    if sorted(echoes) != [1, 2, 3]:
        row["status"] = "skipped"; row["notes"] = f"first run has echoes {sorted(echoes)}"
        manifest.append(row); skipped.append(row); continue
    row["nm_echo1"] = str(echoes[1])

    # T1: same scan date as the NM run
    t1_file, t1_note = find_t1(session_dir, session, t1_overrides)
    if t1_file is None or not t1_file.exists():
        row["status"] = "skipped"; row["notes"] = t1_note or f"T1 not found: {t1_file}"
        manifest.append(row); skipped.append(row); continue
    row["t1"] = str(t1_file)
    if header_field(t1_file, "time")[:8] != first_time[:8]:
        row["status"] = "skipped"; row["notes"] = "T1 and NM come from different scan dates"
        manifest.append(row); skipped.append(row); continue
    row["notes"] = t1_note

    # Copy (never overwrite)
    anat_dir = args.out / subject / "anat"
    targets = {args.out / subject / "anat" / f"{subject}_run-1_echo-{echo}_part-mag_MEGRE.nii.gz": echoes[echo]
               for echo in (1, 2, 3)}
    targets[anat_dir / f"{subject}_acq-T1w.nii.gz"] = t1_file

    if any(target.exists() for target in targets):
        row["status"] = "exists"; row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "output already present, left untouched"
        manifest.append(row); continue

    row["status"] = "would copy" if args.dry_run else "copied"
    if not args.dry_run:
        anat_dir.mkdir(parents=True, exist_ok=False)
        for target, source in targets.items():
            gzip_copy(source, target)
    manifest.append(row)

# =================================================================
# Report
# =================================================================

for row in manifest:
    print(f"{row['session']:9s} -> {str(row['subject']):8s} {row['status']:10s} "
          f"runs={row['n_dc_runs']} nm_time={row['nm_acq_time'][8:14]} {row['notes']}")

n_ready = sum(row["status"] in ("copied", "would copy", "exists") for row in manifest)
print(f"\n{n_ready} subjects ready, {len(skipped)} skipped:")
for row in skipped:
    print(f"  {row['session']}: {row['notes']}")

if not args.dry_run:
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "prep_manifest.tsv"
    if manifest_path.exists():
        manifest_path = args.out / "prep_manifest_new.tsv"
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()), delimiter='\t')
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\nManifest: {manifest_path}")
