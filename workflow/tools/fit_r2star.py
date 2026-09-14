#!/usr/bin/env python3
"""
Fit R2* and T2* maps from multi-echo GRE magnitude data, per subject and run.

For each run the three magnitude echoes are merged into a 4D volume, a crude
brain mask is derived from a percentile threshold on echo 1, the series is
MP-PCA denoised, and R2* is fitted by weighted log-linear regression across
echoes. Per-subject means across runs are written alongside the per-run maps.

Requires FSL (fslmerge) and MRtrix3 (dwidenoise) on PATH.

    python workflow/tools/fit_r2star.py \
        --bids /path/to/bids --derivatives /path/to/derivatives/r2star \
        --echo-times 4.13 8.80 13.47

This is a standalone utility. It is not part of the NeuroMelAnchor pipeline and
nothing in the pipeline reads its outputs; it exists to derive R2* maps from the
same acquisitions for separate analyses.
"""

# =================================================================
# Imports
# =================================================================
import re
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

def sh(cmd):
    """Run a shell command, raising if it fails."""
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

def build_4d_mag(anat, sub, run, out_path):
    """Merge three magnitude echoes into a 4D volume."""
    if out_path.exists():
        return
    echoes = [anat / f"{sub}_run-{run}_echo-{e}_part-mag_MEGRE.nii" for e in (1, 2, 3)]
    missing = [e for e in echoes if not e.exists()]
    if missing:
        raise FileNotFoundError(f"missing echoes for {sub} run-{run}: {missing}")
    sh(f"fslmerge -t {out_path} {' '.join(str(e) for e in echoes)}")

def make_threshold_mask(mag_4d_path, out_path):
    """Brain mask via 20th-percentile threshold on echo-1, then largest-component + fill."""
    if out_path.exists():
        return
    img = nib.load(mag_4d_path)
    e1 = img.get_fdata()[..., 0]
    thresh = np.percentile(e1[e1 > 0], 20)
    m = e1 > thresh
    m = ndimage.binary_erosion(m, iterations=1)
    lbl, n = ndimage.label(m)
    if n > 0:
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        m = lbl == (np.argmax(sizes) + 1)
    m = ndimage.binary_dilation(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    nib.save(nib.Nifti1Image(m.astype(np.uint8), img.affine, img.header), out_path)

def denoise(mag_4d_path, out_path):
    """MP-PCA denoise the 4D magnitude. Writes a noise map for QA next to it."""
    if out_path.exists():
        return
    noise_path = out_path.parent / "noise_map.nii.gz"
    sh(f"dwidenoise {mag_4d_path} {out_path} -noise {noise_path}")

def fit_r2star_wls(mag_4d_path, mask_path, out_r2, out_t2, tes_sec):
    """Vectorized weighted log-linear R2* fit with noise-floor guard."""
    img = nib.load(mag_4d_path)
    mag = img.get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata() > 0

    # Drop voxels where the last echo is simply too noisey (SNR less than 3)
    noise_std = mag[:30, :30, :, 0].std()
    mask = mask & (mag[..., -1] > 3 * noise_std)

    # WLS: weights = S² (Chebyshev / inverse-variance under log transform)
    log_mag = np.log(np.clip(mag, 1e-6, None))
    n_e = mag.shape[-1]
    y = log_mag.reshape(-1, n_e).T          # (n_echoes, n_vox)
    W = (mag.reshape(-1, n_e).T) ** 2

    TE = tes_sec[:, None]
    Sw = W.sum(axis=0)
    Sw_TE = (W * TE).sum(axis=0)
    Sw_TE2 = (W * TE**2).sum(axis=0)
    Sw_y = (W * y).sum(axis=0)
    Sw_TE_y = (W * TE * y).sum(axis=0)

    det = Sw * Sw_TE2 - Sw_TE**2
    slope = (Sw * Sw_TE_y - Sw_TE * Sw_y) / np.where(det != 0, det, 1)
    r2 = -slope.reshape(mag.shape[:3])
    r2 = np.where(mask, r2, 0.0).clip(0, 300)

    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(r2 > 0, 1000.0 / r2, 0).clip(0, 200)

    nib.save(nib.Nifti1Image(r2.astype(np.float32), img.affine, img.header), out_r2)
    nib.save(nib.Nifti1Image(t2.astype(np.float32), img.affine, img.header), out_t2)
    return r2, t2, mask


def process_dataset(bids, derivs, tes_sec):
    """Fit every subject and run under a BIDS root, writing into the derivatives root."""
    print(f"BIDS: {bids}")
    print(f"OUTPUT: {derivs}\n")

    for sub_dir in sorted(bids.glob("sub-*")):
        sub = sub_dir.name
        anat = sub_dir / "anat"
        if not anat.exists():
            print(f"Skipping {sub}: no anat folder")
            continue

        # Discover runs present for this subject from BIDS filenames
        runs = sorted({int(m.group(1)) for f in anat.glob(f"{sub}_run-*_part-mag_MEGRE.nii")
                       for m in [re.search(r"run-(\d+)", f.name)] if m})
        if not runs:
            print(f"Skipping {sub}: no bids-named mag files")
            continue

        print(f"=== {sub}: runs = {runs} ===")
        per_run_r2 = []
        per_run_t2 = []
        for run in runs:
            out_dir = derivs / sub / f"run-{run}"
            out_dir.mkdir(parents=True, exist_ok=True)
            mag_4d = out_dir / "mag_4D.nii.gz"
            mag_4d_dn = out_dir / "mag_4D_dn.nii.gz"
            mask = out_dir / "brain_mask.nii.gz"
            r2_out = out_dir / f"{sub}_run-{run}_R2star.nii.gz"
            t2_out = out_dir / f"{sub}_run-{run}_T2star.nii.gz"

            try:
                build_4d_mag(anat, sub, run, mag_4d)
                make_threshold_mask(mag_4d, mask)
                denoise(mag_4d, mag_4d_dn)
                r2, t2, m = fit_r2star_wls(mag_4d_dn, mask, r2_out, t2_out, tes_sec)
                per_run_r2.append(r2)
                per_run_t2.append(t2)
                print(f"  run-{run}: median R2*={np.median(r2[m]):.1f}  "
                      f"P95={np.percentile(r2[m], 95):.1f}  "
                      f"voxels={int(m.sum())}")
            except Exception as e:
                print(f"  run-{run}: FAILED: {e}")
                continue

        # Within-subject mean across runs
        if per_run_r2:
            ref_img = nib.load(derivs / sub / f"run-{runs[0]}" / f"{sub}_run-{runs[0]}_R2star.nii.gz")

            mean_r2 = np.mean(per_run_r2, axis=0)
            nib.save(
                nib.Nifti1Image(mean_r2.astype(np.float32), ref_img.affine, ref_img.header),
                derivs / sub / f"{sub}_R2star_mean.nii.gz"
            )

            mean_t2 = np.mean(per_run_t2, axis=0)
            nib.save(
                nib.Nifti1Image(mean_t2.astype(np.float32), ref_img.affine, ref_img.header),
                derivs / sub / f"{sub}_T2star_mean.nii.gz"
            )
            print(f"  [{sub}] wrote subject-mean R2* and T2* across {len(per_run_r2)} run(s)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids", type=Path, required=True,
                        help="BIDS root containing sub-*/anat MEGRE magnitude images.")
    parser.add_argument("--derivatives", type=Path, required=True,
                        help="Output derivatives directory for the R2*/T2* maps.")
    parser.add_argument("--echo-times", type=float, nargs="+", default=[4.13, 8.80, 13.47],
                        metavar="MS", help="Echo times in milliseconds (default: 4.13 8.80 13.47).")
    args = parser.parse_args()

    # Fail fast rather than partway through a long loop.
    for tool in ("fslmerge", "dwidenoise"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not on PATH - run: module load fsl mrtrix3")

    process_dataset(args.bids, args.derivatives, np.asarray(args.echo_times) / 1000.0)


if __name__ == "__main__":
    main()
