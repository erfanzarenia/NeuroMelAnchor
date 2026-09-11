#!/usr/bin/env python3
"""
Build the brainstem weighting mask used by the Stage 1 T1 to MNI registration.

Label 8 of the Harvard-Oxford subcortical atlas is the brainstem. It is dilated
slightly so the weighting mask covers the brainstem border rather than stopping
exactly at it, which would bias the SyN cost against the edge voxels that matter
most for midbrain alignment.

    python workflow/tools/make_brainstem_mask.py \
        --out resources/tpl-MNI152NLin2009cAsym/HarvardOxford/Brainstem.nii.gz
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.ndimage
from nilearn import datasets
from nilearn.image import math_img

BRAINSTEM_LABEL = 8


def build_brainstem_mask(atlas_name, dilation_iterations):
    """Return the dilated Harvard-Oxford brainstem as a Nifti1Image."""
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
    brainstem = math_img(f"img == {BRAINSTEM_LABEL}", img=atlas.maps)
    data = brainstem.get_fdata().astype(bool)
    dilated = scipy.ndimage.binary_dilation(
        data, iterations=dilation_iterations).astype(np.int16)
    return nib.Nifti1Image(dilated, brainstem.affine, brainstem.header)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Output mask path.")
    parser.add_argument("--atlas", default="sub-maxprob-thr25-1mm",
                        help="Harvard-Oxford subcortical atlas name (default: sub-maxprob-thr25-1mm).")
    parser.add_argument("--dilate-iterations", type=int, default=2,
                        help="Binary dilation iterations applied to the label (default: 2).")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(build_brainstem_mask(args.atlas, args.dilate_iterations), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
