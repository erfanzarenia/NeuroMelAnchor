#!/usr/bin/env python3
"""
Build the midbrain prior used to crop subjects before template construction.

BrainstemNavigator segments the midbrain into many small nuclei, none of which is
a usable crop region on its own. This takes the union of the midbrain nuclei,
fills each axial cross-section with its convex hull to turn that union into a
solid midbrain body, and closes the result.

It also writes the union of that body with the Harvard-Oxford brainstem, giving a
prior that covers midbrain plus pons and medulla. The two atlases do not share a
voxel grid, so Harvard-Oxford is resampled onto the BrainstemNavigator grid by
world coordinates with nearest-neighbour lookup.

    python workflow/tools/make_midbrain_prior.py \
        --brainstem-navigator /path/to/BrainstemNavigatorv1.0/1.0/2a.BrainstemNucleiAtlas_MNI/labels_thresholded_binary_0.35 \
        --harvard-oxford-brainstem resources/tpl-MNI152NLin2009cAsym/HarvardOxford/Brainstem.nii.gz \
        --out resources/tpl-MNI152NLin2009cAsym/HarvardOxford/Midbrain.nii.gz \
        --out-combined resources/tpl-MNI152NLin2009cAsym/HarvardOxford/Brainstem_Midbrain.nii.gz

BrainstemNavigator is not redistributable, so the committed Midbrain.nii.gz was
produced with this script and is kept in the repository for convenience.
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

# Midbrain nuclei whose union defines the crop region. SN and VTA are the targets;
# the rest are included so the convex hull spans the whole midbrain cross-section
# rather than hugging the dopaminergic nuclei.
MIDBRAIN_NUCLEI = [
    "SN_l", "SN_r", "VTA_PBP_l", "VTA_PBP_r", "RN_l", "RN_r", "PAG",
    "SC_l", "SC_r", "IC_l", "IC_r", "CnF_l", "CnF_r", "mRt_l", "mRt_r",
    "isRt_l", "isRt_r", "MiTg_PBG_l", "MiTg_PBG_r", "CLi_RLi",
]


def union_of_nuclei(atlas_dir, nuclei, reference):
    """Boolean union of the named binary nucleus labels, on the reference grid."""
    union = np.zeros(reference.shape, bool)
    for name in nuclei:
        label_path = atlas_dir / f"{name}.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"BrainstemNavigator label not found: {label_path}")
        union |= np.asarray(nib.load(label_path).dataobj) > 0
    return union


def solidify_axially(mask):
    """Fill each axial slice with its convex hull, then close in 3D."""
    solid = np.zeros_like(mask)
    for k in range(mask.shape[2]):
        if mask[:, :, k].any():
            solid[:, :, k] = convex_hull_image(mask[:, :, k])
    return ndi.binary_closing(solid, np.ones((3, 3, 3)), 1)


def resample_nearest_to(source_img, reference_img):
    """
    Nearest-neighbour resample a binary image onto another image's grid.

    Done by world coordinates rather than by index because the two atlases differ
    in orientation (RAS vs x-flipped LAS), so index-space alignment would mirror
    the mask.
    """
    source_data = np.asarray(source_img.dataobj) > 0
    ijk = np.indices(reference_img.shape).reshape(3, -1).T
    world = nib.affines.apply_affine(reference_img.affine, ijk)
    source_ijk = np.rint(
        nib.affines.apply_affine(np.linalg.inv(source_img.affine), world)).astype(int)
    inside = np.all((source_ijk >= 0) & (source_ijk < np.array(source_img.shape)), axis=1)

    out = np.zeros(ijk.shape[0], bool)
    out[inside] = source_data[source_ijk[inside, 0], source_ijk[inside, 1], source_ijk[inside, 2]]
    return out.reshape(reference_img.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brainstem-navigator", type=Path, required=True,
                        help="Directory of thresholded binary BrainstemNavigator labels.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the solid midbrain prior.")
    parser.add_argument("--harvard-oxford-brainstem", type=Path, default=None,
                        help="Harvard-Oxford brainstem mask; enables the combined output.")
    parser.add_argument("--out-combined", type=Path, default=None,
                        help="Output path for midbrain union brainstem.")
    parser.add_argument("--reference-label", default="PAG",
                        help="Label whose grid defines the output geometry (default: PAG).")
    args = parser.parse_args()

    if bool(args.harvard_oxford_brainstem) != bool(args.out_combined):
        parser.error("--harvard-oxford-brainstem and --out-combined must be given together.")

    reference = nib.load(args.brainstem_navigator / f"{args.reference_label}.nii.gz")
    solid = solidify_axially(
        union_of_nuclei(args.brainstem_navigator, MIDBRAIN_NUCLEI, reference))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(solid.astype("uint8"), reference.affine, reference.header), args.out)
    print(f"wrote {args.out}  ({int(solid.sum())} voxels)")

    if args.out_combined:
        brainstem = resample_nearest_to(nib.load(args.harvard_oxford_brainstem), reference)
        combined = solid | brainstem
        args.out_combined.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(combined.astype("uint8"), reference.affine, reference.header),
                 args.out_combined)
        print(f"wrote {args.out_combined}  ({int(combined.sum())} voxels)")


if __name__ == "__main__":
    main()
