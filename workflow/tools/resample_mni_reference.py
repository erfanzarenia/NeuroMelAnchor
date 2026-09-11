#!/usr/bin/env python3
"""
Write the MNI152NLin2009cAsym brain references used by the pipeline.

The 1 mm reference is copied straight from TemplateFlow; the 0.5 mm one is
produced by B-spline resampling the 1 mm volume onto a finer grid with the same
origin and direction. Both land in ``resources/tpl-MNI152NLin2009cAsym/``.

The 0.5 mm file is not committed to the repository (it is ~30 MB and fully
derived), so run this once after cloning if you intend to use ``--hires`` or the
V4 drawing template.

    python workflow/tools/resample_mni_reference.py \
        --out-dir resources/tpl-MNI152NLin2009cAsym
"""
import argparse
from pathlib import Path

import SimpleITK as sitk
from templateflow.api import get


def fetch_template(resolution=1):
    """Return the TemplateFlow path to the brain-extracted MNI T1 at a resolution."""
    path = get('MNI152NLin2009cAsym', resolution=resolution, desc='brain',
               suffix='T1w', extension='nii.gz')
    if isinstance(path, list):
        path = path[0]
    return Path(path)


def resample_isotropic(image, spacing_mm):
    """Resample an image to an isotropic spacing, preserving origin and direction."""
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [int(round(size * spacing / spacing_mm))
                for size, spacing in zip(old_size, old_spacing)]
    return sitk.Resample(
        image, new_size, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline,
        image.GetOrigin(), (spacing_mm,) * 3, image.GetDirection(), 0.0,
        image.GetPixelID(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write the references into.")
    parser.add_argument("--spacing", type=float, default=0.5,
                        help="Isotropic spacing in mm for the resampled output (default: 0.5).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = sitk.ReadImage(str(fetch_template(resolution=1)))

    one_mm_out = args.out_dir / "MNI152NLin2009cAsym_res-1_desc-brain_T1w.nii.gz"
    sitk.WriteImage(reference, str(one_mm_out))
    print(f"wrote {one_mm_out}")

    label = f"{args.spacing:g}".replace("0.", "0")
    fine_out = args.out_dir / f"MNI152NLin2009cAsym_res-{label}_desc-brain_T1w.nii.gz"
    sitk.WriteImage(resample_isotropic(reference, args.spacing), str(fine_out))
    print(f"wrote {fine_out}")


if __name__ == "__main__":
    main()
