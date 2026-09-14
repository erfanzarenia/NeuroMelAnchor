"""
Checks on the group QC helpers added in v3.0.

These run in the main process rather than as Nipype nodes, so they read
module-level imports and constants; the tests supply those explicitly.
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from scipy.ndimage import binary_erosion, distance_transform_edt

from conftest import load_objects


@pytest.fixture(scope="module")
def nm_blind_control_rows():
    namespace = {"np": np, "nib": nib, "Path": Path,
                 "binary_erosion": binary_erosion, "distance_transform_edt": distance_transform_edt}
    return load_objects("seed_cnr_percentile", "nm_blind_control_rows",
                        namespace=namespace)["nm_blind_control_rows"]


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), path)


def cube(shape, centre, half_width):
    mask = np.zeros(shape, dtype=bool)
    mask[centre - half_width:centre + half_width + 1,
         centre - half_width:centre + half_width + 1,
         centre - half_width:centre + half_width + 1] = True
    return mask


def test_nm_blind_control_keeps_the_deepest_share_of_the_search_region(tmp_path, nm_blind_control_rows):
    shape, centre = (31, 31, 31), 15
    search = cube(shape, centre, 5)    # 11^3 = 1331 voxels
    expert = cube(shape, centre, 3)    # 7^3 = 343 voxels: the three deepest layers of the region

    output_root, manual_dir = tmp_path / "out", tmp_path / "manual"
    anat = output_root / "sub-01" / "anat"
    save_mask(anat / "Native_SN_Search_Region.nii.gz", search)
    save_mask(anat / "MNI_SNc_Manual_trans.nii.gz", search)
    save_mask(anat / "Native_Tractography_Seed_SNc.nii.gz", expert)
    save_mask(manual_dir / "sub-01" / "anat" / "sub-01_space-NM_label-SN_desc-manual_mask.nii.gz", expert)

    # A subject without a manual mask is skipped rather than failing the group run.
    for name in ["Native_SN_Search_Region.nii.gz", "MNI_SNc_Manual_trans.nii.gz",
                 "Native_Tractography_Seed_SNc.nii.gz"]:
        save_mask(output_root / "sub-02" / "anat" / name, search)

    rows = nm_blind_control_rows(["sub-01", "sub-02"], output_root, manual_dir)

    assert [row["subject_id"] for row in rows] == ["sub-01"]
    row = rows[0]
    kept = round(0.45 * 1331)          # seed_cnr_percentile = 55 keeps the top 45%
    assert row["search_region_voxels"] == 1331
    assert row["nm_blind_voxels"] == kept
    assert row["nm_seed_voxels"] == 343
    # Deepest first, so all 343 expert voxels are among those kept.
    assert row["dice_nm_blind"] == pytest.approx(2 * 343 / (kept + 343))
    assert row["dice_nm_seed"] == pytest.approx(1.0)
    assert row["dice_drawing"] == pytest.approx(2 * 343 / (1331 + 343))
    assert row["hd95_nm_seed"] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def build_poster_accuracy_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return load_objects("POSTER_METHODS", "save_poster_figure", "build_poster_accuracy_figure",
                        namespace={"np": np, "plt": plt, "Path": Path})["build_poster_accuracy_figure"]


def test_poster_accuracy_survives_tied_metrics(tmp_path, build_poster_accuracy_figure):
    """A one-subject run whose HD95 ties across methods must not abort group QC."""
    record = {"dice_cnr_vs_manual": 0.72, "dice_atlas_vs_manual": 0.41,
              "dice_prob_atlas_vs_manual": 0.55, "dice_nms_atlas_vs_manual": 0.60,
              "hd95_cnr_vs_manual_mm": 1.0, "hd95_atlas_vs_manual_mm": 1.0,
              "hd95_prob_atlas_vs_manual_mm": 1.0, "hd95_nms_atlas_vs_manual_mm": 1.0}
    output = tmp_path / "Poster_Accuracy.png"

    build_poster_accuracy_figure([("sub-01", record)], output)

    assert output.exists() and output.with_suffix(".svg").exists()
