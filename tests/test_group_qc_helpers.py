"""
Checks on the QC reports in workflow/qc/group_qc.py.

Unlike the pipeline, group_qc.py parses its arguments inside main(), so it is
imported directly.
"""
import importlib.util
import json

import numpy as np
import nibabel as nib
import pytest

from conftest import QC_DIR


@pytest.fixture(scope="module")
def group_qc():
    spec = importlib.util.spec_from_file_location("group_qc", QC_DIR / "group_qc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nm_blind_control_rows(group_qc):
    return group_qc.nm_blind_control_rows


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

    rows = nm_blind_control_rows(["sub-01", "sub-02"], output_root, manual_dir, seed_cnr_percentile=55)

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


def test_accuracy_figure_survives_tied_metrics(tmp_path, group_qc):
    """A one-subject run whose HD95 ties across methods must not abort group QC."""
    record = {"dice_cnr_vs_manual": 0.72, "dice_atlas_vs_manual": 0.41,
              "dice_prob_atlas_vs_manual": 0.55, "dice_nms_atlas_vs_manual": 0.60,
              "hd95_cnr_vs_manual_mm": 1.0, "hd95_atlas_vs_manual_mm": 1.0,
              "hd95_prob_atlas_vs_manual_mm": 1.0, "hd95_nms_atlas_vs_manual_mm": 1.0}
    output = tmp_path / "Group_QC_Accuracy.png"

    group_qc.build_accuracy_figure([("sub-01", record)], output)

    assert output.exists()


def test_group_outputs_go_to_their_own_folder(tmp_path, group_qc):
    """Group QC from a QC file alone: summaries land in group_qc/, not beside the subjects."""
    qc_file = tmp_path / "sub-01" / "cnr" / "SN_Mask_QC.json"
    qc_file.parent.mkdir(parents=True)
    qc_file.write_text(json.dumps({"pairwise": {"cnr_vs_atlas": {"dice": 0.5, "hd95_mm": 2.0}}}))

    group_qc.run_group_qc_stage(tmp_path, None, ["sub-01"])

    assert (tmp_path / "group_qc" / "Group_QC_Pairwise_Summary.csv").exists()
    assert not list(tmp_path.glob("Group_QC_*"))
