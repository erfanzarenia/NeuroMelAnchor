"""
Numeric checks on the mask and QC helpers lifted out of the pipeline.

Each test builds a tiny synthetic volume with a known answer, so a change in
thresholding, connected-component pruning or erosion behaviour shows up here
rather than after a multi-hour run on real data.
"""
import json
import os
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from conftest import load_functions, load_objects


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """Helpers write to os.path.abspath(...), so give each test its own cwd."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------- select_nm_image_files

@pytest.fixture(scope="module")
def select_nm_image_files():
    return load_functions("select_nm_image_files")["select_nm_image_files"]


def test_select_nm_mean_returns_all_sorted(select_nm_image_files):
    assert select_nm_image_files(["b.nii", "a.nii"]) == ["a.nii", "b.nii"]


def test_select_nm_first_returns_one(select_nm_image_files):
    assert select_nm_image_files(["b.nii", "a.nii"], strategy="first") == ["a.nii"]


def test_select_nm_mean_honours_n(select_nm_image_files):
    files = ["c.nii", "a.nii", "b.nii"]
    assert select_nm_image_files(files, strategy="mean", n=2) == ["a.nii", "b.nii"]


def test_select_nm_accepts_a_single_string(select_nm_image_files):
    assert select_nm_image_files("only.nii") == ["only.nii"]


def test_select_nm_rejects_empty_input(select_nm_image_files):
    with pytest.raises(ValueError, match="No CombEchoNM files"):
        select_nm_image_files([])


def test_select_nm_rejects_bad_strategy(select_nm_image_files):
    with pytest.raises(ValueError, match="Unknown NM image strategy"):
        select_nm_image_files(["a.nii"], strategy="median")


def test_select_nm_rejects_n_below_one(select_nm_image_files):
    with pytest.raises(ValueError, match=">= 1"):
        select_nm_image_files(["a.nii"], strategy="mean", n=0)


# ---------------------------------------------------------- threshold_probability_mask

def test_threshold_probability_mask_keeps_voxels_at_the_threshold(nifti_factory):
    threshold_probability_mask = load_functions(
        "threshold_probability_mask")["threshold_probability_mask"]

    probabilities = np.zeros((4, 4, 4), dtype=np.float32)
    probabilities[1, 1, 1] = 0.10
    probabilities[2, 2, 2] = 0.25   # exactly at the threshold: kept
    probabilities[3, 3, 3] = 0.90
    prob_path = nifti_factory("prob.nii.gz", probabilities)

    _, mask_out = threshold_probability_mask(prob_path, threshold=0.25, label="ProbCIT_SNc")

    mask = nib.load(mask_out).get_fdata()
    assert mask.sum() == 2
    assert mask[2, 2, 2] == 1 and mask[3, 3, 3] == 1 and mask[1, 1, 1] == 0
    assert Path(mask_out).name == "Native_ProbCIT_SNc_thr025.nii.gz"
    assert nib.load(mask_out).get_data_dtype() == np.uint8


def test_threshold_probability_mask_sanitises_the_label(nifti_factory):
    threshold_probability_mask = load_functions(
        "threshold_probability_mask")["threshold_probability_mask"]

    prob_path = nifti_factory("prob.nii.gz", np.ones((3, 3, 3), dtype=np.float32))
    _, mask_out = threshold_probability_mask(prob_path, threshold=0.5, label="NMS/SNc atlas")
    assert "/" not in Path(mask_out).name
    assert Path(mask_out).name == "Native_NMS_SNc_atlas_thr050.nii.gz"


# ------------------------------------------------------------- create_core_rim_nm_mask

@pytest.fixture(scope="module")
def create_core_rim_nm_mask():
    return load_functions("create_core_rim_nm_mask")["create_core_rim_nm_mask"]


def _core_rim_inputs(nifti_factory, cnr_core, cnr_rim):
    """A 7x7x7 SN cube whose interior and boundary carry different CNR values."""
    shape = (11, 11, 11)
    sn = np.zeros(shape, dtype=np.float32)
    sn[2:9, 2:9, 2:9] = 1

    cnr = np.zeros(shape, dtype=np.float32)
    cnr[2:9, 2:9, 2:9] = cnr_rim      # whole cube at the rim value
    cnr[3:8, 3:8, 3:8] = cnr_core     # interior (the eroded core) at the core value

    return nifti_factory("cnr.nii.gz", cnr), nifti_factory("sn.nii.gz", sn)


def test_core_rim_applies_the_stricter_threshold_to_the_rim(
        nifti_factory, create_core_rim_nm_mask):
    # Rim CNR sits between the two thresholds, so the rim must be rejected.
    cnr_path, sn_path = _core_rim_inputs(nifti_factory, cnr_core=0.20, cnr_rim=0.07)

    mask_out, summary_out = create_core_rim_nm_mask(
        cnr_path, sn_path, core_erosion_mm=1.0,
        core_cnr_threshold=0.05, rim_cnr_threshold=0.10,
        min_cluster_vox=1, keep_components=2)

    mask = nib.load(mask_out).get_fdata()
    summary = json.loads(Path(summary_out).read_text())

    assert summary["rim_candidate_voxels"] == 0
    assert summary["core_candidate_voxels"] == 5 ** 3
    assert mask.sum() == 5 ** 3
    assert mask[2, 5, 5] == 0      # rim voxel dropped
    assert mask[5, 5, 5] == 1      # core voxel kept


def test_core_rim_keeps_the_rim_when_it_clears_the_strict_threshold(
        nifti_factory, create_core_rim_nm_mask):
    cnr_path, sn_path = _core_rim_inputs(nifti_factory, cnr_core=0.20, cnr_rim=0.20)

    mask_out, _ = create_core_rim_nm_mask(
        cnr_path, sn_path, core_erosion_mm=1.0,
        core_cnr_threshold=0.05, rim_cnr_threshold=0.10,
        min_cluster_vox=1, keep_components=2)

    assert nib.load(mask_out).get_fdata().sum() == 7 ** 3


def test_core_rim_prunes_small_components(nifti_factory, create_core_rim_nm_mask):
    shape = (16, 16, 16)
    sn = np.zeros(shape, dtype=np.float32)
    sn[2:6, 2:6, 2:6] = 1     # 64-voxel blob
    sn[12, 12, 12] = 1        # single-voxel speck, far away

    cnr = np.zeros(shape, dtype=np.float32)
    cnr[sn > 0] = 0.5

    mask_out, summary_out = create_core_rim_nm_mask(
        nifti_factory("cnr.nii.gz", cnr), nifti_factory("sn.nii.gz", sn),
        core_erosion_mm=0.0, core_cnr_threshold=0.05, rim_cnr_threshold=0.10,
        min_cluster_vox=10, keep_components=2)

    mask = nib.load(mask_out).get_fdata()
    assert mask[12, 12, 12] == 0, "single-voxel component should have been pruned"
    assert mask.sum() == 64
    assert json.loads(Path(summary_out).read_text())["kept_components"] == 1


def test_core_rim_rejects_mismatched_geometry(nifti_factory, create_core_rim_nm_mask):
    cnr_path = nifti_factory("cnr.nii.gz", np.zeros((8, 8, 8), dtype=np.float32))
    sn_path = nifti_factory("sn.nii.gz", np.zeros((6, 6, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="Shape mismatch"):
        create_core_rim_nm_mask(cnr_path, sn_path)


def test_core_rim_rejects_mismatched_affines(nifti_factory, create_core_rim_nm_mask):
    data = np.zeros((8, 8, 8), dtype=np.float32)
    cnr_path = nifti_factory("cnr.nii.gz", data, voxel_size=(1.0, 1.0, 1.0))
    sn_path = nifti_factory("sn.nii.gz", data, voxel_size=(2.0, 2.0, 2.0))

    with pytest.raises(ValueError, match="Affine mismatch"):
        create_core_rim_nm_mask(cnr_path, sn_path)


# ------------------------------------------------------------------- build_sn_roi_mask

def test_build_sn_roi_mask_dilates_by_millimetres(nifti_factory):
    build_sn_roi_mask = load_functions("build_sn_roi_mask")["build_sn_roi_mask"]

    seed = np.zeros((21, 21, 21), dtype=np.float32)
    seed[10, 10, 10] = 1
    # 2 mm voxels: a 6 mm dilation reaches 3 voxels, not 6.
    seed_path = nifti_factory("seed.nii.gz", seed, voxel_size=(2.0, 2.0, 2.0))

    roi = nib.load(build_sn_roi_mask(seed_path, dilation_mm=6.0)).get_fdata()

    assert roi[10, 10, 10] == 1
    assert roi[13, 10, 10] == 1     # exactly 6 mm away
    assert roi[14, 10, 10] == 0     # 8 mm away
    assert roi[10, 10, 6] == 0


def test_build_sn_roi_mask_rejects_an_empty_seed(nifti_factory):
    build_sn_roi_mask = load_functions("build_sn_roi_mask")["build_sn_roi_mask"]

    seed_path = nifti_factory("seed.nii.gz", np.zeros((8, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="seed mask is empty"):
        build_sn_roi_mask(seed_path)


# ---------------------------------------------------------------------- binarize_mask

def test_binarize_mask_thresholds_inclusively(nifti_factory):
    binarize_mask = load_functions("binarize_mask")["binarize_mask"]

    data = np.array([[[0.0, 0.49], [0.5, 1.0]]], dtype=np.float32)
    out = binarize_mask(nifti_factory("prob.nii.gz", data), threshold=0.5)

    assert nib.load(out).get_fdata().sum() == 2
    assert nib.load(out).get_data_dtype() == np.uint8


# ------------------------------------------------------------------ compute_motion_params

def test_compute_motion_params_reports_parameter_ranges(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    compute_motion_params = load_functions("compute_motion_params")["compute_motion_params"]

    # Six SPM realignment columns: x, y, z in mm then pitch, roll, yaw in radians.
    parameters = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 0.0, 0.0, np.radians(4.0)],
    ])
    parameter_file = tmp_path / "rp.txt"
    np.savetxt(parameter_file, parameters)

    stats_file = compute_motion_params(str(parameter_file))
    stats = json.loads(Path(stats_file).read_text())

    assert stats["translation_range_mm"]["x"] == pytest.approx(1.0)
    assert stats["translation_range_mm"]["max"] == pytest.approx(3.0)
    assert stats["rotation_range_deg"]["yaw"] == pytest.approx(4.0)
    assert stats["rotation_range_deg"]["max"] == pytest.approx(4.0)


def test_compute_motion_params_handles_a_single_volume(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    compute_motion_params = load_functions("compute_motion_params")["compute_motion_params"]

    parameter_file = tmp_path / "rp.txt"
    np.savetxt(parameter_file, np.zeros((1, 6)))

    stats_file = compute_motion_params(str(parameter_file))
    stats = json.loads(Path(stats_file).read_text())
    assert stats["translation_range_mm"]["max"] == 0.0


# ------------------------------------------------------------- resolve_input_patterns

@pytest.fixture(scope="module")
def resolve_input_patterns():
    return load_objects("COHORT_INPUT_PATTERNS",
                        "resolve_input_patterns")["resolve_input_patterns"]


def test_cohort_patterns_keep_the_subject_placeholder(resolve_input_patterns):
    """SelectFiles fills {subject_id} itself, so it must survive resolution."""
    nm, t1 = resolve_input_patterns("biogen", "combo", 3)
    assert "{subject_id}" in nm and "{subject_id}" in t1
    assert nm.endswith("_acq-CombEchoNM*_GRE.nii.gz")


def test_echo_number_is_substituted(resolve_input_patterns):
    nm, _ = resolve_input_patterns("biogen", "echo", 3)
    assert "echo-3" in nm and "{echo}" not in nm


def test_snit_cohort_uses_its_own_naming(resolve_input_patterns):
    nm, t1 = resolve_input_patterns("snit", "echo", 2)
    assert "rec-DIS2D" in nm and "echo-2" in nm
    assert "MP2RAGEpostproc" in t1


def test_overrides_win_over_the_cohort_defaults(resolve_input_patterns):
    nm, t1 = resolve_input_patterns(
        "biogen", "combo", 3,
        nm_override="{subject_id}/anat/custom_echo-{echo}.nii.gz",
        t1_override="{subject_id}/anat/custom_T1w.nii.gz")
    assert nm == "{subject_id}/anat/custom_echo-3.nii.gz"
    assert t1 == "{subject_id}/anat/custom_T1w.nii.gz"
