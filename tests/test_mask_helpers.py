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

from conftest import load_functions


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


# ---------------------------------------------------------------- dilate_search_region

@pytest.fixture(scope="module")
def dilate_search_region():
    return load_functions("dilate_search_region")["dilate_search_region"]


def test_search_region_without_dilation_is_the_binarised_drawing(nifti_factory, dilate_search_region):
    """--search_dilation_mm 0 must reproduce the propagated drawing exactly."""
    drawing = np.zeros((9, 9, 9), dtype=np.float32)
    drawing[3:6, 3:6, 3:6] = 1

    out = dilate_search_region(nifti_factory("sn.nii.gz", drawing), dilation_mm=0.0)

    region = nib.load(out)
    assert Path(out).name == "Native_SN_Search_Region.nii.gz"
    assert region.get_data_dtype() == np.uint8
    assert np.array_equal(region.get_fdata() > 0, drawing > 0)


def test_search_region_grows_by_millimetres_on_anisotropic_voxels(nifti_factory, dilate_search_region):
    drawing = np.zeros((21, 21, 21), dtype=np.float32)
    drawing[10, 10, 10] = 1
    # 2 mm is 4 voxels along x (0.5 mm), 2 along y (1 mm) and 1 along z (2 mm).
    path = nifti_factory("sn.nii.gz", drawing, voxel_size=(0.5, 1.0, 2.0))

    region = nib.load(dilate_search_region(path, dilation_mm=2.0)).get_fdata() > 0

    assert region[14, 10, 10] and not region[15, 10, 10]
    assert region[10, 12, 10] and not region[10, 13, 10]
    assert region[10, 10, 11] and not region[10, 10, 12]
    assert region[drawing > 0].all(), "the grown region must contain the drawing"


# ----------------------------------------------------------- extract_tractography_seed

def test_seed_closing_never_leaves_the_search_region(nifti_factory):
    """
    The v3.0 morphological closing is intersected with the search region.

    Two bright slabs sit either side of a one-voxel gap that is outside the search
    region. Closing the seed would bridge that gap; the intersection must remove it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import scipy.ndimage as ndi
    extract_tractography_seed = load_functions("extract_tractography_seed")["extract_tractography_seed"]

    shape = (24, 24, 24)
    inner = (slice(6, 19), slice(6, 19))
    nm = np.zeros(shape, dtype=np.float32)
    search = np.zeros(shape, dtype=np.float32)
    cp = np.zeros(shape, dtype=np.float32)

    # Search region: slabs at x = 5..9 and x = 11..15, brightest beside the gap at x = 10.
    for x, intensity in [*zip(range(5, 10), [120, 140, 160, 180, 200]),
                         *zip(range(11, 16), [200, 180, 160, 140, 120])]:
        nm[(x, *inner)] = intensity
        search[(x, *inner)] = 1

    # Cerebral peduncle: background intensity around 100, with spread for the KDE.
    cp_block = (slice(18, 23), *inner)
    nm[cp_block] = 100 + np.random.RandomState(0).normal(0, 3, size=nm[cp_block].shape)
    cp[cp_block] = 1

    _, seed_out, _, diagnostics_out = extract_tractography_seed(
        nifti_factory("nm.nii.gz", nm), nifti_factory("search.nii.gz", search),
        nifti_factory("cp.nii.gz", cp),
        cnr_floor=0.05, cnr_percentile=55, min_cluster_vox=15, keep_components=2)

    seed = nib.load(seed_out).get_fdata() > 0
    inside = search > 0

    assert seed.any()
    assert not (seed & ~inside).any(), "seed voxels outside the search region"
    assert not seed[10].any(), "closing bridged the gap outside the search region"
    # Closing alone would fill the gap, so the assertions above really test the intersection.
    assert ndi.binary_closing(seed, structure=np.ones((3, 3, 3), dtype=bool))[10].any()

    diagnostics = json.loads(Path(diagnostics_out).read_text())
    assert diagnostics["seed_voxels"] == int(seed.sum())
    assert diagnostics["search_roi_voxels"] == int(inside.sum())


# ----------------------------------------------------------- T1-space seed (oct-2)

def test_first_nm_file_returns_the_first_in_sorted_order():
    first_nm_file = load_functions("first_nm_file")["first_nm_file"]
    assert first_nm_file(["run-2.nii.gz", "run-1.nii.gz"]) == "run-1.nii.gz"
    assert first_nm_file("only.nii.gz") == "only.nii.gz"
    with pytest.raises(ValueError):
        first_nm_file([])


def test_threshold_t1_seed_keeps_the_two_largest_components(nifti_factory):
    threshold_t1_seed = load_functions("threshold_t1_seed")["threshold_t1_seed"]

    fraction = np.zeros((30, 30, 30), dtype=np.float32)
    fraction[2:6, 2:6, 2:6] = 0.9          # 64 voxels
    fraction[10:13, 10:13, 10:13] = 0.6    # 27 voxels
    fraction[20:22, 20:22, 20:22] = 0.7    # 8 voxels: third largest, dropped
    fraction[25:28, 25:28, 25:28] = 0.4    # below 0.5, never part of the seed

    out = threshold_t1_seed(nifti_factory("fraction.nii.gz", fraction), keep_components=2)

    seed = nib.load(out)
    assert Path(out).name == "T1w_Tractography_Seed_SNc.nii.gz"
    assert seed.get_data_dtype() == np.uint8
    assert int(np.asanyarray(seed.dataobj).sum()) == 64 + 27


def test_t1_seed_reference_keeps_the_field_of_view(nifti_factory):
    make_t1_seed_reference = load_functions("make_t1_seed_reference")["make_t1_seed_reference"]

    t1 = nifti_factory("t1.nii.gz", np.zeros((20, 30, 40), dtype=np.float32))
    reference = nib.load(make_t1_seed_reference(t1, 0.5))

    assert reference.shape == (40, 60, 80)
    assert np.allclose(reference.header.get_zooms()[:3], 0.5)
    # The outer corner of the grid stays where the T1's was.
    corner = lambda img: nib.affines.apply_affine(img.affine, [-0.5, -0.5, -0.5])
    assert np.allclose(corner(reference), corner(nib.load(t1)))


def test_seed_slice_fill_stays_inside_the_search_region(nifti_factory):
    import matplotlib
    matplotlib.use("Agg")
    extract_tractography_seed = load_functions("extract_tractography_seed")["extract_tractography_seed"]

    shape, inner = (24, 24, 24), (slice(6, 19), slice(6, 19))
    nm = np.zeros(shape, dtype=np.float32); search = np.zeros(shape, dtype=np.float32); cp = np.zeros(shape, dtype=np.float32)
    for x, intensity in [*zip(range(5, 10), [120, 140, 160, 180, 200]), *zip(range(11, 16), [200, 180, 160, 140, 120])]:
        nm[(x, *inner)] = intensity; search[(x, *inner)] = 1
    cp_block = (slice(18, 23), *inner)
    nm[cp_block] = 100 + np.random.RandomState(0).normal(0, 3, size=nm[cp_block].shape); cp[cp_block] = 1

    _, seed_out, _, _ = extract_tractography_seed(
        nifti_factory("nm.nii.gz", nm), nifti_factory("search.nii.gz", search), nifti_factory("cp.nii.gz", cp),
        cnr_floor=0.0, cnr_percentile=55, min_cluster_vox=15, keep_components=2, cnr_fwhm=1.5, slice_fill=True)

    seed = nib.load(seed_out).get_fdata() > 0
    assert seed.any() and not (seed & ~(search > 0)).any()
