"""
Checks on the post-hoc figure scripts.

Unlike the pipeline, these parse their arguments inside main(), so they can be
imported directly.
"""
import csv
import importlib.util

import pytest

from conftest import FIGURES_DIR


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), FIGURES_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return load_script("poster_dilation_sweep.py")


@pytest.fixture(scope="module")
def panels():
    return load_script("poster_workflow_panels.py")


def write_summary(run_dir, rows):
    run_dir.mkdir(parents=True)
    with open(run_dir / "Group_QC_Pairwise_Summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "pair", "dice"])
        writer.writeheader()
        writer.writerows(rows)


def test_runs_are_ordered_by_dilation_and_archives_are_skipped(tmp_path, sweep):
    write_summary(tmp_path / "results_dil-1.0", [])
    write_summary(tmp_path / "results_dil-0.5", [])
    (tmp_path / "results_dil-0").mkdir()                      # no summary yet
    (tmp_path / "results_dil-2.5.tar.gz").write_bytes(b"")    # an archive that matches the glob

    runs = sweep.collect_runs(tmp_path, "results_dil-*")

    assert [dilation for dilation, _ in runs] == [0.5, 1.0]


def test_metrics_are_read_for_the_requested_pair_only(tmp_path, sweep):
    write_summary(tmp_path / "results_dil-0.5", [
        {"subject_id": "sub-01", "pair": "cnr_vs_manual", "dice": "0.7"},
        {"subject_id": "sub-01", "pair": "atlas_vs_manual", "dice": "0.4"},
        {"subject_id": "sub-02", "pair": "cnr_vs_manual", "dice": ""},
    ])
    assert sweep.read_metric(tmp_path / "results_dil-0.5", "dice", "cnr_vs_manual") == {"sub-01": 0.7}


def test_panel_crop_is_pushed_back_inside_the_image(panels):
    # 48 x 36 mm at 0.5 mm voxels is a 96 x 72 voxel crop.
    rows, cols = panels.crop_at((1, 1), (0.5, 0.5), (100, 80), (48.0, 36.0))
    assert (rows.start, rows.stop, cols.start, cols.stop) == (0, 96, 0, 72)

    rows, cols = panels.crop_at((99, 79), (0.5, 0.5), (100, 80), (48.0, 36.0))
    assert (rows.start, rows.stop, cols.start, cols.stop) == (4, 100, 8, 80)


def test_panels_are_saved_edge_to_edge(tmp_path, panels):
    """Panels fill the figure, so the PNG is exactly the figure size, with no white border."""
    import numpy as np
    import matplotlib.image as mpimg

    output = tmp_path / "panel.png"
    panels.save_panel(np.random.RandomState(0).rand(96, 72) + 1.0, [], output,
                      fov_mm=(48.0, 36.0), mm_per_inch=8.0)

    # 48 x 36 mm at 8 mm per inch is 6 x 4.5 in, which is 2400 x 1800 px at 400 dpi.
    assert mpimg.imread(output).shape[:2] == (1800, 2400)
    assert output.with_suffix(".svg").exists()
