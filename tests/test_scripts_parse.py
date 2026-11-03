"""
Static checks on every script in the repository.

The pipeline cannot be exercised end to end without ANTs, SPM, ITK-SNAP, a GPU
and imaging data, so these checks cover what can be verified statically: that
everything parses, that the command-line contract is intact, and that the
functions Nipype re-serialises are safe to re-serialise.
"""
import ast
import subprocess
import sys

import pytest

from conftest import (REPO_ROOT, SCRIPTS_DIR, TOOLS_DIR, FIGURES_DIR, QC_DIR, free_variables,
                      nipype_function_node_names, pipeline_source)

ALL_PYTHON = sorted(p for p in REPO_ROOT.rglob("*.py") if ".git" not in p.parts)

# Options other code and the documentation depend on. Renaming one silently
# breaks every recorded command, so they are pinned here.
REQUIRED_PIPELINE_OPTIONS = [
    "--project_root", "--output_root", "--design_dir", "--work_dir",
    "--n_procs", "--stage", "--smooth", "--manual_dir", "--test",
    "--prob_atlas_mni", "--prob_atlas_threshold", "--skip_prob_atlas",
    "--nms_atlas_mni", "--nms_atlas_threshold", "--skip_nms_atlas",
    "--skip_core_rim_nm", "--nm_image_strategy", "--nm_image_n",
    "--hires", "--skip_template_builder", "--drawing_template",
    "--rebuild_template", "--nm_input", "--nm_echo",
    "--dataset", "--search_dilation_mm",
    "--seed_cnr_floor", "--seed_cnr_percentile", "--seed_min_cluster_voxels",
    "--seed_components_to_keep", "--seed_cnr_fwhm", "--seed_slice_fill", "--seed_image",
    "--trace_route", "--skip_t1_outputs", "--t1_interp", "--t1_seed_res",
]

EXPECTED_STAGES = ["1", "segment", "2", "group_qc", "all"]


@pytest.mark.parametrize("path", ALL_PYTHON, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_python_files_parse(path):
    ast.parse(path.read_text(), filename=str(path))


def test_shell_scripts_parse():
    for path in TOOLS_DIR.rglob("*.sh"):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


@pytest.mark.parametrize("option", REQUIRED_PIPELINE_OPTIONS)
def test_pipeline_option_present(option, pipeline_text):
    assert f'"{option}"' in pipeline_text or f"'{option}'" in pipeline_text


def test_pipeline_stage_choices(pipeline_text):
    for stage in EXPECTED_STAGES:
        assert f"'{stage}'" in pipeline_text


def test_pipeline_invokes_the_renamed_template_builder(pipeline_text):
    """The Stage `segment` V4 build shells out to the builder by filename."""
    assert '"template_builder.py"' in pipeline_text
    assert "Template_Builder_V4.py" not in pipeline_text
    assert (SCRIPTS_DIR / "template_builder.py").is_file()


@pytest.mark.parametrize("name", sorted(nipype_function_node_names()))
def test_nipype_function_nodes_are_self_contained(name):
    """
    Nipype re-executes `Function(function=f)` bodies in a bare process.

    Any name read but not bound inside the body (a module imported only at the
    top of the script, say) becomes a NameError at run time, inside a worker,
    partway through a multi-hour run.
    """
    tree = ast.parse(pipeline_source())
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert node is not None, f"{name} is passed to Function() but is not a top-level def"

    free = free_variables(node)
    assert not free, (f"{name} reads {sorted(free)} from module scope; import them "
                      f"inside the function or Nipype will fail in the worker")


# Standalone command-line scripts: the tools, the QC reports and the figure script.
STANDALONE_SCRIPTS = [
    *(TOOLS_DIR / name for name in [
        "bids_rename.py", "fit_r2star.py", "make_brainstem_mask.py",
        "make_midbrain_prior.py", "prepare_biogen_3t.py", "resample_mni_reference.py"]),
    QC_DIR / "group_qc.py",
    FIGURES_DIR / "mask_comparison_figures.py",
]


@pytest.mark.parametrize("script", STANDALONE_SCRIPTS, ids=lambda p: p.name)
def test_standalone_help_runs(script):
    """Every standalone script must expose a working --help without touching data."""
    result = subprocess.run([sys.executable, str(script), "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("script", STANDALONE_SCRIPTS, ids=lambda p: p.name)
def test_standalone_requires_arguments(script):
    """Standalone scripts must not fall back to a site-specific default when run bare."""
    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0, f"{script.name} ran with no arguments"


# Files the figure script and the QC reports read from a pipeline output root. Renaming
# one in the pipeline would silently break them, so each must still be written there.
# Names ANTs derives itself (the *_trans outputs) do not appear literally and are left out.
FIGURE_INPUT_NAMES = [
    "mean_NM_bias_corrected.nii.gz",
    "Native_CoreRim_NM_Mask.nii.gz",
    "Native_Tractography_Seed_SNc.nii.gz",
    "SN_Mask_QC.json",
    "Subject_CNR_Map_Smoothed.nii.gz",
]
QC_INPUT_NAMES = [
    "Native_SN_Search_Region.nii.gz",
    "Native_Tractography_Seed_SNc.nii.gz",
    "mean_NM_bias_corrected.nii.gz",
    "Seed_Diagnostics.json",
    "SN_Mask_QC.json",
]


def test_figure_script_reads_names_the_pipeline_writes(pipeline_text):
    figure_source = (FIGURES_DIR / "mask_comparison_figures.py").read_text()
    for name in FIGURE_INPUT_NAMES:
        assert name in figure_source, f"{name} is no longer read by the figure script; update this list"
        assert name in pipeline_text, f"the figure script reads {name}, which the pipeline no longer writes"


def test_group_qc_reads_names_the_pipeline_writes(pipeline_text):
    qc_source = (QC_DIR / "group_qc.py").read_text()
    for name in QC_INPUT_NAMES:
        assert name in qc_source, f"{name} is no longer read by group_qc.py; update this list"
        assert name in pipeline_text, f"group_qc.py reads {name}, which the pipeline no longer writes"


def test_pipeline_runs_group_qc_by_default(pipeline_text):
    assert "from group_qc import run_group_qc_stage" in pipeline_text
    assert "'--skip_group_qc'" in pipeline_text
