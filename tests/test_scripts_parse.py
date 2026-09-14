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

from conftest import (REPO_ROOT, SCRIPTS_DIR, TOOLS_DIR, FIGURES_DIR, free_variables,
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
    "--cohort", "--nm_pattern", "--t1_pattern",
    "--search_dilation_mm",
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


# Standalone command-line scripts: the tools plus the post-hoc figure scripts.
STANDALONE_SCRIPTS = [
    *(TOOLS_DIR / name for name in [
        "bids_rename.py", "fit_r2star.py", "make_brainstem_mask.py",
        "make_midbrain_prior.py", "resample_mni_reference.py"]),
    FIGURES_DIR / "poster_dilation_sweep.py",
    FIGURES_DIR / "poster_workflow_panels.py",
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


# Files the figure scripts read from a pipeline output root. Renaming one in the
# pipeline would silently break the figures, so each must still be written there.
FIGURE_INPUT_NAMES = [
    "Group_QC_Pairwise_Summary.csv",
    "Native_SN_Search_Region.nii.gz",
    "Native_Tractography_Seed_SNc.nii.gz",
    "MNI_SNc_Manual_trans.nii.gz",
    "mean_NM_bias_corrected.nii.gz",
    "SN_Template_V4_aligned.nii.gz",
    "MNI_SNc_Manual.nii.gz",
    "MNI_CP_Manual.nii.gz",
]


def test_figure_scripts_read_names_the_pipeline_writes(pipeline_text):
    figure_source = "".join(path.read_text() for path in FIGURES_DIR.glob("*.py"))
    for name in FIGURE_INPUT_NAMES:
        assert name in figure_source, f"{name} is no longer read by the figure scripts; update this list"
        assert name in pipeline_text, f"the figure scripts read {name}, which the pipeline no longer writes"


def test_pairwise_summary_has_the_columns_the_sweep_plots(pipeline_text):
    """poster_dilation_sweep.py plots these columns of Group_QC_Pairwise_Summary.csv."""
    for column in ["subject_id", "pair", "dice", "precision", "recall", "hd95_mm"]:
        assert f'"{column}"' in pipeline_text or f"'{column}'" in pipeline_text, column
