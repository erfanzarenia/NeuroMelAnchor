"""
Structural checks on the repository.

These are cheap guards against the kinds of breakage this repository has
actually suffered: resources drifting away from the paths the pipeline hard-codes,
site-specific absolute paths creeping back into scripts, and generated files being
committed.
"""
import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT, SCRIPTS_DIR, TOOLS_DIR, RESOURCES, pipeline_source

EXPECTED_SCRIPTS = [
    "neuromelanchor_pipeline.py",
    "template_builder.py",
    "template_builder_native.py",
    "native_template_prep.py",
]

EXPECTED_TOOLS = [
    "bids_rename.py",
    "fit_r2star.py",
    "make_brainstem_mask.py",
    "make_midbrain_prior.py",
    "mp2rage_uniden.py",
    "mp2rage_uniden_batch.sh",
    "resample_mni_reference.py",
]

EXPECTED_DOCS = [
    "INSTALL.md", "USAGE.md", "WORKFLOW.md", "DATA_SPECIFICATION.md",
    "EVALUATION.md", "MANUAL_SEGMENTATION.md", "REPOSITORY_STRUCTURE.md",
    "STATUS.md", "PROVENANCE.md", "MIGRATION_AUDIT.md",
]

# Resources the pipeline resolves relative to --project_root. If one of these
# moves, Stage 1 or Stage 2 fails at run time with a FileNotFoundError.
REQUIRED_RESOURCES = [
    "HarvardOxford/Brainstem.nii.gz",
    "HarvardOxford/Midbrain.nii.gz",
    "CIT168/SNc.nii.gz",
    "CIT168_prob/CIT168toMNI152-2009c_prob_0006.nii.gz",
    "NMS_prob/NMS_Snc_prob.nii.gz",
]


@pytest.mark.parametrize("name", EXPECTED_SCRIPTS)
def test_pipeline_scripts_present(name):
    assert (SCRIPTS_DIR / name).is_file()


@pytest.mark.parametrize("name", EXPECTED_TOOLS)
def test_tools_present(name):
    assert (TOOLS_DIR / name).is_file()


@pytest.mark.parametrize("name", EXPECTED_DOCS)
def test_docs_present(name):
    assert (REPO_ROOT / "docs" / name).is_file()


@pytest.mark.parametrize("relative", REQUIRED_RESOURCES)
def test_required_resources_present(relative):
    assert (RESOURCES / relative).is_file(), (
        f"{relative} is read by the pipeline; see resources/README.md")


def test_pipeline_resource_paths_resolve():
    """Every resource path the pipeline builds from mask_dir must exist."""
    source = pipeline_source()
    assert 'mask_dir = project_root / "resources" / "tpl-MNI152NLin2009cAsym"' in source

    referenced = {
        "HarvardOxford/Brainstem.nii.gz",
        "CIT168/SNc.nii.gz",
        "CIT168_prob/CIT168toMNI152-2009c_prob_0006.nii.gz",
        "NMS_prob/NMS_Snc_prob.nii.gz",
        "HarvardOxford/Midbrain.nii.gz",
    }
    for relative in referenced:
        parts = relative.split("/")
        assert all(part in source for part in parts), f"{relative} no longer referenced"
        assert (RESOURCES / relative).is_file(), f"{relative} referenced but missing"


def test_no_site_specific_absolute_paths():
    """Scripts must not hard-code the Nexus working directory."""
    offenders = []
    for path in [*SCRIPTS_DIR.rglob("*.py"), *TOOLS_DIR.rglob("*.py"), *TOOLS_DIR.rglob("*.sh")]:
        if "/scratch/Projects/" in path.read_text():
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"site-specific absolute paths in: {offenders}"


def test_generated_workspace_not_committed():
    """Automated_Setup.itksnap is written per run and must stay out of the tree."""
    assert not (REPO_ROOT / "workflow" / "workspace" / "Automated_Setup.itksnap").exists()
    assert (REPO_ROOT / "workflow" / "workspace" / "NEXUS_NM_Template_ITKsnap.xml").is_file()


def test_no_committed_bytecode():
    """
    No compiled bytecode is tracked by Git.

    Checked against the index rather than the working tree, since running these
    tests creates tests/__pycache__ itself. .gitignore keeps it untracked.
    """
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    offenders = [name for name in tracked
                 if "__pycache__" in name or name.endswith((".pyc", ".pyo"))]
    assert not offenders, f"compiled bytecode tracked by Git: {offenders}"


def test_single_requirements_file():
    """One pinned environment, at the repository root."""
    found = [str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("requirements.txt")
             if ".git" not in p.parts]
    assert found == ["requirements.txt"], f"expected one requirements.txt, found {found}"
