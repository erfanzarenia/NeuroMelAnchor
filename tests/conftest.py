"""
Shared fixtures and helpers for the NeuroMelAnchor checks.

The pipeline is a single script that parses its arguments at import time, so it
cannot simply be imported for testing. Instead these helpers lift individual
functions out of it with `ast` and execute just those definitions. That works
because the pipeline's worker functions are written to be self-contained (they
import what they need inside the body) -- which is exactly what Nipype's
`Function` interface requires of them, so the technique tests the real contract.

Only numpy, nibabel, scipy and matplotlib are needed; Nipype, ANTs, SPM and
ITK-SNAP are not.
"""
import ast
import builtins
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "workflow" / "scripts"
TOOLS_DIR = REPO_ROOT / "workflow" / "tools"
PIPELINE = SCRIPTS_DIR / "neuromelanchor_pipeline.py"
RESOURCES = REPO_ROOT / "resources" / "tpl-MNI152NLin2009cAsym"


def pipeline_source():
    """Source text of the canonical pipeline script."""
    return PIPELINE.read_text()


def nipype_function_node_names(source=None):
    """
    Names passed as `function=` to a Nipype `Function` interface.

    These are the functions Nipype re-serialises by source, so they must not rely
    on anything defined outside their own body.
    """
    return set(re.findall(r"function=(\w+)", source if source is not None else pipeline_source()))


def load_functions(*names, source=None):
    """
    Execute the named top-level functions from the pipeline in a fresh namespace.

    Raises KeyError if a name is not a top-level function, which keeps the tests
    honest if a function is renamed or removed.
    """
    source = pipeline_source() if source is None else source
    tree = ast.parse(source)
    wanted = {node.name: node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in names}

    missing = set(names) - set(wanted)
    if missing:
        raise KeyError(f"not top-level functions in {PIPELINE.name}: {sorted(missing)}")

    namespace = {}
    exec(compile(ast.Module(body=[wanted[n] for n in names], type_ignores=[]),
                 str(PIPELINE), "exec"), namespace)
    return {name: namespace[name] for name in names}


def load_objects(*names, source=None):
    """
    Like `load_functions`, but also accepts module-level assignments.

    Needed for helpers that read a module constant -- `resolve_input_patterns`
    reads the cohort pattern table -- which are not Nipype `Function` nodes and so
    are not required to be self-contained.
    """
    source = pipeline_source() if source is None else source
    tree = ast.parse(source)

    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if set(targets) & set(names):
                selected.append(node)
                found.update(targets)

    missing = set(names) - found
    if missing:
        raise KeyError(f"not top-level names in {PIPELINE.name}: {sorted(missing)}")

    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PIPELINE), "exec"), namespace)
    return {name: namespace[name] for name in names}


def free_variables(func_node):
    """
    Names a function reads without binding them itself.

    Anything returned here would be a NameError once Nipype re-executes the
    function in a bare worker process.
    """
    bound = set()
    args = func_node.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        bound.add(arg.arg)
    if args.vararg:
        bound.add(args.vararg.arg)
    if args.kwarg:
        bound.add(args.kwarg.arg)

    used = set()
    for node in ast.walk(func_node):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else used).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Nested helpers: bind the name and its parameters. This flattens all
            # nested scopes into one, which is deliberately permissive -- it can
            # miss a free name that happens to share a nested parameter's name,
            # but it never reports a false positive, which is what matters here.
            bound.add(node.name)
            nested = node.args
            for arg in [*nested.posonlyargs, *nested.args, *nested.kwonlyargs]:
                bound.add(arg.arg)
            if nested.vararg:
                bound.add(nested.vararg.arg)
            if nested.kwarg:
                bound.add(nested.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                bound.add(arg.arg)
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)

    return used - bound - set(dir(builtins))


@pytest.fixture(scope="session")
def pipeline_text():
    return pipeline_source()


@pytest.fixture
def nifti_factory(tmp_path):
    """Write a small NIfTI volume to tmp_path and return its path."""
    import numpy as np
    import nibabel as nib

    def _make(name, data, voxel_size=(1.0, 1.0, 1.0), dtype=np.float32):
        affine = np.diag([*voxel_size, 1.0])
        path = tmp_path / name
        nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), affine), path)
        return str(path)

    return _make
