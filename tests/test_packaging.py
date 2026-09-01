"""Every non-.py runtime asset must survive the wheel build *without git*.

``imageharbor/dashboard/index.html`` is loaded at runtime by
``dashboard/server.py`` via ``Path(__file__).parent / "index.html"``. If it is
not installed, the module still imports, ``/api/stats`` still serves 200, and
only ``GET /`` fails -- with "dashboard page unavailable". That is exactly how
it shipped on the first real deploy to hpz440 (2026-08-19).

The mechanism is subtle enough to be worth stating, because the obvious test
does not catch it. setuptools' implicit inclusion of non-.py package files
depends on **revision-control metadata**: building from this checkout, where
``.git`` exists, the HTML lands in the wheel whether or not it is declared.
The Docker image builds from a context where ``.dockerignore`` excludes
``.git``, and there the same source tree produces a wheel with no HTML in it.
So the asset's presence depended on an accident of the build environment, and
a test that builds in place is green under both conditions.

These tests therefore build from a **git-less copy** of the tree -- mirroring
the Docker build context -- and assert against the resulting wheel. Two
properties are load-bearing and must not be "simplified" away:

1. the build runs against a copy with no ``.git`` (that is the failing
   condition), and
2. the assertion is against built wheel contents, not the source checkout
   (the checkout always has the file; that is why this bug reached
   production).

The real fix is the explicit ``[tool.setuptools.package-data]`` entry in
pyproject.toml, which makes inclusion independent of git metadata.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Runtime assets that must be inside the wheel. Add any new non-.py file the
# package reads at runtime here AND to [tool.setuptools.package-data].
REQUIRED_WHEEL_MEMBERS = ("imageharbor/dashboard/index.html",)

# What the Docker build context actually carries (see Dockerfile's COPY lines).
_BUILD_INPUTS = ("pyproject.toml", "README.md")


@pytest.fixture(scope="module")
def wheel_built_without_git(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a wheel from a git-less copy, mirroring the Docker build context.

    Uses ``uv build`` -- the project mandates uv (see CLAUDE.md), so it is
    present wherever this suite runs. Deliberately NOT written to skip when a
    builder is missing: a skipped test is indistinguishable from a passing one
    in a summary line, and this test exists precisely because a silent absence
    already shipped once.
    """
    src = tmp_path_factory.mktemp("nogit_src")
    for name in _BUILD_INPUTS:
        shutil.copy2(PROJECT_ROOT / name, src / name)
    shutil.copytree(
        PROJECT_ROOT / "imageharbor",
        src / "imageharbor",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    assert not (src / ".git").exists(), "the copy must have no git metadata"

    out = tmp_path_factory.mktemp("nogit_wheel")
    cmd = ["uv", "build", "--wheel", "--out-dir", str(out), str(src)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except FileNotFoundError:
        # Fail loudly and legibly rather than surfacing a bare traceback --
        # and still fail, never skip (see the fixture docstring).
        pytest.fail(
            "`uv` was not found on PATH, so this test could not build a wheel "
            "and therefore could not check that runtime assets survive "
            "packaging. Install uv (the project mandates it -- see CLAUDE.md) "
            "rather than skipping: the bug this guards shipped as a silent "
            "absence, and a silent absence in the test is the same failure."
        )
    except subprocess.TimeoutExpired:
        pytest.fail("`uv build` timed out after 600s: " + " ".join(cmd))
    if proc.returncode != 0:
        pytest.fail("`uv build` failed:\n" + proc.stdout + "\n" + proc.stderr)
    wheels = list(out.glob("*.whl"))
    assert wheels, "uv build produced no wheel: " + proc.stdout
    return wheels[0]


@pytest.mark.parametrize("member", REQUIRED_WHEEL_MEMBERS)
def test_runtime_asset_survives_a_gitless_wheel_build(
    wheel_built_without_git: Path, member: str
) -> None:
    with zipfile.ZipFile(wheel_built_without_git) as zf:
        names = set(zf.namelist())
    non_py = sorted(n for n in names if not n.endswith(".py"))
    assert member in names, (
        f"{member} is missing from a wheel built without git metadata -- this "
        f"is what the Docker image installs, and the dashboard page 500s "
        f"without it. Declare the file under [tool.setuptools.package-data] "
        f"in pyproject.toml; do not rely on setuptools' implicit inclusion, "
        f"which needs .git and is excluded by .dockerignore. "
        f"Wheel non-.py members: {non_py}"
    )


@pytest.mark.parametrize("member", REQUIRED_WHEEL_MEMBERS)
def test_the_wheels_runtime_asset_is_actually_servable(
    wheel_built_without_git: Path, member: str
) -> None:
    """Presence alone is not enough: a truncated or empty entry still 500s.

    Asserted structurally rather than by byte count -- a size floor would be
    an arbitrary number that fails the day the page is legitimately made
    smaller. What actually matters is that the bytes parse as the HTML
    document the server hands to a browser.
    """
    with zipfile.ZipFile(wheel_built_without_git) as zf:
        data = zf.read(member)
    text = data.decode("utf-8").lower()
    assert text.strip(), f"{member} is empty in the wheel"
    assert "<html" in text, f"{member} in the wheel has no <html> element"
    assert "</html>" in text, (
        f"{member} in the wheel is truncated -- no closing </html>"
    )


# -- the documented dev install must be able to run the suite ---------------
#
# `uv sync --extra dev` is what CLAUDE.md tells a contributor to run, and it
# resolves only [project.dependencies] + the `dev` extra. When the faces work
# landed, fifteen test modules began importing numpy at module scope while
# numpy was declared only under the `faces` extra -- so the documented install
# could not collect the suite at all (16 collection errors, two of them in
# non-faces dashboard tests). Nothing caught it, because every environment
# that ran the tests had been synced with `--all-extras`.
#
# Asserting "numpy is in dev" would only pin the one bug already fixed. This
# derives the requirement instead: whatever the tests import at module scope
# must be installable from the documented command. A test that needs a heavy
# runtime should reach it through `pytest.importorskip`, which is a call and
# not an import, and so is correctly invisible here.

STDLIB = frozenset(sys.stdlib_module_names)

# Import name -> distribution name, for the cases where they differ.
DIST_OF_MODULE = {"PIL": "pillow"}


def _module_scope_imports(tree: ast.Module) -> set[str]:
    """Top-level module names imported unconditionally by `tree`.

    Only statements at module scope count: an import inside a function or a
    `try`/`except ImportError` is a deliberate soft dependency, and the point
    here is what the interpreter must resolve at collection time.
    """
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has no module to install.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    """Distributions `uv sync --extra dev` installs, PEP 503-normalized.

    Read from pyproject.toml, which is the file a contributor edits. tomllib
    is 3.11+ and this project supports 3.10, so on 3.10 the same declarations
    are read back from the installed distribution's metadata instead of
    skipping -- a guard that silently does not run is the failure mode this
    whole module exists to prevent.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        from importlib.metadata import requires

        requirements = [
            r.split(";")[0].strip()
            for r in (requires("imageharbor") or [])
            if ";" not in r or 'extra == "dev"' in r
        ]
    else:
        project = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        requirements = list(project["dependencies"])
        requirements += project["optional-dependencies"]["dev"]

    return {
        _canonical(re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0]) for r in requirements
    }


def test_the_documented_dev_install_covers_every_module_scope_test_import():
    declared = _declared_distributions()
    local = {"imageharbor", "tests", "conftest"}

    missing: dict[str, list[str]] = {}
    for path in sorted((PROJECT_ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _module_scope_imports(tree):
            if module in STDLIB or module in local:
                continue
            dist = DIST_OF_MODULE.get(module, module)
            if _canonical(dist) not in declared:
                missing.setdefault(module, []).append(
                    str(path.relative_to(PROJECT_ROOT))
                )

    assert not missing, (
        "these modules are imported at module scope by the test suite but are "
        "not installed by the documented `uv sync --extra dev`, so collection "
        "fails for anyone who follows CLAUDE.md: "
        + "; ".join(f"{m} ({', '.join(files)})" for m, files in sorted(missing.items()))
        + ". Either add the distribution to [project.optional-dependencies].dev "
        "in pyproject.toml, or reach it through pytest.importorskip so the "
        "test skips instead of erroring."
    )
