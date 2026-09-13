"""The version has exactly one source: git tags, via setuptools-scm at
build/install time, surfaced through importlib.metadata. A literal version
string anywhere in the tree is the bug this file exists to prevent -- the
0.1.0 era had two (pyproject.toml and __init__.py) with nothing binding
them."""

import re
from pathlib import Path

import imageharbor


def test_version_is_a_real_pep440_string():
    assert re.match(r"^\d+\.\d+", imageharbor.__version__) or \
        imageharbor.__version__.startswith("0.0.0")


def test_no_hardcoded_version_literals_remain():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'version = "0.1.0"' not in text
    assert 'dynamic = ["version"]' in text
    init = Path(imageharbor.__file__).read_text(encoding="utf-8")
    assert '__version__ = "0' not in init
