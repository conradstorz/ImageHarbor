"""Shared fixtures. Policy: a fixture lives here only when multiple test
modules used byte-identical copies (the quality review counted `catalog`
x14). A module needing DIFFERENT behavior keeps a local fixture -- pytest's
shadowing makes that safe -- but name it distinctly if it differs
meaningfully, so a reader never has to check which one is in scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog


def _make_jpeg(
    path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"
) -> Path:
    """Write a minimal pseudo-JPEG file. Private to `source_dir` below --
    every test module that needs `_make_jpeg` for its own test bodies (not
    just this fixture) keeps its own identical copy; only the fixture moved
    here, not the general-purpose helper."""
    path.write_bytes(content)
    return path


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach_photo.jpg")
    _make_jpeg(src / "mountain_view.jpg", b"\xff\xd8\xff\xe0" + b"\x01" * 16 + b"\xff\xd9")
    return src


@pytest.fixture()
def dirs(tmp_path: Path):
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    dest.mkdir()
    return archives, dest


@pytest.fixture()
def store(tmp_path: Path):
    # Imported here, not at module level (R4 review Minor #4): a missing
    # numpy (faces' only hard runtime dependency) should only take down
    # collection of the faces/dashboard modules that actually use this
    # fixture, not collection of the entire suite via this shared conftest.
    from imageharbor.faces.store import FaceStore

    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()
