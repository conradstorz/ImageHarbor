"""Integration tests for the processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.hashing import verify_pcs_file
from imageharbor.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9") -> Path:
    """Write a minimal pseudo-JPEG file."""
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach_photo.jpg")
    _make_jpeg(src / "mountain_view.jpg", b"\xff\xd8\xff\xe0" + b"\x01" * 16 + b"\xff\xd9")
    return src


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_copies_files(source_dir: Path, organized_dir: Path, catalog: Catalog) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = pipeline.run()

    assert stats.errors == 0
    assert stats.copied == 2
    assert stats.duplicates == 0


def test_pipeline_creates_organized_structure(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    pipeline.run()

    # Files must exist inside the organized directory
    all_organized = list(organized_dir.rglob("*.jpg"))
    assert len(all_organized) == 2


def test_pipeline_filenames_contain_digest(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    pipeline.run()

    for organized_file in organized_dir.rglob("*.jpg"):
        # Digest embedded in filename should match the file's actual SHA-256
        assert verify_pcs_file(organized_file), (
            f"Integrity check failed for {organized_file.name}"
        )


def test_pipeline_duplicate_detection(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    # First run – copy everything
    stats1 = pipeline.run()
    assert stats1.copied == 2

    # Second run – everything is a duplicate
    stats2 = pipeline.run()
    assert stats2.duplicates == 2
    assert stats2.copied == 0


def test_pipeline_catalog_populated(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    pipeline.run()

    assert catalog.count() == 2


def test_pipeline_dry_run_writes_nothing(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog, dry_run=True)
    stats = pipeline.run()

    assert stats.copied == 2
    # No real files written
    assert not list(organized_dir.rglob("*.jpg"))
    assert catalog.count() == 0


def test_pipeline_sidecar_written(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog, write_sidecars=True)
    pipeline.run()

    sidecars = list(organized_dir.rglob("*.json"))
    assert len(sidecars) == 2
    for sidecar in sidecars:
        data = json.loads(sidecar.read_text())
        assert "sha256_b64url" in data
        assert len(data["sha256_b64url"]) == 43


def test_pipeline_originals_not_modified(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    # Record hashes before pipeline
    originals = list(source_dir.rglob("*.jpg"))
    before = {p: p.read_bytes() for p in originals}

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    pipeline.run()

    # Verify originals unchanged
    for p in originals:
        assert p.read_bytes() == before[p], f"Original was modified: {p}"


def test_pipeline_process_single_file(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    single = source_dir / "beach_photo.jpg"
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    result = pipeline.process_file(single)

    assert result.status == "copied"
    assert result.organized_path is not None
    assert result.organized_path.exists()
