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


def test_pipeline_uses_taxonomy_codes(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = pipeline.run()
    assert stats.copied == 2 and stats.errors == 0
    # beach_photo.jpg -> 300-places/330-beach ; mountain_view.jpg -> 340-mountains
    organized = [p.as_posix() for p in organized_dir.rglob("*.jpg")]
    assert any("300-places/330-beach/330-" in p for p in organized)
    assert any("300-places/340-mountains/340-" in p for p in organized)


def test_pipeline_mints_new_category(
    organized_dir: Path, catalog: Catalog, tmp_path: Path
) -> None:
    # A custom classifier that proposes a brand-new label under events.
    from imageharbor.ai_classifier import AIClassifier, PhotoClassification

    class NewCatClassifier(AIClassifier):
        def classify(self, image_path, exif_data, taxonomy_snapshot):
            return PhotoClassification(top_parent="500", label="holidays", descriptor="xmas")

    src = tmp_path / "src2"
    src.mkdir()
    _make_jpeg(src / "a.jpg")
    pipeline = Pipeline(src, organized_dir, catalog, classifier=NewCatClassifier())
    pipeline.run()
    assert any(
        "500-events/540-holidays/540-" in p.as_posix()
        for p in organized_dir.rglob("*.jpg")
    )


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


# ---------------------------------------------------------------------------
# Failure / error paths
# ---------------------------------------------------------------------------


def test_pipeline_integrity_failure_removes_copy(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the post-copy verification to fail: the pipeline must unlink the
    # copied file and record an error (pipeline.py lines 199-203).
    monkeypatch.setattr("imageharbor.pipeline.verify_file", lambda *a, **k: False)

    single = source_dir / "beach_photo.jpg"
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    result = pipeline.process_file(single)

    assert result.status == "error"
    assert "Integrity check failed" in result.error
    # The organized copy must have been removed, not left behind.
    assert not list(organized_dir.rglob("*.jpg"))
    # Nothing was catalogued for the failed image.
    assert catalog.count() == 0


def test_pipeline_integrity_failure_counted_in_stats(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("imageharbor.pipeline.verify_file", lambda *a, **k: False)

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = pipeline.run()

    assert stats.errors >= 1
    assert stats.copied == 0
    assert not list(organized_dir.rglob("*.jpg"))


def test_pipeline_error_path_captures_message_and_continues(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Make _do_process blow up at step 1 (hashing). The run must not crash; each
    # failure is captured as an "error" result (pipeline.py lines 137-144).
    def _boom(*_a, **_k):
        raise ValueError("hash exploded")

    monkeypatch.setattr("imageharbor.pipeline.compute_sha256_b64url", _boom)

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = pipeline.run()

    assert stats.errors == 2  # both source images fail
    assert stats.copied == 0
    for result in stats.results:
        assert result.status == "error"
        assert "hash exploded" in result.error


def test_pipeline_error_path_logs_error(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(*_a, **_k):
        raise ValueError("hash exploded")

    monkeypatch.setattr("imageharbor.pipeline.compute_sha256_b64url", _boom)

    single = source_dir / "beach_photo.jpg"
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    with caplog.at_level("ERROR"):
        result = pipeline.process_file(single)

    assert result.status == "error"
    assert any("Error" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Duplicate copying
# ---------------------------------------------------------------------------


def test_pipeline_duplicates_copied_to_dir(
    source_dir: Path, organized_dir: Path, catalog: Catalog, tmp_path: Path
) -> None:
    # First run copies everything into the catalog.
    Pipeline(source_dir, organized_dir, catalog).run()

    # Second run, this time with duplicates_dir set: every image is now a
    # duplicate and must be copied into duplicates_dir (pipeline.py lines
    # 156-157, 263-267) and recorded in the catalog history (mark_duplicate).
    dup_dir = tmp_path / "dups"
    dup_pipeline = Pipeline(
        source_dir, organized_dir, catalog, duplicates_dir=dup_dir
    )
    stats = dup_pipeline.run()

    assert stats.duplicates == 2
    copied = sorted(p.name for p in dup_dir.iterdir())
    assert len(copied) == 2

    # Each duplicate filename is prefixed with the FULL digest (never an 8-char
    # prefix) so that distinct-content collisions cannot silently overwrite.
    from imageharbor.hashing import compute_sha256_b64url

    for src in source_dir.rglob("*.jpg"):
        digest = compute_sha256_b64url(src)
        expected = f"{digest}_{src.name}"
        assert (dup_dir / expected).exists()

        # The catalog recorded a duplicate_detected event for this digest.
        row = catalog.get_by_sha256(digest)
        assert row is not None
        assert "duplicate_detected" in row["processing_history"]


# ---------------------------------------------------------------------------
# Dry-run intra-run deduplication
# ---------------------------------------------------------------------------


def test_pipeline_dry_run_intra_run_dedup(
    organized_dir: Path, catalog: Catalog, tmp_path: Path
) -> None:
    # Two source files with IDENTICAL content in a single dry run. Because the
    # catalog is never written during dry_run, an in-memory seen-set must dedup
    # them: one "copied", one "duplicate" -- matching what a real run reports.
    src = tmp_path / "src_identical"
    src.mkdir()
    _make_jpeg(src / "a.jpg")
    _make_jpeg(src / "b.jpg")  # same default content as a.jpg

    pipeline = Pipeline(src, organized_dir, catalog, dry_run=True)
    stats = pipeline.run()

    assert stats.copied == 1
    assert stats.duplicates == 1
    # Still nothing written and the catalog untouched.
    assert not list(organized_dir.rglob("*.jpg"))
    assert catalog.count() == 0

    # A real run over the same input reports the same copied/duplicate counts.
    real_catalog = Catalog(tmp_path / "real.db")
    try:
        real_stats = Pipeline(src, organized_dir, real_catalog).run()
        assert real_stats.copied == stats.copied
        assert real_stats.duplicates == stats.duplicates
    finally:
        real_catalog.close()


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------


def test_pipeline_resume_skips_copy_when_dest_verifies(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First run organizes everything (this is our simulated pre-existing state).
    Pipeline(source_dir, organized_dir, catalog).run()
    organized = list(organized_dir.rglob("*.jpg"))
    assert organized

    # Simulate a crash-after-copy-before-catalog resume: use a FRESH catalog so
    # the images are unknown, but the verified organized files already exist.
    fresh_catalog = Catalog(tmp_path / "fresh.db")

    # copy2 must NOT be called for a destination that already exists and verifies.
    def _no_copy(*_a, **_k):
        raise AssertionError("shutil.copy2 must not be called on a verified resume")

    monkeypatch.setattr("imageharbor.pipeline.shutil.copy2", _no_copy)

    try:
        stats = Pipeline(source_dir, organized_dir, fresh_catalog).run()
        assert stats.errors == 0
        assert stats.copied == 2
        # The pre-existing verified files were re-catalogued, not re-copied.
        assert fresh_catalog.count() == 2
        for result in stats.results:
            assert result.status == "copied"
    finally:
        fresh_catalog.close()


# ---------------------------------------------------------------------------
# Sidecar-failure isolation
# ---------------------------------------------------------------------------


def test_pipeline_sidecar_failure_does_not_fail_image(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A failing sidecar write must not turn an already copied+verified+catalogued
    # image into an "error": status stays "copied", no errors, warning logged.
    def _boom(*_a, **_k):
        raise OSError("sidecar disk full")

    monkeypatch.setattr("imageharbor.pipeline.write_sidecar", _boom)

    single = source_dir / "beach_photo.jpg"
    pipeline = Pipeline(source_dir, organized_dir, catalog, write_sidecars=True)
    with caplog.at_level("WARNING"):
        result = pipeline.process_file(single)

    assert result.status == "copied"
    assert result.organized_path is not None
    assert result.organized_path.exists()
    # Image is catalogued despite the sidecar failure.
    assert catalog.count() == 1
    # And a warning was logged rather than the error propagating.
    assert any("sidecar" in rec.message.lower() for rec in caplog.records)


def test_pipeline_sidecar_failure_run_counts_no_errors(
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a, **_k):
        raise OSError("sidecar disk full")

    monkeypatch.setattr("imageharbor.pipeline.write_sidecar", _boom)

    stats = Pipeline(source_dir, organized_dir, catalog, write_sidecars=True).run()

    assert stats.errors == 0
    assert stats.copied == 2
