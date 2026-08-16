"""Integration tests for the processing pipeline (the facts pass)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from imageharbor import tiers
from imageharbor.catalog import Catalog
from imageharbor.hashing import verify_pcs_file
from imageharbor.pipeline import ExternalEvidence, Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9") -> Path:
    """Write a minimal pseudo-JPEG file."""
    path.write_bytes(content)
    return path


def _make_image(path: Path, content: bytes = b"fake-image-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
# General hashing / dedup / copy / verify behavior
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
    # No destination path is computed in dry-run.
    for result in stats.results:
        assert result.organized_path is None


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
    # copied file and record an error.
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
    # failure is captured as an "error" result.
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
    # duplicate and must be copied into duplicates_dir and recorded in the
    # catalog history (mark_duplicate).
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

    monkeypatch.setattr("imageharbor.pipeline.merge_sidecar", _boom)

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

    monkeypatch.setattr("imageharbor.pipeline.merge_sidecar", _boom)

    stats = Pipeline(source_dir, organized_dir, catalog, write_sidecars=True).run()

    assert stats.errors == 0
    assert stats.copied == 2


# ---------------------------------------------------------------------------
# Facts-pass specific behavior (Task 9)
# ---------------------------------------------------------------------------


def test_facts_pass_makes_no_ai_call() -> None:
    """`imageharbor.pipeline` must not import a classifier at all.

    A previous version of this test patched `StubClassifier.describe` and
    asserted it was never called -- but `Pipeline` never constructs a
    `StubClassifier` (or any classifier), so the patch was never in a
    position to fire and the test was near-vacuous: it would pass even if
    `pipeline.py` imported and called the AI directly through some other
    path. Assert the actual claim instead, as a real import-boundary check:
    importing `imageharbor.pipeline` in a fresh subprocess must never pull
    `imageharbor.ai_classifier` into `sys.modules`. A subprocess is required
    because within the same test process `ai_classifier` is typically
    already imported by other tests/fixtures, which would make an in-process
    check pass or fail for the wrong reason.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import imageharbor.pipeline, sys\n"
            "assert 'imageharbor.ai_classifier' not in sys.modules, (\n"
            "    'imageharbor.pipeline import pulled in imageharbor.ai_classifier'\n"
            ")\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-boundary check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_human_named_undated_file_lands_in_undated(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "Emma's graduation.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        result = stats.results[0]
        assert result.organized_path.parent == dest / "Undated"
        assert result.organized_path.name.startswith("emmas-graduation_")
        assert cat.tiers_for(result.sha256_b64url) == (
            tiers.DATE_NONE,
            tiers.DESC_HUMAN_FILENAME,
        )


def test_camera_named_file_gets_no_descriptor(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "IMG_1234.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        result = stats.results[0]
        assert result.organized_path.stem == result.sha256_b64url
        assert cat.tiers_for(result.sha256_b64url) == (tiers.DATE_NONE, tiers.DESC_NONE)


def test_filename_date_places_the_file(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "IMG_20190704_123456.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        result = stats.results[0]
        assert result.organized_path.parent == dest / "2019" / "2019-07"
        assert result.organized_path.name.startswith("2019-07-04_")


def test_duplicates_record_back_pointers_and_copy_once(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "a" / "IMG_1234.jpg", b"same")
    _make_image(src / "b" / "IMG_5678.jpg", b"same")
    _make_image(src / "c" / "Emma's graduation.jpg", b"same")

    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        assert stats.copied == 1
        assert stats.duplicates == 2
        digest = stats.results[0].sha256_b64url
        assert len(cat.sources_for(digest)) == 3


def test_rerunning_the_facts_pass_changes_nothing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "IMG_20190704_123456.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        first = Pipeline(src, dest, cat).run()
        paths_after_first = sorted(p.name for p in dest.rglob("*.jpg"))
        second = Pipeline(src, dest, cat).run()
        paths_after_second = sorted(p.name for p in dest.rglob("*.jpg"))

    assert first.copied == 1
    assert second.copied == 0
    assert second.duplicates == 1
    assert paths_after_first == paths_after_second


def test_sidecar_records_facts_and_sources(tmp_path):
    from imageharbor.sidecar import read_sidecar

    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "Emma's graduation.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat, write_sidecars=True).run()
    data = read_sidecar(stats.results[0].organized_path)
    assert data["descriptor"]["tier"] == tiers.DESC_HUMAN_FILENAME
    assert data["date"]["source"] == "none"
    assert len(data["sources"]) == 1
    assert "classification" not in data


def test_consume_source_moves_instead_of_copying(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = _make_jpeg(staging / "beach.jpg")

    pipeline = Pipeline(staging, organized_dir, catalog, consume_source=True)
    result = pipeline.process_file(staged)

    assert result.status == "copied"
    assert not staged.exists()                      # consumed
    assert result.organized_path.exists()
    assert verify_pcs_file(result.organized_path)   # verified AFTER the move


def test_consume_source_defaults_to_copying(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    original = source_dir / "beach_photo.jpg"
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    result = pipeline.process_file(original)

    assert original.exists()                        # untouched
    assert result.organized_path.exists()


def test_consume_source_falls_back_to_copy_across_filesystems(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, monkeypatch
) -> None:
    """EXDEV must degrade to copy-then-delete, not fail every member of the run.

    No filesystem available to the test suite produces a genuine cross-device
    rename, so the error is injected. Without this the fallback branch -- and
    specifically its post-verification unlink -- is unreachable code that no
    mutation can be caught by.
    """
    import os as _os

    staged = _make_jpeg(tmp_path / "beach.jpg")

    def _exdev(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(_os, "replace", _exdev)

    result = Pipeline(
        tmp_path, organized_dir, catalog, consume_source=True
    ).process_file(staged)

    assert result.status == "copied"
    assert result.organized_path.exists()
    assert verify_pcs_file(result.organized_path)
    # The fallback must still consume the source, and only after verifying.
    assert not staged.exists()


def test_source_label_is_what_gets_recorded(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The staging path is disposable; the logical source is the archive member."""
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = _make_jpeg(staging / "beach.jpg")
    label = "/nas/takeout/t1.zip!Takeout/AlbumArchive/a/beach.jpg"

    pipeline = Pipeline(staging, organized_dir, catalog)
    result = pipeline.process_file(staged, source_label=label)

    row = catalog.get_by_sha256(result.sha256_b64url)
    assert row["original_path"] == label
    assert [r["source_path"] for r in catalog.sources_for(result.sha256_b64url)] == [label]


def test_evidence_date_places_the_file(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    staged = _make_jpeg(tmp_path / "IMG_1234.jpg")
    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    result = pipeline.process_file(
        staged, evidence=ExternalEvidence(date=datetime(2015, 3, 9, 12, 56, 32))
    )

    assert result.organized_path.parent == organized_dir / "2015" / "2015-03"
    row = catalog.get_by_sha256(result.sha256_b64url)
    assert row["date_tier"] == tiers.DATE_EXTERNAL_SIDECAR
    assert row["date_source"] == "external_sidecar"


def test_evidence_original_name_names_the_file(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    staged = _make_jpeg(tmp_path / "truncated-stem.jpg")
    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    result = pipeline.process_file(
        staged, evidence=ExternalEvidence(original_name="emma birthday party.jpg")
    )

    assert "emma-birthday-party" in result.organized_path.name
    row = catalog.get_by_sha256(result.sha256_b64url)
    assert row["descriptor_tier"] == tiers.DESC_HUMAN_FILENAME


def test_omitting_the_new_arguments_reproduces_todays_behavior(
    tmp_path: Path, organized_dir: Path
) -> None:
    """The regression surface: `process`/`watch` must be byte-for-byte unchanged.

    Runs identical bytes through two independent libraries -- one calling
    `process_file(path)` exactly as `watcher.py` does, one passing the new
    arguments explicitly as None -- and compares every field that decides
    placement, naming, and identity.
    """
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "beach photo.jpg")

    def _run(root: str, **kwargs) -> tuple:
        dest = organized_dir / root
        dest.mkdir()
        with Catalog(tmp_path / f"{root}.db") as cat:
            result = Pipeline(src, dest, cat).process_file(src / "beach photo.jpg", **kwargs)
            row = cat.get_by_sha256(result.sha256_b64url)
            return (
                result.status,
                result.organized_path.relative_to(dest).as_posix(),
                row["original_path"],
                row["date_tier"], row["date_source"], row["date_value"],
                row["descriptor_tier"], row["descriptor_source"], row["descriptor_value"],
                [r["source_path"] for r in cat.sources_for(result.sha256_b64url)],
            )

    omitted = _run("a")
    explicit_none = _run("b", source_label=None, evidence=None)
    assert omitted == explicit_none


def test_evidence_none_on_the_duplicate_branch_is_safe(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The duplicate branch dereferences `evidence`; None must not reach it raw."""
    staged = _make_jpeg(tmp_path / "beach photo.jpg")
    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    first = pipeline.process_file(staged)
    second = pipeline.process_file(staged, evidence=None)

    assert second.status == "duplicate"
    row = catalog.get_by_sha256(first.sha256_b64url)
    assert row["organized_path"] == str(first.organized_path)


def test_result_source_path_is_the_real_path_not_the_label(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """`source_label` replaces the recorded IDENTITY, never the path read from."""
    staged = _make_jpeg(tmp_path / "beach.jpg")
    label = "/nas/takeout/t1.zip!Takeout/AlbumArchive/a/beach.jpg"

    result = Pipeline(tmp_path, organized_dir, catalog).process_file(
        staged, source_label=label
    )

    assert result.source_path == staged
    assert str(result.source_path) != label


def test_consume_source_cleans_up_when_the_destination_already_exists(
    tmp_path: Path, organized_dir: Path
) -> None:
    """The already-present-and-verified branch must still consume the source.

    Reaching that branch requires a destination that exists and verifies while
    the digest is UNKNOWN to the catalog -- a rebuilt catalog over a surviving
    tree, or a crash between the copy and the catalog upsert. A known digest
    would take the duplicate branch and return long before the copy step, so a
    second run against the SAME catalog would not exercise this at all.
    """
    first_staging = tmp_path / "s1"
    first_staging.mkdir()
    staged_once = _make_jpeg(first_staging / "beach.jpg")
    with Catalog(tmp_path / "first.db") as cat:
        first = Pipeline(
            first_staging, organized_dir, cat, consume_source=True
        ).process_file(staged_once)
    assert first.organized_path.exists()

    # Fresh catalog, surviving tree: the digest is unknown but the file is there.
    second_staging = tmp_path / "s2"
    second_staging.mkdir()
    staged_twice = _make_jpeg(second_staging / "beach.jpg")
    with Catalog(tmp_path / "second.db") as cat:
        assert not cat.is_known(first.sha256_b64url)
        second = Pipeline(
            second_staging, organized_dir, cat, consume_source=True
        ).process_file(staged_twice)

    assert second.status == "copied"
    assert not staged_twice.exists()      # consumed, not leaked
    assert first.organized_path.exists()  # destination untouched
