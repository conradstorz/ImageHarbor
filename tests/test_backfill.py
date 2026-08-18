"""Tests for `imageharbor.backfill.backfill_sidecars`.

Covers rebuilding sidecars for a library organized before sidecars were the
default (`process --no-sidecar`), including the honest limit that Google
Takeout provenance is not recoverable this way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imageharbor.backfill import BackfillStats, backfill_sidecars
from imageharbor.catalog import Catalog
from imageharbor.pipeline import Pipeline
from imageharbor.sidecar import merge_sidecar, read_sidecar, sidecar_path_for


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_jpeg(path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9") -> Path:
    path.write_bytes(content)
    return path


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


def _organize_without_sidecars(source_dir: Path, organized_dir: Path, catalog: Catalog) -> None:
    """Populate organized_dir + catalog exactly as a pre-Task-7 library would:
    facts pass only, no sidecars -- write_sidecars defaults to False on
    Pipeline itself (only the CLI flag default changed in Task 7)."""
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = pipeline.run()
    assert stats.errors == 0


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_backfill_writes_sidecars_for_every_resolvable_row(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    _organize_without_sidecars(source_dir, organized_dir, catalog)
    assert not list(organized_dir.rglob("*.json")), "fixture must start with no sidecars"

    stats = backfill_sidecars(organized_dir, catalog)

    assert stats.cataloged == 2
    assert stats.written == 2
    assert stats.unchanged == 0
    assert stats.failed == 0

    sidecars = list(organized_dir.rglob("*.json"))
    assert len(sidecars) == 2
    for path in sidecars:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["identity"]["sha256_b64url"]) == 43
        assert data["sources"], "no sources recorded"
        assert "folder" in data["sources"][0]
        assert data["date"]["tier"] is not None
        assert data["descriptor"]["tier"] is not None
        assert "exif" in data


def test_backfill_provenance_is_empty(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The honest limit: Google Takeout metadata cannot be recovered by
    backfill, because the original archive documents were never stored for
    a library that was never ingested through takeout. Pinned explicitly so
    a future reader finds it here rather than discovering it by accident."""
    _organize_without_sidecars(source_dir, organized_dir, catalog)

    backfill_sidecars(organized_dir, catalog)

    for path in organized_dir.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("provenance", []) == []


def test_backfill_merges_into_an_existing_thin_sidecar(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    _organize_without_sidecars(source_dir, organized_dir, catalog)
    organized = next(organized_dir.rglob("*.jpg"))

    # A thin, hand-written sidecar predating full sidecar support -- only a
    # note, nothing else. Backfill must MERGE into it (preserving the note),
    # not skip it because a sidecar already exists.
    merge_sidecar(organized, {"note": "kept from before backfill existed"})

    stats = backfill_sidecars(organized_dir, catalog)

    # The row with the thin sidecar must be reported "written" (it changed),
    # not silently skipped.
    assert stats.written == 2
    assert stats.unchanged == 0
    assert stats.failed == 0

    doc = read_sidecar(organized)
    assert doc["note"] == "kept from before backfill existed"
    assert "identity" in doc
    assert "sources" in doc


def test_backfill_twice_is_fully_unchanged_and_byte_identical(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    _organize_without_sidecars(source_dir, organized_dir, catalog)

    first = backfill_sidecars(organized_dir, catalog)
    assert first.written == 2

    before = {p: p.read_bytes() for p in organized_dir.rglob("*.json")}

    second = backfill_sidecars(organized_dir, catalog)

    assert second.cataloged == 2
    assert second.written == 0
    assert second.unchanged == 2
    assert second.failed == 0

    after = {p: p.read_bytes() for p in organized_dir.rglob("*.json")}
    assert before == after


def test_backfill_dry_run_writes_nothing_but_reports_accurate_counts(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    _organize_without_sidecars(source_dir, organized_dir, catalog)

    stats = backfill_sidecars(organized_dir, catalog, dry_run=True)

    assert stats.cataloged == 2
    assert stats.written == 2
    assert stats.unchanged == 0
    assert stats.failed == 0
    assert not list(organized_dir.rglob("*.json")), "dry-run must write nothing"


def test_backfill_dry_run_reports_unchanged_for_an_already_backfilled_row(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    _organize_without_sidecars(source_dir, organized_dir, catalog)
    backfill_sidecars(organized_dir, catalog)
    before = {p: p.read_bytes() for p in organized_dir.rglob("*.json")}

    stats = backfill_sidecars(organized_dir, catalog, dry_run=True)

    assert stats.written == 0
    assert stats.unchanged == 2
    after = {p: p.read_bytes() for p in organized_dir.rglob("*.json")}
    assert before == after, "dry-run must not touch existing sidecars"


def test_backfill_missing_organized_file_counts_failed_and_continues(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    _organize_without_sidecars(source_dir, organized_dir, catalog)
    organized_files = list(organized_dir.rglob("*.jpg"))
    assert len(organized_files) == 2

    # Delete one organized file outright so it cannot be found by digest
    # either -- genuinely missing, not merely moved.
    organized_files[0].unlink()

    stats = backfill_sidecars(organized_dir, catalog)

    assert stats.cataloged == 2
    assert stats.failed == 1
    assert stats.written == 1
    assert stats.unchanged == 0


def test_backfill_finds_a_moved_file_by_digest(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """A file relocated since it was cataloged (stale recorded path) must
    still be backfilled, not reported failed -- resolve_organized_path finds
    it by its embedded digest."""
    _organize_without_sidecars(source_dir, organized_dir, catalog)
    organized_files = list(organized_dir.rglob("*.jpg"))
    moved = organized_files[0]
    new_home = organized_dir / "elsewhere"
    new_home.mkdir()
    new_path = new_home / moved.name
    moved.rename(new_path)

    stats = backfill_sidecars(organized_dir, catalog)

    assert stats.failed == 0
    assert stats.written == 2
    assert sidecar_path_for(new_path).exists()


def test_backfill_stats_dataclass_fields() -> None:
    stats = BackfillStats()
    assert stats.cataloged == 0
    assert stats.written == 0
    assert stats.unchanged == 0
    assert stats.failed == 0


# ---------------------------------------------------------------------------
# CLI wiring: `imageharbor sidecar backfill`
# ---------------------------------------------------------------------------


def test_cli_sidecar_backfill_writes_sidecars_and_reports_counts(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from imageharbor.cli import main

    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach.jpg")
    dest = tmp_path / "organized"

    proc = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--no-sidecar"]
    )
    assert proc.exit_code == 0, proc.output
    assert not list(dest.rglob("*.json"))

    result = CliRunner().invoke(main, ["sidecar", "backfill", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "Cataloged=1" in result.output
    assert "Written=1" in result.output
    assert list(dest.rglob("*.json"))


def test_cli_sidecar_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from imageharbor.cli import main

    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach.jpg")
    dest = tmp_path / "organized"

    CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--no-sidecar"]
    )

    result = CliRunner().invoke(
        main, ["sidecar", "backfill", "--dest", str(dest), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "[DRY-RUN]" in result.output
    assert not list(dest.rglob("*.json"))


def test_cli_sidecar_backfill_exits_nonzero_when_failed(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from imageharbor.cli import main

    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach.jpg")
    dest = tmp_path / "organized"

    CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--no-sidecar"]
    )
    for jpg in dest.rglob("*.jpg"):
        jpg.unlink()

    result = CliRunner().invoke(main, ["sidecar", "backfill", "--dest", str(dest)])
    assert result.exit_code == 1
    assert "Failed=1" in result.output
