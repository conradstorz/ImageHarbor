"""EnrichStats.failures carries path + reason for every failed row, in
lockstep with the existing ai_failed / io_failed digest lists."""

from __future__ import annotations

from pathlib import Path

from imageharbor.ai_classifier import StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrichment import EnrichFailure, enrich_library
from imageharbor.pipeline import Pipeline


def _organize(tmp_path: Path, n: int = 3) -> tuple[Path, Path]:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        (src / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")
    dest = tmp_path / "org"
    with Catalog(dest / "catalog.db") as catalog:
        Pipeline(source_dir=src, organized_dir=dest, catalog=catalog).run()
    return src, dest


class _Boom(StubClassifier):
    def describe(self, image_path, exif_data):
        raise RuntimeError("down")


def test_describe_failure_is_an_ai_failure_with_path_and_detail(tmp_path):
    _, dest = _organize(tmp_path, n=1)
    with Catalog(dest / "catalog.db") as catalog:
        stats = enrich_library(catalog, dest, _Boom())
    assert stats.ai_failed and not stats.io_failed
    assert len(stats.failures) == 1
    f = stats.failures[0]
    assert isinstance(f, EnrichFailure)
    assert f.digest == stats.ai_failed[0]
    assert f.reason == "AI"
    assert f.organized_path is not None and f.organized_path.exists()
    assert "down" in f.detail


def test_missing_organized_file_is_an_io_failure_with_no_path(tmp_path):
    _, dest = _organize(tmp_path, n=1)
    with Catalog(dest / "catalog.db") as catalog:
        row = next(iter(catalog.iter_unenriched(None, 0)))
        Path(row["organized_path"]).unlink()
        stats = enrich_library(catalog, dest, StubClassifier())
    assert stats.io_failed and not stats.ai_failed
    assert len(stats.failures) == 1
    f = stats.failures[0]
    assert f.digest == stats.io_failed[0]
    assert f.reason == "IO"
    assert f.organized_path is None
    assert "missing" in f.detail.lower()


def test_failures_and_digest_lists_stay_in_lockstep(tmp_path):
    _, dest = _organize(tmp_path, n=3)
    with Catalog(dest / "catalog.db") as catalog:
        stats = enrich_library(catalog, dest, _Boom())
    assert [f.digest for f in stats.failures] == stats.ai_failed + stats.io_failed
    assert stats.errors == len(stats.failures)


def test_io_failure_after_a_rename_records_the_new_path(tmp_path, monkeypatch):
    """A post-perception failure that happens AFTER the tier-gated rename must
    record where the file now lives, not its pre-rename path."""
    _, dest = _organize(tmp_path, n=1)
    before = next(p for p in dest.rglob("*.jpg"))

    def _boom_set_placement(*args, **kwargs):
        raise RuntimeError("catalog down after rename")

    with Catalog(dest / "catalog.db") as catalog:
        # set_placement is only reached on the rename branch, i.e. after the
        # file has already been moved on disk.
        monkeypatch.setattr(catalog, "set_placement", _boom_set_placement)
        stats = enrich_library(catalog, dest, StubClassifier())
    assert stats.io_failed and len(stats.failures) == 1
    f = stats.failures[0]
    assert f.reason == "IO"
    assert f.organized_path is not None and f.organized_path.exists()
    assert f.organized_path != before
    assert not before.exists()
