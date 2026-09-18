"""Tests for the public facade in imageharbor.api."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imageharbor import api
from imageharbor.ai_classifier import StubClassifier
from imageharbor.hashing import compute_sha256_b64url


def test_exception_hierarchy():
    assert issubclass(api.ConfigError, api.ImageHarborError)
    assert issubclass(api.Aborted, api.ImageHarborError)
    assert api.Aborted("x").report is None


def test_outcome_constants_are_their_own_names():
    for name in ("COPIED", "DUPLICATE", "SKIPPED", "ERROR", "ENRICHED",
                 "RENAMED", "TOTAL", "AI", "IO", "OK", "FAILED"):
        assert getattr(api, name) == name


def test_aiconfig_defaults_match_the_cli():
    cfg = api.AIConfig()
    assert (cfg.backend, cfg.base_url, cfg.model, cfg.timeout, cfg.api_key) == (
        "stub", None, "gpt-4o-mini", 60.0, None
    )


def test_build_classifier_stub_default():
    assert isinstance(api._build_classifier(api.AIConfig()), StubClassifier)


def test_build_classifier_unknown_backend_is_a_config_error():
    with pytest.raises(api.ConfigError):
        api._build_classifier(api.AIConfig(backend="nope"))


def test_guard_rejects_dest_inside_source(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    with pytest.raises(api.ConfigError) as ei:
        api._guard_dest_not_inside_source(src, src / "d")
    assert "--dest" in str(ei.value) and "--source" in str(ei.value)


def test_guard_allows_sibling_dest(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    api._guard_dest_not_inside_source(src, tmp_path / "d")  # no raise


def _jpeg(path: Path, fill: int = 0) -> Path:
    path.write_bytes(b"\xff\xd8\xff\xe0" + bytes([fill]) * 16 + b"\xff\xd9")
    return path


def _source(tmp_path: Path, n: int = 2) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        _jpeg(src / f"img_{i}.jpg", i)
    return src


def test_process_reports_one_row_per_file_and_copies(tmp_path: Path):
    src = _source(tmp_path)
    report = api.process(src, tmp_path / "org")
    assert report.ok
    assert report.counts == {api.COPIED: 2, api.DUPLICATE: 0, api.SKIPPED: 0, api.ERROR: 0, api.TOTAL: 2}
    assert len(report.rows) == 2
    for row in report.rows:
        assert row.outcome == api.COPIED
        assert row.dest_path is not None and Path(row.dest_path).exists()
        assert row.digest == compute_sha256_b64url(Path(row.source_path))
        assert row.detail == ""
    assert report.catalog == str(tmp_path / "org" / "catalog.db")
    assert report.dry_run is False
    assert report.started <= report.finished


def test_process_second_run_reports_duplicates(tmp_path: Path):
    src = _source(tmp_path)
    api.process(src, tmp_path / "org")
    report = api.process(src, tmp_path / "org")
    assert report.ok
    assert report.counts[api.DUPLICATE] == 2 and report.counts[api.COPIED] == 0
    assert {r.outcome for r in report.rows} == {api.DUPLICATE}


def test_process_unreadable_file_is_an_error_row_not_an_exception(tmp_path: Path, monkeypatch):
    src = _source(tmp_path, n=2)
    from imageharbor import pipeline as pipeline_mod

    real = pipeline_mod.compute_sha256_b64url

    def _boom(path):
        if path.name == "img_1.jpg":
            raise OSError("unreadable")
        return real(path)

    monkeypatch.setattr(pipeline_mod, "compute_sha256_b64url", _boom)
    report = api.process(src, tmp_path / "org")
    assert not report.ok
    assert report.counts[api.ERROR] == 1 and report.counts[api.COPIED] == 1
    bad = [r for r in report.rows if r.outcome == api.ERROR]
    assert len(bad) == 1 and "unreadable" in bad[0].detail and bad[0].dest_path is None


def test_process_dry_run_writes_nothing(tmp_path: Path):
    src = _source(tmp_path)
    dest = tmp_path / "org"
    report = api.process(src, dest, dry_run=True)
    assert report.dry_run is True
    assert not dest.exists()
    assert report.counts[api.TOTAL] == 2


def test_process_dest_inside_source_is_a_config_error(tmp_path: Path):
    src = _source(tmp_path)
    with pytest.raises(api.ConfigError):
        api.process(src, src / "org")
    assert not (src / "org").exists()


def test_process_missing_source_is_a_config_error(tmp_path: Path):
    with pytest.raises(api.ConfigError):
        api.process(tmp_path / "nope", tmp_path / "org")


def test_process_accepts_str_paths_and_custom_catalog(tmp_path: Path):
    src = _source(tmp_path)
    cat = tmp_path / "elsewhere" / "cat.db"
    report = api.process(str(src), str(tmp_path / "org"), catalog=str(cat))
    assert report.ok and cat.exists() and report.catalog == str(cat)


def test_process_report_to_dict_round_trips_json(tmp_path: Path):
    report = api.process(_source(tmp_path), tmp_path / "org")
    doc = json.loads(json.dumps(report.to_dict()))
    assert set(doc) == {"source", "dest", "catalog", "dry_run", "started", "finished", "counts", "rows"}
    assert doc["counts"] == report.counts
    assert doc["rows"][0] == {
        "source_path": report.rows[0].source_path, "outcome": "COPIED",
        "dest_path": report.rows[0].dest_path, "digest": report.rows[0].digest, "detail": "",
    }


class _Boom(StubClassifier):
    def describe(self, image_path, exif_data):
        raise RuntimeError("down")


def _organized(tmp_path: Path, n: int = 3) -> Path:
    dest = tmp_path / "org"
    api.process(_source(tmp_path, n), dest)
    return dest


def test_enrich_happy_path_counts_and_no_rows(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    report = api.enrich(dest)
    assert report.ok and not report.aborted
    assert report.ai_backend == "stub"
    assert report.counts[api.ENRICHED] == 2 and report.counts[api.TOTAL] == 2
    assert report.counts[api.ERROR] == 0 and api.RENAMED in report.counts
    assert report.rows == ()
    assert report.catalog == str(dest / "catalog.db")


def test_enrich_is_idempotent_second_run_has_nothing_to_do(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    api.enrich(dest)
    again = api.enrich(dest)
    assert again.ok and again.counts[api.TOTAL] == 0


def test_enrich_breaker_trip_raises_aborted_with_partial_report(tmp_path: Path, monkeypatch):
    dest = _organized(tmp_path, 4)
    monkeypatch.setattr(api, "_build_classifier", lambda ai: _Boom())
    with pytest.raises(api.Aborted) as ei:
        api.enrich(dest, breaker_threshold=2)
    report = ei.value.report
    assert report is not None and report.aborted and not report.ok
    assert report.counts[api.ERROR] == 2
    assert len(report.rows) == 2
    assert all(r.reason == api.AI and "down" in r.detail for r in report.rows)
    assert all(r.dest_path is not None for r in report.rows)


def test_enrich_breaker_disabled_runs_every_row_and_returns(tmp_path: Path, monkeypatch):
    dest = _organized(tmp_path, 3)
    monkeypatch.setattr(api, "_build_classifier", lambda ai: _Boom())
    report = api.enrich(dest, breaker_threshold=0)
    assert not report.aborted and not report.ok
    assert report.counts[api.ERROR] == 3 and len(report.rows) == 3


def test_enrich_missing_organized_file_is_an_io_row(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    victim = next(p for p in dest.rglob("*.jpg"))
    victim.unlink()
    report = api.enrich(dest)
    assert not report.ok and not report.aborted
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.reason == api.IO and row.dest_path is None and "missing" in row.detail.lower()


def test_enrich_missing_catalog_is_a_config_error(tmp_path: Path):
    with pytest.raises(api.ConfigError):
        api.enrich(tmp_path / "nowhere")


def test_enrich_unknown_backend_is_a_config_error(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    with pytest.raises(api.ConfigError):
        api.enrich(dest, ai=api.AIConfig(backend="nope"))


def test_enrich_report_to_dict_round_trips_json(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    victim = next(p for p in dest.rglob("*.jpg"))
    victim.unlink()
    doc = json.loads(json.dumps(api.enrich(dest).to_dict()))
    assert set(doc) == {"dest", "catalog", "ai_backend", "started", "finished", "aborted", "counts", "rows"}
    assert doc["rows"][0]["reason"] == "IO" and doc["rows"][0]["dest_path"] is None
