"""The --json subprocess contract on process / enrich / verify, and the
nas-ingest-aligned exit codes 0 / 1 / 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from imageharbor import api
from imageharbor.cli import main


def _jpeg(path: Path, fill: int = 0) -> Path:
    path.write_bytes(b"\xff\xd8\xff\xe0" + bytes([fill]) * 16 + b"\xff\xd9")
    return path


def _source(tmp_path: Path, n: int = 2) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        _jpeg(src / f"img_{i}.jpg", i)
    return src


@pytest.fixture()
def runner() -> CliRunner:
    # click >= 8.2 (we pin 8.4.x): .stdout and .stderr are always separate;
    # .output is the interleaved view.
    return CliRunner()


def _doc(result) -> dict:
    """stdout must be exactly one JSON document."""
    return json.loads(result.stdout)


# --- process -----------------------------------------------------------------

def test_process_json_is_the_report_to_dict(runner, tmp_path):
    src = _source(tmp_path)
    dest = tmp_path / "org"
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest), "--json"])
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert set(doc) == {"source", "dest", "catalog", "dry_run", "started", "finished", "counts", "rows"}
    assert doc["counts"]["COPIED"] == 2 and len(doc["rows"]) == 2
    assert "Done." not in result.stdout


def test_process_json_exit_1_when_a_row_is_an_error(runner, tmp_path, monkeypatch):
    src = _source(tmp_path, 2)
    from imageharbor import pipeline as pipeline_mod

    real = pipeline_mod.compute_sha256_b64url
    monkeypatch.setattr(
        pipeline_mod, "compute_sha256_b64url",
        lambda p: (_ for _ in ()).throw(OSError("unreadable")) if p.name == "img_1.jpg" else real(p),
    )
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(tmp_path / "org"), "--json"])
    assert result.exit_code == 1
    doc = _doc(result)
    assert doc["counts"]["ERROR"] == 1


def test_process_json_exit_2_on_config_error_prints_no_document(runner, tmp_path):
    src = _source(tmp_path)
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(src / "org"), "--json"])
    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "--dest" in result.stderr


def test_process_prose_is_unchanged_without_json(runner, tmp_path):
    src = _source(tmp_path)
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(tmp_path / "org")])
    assert result.exit_code == 0
    assert result.stdout.startswith("Done. Total=2  Copied=2  Duplicates=0  Errors=0")


def test_process_json_dry_run_prints_document_not_banner(runner, tmp_path):
    src = _source(tmp_path)
    dest = tmp_path / "org"
    result = runner.invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert doc["dry_run"] is True
    assert doc["counts"]["TOTAL"] == 2
    assert "[DRY-RUN]" not in result.stdout
    assert not dest.exists()


# --- enrich ------------------------------------------------------------------

def test_enrich_json_is_the_report_to_dict(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path), dest)
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--json"])
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert set(doc) == {"dest", "catalog", "ai_backend", "started", "finished", "aborted", "counts", "rows"}
    assert doc["counts"]["ENRICHED"] == 2 and doc["aborted"] is False


def test_enrich_json_breaker_trip_exits_2_with_document_and_error(runner, tmp_path, monkeypatch):
    dest = tmp_path / "org"
    api.process(_source(tmp_path, 4), dest)
    from imageharbor.ai_classifier import StubClassifier

    monkeypatch.setattr(StubClassifier, "describe", lambda self, p, e: (_ for _ in ()).throw(RuntimeError("down")))
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--breaker-threshold", "2", "--json"])
    assert result.exit_code == 2
    doc = _doc(result)
    assert doc["aborted"] is True
    assert doc["counts"]["ERROR"] == 2 and len(doc["rows"]) == 2
    assert "backend appears down" in doc["error"].lower()
    assert "backend appears down" in result.stderr.lower()


def test_enrich_json_io_failure_exits_1(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path, 1), dest)
    next(dest.rglob("*.jpg")).unlink()
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--json"])
    assert result.exit_code == 1
    doc = _doc(result)
    assert doc["rows"][0]["reason"] == "IO" and "error" not in doc


def test_enrich_prose_breaker_trip_now_exits_2(runner, tmp_path, monkeypatch):
    dest = tmp_path / "org"
    api.process(_source(tmp_path, 4), dest)
    from imageharbor.ai_classifier import StubClassifier

    monkeypatch.setattr(StubClassifier, "describe", lambda self, p, e: (_ for _ in ()).throw(RuntimeError("down")))
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--breaker-threshold", "2"])
    assert result.exit_code == 2
    assert "backend appears down" in result.stderr.lower()


def test_enrich_json_missing_catalog_exits_2_with_no_document(runner, tmp_path):
    dest = tmp_path / "org"
    dest.mkdir()
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--json"])
    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "catalog" in result.stderr.lower()


# --- verify ------------------------------------------------------------------

def test_verify_json_is_the_report_to_dict(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path), dest)
    result = runner.invoke(main, ["verify", str(dest), "--json"])
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert set(doc) == {"path", "started", "finished", "counts", "rows"}
    assert doc["counts"]["OK"] == 2 and doc["counts"]["FAILED"] == 0
    assert set(doc["counts"]) == {"OK", "FAILED", "SKIPPED", "TOTAL"}
    assert not result.stdout.startswith("OK ")


def test_verify_json_failed_row_exits_1(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path), dest)
    next(dest.rglob("*.jpg")).write_bytes(b"\xff\xd8bad\xff\xd9")
    result = runner.invoke(main, ["verify", str(dest), "--json"])
    assert result.exit_code == 1
    assert _doc(result)["counts"]["FAILED"] == 1


def test_verify_nothing_verifiable_exits_2_in_both_modes(runner, tmp_path):
    plain = _jpeg(tmp_path / "just_a_photo.jpg")
    prose = runner.invoke(main, ["verify", str(plain)])
    assert prose.exit_code == 2
    assert "No organized image files" in prose.stderr
    js = runner.invoke(main, ["verify", str(plain), "--json"])
    assert js.exit_code == 2
    doc = _doc(js)
    assert doc["counts"]["SKIPPED"] == 1 and "No organized image files" in doc["error"]
