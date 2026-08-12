"""CLI (Click) tests for ImageHarbor.

Exercises the `imageharbor.cli.main` command group end-to-end using
``click.testing.CliRunner``.  The default *stub* AI backend is used
throughout so no network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from imageharbor.cli import main
from imageharbor.hashing import compute_sha256_b64url, verify_pcs_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(
    path: Path,
    content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9",
) -> Path:
    """Write a minimal pseudo-JPEG file."""
    path.write_bytes(content)
    return path


def _source_with_two_jpegs(tmp_path: Path) -> Path:
    """Create a source directory containing two distinct minimal JPEGs."""
    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach_photo.jpg")
    _make_jpeg(
        src / "mountain_view.jpg",
        b"\xff\xd8\xff\xe0" + b"\x01" * 16 + b"\xff\xd9",
    )
    return src


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def test_process_copies_and_catalogs(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"

    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])

    assert result.exit_code == 0, result.output
    assert "Copied=2" in result.output
    assert "Errors=0" in result.output

    # Organized image files exist under dest
    organized = list(dest.rglob("*.jpg"))
    assert len(organized) == 2

    # Catalog created and populated
    catalog_db = dest / "catalog.db"
    assert catalog_db.exists()

    # Each organized file's embedded digest matches its actual bytes
    for f in organized:
        assert verify_pcs_file(f), f"digest mismatch for {f.name}"


def test_process_dry_run_writes_nothing(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"

    result = runner.invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "[DRY-RUN]" in result.output
    assert "Copied=2" in result.output

    # Dry-run must not create the dest directory at all.
    assert not dest.exists()

    # No organized image files written, and no catalog.db (or WAL) on disk.
    assert not list(dest.rglob("*.jpg"))
    assert not (dest / "catalog.db").exists()


def test_process_twice_reports_duplicates(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    args = ["process", "--source", str(src), "--dest", str(dest)]

    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    assert "Copied=2" in first.output

    second = runner.invoke(main, args)
    assert second.exit_code == 0, second.output
    assert "Duplicates=2" in second.output
    assert "Copied=0" in second.output


def test_process_duplicates_dir_receives_copies(
    runner: CliRunner, tmp_path: Path
) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    dups = tmp_path / "dups"
    args = [
        "process",
        "--source",
        str(src),
        "--dest",
        str(dest),
        "--duplicates",
        str(dups),
    ]

    # First run: nothing is a duplicate yet.
    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    assert not dups.exists() or not list(dups.iterdir())

    # Second run: both images are duplicates -> copied into the dups dir.
    second = runner.invoke(main, args)
    assert second.exit_code == 0, second.output
    assert "Duplicates=2" in second.output
    assert dups.exists()
    assert len(list(dups.iterdir())) == 2


def test_process_sidecar_written(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"

    result = runner.invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--sidecar"]
    )
    assert result.exit_code == 0, result.output

    sidecars = list(dest.rglob("*.json"))
    # catalog.db is not a json file; only sidecars should match.
    assert len(sidecars) == 2
    for s in sidecars:
        data = json.loads(s.read_text(encoding="utf-8"))
        assert "sha256_b64url" in data["identity"]
        assert len(data["identity"]["sha256_b64url"]) == 43


def test_process_no_sidecar_default(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"

    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert not list(dest.rglob("*.json"))


def test_process_single_file_source(runner: CliRunner, tmp_path: Path) -> None:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    single = _make_jpeg(src_dir / "one.jpg")
    dest = tmp_path / "organized"

    result = runner.invoke(
        main, ["process", "--source", str(single), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert "Copied=1" in result.output
    assert len(list(dest.rglob("*.jpg"))) == 1


def test_process_custom_catalog_path(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    catalog_db = tmp_path / "my_catalog.db"

    result = runner.invoke(
        main,
        [
            "process",
            "--source",
            str(src),
            "--dest",
            str(dest),
            "--catalog",
            str(catalog_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert catalog_db.exists()
    assert not (dest / "catalog.db").exists()


def test_process_missing_source_errors(runner: CliRunner, tmp_path: Path) -> None:
    dest = tmp_path / "organized"
    result = runner.invoke(
        main,
        ["process", "--source", str(tmp_path / "nope"), "--dest", str(dest)],
    )
    # Click validates existence of --source (exists=True) -> usage error.
    assert result.exit_code != 0


def test_process_rejects_dest_inside_source(runner: CliRunner, tmp_path: Path) -> None:
    """enrich/watch rename files under --dest; a --dest nested inside
    --source would put those renames into the read-only source tree."""
    src = tmp_path / "source"
    src.mkdir()
    dest = src / "organized"  # nested inside source

    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])

    assert result.exit_code != 0
    assert "--dest" in result.output and "--source" in result.output
    assert not dest.exists()  # must fail before writing anything


def test_process_rejects_dest_equal_to_source(runner: CliRunner, tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()

    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(src)])

    assert result.exit_code != 0


def test_process_rejects_dest_inside_source_across_relative_and_absolute(
    runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """The guard must compare resolved paths, not the strings as typed.

    A relative --source and an absolute --dest can name the same tree while
    sharing no textual prefix, which is exactly the shape a naive string
    comparison misses.
    """
    src = tmp_path / "source"
    src.mkdir()
    dest = src / "organized"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main, ["process", "--source", "source", "--dest", str(dest)])

    assert result.exit_code != 0
    assert "--dest" in result.output and "--source" in result.output
    assert not dest.exists()


def test_process_allows_sibling_dest(runner: CliRunner, tmp_path: Path) -> None:
    """A --dest that merely shares a parent with --source (not nested inside
    it) must be unaffected by the new guard."""
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"  # sibling of source, not nested inside it

    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])

    assert result.exit_code == 0, result.output


def test_watch_rejects_dest_inside_source(runner: CliRunner, tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    dest = src / "organized"

    # No --stop-after-passes/etc exists, but the guard must fire before the
    # watch loop is ever entered, so this invocation must return promptly.
    result = runner.invoke(main, ["watch", "--source", str(src), "--dest", str(dest)])

    assert result.exit_code != 0
    assert "--dest" in result.output and "--source" in result.output
    assert not dest.exists()


def test_process_no_longer_accepts_ai_flags(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    result = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(tmp_path / "d"), "--ai", "stub"]
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_process_organizes_without_any_ai(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    result = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert (dest / "2019" / "2019-07").exists()


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------


def test_enrich_command_exists_and_reports(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    runner = CliRunner()
    runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    result = runner.invoke(
        main, ["enrich", "--dest", str(dest), "--ai", "stub"]
    )
    assert result.exit_code == 0, result.output
    assert "enriched" in result.output.lower()


def test_enrich_accepts_limit_and_reclassify(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(
        main,
        ["enrich", "--dest", str(dest), "--ai", "stub", "--limit", "1", "--reclassify"],
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_organized_dir_all_ok(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output

    result = runner.invoke(main, ["verify", str(dest)])
    assert result.exit_code == 0, result.output
    assert "OK   " in result.output
    assert "2 OK, 0 FAILED" in result.output


def test_verify_corrupted_file_fails(runner: CliRunner, tmp_path: Path) -> None:
    # Build a PCS-named file whose digest matches its content, then corrupt it.
    content = b"\xff\xd8\xff\xe0" + b"\x02" * 16 + b"\xff\xd9"
    tmp_file = tmp_path / "seed.jpg"
    tmp_file.write_bytes(content)
    digest = compute_sha256_b64url(tmp_file)

    pcs_file = tmp_path / f"920-x_{digest}.jpg"
    pcs_file.write_bytes(content)
    # Sanity: name matches content before corruption.
    assert verify_pcs_file(pcs_file)

    # Corrupt the bytes so the embedded digest no longer matches.
    pcs_file.write_bytes(content + b"\x00")
    assert not verify_pcs_file(pcs_file)

    result = runner.invoke(main, ["verify", str(pcs_file)])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "1 FAILED" in result.output


def test_verify_non_pcs_file_skipped(runner: CliRunner, tmp_path: Path) -> None:
    # A supported-extension image whose name is not in PCS format -> skipped.
    plain = _make_jpeg(tmp_path / "just_a_photo.jpg")

    result = runner.invoke(main, ["verify", str(plain)])
    # Nothing was actually verified -> non-zero exit with a clear warning.
    assert result.exit_code != 0, result.output
    # No per-file FAIL line (the summary word "FAILED" does not count).
    assert not any(ln.startswith("FAIL ") for ln in result.output.splitlines())
    assert "0 OK, 0 FAILED" in result.output
    assert "No organized image files (with an embedded digest) found to verify." in result.output


def test_verify_directory_mixes_ok_and_skip(runner: CliRunner, tmp_path: Path) -> None:
    d = tmp_path / "lib"
    d.mkdir()

    # One valid PCS file.
    content = b"\xff\xd8\xff\xe0" + b"\x03" * 16 + b"\xff\xd9"
    seed = tmp_path / "seed.jpg"
    seed.write_bytes(content)
    digest = compute_sha256_b64url(seed)
    (d / f"100-scene_{digest}.jpg").write_bytes(content)

    # A non-image file and a non-PCS image, both should be skipped.
    (d / "notes.txt").write_text("hello", encoding="utf-8")
    _make_jpeg(d / "random.jpg")

    result = runner.invoke(main, ["verify", str(d)])
    assert result.exit_code == 0, result.output
    assert "1 OK, 0 FAILED" in result.output


# ---------------------------------------------------------------------------
# catalog list / get
# ---------------------------------------------------------------------------


def test_catalog_list_empty(runner: CliRunner, tmp_path: Path) -> None:
    from imageharbor.catalog import Catalog

    db = tmp_path / "catalog.db"
    Catalog(db).close()  # create empty catalog

    result = runner.invoke(main, ["catalog", "list", "--catalog", str(db)])
    assert result.exit_code == 0, result.output
    assert "(empty catalog)" in result.output


def test_catalog_list_populated(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output
    db = dest / "catalog.db"

    result = runner.invoke(main, ["catalog", "list", "--catalog", str(db)])
    assert result.exit_code == 0, result.output
    assert "(empty catalog)" not in result.output
    # Two rows -> two non-empty content lines.
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 2


def test_catalog_list_shows_dash_for_unenriched_rows(runner: CliRunner, tmp_path: Path) -> None:
    """A facts-pass-only row was never classified; `catalog list` must not
    assert a fake "900" classification for it."""
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output
    db = dest / "catalog.db"

    result = runner.invoke(main, ["catalog", "list", "--catalog", str(db)])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert all("—" in ln for ln in lines)
    assert not any("900" in ln for ln in lines)


def test_catalog_list_shows_real_class_for_enriched_rows(runner: CliRunner, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "organized"

    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output
    db = dest / "catalog.db"
    en = runner.invoke(main, ["enrich", "--dest", str(dest), "--catalog", str(db), "--ai", "stub"])
    assert en.exit_code == 0, en.output

    result = runner.invoke(main, ["catalog", "list", "--catalog", str(db)])
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "—" not in lines[0]


def test_catalog_list_limit(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output
    db = dest / "catalog.db"

    result = runner.invoke(
        main, ["catalog", "list", "--catalog", str(db), "--limit", "1"]
    )
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    assert len(lines) == 1


def test_catalog_get_existing(runner: CliRunner, tmp_path: Path) -> None:
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output
    db = dest / "catalog.db"

    # Recover a known sha256 from a source file.
    sha = compute_sha256_b64url(src / "beach_photo.jpg")

    result = runner.invoke(main, ["catalog", "get", "--catalog", str(db), sha])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["sha256_b64url"] == sha


def test_catalog_get_decodes_json_columns(runner: CliRunner, tmp_path: Path) -> None:
    """`catalog get` must emit structured values for JSON-TEXT columns rather
    than escaped strings."""
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output
    db = dest / "catalog.db"

    sha = compute_sha256_b64url(src / "beach_photo.jpg")

    result = runner.invoke(main, ["catalog", "get", "--catalog", str(db), sha])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)

    # JSON-TEXT columns decode to native structures, not escaped strings.
    assert isinstance(data["exif"], dict)
    assert isinstance(data["processing_history"], list)
    assert isinstance(data["secondary_tags"], list)
    assert isinstance(data["objects"], list)


def test_catalog_get_missing(runner: CliRunner, tmp_path: Path) -> None:
    from imageharbor.catalog import Catalog

    db = tmp_path / "catalog.db"
    Catalog(db).close()

    result = runner.invoke(
        main, ["catalog", "get", "--catalog", str(db), "does-not-exist"]
    )
    assert result.exit_code == 1
    assert "Not found: does-not-exist" in result.output


# ---------------------------------------------------------------------------
# enrich --ai openai (no openai package required)
# ---------------------------------------------------------------------------


def test_enrich_ai_openai_without_package_fails_gracefully(
    runner: CliRunner, tmp_path: Path
) -> None:
    """With --ai openai, if the openai backend can't be constructed (e.g. the
    optional package is unavailable or no key), the run must fail, not crash
    silently.  We only assert a non-zero exit and that no crash produced a
    successful summary."""
    dest = tmp_path / "organized"
    dest.mkdir()

    result = runner.invoke(
        main,
        [
            "enrich",
            "--dest",
            str(dest),
            "--ai",
            "openai",
            "--openai-key",
            "sk-test-not-real",
        ],
    )

    # Either an ImportError (package missing) or another failure to build the
    # classifier surfaces as a non-zero exit.  If the environment somehow has a
    # usable openai backend, the run may succeed; in that case there is nothing
    # to assert about failure, so guard the assertion.
    if result.exit_code == 0:
        pytest.skip("openai backend is available in this environment")
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _build_classifier / watch
# ---------------------------------------------------------------------------


def test_build_classifier_stub_default() -> None:
    from imageharbor.cli import _build_classifier
    from imageharbor.ai_classifier import StubClassifier

    clf = _build_classifier("stub", None, None, "gpt-4o-mini", 60.0)
    assert isinstance(clf, StubClassifier)


def test_cli_watch_wires_args(monkeypatch, tmp_path):
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9")
    dest = tmp_path / "dest"

    captured = {}

    def _fake_watch(**kwargs):
        captured.update(kwargs)
        return WatchStats(passes=1)

    monkeypatch.setattr(_watcher, "watch", _fake_watch)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["watch", "--source", str(src), "--dest", str(dest), "--interval", "5"],
    )
    assert result.exit_code == 0, result.output
    assert captured["interval"] == 5.0
    assert captured["source"] == src


# ---------------------------------------------------------------------------
# circuit breaker wiring
# ---------------------------------------------------------------------------


# NOTE: `Pipeline` (the facts pass) no longer accepts a `classifier` and
# `Pipeline.run()` no longer accepts a `breaker` -- the facts pass makes no
# AI calls at all now (see imageharbor/pipeline.py). Breaker-trip-aborts-the-
# pass behavior lives in the enrichment pass and is covered directly by
# tests/test_enrich.py::test_a_tripped_breaker_aborts_the_pass. What remains
# here is CLI-level: `enrich` must report and exit non-zero when the breaker
# trips.


def test_enrich_command_aborts_and_reports_when_backend_down(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from imageharbor.cli import main

    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        (src / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")
    dest = tmp_path / "org"

    runner = CliRunner()
    proc = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert proc.exit_code == 0, proc.output

    # Force the stub classifier to fail so the breaker trips.
    from imageharbor.ai_classifier import StubClassifier

    def _boom(self, image_path, exif_data):
        raise RuntimeError("down")

    monkeypatch.setattr(StubClassifier, "describe", _boom)

    result = runner.invoke(
        main,
        ["enrich", "--dest", str(dest), "--breaker-threshold", "2"],
    )
    assert result.exit_code == 1
    assert "backend appears down" in result.output.lower()
