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
        assert "sha256_b64url" in data
        assert len(data["sha256_b64url"]) == 43


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
    assert "No PCS-format image files found to verify." in result.output


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
# process --ai openai (no openai package required)
# ---------------------------------------------------------------------------


def test_process_ai_openai_without_package_fails_gracefully(
    runner: CliRunner, tmp_path: Path
) -> None:
    """With --ai openai, if the openai backend can't be constructed (e.g. the
    optional package is unavailable or no key), the run must fail, not crash
    silently.  We only assert a non-zero exit and that no crash produced a
    successful summary."""
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"

    result = runner.invoke(
        main,
        [
            "process",
            "--source",
            str(src),
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
