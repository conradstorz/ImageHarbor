"""CLI (Click) tests for ImageHarbor.

Exercises the `imageharbor.cli.main` command group end-to-end using
``click.testing.CliRunner``.  The default *stub* AI backend is used
throughout so no network access is required.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from imageharbor.cli import main
from imageharbor.hashing import compute_sha256_b64url, verify_pcs_file

# Mirrors tests/faces/test_detect.py's own skip gate: real Detector/Embedder
# construction downloads and loads an ONNX session, so the one CLI test that
# exercises it for real (rather than via the `HAS_ONNX=False` branch above)
# only runs when weights are already staged locally.
_FACE_MODEL_DIR = (
    Path(os.environ["IMAGEHARBOR_FACE_MODEL_DIR"])
    if os.environ.get("IMAGEHARBOR_FACE_MODEL_DIR")
    else None
)


def _face_weights_present() -> bool:
    if _FACE_MODEL_DIR is None:
        return False
    from imageharbor.faces import models as face_models

    return (_FACE_MODEL_DIR / face_models.DETECTORS["yunet"].filename).exists()


needs_face_weights = pytest.mark.skipif(
    not _face_weights_present(),
    reason="set IMAGEHARBOR_FACE_MODEL_DIR to a directory holding the weights",
)


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


def test_process_writes_a_sidecar_by_default(tmp_path: Path) -> None:
    """The flag flip, stated as behavior rather than as a default value."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b" " * 16 + b"\xff\xd9")
    dest = tmp_path / "organized"

    result = CliRunner().invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert [p for p in dest.rglob("*.json")], "no sidecar written without a flag"


def test_no_sidecar_still_suppresses(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b" " * 16 + b"\xff\xd9")
    dest = tmp_path / "organized"

    result = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--no-sidecar"]
    )
    assert result.exit_code == 0, result.output
    assert [p for p in dest.rglob("*.json")] == []


def test_process_no_sidecar_flag_suppresses_for_every_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--no-sidecar suppresses sidecars for every organized file, not just one.

    Was `test_process_no_sidecar_default`, which asserted the OLD default (no
    flag -> no sidecar). Sidecars are now written by default (see
    `test_process_writes_a_sidecar_by_default` above), so this test's job
    shifted to covering the explicit opt-out across multiple files -- the
    new default-on behavior for a single file is already covered above.
    """
    src = _source_with_two_jpegs(tmp_path)
    dest = tmp_path / "organized"

    result = runner.invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--no-sidecar"]
    )
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
        # --no-dashboard: this test is about arg wiring into `_watcher.watch`,
        # not the dashboard -- without it, a real `serve()` call would bind
        # an actual port 8080 as an unrelated side effect of running this
        # test (see the dashboard-specific tests below for real bind/port
        # coverage).
        [
            "watch", "--source", str(src), "--dest", str(dest),
            "--interval", "5", "--no-dashboard",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["interval"] == 5.0
    # `watch` takes no `source` kwarg: the tree to walk is derived from the
    # pipeline, so that discovery and the pipeline can never disagree.
    assert "source" not in captured
    assert captured["pipeline"].source_dir == src
    # Task 8: `watch()` now always receives the live control object (not a
    # frozen `interval`/`enrich_enabled` value) so a dashboard change takes
    # effect without a restart -- see watcher.watch's docstring.
    from imageharbor.dashboard.control import ControlPlane

    assert isinstance(captured["control"], ControlPlane)


# ---------------------------------------------------------------------------
# faces flags and wiring (Task 16)
# ---------------------------------------------------------------------------


def test_cli_watch_faces_off_by_default(monkeypatch, tmp_path):
    """`--faces` defaults to off -- a new, heavier, opt-in extra must not
    start running face detection just because `watch` was invoked."""
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
        ["watch", "--source", str(src), "--dest", str(dest), "--no-dashboard"],
    )
    assert result.exit_code == 0, result.output
    assert captured["faces_enabled"] is False
    assert captured["face_config"] is None


def test_cli_watch_faces_requested_but_extra_unavailable_still_organizes(
    monkeypatch, tmp_path
):
    """`--faces` with the extra missing must not stop `watch` from starting
    -- it degrades to a warning and `face_config=None`, exactly like a
    dashboard bind failure degrades instead of crashing (see
    dashboard/server.py's module docstring) and like `watcher.watch`'s own
    faces-pass handling of `faces_available() is False`."""
    from imageharbor import faces as faces_pkg
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)

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
        [
            "watch", "--source", str(src), "--dest", str(dest), "--no-dashboard",
            "--faces",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "faces' extra is not installed" in result.output
    # faces_enabled still reflects the *request* -- watcher.watch is what
    # decides (via faces_available()) to skip and warn once, not this layer.
    assert captured["faces_enabled"] is True
    assert captured["face_config"] is None


def test_cli_watch_face_threshold_rejects_a_hostile_value(monkeypatch, tmp_path):
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    monkeypatch.setattr(_watcher, "watch", lambda **kwargs: WatchStats(passes=1))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "watch", "--source", str(src), "--dest", str(dest), "--no-dashboard",
            "--face-threshold", "not-a-number",
        ],
    )
    assert result.exit_code != 0
    assert "must be numeric" in result.output


def test_cli_watch_face_threshold_empty_string_is_not_a_crash(monkeypatch, tmp_path):
    """`docker-compose.yml` ships `IMAGEHARBOR_FACE_THRESHOLD=""` on purpose
    -- an empty env value must resolve to "not configured", never a startup
    crash (see `_parse_face_threshold`'s docstring)."""
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    monkeypatch.setattr(_watcher, "watch", lambda **kwargs: WatchStats(passes=1))

    runner = CliRunner(env={"IMAGEHARBOR_FACE_THRESHOLD": ""})
    result = runner.invoke(
        main,
        ["watch", "--source", str(src), "--dest", str(dest), "--no-dashboard"],
    )
    assert result.exit_code == 0, result.output


@needs_face_weights
def test_cli_watch_wires_a_real_face_config(monkeypatch, tmp_path):
    """With the extra installed and weights staged, `--faces` must build a
    real `FacesConfig` (a live `FaceStore`, a loaded `Detector`/`Embedder`,
    the parsed threshold, and the library's own `--dest`) and forward it to
    `_watcher.watch` unchanged."""
    from imageharbor import watcher as _watcher
    from imageharbor.faces.store import FaceStore
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
        [
            "watch", "--source", str(src), "--dest", str(dest), "--no-dashboard",
            "--faces", "--face-model-dir", str(_FACE_MODEL_DIR),
            "--face-threshold", "0.42", "--face-recluster-threshold", "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["faces_enabled"] is True
    face_config = captured["face_config"]
    assert face_config is not None
    assert isinstance(face_config.store, FaceStore)
    assert face_config.dest == dest
    assert face_config.cluster_threshold == pytest.approx(0.42)
    assert face_config.recluster_threshold == 10
    face_config.store.close()


# ---------------------------------------------------------------------------
# dashboard flags and wiring (Task 8)
# ---------------------------------------------------------------------------


def _fake_watch_cli(monkeypatch, tmp_path):
    """Patch `_watcher.watch` so `watch` returns immediately without running
    the real (blocking) loop -- these tests are about the dashboard/CLI
    wiring around it, not about the loop itself (see test_watcher.py)."""
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9")
    dest = tmp_path / "dest"

    monkeypatch.setattr(_watcher, "watch", lambda **kwargs: WatchStats(passes=1))
    return src, dest


def test_watch_no_dashboard_starts_no_server(monkeypatch, tmp_path):
    from imageharbor.dashboard import server as dashboard_server

    src, dest = _fake_watch_cli(monkeypatch, tmp_path)

    calls = {"n": 0}
    monkeypatch.setattr(
        dashboard_server, "serve", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["watch", "--source", str(src), "--dest", str(dest), "--no-dashboard"],
    )
    assert result.exit_code == 0, result.output
    assert calls["n"] == 0


def test_watch_dashboard_port_is_accepted_and_forwarded(monkeypatch, tmp_path):
    from imageharbor.dashboard import server as dashboard_server

    src, dest = _fake_watch_cli(monkeypatch, tmp_path)

    captured = {}

    def _fake_serve(catalog, control, *, port, breaker=None, store=None, crop_dir=None, stop_event):
        captured["port"] = port
        return None  # a dashboard failure must never stop the watcher

    monkeypatch.setattr(dashboard_server, "serve", _fake_serve)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "watch", "--source", str(src), "--dest", str(dest),
            "--dashboard-port", "12345",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["port"] == 12345


def test_watch_still_runs_when_the_dashboard_port_is_already_bound(monkeypatch, tmp_path):
    """The important one: a real, already-bound port must not stop `watch`
    from organizing photos -- it degrades to a warning (see
    dashboard/server.py's module docstring: "A dashboard failure must never
    stop the watcher"). `serve()` itself already guarantees this (see
    tests/test_dashboard_server.py::
    test_serve_on_already_bound_port_returns_none_and_does_not_raise); this
    test exercises that guarantee THROUGH the CLI, using a real bound socket
    and the real (unpatched) `serve()`, so a regression in either `serve()`
    or in how `watch` calls it is caught here too.

    Mutation-tested manually: forcing the bind failure to propagate (e.g. by
    calling the socket-binding server constructor directly instead of going
    through `serve()`'s guarded bind) makes this test fail with a non-zero
    exit code instead of passing -- confirming the assertions below actually
    depend on the "never raises" contract rather than passing vacuously.
    """
    import socket

    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9")
    dest = tmp_path / "dest"

    watch_calls = {"n": 0}

    def _fake_watch(**kwargs):
        watch_calls["n"] += 1
        return WatchStats(passes=1)

    monkeypatch.setattr(_watcher, "watch", _fake_watch)

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Wildcard bind, matching what `serve()` itself binds -- see
    # test_dashboard_server.py's identical comment for why "127.0.0.1"
    # would not actually conflict on Windows.
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "watch", "--source", str(src), "--dest", str(dest),
                "--dashboard-port", str(port),
            ],
        )
        assert result.exit_code == 0, result.output
        # The watcher still ran (organized photos) despite the bind failure.
        assert watch_calls["n"] == 1
        assert "could not bind" in result.output.lower()
    finally:
        blocker.close()


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


# ---------------------------------------------------------------------------
# takeout ingest / status
# ---------------------------------------------------------------------------


def _takeout_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_takeout_ingest(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {
            "Takeout/AlbumArchive/a/2015-03-09.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9",
            "Takeout/AlbumArchive/a/2015-03-09.jpg.json": json.dumps(
                {"title": "2015-03-09.jpg",
                 "photoTakenTime": {"timestampSeconds": "1425905792"}}
            ).encode(),
        },
    )

    result = CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert "ingested 1" in result.output
    assert (dest / "2015" / "2015-03").exists()


def test_takeout_ingest_writes_a_sidecar_by_default(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(archives / "t.zip", {
        "Takeout/A/a.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9",
        "Takeout/A/a.jpg.json": json.dumps(
            {"title": "a.jpg", "photoTakenTime": {"timestampSeconds": "1425905792"}}
        ).encode(),
    })

    result = CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    assert organized.with_suffix(".json").exists()


def test_takeout_ingest_reports_missing_metadata_neutrally(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {
            "Takeout/AlbumArchive/a/2015-03-09.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9",
            "Takeout/AlbumArchive/a/2015-03-09.jpg.json": json.dumps(
                {"title": "2015-03-09.jpg",
                 "photoTakenTime": {"timestampSeconds": "1425905792"}}
            ).encode(),
            "Takeout/AlbumArchive/a/no-metadata.jpg": b"\xff\xd8\xff\xe0" + b"\x05" * 16 + b"\xff\xd9",
        },
    )

    result = CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert "1 organized without Google metadata" in result.output
    # Pins the fix rather than merely exercising the line: the old wording
    # must not reappear.
    assert "ingested without Google metadata" not in result.output


def test_takeout_ingest_dry_run_writes_nothing(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {"Takeout/a/x.jpg": b"\xff\xd8\xff\xe0" + b"\x01" * 16 + b"\xff\xd9"},
    )

    result = CliRunner().invoke(
        main,
        ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "[DRY-RUN]" in result.output
    assert not (dest / "catalog.db").exists()


def test_takeout_status(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {"Takeout/a/x.jpg": b"\xff\xd8\xff\xe0" + b"\x02" * 16 + b"\xff\xd9"},
    )
    setup = CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )
    # Assert the setup succeeded, so a failure here is diagnosed here rather
    # than surfacing indirectly as a confusing assertion failure below.
    assert setup.exit_code == 0, setup.output

    result = CliRunner().invoke(
        main, ["takeout", "status", "--catalog", str(dest / "catalog.db")]
    )
    assert result.exit_code == 0, result.output
    assert "1 archive" in result.output


def test_takeout_group_has_no_default_subcommand() -> None:
    result = CliRunner().invoke(main, ["takeout"])
    assert "ingest" in result.output
    assert "status" in result.output


_SURVEY_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01" + b"\x00" * 64


def test_takeout_survey_runs_standalone_and_prints_a_summary(tmp_path):
    """No catalog, no dest, no network -- an archive directory is enough."""
    archives = tmp_path / "archives"
    archives.mkdir()
    with zipfile.ZipFile(archives / "takeout-20260818T012414Z-2-001.zip", "w") as zf:
        zf.writestr("Takeout/Google Photos/Photos from 2019/a.jpg", _SURVEY_JPEG)
        zf.writestr("Takeout/Google Photos/Photos from 2019/x.screen", _SURVEY_JPEG)

    out_json = tmp_path / "survey.json"
    result = CliRunner().invoke(
        main, ["takeout", "survey", "--archives", str(archives), "--json", str(out_json)]
    )

    assert result.exit_code == 0, result.output
    assert "INVENTORY" in result.output
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["inventory"]["by_kind"]["image"] == 1
    assert doc["anomalies"]["misnamed_media"]["total"] == 1


def test_takeout_survey_requires_an_existing_archives_dir(tmp_path):
    result = CliRunner().invoke(
        main, ["takeout", "survey", "--archives", str(tmp_path / "nope")]
    )
    assert result.exit_code != 0


def test_takeout_survey_writes_nothing_when_json_is_omitted(tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    with zipfile.ZipFile(archives / "takeout-20260818T012414Z-2-001.zip", "w") as zf:
        zf.writestr("Takeout/Google Photos/a.jpg", _SURVEY_JPEG)

    before = sorted(p.name for p in archives.iterdir())
    result = CliRunner().invoke(main, ["takeout", "survey", "--archives", str(archives)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in archives.iterdir()) == before
