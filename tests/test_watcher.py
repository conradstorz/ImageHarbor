"""Tests for the continuous polling watcher."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import WatchStats, run_pass, watch


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
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


def test_run_pass_processes_new_files(source_dir: Path, organized_dir: Path, catalog: Catalog) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)
    assert stats.processed == 2
    assert stats.skipped_unchanged == 0
    assert stats.errors == 0


def test_run_pass_second_pass_skips_unchanged_without_hashing(
    source_dir: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    # On the second pass nothing changed: process_file must NOT be called.
    calls = []
    real_process = pipeline.process_file

    def _spy(path):
        calls.append(path)
        return real_process(path)

    monkeypatch.setattr(pipeline, "process_file", _spy)
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    assert calls == []  # unchanged files never re-processed / re-hashed
    assert stats.skipped_unchanged == 2
    assert stats.processed == 0


def test_run_pass_reprocesses_changed_file(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    # Change one file's bytes (new content -> new size/mtime).
    target = source_dir / "beach_photo.jpg"
    _make_jpeg(target, b"\xff\xd8\xff\xe0" + b"\x02" * 40 + b"\xff\xd9")

    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)
    assert stats.processed == 1
    assert stats.skipped_unchanged == 1


def test_watch_runs_one_pass_then_stops(source_dir: Path, organized_dir: Path, catalog: Catalog) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stop = threading.Event()

    # Fake sleep sets the stop event so the loop exits after exactly one pass.
    def _sleep(_interval: float) -> bool:
        stop.set()
        return True

    wstats = watch(
        pipeline=pipeline,
        catalog=catalog,
        source=source_dir,
        interval=1.0,
        stop_event=stop,
        sleep=_sleep,
    )
    assert wstats.passes == 1


def test_run_pass_counts_stat_race_as_error_without_crashing(
    source_dir: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a file that vanished between discovery and stat() (common on a
    # networked SMB/CIFS mount): discover_images yields a Path that no longer
    # exists, so path.stat() raises FileNotFoundError.
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    missing = source_dir / "gone_before_stat.jpg"

    def _fake_discover_images(source, recursive=True):
        yield missing

    monkeypatch.setattr("imageharbor.watcher.discover_images", _fake_discover_images)

    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    assert stats.errors == 1
    assert stats.processed == 0
    assert stats.skipped_unchanged == 0


def test_run_pass_counts_source_unavailable_as_error_without_crashing(
    source_dir: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate the whole source root / mount dropping mid-pass: discover_images
    # itself raises OSError (e.g. a dropped SMB mount), not just a single missing
    # file. run_pass must count it and return, not crash the watcher. Use a bare
    # OSError (not FileNotFoundError) to prove the broadened catch.
    pipeline = Pipeline(source_dir, organized_dir, catalog)

    def _boom_discover_images(source, recursive=True):
        raise OSError("mount dropped")
        yield  # pragma: no cover - makes this a generator like the real one

    monkeypatch.setattr("imageharbor.watcher.discover_images", _boom_discover_images)

    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    assert stats.errors == 1
    assert stats.processed == 0
    assert stats.skipped_unchanged == 0


def test_watch_exits_immediately_if_stop_already_set(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stop = threading.Event()
    stop.set()
    wstats = watch(pipeline=pipeline, catalog=catalog, source=source_dir, interval=1.0, stop_event=stop)
    assert wstats.passes == 0
