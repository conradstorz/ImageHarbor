"""Tests for the continuous polling watcher."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
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


class _AlwaysFails:
    """Classifier whose describe() always raises — simulates a dead backend."""
    def describe(self, image_path, exif_data):
        raise RuntimeError("backend down")
    def adjudicate(self, label, candidates):
        return None
    def pick_class(self, content, classes):
        return "900"


def _src_with(tmp_path: Path, n: int) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        _make_jpeg(src / f"img_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")
    return src


def test_run_pass_aborts_remaining_files_when_breaker_trips(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 5)
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AlwaysFails())
    breaker = CircuitBreaker(trip_threshold=2, now=lambda: 0.0)
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker)
    # 2 failures trip the breaker; the pass aborts before the other 3 files.
    assert stats.errors == 2
    assert breaker.is_open()


def test_run_pass_half_open_failure_tries_only_one_file(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 5)
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AlwaysFails())
    clock = [0.0]
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=60.0, now=lambda: clock[0])
    breaker.record_failure(); breaker.record_failure()   # OPEN
    clock[0] = 60.0
    breaker.begin_probe()                                  # HALF_OPEN
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker)
    assert stats.errors == 1          # only the probe file was tried
    assert breaker.is_open()          # probe failed -> reopened


def test_run_pass_half_open_success_resumes_full_pass(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 3)
    pipeline = Pipeline(src, organized_dir, catalog)   # StubClassifier: all succeed
    clock = [0.0]
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=60.0, now=lambda: clock[0])
    breaker.record_failure(); breaker.record_failure()
    clock[0] = 60.0
    breaker.begin_probe()                               # HALF_OPEN
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker)
    assert stats.processed == 3                         # probe closed it, rest ran
    assert breaker.state.name == "CLOSED"


def test_watch_sleeps_breaker_backoff_when_open(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    clock = [1000.0]
    breaker = CircuitBreaker(trip_threshold=1, backoff_base=60.0, now=lambda: clock[0])
    breaker.record_failure()          # OPEN at t=1000, backoff=60
    stop = threading.Event()
    slept: list[float] = []

    def _sleep(d: float) -> bool:
        slept.append(d)
        stop.set()                    # exit after first sleep
        return True

    watch(pipeline=pipeline, catalog=catalog, source=src, interval=300.0,
          stop_event=stop, sleep=_sleep, breaker=breaker)
    assert slept and abs(slept[0] - 60.0) < 1.0   # slept the backoff, not the interval


def test_watch_probe_uses_backoff_not_interval_after_midpass_trip(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    # A pass that trips the breaker mid-run must NOT then sleep the full poll
    # interval; the next wait should be the (short) backoff so the recovery
    # probe fires promptly.
    src = _src_with(tmp_path, 3)
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AlwaysFails())
    clock = [1000.0]
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=60.0, now=lambda: clock[0])
    stop = threading.Event()
    slept: list[float] = []

    def _sleep(d: float) -> bool:
        slept.append(d)
        clock[0] += d          # advance clock so backoff elapses realistically
        stop.set()             # stop after observing the first between-pass wait
        return True

    watch(pipeline=pipeline, catalog=catalog, source=src, interval=300.0,
          stop_event=stop, sleep=_sleep, breaker=breaker, poison_max_fails=5)
    # First pass trips (2 fails). The observed wait must be the ~60s backoff,
    # not the 300s interval.
    assert slept and abs(slept[0] - 60.0) < 1.0
