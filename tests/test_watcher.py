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


# NOTE: run_pass no longer takes a breaker at all (the facts phase makes no
# AI calls, so there is nothing for it to feed the breaker), so the three
# tests that used to exercise breaker-tripping/half-open behavior *during the
# facts pass* (via Pipeline(classifier=_AlwaysFails())) are obsolete -- that
# mechanism doesn't exist anymore. The equivalent properties now live at the
# run_once/watch level:
#   - test_facts_phase_runs_even_when_the_breaker_is_open (below) covers "the
#     facts phase runs regardless of breaker state".
#   - test_watch_probe_uses_backoff_not_interval_after_midpass_trip (below,
#     rewritten) covers "a mid-pass trip still yields a backoff wait, not the
#     full interval" -- now tripped by the enrichment phase.


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
    # probe fires promptly. The trip now comes from the ENRICHMENT phase (the
    # facts phase has no AI to fail), fed via watch()'s classifier= param
    # rather than through Pipeline (which no longer accepts a classifier).
    src = _src_with(tmp_path, 3)
    pipeline = Pipeline(src, organized_dir, catalog)
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
          stop_event=stop, sleep=_sleep, classifier=_AlwaysFails(), breaker=breaker,
          poison_max_fails=5)
    # Facts organizes all 3 files; enrichment then trips (2 fails) on them.
    # The observed wait must be the ~60s backoff, not the 300s interval.
    assert slept and abs(slept[0] - 60.0) < 1.0


def test_facts_phase_runs_even_when_the_breaker_is_open(tmp_path, monkeypatch):
    """A dead AI backend must not stop the library being organized."""
    from imageharbor.catalog import Catalog
    from imageharbor.circuit_breaker import CircuitBreaker
    from imageharbor import watcher

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    breaker = CircuitBreaker(trip_threshold=1, backoff_base=1.0, backoff_cap=1.0)
    breaker.record_failure()
    assert breaker.is_open()

    calls = {"enrich": 0}

    def fake_enrich(*args, **kwargs):
        calls["enrich"] += 1
        from imageharbor.enrich import EnrichStats

        return EnrichStats()

    monkeypatch.setattr(watcher, "enrich_library", fake_enrich)

    with Catalog(tmp_path / "c.db") as cat:
        watcher.run_once(src, dest, cat, classifier=None, breaker=breaker)

    assert (dest / "2019" / "2019-07").exists()
    assert calls["enrich"] == 0  # breaker open -> enrichment skipped


def test_enrich_phase_runs_after_the_facts_phase(tmp_path, monkeypatch):
    from imageharbor.catalog import Catalog
    from imageharbor.ai_classifier import StubClassifier
    from imageharbor import watcher

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    order = []
    real_enrich = watcher.enrich_library

    def tracking_enrich(*args, **kwargs):
        order.append("enrich")
        return real_enrich(*args, **kwargs)

    monkeypatch.setattr(watcher, "enrich_library", tracking_enrich)

    with Catalog(tmp_path / "c.db") as cat:
        watcher.run_once(src, dest, cat, classifier=StubClassifier(), breaker=None)

    assert order == ["enrich"]
    assert (dest / "2019" / "2019-07").exists()
