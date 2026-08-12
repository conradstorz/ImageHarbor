"""Tests for the continuous polling watcher."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import CONSECUTIVE_ABORT_WARNING_THRESHOLD, WatchStats, run_pass, watch


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


def test_watch_warns_once_after_many_consecutive_aborted_passes(
    tmp_path: Path,
    organized_dir: Path,
    catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the files tripping the breaker are (or become) the entire
    remaining unenriched queue, quarantine can never fire for them (see
    `_reconcile_poison`'s "Known, deliberate limitation" note) and the
    rotating probe offset just keeps cycling. That is bounded in cost, but
    should not be silent forever: watch() logs ONE diagnostic warning once
    the run of consecutive ABORTED passes crosses
    CONSECUTIVE_ABORT_WARNING_THRESHOLD, and must not repeat it on every
    later pass while the run continues (only on the next fresh crossing,
    which this test does not exercise).
    """
    from imageharbor import watcher

    # 50 always-failing files and NO good file anywhere -- a genuine
    # "entire remaining queue is poison" scenario, not just a head cluster.
    # The count matters: with trip_threshold=2 the offset advances by 2 per
    # abort, and iter_unenriched(offset=...) returning 0 rows resets
    # consecutive_aborted_passes back to 0 (by design -- see watch()'s
    # docstring), so the pool must stay bigger than the max offset reached
    # across this test's passes (well under 50) or the counter would never
    # climb monotonically to the threshold.
    src = _src_with(tmp_path, 50)
    pipeline = Pipeline(src, organized_dir, catalog)
    # backoff_base=0.0: the half-open probe wait is always 0, so the loop
    # never needs to actually sleep to keep cycling.
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=0.0, now=lambda: 0.0)

    stop_event = threading.Event()
    calls = {"n": 0}
    real_run_once = watcher.run_once
    # Run well past the threshold (10) before stopping, to prove the
    # warning does not repeat on every subsequent aborted pass. This counts
    # passes directly rather than relying on `sleep` being called, because
    # once the breaker trips it stays OPEN pass after pass and the loop's
    # `continue` branches skip `sleep` entirely in this scenario.
    stop_after = watcher.CONSECUTIVE_ABORT_WARNING_THRESHOLD + 5

    def _counting_run_once(*args, **kwargs):
        calls["n"] += 1
        result = real_run_once(*args, **kwargs)
        if calls["n"] >= stop_after:
            stop_event.set()
        return result

    monkeypatch.setattr(watcher, "run_once", _counting_run_once)

    with caplog.at_level("WARNING"):
        watcher.watch(
            pipeline=pipeline,
            catalog=catalog,
            source=src,
            interval=0.0,
            stop_event=stop_event,
            classifier=_AlwaysFails(),
            breaker=breaker,
        )

    assert calls["n"] >= stop_after
    progress_warnings = [
        r for r in caplog.records if "consecutive aborted passes" in r.message
    ]
    assert len(progress_warnings) == 1, (
        f"expected exactly one progress warning, got {len(progress_warnings)}: "
        f"{[r.message for r in progress_warnings]}"
    )
    assert (
        f"{watcher.CONSECUTIVE_ABORT_WARNING_THRESHOLD} consecutive"
        in progress_warnings[0].message
    )
