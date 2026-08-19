"""Tests for the continuous polling watcher."""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import (
    CONSECUTIVE_ABORT_WARNING_THRESHOLD,
    WatchStats,
    _MAX_SLEEP_SECONDS,
    _safe_sleep,
    run_pass,
    watch,
)


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
    wstats = watch(pipeline=pipeline, catalog=catalog, interval=1.0, stop_event=stop)
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

    watch(pipeline=pipeline, catalog=catalog, interval=300.0,
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

    watch(pipeline=pipeline, catalog=catalog, interval=300.0,
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


# ---------------------------------------------------------------------------
# Pause plumbing (dashboard Task 4)
# ---------------------------------------------------------------------------


def test_watch_paused_control_skips_the_pass_and_sleeps(
    source_dir: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paused watcher must sleep the interval and run NO pass at all --
    not start one and immediately break out.
    """
    from imageharbor import watcher as watcher_module
    from imageharbor.dashboard.control import ControlPlane

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    control = ControlPlane(catalog, env_interval=1.0, env_enrich=True)
    control.set_paused(True)
    stop = threading.Event()
    slept: list[float] = []
    calls = {"n": 0}

    real_run_once = watcher_module.run_once

    def _counting_run_once(*args, **kwargs):
        calls["n"] += 1
        return real_run_once(*args, **kwargs)

    monkeypatch.setattr(watcher_module, "run_once", _counting_run_once)

    def _sleep(interval: float) -> bool:
        slept.append(interval)
        stop.set()
        return True

    wstats = watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_sleep,
        control=control,
    )

    assert calls["n"] == 0           # no pass was ever started
    assert slept == [1.0]            # but the loop did sleep the interval
    assert wstats.passes == 0


# ---------------------------------------------------------------------------
# CRITICAL finding #1 (belt-and-braces): a non-finite interval reaching
# sleep() must never crash the watch loop, even if it somehow got past
# ControlPlane's own math.isfinite validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_safe_sleep_replaces_non_finite_seconds_with_the_ceiling(bad: float) -> None:
    calls: list[float] = []

    def _sleep(seconds: float) -> bool:
        calls.append(seconds)
        return True

    _safe_sleep(_sleep, bad)
    assert calls == [_MAX_SLEEP_SECONDS]


def test_safe_sleep_catches_overflow_error_from_the_underlying_sleep() -> None:
    """The pre-check (`math.isfinite` + clamping to `_MAX_SLEEP_SECONDS`)
    already prevents an out-of-range value from reaching `sleep` in every
    case this module can construct -- but the call itself is *also* wrapped
    in `try/except OverflowError`, as a second, independent guard against a
    platform whose real `Event.wait` overflows even for an in-range value
    this module considers safe. This test exercises that second guard
    directly, since nothing in this module's own clamping can trigger it.
    """
    calls: list[float] = []

    def _sleep(seconds: float) -> bool:
        calls.append(seconds)
        raise OverflowError("timestamp out of range for platform time_t")

    with pytest.raises(OverflowError):
        # The retry itself calls `sleep(_MAX_SLEEP_SECONDS)`, which this
        # fake still rejects unconditionally -- confirming _safe_sleep does
        # not swallow a *persistent* failure, only retries it once with the
        # ceiling value, exactly like the real call site is expected to.
        _safe_sleep(_sleep, 5.0)
    assert calls == [5.0, _MAX_SLEEP_SECONDS]


def test_watch_survives_a_non_finite_interval_without_crashing(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """Even with `control=None` (bypassing `ControlPlane`'s own validation
    entirely) and a `sleep` double that would raise `OverflowError` (as the
    real `Event.wait` does) if it were ever actually handed a non-finite
    value, `watch()` must not crash. `_safe_sleep` replaces the bad value
    with `_MAX_SLEEP_SECONDS` before the call reaches `sleep` at all, so the
    double's `OverflowError` branch is never triggered here -- proving the
    watcher-side guard works end-to-end through the public `watch()` entry
    point, on top of (not instead of) the control.py store-side fix.
    """
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stop = threading.Event()
    calls: list[float] = []

    def _sleep(seconds: float) -> bool:
        calls.append(seconds)
        if not math.isfinite(seconds):
            raise OverflowError("timestamp out of range for platform time_t")
        stop.set()
        return True

    # No exception escapes watch() -- this call itself is the assertion.
    wstats = watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=math.inf,
        stop_event=stop,
        sleep=_sleep,
    )
    assert wstats.passes == 1
    assert calls[-1] == _MAX_SLEEP_SECONDS


def test_watch_reads_live_interval_from_control_each_iteration(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """A dashboard interval change must take effect on the very next sleep,
    not stay frozen at whatever `watch()` was started with.

    If `watch` only ever read `control.interval` once before the loop (instead
    of on each iteration), the second sleep below would still see 1.0, not
    the 9.0 written mid-loop.
    """
    from imageharbor.dashboard.control import ControlPlane

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    control = ControlPlane(catalog, env_interval=1.0, env_enrich=True)
    stop = threading.Event()
    slept: list[float] = []

    def _sleep(interval: float) -> bool:
        slept.append(interval)
        if len(slept) == 1:
            # A dashboard edit lands between the first and second pass.
            control.set_override("interval", 9.0)
        else:
            stop.set()
        return True

    watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_sleep,
        control=control,
    )

    assert slept == [1.0, 9.0]


def test_watch_control_enrich_enabled_read_live(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """enrich_enabled must also be read from `control` each iteration -- a
    dashboard toggle that only took effect at startup would silently defeat
    that dial exactly the way a frozen interval would.
    """
    from imageharbor.dashboard.control import ControlPlane

    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    control = ControlPlane(catalog, env_interval=1.0, env_enrich=True)
    control.set_override("enrich", False)

    describe_calls = {"n": 0}

    class _CountingClassifier(_AlwaysFails):
        def describe(self, image_path, exif_data):
            describe_calls["n"] += 1
            return super().describe(image_path, exif_data)

    stop = threading.Event()

    def _sleep(interval: float) -> bool:
        stop.set()
        return True

    watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_sleep,
        classifier=_CountingClassifier(),
        control=control,
    )

    assert describe_calls["n"] == 0  # enrichment never ran: control says off


def test_watch_control_none_behaves_exactly_as_before(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """No `control` supplied -> the existing `interval`/`enrich_enabled`
    parameters govern the loop exactly as they did before this feature.
    """
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stop = threading.Event()

    def _sleep(_interval: float) -> bool:
        stop.set()
        return True

    wstats = watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_sleep,
    )
    assert wstats.passes == 1


# ---------------------------------------------------------------------------
# The watcher records each pass in `runs` (dashboard Task 7)
# ---------------------------------------------------------------------------


def test_run_once_writes_a_facts_run_row_with_counts(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """A completed facts pass writes a closed `runs` row whose counts match
    the `WatchStats` the pass actually produced."""
    from imageharbor import watcher as watcher_module

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    facts, enrich_stats = watcher_module.run_once(
        source_dir, organized_dir, catalog, classifier=None, pipeline=pipeline,
    )

    assert enrich_stats is None  # no classifier -> no enrichment phase at all
    rows = catalog.recent_runs(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "facts"
    assert row["ended_at"] is not None
    assert row["scanned"] == facts.processed + facts.skipped_unchanged + facts.errors
    assert row["copied"] == facts.copied == 2
    assert row["duplicates"] == facts.duplicates == 0
    assert row["errors"] == facts.errors == 0
    assert row["enriched"] == 0
    assert row["enrich_failed"] == 0
    assert row["breaker_state"] == "closed"
    assert row["paused"] == 0


def test_run_once_writes_an_enrich_run_row_when_enrichment_runs(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    from imageharbor.ai_classifier import StubClassifier
    from imageharbor import watcher as watcher_module

    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    facts, enrich_stats = watcher_module.run_once(
        src, organized_dir, catalog, classifier=StubClassifier(), pipeline=pipeline,
    )

    assert enrich_stats is not None
    rows = {row["kind"]: row for row in catalog.recent_runs(limit=5)}
    assert set(rows) == {"facts", "enrich"}
    enrich_row = rows["enrich"]
    assert enrich_row["ended_at"] is not None
    assert enrich_row["scanned"] == enrich_stats.total
    assert enrich_row["enriched"] == enrich_stats.enriched
    assert enrich_row["enrich_failed"] == enrich_stats.errors
    assert enrich_row["copied"] == 0
    assert enrich_row["duplicates"] == 0
    assert enrich_row["breaker_state"] == "closed"
    assert enrich_row["paused"] == 0


def test_run_once_writes_no_enrich_row_when_the_breaker_is_open(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """A phase that is SKIPPED (breaker OPEN) must not produce a row at all --
    a row means a pass actually happened."""
    from imageharbor.circuit_breaker import CircuitBreaker
    from imageharbor import watcher as watcher_module

    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    breaker = CircuitBreaker(trip_threshold=1, backoff_base=1.0, backoff_cap=1.0)
    breaker.record_failure()
    assert breaker.is_open()

    watcher_module.run_once(
        src, organized_dir, catalog,
        classifier=_AlwaysFails(), pipeline=pipeline, breaker=breaker,
    )

    rows = catalog.recent_runs(limit=5)
    assert [row["kind"] for row in rows] == ["facts"]
    assert rows[0]["breaker_state"] == "open"


def test_run_finish_records_the_breaker_state_at_pass_end(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The facts row is closed BEFORE the breaker can trip (the facts phase
    never touches it); the enrich row is closed AFTER, so it must reflect
    the breaker's state as of the END of that phase, not its state when the
    phase started."""
    from imageharbor.circuit_breaker import CircuitBreaker
    from imageharbor import watcher as watcher_module

    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    breaker = CircuitBreaker(trip_threshold=1, backoff_base=1.0, backoff_cap=1.0)
    assert not breaker.is_open()

    watcher_module.run_once(
        src, organized_dir, catalog,
        classifier=_AlwaysFails(), pipeline=pipeline, breaker=breaker,
    )

    assert breaker.is_open()  # the single failure tripped it during enrichment
    rows = {row["kind"]: row for row in catalog.recent_runs(limit=5)}
    assert rows["facts"]["breaker_state"] == "closed"
    assert rows["enrich"]["breaker_state"] == "open"


def test_pause_mid_facts_pass_records_paused_flag(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """A pause that lands WHILE the facts phase is running must close that
    pass's `runs` row with `paused=1` -- not leave it looking like a clean,
    unpaused completion.
    """
    from imageharbor import watcher as watcher_module

    pipeline = Pipeline(source_dir, organized_dir, catalog)
    calls = {"n": 0}

    def _pause_after_one_file() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    facts, _ = watcher_module.run_once(
        source_dir, organized_dir, catalog,
        classifier=None, pipeline=pipeline, pause_check=_pause_after_one_file,
    )

    assert facts.processed == 1  # stopped after the first file, not both
    rows = catalog.recent_runs(limit=5)
    assert len(rows) == 1
    assert rows[0]["kind"] == "facts"
    assert rows[0]["ended_at"] is not None
    assert rows[0]["paused"] == 1


def test_pause_mid_enrich_pass_records_paused_flag(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imageharbor.ai_classifier import StubClassifier
    from imageharbor import watcher as watcher_module

    src = _src_with(tmp_path, 2)
    pipeline = Pipeline(src, organized_dir, catalog)
    # `pause_check` must be idempotent/side-effect-free in real use (the real
    # ControlPlane.pause_check() is a bare attribute read), so the pause
    # transition here is driven by real state, not by counting pause_check
    # calls -- run_once's own `finally` blocks re-read pause_check once more
    # after each phase (to decide that phase's `paused` column), and a
    # call-counting fake would be thrown off by those extra reads.
    state = {"paused": False}

    def _pause_check() -> bool:
        return state["paused"]

    real_enrich_library = watcher_module.enrich_library

    def _flip_paused_then_enrich(*args, **kwargs):
        # The facts phase (and its `runs` row) has already fully finished by
        # the time this wrapper runs, so flipping here pauses the enrichment
        # phase from its very first row onward without touching the facts
        # row's paused=0 result.
        state["paused"] = True
        return real_enrich_library(*args, **kwargs)

    monkeypatch.setattr(watcher_module, "enrich_library", _flip_paused_then_enrich)

    facts, enrich_stats = watcher_module.run_once(
        src, organized_dir, catalog,
        classifier=StubClassifier(), pipeline=pipeline,
        pause_check=_pause_check,
    )

    assert facts.processed == 2       # facts phase was not paused
    assert enrich_stats.total == 0    # enrichment paused before its first row
    rows = {row["kind"]: row for row in catalog.recent_runs(limit=5)}
    assert rows["facts"]["paused"] == 0
    assert rows["enrich"]["paused"] == 1


def test_facts_pass_that_raises_still_closes_its_row(
    source_dir: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_finish` must run in a `finally`: a crash mid-facts-pass must not
    leave `ended_at` NULL forever (the dashboard's signal that a previous
    run died outright) -- and the crash itself counts as one error.
    """
    from imageharbor import watcher as watcher_module

    pipeline = Pipeline(source_dir, organized_dir, catalog)

    def _boom(*args, **kwargs):
        raise RuntimeError("catalog write failed")

    monkeypatch.setattr(catalog, "record_source_seen", _boom)

    with pytest.raises(RuntimeError):
        watcher_module.run_once(
            source_dir, organized_dir, catalog, classifier=None, pipeline=pipeline,
        )

    rows = catalog.recent_runs(limit=5)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "facts"
    assert row["ended_at"] is not None       # closed, not left open
    assert row["errors"] == 1                # the crash itself
    assert row["copied"] == 0                # nothing had completed yet


def test_enrich_pass_that_raises_still_closes_its_row(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imageharbor.ai_classifier import StubClassifier
    from imageharbor import watcher as watcher_module

    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)

    def _boom(*args, **kwargs):
        raise RuntimeError("enrichment crashed")

    monkeypatch.setattr(watcher_module, "enrich_library", _boom)

    with pytest.raises(RuntimeError):
        watcher_module.run_once(
            src, organized_dir, catalog, classifier=StubClassifier(), pipeline=pipeline,
        )

    rows = {row["kind"]: row for row in catalog.recent_runs(limit=5)}
    assert rows["facts"]["ended_at"] is not None    # facts phase completed cleanly
    assert rows["enrich"]["ended_at"] is not None    # closed despite the crash
    # IMPORTANT finding #7: `errors` on a `runs` row means "facts-phase
    # errors"; an 'enrich'-kind row has no facts phase and must always be 0
    # here -- the crash itself is recorded ONLY in `enrich_failed`, never in
    # both (that double-write is what finding #7 removed: it made the
    # dashboard history panel's 24h error figure count every enrichment
    # failure twice).
    assert rows["enrich"]["errors"] == 0
    assert rows["enrich"]["enrich_failed"] == 1
    assert rows["enrich"]["enriched"] == 0


def test_watch_server_survives_a_crashed_pass_and_keeps_reporting(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The moment an operator most wants the dashboard is exactly when a pass
    is failing. A crashed pass must not kill the watch loop (which would
    freeze `current_run`/`last_run` forever) or the dashboard server thread,
    and `/api/stats` must keep returning real data afterward.
    """
    import http.client
    import json
    import socket

    from imageharbor.dashboard.control import ControlPlane
    from imageharbor.dashboard.server import serve

    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    control = ControlPlane(catalog, env_interval=1.0, env_enrich=False)

    # Reserve a real, free port up front (rather than binding it inside
    # `serve` with port=0) so the test knows which port to connect to.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("0.0.0.0", 0))
    port = probe.getsockname()[1]
    probe.close()

    # Two DISTINCT events: `watch()` must stop after two passes, but the
    # dashboard server must stay up afterward so the test can still query
    # it -- sharing one event would shut the server down (via `serve()`'s
    # own stop-watcher thread) at the exact moment `watch()` stops, before
    # the assertions below ever get to make a request.
    watch_stop = threading.Event()
    server_stop = threading.Event()
    server_thread = serve(catalog, control, port=port, stop_event=server_stop)
    assert server_thread is not None

    real_record_source_seen = catalog.record_source_seen
    calls = {"n": 0}

    def _boom_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash on the first pass")
        return real_record_source_seen(*args, **kwargs)

    monkeypatch.setattr(catalog, "record_source_seen", _boom_once)

    passes_done = {"n": 0}

    def _sleep(_interval: float) -> bool:
        passes_done["n"] += 1
        if passes_done["n"] >= 2:
            watch_stop.set()
        return True

    try:
        wstats = watch(
            pipeline=pipeline,
            catalog=catalog,
            interval=1.0,
            stop_event=watch_stop,
            sleep=_sleep,
            control=control,
        )

        assert wstats.passes == 2   # the crashed pass, plus the one after it
        assert calls["n"] >= 2      # the loop did not stop after the crash

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/api/stats")
            resp = conn.getresponse()
            body = json.loads(resp.read())
        finally:
            conn.close()
        assert resp.status == 200
        assert body["now"] is not None   # the page still reports
    finally:
        server_stop.set()
        server_thread.join(timeout=5)
