"""Tests for poison-file quarantine in the watcher.

Poison-quarantine is driven entirely by the ENRICHMENT phase now: the facts
phase makes no AI calls, so it can never produce the failure signal quarantine
depends on. Quarantine means "stop asking the model about this one", not "set
this file aside" -- a file that can never be described is still fully
organized, verified, and catalogued by the facts phase.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.hashing import compute_sha256_b64url, verify_file
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import run_once, watch


def _make_jpeg(path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9") -> Path:
    path.write_bytes(content)
    return path


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


class _FailsForContent:
    """StubClassifier-like backend that raises only for named byte content.

    Enrichment reads the ORGANIZED copy, whose filename is date/descriptor/
    digest derived -- not the original name -- so which file should fail must
    be selected by content, not by name.
    """
    def __init__(self, bad_content: set[bytes]) -> None:
        self._bad = bad_content

    def describe(self, image_path, exif_data):
        if image_path.read_bytes() in self._bad:
            raise RuntimeError("cannot decode")
        from imageharbor.ai_classifier import StubClassifier
        return StubClassifier().describe(image_path, exif_data)

    def adjudicate(self, label, candidates):
        return None

    def pick_class(self, content, classes):
        return "900"


class _AllFail:
    """Classifier whose describe() always raises -- simulates a dead backend."""
    def describe(self, image_path, exif_data):
        raise RuntimeError("backend down")

    def adjudicate(self, label, candidates):
        return None

    def pick_class(self, content, classes):
        return "900"


def _fresh_breaker() -> CircuitBreaker:
    # High threshold so a single poison file never trips it in these tests.
    return CircuitBreaker(trip_threshold=100, now=lambda: 0.0)


def test_poison_file_quarantined_after_k_healthy_enrichment_passes(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    bad_bytes = b"\xff\xd8\xff\xe0" + b"\x07" * 16 + b"\xff\xd9"
    bad = _make_jpeg(src / "bad.jpg", bad_bytes)
    classifier = _FailsForContent({bad_bytes})
    breaker = _fresh_breaker()

    for i in range(4):
        # Fresh unseen success each pass proves the backend is up (healthy-pass gate).
        _make_jpeg(src / f"good_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i + 1]) * 16 + b"\xff\xd9")
        run_once(src, organized_dir, catalog, classifier=classifier, breaker=breaker,
                  poison_max_fails=5)
        assert catalog.is_quarantined(
            str(bad), bad.stat().st_size, bad.stat().st_mtime_ns
        ) is False

    _make_jpeg(src / "good_final.jpg", b"\xff\xd8\xff\xe0" + b"\x63" * 16 + b"\xff\xd9")
    run_once(src, organized_dir, catalog, classifier=classifier, breaker=breaker,
              poison_max_fails=5)   # 5th healthy-pass failure -> quarantine
    assert catalog.is_quarantined(str(bad), bad.stat().st_size, bad.stat().st_mtime_ns)


def test_systemic_outage_does_not_quarantine(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    a = _make_jpeg(src / "a.jpg")
    _make_jpeg(src / "b.jpg", b"\xff\xd8\xff\xe0" + b"\x02" * 16 + b"\xff\xd9")

    breaker = CircuitBreaker(trip_threshold=2, now=lambda: 0.0)
    for _ in range(10):
        run_once(src, organized_dir, catalog, classifier=_AllFail(), breaker=breaker,
                  poison_max_fails=1)
    # Every enrichment attempt fails: either the breaker trips this pass (its
    # failures are discarded outright) or, once OPEN, run_once skips the
    # enrichment phase entirely -- either way NOTHING is ever quarantined.
    assert catalog.is_quarantined(str(a), a.stat().st_size, a.stat().st_mtime_ns) is False


def test_poison_at_the_head_does_not_halt_enrichment(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The mirror image of test_systemic_outage_does_not_quarantine: this is
    NOT a systemic outage (most content is describable), but a cluster of
    genuinely-undescribable files that happens to sort to the very head of
    `iter_unenriched`'s fixed `ORDER BY p.id` queue.

    Pre-fix, this is a livelock: every enrichment pass re-hits the same head
    cluster first, trips the breaker, aborts, and `_reconcile_poison`
    discards the whole pass's failures unconditionally on `tripped` -- so
    nothing behind the cluster (good or poison) is EVER reached, and
    `fail_count` never climbs to the quarantine threshold either. The fix is
    a rotating probe offset (owned by `watch`) that skips a whole cluster
    after an abort, so a half-open probe eventually lands on a working file
    elsewhere in the queue, closes the breaker, and lets the pass proceed.

    Layout (alphabetical == id order, all present before the first pass):
      a0/a1 trigger  -- two always-failing files, exactly `trip_threshold`
                        long, contiguous at the head. These cause the very
                        first trip; a pass that touches both together is
                        always discarded, and a half-open probe landing on
                        either alone reopens on that single failure -- so
                        this pair can never itself be quarantined, and the
                        test does not assert on it.
      b0 good, b1 target, b2 good, b3 target, b4 good, b5 target, b6 good
                     -- alternating tail: every target poison file has a
                        good neighbour on both sides, so once a probe's
                        offset clears the trigger pair and lands on b0, the
                        pass proceeds through the whole tail without
                        re-tripping (no two failures ever land back to
                        back) -- and every target file in it fails during
                        that one HEALTHY pass, which is exactly the
                        condition `_reconcile_poison` requires to count a
                        failure toward quarantine.
    """
    src = tmp_path / "src"
    src.mkdir()

    trigger_contents = [
        b"\xff\xd8\xff\xe0" + bytes([0xA0 + i]) * 16 + b"\xff\xd9" for i in range(2)
    ]
    for i, content in enumerate(trigger_contents):
        _make_jpeg(src / f"a{i}_trigger.jpg", content)

    target_contents = [
        b"\xff\xd8\xff\xe0" + bytes([0xB0 + i]) * 16 + b"\xff\xd9" for i in range(3)
    ]
    good_paths: list[Path] = []
    for i, content in enumerate(target_contents):
        good_path = _make_jpeg(
            src / f"b{2 * i}_good.jpg",
            b"\xff\xd8\xff\xe0" + bytes([0xC0 + i]) * 16 + b"\xff\xd9",
        )
        good_paths.append(good_path)
        _make_jpeg(src / f"b{2 * i + 1}_target.jpg", content)
    final_good = _make_jpeg(
        src / "b6_good.jpg", b"\xff\xd8\xff\xe0" + b"\xcf" * 16 + b"\xff\xd9"
    )
    good_paths.append(final_good)

    classifier = _FailsForContent(set(trigger_contents) | set(target_contents))
    # backoff_base=0.0 means the breaker's probe wait is always 0, so the
    # test does not need to fake real elapsed time to reach a half-open
    # probe -- only `sleep`/`stop_event` need faking, to bound the loop.
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=0.0, now=lambda: 0.0)
    pipeline = Pipeline(src, organized_dir, catalog)

    stop_event = threading.Event()
    passes = {"n": 0}

    def fake_sleep(_seconds: float) -> bool:
        passes["n"] += 1
        if passes["n"] >= 10:
            stop_event.set()
        return False

    # Pre-fix, this scenario is a genuine infinite loop: with the offset
    # stuck at 0, every half-open probe re-hits the trigger pair's first
    # file, reopens on that single failure, and `continue`s straight back
    # to the top without ever calling `sleep` -- so `fake_sleep`'s own
    # pass-count cap is never reached either. A wall-clock watchdog is the
    # only thing that can end that loop, which is exactly why this
    # regression matters: nothing internal to the loop bounds it.
    watchdog = threading.Timer(5.0, stop_event.set)
    watchdog.start()
    try:
        watch(
            pipeline=pipeline,
            catalog=catalog,
            source=src,
            interval=0.0,
            stop_event=stop_event,
            sleep=fake_sleep,
            classifier=classifier,
            breaker=breaker,
            poison_max_fails=1,
        )
    finally:
        watchdog.cancel()

    # The good files were never poison and must not be held hostage by the
    # trigger cluster ahead of them in id order.
    for good_path in good_paths:
        digest = compute_sha256_b64url(good_path)
        row = catalog.get_by_sha256(digest)
        assert row is not None
        assert row["enriched_at"] is not None, (
            f"{good_path.name} should have been enriched"
        )

    # The target poison files -- distinct from the un-quarantinable trigger
    # pair -- must eventually reach quarantine: they can only do so by
    # failing during a pass that also has a success, which requires
    # enrichment to make it past the head cluster at all.
    for i in range(3):
        target = src / f"b{2 * i + 1}_target.jpg"
        assert catalog.is_quarantined(
            str(target), target.stat().st_size, target.stat().st_mtime_ns
        ), f"{target.name} should have been quarantined"


def test_quarantine_copies_to_dir_when_set(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    bad_bytes = b"\xff\xd8\xff\xe0" + b"\x09" * 16 + b"\xff\xd9"
    _make_jpeg(src / "bad.jpg", bad_bytes)
    qdir = tmp_path / "quarantine"
    classifier = _FailsForContent({bad_bytes})
    breaker = _fresh_breaker()
    for i in range(3):
        _make_jpeg(src / f"good_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i + 1]) * 16 + b"\xff\xd9")
        run_once(src, organized_dir, catalog, classifier=classifier, breaker=breaker,
                  poison_max_fails=3, quarantine_dir=qdir)
    copied = list(qdir.glob("*_bad.jpg"))
    assert len(copied) == 1


def test_quarantined_file_remains_organized_and_verifiable(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """Quarantine means 'stop asking the model about this one', not 'set this
    file aside': a file that can never be described is still fully organized,
    verified, and catalogued by the facts phase -- quarantine must not remove
    or relocate it."""
    src = tmp_path / "src"
    src.mkdir()
    bad_bytes = b"\xff\xd8\xff\xe0" + b"\x11" * 16 + b"\xff\xd9"
    bad = _make_jpeg(src / "bad.jpg", bad_bytes)
    classifier = _FailsForContent({bad_bytes})
    breaker = _fresh_breaker()

    for i in range(3):
        _make_jpeg(src / f"good_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i + 20]) * 16 + b"\xff\xd9")
        run_once(src, organized_dir, catalog, classifier=classifier, breaker=breaker,
                  poison_max_fails=3)

    assert catalog.is_quarantined(str(bad), bad.stat().st_size, bad.stat().st_mtime_ns)

    digest = compute_sha256_b64url(bad)
    row = catalog.get_by_sha256(digest)
    assert row is not None, "the facts phase must still have catalogued the file"
    organized_path = Path(row["organized_path"])
    assert organized_path.exists(), "quarantine must not remove the organized copy"
    assert verify_file(organized_path, digest), "organized copy must still verify"


def test_a_quarantined_file_is_never_described_again(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """Quarantine means "stop asking the model about this one".

    The bookkeeping (`failed_files.quarantined = 1`) can look right while the
    model is still being hit every pass -- that is the entire failure mode
    this test exists to catch, so it asserts on the AI-call count, never on a
    status/flag.
    """
    src = tmp_path / "src"
    src.mkdir()
    bad_bytes = b"\xff\xd8\xff\xe0" + b"\x42" * 16 + b"\xff\xd9"
    bad = _make_jpeg(src / "bad.jpg", bad_bytes)
    breaker = _fresh_breaker()

    calls: list[str] = []

    class Counting(_FailsForContent):
        """Wraps _FailsForContent to record a call only when it is actually
        asked about the poison content -- new good files introduced on the
        final pass must not be able to pollute the count."""
        def describe(self, image_path, exif_data=None):
            if image_path.read_bytes() in self._bad:
                calls.append(image_path.name)
            return super().describe(image_path, exif_data)

    classifier = Counting({bad_bytes})

    # Drive passes until bad.jpg is quarantined (poison_max_fails=3).
    for i in range(3):
        _make_jpeg(src / f"good_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i + 1]) * 16 + b"\xff\xd9")
        run_once(src, organized_dir, catalog, classifier=classifier, breaker=breaker,
                  poison_max_fails=3)
    assert catalog.is_quarantined(str(bad), bad.stat().st_size, bad.stat().st_mtime_ns)
    # Sanity: it really was being described (and failing) on every pass so far,
    # so a flat "still zero" count below wouldn't just mean the double is unused.
    assert len(calls) == 3

    before = len(calls)
    _make_jpeg(src / "good_extra.jpg", b"\xff\xd8\xff\xe0" + b"\x99" * 16 + b"\xff\xd9")
    run_once(src, organized_dir, catalog, classifier=classifier, breaker=breaker,
              poison_max_fails=3)
    assert len(calls) == before  # not one further AI call for the quarantined file
