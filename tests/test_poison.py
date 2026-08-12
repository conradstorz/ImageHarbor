"""Tests for poison-file quarantine in the watcher.

Poison-quarantine is driven entirely by the ENRICHMENT phase now: the facts
phase makes no AI calls, so it can never produce the failure signal quarantine
depends on. Quarantine means "stop asking the model about this one", not "set
this file aside" -- a file that can never be described is still fully
organized, verified, and catalogued by the facts phase.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.hashing import compute_sha256_b64url, verify_file
from imageharbor.watcher import run_once


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
