"""Tests for poison-file quarantine in the watcher."""
from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import run_pass


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


class _FailsFor:
    """StubClassifier-like backend that raises only for named files."""
    def __init__(self, bad_names: set[str]) -> None:
        self._bad = bad_names
    def describe(self, image_path, exif_data):
        if image_path.name in self._bad:
            raise RuntimeError("cannot decode")
        from imageharbor.ai_classifier import StubClassifier
        return StubClassifier().describe(image_path, exif_data)
    def adjudicate(self, label, candidates):
        return None
    def pick_class(self, content, classes):
        return "900"


def _fresh_breaker() -> CircuitBreaker:
    # High threshold so a single poison file never trips it in these tests.
    return CircuitBreaker(trip_threshold=100, now=lambda: 0.0)


def test_poison_file_quarantined_after_k_healthy_passes(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "good.jpg")
    _make_jpeg(src / "bad.jpg", b"\xff\xd8\xff\xe0" + b"\x07" * 16 + b"\xff\xd9")
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_FailsFor({"bad.jpg"}))
    breaker = _fresh_breaker()

    for _ in range(4):
        # 'good.jpg' succeeds each pass -> pass_had_success -> 'bad.jpg' counts.
        # Touch mtime so 'good.jpg' is re-seen? No: good is recorded seen after
        # pass 1, but bad.jpg is retried every pass (never recorded). We need a
        # success in EVERY pass, so re-create good.jpg unseen each pass:
        _make_jpeg(src / f"good_{_}.jpg", b"\xff\xd8\xff\xe0" + bytes([_ + 1]) * 16 + b"\xff\xd9")
        run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
                 poison_max_fails=5)
        assert catalog.is_quarantined(str(src / "bad.jpg"),
                                      (src / "bad.jpg").stat().st_size,
                                      (src / "bad.jpg").stat().st_mtime_ns) is False

    _make_jpeg(src / "good_final.jpg", b"\xff\xd8\xff\xe0" + b"\x63" * 16 + b"\xff\xd9")
    run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
             poison_max_fails=5)   # 5th healthy-pass failure -> quarantine
    bad = src / "bad.jpg"
    assert catalog.is_quarantined(str(bad), bad.stat().st_size, bad.stat().st_mtime_ns)


def test_systemic_outage_does_not_quarantine(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "a.jpg")
    _make_jpeg(src / "b.jpg", b"\xff\xd8\xff\xe0" + b"\x02" * 16 + b"\xff\xd9")

    class _AllFail:
        def describe(self, image_path, exif_data):
            raise RuntimeError("backend down")
        def adjudicate(self, label, candidates):
            return None
        def pick_class(self, content, classes):
            return "900"

    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AllFail())
    breaker = CircuitBreaker(trip_threshold=2, now=lambda: 0.0)
    for _ in range(10):
        run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
                 poison_max_fails=1)
    # Every pass tripped (all fail) -> failures are systemic -> NOTHING quarantined.
    a = src / "a.jpg"
    assert catalog.is_quarantined(str(a), a.stat().st_size, a.stat().st_mtime_ns) is False


def test_quarantine_copies_to_dir_when_set(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "bad.jpg", b"\xff\xd8\xff\xe0" + b"\x09" * 16 + b"\xff\xd9")
    qdir = tmp_path / "quarantine"
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_FailsFor({"bad.jpg"}))
    breaker = _fresh_breaker()
    for i in range(3):
        _make_jpeg(src / f"good_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i + 1]) * 16 + b"\xff\xd9")
        run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
                 poison_max_fails=3, quarantine_dir=qdir)
    copied = list(qdir.glob("*_bad.jpg"))
    assert len(copied) == 1
