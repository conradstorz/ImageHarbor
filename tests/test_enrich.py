"""Tests for the AI enrichment pass and its non-degradation guarantee."""

from imageharbor import concept_map, tiers
from imageharbor.ai_classifier import AIClassifier, ContentDescription, StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrich import enrich_library
from imageharbor.pipeline import Pipeline


class FixedClassifier(StubClassifier):
    """A classifier that always reports the same subject."""

    def __init__(self, subject="beach"):
        self._subject = subject

    def describe(self, image_path, exif_data=None):
        return ContentDescription(
            primary_subject=self._subject,
            scene="outdoor",
            objects=["sand"],
            caption="a beach",
            tags=["sand"],
            ocr_text="",
            model_version="fixed-1",
        )


class BrokenClassifier(AIClassifier):
    """Every call fails, as during a backend outage."""

    def describe(self, image_path, exif_data=None):
        raise RuntimeError("backend down")


def _make(tmp_path, name, content=b"fake-image-bytes"):
    src = tmp_path / "src"
    path = src / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return src


def _facts(tmp_path, name, content=b"fake-image-bytes"):
    src = _make(tmp_path, name, content)
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    stats = Pipeline(src, dest, cat, write_sidecars=True).run()
    return cat, dest, stats.results[0]


def test_enrichment_names_a_camera_named_file(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    assert result.organized_path.name.startswith("2019-07-04_")

    stats = enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    assert stats.enriched == 1
    assert stats.renamed == 1
    row = cat.get_by_sha256(result.sha256_b64url)
    assert row["organized_path"].endswith(f"2019-07-04-beach_{result.sha256_b64url}.jpg")
    assert cat.tiers_for(result.sha256_b64url) == (
        tiers.DATE_FILENAME_PATTERN,
        tiers.DESC_AI_SUBJECT,
    )
    cat.close()


def test_enrichment_never_displaces_a_human_filename(tmp_path):
    cat, dest, result = _facts(tmp_path, "Emma's graduation.jpg")
    before = result.organized_path.name

    stats = enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    assert stats.enriched == 1
    assert stats.renamed == 0
    row = cat.get_by_sha256(result.sha256_b64url)
    assert row["organized_path"].endswith(before)
    # The classification is still recorded -- only the *name* is protected.
    assert row["pcs_primary"]
    cat.close()


def test_a_second_enrichment_run_is_a_no_op(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier())
    after_first = cat.get_by_sha256(result.sha256_b64url)["organized_path"]

    second = enrich_library(cat, dest, FixedClassifier(subject="mountain"))

    assert second.total == 0
    assert cat.get_by_sha256(result.sha256_b64url)["organized_path"] == after_first
    cat.close()


def test_a_backend_outage_degrades_nothing(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    before_path = result.organized_path
    before_tiers = cat.tiers_for(result.sha256_b64url)

    stats = enrich_library(cat, dest, BrokenClassifier())

    assert stats.errors == 1
    assert stats.enriched == 0
    assert before_path.exists()
    assert cat.get_by_sha256(result.sha256_b64url)["organized_path"] == str(before_path)
    assert cat.tiers_for(result.sha256_b64url) == before_tiers
    cat.close()


def test_a_tripped_breaker_aborts_the_pass(tmp_path):
    from imageharbor.circuit_breaker import CircuitBreaker

    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    (src / "IMG_3.jpg").write_bytes(b"three")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    breaker = CircuitBreaker(trip_threshold=2, backoff_base=1.0, backoff_cap=1.0)
    stats = enrich_library(cat, dest, BrokenClassifier(), breaker=breaker)

    assert stats.aborted is True
    assert stats.errors == 2
    cat.close()


def test_enrichment_adds_classification_to_the_sidecar(tmp_path):
    from imageharbor.sidecar import read_sidecar

    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    new_path = cat.get_by_sha256(result.sha256_b64url)["organized_path"]
    from pathlib import Path

    data = read_sidecar(Path(new_path))
    assert data["classification"]["primary_subject"] == "beach"
    # Facts written by the earlier pass survive the merge.
    assert data["identity"]["sha256_b64url"] == result.sha256_b64url
    assert data["date"]["tier"] == tiers.DATE_FILENAME_PATTERN
    cat.close()


def test_classification_records_the_model_version(tmp_path):
    """A classification without its model version cannot be re-evaluated later.

    When a better model arrives, the sidecar's classification history has to say
    which answer came from which model, or there is no way to tell an
    improvement from a regression.
    """
    from pathlib import Path

    from imageharbor.sidecar import read_sidecar

    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    new_path = cat.get_by_sha256(result.sha256_b64url)["organized_path"]
    data = read_sidecar(Path(new_path))
    assert data["classification"]["model_version"]
    cat.close()


def test_enrichment_self_heals_a_stale_catalog_path(tmp_path):
    """Simulates a crash between the rename and the catalog update."""
    import shutil

    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    old = result.organized_path
    moved = old.parent / f"2019-07-04-moved_{result.sha256_b64url}.jpg"
    shutil.move(str(old), str(moved))

    stats = enrich_library(cat, dest, FixedClassifier())

    assert stats.errors == 0
    assert stats.enriched == 1
    cat.close()


def test_self_heal_without_upgrade_carries_the_sidecar(tmp_path):
    """A stale path repaired at a tier that blocks renaming must keep its sidecar.

    A human-named file cannot be renamed by the AI pass, so it takes the
    self-heal branch rather than the rename branch. If that branch left the
    sidecar behind, the merge would rebuild it from an empty base and lose
    every fact the first pass recorded.
    """
    import shutil
    from pathlib import Path

    from imageharbor.sidecar import read_sidecar, sidecar_path_for

    cat, dest, result = _facts(tmp_path, "Emma's graduation.jpg")
    old = result.organized_path
    assert read_sidecar(old)["identity"]["sha256_b64url"] == result.sha256_b64url

    # Relocate the file and its sidecar is left behind by an external actor.
    moved = old.parent / f"moved-{old.name}"
    shutil.move(str(old), str(moved))

    enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    healed = Path(cat.get_by_sha256(result.sha256_b64url)["organized_path"])
    data = read_sidecar(healed)
    assert data["identity"]["sha256_b64url"] == result.sha256_b64url
    assert data["descriptor"]["tier"] == tiers.DESC_HUMAN_FILENAME
    assert data["classification"]["primary_subject"] == "beach"
    assert not sidecar_path_for(old).exists()
    cat.close()


def test_a_local_failure_does_not_wedge_the_pass(tmp_path):
    """An exception after perception must not escape or block later rows.

    The queue is ordered by id and a row that raises is never marked enriched
    or failed, so an escaping exception would crash on the same row forever.
    This is a POST-perception (local/catalog) failure, not an AI-perception
    one, so it must land in io_failed, not ai_failed -- it says nothing about
    the AI backend and must never feed poison-file quarantine.
    """
    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    calls = {"n": 0}
    real_mark = cat.mark_enriched

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("catalog is busy")
        return real_mark(*args, **kwargs)

    cat.mark_enriched = flaky

    stats = enrich_library(cat, dest, FixedClassifier())

    assert stats.total == 2
    assert stats.errors == 1
    assert stats.enriched == 1
    assert len(stats.io_failed) == 1  # counted, but not AI-perception evidence
    assert stats.ai_failed == []  # must never feed poison-file quarantine
    cat.close()


def test_reclassify_skips_rows_with_no_organized_copy(tmp_path):
    """--reclassify walks the whole catalog, including rows iter_unenriched hides."""
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    cat.upsert(sha256_b64url="ORPHAN", original_path="/gone.jpg")

    stats = enrich_library(cat, dest, FixedClassifier(), reclassify=True)

    assert stats.total == 1  # the orphan is skipped, not crashed on
    cat.close()


def test_reclassify_forces_a_second_pass(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier(subject="beach"))

    stats = enrich_library(cat, dest, FixedClassifier(subject="mountain"), reclassify=True)

    assert stats.total == 1
    assert cat.get_by_sha256(result.sha256_b64url)["pcs_name"]
    cat.close()


# ---------------------------------------------------------------------------
# Pause plumbing (dashboard Task 4)
# ---------------------------------------------------------------------------


def test_pause_stops_enrichment_between_rows(tmp_path):
    """pause_check must be consulted between rows, never mid-row.

    Same guarantee Task 4 protects in the facts pass: pausing stops the NEXT
    row from starting, but never leaves a row half-processed.
    """
    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    (src / "IMG_3.jpg").write_bytes(b"three")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    seen = 0

    def _pause_after_two():
        nonlocal seen
        seen += 1
        return seen > 2

    stats = enrich_library(cat, dest, FixedClassifier(), pause_check=_pause_after_two)

    assert stats.total == 2
    assert stats.enriched == 2
    cat.close()


def test_no_pause_check_enriches_everything(tmp_path):
    """The default path is unchanged for every existing caller."""
    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    stats = enrich_library(cat, dest, FixedClassifier())

    assert stats.total == 2
    assert stats.enriched == 2
    cat.close()


def test_pause_check_true_from_the_start_enriches_nothing(tmp_path):
    """Pins the check to BEFORE a row is processed, not after."""
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")

    stats = enrich_library(cat, dest, FixedClassifier(), pause_check=lambda: True)

    assert stats.total == 0
    assert stats.enriched == 0
    assert cat.get_by_sha256(result.sha256_b64url)["enriched_at"] is None
    cat.close()


# ---------------------------------------------------------------------------
# pick_class fallback failures are AI evidence (R3 Task 1)
# ---------------------------------------------------------------------------


class DescribesButCannotPick(StubClassifier):
    """describe() succeeds (perception is fine); pick_class() dies.

    Models a real OpenAIClassifier whose backend goes down between the
    describe() chat call and the pick_class() chat call for a concept-map
    miss -- pick_class is a network call too, so its failure is the same
    kind of AI evidence describe()'s failure is.
    """

    def pick_class(self, content, classes):
        raise RuntimeError("backend died mid-pass")


def test_a_pick_class_failure_is_ai_evidence_and_feeds_the_breaker(tmp_path):
    from imageharbor.circuit_breaker import CircuitBreaker

    # Nonsense stems so concept_map.class_for misses for every row (StubClassifier
    # derives primary_subject from the filename, and none of these words are in
    # STATIC_SEED or the learned-concepts store).
    src = _make(tmp_path, "zzxxqq1.jpg", b"one")
    (src / "zzxxqq2.jpg").write_bytes(b"two")
    (src / "zzxxqq3.jpg").write_bytes(b"three")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    breaker = CircuitBreaker(trip_threshold=3, backoff_base=1.0, backoff_cap=1.0)
    stats = enrich_library(cat, dest, DescribesButCannotPick(), breaker=breaker)

    assert stats.ai_failed  # not io_failed
    assert stats.io_failed == []
    assert stats.aborted is True  # breaker opened and the pass stopped
    assert breaker.is_open()
    cat.close()


def test_a_concept_map_hit_never_calls_pick_class(tmp_path):
    """A learned-concepts hit must enrich fine even with a broken pick_class.

    Proves the pick_class fallback is the only new breaker-feeding path --
    a subject the concept map already knows never reaches pick_class at all.
    """
    cat, dest, result = _facts(tmp_path, "beachy.jpg")
    cat.learned_concept_remember("beachy", "600")

    stats = enrich_library(cat, dest, DescribesButCannotPick())

    assert stats.enriched == 1
    assert stats.ai_failed == []
    assert stats.errors == 0
    cat.close()


# ---------------------------------------------------------------------------
# Fix round 1: class_for()/remember() are LOCAL SQLite I/O, not backend calls
# -- a raise there must stay inside the per-row outer try/except (io_failed,
# breaker untouched, pass continues to later rows), never escape
# enrich_library entirely.
# ---------------------------------------------------------------------------


def test_a_class_for_failure_is_io_evidence_and_the_pass_continues(tmp_path, monkeypatch):
    from imageharbor.circuit_breaker import BreakerState, CircuitBreaker

    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    def boom(*args, **kwargs):
        raise RuntimeError("catalog I/O exploded in class_for")

    monkeypatch.setattr(concept_map, "class_for", boom)

    # trip_threshold=1 means a single fed failure would open the breaker --
    # this proves class_for's failure never reaches record_failure().
    breaker = CircuitBreaker(trip_threshold=1, backoff_base=1.0, backoff_cap=1.0)
    stats = enrich_library(cat, dest, FixedClassifier(), breaker=breaker)

    # Both rows were reached -- the exception in row 1 did not abort the pass.
    assert stats.total == 2
    assert len(stats.io_failed) == 2
    assert stats.ai_failed == []
    assert stats.enriched == 0
    assert stats.aborted is False
    assert breaker.state == BreakerState.CLOSED
    cat.close()


def test_a_remember_failure_is_io_evidence_and_the_pass_continues(tmp_path, monkeypatch):
    from imageharbor.circuit_breaker import CircuitBreaker

    # Nonsense stems so concept_map.class_for misses for every row (as in
    # test_a_pick_class_failure_is_ai_evidence_and_feeds_the_breaker above),
    # which drives every row through the pick_class fallback. StubClassifier's
    # pick_class inherits the ABC default (900) and never raises, so it
    # succeeds and remember() is reached.
    src = _make(tmp_path, "zzxxqq1.jpg", b"one")
    (src / "zzxxqq2.jpg").write_bytes(b"two")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    def boom(*args, **kwargs):
        raise RuntimeError("catalog I/O exploded in remember")

    monkeypatch.setattr(concept_map, "remember", boom)

    breaker = CircuitBreaker(trip_threshold=1, backoff_base=1.0, backoff_cap=1.0)
    stats = enrich_library(cat, dest, StubClassifier(), breaker=breaker)

    assert stats.total == 2
    assert len(stats.io_failed) == 2
    assert stats.ai_failed == []
    assert stats.enriched == 0
    assert stats.aborted is False
    # pick_class succeeded, so record_success() ran before remember() blew up --
    # a real backend success, not a failure, so a CLOSED breaker here is
    # expected either way; the load-bearing assertion is that it never opened.
    assert not breaker.is_open()
    cat.close()
