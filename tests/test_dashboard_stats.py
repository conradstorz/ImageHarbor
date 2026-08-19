"""Tests for imageharbor.dashboard.stats.collect().

Two things this suite exists to pin down (see the module docstring and
CLAUDE.md's dashboard-stats task brief):

- **A failing section must not fail the document.** `test_a_failing_section_...`
  forces one section to raise and asserts the rest of the document is still a
  complete, populated set of sections with only the broken one reporting
  `None`.
- **The evidence tier distributions must read the right columns.** The
  fixture in `test_tier_distributions_match_a_real_pipeline_run` produces
  deliberately different date-tier and descriptor-tier shapes (date:
  {0: 2, 10: 1}, descriptor: {0: 1, 30: 2}) specifically so that a query
  which accidentally swapped `date_tier`/`descriptor_tier` would fail this
  test rather than passing by coincidence.

Catalog fixtures for tier-count assertions are built by running the real
`Pipeline`, not by hand-writing catalog rows, so the tiers under test are
exactly what the system itself produces from EXIF-free JPEGs and their
original filenames.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.dashboard import stats
from imageharbor.dashboard.control import ControlPlane
from imageharbor.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _jpeg(marker: bytes) -> bytes:
    """A minimal pseudo-JPEG whose bytes (and therefore digest) vary with *marker*."""
    return b"\xff\xd8\xff\xe0" + marker * 16 + b"\xff\xd9"


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def control(catalog: Catalog) -> ControlPlane:
    return ControlPlane(catalog, env_interval=300, env_enrich=True)


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


def _run_pipeline_with_three_photos(source: Path, organized: Path, catalog: Catalog):
    """Three photos chosen so date-tier and descriptor-tier land on different
    combinations, and the two distributions have different shapes:

    - IMG_0001.jpg                  -> camera-generated name, no date in name
                                        => date_tier=0 (none), descriptor_tier=0 (none)
    - family_reunion.jpg            -> human name, no date in name
                                        => date_tier=0 (none), descriptor_tier=30 (human)
    - vacation_20200501_beach.jpg   -> human name, compact date embedded
                                        => date_tier=10 (filename), descriptor_tier=30 (human)

    date_tier distribution:       {0: 2, 10: 1}
    descriptor_tier distribution: {0: 1, 30: 2}
    """
    _write(source / "IMG_0001.jpg", _jpeg(b"\x00"))
    _write(source / "family_reunion.jpg", _jpeg(b"\x01"))
    _write(source / "vacation_20200501_beach.jpg", _jpeg(b"\x02"))
    result = Pipeline(source, organized, catalog).run()
    assert result.errors == 0
    assert result.copied == 3
    return result


# ---------------------------------------------------------------------------
# Empty catalog
# ---------------------------------------------------------------------------


def test_empty_catalog_returns_a_complete_document_of_zeros(
    catalog: Catalog, control: ControlPlane
) -> None:
    now = datetime(2026, 8, 19, tzinfo=timezone.utc)
    doc = stats.collect(catalog, control, now=now)

    # No section is None merely because the catalog is empty -- None is
    # reserved for "this section's query raised".
    for key in ("now", "library", "evidence", "queues", "history", "projection", "overrides"):
        assert doc[key] is not None, f"{key!r} section was None on an empty catalog"

    assert doc["now"]["state"] == "running"
    assert doc["now"]["current_run"] is None
    assert doc["now"]["last_run"] is None
    assert doc["now"]["phase"] is None

    assert doc["library"] == {
        "total_photos": 0,
        "total_bytes": 0,
        "distinct_source_paths": 0,
        "date_range": {"earliest": None, "latest": None},
        "undated_count": 0,
        "duplicates_collapsed": 0,
        "bytes_saved": 0,
        "enriched_count": 0,
        "unenriched_count": 0,
    }

    assert doc["evidence"]["date_tiers"] == [
        {"tier": 40, "source": "exif_original", "count": 0},
        {"tier": 30, "source": "external_sidecar", "count": 0},
        {"tier": 20, "source": "exif_other", "count": 0},
        {"tier": 10, "source": "filename_pattern", "count": 0},
        {"tier": 0, "source": "none", "count": 0},
    ]
    assert doc["evidence"]["descriptor_tiers"] == [
        {"tier": 30, "source": "human_filename", "count": 0},
        {"tier": 20, "source": "ai_subject", "count": 0},
        {"tier": 0, "source": "none", "count": 0},
    ]

    assert doc["queues"] == {
        "unenriched_count": 0,
        "quarantined_count": 0,
        "quarantined": [],
        "failed_active_count": 0,
        "failed_active": [],
        "takeout_pending": 0,
        "takeout": {"archives": {}, "members": {}, "missing_metadata": 0},
    }

    assert doc["history"]["runs"] == []
    assert doc["history"]["last_24h"] == {
        "passes": 0, "copied": 0, "duplicates": 0, "errors": 0,
        "enriched": 0, "enrich_failed": 0,
    }
    assert doc["history"]["last_30d"] == doc["history"]["last_24h"]

    assert doc["projection"]["backlog"] == 0
    assert doc["projection"]["status"] == "complete"

    assert doc["overrides"]["interval"] == {
        "value": 300, "env_value": 300, "overridden": False,
    }
    assert doc["overrides"]["enrich"] == {
        "value": True, "env_value": True, "overridden": False,
    }


# ---------------------------------------------------------------------------
# Evidence: tier distributions against a real pipeline run
# ---------------------------------------------------------------------------


def test_tier_distributions_match_a_real_pipeline_run(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, control: ControlPlane
) -> None:
    source = tmp_path / "source"
    _run_pipeline_with_three_photos(source, organized_dir, catalog)

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))

    date_counts = {row["tier"]: row["count"] for row in doc["evidence"]["date_tiers"]}
    descriptor_counts = {
        row["tier"]: row["count"] for row in doc["evidence"]["descriptor_tiers"]
    }

    assert date_counts == {40: 0, 30: 0, 20: 0, 10: 1, 0: 2}
    assert descriptor_counts == {30: 2, 20: 0, 0: 1}

    assert doc["library"]["total_photos"] == 3
    assert doc["library"]["undated_count"] == 2


def test_library_section_reports_dedup_savings(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, control: ControlPlane
) -> None:
    source = tmp_path / "source"
    content = _jpeg(b"\x07")
    _write(source / "a.jpg", content)
    _write(source / "copy_of_a.jpg", content)  # byte-identical -> same digest

    Pipeline(source, organized_dir, catalog).run()

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))
    lib = doc["library"]

    assert lib["total_photos"] == 1
    assert lib["distinct_source_paths"] == 2
    assert lib["duplicates_collapsed"] == 1
    assert lib["bytes_saved"] == len(content)
    assert lib["total_bytes"] == len(content)


# ---------------------------------------------------------------------------
# Queues: unenriched / quarantined counts
# ---------------------------------------------------------------------------


def test_unenriched_and_quarantined_counts(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, control: ControlPlane
) -> None:
    source = tmp_path / "source"
    _run_pipeline_with_three_photos(source, organized_dir, catalog)

    rows = list(catalog.iter_all())
    assert len(rows) == 3

    # Mark one photo enriched -- it should drop out of the unenriched queue.
    catalog.mark_enriched(
        rows[0]["sha256_b64url"],
        pcs_primary="200", pcs_name="test", secondary_tags=[],
        ai_caption="", objects=[], ocr_text="", model_version="stub",
    )

    # Quarantine a second photo's only source path -- it should drop out of
    # the unenriched queue too (see iter_unenriched's docstring) and appear
    # under queues.quarantined with its reason.
    quarantine_target = rows[1]
    source_row = catalog.sources_for(quarantine_target["sha256_b64url"])[0]
    catalog.record_file_failure(
        source_row["source_path"], source_row["size"], source_row["mtime_ns"], "boom"
    )
    catalog.quarantine_file(source_row["source_path"])

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))
    queues = doc["queues"]

    assert queues["unenriched_count"] == 1  # only rows[2] remains
    assert queues["quarantined_count"] == 1
    assert queues["quarantined"][0]["source_path"] == source_row["source_path"]
    assert queues["quarantined"][0]["last_error"] == "boom"
    assert queues["failed_active_count"] == 0

    assert doc["library"]["enriched_count"] == 1
    assert doc["library"]["unenriched_count"] == 2  # library counts all non-enriched


# ---------------------------------------------------------------------------
# Now: current/last run, breaker
# ---------------------------------------------------------------------------


def test_now_section_reports_last_and_current_run(
    catalog: Catalog, control: ControlPlane
) -> None:
    run1 = catalog.run_start("facts")
    catalog.run_finish(
        run1, scanned=10, copied=8, duplicates=2, errors=0,
        enriched=0, enrich_failed=0, breaker_state="CLOSED", paused=False,
    )
    run2 = catalog.run_start("enrich")  # never finished: simulates an in-flight pass

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))
    now_section = doc["now"]

    assert now_section["state"] == "running"
    assert now_section["phase"] == "enrich"
    assert now_section["current_run"]["id"] == run2
    assert now_section["last_run"]["id"] == run1
    assert now_section["last_run"]["copied"] == 8


def test_now_section_breaker_absent_is_distinct_from_a_real_state(
    catalog: Catalog, control: ControlPlane
) -> None:
    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))
    assert doc["now"]["breaker"] == {
        "state": None, "enabled": None, "seconds_until_probe": None,
    }


def test_now_section_reports_open_breaker_state(
    catalog: Catalog, control: ControlPlane
) -> None:
    breaker = CircuitBreaker(trip_threshold=1)
    breaker.record_failure()  # trips open immediately with threshold=1

    doc = stats.collect(catalog, control, breaker=breaker, now=datetime.now(timezone.utc))
    assert doc["now"]["breaker"]["state"] == "open"
    assert doc["now"]["breaker"]["enabled"] is True


def test_now_section_pausing_when_paused_mid_run(
    catalog: Catalog, control: ControlPlane
) -> None:
    catalog.run_start("facts")  # left unfinished -> a pass is in flight
    control.set_paused(True)

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))
    assert doc["now"]["state"] == "pausing"

    control.set_paused(False)
    catalog.run_start("facts")  # a second unfinished row, no pause this time
    doc2 = stats.collect(catalog, control, now=datetime.now(timezone.utc))
    # Not paused: an in-flight pass is just "running".
    assert doc2["now"]["state"] == "running"


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_projection_is_stalled_when_paused(
    tmp_path: Path, organized_dir: Path, catalog: Catalog, control: ControlPlane
) -> None:
    source = tmp_path / "source"
    _write(source / "family_reunion.jpg", _jpeg(b"\x03"))
    Pipeline(source, organized_dir, catalog).run()

    control.set_paused(True)

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))

    assert doc["projection"] is not None
    assert doc["projection"]["status"] == "stalled"
    assert doc["projection"]["reason"] == "paused"
    assert doc["projection"]["backlog"] == 1


def test_projection_staleness_window_derives_from_poll_interval(
    catalog: Catalog,
) -> None:
    """A small poll interval must not inherit projections' 1-day default stale
    window -- see the comment in `_projection_section`. Two completed passes
    that both ended two hours ago are "stale" under our interval-derived
    window (max(4*interval, 3600) = 3600s here) even though two hours would
    NOT trip projections.DEFAULT_STALE_AFTER_SECONDS (one day).
    """
    tiny_interval_control = ControlPlane(catalog, env_interval=10, env_enrich=True)

    end = datetime.now(timezone.utc) - timedelta(hours=2)
    start = end - timedelta(seconds=60)
    for _ in range(2):
        run_id = catalog.run_start("enrich")
        catalog.run_finish(
            run_id, scanned=5, copied=0, duplicates=0, errors=0, enriched=5,
            enrich_failed=0, breaker_state="CLOSED", paused=False,
        )
        catalog._conn.execute(
            "UPDATE runs SET started_at=?, ended_at=? WHERE id=?",
            (start.isoformat(), end.isoformat(), run_id),
        )
    catalog._conn.commit()

    # A backlog > 0 is required for the staleness path to matter at all.
    catalog.upsert(
        sha256_b64url="a" * 43, original_path="/src/a.jpg", organized_path="Undated/a.jpg",
    )

    doc = stats.collect(catalog, tiny_interval_control, now=datetime.now(timezone.utc))

    assert doc["projection"]["status"] == "stalled"
    assert "ended" in doc["projection"]["reason"]


# ---------------------------------------------------------------------------
# Hard rule: a failing section must not fail the document
# ---------------------------------------------------------------------------


def test_a_failing_section_does_not_fail_the_document(
    catalog: Catalog, control: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated query failure")

    monkeypatch.setattr(stats, "_library_section", _boom)

    doc = stats.collect(catalog, control, now=datetime.now(timezone.utc))

    assert doc["library"] is None
    assert doc["now"] is not None
    assert doc["evidence"] is not None
    assert doc["queues"] is not None
    assert doc["history"] is not None
    assert doc["projection"] is not None
    assert doc["overrides"] is not None
