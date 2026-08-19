"""Tests for imageharbor.catalog."""

from pathlib import Path

import pytest

from imageharbor import catalog as catalog_module
from imageharbor.catalog import Catalog, _from_json


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    db = tmp_path / "test_catalog.db"
    cat = Catalog(db)
    yield cat
    cat.close()


def _fake_digest(n: int = 0) -> str:
    return ("A" * 43)[:-len(str(n))] + str(n) if n else "A" * 43


# ---------------------------------------------------------------------------
# upsert / get_by_sha256
# ---------------------------------------------------------------------------


def test_upsert_and_retrieve(catalog: Catalog) -> None:
    digest = _fake_digest(1)
    catalog.upsert(
        sha256_b64url=digest,
        original_path="/photos/beach.jpg",
        organized_path="/organized/300-places/330-beach/330-beach_AAAA.jpg",
        pcs_primary="330",
        pcs_name="beach",
        ai_caption="A sunny beach",
        objects=["sand", "waves"],
    )
    row = catalog.get_by_sha256(digest)
    assert row is not None
    assert row["pcs_primary"] == "330"
    assert row["ai_caption"] == "A sunny beach"
    assert row["original_path"] == "/photos/beach.jpg"


def test_upsert_defaults_pcs_fields_to_null(catalog: Catalog) -> None:
    """A facts-pass-only row (no pcs_primary/pcs_name given) must not assert
    a classification that was never made -- an unenriched row's classification
    columns must be NULL, not the old '900'/'miscellaneous' sentinel."""
    digest = _fake_digest(2)
    catalog.upsert(sha256_b64url=digest, original_path="/photos/beach.jpg")
    row = catalog.get_by_sha256(digest)
    assert row is not None
    assert row["pcs_primary"] is None
    assert row["pcs_name"] is None
    assert row["enriched_at"] is None


def test_upsert_serializes_bytes_in_exif(catalog: Catalog) -> None:
    # Real EXIF tags (e.g. ExifVersion, SceneType, MakerNote) are raw bytes,
    # which are not JSON-serializable. upsert must not crash on them — a single
    # odd metadata value should never fail an image.
    import json

    digest = _fake_digest(40)
    catalog.upsert(
        sha256_b64url=digest,
        original_path="/photos/real.jpg",
        exif={"ExifVersion": b"0230", "SceneType": b"\x01", "Make": "FUJIFILM"},
    )
    row = catalog.get_by_sha256(digest)
    assert row is not None
    exif = json.loads(row["exif"])
    assert exif["Make"] == "FUJIFILM"
    assert exif["ExifVersion"] == "0230"  # bytes decoded to text
    assert "SceneType" in exif


def test_upsert_updates_on_duplicate_sha256(catalog: Catalog) -> None:
    digest = _fake_digest(2)
    catalog.upsert(sha256_b64url=digest, original_path="/a.jpg", ai_caption="first")
    catalog.upsert(sha256_b64url=digest, original_path="/a.jpg", ai_caption="updated")
    row = catalog.get_by_sha256(digest)
    assert row["ai_caption"] == "updated"


def test_count(catalog: Catalog) -> None:
    assert catalog.count() == 0
    catalog.upsert(sha256_b64url=_fake_digest(3), original_path="/x.jpg")
    assert catalog.count() == 1
    catalog.upsert(sha256_b64url=_fake_digest(4), original_path="/y.jpg")
    assert catalog.count() == 2


def test_is_known_true(catalog: Catalog) -> None:
    digest = _fake_digest(5)
    catalog.upsert(sha256_b64url=digest, original_path="/z.jpg")
    assert catalog.is_known(digest) is True


def test_is_known_false(catalog: Catalog) -> None:
    assert catalog.is_known("B" * 43) is False


def test_iter_all_empty(catalog: Catalog) -> None:
    assert list(catalog.iter_all()) == []


def test_iter_all_returns_all_rows(catalog: Catalog) -> None:
    for i in range(5):
        catalog.upsert(sha256_b64url=_fake_digest(10 + i), original_path=f"/{i}.jpg")
    rows = list(catalog.iter_all())
    assert len(rows) == 5


def test_mark_duplicate_appends_history(catalog: Catalog) -> None:
    import json

    digest = _fake_digest(20)
    catalog.upsert(sha256_b64url=digest, original_path="/orig.jpg")
    catalog.mark_duplicate(digest, "/dup.jpg")
    row = catalog.get_by_sha256(digest)
    history = json.loads(row["processing_history"])
    assert any(e.get("event") == "duplicate_detected" for e in history)


def test_mark_duplicate_is_a_noop_for_a_repeated_identical_path(catalog: Catalog) -> None:
    """Re-running `process` against an unchanged source must not grow
    processing_history without bound: a repeat of the SAME duplicate path
    right after itself is skipped rather than appended again."""
    import json

    digest = _fake_digest(21)
    catalog.upsert(sha256_b64url=digest, original_path="/orig.jpg")
    catalog.mark_duplicate(digest, "/dup.jpg")
    catalog.mark_duplicate(digest, "/dup.jpg")
    catalog.mark_duplicate(digest, "/dup.jpg")
    row = catalog.get_by_sha256(digest)
    history = json.loads(row["processing_history"])
    assert sum(1 for e in history if e.get("event") == "duplicate_detected") == 1


def test_mark_duplicate_still_appends_for_a_different_path(catalog: Catalog) -> None:
    """A genuinely NEW duplicate path must still be recorded even when the
    most recent entry was also a duplicate_detected event."""
    import json

    digest = _fake_digest(22)
    catalog.upsert(sha256_b64url=digest, original_path="/orig.jpg")
    catalog.mark_duplicate(digest, "/dup1.jpg")
    catalog.mark_duplicate(digest, "/dup2.jpg")
    row = catalog.get_by_sha256(digest)
    history = json.loads(row["processing_history"])
    dup_paths = [e.get("duplicate_path") for e in history if e.get("event") == "duplicate_detected"]
    assert dup_paths == ["/dup1.jpg", "/dup2.jpg"]


def test_get_by_original_path(catalog: Catalog) -> None:
    digest = _fake_digest(30)
    catalog.upsert(sha256_b64url=digest, original_path="/specific.jpg")
    row = catalog.get_by_original_path("/specific.jpg")
    assert row is not None
    assert row["sha256_b64url"] == digest


def test_get_by_original_path_missing(catalog: Catalog) -> None:
    assert catalog.get_by_original_path("/nonexistent.jpg") is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_from_json_valid_roundtrip() -> None:
    assert _from_json('[1, 2, 3]') == [1, 2, 3]
    assert _from_json('{"a": 1}') == {"a": 1}


def test_from_json_malformed_returns_raw_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Malformed JSON must not be silently swallowed: the raw text is returned
    # (behavior unchanged) but a warning is logged so data loss is visible.
    with caplog.at_level("WARNING"):
        result = _from_json("not-valid-json{")
    assert result == "not-valid-json{"
    assert any("Malformed JSON" in rec.message for rec in caplog.records)


def test_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    with Catalog(db) as cat:
        cat.upsert(sha256_b64url="C" * 43, original_path="/cm.jpg")
        assert cat.count() == 1
    # After close, any further use would raise; just check it doesn't explode


# ---------------------------------------------------------------------------
# source_seen cache
# ---------------------------------------------------------------------------


def test_source_seen_unknown_is_not_unchanged(catalog: Catalog) -> None:
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 111) is False


def test_source_seen_roundtrip_unchanged(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111, "A" * 43)
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 111) is True


def test_source_seen_changed_size_is_not_unchanged(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111)
    assert catalog.source_is_unchanged("/src/a.jpg", 200, 111) is False


def test_source_seen_changed_mtime_is_not_unchanged(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111)
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 222) is False


def test_source_seen_upsert_updates(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111)
    catalog.record_source_seen("/src/a.jpg", 100, 999)  # file changed
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 999) is True
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 111) is False


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------


def test_taxonomy_seed_insert_get_children(catalog: Catalog) -> None:
    assert catalog.taxonomy_is_empty() is True
    catalog.taxonomy_insert("500", None, "events", "500-events")
    catalog.taxonomy_insert("540", "500", "holidays", "540-holidays")
    assert catalog.taxonomy_is_empty() is False
    row = catalog.taxonomy_get("540")
    assert row["label"] == "holidays"
    assert row["parent_code"] == "500"
    kids = catalog.taxonomy_children("500")
    assert [k["code"] for k in kids] == ["540"]
    tops = catalog.taxonomy_children(None)
    assert [t["code"] for t in tops] == ["500"]


def test_taxonomy_set_alias(catalog: Catalog) -> None:
    catalog.taxonomy_insert("540", "500", "holidays", "540-holidays")
    catalog.taxonomy_insert("550", "500", "festivities", "550-festivities")
    catalog.taxonomy_set_alias("550", "540")
    row = catalog.taxonomy_get("550")
    assert row["alias_of"] == "540"
    assert row["active"] == 0


def test_taxonomy_set_aliases(catalog: Catalog) -> None:
    import json
    catalog.taxonomy_insert("540", "500", "holidays", "540-holidays")
    catalog.taxonomy_set_aliases("540", ["festivities", "xmas"])
    assert json.loads(catalog.taxonomy_get("540")["aliases"]) == ["festivities", "xmas"]


# ---------------------------------------------------------------------------
# learned_concepts
# ---------------------------------------------------------------------------


def test_learned_concept_roundtrip(catalog: Catalog) -> None:
    assert catalog.learned_concept_get("marching band") is None
    catalog.learned_concept_remember("marching band", "500")
    assert catalog.learned_concept_get("marching band") == "500"


def test_learned_concept_remember_updates_and_counts(catalog: Catalog) -> None:
    catalog.learned_concept_remember("widget", "200")
    catalog.learned_concept_remember("widget", "300")  # correction / re-seen
    assert catalog.learned_concept_get("widget") == "300"


# ---------------------------------------------------------------------------
# failed_files (poison-file tracking)
# ---------------------------------------------------------------------------


def test_record_file_failure_increments(catalog: Catalog) -> None:
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 1
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 2
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 3


def test_changed_file_resets_fail_count_and_quarantine(catalog: Catalog) -> None:
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    catalog.quarantine_file("/src/a.jpg")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is True
    # File changed (new size/mtime): count resets, quarantine cleared.
    assert catalog.record_file_failure("/src/a.jpg", 200, 222, "boom") == 1
    assert catalog.is_quarantined("/src/a.jpg", 200, 222) is False


def test_is_quarantined_requires_flag_and_matching_stat(catalog: Catalog) -> None:
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is False  # not flagged yet
    catalog.quarantine_file("/src/a.jpg")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is True
    assert catalog.is_quarantined("/src/a.jpg", 999, 111) is False  # size differs
    assert catalog.is_quarantined("/src/missing.jpg", 100, 111) is False  # no row


def test_clear_file_failure_removes_row(catalog: Catalog) -> None:
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    catalog.quarantine_file("/src/a.jpg")
    catalog.clear_file_failure("/src/a.jpg")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is False
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 1  # fresh row


# ---------------------------------------------------------------------------
# sources, tiers, enrichment queue
# ---------------------------------------------------------------------------


def test_record_source_accumulates_back_pointers(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a/one.jpg")
        cat.record_source("D1", "/a/one.jpg", 100, 111)
        cat.record_source("D1", "/b/two.jpg", 100, 222)
        cat.record_source("D1", "/c/three.jpg", 100, 333)
        rows = cat.sources_for("D1")
        assert {r["source_path"] for r in rows} == {"/a/one.jpg", "/b/two.jpg", "/c/three.jpg"}


def test_record_source_is_idempotent_and_updates_last_seen(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a/one.jpg")
        cat.record_source("D1", "/a/one.jpg", 100, 111)
        first = cat.sources_for("D1")[0]["first_seen_at"]
        cat.record_source("D1", "/a/one.jpg", 100, 111)
        rows = cat.sources_for("D1")
        assert len(rows) == 1
        assert rows[0]["first_seen_at"] == first


def test_upsert_stores_tiers(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(
            sha256_b64url="D1",
            original_path="/a/one.jpg",
            date_value="2019-07-04",
            date_tier=40,
            date_source="exif_original",
            descriptor_value="emmas-graduation",
            descriptor_tier=30,
            descriptor_source="human_filename",
        )
        assert cat.tiers_for("D1") == (40, 30)
        row = cat.get_by_sha256("D1")
        assert row["date_value"] == "2019-07-04"
        assert row["descriptor_source"] == "human_filename"


def test_tiers_for_unknown_digest_is_zero(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        assert cat.tiers_for("nope") == (0, 0)


def test_iter_unenriched_is_the_work_queue(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.upsert(sha256_b64url="D2", original_path="/b.jpg", organized_path="/lib/b.jpg")
        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D1", "D2"}

        cat.mark_enriched(
            "D1",
            pcs_primary="330",
            pcs_name="beach",
            secondary_tags=["sand"],
            ai_caption="a beach",
            objects=["sand"],
            ocr_text="",
            model_version="stub-1",
            scene="outdoor",
        )
        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D2"}


def test_iter_unenriched_respects_limit(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        for i in range(5):
            cat.upsert(
                sha256_b64url=f"D{i}",
                original_path=f"/{i}.jpg",
                organized_path=f"/lib/{i}.jpg",
            )
        assert len(cat.iter_unenriched(limit=2)) == 2


def test_iter_unenriched_excludes_quarantined_content(tmp_path):
    """Quarantine means "stop asking the model", so the row leaves the queue.

    failed_files is keyed by source path and photos by digest, so this only
    works if the exclusion joins through the sources table.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D1"}

        cat.record_file_failure("/a.jpg", 10, 111, "boom")
        cat.quarantine_file("/a.jpg")

        assert cat.iter_unenriched() == []


def test_iter_unenriched_excludes_content_quarantined_via_any_source(tmp_path):
    """Identical bytes fail identically, so one quarantined path condemns them."""
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_source("D1", "/b.jpg", 10, 222)

        cat.record_file_failure("/b.jpg", 10, 222, "boom")
        cat.quarantine_file("/b.jpg")

        assert cat.iter_unenriched() == []


def test_quarantine_survives_a_metadata_only_mtime_change(tmp_path):
    """A touch must not lift a quarantine whose bytes never changed.

    iter_unenriched correlates the exclusion on (path, size, mtime_ns). If
    record_source overwrote those stats on re-observation, a backup tool or a
    CIFS remount touching the file would silently re-admit known-poison content
    to the AI queue.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_file_failure("/a.jpg", 10, 111, "boom")
        cat.quarantine_file("/a.jpg")
        assert cat.iter_unenriched() == []

        # Same bytes, same digest, only the mtime moved.
        cat.record_source("D1", "/a.jpg", 10, 999)

        assert cat.iter_unenriched() == []


def test_record_source_freezes_stats_for_unchanged_content(tmp_path):
    """The row is keyed by digest, so its stats describe fixed content."""
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        first = cat.sources_for("D1")[0]["last_seen_at"]
        cat.record_source("D1", "/a.jpg", 10, 999)

        row = cat.sources_for("D1")[0]
        assert row["mtime_ns"] == 111
        assert row["size"] == 10
        assert row["last_seen_at"] >= first


def test_new_content_at_a_quarantined_path_re_enters_the_queue(tmp_path):
    """Quarantine is scoped to the exact bytes that failed.

    CLAUDE.md: a quarantined file is skipped thereafter "until its bytes
    change". Replacing a poison photo with a fixed one under the same filename
    must lift the exclusion for the new content -- and nothing else can, since
    record_file_failure's stale-stat reset only runs if the file is attempted.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_file_failure("/a.jpg", 10, 111, "boom")
        cat.quarantine_file("/a.jpg")
        assert cat.iter_unenriched() == []

        # Same path, new bytes: new digest, new size and mtime.
        cat.upsert(sha256_b64url="D2", original_path="/a.jpg", organized_path="/lib/b.jpg")
        cat.record_source("D2", "/a.jpg", 20, 222)

        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D2"}


def test_a_merely_failing_file_stays_in_the_queue(tmp_path):
    """Only QUARANTINED content leaves; a file still accruing failures retries."""
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_file_failure("/a.jpg", 10, 111, "boom")  # not yet quarantined

        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D1"}


def test_iter_unenriched_skips_rows_with_no_organized_copy(tmp_path):
    """The enrichment pass reads the ORGANIZED copy, not the source.

    A row with no organized_path has no file for it to open, so it must never
    reach the queue — `Path(None)` raises TypeError.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg")
        assert cat.iter_unenriched() == []


def test_set_placement_updates_path_and_tiers(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/Undated/x.jpg")
        cat.set_placement(
            "D1",
            organized_path="/lib/2019/2019-07/2019-07-04-beach_x.jpg",
            date_value="2019-07-04",
            date_tier=40,
            date_source="exif_original",
            descriptor_value="beach",
            descriptor_tier=20,
            descriptor_source="ai_subject",
        )
        row = cat.get_by_sha256("D1")
        assert row["organized_path"] == "/lib/2019/2019-07/2019-07-04-beach_x.jpg"
        assert cat.tiers_for("D1") == (40, 20)


_OLD_PHOTOS_TABLE_DDL = (
    "CREATE TABLE photos (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "sha256_b64url TEXT NOT NULL UNIQUE, original_path TEXT NOT NULL, "
    "organized_path TEXT, pcs_version TEXT NOT NULL DEFAULT '1', "
    "pcs_primary TEXT NOT NULL DEFAULT '900', pcs_name TEXT NOT NULL DEFAULT 'miscellaneous', "
    "secondary_tags TEXT NOT NULL DEFAULT '[]', ai_caption TEXT NOT NULL DEFAULT '', "
    "objects TEXT NOT NULL DEFAULT '[]', ocr_text TEXT NOT NULL DEFAULT '', "
    "exif TEXT NOT NULL DEFAULT '{}', model_version TEXT NOT NULL DEFAULT 'unknown', "
    "processing_history TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, processed_at TEXT)"
)


def test_existing_catalog_gains_new_columns(tmp_path):
    """An older DB must open and upgrade without losing rows.

    `sources` is pre-populated here so this test isolates the additive
    ALTER TABLE column-upgrade mechanism from the separate legacy-catalog
    guard (see test_legacy_catalog_with_rows_and_no_sources_raises below,
    which covers the case this DB would otherwise also trigger).
    """
    import sqlite3
    from imageharbor.catalog import Catalog

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_OLD_PHOTOS_TABLE_DDL)
    conn.execute(
        "INSERT INTO photos (sha256_b64url, original_path, created_at) VALUES ('OLD', '/x.jpg', 'now')"
    )
    conn.execute(
        "CREATE TABLE sources (sha256_b64url TEXT NOT NULL, source_path TEXT NOT NULL, "
        "size INTEGER, mtime_ns INTEGER, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, "
        "PRIMARY KEY (sha256_b64url, source_path))"
    )
    conn.execute(
        "INSERT INTO sources (sha256_b64url, source_path, size, mtime_ns, first_seen_at, last_seen_at) "
        "VALUES ('OLD', '/x.jpg', 1, 1, 'now', 'now')"
    )
    conn.commit()
    conn.close()

    with Catalog(db) as cat:
        assert cat.get_by_sha256("OLD") is not None
        assert cat.tiers_for("OLD") == (0, 0)


def test_legacy_catalog_with_rows_and_no_sources_raises(tmp_path):
    """A pre-redesign catalog (photo rows, no `sources`, no schema_version)
    must never be opened in place -- see `LegacyCatalogError`. Opening it
    would make every existing row look unenriched and undated, and the next
    enrichment pass would relocate the whole organized tree into Undated/.
    """
    import sqlite3

    from imageharbor.catalog import Catalog, LegacyCatalogError

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_OLD_PHOTOS_TABLE_DDL)
    conn.execute(
        "INSERT INTO photos (sha256_b64url, original_path, created_at) VALUES ('OLD', '/x.jpg', 'now')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(LegacyCatalogError, match="docs/rebuild.md"):
        Catalog(db)


def test_legacy_catalog_empty_opens(tmp_path):
    """An old-schema DB with a `photos` table but zero rows is not a hazard
    (there is nothing to misinterpret) -- it must open normally and get
    stamped at the current schema version."""
    import sqlite3

    from imageharbor.catalog import SCHEMA_VERSION, Catalog

    db = tmp_path / "legacy_empty.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_OLD_PHOTOS_TABLE_DDL)
    conn.commit()
    conn.close()

    with Catalog(db) as cat:
        assert cat.count() == 0
        row = cat._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert row is not None and row["value"] == SCHEMA_VERSION


def test_fresh_catalog_carries_schema_version(tmp_path):
    """A brand-new catalog must open normally and be stamped at the current
    schema version."""
    from imageharbor.catalog import SCHEMA_VERSION, Catalog

    with Catalog(tmp_path / "fresh.db") as cat:
        row = cat._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        assert row is not None and row["value"] == SCHEMA_VERSION


def test_reopening_a_v2_catalog_is_a_noop(tmp_path):
    """Reopening an already-stamped catalog must not raise and must not
    rewrite the stamp."""
    from imageharbor.catalog import SCHEMA_VERSION, Catalog

    db = tmp_path / "v2.db"
    with Catalog(db) as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg")
        cat.record_source("D1", "/a.jpg", 1, 1)

    with Catalog(db) as cat:
        assert cat.get_by_sha256("D1") is not None
        rows = cat._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["value"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Google Takeout ingestion tables
# ---------------------------------------------------------------------------


def test_takeout_archive_roundtrip(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/nas/t1.zip", size=79, mtime_ns=1, member_count=196
        )
        row = cat.takeout_archive_get("A" * 43)
        assert row["status"] == "partial"
        assert row["member_count"] == 196
        assert row["last_path"] == "/nas/t1.zip"


def test_takeout_archive_stat_fast_path(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/nas/t1.zip", size=79, mtime_ns=1
        )
        assert cat.takeout_archive_get_by_stat("/nas/t1.zip", 79, 1)["archive_id"] == "A" * 43
        assert cat.takeout_archive_get_by_stat("/nas/t1.zip", 79, 2) is None
        assert cat.takeout_archive_get_by_stat("/other.zip", 79, 1) is None


def test_takeout_archive_upsert_updates_location_not_identity(tmp_path, monkeypatch) -> None:
    """A moved archive keeps its id; only where it lives changes.

    The two upserts below run back-to-back in the same process, microseconds
    apart, so a real wall-clock read could plausibly land on the same string
    twice -- which would let a broken `first_seen_at = excluded.first_seen_at`
    slip past an equality assertion by accident. `_now_iso` is patched to
    return two distinct, known timestamps so the `first_seen_at`/`last_seen_at`
    assertions below are genuinely discriminating rather than clock-luck.
    """
    timestamps = iter(["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"])
    monkeypatch.setattr(catalog_module, "_now_iso", lambda: next(timestamps))
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/old/t1.zip", size=79, mtime_ns=1
        )
        first_seen_at = cat.takeout_archive_get("A" * 43)["first_seen_at"]
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/new/t1.zip", size=79, mtime_ns=9
        )
        assert len(cat.takeout_archives_all()) == 1
        row = cat.takeout_archive_get("A" * 43)
        assert row["last_path"] == "/new/t1.zip"
        assert row["first_seen_at"] == first_seen_at
        assert row["last_seen_at"] != first_seen_at


def test_takeout_member_add_never_resets_a_terminal_status(tmp_path) -> None:
    """Re-surveying an archive must not drag ingested members back to pending."""
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="a/b.jpg", kind="image",
            size=10, crc32=1, status="pending",
        )
        cat.takeout_member_set("A" * 43, "a/b.jpg", status="ingested", sha256_b64url="D" * 43)
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="a/b.jpg", kind="image",
            size=10, crc32=1, status="pending",
        )
        rows = cat.takeout_members_all("A" * 43)
        assert len(rows) == 1
        assert rows[0]["status"] == "ingested"
        assert rows[0]["sha256_b64url"] == "D" * 43


def test_takeout_members_pending_returns_pending_and_failed_only(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        for name, status in (
            ("p.jpg", "pending"), ("f.jpg", "failed"), ("i.jpg", "ingested"),
            ("d.jpg", "duplicate"), ("v.mp4", "deferred"), ("m.json", "parsed"),
            ("o.txt", "ignored"), ("t.jpg", "skipped_trash"),
        ):
            cat.takeout_member_add(
                archive_id="A" * 43, member_path=name, kind="image",
                size=1, crc32=1, status=status,
            )
        assert {r["member_path"] for r in cat.takeout_members_pending("A" * 43)} == {
            "p.jpg", "f.jpg",
        }


def test_takeout_members_unskip_trash(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="Trash/x.jpg", kind="image",
            size=1, crc32=1, status="skipped_trash",
        )
        assert cat.takeout_members_unskip_trash("A" * 43) == 1
        assert cat.takeout_members_pending("A" * 43)[0]["member_path"] == "Trash/x.jpg"


def test_takeout_members_unskip_trash_is_kind_aware(tmp_path) -> None:
    """A trash image goes back to 'pending' (a work item); a trash sidecar
    goes back to 'parsed', never 'pending' -- nothing drains a pending
    metadata/album row, so resetting it that way would strand it in the
    queue forever and the archive would never reach 'complete'.
    """
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="Trash/x.jpg", kind="image",
            size=1, crc32=1, status="skipped_trash",
        )
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="Trash/x.jpg.json", kind="metadata",
            size=1, crc32=1, status="skipped_trash",
        )
        assert cat.takeout_members_unskip_trash("A" * 43) == 2
        rows = {r["member_path"]: r["status"] for r in cat.takeout_members_all("A" * 43)}
        assert rows["Trash/x.jpg"] == "pending"
        assert rows["Trash/x.jpg.json"] == "parsed"


def test_takeout_status_counts(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/t1.zip", size=1, mtime_ns=1, status="complete"
        )
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="a.jpg", kind="image",
            size=1, crc32=1, status="ingested",
        )
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="b.mp4", kind="video",
            size=1, crc32=1, status="deferred",
        )
        counts = cat.takeout_status_counts()
        assert counts["archives"]["complete"] == 1
        assert counts["members"]["ingested"] == 1
        assert counts["members"]["deferred"] == 1
        # a.jpg was ingested with no sidecar recorded
        assert counts["missing_metadata"] == 1


def test_takeout_tables_do_not_bump_the_schema_version(tmp_path) -> None:
    from imageharbor.catalog import SCHEMA_VERSION

    assert SCHEMA_VERSION == "2"
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/t.zip", size=1, mtime_ns=1
        )
    # Reopening must not raise LegacyCatalogError or lose the row.
    with Catalog(tmp_path / "c.db") as cat:
        assert cat.takeout_archive_get("A" * 43) is not None


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def test_run_start_returns_id_with_ended_at_null_and_is_unfinished(catalog: Catalog) -> None:
    run_id = catalog.run_start("facts")
    assert isinstance(run_id, int)
    unfinished = catalog.unfinished_runs()
    assert len(unfinished) == 1
    assert unfinished[0]["id"] == run_id
    assert unfinished[0]["ended_at"] is None
    assert unfinished[0]["kind"] == "facts"


def test_run_finish_populates_counters_and_clears_unfinished(catalog: Catalog) -> None:
    run_id = catalog.run_start("enrich")
    catalog.run_finish(
        run_id,
        scanned=10,
        copied=3,
        duplicates=2,
        errors=1,
        enriched=4,
        enrich_failed=1,
        breaker_state="OPEN",
        paused=False,
    )
    assert catalog.unfinished_runs() == []
    rows = catalog.recent_runs()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == run_id
    assert row["ended_at"] is not None
    assert row["scanned"] == 10
    assert row["copied"] == 3
    assert row["duplicates"] == 2
    assert row["errors"] == 1
    assert row["enriched"] == 4
    assert row["enrich_failed"] == 1
    assert row["breaker_state"] == "OPEN"
    assert row["paused"] == 0  # stored as an INTEGER, False -> 0


def test_recent_runs_is_newest_first_and_respects_limit(catalog: Catalog) -> None:
    ids = []
    for i in range(5):
        run_id = catalog.run_start("facts")
        catalog.run_finish(
            run_id, scanned=i, copied=0, duplicates=0, errors=0,
            enriched=0, enrich_failed=0, breaker_state="CLOSED", paused=False,
        )
        ids.append(run_id)
    rows = catalog.recent_runs(limit=3)
    assert len(rows) == 3
    assert [r["id"] for r in rows] == list(reversed(ids))[:3]


def test_unfinished_runs_only_returns_runs_with_ended_at_null(catalog: Catalog) -> None:
    finished_id = catalog.run_start("facts")
    catalog.run_finish(
        finished_id, scanned=1, copied=0, duplicates=0, errors=0,
        enriched=0, enrich_failed=0, breaker_state="CLOSED", paused=False,
    )
    unfinished_id = catalog.run_start("enrich")
    unfinished = catalog.unfinished_runs()
    assert [r["id"] for r in unfinished] == [unfinished_id]


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


def test_setting_roundtrip(catalog: Catalog) -> None:
    assert catalog.setting_get("interval") is None
    catalog.setting_set("interval", "120")
    assert catalog.setting_get("interval") == "120"


def test_setting_set_overwrites_existing_value(catalog: Catalog) -> None:
    catalog.setting_set("enrich", "1")
    catalog.setting_set("enrich", "0")
    assert catalog.setting_get("enrich") == "0"


def test_setting_delete_removes_the_row_not_blanks_it(catalog: Catalog) -> None:
    """A revert must DELETE the row so the env var is consulted again -- see
    the `settings` table comment. Writing an empty string instead would look
    identical today and silently shadow every future compose change."""
    catalog.setting_set("interval", "300")
    catalog.setting_delete("interval")
    assert catalog.setting_get("interval") is None
    # Confirm the row is truly gone, not present with an empty/old value.
    row = catalog._conn.execute(
        "SELECT * FROM settings WHERE key='interval'"
    ).fetchone()
    assert row is None


def test_setting_delete_of_missing_key_is_a_noop(catalog: Catalog) -> None:
    catalog.setting_delete("does-not-exist")  # must not raise
    assert catalog.setting_get("does-not-exist") is None


def test_settings_all_returns_dict_of_all_settings(catalog: Catalog) -> None:
    catalog.setting_set("interval", "60")
    catalog.setting_set("enrich", "1")
    assert catalog.settings_all() == {"interval": "60", "enrich": "1"}


def test_settings_all_empty(catalog: Catalog) -> None:
    assert catalog.settings_all() == {}


# ---------------------------------------------------------------------------
# busy timeout
#
# No test here: `sqlite3.connect()`'s default `timeout=5.0` already calls
# `sqlite3_busy_timeout(db, 5000)` on every connection, so `PRAGMA
# busy_timeout` reads 5000 whether or not `Catalog.__init__` sets it
# explicitly -- a test asserting that value would pass identically with the
# line removed. See the mutation-testing note in the Task 2 report.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# runs / settings: schema version and legacy safety
# ---------------------------------------------------------------------------


def test_a_catalog_predating_runs_and_settings_reopens_without_raising(tmp_path):
    """A catalog written before this change has `photos`/`sources` but no
    `runs`/`settings` tables at all. Opening it must add both tables
    additively (not raise `LegacyCatalogError`), and they must be usable
    immediately.
    """
    import sqlite3

    from imageharbor.catalog import Catalog

    db = tmp_path / "pre_runs_settings.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_OLD_PHOTOS_TABLE_DDL)
    conn.execute(
        "INSERT INTO photos (sha256_b64url, original_path, created_at) VALUES ('OLD', '/x.jpg', 'now')"
    )
    conn.execute(
        "CREATE TABLE sources (sha256_b64url TEXT NOT NULL, source_path TEXT NOT NULL, "
        "size INTEGER, mtime_ns INTEGER, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, "
        "PRIMARY KEY (sha256_b64url, source_path))"
    )
    conn.execute(
        "INSERT INTO sources (sha256_b64url, source_path, size, mtime_ns, first_seen_at, last_seen_at) "
        "VALUES ('OLD', '/x.jpg', 1, 1, 'now', 'now')"
    )
    conn.commit()
    conn.close()

    with Catalog(db) as cat:  # must not raise LegacyCatalogError
        assert cat.get_by_sha256("OLD") is not None
        run_id = cat.run_start("facts")
        assert cat.unfinished_runs()[0]["id"] == run_id
        cat.setting_set("interval", "120")
        assert cat.setting_get("interval") == "120"


def test_runs_and_settings_tables_do_not_bump_the_schema_version(tmp_path: Path) -> None:
    from imageharbor.catalog import SCHEMA_VERSION

    assert SCHEMA_VERSION == "2"
    with Catalog(tmp_path / "c.db") as cat:
        run_id = cat.run_start("facts")
        cat.setting_set("interval", "120")
    # Reopening must not raise LegacyCatalogError or lose the rows.
    with Catalog(tmp_path / "c.db") as cat:
        assert cat.setting_get("interval") == "120"
        assert cat.unfinished_runs()[0]["id"] == run_id


# ---------------------------------------------------------------------------
# Concurrency -- CRITICAL finding #2
#
# `cli.py` passes ONE `Catalog` to `ControlPlane`, the dashboard HTTP server
# (`daemon_threads = True`, one thread per request), and the watcher loop.
# `check_same_thread=False` PERMITS concurrent access from multiple threads;
# it does not make it safe. Measured under realistic load before the fix (4
# pollers at 5 Hz + pause POSTs, 25s): 55 exceptions out of the writer,
# including "cannot commit - no transaction is active", "cannot start a
# transaction within a transaction", "another row available", and
# "SystemError: error return without exception set" out of `run_finish`.
# `Catalog.lock` (a `threading.RLock`) now guards every public method.
#
# This test is deterministic on purpose -- a FIXED thread count and a FIXED
# operation count per thread, not a timed loop -- so it fails the same way
# every time it fails, rather than being a flaky proxy for real contention.
# ---------------------------------------------------------------------------


def test_concurrent_reads_and_writes_from_multiple_threads_raise_nothing(tmp_path: Path) -> None:
    import queue
    import threading

    THREADS_EACH = 4          # writer threads and reader threads, each
    OPS_PER_THREAD = 40        # fixed, not time-based

    cat = Catalog(tmp_path / "concurrent.db")
    errors: "queue.Queue[BaseException]" = queue.Queue()

    def _writer(thread_id: int) -> None:
        try:
            for i in range(OPS_PER_THREAD):
                digest = f"W{thread_id}-{i}"
                cat.upsert(
                    sha256_b64url=digest,
                    original_path=f"/w{thread_id}/{i}.jpg",
                    organized_path=f"/lib/w{thread_id}/{i}.jpg",
                )
                run_id = cat.run_start("facts")
                cat.run_finish(
                    run_id, scanned=1, copied=1, duplicates=0, errors=0,
                    enriched=0, enrich_failed=0, breaker_state="CLOSED", paused=False,
                )
                cat.setting_set("interval", str(100 + i))
                cat.record_file_failure(f"/w{thread_id}/{i}.jpg", 10, i, "boom")
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertion below
            errors.put(exc)

    def _reader(thread_id: int) -> None:
        try:
            for _ in range(OPS_PER_THREAD):
                cat.recent_runs(limit=5)
                cat.unfinished_runs()
                cat.iter_unenriched()
                cat.count_unenriched()
                cat.settings_all()
                list(cat.iter_all())
        except BaseException as exc:  # noqa: BLE001
            errors.put(exc)

    threads = [
        threading.Thread(target=_writer, args=(n,), name=f"writer-{n}")
        for n in range(THREADS_EACH)
    ] + [
        threading.Thread(target=_reader, args=(n,), name=f"reader-{n}")
        for n in range(THREADS_EACH)
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert all(not t.is_alive() for t in threads), "a thread did not finish within 60s"

        collected: list[BaseException] = []
        while not errors.empty():
            collected.append(errors.get_nowait())
        assert collected == [], (
            f"{len(collected)} exception(s) raised by concurrent Catalog access: "
            f"{[repr(e) for e in collected]}"
        )
    finally:
        cat.close()
