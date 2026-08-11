"""Tests for imageharbor.catalog."""

from pathlib import Path

import pytest

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


def test_existing_catalog_gains_new_columns(tmp_path):
    """An older DB must open and upgrade without losing rows."""
    import sqlite3
    from imageharbor.catalog import Catalog

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "sha256_b64url TEXT NOT NULL UNIQUE, original_path TEXT NOT NULL, "
        "organized_path TEXT, pcs_version TEXT NOT NULL DEFAULT '1', "
        "pcs_primary TEXT NOT NULL DEFAULT '900', pcs_name TEXT NOT NULL DEFAULT 'miscellaneous', "
        "secondary_tags TEXT NOT NULL DEFAULT '[]', ai_caption TEXT NOT NULL DEFAULT '', "
        "objects TEXT NOT NULL DEFAULT '[]', ocr_text TEXT NOT NULL DEFAULT '', "
        "exif TEXT NOT NULL DEFAULT '{}', model_version TEXT NOT NULL DEFAULT 'unknown', "
        "processing_history TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, processed_at TEXT)"
    )
    conn.execute(
        "INSERT INTO photos (sha256_b64url, original_path, created_at) VALUES ('OLD', '/x.jpg', 'now')"
    )
    conn.commit()
    conn.close()

    with Catalog(db) as cat:
        assert cat.get_by_sha256("OLD") is not None
        assert cat.tiers_for("OLD") == (0, 0)
