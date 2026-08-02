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
