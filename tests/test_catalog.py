"""Tests for imageharbor.catalog."""

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog


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
        pcs_primary=330,
        pcs_name="beach",
        ai_caption="A sunny beach",
        objects=["sand", "waves"],
    )
    row = catalog.get_by_sha256(digest)
    assert row is not None
    assert row["pcs_primary"] == 330
    assert row["ai_caption"] == "A sunny beach"
    assert row["original_path"] == "/photos/beach.jpg"


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


def test_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    with Catalog(db) as cat:
        cat.upsert(sha256_b64url="C" * 43, original_path="/cm.jpg")
        assert cat.count() == 1
    # After close, any further use would raise; just check it doesn't explode
