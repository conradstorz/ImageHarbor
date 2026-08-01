"""Tests for the extensible taxonomy."""
from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.taxonomy import Taxonomy, slug


@pytest.fixture()
def tax(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    t = Taxonomy(cat)
    t.ensure_seeded()
    yield t
    cat.close()


def test_slug() -> None:
    assert slug("Christmas Eve!") == "christmas-eve"


def test_seed_has_fixed_spine(tax: Taxonomy) -> None:
    tops = [n.code for n in tax.children(None)]
    for c in ("100", "200", "300", "400", "500", "600", "700", "800", "900"):
        assert c in tops
    assert tax.get("330").label == "beach"


def test_folder_path(tax: Taxonomy) -> None:
    assert tax.folder_path("300") == "300-places"
    assert tax.folder_path("330") == "300-places/330-beach"


def test_mint_next_sub_and_leaf(tax: Taxonomy) -> None:
    # 500-events currently has 510/520/530 seeded; next sub is 540
    code = tax.mint_child("500", "holidays")
    assert code == "540"
    assert tax.folder_path("540") == "500-events/540-holidays"
    # a leaf under 540
    leaf = tax.mint_child("540", "christmas")
    assert leaf == "541"
    assert tax.folder_path("541") == "500-events/540-holidays/541-christmas"


def test_mint_overflow_uses_tilde(tax: Taxonomy) -> None:
    # Fill all 9 integer leaves under 540, then overflow -> 540~1
    tax.mint_child("500", "holidays")  # 540
    for i in range(9):
        tax.mint_child("540", f"leaf{i}")   # 541..549
    overflow = tax.mint_child("540", "tenth")
    assert overflow == "540~1"
    assert tax.folder_path("540~1") == "500-events/540-holidays/540~1-tenth"


def test_mint_is_append_only(tax: Taxonomy) -> None:
    a = tax.mint_child("500", "alpha")   # 540
    b = tax.mint_child("500", "beta")    # 550
    assert a == "540" and b == "550"


def test_resolve_reuses_existing_child(tax: Taxonomy) -> None:
    # "beach" already exists as 330 under 300
    assert tax.resolve_or_create("300", "Beaches") == "330"  # normalized reuse


def test_resolve_mints_when_new(tax: Taxonomy) -> None:
    code = tax.resolve_or_create("500", "holidays")
    assert code == "540"
    # second time reuses
    assert tax.resolve_or_create("500", "holidays") == "540"


def test_resolve_adjudicator_merges_synonym(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")  # 540
    calls = []
    def adj(label, candidates):
        calls.append((label, tuple(candidates)))
        return "holidays"  # the model says festivities == holidays
    code = tax.resolve_or_create("500", "festivities", adjudicator=adj)
    assert code == "540"          # reused, not minted
    assert calls                  # adjudicator consulted
    assert "festivities" in tax.get("540").aliases  # alias recorded


def test_resolve_no_adjudicator_mints_new(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")           # 540
    code = tax.resolve_or_create("500", "festivities")  # no adjudicator
    assert code == "550"  # minted as new sibling


def test_resolve_adjudicator_exception_falls_back_to_mint(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")  # 540

    def boom(label, candidates):
        raise RuntimeError("Jetson is down")

    code = tax.resolve_or_create("500", "festivities", adjudicator=boom)
    assert code == "550"  # minted as new sibling, exception did not propagate


def test_resolve_invalid_top_parent_falls_back_to_900(tax: Taxonomy) -> None:
    # A backend may return a top_parent that is not one of the 9 fixed classes
    # (e.g. a label like "events" or an out-of-range "999"). It must resolve
    # under 900 (miscellaneous), NOT mint an orphan under a nonexistent parent.
    code = tax.resolve_or_create("events", "whatever")
    node = tax.get(code)
    assert node is not None
    assert node.parent_code == "900"
    # No node anywhere should have "events" as its parent.
    assert all(n.parent_code != "events" for n in tax.children("900"))
    assert tax.get("events") is None


def test_resolve_invalid_sub_and_top_parent_falls_back_to_900(tax: Taxonomy) -> None:
    # Both a truthy-but-nonexistent sub_parent AND an invalid non-numeric
    # top_parent: the RESOLVED target must be validated so we never mint an
    # unparseable orphan (e.g. "events~1") that would break filename verify.
    code = tax.resolve_or_create("events", "whatever", sub_parent="540")
    node = tax.get(code)
    assert node is not None
    assert node.parent_code == "900"
    # No node anywhere should be parented on the bogus "events" or "540" codes.
    all_parents = {n.parent_code for n in tax.children("900")}
    assert "events" not in all_parents and "540" not in all_parents
    assert tax.get("events") is None
    assert tax.get("540") is None


def test_resolve_sub_parent_places_leaf(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")  # 540
    code = tax.resolve_or_create("500", "christmas", sub_parent="540")
    assert code == "541"


def test_merge_redirects_future_resolution(tax: Taxonomy) -> None:
    a = tax.resolve_or_create("500", "holidays")     # 540
    b = tax.resolve_or_create("500", "festivities")  # 550
    tax.merge(b, a)
    assert tax.resolve_alias(b) == a
    # a future exact-hit on the merged label redirects
    assert tax.resolve_or_create("500", "festivities") == a


def test_normalize_singularizes_plural_es() -> None:
    # "Beaches" -> lowercase, strip trailing "es" -> "beach" (why "Beaches"
    # reuses the seeded 330 "beach" leaf).
    assert Taxonomy._normalize("Beaches") == "beach"


def test_normalize_singularizes_plural_s() -> None:
    # "Holidays" -> "holiday" (trailing "s" dropped for words len>3).
    assert Taxonomy._normalize("Holidays") == "holiday"


def test_normalize_lossy_singularization_collision() -> None:
    # KNOWN LIMITATION, not desired behavior: the naive singularization is lossy
    # and collides unrelated words. "wines" -> "win" (strip "es") and
    # "wins" -> "win" (strip "s") both normalize to the same token. Pinned here
    # so any future change to _normalize is a conscious one.
    assert Taxonomy._normalize("wines") == Taxonomy._normalize("wins")


def test_snapshot_text_lists_categories(tax: Taxonomy) -> None:
    s = tax.snapshot_text()
    assert "100" in s and "people" in s
    assert "330" in s and "beach" in s
