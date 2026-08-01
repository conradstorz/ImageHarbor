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


def test_snapshot_text_lists_categories(tax: Taxonomy) -> None:
    s = tax.snapshot_text()
    assert "100" in s and "people" in s
    assert "330" in s and "beach" in s
