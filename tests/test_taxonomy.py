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
