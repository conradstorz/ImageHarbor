"""Composition tests for `Catalog.takeout` (the `TakeoutStore` extraction)."""

from __future__ import annotations

from imageharbor.catalog import Catalog


def test_catalog_composes_a_takeout_store_on_the_same_connection(catalog: Catalog) -> None:
    from imageharbor.takeout.store import TakeoutStore

    assert isinstance(catalog.takeout, TakeoutStore)
    # same connection + same lock object -- the single-writer invariant:
    assert catalog.takeout._conn is catalog._conn
    assert catalog.takeout.lock is catalog.lock


def test_no_takeout_methods_remain_on_catalog(catalog: Catalog) -> None:
    assert not [m for m in dir(catalog) if m.startswith("takeout_")]
