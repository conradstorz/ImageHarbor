from __future__ import annotations
from pathlib import Path
import pytest
from imageharbor.catalog import Catalog
from imageharbor import concept_map


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    yield cat
    cat.close()


def test_static_seed_from_pcs_subnames() -> None:
    # legacy sub-category "beach" (330) -> its class 300; "portraits" -> 100
    assert concept_map.STATIC_SEED["beach"] == "300"
    assert concept_map.STATIC_SEED["portrait"] == "100"  # normalized (no trailing s)


def test_class_for_static_hit(catalog: Catalog) -> None:
    assert concept_map.class_for("beach", [], "", catalog) == "300"
    assert concept_map.class_for("marching band", [], "", catalog) == "500"  # synonym


def test_class_for_object_or_scene_keyword(catalog: Catalog) -> None:
    assert concept_map.class_for("xyzzy", ["dog"], "", catalog) == "200"
    assert concept_map.class_for("xyzzy", [], "at the beach", catalog) == "300"


def test_class_for_miss_returns_none(catalog: Catalog) -> None:
    assert concept_map.class_for("qwertyunknownthing", [], "", catalog) is None


def test_remember_rejects_invalid_class(catalog: Catalog) -> None:
    from imageharbor.taxonomy import _normalize_label
    concept_map.remember(catalog, "gizmo", "999")  # not a fixed top-level class
    assert catalog.learned_concept_get(_normalize_label("gizmo")) is None
    assert concept_map.class_for("gizmo", [], "", catalog) is None


def test_class_for_ignores_invalid_learned_value(catalog: Catalog) -> None:
    from imageharbor.taxonomy import _normalize_label
    # Poison the store directly (bypassing remember's guard) with a bad code.
    catalog.learned_concept_remember(_normalize_label("gremlin"), "zzz")
    assert concept_map.class_for("gremlin", [], "", catalog) is None  # ignored, not a hit


def test_learned_beats_static_and_roundtrips(catalog: Catalog) -> None:
    concept_map.remember(catalog, "Beach", "600")  # user/AI override
    assert concept_map.class_for("beach", [], "", catalog) == "600"  # learned wins
    concept_map.remember(catalog, "novel gizmo", "200")
    assert concept_map.class_for("novel gizmo", [], "", catalog) == "200"
