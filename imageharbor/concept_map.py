"""Deterministic-first concept -> class mapping (self-learning).

A static seed (bootstrapped from the legacy PCS sub-categories) maps common
subject/object keywords to one of the 9 fixed top-level classes. A learned store
in the catalog memoizes every AI-decided subject->class, so repeats become
deterministic hits and stop calling the AI. Genuine misses return None and the
pipeline falls through to the text-only pick_class step.
"""
from __future__ import annotations

import logging

from .catalog import Catalog
from .pcs import PCS_CATEGORIES
from .taxonomy import _normalize_label

logger = logging.getLogger(__name__)


def _build_static_seed() -> dict[str, str]:
    seed: dict[str, str] = {}
    # Every legacy sub-category name -> its top-level class code.
    for code, cat in PCS_CATEGORIES.items():
        if cat.parent is not None:
            seed[_normalize_label(cat.name)] = str((code // 100) * 100)
    # Curated common subjects / synonyms.
    extra = {
        "band": "500", "marching band": "500", "parade": "500", "wedding": "500",
        "concert": "500", "party": "500", "festival": "500", "graduation": "500",
        "game": "500", "match": "500",
        "dog": "200", "cat": "200", "puppy": "200", "kitten": "200",
        "horse": "200", "fish": "200",
        "sunset": "600", "sunrise": "600", "flower": "600", "tree": "600",
        "forest": "600", "river": "600", "lake": "600", "waterfall": "600",
        "car": "400", "truck": "400", "boat": "400", "plane": "400",
        "train": "400", "bicycle": "400", "motorcycle": "400",
        "baby": "100", "child": "100", "family": "100", "selfie": "100",
        "receipt": "700", "document": "700", "sign": "700", "menu": "700",
        "house": "800", "building": "800", "church": "800", "bridge": "800",
        "food": "900", "meal": "900",
    }
    for k, v in extra.items():
        seed.setdefault(_normalize_label(k), v)
    return seed


STATIC_SEED: dict[str, str] = _build_static_seed()

# The 9 fixed top-level class codes. Learned/returned classes are validated
# against these so an invalid value can never be stored or treated as a hit
# (which would permanently poison a subject's mapping).
_VALID_CLASSES: frozenset[str] = frozenset(
    str(code) for code, cat in PCS_CATEGORIES.items() if cat.parent is None
)


def class_for(
    primary_subject: str, objects: list[str], scene: str, catalog: Catalog
) -> str | None:
    """Return a top-level class code for this content, or None (a miss)."""
    subj = _normalize_label(primary_subject)
    # 1. Learned store wins (exact normalized subject) — but only if it's a
    #    real class; an invalid stored value is ignored, not treated as a hit.
    learned = catalog.learned_concept_get(subj)
    if learned in _VALID_CLASSES:
        return learned
    # 2. Static seed: the subject, then object/scene keywords.
    if subj in STATIC_SEED:
        return STATIC_SEED[subj]
    for token in list(objects) + scene.split():
        norm = _normalize_label(token)
        if norm in STATIC_SEED:
            return STATIC_SEED[norm]
    return None


def remember(catalog: Catalog, primary_subject: str, class_code: str) -> None:
    """Memoize an AI-decided subject -> class so the next repeat is deterministic."""
    if class_code not in _VALID_CLASSES:
        logger.warning(
            "refusing to learn %r -> invalid class %r (not a fixed top-level class)",
            primary_subject, class_code,
        )
        return
    subj = _normalize_label(primary_subject)
    logger.info("concept-map learned: %r -> class %s", subj, class_code)
    catalog.learned_concept_remember(subj, class_code)
