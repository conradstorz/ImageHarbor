"""The `machine_generated` flag: a decorated camera name is recorded as such.

FLAG ONLY: the tier verdict (human filename, tier 30) is unchanged.
"""

import json
from pathlib import Path

import pytest

from imageharbor import sidecar_schema, tiers
from imageharbor.ai_classifier import ContentDescription, StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.descriptor import resolve_descriptor, strip_google_decorations
from imageharbor.enrichment import enrich_library
from imageharbor.pipeline import Pipeline

FLAGGED = [
    "IMG_1234-edited",
    "IMG_0042(1)",
    "IMG_0042(12)",
    "IMG_1234-edited(1)",
    "PXL_20200101_123456789-ANIMATION",
    "PXL_20200101_123456789-MOTION",
    "IMG_20190704_123456~2",
    "DSC0042-edited",
    "MVIMG_20190704_123456-ANIMATION",
    "img_1234-EDITED",
]

NOT_FLAGGED_HUMAN = [
    "Beach trip-edited",
    "Birthday(2)",
    "Sam_1(1)",
    "Emma's graduation-ANIMATION",
    "Screenshot - grandpas text-edited",
    "Kratz Cropped Scans_116~2",
]


@pytest.mark.parametrize("stem", FLAGGED)
def test_decorated_camera_name_is_flagged_but_keeps_tier_30(stem):
    d = resolve_descriptor(Path(f"{stem}.jpg"))
    assert d.machine_generated is True
    assert d.tier == tiers.DESC_HUMAN_FILENAME


@pytest.mark.parametrize("stem", NOT_FLAGGED_HUMAN)
def test_decorated_human_name_is_not_flagged(stem):
    d = resolve_descriptor(Path(f"{stem}.jpg"))
    assert d.machine_generated is False
    assert d.tier == tiers.DESC_HUMAN_FILENAME


@pytest.mark.parametrize("stem", ["IMG_1234", "PXL_20200101_123456789", "2019-07-04"])
def test_plain_camera_name_is_tier_zero_and_unflagged(stem):
    d = resolve_descriptor(Path(f"{stem}.jpg"))
    assert d.tier == tiers.DESC_NONE
    assert d.machine_generated is False


def test_human_name_is_unflagged():
    assert resolve_descriptor(Path("Emma's graduation.jpg")).machine_generated is False


def test_title_with_decorated_camera_name_is_flagged():
    d = resolve_descriptor(Path("truncated.jpg"), original_name="IMG_1234-edited.jpg")
    assert d.machine_generated is True


def test_strip_only_removes_decorations():
    assert strip_google_decorations("IMG_1234-edited(1)") == "IMG_1234"
    assert strip_google_decorations("Beach trip") == "Beach trip"


# -- sidecar merge -----------------------------------------------------------

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-02-02T00:00:00Z"


def _desc(flag):
    block = {"value": "img-1234-edited", "tier": 30, "source": "human_filename"}
    if flag:
        block["machine_generated"] = True
    return {"descriptor": block}


def test_flagged_merge_is_idempotent():
    once = sidecar_schema.merge(sidecar_schema.merge({}, _desc(True), observed_at=T0), _desc(True), observed_at=T1)
    twice = sidecar_schema.merge(once, _desc(True), observed_at=T1)
    assert once == twice
    assert once["descriptor"]["machine_generated"] is True
    assert once["descriptor"]["history"] == []


def test_flag_is_added_to_an_existing_unflagged_block_without_history_growth():
    base = sidecar_schema.merge({}, _desc(False), observed_at=T0)
    once = sidecar_schema.merge(base, _desc(True), observed_at=T1)
    twice = sidecar_schema.merge(once, _desc(True), observed_at=T1)
    assert once["descriptor"]["machine_generated"] is True
    assert once == twice
    assert once["descriptor"]["history"] == []


def test_unflagged_update_never_drops_the_flag():
    base = sidecar_schema.merge({}, _desc(True), observed_at=T0)
    after = sidecar_schema.merge(base, _desc(False), observed_at=T1)
    assert after["descriptor"]["machine_generated"] is True
    again = sidecar_schema.merge(after, _desc(False), observed_at=T1)
    assert again == after


# -- pipeline ----------------------------------------------------------------


def _sidecars(dest: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in dest.rglob("*.json")}


def test_facts_pass_records_flag_and_rerun_is_byte_identical(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_1234-edited.jpg").write_bytes(b"machine")
    (src / "Beach trip-edited.jpg").write_bytes(b"human")
    dest = tmp_path / "dest"
    with Catalog(tmp_path / "c.db") as cat:
        Pipeline(src, dest, cat, write_sidecars=True).run()
        first = _sidecars(dest)
        assert len(first) == 2
        docs = [json.loads(b) for b in first.values()]
        flagged = [d for d in docs if d["descriptor"].get("machine_generated")]
        assert len(flagged) == 1
        assert flagged[0]["descriptor"]["tier"] == tiers.DESC_HUMAN_FILENAME
        for d in docs:
            if d not in flagged:
                assert "machine_generated" not in d["descriptor"]
        Pipeline(src, dest, cat, write_sidecars=True).run()
        assert _sidecars(dest) == first


class _Fixed(StubClassifier):
    def describe(self, image_path, exif_data=None):
        return ContentDescription(
            primary_subject="beach", scene="s", objects=[], caption="c",
            tags=[], ocr_text="", model_version="fixed-1",
        )


def test_enrichment_cannot_displace_a_flagged_tier_30_descriptor(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_1234-edited.jpg").write_bytes(b"machine")
    dest = tmp_path / "dest"
    with Catalog(tmp_path / "c.db") as cat:
        Pipeline(src, dest, cat, write_sidecars=True).run()
        before = {p.name for p in dest.rglob("*") if p.is_file()}
        report = enrich_library(cat, dest, _Fixed(), write_sidecars=True)
        assert report.renamed == 0
        assert {p.name for p in dest.rglob("*") if p.is_file()} == before
        doc = json.loads(next(dest.rglob("*.json")).read_bytes())
        assert doc["descriptor"]["tier"] == tiers.DESC_HUMAN_FILENAME
        assert doc["descriptor"]["machine_generated"] is True
