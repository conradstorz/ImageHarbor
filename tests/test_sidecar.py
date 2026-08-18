"""Tests for sidecar I/O: atomicity and corrupt-file handling."""

from __future__ import annotations

import json
from pathlib import Path

from imageharbor.sidecar import merge_sidecar, read_sidecar, sidecar_path_for


def _img(tmp_path: Path) -> Path:
    p = tmp_path / "2015-03-09_abc.jpg"
    p.write_bytes(b"x")
    return p


def test_merge_creates_then_accretes(tmp_path: Path) -> None:
    img = _img(tmp_path)
    merge_sidecar(img, {"exif": {"Make": "Canon"}})
    merge_sidecar(img, {"exif": {"Model": "5D"}})
    doc = read_sidecar(img)
    assert doc["exif"] == {"Make": "Canon", "Model": "5D"}


def test_a_corrupt_sidecar_is_quarantined_not_overwritten(tmp_path: Path) -> None:
    """Treating a corrupt sidecar as empty would destroy data under the new rule.

    The old behavior logged and returned {}, so the next merge silently wrote a
    fresh document over whatever the unreadable bytes contained.
    """
    img = _img(tmp_path)
    path = sidecar_path_for(img)
    path.write_text('{"exif": {"Make": "Canon"} TRUNCATED', encoding="utf-8")

    merge_sidecar(img, {"exif": {"Model": "5D"}})

    quarantined = list(tmp_path.glob("*.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "TRUNCATED" in quarantined[0].read_text(encoding="utf-8")
    assert read_sidecar(img)["exif"] == {"Model": "5D"}


def test_a_hand_edit_survives_a_merge(tmp_path: Path) -> None:
    img = _img(tmp_path)
    merge_sidecar(img, {"exif": {"Make": "Canon"}})
    doc = read_sidecar(img)
    doc["my_note"] = "grandma's camera"
    sidecar_path_for(img).write_text(json.dumps(doc), encoding="utf-8")

    merge_sidecar(img, {"exif": {"Model": "5D"}})
    assert read_sidecar(img)["my_note"] == "grandma's camera"


def test_repeated_merge_is_byte_identical(tmp_path: Path) -> None:
    img = _img(tmp_path)
    update = {"sources": [{"path": "/a.jpg", "folder": "d", "first_seen": "T", "last_seen": "T"}]}
    merge_sidecar(img, update)
    first = sidecar_path_for(img).read_bytes()
    merge_sidecar(img, update)
    assert sidecar_path_for(img).read_bytes() == first


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    img = _img(tmp_path)
    merge_sidecar(img, {"exif": {"Make": "Canon"}})
    assert list(tmp_path.glob("*.tmp")) == []


def test_sidecar_path_appends_json_rather_than_replacing_a_suffix(tmp_path: Path) -> None:
    """A stem containing dots must not lose part of its name."""
    img = tmp_path / "2019-07-04-v1.2_abc.jpg"
    assert sidecar_path_for(img).name == "2019-07-04-v1.2_abc.json"


def test_read_of_a_missing_sidecar_is_empty(tmp_path: Path) -> None:
    assert read_sidecar(tmp_path / "nope.jpg") == {}


def test_bytes_valued_exif_serializes_as_text_not_python_repr(tmp_path: Path) -> None:
    """Real EXIF carries raw bytes (ExifVersion, SceneType, MakerNote).

    A bare default=str would not raise, but would write Python repr syntax
    into the file -- "b'0230'" instead of "0230" -- and a sidecar is meant to
    be portable and human-readable.
    """
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"exif": {"ExifVersion": b"0230", "MakerNote": b"\x00\xff"}})
    raw = sidecar_path_for(img).read_text(encoding="utf-8")
    assert "b'" not in raw
    assert read_sidecar(img)["exif"]["ExifVersion"] == "0230"


def test_a_scalar_replaced_by_a_dict_does_not_corrupt(tmp_path: Path) -> None:
    """Type mismatches between runs resolve cleanly rather than raising.

    A v1 sidecar can hold a bare string where v2 expects a tiered block, so
    this is a real migration shape, not a hypothetical one. `_coerce_block`
    lifts the bare scalar to a tier-0 block (`value`/`tier`/`source`)
    rather than discarding it, so the first merge's scalar is real evidence
    on record -- and when the second merge's higher-tier block wins, that
    tier-0 reading is relocated into `history` like any other superseded
    value, not lost.
    """
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"date": "2019-07-04"})
    merge_sidecar(img, {"date": {"value": "2019-07-04", "tier": 40}})

    block = read_sidecar(img)["date"]
    assert block["value"] == "2019-07-04"
    assert block["tier"] == 40
    assert any(h.get("tier") == 0 for h in block["history"]), (
        "the bare scalar from the first merge must survive in history"
    )
