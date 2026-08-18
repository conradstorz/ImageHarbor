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
