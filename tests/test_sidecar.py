"""Tests for cumulative sidecar merging."""

import json

from imageharbor.sidecar import (
    SIDECAR_SCHEMA_VERSION,
    merge_sidecar,
    read_sidecar,
    sidecar_path_for,
)


def test_sidecar_path_appends_json_rather_than_replacing_a_suffix(tmp_path):
    """A stem containing dots must not lose part of its name."""
    img = tmp_path / "2019-07-04-v1.2_abc.jpg"
    assert sidecar_path_for(img).name == "2019-07-04-v1.2_abc.json"


def test_merge_creates_and_stamps_schema_version(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    data = read_sidecar(img)
    assert data["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert data["identity"]["sha256_b64url"] == "D1"


def test_merge_is_cumulative_across_runs(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}, "date": {"tier": 40}})
    merge_sidecar(img, {"classification": {"pcs_code": "330"}})
    data = read_sidecar(img)
    assert data["identity"]["sha256_b64url"] == "D1"
    assert data["date"]["tier"] == 40
    assert data["classification"]["pcs_code"] == "330"


def test_merge_deep_merges_nested_dicts(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1", "size": 10}})
    merge_sidecar(img, {"identity": {"size": 20}})
    data = read_sidecar(img)
    assert data["identity"] == {"sha256_b64url": "D1", "size": 20}


def test_merge_preserves_hand_added_keys(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    path = sidecar_path_for(img)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["my_note"] = "grandma's kitchen, not a beach"
    data["identity"]["my_field"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")

    # The second merge must REWRITE the same nested dict the hand edit lives in
    # -- an update that only touched an unrelated key would not prove anything.
    merge_sidecar(img, {"identity": {"sha256_b64url": "D2"}, "classification": {"pcs_code": "330"}})
    after = read_sidecar(img)
    assert after["my_note"] == "grandma's kitchen, not a beach"
    assert after["identity"]["my_field"] == 1
    assert after["identity"]["sha256_b64url"] == "D2"


def test_lists_are_replaced_not_appended(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"sources": [{"path": "/a.jpg"}]})
    merge_sidecar(img, {"sources": [{"path": "/a.jpg"}, {"path": "/b.jpg"}]})
    assert len(read_sidecar(img)["sources"]) == 2


def test_read_of_a_missing_sidecar_is_empty(tmp_path):
    assert read_sidecar(tmp_path / "nope.jpg") == {}


def test_corrupt_sidecar_does_not_raise_and_is_rebuilt(tmp_path):
    img = tmp_path / "a.jpg"
    sidecar_path_for(img).write_text("{not json", encoding="utf-8")
    assert read_sidecar(img) == {}
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    assert read_sidecar(img)["identity"]["sha256_b64url"] == "D1"


def test_no_temp_file_is_left_behind(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_bytes_valued_exif_serializes_as_text_not_python_repr(tmp_path):
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


def test_a_scalar_replaced_by_a_dict_does_not_corrupt(tmp_path):
    """Type mismatches between runs replace cleanly rather than raising."""
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"date": "2019-07-04"})
    merge_sidecar(img, {"date": {"value": "2019-07-04", "tier": 40}})
    assert read_sidecar(img)["date"] == {"value": "2019-07-04", "tier": 40}
