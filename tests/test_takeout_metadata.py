"""Tests for the Google Takeout per-media JSON parser.

The first payload below is verbatim from the real export this design was
calibrated against (takeout-20230618T004316Z-001.zip, AlbumArchive schema).
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from imageharbor.takeout.metadata import (
    EMPTY,
    parse_album_metadata,
    parse_photo_metadata,
)

ALBUM_ARCHIVE_PAYLOAD = b"""{
  "title": "2015-03-09.jpg",
  "imageViews": "12",
  "creationTime":   { "timestampSeconds": "1425920628", "formatted": "Mar 9, 2015, 5:03:48 PM UTC" },
  "photoTakenTime": { "timestampSeconds": "1425905792", "formatted": "Mar 9, 2015, 12:56:32 PM UTC" },
  "geoData": { "latitude": 38.2768361, "longitude": -85.73573890000002 },
  "height": "2432", "width": "4320",
  "exif": { "apertureFNumber": 2.4, "cameraModel": "XT1056", "exposureTime": 0.01666,
            "focalLength": 4.499, "isoEquivalent": 640 },
  "sizeBytes": "3698139"
}"""


def test_parses_the_real_album_archive_payload() -> None:
    meta = parse_photo_metadata(ALBUM_ARCHIVE_PAYLOAD)
    assert meta.title == "2015-03-09.jpg"
    assert meta.photo_taken_at == datetime(2015, 3, 9, 12, 56, 32)
    assert meta.latitude == pytest.approx(38.2768361)
    assert meta.longitude == pytest.approx(-85.73573890000002)
    assert meta.size_bytes == 3698139
    assert meta.google_exif["cameraModel"] == "XT1056"
    assert meta.google_exif["isoEquivalent"] == 640
    # Absent in this schema; every field is optional.
    assert meta.description is None
    assert meta.people == ()
    assert meta.favorited is False


def test_creation_time_is_parsed_but_is_not_the_placement_date() -> None:
    """creationTime is upload time. It is recorded and never placed with.

    In the real export the two differ by four hours on the same file -- direct
    evidence for the rule.
    """
    meta = parse_photo_metadata(ALBUM_ARCHIVE_PAYLOAD)
    assert meta.creation_at == datetime(2015, 3, 9, 17, 3, 48)
    assert meta.creation_at != meta.photo_taken_at


def test_accepts_the_google_photos_timestamp_key() -> None:
    """Newer Google Photos exports use `timestamp`, not `timestampSeconds`."""
    raw = json.dumps(
        {
            "title": "IMG_1234.jpg",
            "description": "at the lake",
            "photoTakenTime": {"timestamp": "1425905792", "formatted": "..."},
            "people": [{"name": "Emma"}, {"name": "Sam"}],
            "favorited": True,
        }
    ).encode()
    meta = parse_photo_metadata(raw)
    assert meta.photo_taken_at == datetime(2015, 3, 9, 12, 56, 32)
    assert meta.description == "at the lake"
    assert meta.people == ("Emma", "Sam")
    assert meta.favorited is True


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json at all",
        b'{"photoTakenTime": {"timestampSeconds": "1425905792"',  # truncated
        b"[]",                       # top-level list
        b'"a string"',               # top-level scalar
        b"null",
        b'{"photoTakenTime": "not a dict"}',
        b'{"photoTakenTime": {"timestampSeconds": "not a number"}}',
        b'{"geoData": "not a dict"}',
        b'{"people": "not a list"}',
        b'{"sizeBytes": "not a number"}',
        b"\xff\xfe\x00garbage",      # not valid UTF-8
    ],
)
def test_malformed_input_never_raises(raw: bytes) -> None:
    meta = parse_photo_metadata(raw)
    assert meta.photo_taken_at is None
    assert meta.title is None


def test_implausible_timestamp_is_dropped() -> None:
    raw = json.dumps({"photoTakenTime": {"timestampSeconds": "-99999999999"}}).encode()
    assert parse_photo_metadata(raw).photo_taken_at is None


def test_null_island_geodata_is_treated_as_absent() -> None:
    """Google writes 0.0/0.0 for 'no location', which is not a location."""
    raw = json.dumps({"geoData": {"latitude": 0.0, "longitude": 0.0}}).encode()
    meta = parse_photo_metadata(raw)
    assert meta.latitude is None
    assert meta.longitude is None


def test_empty_strings_become_none() -> None:
    raw = json.dumps({"title": "", "description": "   "}).encode()
    meta = parse_photo_metadata(raw)
    assert meta.title is None
    assert meta.description is None


def test_empty_singleton_is_all_defaults() -> None:
    assert EMPTY.title is None
    assert EMPTY.photo_taken_at is None
    assert EMPTY.people == ()
    assert EMPTY.favorited is False
    assert EMPTY.google_exif == {}


def test_parse_album_metadata() -> None:
    raw = json.dumps({"title": "Hangout: Emma", "description": "chat images"}).encode()
    album = parse_album_metadata(raw)
    assert album.title == "Hangout: Emma"
    assert album.description == "chat images"


def test_parse_album_metadata_never_raises() -> None:
    assert parse_album_metadata(b"{{{").title is None


def test_deeply_nested_array_never_raises() -> None:
    """Adversarial input, not fuzzing -- this pins the never-raise contract.

    `json.loads` on deeply nested input raises `RecursionError`, which is a
    `RuntimeError` subclass, not a `ValueError`/`JSONDecodeError`/
    `UnicodeError`. A corrupted or partially-rewritten sidecar can plausibly
    produce this shape. If this test is ever deleted because it "looks like
    arbitrary fuzzing," the module's absolute never-raise contract is no
    longer verified against the exact input that once defeated it.
    """
    meta = parse_photo_metadata(b"[" * 200000)
    assert meta == EMPTY


def test_deeply_nested_object_never_raises_for_album_metadata() -> None:
    """Same contract as above, exercised through the object-decoding path.

    Adversarial input, not fuzzing -- do not delete this thinking it is a
    stray fuzz test. `parse_album_metadata` shares `_load` with
    `parse_photo_metadata`; this pins the object-nesting form of the same
    `RecursionError` hole (the array form is covered separately).
    """
    raw = b'{"a":' * 200000 + b"1" + b"}" * 200000
    album = parse_album_metadata(raw)
    assert album.title is None
    assert album.description is None


def test_geodata_with_only_latitude_is_treated_as_absent() -> None:
    """A partial geoData block (latitude without longitude) must not leak a
    half-populated coordinate -- both fields resolve to None together."""
    raw = json.dumps({"geoData": {"latitude": 38.2768361}}).encode()
    meta = parse_photo_metadata(raw)
    assert meta.latitude is None
    assert meta.longitude is None


def test_oversized_latitude_never_raises() -> None:
    """Adversarial input, not fuzzing -- pins the never-raise contract.

    `float()` raises `OverflowError` (not `ValueError`/`TypeError`) for a
    Python int too large for a C double, and a bare (unquoted) 400-digit JSON
    integer literal survives `json.loads` to reach exactly that call. If this
    test is ever deleted as arbitrary fuzzing, this exact escape from the
    module's absolute never-raise contract is no longer verified.
    """
    raw = b'{"geoData": {"latitude": ' + str(10**400).encode() + b', "longitude": 1}}'
    meta = parse_photo_metadata(raw)
    assert meta.latitude is None
    assert meta.longitude is None


def test_oversized_longitude_never_raises() -> None:
    """Same escape as above, exercised through `longitude` instead of
    `latitude` -- both operands of `_geo`'s `float()` calls are reachable by
    this class of oversized-int input, so both are pinned."""
    raw = b'{"geoData": {"latitude": 1, "longitude": ' + str(10**400).encode() + b"}}"
    meta = parse_photo_metadata(raw)
    assert meta.latitude is None
    assert meta.longitude is None


def test_non_dict_exif_block_yields_empty_google_exif() -> None:
    """A malformed `exif` value must degrade to `{}`, not raise and not
    store a non-dict value."""
    raw = json.dumps({"exif": "not a dict"}).encode()
    meta = parse_photo_metadata(raw)
    assert meta.google_exif == {}


def test_album_metadata_parses_access_and_date() -> None:
    raw = json.dumps({
        "title": "Hangout: Conrad Storz ● Herbie (Tony) Hughes",
        "access": "protected",
        "date": {"timestampSeconds": "1524674607", "formatted": "…"},
    }).encode()
    album = parse_album_metadata(raw)
    assert album.title.startswith("Hangout:")
    assert album.access == "protected"
    assert album.date == datetime(2018, 4, 25, 16, 43, 27)


def test_album_metadata_malformed_json_never_raises() -> None:
    """A corrupt Albums.json must degrade to 'no album metadata', never fail
    the batch it belongs to."""
    album = parse_album_metadata(b"{not json")
    assert album.title is None
    assert album.access is None
    assert album.date is None
