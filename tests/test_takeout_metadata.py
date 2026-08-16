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


def test_parse_album_metadata() -> None:
    raw = json.dumps({"title": "Hangout: Emma", "description": "chat images"}).encode()
    album = parse_album_metadata(raw)
    assert album.title == "Hangout: Emma"
    assert album.description == "chat images"


def test_parse_album_metadata_never_raises() -> None:
    assert parse_album_metadata(b"{{{").title is None
