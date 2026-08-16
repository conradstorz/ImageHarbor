"""Parse Google Takeout's per-media and per-album JSON sidecars.

Pure: handed ``bytes``, returns a dataclass, touches no filesystem. It never
raises -- malformed, truncated, empty, or absent input returns an empty
result, the same discipline ``exif_reader.read_exif`` uses. A sidecar is
supplementary evidence; a corrupt one must degrade a photo to "no external
date", never fail it.

Two export generations are in circulation and both are accepted: AlbumArchive
uses ``timestampSeconds``, newer Google Photos exports use ``timestamp``. Every
field is optional in both -- the AlbumArchive schema has no ``description`` and
no ``people`` at all.

All datetimes returned here are **naive UTC**. The rest of the date ladder is
naive (EXIF carries no timezone) and ``date_resolver.date_from_row`` rebuilds
naive values from the catalog, so returning aware datetimes would put two
incompatible kinds of value in one column.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Matches date_resolver's plausibility window: photography began in 1826, and
# anything past 2100 is a dead clock or a bad parse.
_MIN_YEAR = 1826
_MAX_YEAR = 2100

# Google writes 0.0/0.0 when it has no location. Null Island is not a location.
_NULL_ISLAND = (0.0, 0.0)


@dataclass(frozen=True)
class TakeoutMetadata:
    """What Google recorded about one media file.

    Only ``photo_taken_at`` and ``title`` are load-bearing (they feed the date
    and descriptor ladders). Everything else is recorded as provenance and can
    never move or rename a file.
    """

    title: str | None = None
    description: str | None = None
    photo_taken_at: datetime | None = None
    # RECORDED ONLY. creationTime is when the file was uploaded to Google
    # Photos, not when the photo was taken -- the same category of claim as
    # file mtime, which date_resolver.py deliberately refuses. In the real
    # export the two differ by four hours on the same file. It must never be
    # passed to resolve_date().
    creation_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    people: tuple[str, ...] = ()
    favorited: bool = False
    size_bytes: int | None = None


@dataclass(frozen=True)
class AlbumMetadata:
    """What Google recorded about one album (Albums.json / metadata.json)."""

    title: str | None = None
    description: str | None = None


EMPTY = TakeoutMetadata()
EMPTY_ALBUM = AlbumMetadata()


def _load(raw: bytes) -> dict[str, Any] | None:
    """Decode *raw* into a JSON object, or None if it is not one.

    The `except Exception` is deliberate and is scoped to the single decode
    call. This module's contract is absolute -- it NEVER raises -- and any
    explicit exception tuple is a standing guess about what `json.loads` can
    throw. That guess was already wrong once: a tuple of
    `(JSONDecodeError, UnicodeError, ValueError)` does not catch the
    `RecursionError` that deeply nested input raises (`b"[" * 200000`), which
    a corrupted or partially-rewritten sidecar can easily produce. A sidecar
    is supplementary evidence; a bad one must degrade a photo to "no Google
    metadata", never fail it, and never take down a 100k-member ingest.

    `exif_reader.read_exif` catches broadly for exactly this reason. Keep the
    two consistent.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except Exception as exc:
        logger.debug("Unparseable Takeout sidecar (%s); treating as absent", exc)
        return None
    return data if isinstance(data, dict) else None


def _text(value: Any) -> str | None:
    """A non-blank string, or None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _timestamp(block: Any) -> datetime | None:
    """Epoch seconds out of a Google timestamp block, as naive UTC.

    Accepts ``timestampSeconds`` (AlbumArchive) and ``timestamp`` (Google
    Photos); both are strings holding epoch seconds UTC.
    """
    if not isinstance(block, dict):
        return None
    for key in ("timestampSeconds", "timestamp"):
        raw = block.get(key)
        if raw is None:
            continue
        try:
            seconds = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        try:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            continue
        if _MIN_YEAR <= dt.year <= _MAX_YEAR:
            return dt
    return None


def _geo(block: Any) -> tuple[float | None, float | None]:
    if not isinstance(block, dict):
        return (None, None)
    try:
        lat = float(block["latitude"])
        lon = float(block["longitude"])
    except (KeyError, TypeError, ValueError):
        return (None, None)
    if (lat, lon) == _NULL_ISLAND:
        return (None, None)
    return (lat, lon)


def _people(block: Any) -> tuple[str, ...]:
    if not isinstance(block, list):
        return ()
    names = []
    for entry in block:
        if isinstance(entry, dict):
            name = _text(entry.get("name"))
        else:
            name = _text(entry)
        if name:
            names.append(name)
    return tuple(names)


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_photo_metadata(raw: bytes) -> TakeoutMetadata:
    """Parse one per-media JSON sidecar. Never raises."""
    data = _load(raw)
    if data is None:
        return EMPTY

    latitude, longitude = _geo(data.get("geoData"))
    # Some exports carry an emptied `geoData` alongside a populated
    # `geoDataExif`. Fall back to it, still as provenance only.
    if latitude is None:
        latitude, longitude = _geo(data.get("geoDataExif"))

    return TakeoutMetadata(
        title=_text(data.get("title")),
        description=_text(data.get("description")),
        photo_taken_at=_timestamp(data.get("photoTakenTime")),
        creation_at=_timestamp(data.get("creationTime")),
        latitude=latitude,
        longitude=longitude,
        people=_people(data.get("people")),
        favorited=data.get("favorited") is True,
        size_bytes=_int(data.get("sizeBytes")),
    )


def parse_album_metadata(raw: bytes) -> AlbumMetadata:
    """Parse an Albums.json / metadata.json album descriptor. Never raises."""
    data = _load(raw)
    if data is None:
        return EMPTY_ALBUM
    return AlbumMetadata(
        title=_text(data.get("title")),
        description=_text(data.get("description")),
    )
