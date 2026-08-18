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
from dataclasses import dataclass, field
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
    # Google's own EXIF summary (apertureFNumber, cameraModel, exposureTime,
    # focalLength, isoEquivalent). Passed through verbatim and never parsed
    # for meaning -- it is recorded provenance, exactly like geo and people,
    # and cannot move or rename a file. Usually redundant with the real EXIF
    # `exif_reader` reads from the image bytes; occasionally it is the only
    # camera data left, when Google's export pipeline stripped the original.
    google_exif: dict[str, Any] = field(default_factory=dict)

    # Explicitly unhashable. frozen=True + eq=True would otherwise autogenerate
    # a __hash__ that raises TypeError at call time, because `google_exif` is a
    # dict -- a landmine for the first caller to put one of these in a set or
    # use it as a dict key. Equality still works and is what the tests use.
    __hash__ = None


@dataclass(frozen=True)
class AlbumMetadata:
    """What Google recorded about one album (Albums.json / metadata.json)."""

    title: str | None = None
    description: str | None = None
    access: str | None = None
    date: datetime | None = None


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
    # OverflowError is in the tuple because `float()` raises it for a Python
    # int too large for a C double, and a bare (unquoted) 400-digit JSON
    # integer literal survives json.loads to reach exactly that call. Unlike
    # `_load`'s unknowable surface, this one IS bounded and enumerable: the
    # input is a json.loads product, so `float()` can see only dict/list/None
    # (TypeError), a non-numeric str (ValueError), or an oversized int
    # (OverflowError). A complete narrow tuple is therefore correct here --
    # but it must actually be complete.
    if not isinstance(block, dict):
        return (None, None)
    try:
        lat = float(block["latitude"])
        lon = float(block["longitude"])
    except (KeyError, TypeError, ValueError, OverflowError):
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

    google_exif = data.get("exif")
    if not isinstance(google_exif, dict):
        google_exif = {}

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
        google_exif=google_exif,
    )


def parse_album_metadata(raw: bytes) -> AlbumMetadata:
    """Parse an Albums.json / metadata.json album descriptor. Never raises."""
    data = _load(raw)
    if data is None:
        return EMPTY_ALBUM
    return AlbumMetadata(
        title=_text(data.get("title")),
        description=_text(data.get("description")),
        access=_text(data.get("access")),
        date=_timestamp(data.get("date")),
    )
