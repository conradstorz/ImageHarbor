"""Capture-date resolution.

The date is the load-bearing fact of the organized library: it decides the
folder a file lands in, and placement is meant to be permanent.  So every rung
of this ladder is evidence about when the photo was *taken*, and file mtime --
which records when a file was last copied -- is deliberately absent.

A file with no trustworthy date goes to ``Undated/`` and waits.  Asserting a
year we cannot support would be exactly the quiet corruption the project's
SHA-256 discipline exists to prevent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import tiers

logger = logging.getLogger(__name__)

UNDATED_FOLDER = "Undated"

# EXIF stores timestamps as "YYYY:MM:DD HH:MM:SS".
_EXIF_FORMAT = "%Y:%m:%d %H:%M:%S"

# Photography began in 1826; anything earlier is a dead clock or a bad parse.
_MIN_YEAR = 1826
_MAX_YEAR = 2100

# EXIF fields in ladder order: (field name, tier).
_EXIF_FIELDS: tuple[tuple[str, int], ...] = (
    ("DateTimeOriginal", tiers.DATE_EXIF_ORIGINAL),
    ("DateTimeDigitized", tiers.DATE_EXIF_OTHER),
    ("DateTime", tiers.DATE_EXIF_OTHER),
)

# Filename date patterns, most specific first. Each yields named groups.
# A bare epoch is deliberately NOT decoded: it is indistinguishable from an
# ordinary counter, so treating it as a timestamp would invent evidence.
_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 20190704_123456 / IMG_20190704_123456 / PXL_20190704_123456789
    re.compile(
        r"(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})[-_](?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"
    ),
    # 2019-07-04-12-33-11 / 2019-07-04_12-33-11
    re.compile(
        r"(?P<Y>\d{4})-(?P<M>\d{2})-(?P<D>\d{2})[-_](?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})"
    ),
    # 2019-07-04 12.33.11
    re.compile(
        r"(?P<Y>\d{4})-(?P<M>\d{2})-(?P<D>\d{2})[ _](?P<h>\d{2})\.(?P<m>\d{2})\.(?P<s>\d{2})"
    ),
    # Date only: 2019-07-04
    re.compile(r"(?P<Y>\d{4})-(?P<M>\d{2})-(?P<D>\d{2})"),
    # Date only, dotted or space-separated: 2019.07.04 / 2019 07 04.
    # Without this rung such a file is Undated, and its descriptor normalizes
    # to a date-shaped token that build_filename then has to disambiguate.
    # Reading the date properly is the better outcome.
    re.compile(r"(?P<Y>\d{4})[.\s](?P<M>\d{2})[.\s](?P<D>\d{2})"),
    # Date only, compact and delimited: IMG-20190704-WA0001
    re.compile(r"[-_](?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})[-_]"),
)


@dataclass(frozen=True)
class ResolvedDate:
    """A capture date together with the provenance that justifies its tier."""

    value: datetime | None
    tier: int
    source: str

    @property
    def date_str(self) -> str | None:
        """``YYYY-MM-DD`` for the filename, or None when undated."""
        return self.value.strftime("%Y-%m-%d") if self.value else None

    @property
    def folder(self) -> str:
        """Destination folder relative to the organized root."""
        if self.value is None:
            return UNDATED_FOLDER
        return f"{self.value.year:04d}/{self.value.year:04d}-{self.value.month:02d}"


_UNDATED = ResolvedDate(
    value=None, tier=tiers.DATE_NONE, source=tiers.DATE_SOURCE_NAMES[tiers.DATE_NONE]
)


def _plausible(dt: datetime) -> bool:
    return _MIN_YEAR <= dt.year <= _MAX_YEAR


def _parse_exif_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.strptime(raw.strip(), _EXIF_FORMAT)
    except ValueError:
        return None
    return dt if _plausible(dt) else None


def date_from_filename(stem: str) -> datetime | None:
    """Extract a capture date from a filename stem, or None."""
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        parts = match.groupdict()
        try:
            dt = datetime(
                int(parts["Y"]),
                int(parts["M"]),
                int(parts["D"]),
                int(parts.get("h") or 0),
                int(parts.get("m") or 0),
                int(parts.get("s") or 0),
            )
        except ValueError:
            continue  # e.g. month 13, or 07-32
        if _plausible(dt):
            return dt
    return None


def resolve_date(source_path: Path, exif_data: dict[str, Any]) -> ResolvedDate:
    """Resolve *source_path*'s capture date from EXIF, then from its filename.

    Rungs are tried highest-first and the first plausible hit wins.  File mtime
    is never consulted.
    """
    for field, tier in _EXIF_FIELDS:
        dt = _parse_exif_datetime(exif_data.get(field))
        if dt is not None:
            return ResolvedDate(value=dt, tier=tier, source=tiers.DATE_SOURCE_NAMES[tier])

    dt = date_from_filename(source_path.stem)
    if dt is not None:
        return ResolvedDate(
            value=dt,
            tier=tiers.DATE_FILENAME_PATTERN,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_FILENAME_PATTERN],
        )

    logger.debug("No trustworthy date for %s -> %s", source_path.name, UNDATED_FOLDER)
    return _UNDATED
