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

# The top EXIF rung, kept separate because the external-sidecar rung sits
# between it and the rest of the ladder.
_EXIF_PRIMARY_FIELD = "DateTimeOriginal"

# The remaining EXIF fields, all at the same lower tier.
_EXIF_OTHER_FIELDS: tuple[str, ...] = ("DateTimeDigitized", "DateTime")

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
    #
    # KNOWN FALSE POSITIVE, accepted deliberately: this matches ANY 8-digit run
    # bounded by - or _ that is calendar-valid and in range, so
    # "Order_20230615_001" reads as 2023-06-15. Accepted because in a photo
    # library a bounded _YYYYMMDD_ token is overwhelmingly a real date, and
    # anchoring this to camera prefixes would stop dating legitimate files like
    # "vacation_20190704_beach.jpg". It lands at DATE_FILENAME_PATTERN (10) --
    # the weakest non-zero rung, below every EXIF source -- and the source is
    # recorded, so a higher-ranked date can correct it later.
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


def date_from_row(row: Any) -> ResolvedDate:
    """Rebuild a :class:`ResolvedDate` from stored catalog columns.

    Shared by the facts pass and the enrichment pass -- both need to compare a
    freshly-resolved date against the one already on record, and this module
    is the dependency-light home for that logic (no AI, no catalog import).
    """
    raw = row["date_value"]
    value = None
    if raw:
        try:
            value = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            logger.warning("Unparseable stored date %r; treating as undated", raw)
    tier = row["date_tier"] or tiers.DATE_NONE
    return ResolvedDate(
        value=value,
        tier=tier,
        source=row["date_source"] or tiers.DATE_SOURCE_NAMES[tiers.DATE_NONE],
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


def resolve_date(
    source_path: Path,
    exif_data: dict[str, Any],
    *,
    external_date: datetime | None = None,
) -> ResolvedDate:
    """Resolve *source_path*'s capture date from EXIF, an external sidecar, then
    its filename.

    Rungs are tried highest-first and the first plausible hit wins.  File mtime
    is never consulted.

    *external_date* is a capture date asserted by a trustworthy source outside
    the file's own bytes and path -- in practice Google Takeout's
    ``photoTakenTime``.  It sits below EXIF ``DateTimeOriginal``, which is the
    camera's own record, and above ``DateTimeDigitized``/``DateTime``, which
    frequently record a scan or an edit rather than the capture.  An
    implausible value is ignored rather than asserted, exactly like an
    implausible EXIF value.

    Google's ``creationTime`` must NEVER be passed here: it records when a file
    was uploaded, which is the same category of claim as file mtime.
    """
    dt = _parse_exif_datetime(exif_data.get(_EXIF_PRIMARY_FIELD))
    if dt is not None:
        return ResolvedDate(
            value=dt,
            tier=tiers.DATE_EXIF_ORIGINAL,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_EXIF_ORIGINAL],
        )

    if external_date is not None and _plausible(external_date):
        return ResolvedDate(
            value=external_date,
            tier=tiers.DATE_EXTERNAL_SIDECAR,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_EXTERNAL_SIDECAR],
        )

    for field in _EXIF_OTHER_FIELDS:
        dt = _parse_exif_datetime(exif_data.get(field))
        if dt is not None:
            return ResolvedDate(
                value=dt,
                tier=tiers.DATE_EXIF_OTHER,
                source=tiers.DATE_SOURCE_NAMES[tiers.DATE_EXIF_OTHER],
            )

    dt = date_from_filename(source_path.stem)
    if dt is not None:
        return ResolvedDate(
            value=dt,
            tier=tiers.DATE_FILENAME_PATTERN,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_FILENAME_PATTERN],
        )

    logger.debug("No trustworthy date for %s -> %s", source_path.name, UNDATED_FOLDER)
    return _UNDATED
