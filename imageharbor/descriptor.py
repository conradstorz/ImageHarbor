"""Descriptor resolution from the original filename.

An original filename is itself a fact, and often the best one available: a
person who typed "Emma's graduation" knew something no model will recover from
the pixels.  Camera-generated names carry no such information, so they are
discarded and the descriptor waits for the AI enrichment pass.

The pattern list below is the one empirical claim in this module.  Keep it
adjacent to its fixture table in tests/test_descriptor.py so additions arrive
with tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import tiers
from .filename import normalize_descriptor

# Matched case-insensitively against the full original stem. A match means
# "no human information here".
CAMERA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^_?img[-_]?\d+$", re.I),                      # IMG_1234, _IMG0042
    re.compile(r"^img[-_]\d{8}[-_]wa\d+$", re.I),              # IMG-20190704-WA0001
    re.compile(r"^img[-_]\d{8}[-_]\d{6}$", re.I),              # IMG_20190704_123456
    re.compile(r"^_?dsc[nf]?[-_]?\d+$", re.I),                 # DSC0042, DSCN, DSCF, _DSC
    re.compile(r"^p?xl[-_]\d{8}[-_]\d+$", re.I),               # PXL_20190704_123456789
    re.compile(r"^mvimg[-_]\d{8}[-_]\d{6}$", re.I),            # MVIMG_20190704_123456
    re.compile(r"^p\d{7}$", re.I),                             # P1000042 (Panasonic)
    re.compile(r"^pict\d+$", re.I),                            # PICT0042
    re.compile(r"^\d{3}[-_]\d{4}$", re.I),                     # 100_0042
    re.compile(r"^cimg\d+$", re.I),                            # CIMG0042 (Casio)
    # Samsung's format is exactly 4 digits. Do NOT relax this to \d+ -- "sam" is
    # also a person's name, and "Sam_1.jpg" is an ordinary way to label photos
    # of someone. A tier-0 verdict discards that name permanently.
    re.compile(r"^sam[-_]\d{4}$", re.I),                       # SAM_0042 (Samsung)
    re.compile(r"^gopr\d+$", re.I),                            # GOPR0042
    re.compile(r"^dji[-_]\d+$", re.I),                         # DJI_0042
    # These three require a DIGIT after the auto-generated prefix, so the
    # timestamp forms match but an appended human suffix survives:
    # "Screenshot - grandpas last text message" stays tier 30. An open .*$ here
    # would swallow it. One pattern covers "Screenshot" and "Screen Shot" both,
    # since [-_ ]? already matches zero separator characters.
    re.compile(r"^screen[-_ ]?shot[-_ ]?\d.*$", re.I),         # Screenshot_2019-...
    re.compile(r"^whatsapp[ -](image|video)[ -]?\d.*$", re.I), # WhatsApp Image 2019-...
    re.compile(r"^signal[-_]\d{4}-\d{2}-\d{2}.*$", re.I),      # Signal-2019-07-04-...
    re.compile(r"^fb[-_]img[-_]\d+$", re.I),                   # FB_IMG_1562243591
    re.compile(r"^received[-_]\d+$", re.I),                    # received_101234567890
    re.compile(r"^\d{8}[-_]\d{6}$", re.I),                     # 20190704_123456
    re.compile(r"^\d{4}-\d{2}-\d{2}[ _]\d{2}\.\d{2}\.\d{2}$"), # 2019-07-04 12.33.11
    re.compile(r"^\d{9,13}$"),                                 # bare epoch seconds/ms
    # Hangouts / AlbumArchive row ids, present at volume in Google Takeout
    # exports: 865948477697870747_account_id=1.jpg
    #
    # The separators are deliberately loose: the zip member name reads
    # `..._account_id=1` because the filesystem cannot hold the `?` that
    # Google's own `title` field preserves (`...?account_id=1`). Anchoring on
    # one spelling would match only one of the two names the resolver sees.
    re.compile(r"^\d{10,}[\W_]?account[\W_]?id[\W_]?\d+$", re.I),
    # A BARE date, with or without Google's (N) copy suffix. A date is not a
    # description -- the date ladder already captured it, and keeping it here
    # would state the same fact twice in one filename. A date followed by
    # human words ("2015-03-09 emma birthday") does NOT match and survives.
    re.compile(r"^\d{4}-\d{2}-\d{2}(\(\d+\))?$"),
)


@dataclass(frozen=True)
class ResolvedDescriptor:
    """A descriptor together with the provenance that justifies its tier."""

    value: str
    tier: int
    source: str


_NONE = ResolvedDescriptor(value="", tier=tiers.DESC_NONE, source=tiers.DESC_SOURCE_NAMES[tiers.DESC_NONE])


def is_camera_generated(stem: str) -> bool:
    """Return True if *stem* looks machine-generated rather than human-authored."""
    candidate = stem.strip()
    return any(pattern.match(candidate) for pattern in CAMERA_PATTERNS)


def resolve_descriptor(
    source_path: Path,
    *,
    original_name: str | None = None,
    date_str: str | None = None,
) -> ResolvedDescriptor:
    """Derive a descriptor from *source_path*'s original filename.

    Returns tier ``DESC_HUMAN_FILENAME`` when the stem carries human intent,
    and ``DESC_NONE`` when it does not -- leaving the slot open for the AI
    enrichment pass to fill at the lower ``DESC_AI_SUBJECT`` tier.

    Parameters
    ----------
    original_name:
        A filename known to be closer to the original than *source_path*'s own
        -- Google Takeout's ``title``, which is the pre-truncation name of a
        member whose stem the export truncated.

        It supplies a BETTER SPELLING of the name; it is not a vote that the
        name is human-authored. A camera verdict from EITHER name therefore
        wins. This is load-bearing: Google's ``title`` keeps characters the zip
        member name had to sanitize for the filesystem, so the two can differ
        in exactly the characters a pattern anchors on. In the calibrating
        export every Hangouts row id reads ``...?account_id=1`` in the title but
        ``..._account_id=1`` in the member name -- and letting the title win
        would lock 42 of 52 files to a row id at tier 30, where no later pass
        could ever rename them.
    date_str:
        The ``YYYY-MM-DD`` the date ladder actually resolved for this file, when
        the caller already knows it. A descriptor that merely restates the date
        carries no information beyond what the folder and the filename's date
        prefix already say, so it is discarded as ``DESC_NONE``.
    """
    path_stem = source_path.stem
    title_stem = Path(original_name.strip()).stem if original_name else ""

    if not path_stem and not title_stem:
        return _NONE
    if is_camera_generated(path_stem) or (title_stem and is_camera_generated(title_stem)):
        return _NONE

    stem = title_stem or path_stem

    normalized = normalize_descriptor(stem)
    # normalize_descriptor falls back to "photo" for input with no usable
    # characters; that is not information, so treat it as absent.
    if not normalized or normalized == "photo":
        return _NONE

    if date_str and normalized == date_str:
        return _NONE

    return ResolvedDescriptor(
        value=normalized,
        tier=tiers.DESC_HUMAN_FILENAME,
        source=tiers.DESC_SOURCE_NAMES[tiers.DESC_HUMAN_FILENAME],
    )
