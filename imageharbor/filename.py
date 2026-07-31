"""Deterministic PCS filename generation and parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from .hashing import SHA256_B64URL_LEN

_DESCRIPTOR_RE = re.compile(r"[^a-z0-9]+")
_MAX_DESCRIPTOR_LEN = 30
_MAX_FILENAME_LEN = 100


# ---------------------------------------------------------------------------
# Descriptor normalisation
# ---------------------------------------------------------------------------


def normalize_descriptor(text: str) -> str:
    """Normalise *text* into a PCS-compliant descriptor.

    Rules (per spec):
    * lowercase
    * ASCII letters/digits only
    * words joined with hyphens
    * 1–3 words
    * max 30 characters total
    * must not be empty; falls back to ``photo``
    """
    lowered = text.lower()
    # Replace any run of non-alphanumeric characters with a single space
    cleaned = _DESCRIPTOR_RE.sub(" ", lowered)
    words = [w for w in cleaned.split() if w][:3]
    descriptor = "-".join(words)
    descriptor = descriptor[:_MAX_DESCRIPTOR_LEN].rstrip("-")
    return descriptor or "photo"


# ---------------------------------------------------------------------------
# Filename generation
# ---------------------------------------------------------------------------


def generate_filename(
    pcs_code: int,
    descriptor: str,
    sha256_b64url: str,
    extension: str,
) -> str:
    """Return a deterministic PCS filename.

    Format: ``<pcs>-<descriptor>_<sha256_b64url>.<ext>``

    The total length is guaranteed ≤ 100 characters.  If the descriptor causes
    the filename to exceed the limit it is silently truncated.
    """
    ext = extension.lower().lstrip(".")
    desc = normalize_descriptor(descriptor)
    name = f"{pcs_code}-{desc}_{sha256_b64url}.{ext}"

    if len(name) > _MAX_FILENAME_LEN:
        # Calculate how many characters the descriptor may use
        overhead = len(f"{pcs_code}-_{sha256_b64url}.{ext}")
        max_desc = _MAX_FILENAME_LEN - overhead
        desc = desc[: max(1, max_desc)].rstrip("-")
        name = f"{pcs_code}-{desc}_{sha256_b64url}.{ext}"

    return name


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


class ParsedFilename(TypedDict):
    pcs_code: int
    descriptor: str
    sha256_b64url: str
    extension: str


def parse_filename(filename: str) -> ParsedFilename | None:
    """Parse a PCS filename and return its components, or None on failure.

    Accepts both bare filenames and full paths.  The digest is located by
    counting back exactly :data:`~imageharbor.hashing.SHA256_B64URL_LEN`
    characters from the end of the stem, since base64url may contain ``_``.
    """
    p = Path(filename)
    stem = p.stem
    ext = p.suffix.lstrip(".").lower()

    # Locate the separator '_' that precedes the 43-char digest
    if len(stem) <= SHA256_B64URL_LEN:
        return None
    sep_idx = len(stem) - SHA256_B64URL_LEN - 1
    if stem[sep_idx] != "_":
        return None

    sha256_b64url = stem[sep_idx + 1 :]
    prefix = stem[:sep_idx]

    dash_idx = prefix.find("-")
    if dash_idx < 0:
        return None

    pcs_str = prefix[:dash_idx]
    descriptor = prefix[dash_idx + 1 :]

    try:
        pcs_code = int(pcs_str)
    except ValueError:
        return None

    return ParsedFilename(
        pcs_code=pcs_code,
        descriptor=descriptor,
        sha256_b64url=sha256_b64url,
        extension=ext,
    )
