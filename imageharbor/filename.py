"""Deterministic PCS filename generation and parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from .hashing import SHA256_B64URL_LEN, extract_digest_from_stem

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
    pcs_code: str,
    descriptor: str,
    sha256_b64url: str,
    extension: str,
) -> str:
    """Return a deterministic PCS filename.

    Format: ``<pcs>-<descriptor>_<sha256_b64url>.<ext>``

    The total length is guaranteed ≤ 100 characters for ALL inputs.  If the
    descriptor causes the filename to exceed the limit it is silently truncated;
    if a pathologically long extension still overflows after the descriptor has
    been shrunk to a single character, the extension itself is truncated.
    """
    # Derive a single, safe extension component: take the part after the last
    # dot, lowercase it, and keep only [a-z0-9] characters. This sanitises
    # invalid Windows characters, path separators, and collapses multi-dot
    # extensions (e.g. "tar.gz" -> "gz").
    ext = re.sub(r"[^a-z0-9]", "", extension.lower().rsplit(".", 1)[-1])
    suffix = f".{ext}" if ext else ""
    desc = normalize_descriptor(descriptor)
    name = f"{pcs_code}-{desc}_{sha256_b64url}{suffix}"

    if len(name) > _MAX_FILENAME_LEN:
        # Calculate how many characters the descriptor may use
        overhead = len(f"{pcs_code}-_{sha256_b64url}{suffix}")
        max_desc = _MAX_FILENAME_LEN - overhead
        desc = desc[: max(1, max_desc)].rstrip("-") or desc[:1]
        name = f"{pcs_code}-{desc}_{sha256_b64url}{suffix}"

    if len(name) > _MAX_FILENAME_LEN and ext:
        # Descriptor is already at its minimum (>=1 char) but the extension is
        # still pushing us over the limit: truncate the extension itself by the
        # overflow amount and rebuild.
        overflow = len(name) - _MAX_FILENAME_LEN
        ext = ext[: max(0, len(ext) - overflow)]
        suffix = f".{ext}" if ext else ""
        name = f"{pcs_code}-{desc}_{sha256_b64url}{suffix}"

    return name


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


class ParsedFilename(TypedDict):
    pcs_code: str
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

    # Reuse extract_digest_from_stem so parse and extract can never diverge in
    # what they accept (empty descriptor, non-ASCII/'+' pcs, short stems, etc.).
    digest = extract_digest_from_stem(stem)
    if digest is None:
        return None

    # Recover the prefix "<pcs>-<descriptor>" (everything before "_<digest>").
    prefix = stem[: len(stem) - SHA256_B64URL_LEN - 1]
    # Split on the FIRST "-"; both parts are guaranteed valid and non-empty
    # because extract_digest_from_stem already validated them.
    pcs_str, descriptor = prefix.split("-", 1)
    pcs_code = pcs_str  # keep as string; codes may contain '~'

    return ParsedFilename(
        pcs_code=pcs_code,
        descriptor=descriptor,
        sha256_b64url=digest,
        extension=ext,
    )
