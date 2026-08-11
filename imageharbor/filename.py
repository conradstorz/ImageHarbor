"""Deterministic organized filename generation and parsing."""

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

# A date prefix is exactly YYYY-MM-DD. Anchored so a descriptor that merely
# starts with digits (e.g. "2019-summer") is not misread as a date.
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:-(.+))?$")


def build_filename(
    date_str: str | None,
    descriptor: str | None,
    sha256_b64url: str,
    extension: str,
) -> str:
    """Return an organized filename.

    Format: ``[<date>][-<descriptor>]_<digest>.<ext>``.  Both prefix
    components are optional; with neither, the stem is the bare digest.

    The total length is guaranteed <= 100 characters.  The descriptor is
    truncated first (the date is never sacrificed, since it must agree with the
    folder the file lives in); a pathologically long extension is truncated
    last.
    """
    ext = re.sub(r"[^a-z0-9]", "", extension.lower().rsplit(".", 1)[-1])
    suffix = f".{ext}" if ext else ""
    desc = normalize_descriptor(descriptor) if descriptor else ""

    def assemble(d: str) -> str:
        prefix = "-".join(part for part in (date_str or "", d) if part)
        return f"{prefix}_{sha256_b64url}{suffix}" if prefix else f"{sha256_b64url}{suffix}"

    name = assemble(desc)
    if len(name) > _MAX_FILENAME_LEN and desc:
        overflow = len(name) - _MAX_FILENAME_LEN
        desc = desc[: max(0, len(desc) - overflow)].rstrip("-")
        name = assemble(desc)

    if len(name) > _MAX_FILENAME_LEN and ext:
        overflow = len(name) - _MAX_FILENAME_LEN
        ext = ext[: max(0, len(ext) - overflow)]
        suffix = f".{ext}" if ext else ""
        name = assemble(desc)

    return name


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


class ParsedFilename(TypedDict):
    date: str | None
    descriptor: str
    sha256_b64url: str
    extension: str


def parse_filename(filename: str) -> ParsedFilename | None:
    """Parse an organized filename, or return None if it is not one.

    Accepts bare filenames and full paths.  The digest is located by counting
    back from the end of the stem (see
    :func:`~imageharbor.hashing.extract_digest_from_stem`); everything before
    the separator is split into an optional ``YYYY-MM-DD`` date and an optional
    descriptor.
    """
    p = Path(filename)
    stem = p.stem
    ext = p.suffix.lstrip(".").lower()

    digest = extract_digest_from_stem(stem)
    if digest is None:
        return None

    # Recover the prefix: everything before "_<digest>", or "" for a bare digest.
    prefix = "" if len(stem) == SHA256_B64URL_LEN else stem[: len(stem) - SHA256_B64URL_LEN - 1]

    date: str | None = None
    descriptor = prefix
    match = _DATE_PREFIX_RE.match(prefix)
    if match:
        date = match.group(1)
        descriptor = match.group(2) or ""

    return ParsedFilename(
        date=date,
        descriptor=descriptor,
        sha256_b64url=digest,
        extension=ext,
    )
