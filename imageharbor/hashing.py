"""SHA-256 hashing and Base64url encoding utilities."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

# SHA-256 produces 32 bytes → 43 unpadded Base64url characters (always).
SHA256_B64URL_LEN = 43

# The digest is unpadded Base64url: exactly 43 characters from the RFC 4648 §5
# alphabet. Validating the character class is what lets the *prefix* be
# unconstrained -- the filename grammar allows a bare digest, a date, a
# descriptor, both, or (for legacy files) a PCS code.
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def compute_sha256_bytes(path: Path) -> bytes:
    """Return the raw 32-byte SHA-256 digest of *path*."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.digest()


def encode_base64url(digest: bytes) -> str:
    """Encode *digest* as unpadded Base64url (RFC 4648 §5).

    A SHA-256 digest (32 bytes) always encodes to exactly 43 characters.
    """
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def decode_base64url(encoded: str) -> bytes:
    """Decode an unpadded Base64url string back to bytes."""
    # Re-add padding so standard library decodes correctly
    padding = 4 - len(encoded) % 4
    if padding != 4:
        encoded += "=" * padding
    return base64.urlsafe_b64decode(encoded)


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------


def compute_sha256_b64url(path: Path) -> str:
    """Compute SHA-256 of *path* and return as 43-character Base64url string."""
    return encode_base64url(compute_sha256_bytes(path))


def compute_sha256_b64url_bytes(data: bytes) -> str:
    """Compute SHA-256 of *data* (already in memory) as 43-char Base64url.

    Mirrors :func:`compute_sha256_b64url`, which takes a path. This is for
    bytes that did not come from a plain file on disk -- e.g. one member of
    a zip archive -- where writing them to a temp file just to hash them
    would be wasted I/O.
    """
    return encode_base64url(hashlib.sha256(data).digest())


def verify_file(path: Path, expected_b64url: str) -> bool:
    """Return True if the SHA-256 of *path* matches *expected_b64url*."""
    return compute_sha256_b64url(path) == expected_b64url


# ---------------------------------------------------------------------------
# Filename-embedded digest extraction
# ---------------------------------------------------------------------------


def extract_digest_from_stem(stem: str) -> str | None:
    """Extract the Base64url digest from an organized filename stem.

    The grammar is ``[<date>][-<descriptor>]_<digest>``, where both prefix
    components are optional -- so a stem may be nothing but the digest itself.
    Because base64url may contain ``_``, the separator is located by counting
    back exactly :data:`SHA256_B64URL_LEN` characters from the end of the stem
    rather than splitting on the last underscore.

    Legacy ``<pcs>-<descriptor>_<digest>`` stems parse unchanged, so files
    organized by the previous scheme remain verifiable.

    Returns the 43-character digest, or None if the stem does not match.
    """
    # A stem that is nothing but the digest: Undated/<digest>.jpg
    if len(stem) == SHA256_B64URL_LEN:
        return stem if _B64URL_RE.match(stem) else None

    # Otherwise we need at least a one-character prefix plus the separator.
    if len(stem) < SHA256_B64URL_LEN + 2:
        return None
    sep_idx = len(stem) - SHA256_B64URL_LEN - 1
    if stem[sep_idx] != "_":
        return None
    if not stem[:sep_idx]:
        return None
    digest = stem[sep_idx + 1 :]
    return digest if _B64URL_RE.match(digest) else None


def verify_pcs_file(path: Path) -> bool:
    """Verify an organized file by extracting its embedded digest.

    Returns True if the file's SHA-256 matches the digest encoded in its name.
    Returns False if verification fails or the filename is not in organized format.
    """
    digest = extract_digest_from_stem(path.stem)
    if not digest:
        return False
    return verify_file(path, digest)
