"""SHA-256 hashing and Base64url encoding utilities."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

# SHA-256 produces 32 bytes → 43 unpadded Base64url characters (always).
SHA256_B64URL_LEN = 43

# PCS codes are plain integers ("330") or, on taxonomy overflow, a base code
# with a "~N" suffix ("540~1"). '~' is filesystem-safe and NOT in the
# base64url alphabet, so the 43-char digest counting-back logic is unaffected.
_PCS_CODE_RE = re.compile(r"^\d+(~\d+)*$")


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


def verify_file(path: Path, expected_b64url: str) -> bool:
    """Return True if the SHA-256 of *path* matches *expected_b64url*."""
    return compute_sha256_b64url(path) == expected_b64url


# ---------------------------------------------------------------------------
# Filename-embedded digest extraction
# ---------------------------------------------------------------------------


def extract_digest_from_stem(stem: str) -> str | None:
    """Extract the Base64url digest from a PCS filename stem.

    The stem format is ``<pcs>-<descriptor>_<digest>``.  Because base64url
    itself may contain ``_`` characters, we locate the separator by counting
    back exactly :data:`SHA256_B64URL_LEN` characters from the end rather than
    splitting on the last underscore.

    Returns the digest portion (43 chars) or None if the stem does not match.
    """
    # Minimum viable stem: "<pcs>-<d>_" (at least 4 chars) + 43-char digest.
    if len(stem) < SHA256_B64URL_LEN + 4:
        return None
    sep_idx = len(stem) - SHA256_B64URL_LEN - 1
    if stem[sep_idx] != "_":
        return None
    prefix = stem[:sep_idx]
    dash_idx = prefix.find("-")
    if dash_idx < 0:
        return None
    # The part before '-' must be a valid PCS code (ASCII digits, optionally
    # with '~N' overflow suffixes); after '-' must be a non-empty descriptor.
    pcs_part = prefix[:dash_idx]
    if not (pcs_part.isascii() and _PCS_CODE_RE.match(pcs_part)) or not prefix[dash_idx + 1:]:
        return None
    return stem[sep_idx + 1 :]


def verify_pcs_file(path: Path) -> bool:
    """Verify a PCS-named file by extracting its embedded digest.

    Returns True if the file's SHA-256 matches the digest encoded in its name.
    Returns False if verification fails or the filename is not in PCS format.
    """
    digest = extract_digest_from_stem(path.stem)
    if not digest:
        return False
    return verify_file(path, digest)
