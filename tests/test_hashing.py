"""Tests for imageharbor.hashing."""

import hashlib
from pathlib import Path

import pytest

from imageharbor.hashing import (
    compute_sha256_bytes,
    compute_sha256_b64url,
    decode_base64url,
    encode_base64url,
    extract_digest_from_stem,
    verify_file,
    verify_pcs_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_image(tmp_path: Path) -> Path:
    """Create a tiny but valid JPEG-like file for testing."""
    p = tmp_path / "test.jpg"
    # Minimal JPEG: SOI + EOI markers
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 12 + b"\xff\xd9")
    return p


# ---------------------------------------------------------------------------
# encode_base64url / decode_base64url
# ---------------------------------------------------------------------------


def test_encode_base64url_no_padding() -> None:
    digest = b"\x00" * 32  # 32 zero bytes
    encoded = encode_base64url(digest)
    assert "=" not in encoded
    assert len(encoded) == 43


def test_encode_decode_roundtrip() -> None:
    digest = hashlib.sha256(b"hello world").digest()
    encoded = encode_base64url(digest)
    decoded = decode_base64url(encoded)
    assert decoded == digest


def test_sha256_digest_always_43_chars() -> None:
    for data in (b"", b"a", b"x" * 1000):
        digest = hashlib.sha256(data).digest()
        encoded = encode_base64url(digest)
        assert len(encoded) == 43, f"Expected 43 chars, got {len(encoded)}"


# ---------------------------------------------------------------------------
# compute_sha256_bytes / compute_sha256_b64url
# ---------------------------------------------------------------------------


def test_compute_sha256_bytes(tmp_image: Path) -> None:
    expected = hashlib.sha256(tmp_image.read_bytes()).digest()
    assert compute_sha256_bytes(tmp_image) == expected


def test_compute_sha256_b64url(tmp_image: Path) -> None:
    expected = encode_base64url(hashlib.sha256(tmp_image.read_bytes()).digest())
    assert compute_sha256_b64url(tmp_image) == expected


def test_compute_sha256_b64url_length(tmp_image: Path) -> None:
    result = compute_sha256_b64url(tmp_image)
    assert len(result) == 43


# ---------------------------------------------------------------------------
# verify_file
# ---------------------------------------------------------------------------


def test_verify_file_correct(tmp_image: Path) -> None:
    digest = compute_sha256_b64url(tmp_image)
    assert verify_file(tmp_image, digest) is True


def test_verify_file_wrong_digest(tmp_image: Path) -> None:
    assert verify_file(tmp_image, "A" * 43) is False


def test_verify_file_after_modification(tmp_path: Path) -> None:
    p = tmp_path / "img.jpg"
    p.write_bytes(b"original content")
    digest = compute_sha256_b64url(p)
    # Modify the file
    p.write_bytes(b"modified content")
    assert verify_file(p, digest) is False


# ---------------------------------------------------------------------------
# extract_digest_from_stem
# ---------------------------------------------------------------------------


def test_extract_digest_from_stem_normal() -> None:
    digest = "q" * 43  # 43-char placeholder representing a valid b64url digest
    stem = f"330-indiana-dunes_{digest}"
    result = extract_digest_from_stem(stem)
    assert result == digest


def test_extract_digest_from_stem_no_underscore() -> None:
    assert extract_digest_from_stem("no-underscore-here") is None


def test_extract_digest_from_stem_digest_contains_underscore() -> None:
    # base64url can contain '_'; the separator is at position -(43+1) from end
    digest = "A" * 21 + "_" + "B" * 21  # 43 chars with embedded underscore
    stem = f"330-beach_{digest}"
    assert extract_digest_from_stem(stem) == digest


def test_extract_digest_from_stem_too_short() -> None:
    # A bare digest must be exactly SHA256_B64URL_LEN (= 43) chars, and a
    # prefixed stem must be at least SHA256_B64URL_LEN + 2 (= 45) chars (a
    # one-character prefix plus the '_' separator). This 46-char stem of all
    # 'a's is long enough to clear that minimum, but there's no '_' at the
    # expected separator position, so it's still rejected.
    assert extract_digest_from_stem("a" * 46) is None


def test_extract_digest_from_stem_prefix_without_dash() -> None:
    # With relaxed grammar, a prefix without '-' is now accepted.
    stem = f"nodash_{'A' * 43}"
    assert extract_digest_from_stem(stem) == "A" * 43


def test_extract_digest_from_stem_separator_not_underscore() -> None:
    # The character at the computed separator position is not '_'.
    stem = f"330-beachX{'A' * 43}"
    assert extract_digest_from_stem(stem) is None


def test_extract_digest_from_stem_non_numeric_pcs() -> None:
    # With relaxed grammar, prefix validation is removed; only digest is validated.
    stem = f"abc-desc_{'A' * 43}"
    assert extract_digest_from_stem(stem) == "A" * 43


def test_extract_digest_from_stem_empty_descriptor() -> None:
    # With relaxed grammar, any non-empty prefix is accepted.
    stem = f"330-_{'A' * 43}"
    assert extract_digest_from_stem(stem) == "A" * 43


def test_extract_digest_from_stem_non_ascii_pcs() -> None:
    # With relaxed grammar, any prefix (including non-ASCII) is accepted.
    stem = f"٣٣٠-beach_{'A' * 43}"  # "٣٣٠" == 330 in Arabic-Indic
    assert extract_digest_from_stem(stem) == "A" * 43


def test_extract_digest_accepts_tilde_code() -> None:
    digest = "A" * 43
    assert extract_digest_from_stem(f"540~1-holiday_{digest}") == digest


def test_extract_digest_rejects_dotted_code() -> None:
    # With relaxed grammar, dotted codes in prefix are accepted (prefix is unconstrained).
    digest = "A" * 43
    assert extract_digest_from_stem(f"540.1-holiday_{digest}") == digest


# ---------------------------------------------------------------------------
# verify_pcs_file
# ---------------------------------------------------------------------------


def test_verify_pcs_file_valid(tmp_path: Path) -> None:
    content = b"some image bytes"
    digest = encode_base64url(hashlib.sha256(content).digest())
    filename = f"920-test-object_{digest}.jpg"
    p = tmp_path / filename
    p.write_bytes(content)
    assert verify_pcs_file(p) is True


def test_verify_pcs_file_invalid_digest(tmp_path: Path) -> None:
    content = b"some image bytes"
    p = tmp_path / f"920-test-object_{'A' * 43}.jpg"
    p.write_bytes(content)
    assert verify_pcs_file(p) is False


def test_verify_pcs_file_no_underscore(tmp_path: Path) -> None:
    p = tmp_path / "notavalidname.jpg"
    p.write_bytes(b"data")
    assert verify_pcs_file(p) is False


# ---------------------------------------------------------------------------
# New tests for relaxed digest extraction grammar
# ---------------------------------------------------------------------------

_D = "qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI"  # 43 chars, valid base64url


def test_extract_accepts_bare_digest_stem():
    """Undated/<digest>.jpg has no prefix at all."""
    assert extract_digest_from_stem(_D) == _D


def test_extract_accepts_date_and_descriptor():
    assert extract_digest_from_stem(f"2019-07-04-emmas-graduation_{_D}") == _D


def test_extract_accepts_date_only():
    assert extract_digest_from_stem(f"2019-07-04_{_D}") == _D


def test_extract_accepts_descriptor_only():
    """Undated file that still has a human name."""
    assert extract_digest_from_stem(f"beach-trip-scan_{_D}") == _D


def test_extract_still_accepts_legacy_pcs_names():
    """Files organized by the old scheme must remain verifiable."""
    assert extract_digest_from_stem(f"330-beach_{_D}") == _D


def test_extract_rejects_non_base64url_tail():
    bad = "!" * 43
    assert extract_digest_from_stem(f"2019-07-04_{bad}") is None


def test_extract_rejects_empty_prefix():
    assert extract_digest_from_stem(f"_{_D}") is None


def test_extract_rejects_short_stem():
    assert extract_digest_from_stem("abc") is None


def test_extract_rejects_missing_separator():
    assert extract_digest_from_stem(f"2019-07-04x{_D}") is None
