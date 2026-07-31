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
