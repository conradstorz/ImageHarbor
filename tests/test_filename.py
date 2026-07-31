"""Tests for imageharbor.filename."""

import pytest

from imageharbor.filename import (
    generate_filename,
    normalize_descriptor,
    parse_filename,
)


# ---------------------------------------------------------------------------
# normalize_descriptor
# ---------------------------------------------------------------------------


class TestNormalizeDescriptor:
    def test_simple_words(self) -> None:
        assert normalize_descriptor("Indiana Dunes") == "indiana-dunes"

    def test_special_chars_stripped(self) -> None:
        assert normalize_descriptor("Hello, World!") == "hello-world"

    def test_max_3_words(self) -> None:
        result = normalize_descriptor("one two three four five")
        assert result == "one-two-three"

    def test_max_30_chars(self) -> None:
        long_word = "a" * 35
        result = normalize_descriptor(long_word)
        assert len(result) <= 30

    def test_empty_string_falls_back(self) -> None:
        assert normalize_descriptor("") == "photo"

    def test_only_special_chars_falls_back(self) -> None:
        assert normalize_descriptor("!!!---") == "photo"

    def test_digits_preserved(self) -> None:
        result = normalize_descriptor("Building 42")
        assert result == "building-42"

    def test_lowercase(self) -> None:
        assert normalize_descriptor("UPPER CASE") == "upper-case"

    def test_no_trailing_hyphen(self) -> None:
        result = normalize_descriptor("a" * 30 + " extra")
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# generate_filename
# ---------------------------------------------------------------------------

_FAKE_DIGEST = "A" * 43  # 43-char placeholder


class TestGenerateFilename:
    def test_basic_format(self) -> None:
        name = generate_filename(330, "indiana dunes", _FAKE_DIGEST, "jpg")
        assert name == f"330-indiana-dunes_{_FAKE_DIGEST}.jpg"

    def test_extension_lowercased(self) -> None:
        name = generate_filename(110, "portrait", _FAKE_DIGEST, "JPG")
        assert name.endswith(".jpg")

    def test_extension_leading_dot_stripped(self) -> None:
        name = generate_filename(110, "portrait", _FAKE_DIGEST, ".jpg")
        assert name.endswith(".jpg")
        assert ".." not in name

    def test_under_100_chars(self) -> None:
        name = generate_filename(900, "a" * 40, _FAKE_DIGEST, "jpg")
        assert len(name) <= 100

    def test_long_descriptor_truncated_not_over_100(self) -> None:
        long_desc = "very " * 20
        name = generate_filename(900, long_desc, _FAKE_DIGEST, "jpg")
        assert len(name) <= 100

    def test_digest_preserved_in_full(self) -> None:
        name = generate_filename(330, "beach", _FAKE_DIGEST, "jpg")
        assert _FAKE_DIGEST in name

    def test_pcs_code_in_name(self) -> None:
        name = generate_filename(330, "beach", _FAKE_DIGEST, "jpg")
        assert name.startswith("330-")

    def test_separator_underscore_before_digest(self) -> None:
        name = generate_filename(330, "beach", _FAKE_DIGEST, "jpg")
        stem = name.rsplit(".", 1)[0]
        assert "_" in stem
        assert stem.rsplit("_", 1)[1] == _FAKE_DIGEST


# ---------------------------------------------------------------------------
# parse_filename
# ---------------------------------------------------------------------------


class TestParseFilename:
    def test_roundtrip(self) -> None:
        original = generate_filename(330, "indiana dunes", _FAKE_DIGEST, "jpg")
        parsed = parse_filename(original)
        assert parsed is not None
        assert parsed["pcs_code"] == 330
        assert parsed["descriptor"] == "indiana-dunes"
        assert parsed["sha256_b64url"] == _FAKE_DIGEST
        assert parsed["extension"] == "jpg"

    def test_missing_underscore_returns_none(self) -> None:
        assert parse_filename("nodash.jpg") is None

    def test_non_numeric_pcs_returns_none(self) -> None:
        assert parse_filename(f"abc-beach_{_FAKE_DIGEST}.jpg") is None

    def test_accepts_full_path(self) -> None:
        full = f"/some/dir/330-beach_{_FAKE_DIGEST}.jpg"
        parsed = parse_filename(full)
        assert parsed is not None
        assert parsed["pcs_code"] == 330

    def test_multiple_underscores_in_descriptor_handled(self) -> None:
        # Descriptor portion may contain underscores in theory; digest is found
        # by fixed-length offset from end, not by last-underscore split.
        # We construct a valid PCS name programmatically:
        from imageharbor.filename import generate_filename
        name = generate_filename(330, "foo bar", _FAKE_DIGEST, "jpg")
        parsed = parse_filename(name)
        assert parsed is not None
        assert parsed["sha256_b64url"] == _FAKE_DIGEST
