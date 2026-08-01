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
        name = generate_filename("330", "indiana dunes", _FAKE_DIGEST, "jpg")
        assert name == f"330-indiana-dunes_{_FAKE_DIGEST}.jpg"

    def test_extension_lowercased(self) -> None:
        name = generate_filename("110", "portrait", _FAKE_DIGEST, "JPG")
        assert name.endswith(".jpg")

    def test_extension_leading_dot_stripped(self) -> None:
        name = generate_filename("110", "portrait", _FAKE_DIGEST, ".jpg")
        assert name.endswith(".jpg")
        assert ".." not in name

    def test_under_100_chars(self) -> None:
        name = generate_filename("900", "a" * 40, _FAKE_DIGEST, "jpg")
        assert len(name) <= 100

    def test_long_descriptor_capped_by_normalize_descriptor(self) -> None:
        # NOTE: this does NOT exercise the truncation branch in generate_filename.
        # "very " * 20 normalises to just "very-very-very" (3 words, 14 chars)
        # because normalize_descriptor keeps at most 3 words and 30 chars. The
        # resulting name is well under 100, so no truncation occurs.
        long_desc = "very " * 20
        name = generate_filename("900", long_desc, _FAKE_DIGEST, "jpg")
        assert "very-very-very" in name
        assert len(name) <= 100

    def test_truncation_branch_triggered_by_long_extension(self) -> None:
        # The truncation branch (generate_filename lines ~63-68) is reachable,
        # but NOT via a long descriptor (normalize_descriptor caps it at 30
        # chars). It is reached when the *extension* is long enough to push the
        # total past 100. Here: 3-digit pcs + 30-char descriptor + 43-char
        # digest + 25-char extension = 104 pre-truncation, which trims the
        # descriptor down to fit exactly 100.
        name = generate_filename("900", "a" * 30, _FAKE_DIGEST, "e" * 25)
        assert len(name) == 100
        assert len(name) <= 100
        # Descriptor was truncated from 30 to 26 'a's.
        assert "900-" + "a" * 26 + "_" in name
        assert "a" * 27 not in name

    def test_very_long_extension_still_capped_at_100(self) -> None:
        # Even with a pathologically long extension the <=100 guarantee holds:
        # after the descriptor is shrunk to 1 char, the extension itself is
        # truncated by the overflow amount.
        name = generate_filename("900", "a" * 30, _FAKE_DIGEST, "e" * 60)
        assert len(name) <= 100

    def test_multi_dot_extension_collapsed(self) -> None:
        # "tar.gz" collapses to the part after the last dot -> "gz".
        name = generate_filename("900", "beach", _FAKE_DIGEST, "tar.gz")
        assert name.endswith(".gz")
        assert ".tar" not in name

    def test_invalid_chars_stripped_from_extension(self) -> None:
        # Non [a-z0-9] characters are removed from the extension.
        name = generate_filename("900", "beach", _FAKE_DIGEST, "jp<g>")
        assert name.endswith(".jpg")

    def test_path_separator_stripped_from_extension(self) -> None:
        # Path separators must never appear in the generated name.
        name = generate_filename("900", "beach", _FAKE_DIGEST, "jpg/evil")
        assert "/" not in name
        assert "\\" not in name
        # Both '/' path-injection and the sanitiser collapse to "jpgevil".
        assert name.endswith(".jpgevil")

    def test_empty_extension_no_trailing_dot(self) -> None:
        # An empty (or fully invalid) extension yields NO trailing dot.
        name = generate_filename("900", "beach", _FAKE_DIGEST, "")
        assert not name.endswith(".")
        assert "." not in name

    def test_digest_preserved_in_full(self) -> None:
        name = generate_filename("330", "beach", _FAKE_DIGEST, "jpg")
        assert _FAKE_DIGEST in name

    def test_pcs_code_in_name(self) -> None:
        name = generate_filename("330", "beach", _FAKE_DIGEST, "jpg")
        assert name.startswith("330-")

    def test_separator_underscore_before_digest(self) -> None:
        name = generate_filename("330", "beach", _FAKE_DIGEST, "jpg")
        stem = name.rsplit(".", 1)[0]
        assert "_" in stem
        assert stem.rsplit("_", 1)[1] == _FAKE_DIGEST


# ---------------------------------------------------------------------------
# parse_filename
# ---------------------------------------------------------------------------


class TestParseFilename:
    def test_roundtrip(self) -> None:
        original = generate_filename("330", "indiana dunes", _FAKE_DIGEST, "jpg")
        parsed = parse_filename(original)
        assert parsed is not None
        assert parsed["pcs_code"] == "330"
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
        assert parsed["pcs_code"] == "330"

    def test_multiple_underscores_in_descriptor_handled(self) -> None:
        # Descriptor portion may contain underscores in theory; digest is found
        # by fixed-length offset from end, not by last-underscore split.
        # We construct a valid PCS name programmatically:
        from imageharbor.filename import generate_filename
        name = generate_filename("330", "foo bar", _FAKE_DIGEST, "jpg")
        parsed = parse_filename(name)
        assert parsed is not None
        assert parsed["sha256_b64url"] == _FAKE_DIGEST

    def test_stem_too_short_returns_none(self) -> None:
        # Stem length <= SHA256_B64URL_LEN (43) cannot contain a digest.
        assert parse_filename("short.jpg") is None

    def test_separator_not_underscore_returns_none(self) -> None:
        # 44-char stem: sep position holds a non-underscore character.
        assert parse_filename(f"X{_FAKE_DIGEST}.jpg") is None

    def test_accepts_windows_style_path(self) -> None:
        full = rf"C:\Users\photos\330-beach_{_FAKE_DIGEST}.jpg"
        parsed = parse_filename(full)
        assert parsed is not None
        assert parsed["pcs_code"] == "330"
        assert parsed["descriptor"] == "beach"

    def test_extension_normalized_to_lowercase(self) -> None:
        parsed = parse_filename(f"330-beach_{_FAKE_DIGEST}.JPG")
        assert parsed is not None
        assert parsed["extension"] == "jpg"

    def test_multi_dot_extension_roundtrip(self) -> None:
        # generate_filename collapses "tar.gz" -> "gz"; parse recovers it.
        name = generate_filename("900", "beach", _FAKE_DIGEST, "tar.gz")
        parsed = parse_filename(name)
        assert parsed is not None
        assert parsed["descriptor"] == "beach"
        assert parsed["extension"] == "gz"

    def test_empty_descriptor_returns_none(self) -> None:
        # Consistent with extract_digest_from_stem: empty descriptor rejected.
        assert parse_filename(f"330-_{_FAKE_DIGEST}.jpg") is None

    def test_non_ascii_pcs_returns_none(self) -> None:
        # Arabic-Indic digits are not ASCII digits and must be rejected.
        assert parse_filename(f"٣٣٠-beach_{_FAKE_DIGEST}.jpg") is None

    def test_plus_signed_pcs_returns_none(self) -> None:
        # Permissive int() would accept "+330"; the reused validator rejects it.
        assert parse_filename(f"+330-beach_{_FAKE_DIGEST}.jpg") is None


def test_generate_and_parse_tilde_code() -> None:
    name = generate_filename("540~1", "christmas eve", "A" * 43, "jpg")
    assert name.startswith("540~1-")
    parsed = parse_filename(name)
    assert parsed is not None
    assert parsed["pcs_code"] == "540~1"
    assert parsed["descriptor"] == "christmas-eve"


def test_parse_plain_code_is_string() -> None:
    parsed = parse_filename(f"330-beach_{'A' * 43}.jpg")
    assert parsed is not None
    assert parsed["pcs_code"] == "330"  # str, not int


def test_parse_rejects_dotted_code() -> None:
    assert parse_filename(f"540.1-x_{'A' * 43}.jpg") is None
