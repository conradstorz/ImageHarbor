"""Tests for the organized filename grammar."""

import pytest

from imageharbor.filename import build_filename, normalize_descriptor, parse_filename

_D = "qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI"


# --- normalize_descriptor (behavior preserved from the previous scheme) ----

def test_normalize_lowercases_and_hyphenates():
    assert normalize_descriptor("Indiana Dunes") == "indiana-dunes"


def test_normalize_keeps_at_most_three_words():
    assert normalize_descriptor("a b c d e") == "a-b-c"


def test_normalize_falls_back_to_photo():
    assert normalize_descriptor("!!!") == "photo"


# --- build_filename -------------------------------------------------------

def test_build_with_date_and_descriptor():
    assert (
        build_filename("2019-07-04", "emmas-graduation", _D, "jpg")
        == f"2019-07-04-emmas-graduation_{_D}.jpg"
    )


def test_build_with_date_only():
    assert build_filename("2019-07-04", None, _D, "jpg") == f"2019-07-04_{_D}.jpg"


def test_build_with_descriptor_only():
    assert build_filename(None, "beach-scan", _D, "jpg") == f"beach-scan_{_D}.jpg"


def test_build_with_neither_is_a_bare_digest():
    assert build_filename(None, None, _D, "jpg") == f"{_D}.jpg"


def test_build_treats_empty_descriptor_as_absent():
    assert build_filename("2019-07-04", "", _D, "jpg") == f"2019-07-04_{_D}.jpg"


def test_build_normalizes_the_extension():
    assert build_filename(None, None, _D, ".JPEG") == f"{_D}.jpeg"


def test_build_stays_within_100_chars_and_truncates_descriptor_not_date():
    name = build_filename("2019-07-04", "a" * 80, _D, "jpg")
    assert len(name) <= 100
    assert name.startswith("2019-07-04-")
    assert name.endswith(f"_{_D}.jpg")


def test_build_disambiguates_a_date_shaped_descriptor_with_no_date():
    """A scan named "2019.07.04.jpg" normalizes to a date-shaped descriptor.

    Emitting it verbatim would produce a name identical to a genuinely dated
    file's, asserting a date the system never established and contradicting the
    Undated/ folder it lives in.
    """
    name = build_filename(None, "2019-07-04", _D, "jpg")
    assert name == f"20190704_{_D}.jpg"
    parsed = parse_filename(name)
    assert parsed["date"] is None
    assert parsed["descriptor"] == "20190704"


def test_a_real_date_is_still_emitted_verbatim():
    """The guard must only fire when no date was supplied."""
    assert build_filename("2019-07-04", None, _D, "jpg") == f"2019-07-04_{_D}.jpg"


def test_build_output_round_trips():
    name = build_filename("2019-07-04", "emmas-graduation", _D, "jpg")
    parsed = parse_filename(name)
    assert parsed == {
        "date": "2019-07-04",
        "descriptor": "emmas-graduation",
        "sha256_b64url": _D,
        "extension": "jpg",
    }


# --- parse_filename -------------------------------------------------------

@pytest.mark.parametrize(
    "name,date,descriptor",
    [
        (f"2019-07-04-emmas-graduation_{_D}.jpg", "2019-07-04", "emmas-graduation"),
        (f"2019-07-04_{_D}.jpg", "2019-07-04", ""),
        (f"beach-scan_{_D}.jpg", None, "beach-scan"),
        (f"{_D}.jpg", None, ""),
        # A legacy PCS name has no date, so the whole prefix is the descriptor.
        (f"330-beach_{_D}.jpg", None, "330-beach"),
    ],
)
def test_parse_variants(name, date, descriptor):
    parsed = parse_filename(name)
    assert parsed is not None
    assert parsed["date"] == date
    assert parsed["descriptor"] == descriptor
    assert parsed["sha256_b64url"] == _D


def test_parse_accepts_a_full_path():
    parsed = parse_filename(f"/lib/2019/2019-07/2019-07-04_{_D}.jpg")
    assert parsed is not None
    assert parsed["date"] == "2019-07-04"


def test_parse_rejects_a_non_organized_name():
    assert parse_filename("IMG_1234.jpg") is None


def test_parse_does_not_mistake_a_numeric_descriptor_for_a_date():
    parsed = parse_filename(f"2019-summer_{_D}.jpg")
    assert parsed is not None
    assert parsed["date"] is None
    assert parsed["descriptor"] == "2019-summer"
