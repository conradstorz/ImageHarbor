"""Tests for camera-pattern detection and descriptor resolution."""

from pathlib import Path

import pytest

from imageharbor import tiers
from imageharbor.descriptor import is_camera_generated, resolve_descriptor

CAMERA_STEMS = [
    "IMG_1234",
    "IMG-20190704-WA0001",
    "img_20190704_123456",
    "DSC0042",
    "DSCN0042",
    "DSCF0042",
    "_DSC0042",
    "PXL_20190704_123456789",
    "MVIMG_20190704_123456",
    "P1000042",
    "PICT0042",
    "100_0042",
    "CIMG0042",
    "SAM_0042",
    "GOPR0042",
    "DJI_0042",
    "Screenshot_2019-07-04-12-33-11",
    "Screen Shot 2019-07-04 at 12.33.11",
    "WhatsApp Image 2019-07-04 at 12.33.11",
    "Signal-2019-07-04-123311",
    "FB_IMG_1562243591",
    "received_101234567890",
    "20190704_123456",
    "2019-07-04 12.33.11",
    "1562243591",
    "865948477697870747_account_id=1",
    "112233445566778899_account_id=0",
    "2015-03-09",
    "2015-03-09(1)",
    "2015-03-09(12)",
]

HUMAN_STEMS = [
    "Emma's graduation",
    "beach trip 2019",
    "grandpa and the tractor",
    "kitchen remodel before",
    "scan0001 aunt martha",
    "Christmas",
    # Regression cases: each of these was destroyed by an earlier, looser
    # pattern. A false positive here discards a human's name permanently in
    # favour of an AI guess, so they are pinned deliberately.
    "Screenshot - grandpas last text message",
    "WhatsApp Image of the new puppy",
    "Sam_1",
    "Sam_2",
]


@pytest.mark.parametrize("stem", CAMERA_STEMS)
def test_camera_stems_are_detected(stem):
    assert is_camera_generated(stem) is True


@pytest.mark.parametrize("stem", HUMAN_STEMS)
def test_human_stems_are_not_camera_generated(stem):
    assert is_camera_generated(stem) is False


def test_camera_detection_is_case_insensitive():
    assert is_camera_generated("img_1234") is True
    assert is_camera_generated("dsc0042") is True


def test_resolve_keeps_a_human_name_at_the_top_tier(tmp_path):
    path = tmp_path / "Emma's graduation.jpg"
    path.write_bytes(b"x")
    resolved = resolve_descriptor(path)
    assert resolved.value == "emmas-graduation"
    assert resolved.tier == tiers.DESC_HUMAN_FILENAME
    assert resolved.source == "human_filename"


def test_resolve_discards_a_camera_name(tmp_path):
    path = tmp_path / "IMG_1234.jpg"
    path.write_bytes(b"x")
    resolved = resolve_descriptor(path)
    assert resolved.value == ""
    assert resolved.tier == tiers.DESC_NONE
    assert resolved.source == "none"


def test_resolve_discards_a_stem_that_normalizes_to_nothing(tmp_path):
    """A stem of pure punctuation carries no information."""
    path = tmp_path / "___.jpg"
    path.write_bytes(b"x")
    resolved = resolve_descriptor(path)
    assert resolved.tier == tiers.DESC_NONE


def test_resolve_truncates_to_three_words(tmp_path):
    path = tmp_path / "the big family reunion picnic.jpg"
    path.write_bytes(b"x")
    assert resolve_descriptor(path).value == "the-big-family"


def test_account_id_stem_is_not_a_human_name(tmp_path: Path) -> None:
    """A Hangouts row id is machine-generated; it must not lock out enrichment."""
    path = tmp_path / "865948477697870747_account_id=1.jpg"
    assert resolve_descriptor(path).tier == tiers.DESC_NONE


def test_bare_date_stem_is_not_a_descriptor(tmp_path: Path) -> None:
    """A date is not a description -- the date ladder already captured it."""
    path = tmp_path / "2015-03-09.jpg"
    assert resolve_descriptor(path).tier == tiers.DESC_NONE


def test_bare_date_with_copy_suffix_is_not_a_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "2015-03-09(1).jpg"
    assert resolve_descriptor(path).tier == tiers.DESC_NONE


def test_bare_date_pattern_does_not_over_match(tmp_path: Path) -> None:
    """The new bare-date pattern is anchored: a date PLUS words is not a match.

    `normalize_descriptor` then reduces this to the date alone, because its
    three-word cap is entirely consumed by the date's own three numeric
    tokens (2015, 03, 09). That is pre-existing, deliberately frozen behavior
    (see tests/test_filename.py) and is NOT changed here -- the point of this
    test is only that `is_camera_generated` leaves the stem alive.
    """
    path = tmp_path / "2015-03-09 emma birthday.jpg"
    assert is_camera_generated("2015-03-09 emma birthday") is False
    resolved = resolve_descriptor(path)
    assert resolved.tier == tiers.DESC_HUMAN_FILENAME
    assert resolved.value == "2015-03-09"


def test_a_date_only_descriptor_yields_to_the_enrichment_pass(tmp_path: Path) -> None:
    """The accepted consequence of the above, pinned deliberately.

    Once the pipeline supplies the date it actually resolved, a descriptor that
    reduced to nothing but that same date is discarded -- which is the RIGHT
    outcome: it leaves descriptor_tier at 0, so the enrichment pass can later
    supply a real subject at DESC_AI_SUBJECT (20). The alternative is locking
    the file at tier 30 with a filename that states the date twice and says
    nothing else, which no later pass could ever improve.
    """
    path = tmp_path / "2015-03-09 emma birthday.jpg"
    assert resolve_descriptor(path, date_str="2015-03-09").tier == tiers.DESC_NONE


def test_descriptor_equal_to_resolved_date_is_discarded(tmp_path: Path) -> None:
    """A descriptor that merely restates the date carries no information."""
    path = tmp_path / "2015.03.09.jpg"
    # Without date_str this normalizes to "2015-03-09" at tier 30.
    assert resolve_descriptor(path).tier == tiers.DESC_HUMAN_FILENAME
    # With the date the ladder actually resolved, it is redundant.
    assert resolve_descriptor(path, date_str="2015-03-09").tier == tiers.DESC_NONE


def test_original_name_overrides_a_truncated_member_stem(tmp_path: Path) -> None:
    """Google's `title` is the pre-truncation filename: strictly better evidence."""
    path = tmp_path / "emma-graduation-ceremony-at-the-high-scho.jpg"
    resolved = resolve_descriptor(
        path, original_name="emma graduation ceremony at the high school.jpg"
    )
    assert resolved.tier == tiers.DESC_HUMAN_FILENAME
    assert resolved.value == "emma-graduation-ceremony"


def test_original_name_that_is_camera_generated_still_yields_none(tmp_path: Path) -> None:
    path = tmp_path / "truncated-thing.jpg"
    assert resolve_descriptor(path, original_name="IMG_1234.jpg").tier == tiers.DESC_NONE


def test_blank_original_name_falls_back_to_the_member_stem(tmp_path: Path) -> None:
    path = tmp_path / "beach trip.jpg"
    assert resolve_descriptor(path, original_name="  ").value == "beach-trip"
