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
]

HUMAN_STEMS = [
    "Emma's graduation",
    "beach trip 2019",
    "grandpa and the tractor",
    "kitchen remodel before",
    "scan0001 aunt martha",
    "Christmas",
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
