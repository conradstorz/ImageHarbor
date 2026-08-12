"""Tests for the tier ladders and the monotonicity predicate."""

import pytest

from imageharbor import tiers


def test_date_ladder_is_ordered():
    assert (
        tiers.DATE_EXIF_ORIGINAL
        > tiers.DATE_EXTERNAL_SIDECAR
        > tiers.DATE_EXIF_OTHER
        > tiers.DATE_FILENAME_PATTERN
        > tiers.DATE_NONE
    )


def test_descriptor_ladder_puts_humans_above_ai():
    assert tiers.DESC_HUMAN_FILENAME > tiers.DESC_AI_SUBJECT > tiers.DESC_NONE


def test_source_names_cover_every_rank():
    assert set(tiers.DATE_SOURCE_NAMES) == {40, 30, 20, 10, 0}
    assert set(tiers.DESC_SOURCE_NAMES) == {30, 20, 0}


# (old_date, old_desc), (new_date, new_desc), expected
UPGRADE_CASES = [
    # Equal in both dimensions is never an upgrade -- this is what makes a
    # re-run a no-op.
    ((40, 30), (40, 30), False),
    ((0, 0), (0, 0), False),
    # Strictly better in one dimension, equal in the other.
    ((40, 0), (40, 20), True),
    ((0, 20), (40, 20), True),
    # Strictly better in both.
    ((0, 0), (40, 30), True),
    # Worse in either dimension is never an upgrade, even if the other improves.
    ((40, 30), (40, 20), False),
    ((40, 30), (0, 30), False),
    ((0, 30), (40, 20), False),
    ((40, 20), (20, 30), False),
]


@pytest.mark.parametrize("old,new,expected", UPGRADE_CASES)
def test_is_upgrade(old, new, expected):
    assert tiers.is_upgrade(old, new) is expected


def test_ai_can_never_displace_a_human_filename():
    """The central information-preservation guarantee."""
    human = (tiers.DATE_EXIF_ORIGINAL, tiers.DESC_HUMAN_FILENAME)
    ai = (tiers.DATE_EXIF_ORIGINAL, tiers.DESC_AI_SUBJECT)
    assert tiers.is_upgrade(human, ai) is False
    assert tiers.is_upgrade((tiers.DATE_EXIF_ORIGINAL, tiers.DESC_NONE), ai) is True
