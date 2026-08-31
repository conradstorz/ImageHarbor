"""Tests for the date ladder and folder derivation."""

from datetime import datetime
from pathlib import Path

import pytest

from imageharbor import tiers
from imageharbor.date_resolver import UNDATED_FOLDER, date_from_filename, resolve_date


def _p(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"x")
    return path


# --- the ladder -----------------------------------------------------------

def test_exif_original_is_the_top_rung(tmp_path):
    exif = {
        "DateTimeOriginal": "2019:07:04 12:33:11",
        "DateTimeDigitized": "2020:01:01 00:00:00",
        "DateTime": "2021:01:01 00:00:00",
    }
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), exif)
    assert resolved.value == datetime(2019, 7, 4, 12, 33, 11)
    assert resolved.tier == tiers.DATE_EXIF_ORIGINAL
    assert resolved.source == "exif_original"


def test_falls_back_to_other_exif_fields(tmp_path):
    exif = {"DateTime": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), exif)
    assert resolved.tier == tiers.DATE_EXIF_OTHER
    assert resolved.source == "exif_other"


def test_falls_back_to_the_filename(tmp_path):
    resolved = resolve_date(_p(tmp_path, "IMG_20190704_123456.jpg"), {})
    assert resolved.value == datetime(2019, 7, 4, 12, 34, 56)
    assert resolved.tier == tiers.DATE_FILENAME_PATTERN
    assert resolved.source == "filename_pattern"


def test_no_evidence_means_undated(tmp_path):
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), {})
    assert resolved.value is None
    assert resolved.tier == tiers.DATE_NONE
    assert resolved.source == "none"


def test_mtime_is_never_used(tmp_path):
    """mtime is evidence of copying, not of capture."""
    path = _p(tmp_path, "IMG_1234.jpg")
    import os
    os.utime(path, (1562243591, 1562243591))
    assert resolve_date(path, {}).tier == tiers.DATE_NONE


# --- filename patterns ----------------------------------------------------

@pytest.mark.parametrize(
    "stem,expected",
    [
        ("IMG_20190704_123456", datetime(2019, 7, 4, 12, 34, 56)),
        ("PXL_20190704_123456789", datetime(2019, 7, 4, 12, 34, 56)),
        ("20190704_123456", datetime(2019, 7, 4, 12, 34, 56)),
        ("Screenshot_2019-07-04-12-33-11", datetime(2019, 7, 4, 12, 33, 11)),
        ("2019-07-04 12.33.11", datetime(2019, 7, 4, 12, 33, 11)),
        ("WhatsApp Image 2019-07-04 at 12.33.11", datetime(2019, 7, 4)),
        ("beach trip 2019-07-04", datetime(2019, 7, 4)),
        ("IMG-20190704-WA0001", datetime(2019, 7, 4)),
        ("2019.07.04", datetime(2019, 7, 4)),
        ("2019 07 04", datetime(2019, 7, 4)),
    ],
)
def test_date_from_filename_hits(stem, expected):
    assert date_from_filename(stem) == expected


@pytest.mark.parametrize(
    "stem",
    ["IMG_1234", "DSC0042", "Emma's graduation", "1562243591", "20190732_123456"],
)
def test_date_from_filename_misses(stem):
    """A bare epoch is deliberately not decoded, and 07-32 is not a date."""
    assert date_from_filename(stem) is None


def test_implausible_years_are_rejected(tmp_path):
    exif = {"DateTimeOriginal": "1601:01:01 00:00:00"}
    assert resolve_date(_p(tmp_path, "x.jpg"), exif).tier == tiers.DATE_NONE


def test_malformed_exif_date_is_ignored(tmp_path):
    exif = {"DateTimeOriginal": "not a date", "DateTime": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "x.jpg"), exif)
    assert resolved.tier == tiers.DATE_EXIF_OTHER


def test_exif_zero_date_is_ignored(tmp_path):
    """Cameras with a dead clock emit all-zero timestamps."""
    exif = {"DateTimeOriginal": "0000:00:00 00:00:00"}
    assert resolve_date(_p(tmp_path, "x.jpg"), exif).tier == tiers.DATE_NONE


# --- folder derivation ----------------------------------------------------

def test_folder_is_year_and_month(tmp_path):
    exif = {"DateTimeOriginal": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "x.jpg"), exif)
    assert resolved.folder == "2019/2019-07"
    assert resolved.date_str == "2019-07-04"


def test_undated_folder(tmp_path):
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), {})
    assert resolved.folder == UNDATED_FOLDER
    assert resolved.date_str is None


# --- external sidecar rung -------------------------------------------------

def test_exif_original_outranks_external_sidecar(tmp_path) -> None:
    exif = {"DateTimeOriginal": "2019:07:04 12:33:11"}
    resolved = resolve_date(
        _p(tmp_path, "IMG_1234.jpg"), exif, external_date=datetime(2015, 3, 9, 12, 56, 32)
    )
    assert resolved.tier == tiers.DATE_EXIF_ORIGINAL
    assert resolved.value == datetime(2019, 7, 4, 12, 33, 11)


def test_external_sidecar_outranks_exif_digitized(tmp_path) -> None:
    exif = {"DateTimeDigitized": "2019:07:04 12:33:11"}
    resolved = resolve_date(
        _p(tmp_path, "IMG_1234.jpg"), exif, external_date=datetime(2015, 3, 9, 12, 56, 32)
    )
    assert resolved.tier == tiers.DATE_EXTERNAL_SIDECAR
    assert resolved.source == "external_sidecar"
    assert resolved.value == datetime(2015, 3, 9, 12, 56, 32)
    assert resolved.folder == "2015/2015-03"


def test_external_sidecar_outranks_a_filename_pattern(tmp_path) -> None:
    resolved = resolve_date(
        _p(tmp_path, "IMG_20190704_123456.jpg"), {}, external_date=datetime(2015, 3, 9)
    )
    assert resolved.tier == tiers.DATE_EXTERNAL_SIDECAR
    assert resolved.value == datetime(2015, 3, 9)


def test_external_sidecar_is_used_when_there_is_nothing_else(tmp_path) -> None:
    resolved = resolve_date(_p(tmp_path, "photo.jpg"), {}, external_date=datetime(2015, 3, 9))
    assert resolved.tier == tiers.DATE_EXTERNAL_SIDECAR


def test_implausible_external_date_is_ignored(tmp_path) -> None:
    """An out-of-range external date must fall through, not be asserted."""
    resolved = resolve_date(_p(tmp_path, "photo.jpg"), {}, external_date=datetime(1600, 1, 1))
    assert resolved.tier == tiers.DATE_NONE
    assert resolved.folder == UNDATED_FOLDER


def test_external_date_none_leaves_the_ladder_unchanged(tmp_path) -> None:
    exif = {"DateTimeDigitized": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), exif, external_date=None)
    assert resolved.tier == tiers.DATE_EXIF_OTHER


def test_unknown_external_date_tier_degrades_instead_of_raising(tmp_path) -> None:
    """`external_date_tier` is caller-supplied; every current caller passes
    DATE_RELATED_SIDECAR (25) or DATE_EXTERNAL_SIDECAR (30), but an unknown
    tier must degrade to a synthesized source name rather than raise
    `KeyError` out of `DATE_SOURCE_NAMES` mid-resolution."""
    resolved = resolve_date(
        _p(tmp_path, "photo.jpg"), {},
        external_date=datetime(2015, 3, 9), external_date_tier=999,
    )
    assert resolved.tier == 999
    assert resolved.source == "external_tier_999"
