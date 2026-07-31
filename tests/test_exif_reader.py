"""Tests for imageharbor.exif_reader."""

from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from imageharbor.exif_reader import (
    _dms_to_decimal,
    _rational_to_float,
    read_exif,
)


# ---------------------------------------------------------------------------
# _rational_to_float
# ---------------------------------------------------------------------------


def test_rational_to_float_numerator_denominator() -> None:
    # Fraction exposes .numerator / .denominator
    assert _rational_to_float(Fraction(3, 4)) == pytest.approx(0.75)


def test_rational_to_float_two_tuple() -> None:
    assert _rational_to_float((3, 4)) == pytest.approx(0.75)


def test_rational_to_float_zero_denominator_object() -> None:
    # Fraction(0, x) has denominator 1, so build a tiny stand-in object
    class _Rational:
        numerator = 5
        denominator = 0

    assert _rational_to_float(_Rational()) == 0.0


def test_rational_to_float_zero_denominator_tuple() -> None:
    assert _rational_to_float((5, 0)) == 0.0


def test_rational_to_float_int_passthrough() -> None:
    assert _rational_to_float(7) == 7


def test_rational_to_float_float_passthrough() -> None:
    assert _rational_to_float(2.5) == 2.5


# ---------------------------------------------------------------------------
# _dms_to_decimal
# ---------------------------------------------------------------------------


def test_dms_to_decimal_north() -> None:
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, "N") == pytest.approx(41.5)


def test_dms_to_decimal_east() -> None:
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, "E") == pytest.approx(41.5)


def test_dms_to_decimal_south_negates() -> None:
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, "S") == pytest.approx(-41.5)


def test_dms_to_decimal_west_negates() -> None:
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, "W") == pytest.approx(-41.5)


def test_dms_to_decimal_with_seconds() -> None:
    # 41 deg 30 min 36 sec = 41 + 0.5 + 0.01 = 41.51
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(36, 1))
    assert _dms_to_decimal(dms, "N") == pytest.approx(41.51)


def test_dms_to_decimal_ref_case_insensitive() -> None:
    dms = (Fraction(10, 1), Fraction(0, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, "s") == pytest.approx(-10.0)


def test_dms_to_decimal_bytes_ref_south_negates() -> None:
    # Some encoders emit the hemisphere ref as bytes (e.g. b"S"). The sign
    # must still flip (bytes.upper() would not equal "S" without normalization).
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, b"S") == pytest.approx(-41.5)


def test_dms_to_decimal_bytes_ref_north_no_flip() -> None:
    dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    assert _dms_to_decimal(dms, b"N") == pytest.approx(41.5)


def test_dms_to_decimal_lat_and_lon_independent() -> None:
    # Each coordinate is computed on its own; a valid latitude tuple and a
    # valid longitude tuple resolve independently at the helper level.
    lat_dms = (Fraction(41, 1), Fraction(30, 1), Fraction(0, 1))
    lon_dms = (Fraction(87, 1), Fraction(0, 1), Fraction(0, 1))
    assert _dms_to_decimal(lat_dms, "N") == pytest.approx(41.5)
    assert _dms_to_decimal(lon_dms, "W") == pytest.approx(-87.0)


# ---------------------------------------------------------------------------
# read_exif — no EXIF
# ---------------------------------------------------------------------------


def test_read_exif_no_exif(tmp_path: Path) -> None:
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (4, 4), "red").save(p, "JPEG")

    result = read_exif(p)

    assert result["format"] == "JPEG"
    assert result["mode"] == "RGB"
    assert result["width"] == 4
    assert result["height"] == 4


# ---------------------------------------------------------------------------
# read_exif — with EXIF
# ---------------------------------------------------------------------------


def test_read_exif_with_exif(tmp_path: Path) -> None:
    p = tmp_path / "withexif.jpg"
    exif = Image.Exif()
    exif[271] = "TestMake"      # Make
    exif[272] = "TestModel"     # Model
    exif[305] = "TestSoftware"  # Software
    Image.new("RGB", (8, 8), "blue").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    assert result["Make"] == "TestMake"
    assert result["Model"] == "TestModel"
    assert result["Software"] == "TestSoftware"
    # Basic image metadata still present
    assert result["width"] == 8
    assert result["height"] == 8


# ---------------------------------------------------------------------------
# read_exif — corrupt / non-image
# ---------------------------------------------------------------------------


def test_read_exif_corrupt_file_does_not_raise(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"\x00\x01\x02not a real image\xff\xfe\xab")

    result = read_exif(p)

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# read_exif — nonexistent path
# ---------------------------------------------------------------------------


def test_read_exif_nonexistent_path_does_not_raise(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.jpg"

    result = read_exif(p)

    # Image.open raises FileNotFoundError, caught internally -> empty dict
    assert result == {}
