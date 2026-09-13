"""Tests for imageharbor.exif_reader."""

from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

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


def test_read_exif_bytes_valued_tag_normalizes_to_text(tmp_path: Path) -> None:
    """ExifVersion (and FlashPixVersion/SceneType/GPSVersionID) are commonly
    stored as raw bytes on real cameras. `read_exif` must decode them to text
    -- matching `sidecar._json_default`'s write-time decode -- so a value
    written to a sidecar and read back on the next pass compares equal
    instead of tripping a false supersession (`"0230" != b"0230"`)."""
    p = tmp_path / "withversion.jpg"
    exif = Image.Exif()
    exif[36864] = b"0230"  # ExifVersion
    Image.new("RGB", (4, 4), "green").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    assert result["ExifVersion"] == "0230"
    assert isinstance(result["ExifVersion"], str)


def test_read_exif_merged_twice_records_no_false_supersession(tmp_path: Path) -> None:
    """End-to-end THROUGH DISK: a bytes-valued EXIF tag merged and written to
    a real sidecar file, then read back via a fresh `read_exif` call and
    merged again, must not grow `exif_history`.

    This must round-trip through actual JSON on disk (`merge_sidecar` +
    `read_sidecar`), not just call `sidecar_schema.merge` in memory twice --
    an in-memory merge sees the same Python `bytes` object both times
    regardless of normalization and would pass even with the bug present.
    The bug only shows once the FIRST write has gone through
    `sidecar._json_default` (bytes -> text) and comes back off disk as text,
    to be compared against a second, unnormalized `bytes` read.
    """
    from imageharbor.sidecar import merge_sidecar, read_sidecar

    p = tmp_path / "withversion2.jpg"
    exif = Image.Exif()
    exif[36864] = b"0230"  # ExifVersion
    Image.new("RGB", (4, 4), "green").save(p, "JPEG", exif=exif.tobytes())

    first_read = read_exif(p)
    merge_sidecar(p, {"exif": first_read})

    second_read = read_exif(p)
    merge_sidecar(p, {"exif": second_read})

    doc = read_sidecar(p)
    assert doc.get("exif_history", []) == []


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


# ---------------------------------------------------------------------------
# read_exif — GPS sub-IFD walk (exif_reader.py:149-182)
#
# These write a real GPSInfo (tag 34853) sub-IFD through Pillow's Image.Exif
# and re-read it via read_exif's actual Image.open()/_getexif() path, so
# each test genuinely exercises the GPS block rather than the pure
# `_dms_to_decimal` helper already covered above.
# ---------------------------------------------------------------------------


def test_gps_dms_rationals_become_signed_decimal_degrees(tmp_path: Path) -> None:
    p = tmp_path / "gps_ne.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (Fraction(40, 1), Fraction(26, 1), Fraction(46, 1)),
        3: "W",
        4: (Fraction(79, 1), Fraction(58, 1), Fraction(56, 1)),
    }
    Image.new("RGB", (4, 4), "red").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    gps = result["GPS"]
    assert gps["latitude_decimal"] == pytest.approx(40.4461111, abs=1e-6)
    # West longitude flips sign even though the ref itself reads positive.
    assert gps["longitude_decimal"] == pytest.approx(-79.9822222, abs=1e-6)


def test_southern_and_western_hemispheres_are_negative(tmp_path: Path) -> None:
    p = tmp_path / "gps_sw.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "S",
        2: (Fraction(33, 1), Fraction(52, 1), Fraction(4, 1)),
        3: "W",
        4: (Fraction(151, 1), Fraction(12, 1), Fraction(36, 1)),
    }
    Image.new("RGB", (4, 4), "red").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    gps = result["GPS"]
    assert gps["latitude_decimal"] < 0
    assert gps["longitude_decimal"] < 0


def test_a_malformed_longitude_does_not_drop_a_valid_latitude(tmp_path: Path) -> None:
    # GPSLongitude has only 2 components instead of 3 (degrees, minutes) --
    # _dms_to_decimal indexes dms[2] for seconds and raises IndexError, which
    # is caught by the longitude coordinate's own `except Exception: pass`
    # arm (exif_reader.py:173-180) without disturbing the latitude arm above
    # it (:165-172).
    p = tmp_path / "gps_malformed_lon.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (Fraction(40, 1), Fraction(26, 1), Fraction(46, 1)),
        3: "W",
        4: (Fraction(79, 1), Fraction(58, 1)),  # missing seconds component
    }
    Image.new("RGB", (4, 4), "red").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    gps = result["GPS"]
    assert gps["latitude_decimal"] == pytest.approx(40.4461111, abs=1e-6)
    assert "longitude_decimal" not in gps


def test_a_malformed_latitude_does_not_drop_a_valid_longitude(tmp_path: Path) -> None:
    # Mirror of the malformed-longitude case above, exercising the
    # latitude coordinate's own `except Exception: pass` arm
    # (exif_reader.py:165-172) instead of the longitude arm.
    p = tmp_path / "gps_malformed_lat.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (Fraction(40, 1), Fraction(26, 1)),  # missing seconds component
        3: "W",
        4: (Fraction(79, 1), Fraction(58, 1), Fraction(56, 1)),
    }
    Image.new("RGB", (4, 4), "red").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    gps = result["GPS"]
    assert "latitude_decimal" not in gps
    assert gps["longitude_decimal"] == pytest.approx(-79.9822222, abs=1e-6)


def test_zero_denominator_rationals_are_dropped_not_raised(tmp_path: Path) -> None:
    # A zero-denominator rational (IFDRational allows constructing one; real
    # cameras occasionally emit these) must not raise inside the GPS block --
    # _rational_to_float treats it as 0.0 rather than dividing by zero, so
    # the coordinate still resolves using its other components.
    p = tmp_path / "gps_zero_denom.jpg"
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (IFDRational(5, 0), Fraction(26, 1), Fraction(46, 1)),
        3: "E",
        4: (Fraction(79, 1), Fraction(58, 1), Fraction(56, 1)),
    }
    Image.new("RGB", (4, 4), "red").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    gps = result["GPS"]
    # Degrees component collapsed to 0.0 instead of raising; minutes/seconds
    # still contribute.
    assert gps["latitude_decimal"] == pytest.approx(0.4461111, abs=1e-6)
    assert gps["longitude_decimal"] == pytest.approx(79.9822222, abs=1e-6)


def test_gps_ref_missing_defaults_do_not_invent_a_hemisphere(tmp_path: Path) -> None:
    # No GPSLatitudeRef (tag 1) or GPSLongitudeRef (tag 3) at all. Pinning
    # today's actual behavior: `gps.get("GPSLatitudeRef", "N")` /
    # `gps.get("GPSLongitudeRef", "E")` silently default to the positive
    # (Northern/Eastern) hemisphere rather than surfacing the ref as absent.
    # NOTE (see task report): this is a debatable default -- a genuinely
    # missing ref could just as easily belong to S/W -- but it is pinned as
    # existing behavior, not "fixed", per the task brief.
    p = tmp_path / "gps_no_ref.jpg"
    exif = Image.Exif()
    exif[34853] = {
        2: (Fraction(40, 1), Fraction(26, 1), Fraction(46, 1)),
        4: (Fraction(79, 1), Fraction(58, 1), Fraction(56, 1)),
    }
    Image.new("RGB", (4, 4), "red").save(p, "JPEG", exif=exif.tobytes())

    result = read_exif(p)

    gps = result["GPS"]
    assert "GPSLatitudeRef" not in gps
    assert "GPSLongitudeRef" not in gps
    assert gps["latitude_decimal"] == pytest.approx(40.4461111, abs=1e-6)
    assert gps["longitude_decimal"] == pytest.approx(79.9822222, abs=1e-6)
