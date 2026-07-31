"""EXIF/XMP metadata reader using Pillow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pillow EXIF tag numbers that are most useful for cataloguing
_INTERESTING_TAGS: dict[int, str] = {
    271: "Make",
    272: "Model",
    274: "Orientation",
    282: "XResolution",
    283: "YResolution",
    296: "ResolutionUnit",
    305: "Software",
    306: "DateTime",
    315: "Artist",
    33432: "Copyright",
    33434: "ExposureTime",
    33437: "FNumber",
    34850: "ExposureProgram",
    34855: "ISOSpeedRatings",
    36864: "ExifVersion",
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    37383: "MeteringMode",
    37384: "LightSource",
    37385: "Flash",
    37386: "FocalLength",
    40960: "FlashPixVersion",
    40961: "ColorSpace",
    41495: "SensingMethod",
    41729: "SceneType",
    34853: "GPSInfo",
}

_GPS_TAGS: dict[int, str] = {
    0: "GPSVersionID",
    1: "GPSLatitudeRef",
    2: "GPSLatitude",
    3: "GPSLongitudeRef",
    4: "GPSLongitude",
    5: "GPSAltitudeRef",
    6: "GPSAltitude",
    7: "GPSTimeStamp",
    12: "GPSSpeedRef",
    13: "GPSSpeed",
    17: "GPSImgDirectionRef",
    18: "GPSImgDirection",
    29: "GPSDateStamp",
}


def _rational_to_float(value: Any) -> float | Any:
    """Convert an IFDRational or (num, denom) tuple to a float."""
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denom = value.denominator
        return float(value.numerator) / denom if denom else 0.0
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / value[1] if value[1] else 0.0
    return value


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    degrees = _rational_to_float(dms[0])
    minutes = _rational_to_float(dms[1]) / 60.0
    seconds = _rational_to_float(dms[2]) / 3600.0
    decimal = degrees + minutes + seconds
    if ref.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def read_exif(path: Path) -> dict[str, Any]:
    """Return a dict of EXIF fields from *path*.

    Returns an empty dict if EXIF data is absent or unreadable.
    Uses Pillow so no native library bindings are required.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        logger.warning("Pillow not installed; EXIF reading disabled")
        return {}

    result: dict[str, Any] = {}
    try:
        with Image.open(path) as img:
            result["format"] = img.format
            result["mode"] = img.mode
            result["width"], result["height"] = img.size

            exif_data = img._getexif()  # type: ignore[attr-defined]
            if not exif_data:
                return result

            gps_raw: dict[int, Any] | None = None
            for tag_id, value in exif_data.items():
                tag_name = _INTERESTING_TAGS.get(tag_id)
                if tag_name == "GPSInfo":
                    gps_raw = value
                    continue
                if tag_name is None:
                    continue
                # Serialise rationals
                if hasattr(value, "numerator"):
                    value = _rational_to_float(value)
                elif isinstance(value, tuple):
                    value = [
                        _rational_to_float(v)
                        if hasattr(v, "numerator")
                        else v
                        for v in value
                    ]
                result[tag_name] = value

            # Parse GPS sub-IFD
            if gps_raw:
                gps: dict[str, Any] = {}
                for gps_id, gps_val in gps_raw.items():
                    gps_name = _GPS_TAGS.get(gps_id, f"GPS_{gps_id}")
                    if isinstance(gps_val, tuple) and gps_val and not isinstance(gps_val[0], int):
                        gps_val = [
                            _rational_to_float(v)
                            if hasattr(v, "numerator")
                            else v
                            for v in gps_val
                        ]
                    gps[gps_name] = gps_val

                # Compute decimal lat/lon when possible
                try:
                    lat = _dms_to_decimal(
                        tuple(gps.get("GPSLatitude", [])),
                        gps.get("GPSLatitudeRef", "N"),
                    )
                    lon = _dms_to_decimal(
                        tuple(gps.get("GPSLongitude", [])),
                        gps.get("GPSLongitudeRef", "E"),
                    )
                    gps["latitude_decimal"] = round(lat, 7)
                    gps["longitude_decimal"] = round(lon, 7)
                except Exception:
                    pass

                result["GPS"] = gps

    except Exception as exc:
        logger.debug("EXIF read failed for %s: %s", path, exc)

    return result
