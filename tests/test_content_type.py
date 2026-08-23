"""Content sniffing is pure: bytes in, verdict out."""

import pytest

from imageharbor import content_type


@pytest.mark.parametrize(
    "head, expected",
    [
        (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00", "image"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "image"),
        (b"GIF89a\x10\x00\x10\x00", "image"),
        (b"GIF87a\x10\x00\x10\x00", "image"),
        (b"BM\x36\x00\x00\x00\x00\x00\x00\x00", "image"),
        (b"II*\x00\x08\x00\x00\x00", "image"),
        (b"MM\x00*\x00\x00\x00\x08", "image"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image"),
    ],
)
def test_image_signatures(head, expected):
    assert content_type.sniff(head) == expected


@pytest.mark.parametrize(
    "head, expected",
    [
        (b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00", "video"),
        (b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00", "video"),
        (b"\x00\x00\x00\x18ftyp3gp4\x00\x00\x00\x00", "video"),
        (b"\x1aE\xdf\xa3\x01\x00\x00\x00", "video"),
        (b"RIFF\x24\x00\x00\x00AVI LIST", "video"),
        (b"FLV\x01\x05\x00\x00\x00\x09", "video"),
        (b"0&\xb2u\x8ef\xcf\x11\xa6\xd9\x00\xaa", "video"),
        (b"\x00\x00\x01\xba\x44\x00\x04\x00", "video"),
        (b"\x00\x00\x01\xb3\x12\x00\xf0\x13", "video"),
    ],
)
def test_video_signatures(head, expected):
    assert content_type.sniff(head) == expected


def test_heic_ftyp_brand_is_an_image_not_a_video():
    """HEIC and MP4 share the ISO-BMFF container; only the brand separates them."""
    assert content_type.sniff(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00") == "image"
    assert content_type.sniff(b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00") == "image"


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"\xff",
        b"not media at all",
        b"{\n  \"title\": \"x\"\n}",
        b"<!DOCTYPE html>",
        # A favicon is not a photograph. Deliberately unclassified so that a
        # saved web page's icon is never pulled into a photo library.
        b"\x00\x00\x01\x00\x07\x00\x30\x30",
    ],
)
def test_non_media_returns_none(head):
    assert content_type.sniff(head) is None


def test_sniff_never_raises_on_short_input():
    for n in range(0, 40):
        content_type.sniff(b"\x00" * n)


def test_canonical_extension():
    assert content_type.canonical_extension(b"\xff\xd8\xff\xe0\x00\x10JFIF") == ".jpg"
    assert content_type.canonical_extension(b"\x89PNG\r\n\x1a\n") == ".png"
    assert content_type.canonical_extension(b"\x00\x00\x01\xba\x44\x00") == ".mpeg"
    assert content_type.canonical_extension(b"nope") is None


def test_observed_takeout_bytes():
    """The exact prefixes measured in the real archive set (see the spec)."""
    # .screen and .tile members are plain JPEGs
    assert content_type.sniff(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01") == "image"
    # extensionless MVIMG_* Motion Photos are ISO-BMFF
    assert content_type.sniff(b"\x00\x00\x00\x1cftypmp42\x00\x00\x00\x00mp42") == "video"
    # .vob / .mod are MPEG program streams
    assert content_type.sniff(b"\x00\x00\x01\xba\x21\x00\x01\x00") == "video"
