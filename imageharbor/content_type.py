"""Identify media by content signature rather than by filename extension.

Pure and I/O-free: handed the first bytes of a file, returns a verdict. No
filesystem access, no imports from the rest of the package -- the same split
that makes ``sidecar_schema.py``, ``takeout/metadata.py``, and
``takeout/pairing.py`` exhaustively testable without fixtures on disk.

**The extension stays the first rung wherever this is used.** Only a file whose
extension is unrecognized should be read and sniffed, so the fast path stays
free and a recognized extension is never second-guessed. This module exists
because a real Google Takeout export delivers thousands of genuine photographs
under extensions like ``.screen``, ``.tile``, and ``.tmp``, which extension-only
classification files as documents.

``sniff`` is total: it never raises, whatever it is handed, including ``b""``.
An unrecognized signature is ``None`` -- "not media", never a guess.
"""

from __future__ import annotations

# Callers should read this many bytes from the head of a file. Twelve would be
# enough for every signature below (the ISO-BMFF brand ends at offset 12), but
# 32 leaves room to add signatures without changing every call site.
HEAD_BYTES = 32

# These strings match archive.KIND_IMAGE / archive.KIND_VIDEO by value so a
# sniffed verdict can be used wherever a classify() verdict is expected.
IMAGE = "image"
VIDEO = "video"

# (prefix, kind, canonical extension), longest/most specific first.
_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", IMAGE, ".jpg"),
    (b"\x89PNG\r\n\x1a\n", IMAGE, ".png"),
    (b"GIF87a", IMAGE, ".gif"),
    (b"GIF89a", IMAGE, ".gif"),
    (b"II*\x00", IMAGE, ".tif"),
    (b"MM\x00*", IMAGE, ".tif"),
    (b"BM", IMAGE, ".bmp"),
    (b"\x1aE\xdf\xa3", VIDEO, ".mkv"),
    (b"FLV\x01", VIDEO, ".flv"),
    (b"0&\xb2u\x8ef\xcf\x11", VIDEO, ".wmv"),
    # MPEG program stream / elementary stream. Covers .vob and .mod, which a
    # real export carries at multi-GB sizes.
    (b"\x00\x00\x01\xba", VIDEO, ".mpeg"),
    (b"\x00\x00\x01\xb3", VIDEO, ".mpeg"),
)

# ISO base media file format: the brand at offset 8 is the only thing
# separating a HEIC still from an MP4 video -- the container is identical.
_ISOBMFF_IMAGE_BRANDS = frozenset(
    {"heic", "heix", "heim", "heis", "hevc", "mif1", "msf1", "avif", "avis"}
)
_ISOBMFF_VIDEO_BRANDS = frozenset(
    {
        "isom", "iso2", "iso4", "iso5", "iso6", "mp41", "mp42", "mp71",
        "avc1", "3gp4", "3gp5", "3gp6", "3g2a", "qt  ", "m4v ", "m4a ",
        "mmp4", "dash", "f4v ",
    }
)

# A favicon is deliberately absent from the tables above. `.ico` appears in a
# real export as the icon of a saved web page; it is an image file but not a
# photograph, and classifying it as media would pull browser furniture into a
# photo library.


def _isobmff(head: bytes) -> str | None:
    """Return the kind for an ISO-BMFF header, or None if this isn't one."""
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    brand = head[8:12].decode("ascii", "replace").lower()
    if brand in _ISOBMFF_IMAGE_BRANDS:
        return IMAGE
    if brand in _ISOBMFF_VIDEO_BRANDS:
        return VIDEO
    # An unknown brand in a well-formed ISO-BMFF container is far more often a
    # video than a still. Guessing "video" here only ever affects reporting,
    # never placement.
    return VIDEO


def _riff(head: bytes) -> str | None:
    """RIFF containers hold both WebP stills and AVI video."""
    if len(head) < 12 or head[:4] != b"RIFF":
        return None
    form = head[8:12]
    if form == b"WEBP":
        return IMAGE
    if form == b"AVI ":
        return VIDEO
    return None


def _valid_ts_packet_header(head: bytes, offset: int) -> bool:
    """Is there a well-formed MPEG-TS packet header at *offset*?

    A TS packet header is 4 bytes: the sync byte ``0x47``, then byte 1
    (transport_error_indicator / payload_unit_start_indicator /
    transport_priority / PID high bits), byte 2 (PID low bits), byte 3
    (transport_scrambling_control / adaptation_field_control /
    continuity_counter). Matching the sync byte alone is a 1-in-256 false
    positive on arbitrary data, so this also requires
    transport_error_indicator == 0 and adaptation_field_control != 0 (0 is
    a reserved/invalid value) before calling it a real header.
    """
    if len(head) < offset + 4:
        return False
    if head[offset] != 0x47:
        return False
    transport_error_indicator = head[offset + 1] & 0x80
    if transport_error_indicator:
        return False
    adaptation_field_control = (head[offset + 3] >> 4) & 0x03
    if adaptation_field_control == 0:
        return False
    return True


def _mpegts(head: bytes) -> bool:
    """Is *head* an MPEG transport stream (plain .ts or AVCHD/M2TS)?

    AVCHD/M2TS files (Sony/Panasonic camcorders, and what a real Google
    Takeout export carries as ``.MTS``) prefix every TS packet with a 4-byte
    copyright/timecode field, so the ``0x47`` sync byte lands at offset 4,
    not offset 0. Do NOT simplify this to checking offset 0 only -- that
    silently stops recognizing every M2TS file again. A plain ``.ts``
    elementary stream has no such prefix, so offset 0 is checked too.
    """
    return _valid_ts_packet_header(head, 4) or _valid_ts_packet_header(head, 0)


def _match(head: bytes) -> tuple[str, str] | None:
    """Return (kind, canonical extension) for *head*, or None."""
    if not head:
        return None
    kind = _isobmff(head)
    if kind is not None:
        return (kind, ".heic" if kind is IMAGE else ".mp4")
    kind = _riff(head)
    if kind is not None:
        return (kind, ".webp" if kind is IMAGE else ".avi")
    for prefix, sig_kind, ext in _SIGNATURES:
        if head.startswith(prefix):
            return (sig_kind, ext)
    # Checked last, after every literal-prefix signature above: GIF87a/
    # GIF89a also start with the byte 0x47 (ASCII 'G'), so an offset-0 TS
    # sync-byte match must never be allowed to shadow the GIF entries.
    if _mpegts(head):
        return (VIDEO, ".mts")
    return None


def sniff(head: bytes) -> str | None:
    """Return ``IMAGE``, ``VIDEO``, or ``None`` for the head of a file.

    Total: never raises, whatever it is handed. ``None`` means "not recognized
    media", which is a verdict, not a failure.
    """
    matched = _match(head)
    return matched[0] if matched else None


def canonical_extension(head: bytes) -> str | None:
    """Return the extension *head*'s content deserves, or ``None``.

    Used to give an organized copy an openable name when the archive member's
    own extension was wrong (a ``.screen`` member holding JPEG bytes).
    """
    matched = _match(head)
    return matched[1] if matched else None
