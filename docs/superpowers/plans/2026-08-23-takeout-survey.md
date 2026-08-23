# Takeout Survey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, read-only `imageharbor takeout survey` command that measures a Google Takeout archive set and reports what the current pipeline would do with it — without changing any pipeline behavior.

**Architecture:** Three new modules following the project's established pure/IO split. `content_type.py` is pure magic-byte sniffing (bytes in, verdict out). `takeout/report.py` is pure reporting logic (inventory in, document out). `takeout/survey.py` holds all the I/O. A Click verb wires them together. Everything reuses the existing `archive.classify` / `archive.iter_members` / `pairing.build_index` / `metadata.parse_photo_metadata` / `descriptor.is_camera_generated` APIs rather than reimplementing them.

**Tech Stack:** Python 3.11+, Click, stdlib `zipfile`, pytest. No new dependencies.

## Global Constraints

- **Read-only, absolutely.** Archives are opened with `zipfile.ZipFile(path)` (mode `'r'`, the default) and never written to. The command writes nothing to any library, catalog, or archive. The only file it may write is the optional `--json` output path.
- **Standalone.** No catalog, no destination, no watcher, no Docker, no AI backend, no network. `--archives` is the only required option.
- **Changes no pipeline behavior.** No existing module's behavior may change. `content_type.py` is new and called only by the survey.
- **Spec:** `docs/superpowers/specs/2026-08-23-takeout-survey-and-media-coverage-design.md`
- **Package data:** any new non-`.py` file must be declared in `[tool.setuptools.package-data]` — see commit `9d36c60` for why implicit inclusion is not reliable. This plan adds no such files.
- **Commands:** `uv sync --extra dev` to install; `uv run pytest` to test. Never `pip`/`venv` directly.
- **Do not chain shell commands with `&&`** — run them as separate calls.

---

### Task 1: `content_type.py` — pure magic-byte sniffing

**Files:**
- Create: `imageharbor/content_type.py`
- Test: `tests/test_content_type.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `HEAD_BYTES: int` — how many bytes callers should read (32)
  - `IMAGE: str` = `"image"`, `VIDEO: str` = `"video"` (deliberately equal to `archive.KIND_IMAGE` / `archive.KIND_VIDEO` string values)
  - `sniff(head: bytes) -> str | None`
  - `canonical_extension(head: bytes) -> str | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_content_type.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_content_type.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.content_type'`

- [ ] **Step 3: Write the implementation**

Create `imageharbor/content_type.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_content_type.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run the full suite to confirm nothing else moved**

Run: `uv run pytest`
Expected: PASS. This module is new and imported by nothing yet, so the count should rise and nothing should fail.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/content_type.py tests/test_content_type.py
git commit -m "feat: pure content-type sniffing by magic bytes"
```

---

### Task 2: `takeout/report.py` — pure reporting logic

**Files:**
- Create: `imageharbor/takeout/report.py`
- Test: `tests/test_takeout_report.py`

**Interfaces:**
- Consumes: `content_type.IMAGE`, `content_type.VIDEO` (string values only).
- Produces:
  - `@dataclass ArchiveFact(name: str, size: int, members: int, error: str | None = None)`
  - `@dataclass LoosePart(name: str, size: int, part: str | None, kind: str | None)`
  - `@dataclass SurveyInventory` — the mutable accumulator `survey.py` fills; field list below.
  - `find_distrusted_timestamps(counts: Mapping[str, int], threshold: int) -> frozenset[str]`
  - `build_report(inv: SurveyInventory, *, distrust_threshold: int) -> dict[str, Any]`
  - `format_summary(report: Mapping[str, Any]) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_report.py`:

```python
"""Report logic is pure: an inventory in, a document out. No filesystem."""

from collections import Counter

from imageharbor.takeout import report

# Fields that are Counters on SurveyInventory. The helper below wraps plain
# dicts so a test can pass {".screen": 3963} without losing .most_common().
_COUNTER_FIELDS = {
    "ext_counts", "ext_bytes", "kind_counts", "kind_bytes", "area_counts",
    "misnamed_counts", "misnamed_bytes", "timestamp_counts", "year_counts",
}


def _inv(**overrides):
    inv = report.SurveyInventory()
    for key, value in overrides.items():
        if key in _COUNTER_FIELDS and not isinstance(value, Counter):
            value = Counter(value)
        setattr(inv, key, value)
    return inv


# --- distrusted timestamp clusters ---------------------------------------

def test_cluster_at_threshold_is_distrusted():
    counts = {"1968-01-12T10:35:03": 25}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset({"1968-01-12T10:35:03"})


def test_cluster_below_threshold_is_not_distrusted():
    counts = {"1968-01-12T10:35:03": 24}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset()


def test_a_burst_of_shots_sharing_a_second_is_not_a_cluster():
    """Real bursts share a second; they do not share it 200 times."""
    counts = {"2019-07-04T12:33:11": 4, "2019-07-04T12:33:12": 6}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset()


def test_multiple_clusters_are_all_reported():
    counts = {"1968-01-12T10:35:03": 210, "2000-01-01T00:00:00": 40, "2019-01-01T09:00:00": 3}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset(
        {"1968-01-12T10:35:03", "2000-01-01T00:00:00"}
    )


def test_threshold_of_zero_does_not_distrust_everything():
    """A zero threshold disables the rule rather than condemning every date."""
    counts = {"2019-01-01T09:00:00": 3}
    assert report.find_distrusted_timestamps(counts, 0) == frozenset()


# --- refuse to guess ------------------------------------------------------

def test_duplicates_are_reported_as_an_upper_bound_never_as_a_count():
    inv = _inv(basename_collisions=9850, basename_collision_members=22402)
    doc = report.build_report(inv, distrust_threshold=25)
    dupes = doc["projection"]["duplicates"]
    assert dupes["name_collision_upper_bound"] == 22402
    assert dupes["exact"] is None
    assert "upper bound" in dupes["note"].lower()


def test_an_empty_archive_set_is_labelled_empty_not_reported_as_a_clean_result():
    """Zero members and "no archives found" must not read the same."""
    doc = report.build_report(_inv(), distrust_threshold=25)
    assert doc["archives"]["status"] == "empty"
    assert doc["projection"]["organized_estimate"] == 0


# --- sequence gaps --------------------------------------------------------

def test_a_gap_filled_by_a_loose_part_is_not_reported_as_missing():
    inv = _inv(
        part_numbers={"001", "002", "004"},
        loose_parts=[report.LoosePart(name="VID-003.mp4", size=10, part="003", kind="video")],
    )
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == []
    assert doc["archives"]["loose_parts"] == 1


def test_a_genuinely_missing_part_is_reported():
    inv = _inv(part_numbers={"001", "002", "004"}, loose_parts=[])
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == ["003"]


# --- misnamed media -------------------------------------------------------

def test_misnamed_media_is_surfaced_by_declared_extension():
    inv = _inv(misnamed_counts={".screen": 3963, ".tile": 26})
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["anomalies"]["misnamed_media"]["total"] == 3989
    assert doc["anomalies"]["misnamed_media"]["by_extension"][".screen"] == 3963


# --- descriptor distribution ---------------------------------------------

def test_descriptor_distribution_is_reported_without_judging_it():
    inv = _inv(descriptor_human=47460, descriptor_machine=31751)
    doc = report.build_report(inv, distrust_threshold=25)
    desc = doc["anomalies"]["descriptors"]
    assert desc["human_tier30"] == 47460
    assert desc["machine_tier0"] == 31751
    assert round(desc["human_share"], 2) == 0.6


def test_descriptor_share_is_zero_when_nothing_was_named():
    """Not a division by zero, and not a misleading 100%."""
    doc = report.build_report(_inv(), distrust_threshold=25)
    assert doc["anomalies"]["descriptors"]["human_share"] == 0.0


# --- summary --------------------------------------------------------------

def test_format_summary_is_text_and_mentions_the_headline_numbers():
    inv = _inv(kind_counts={"image": 5, "video": 2}, archives=[
        report.ArchiveFact(name="a.zip", size=10, members=7)
    ])
    text = report.format_summary(report.build_report(inv, distrust_threshold=25))
    assert isinstance(text, str)
    assert "image" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_takeout_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'report' from 'imageharbor.takeout'`

- [ ] **Step 3: Write the implementation**

Create `imageharbor/takeout/report.py`:

```python
"""Turn a collected archive inventory into a report document.

Pure and I/O-free, split from ``survey.py`` the same way ``projections.py`` is
split from ``stats.py`` and ``sidecar_schema.py`` from ``sidecar.py``: the
anomaly rules and the projection arithmetic are the logic most likely to be
wrong, so they live where they can be tested exhaustively without a filesystem.

**This module inherits the refuse-to-guess rule.** Where the survey cannot know
an answer, the report says so rather than producing a confident number. The
duplicate count is the load-bearing example: knowing it truly would require
hashing every byte of a 345 GiB archive set, so what is reported is a
name-collision *upper bound*, labelled as such. A confident wrong duplicate
figure would send an operator into an ingest with the wrong space budget --
exactly the failure ``dashboard/projections.py`` exists to avoid.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ArchiveFact:
    """One archive, as the survey found it."""

    name: str
    size: int
    members: int
    error: str | None = None


@dataclass(frozen=True)
class LoosePart:
    """A non-zip file sitting beside the archives.

    Google delivers a media file larger than the part size as its own raw
    part rather than wrapping it in a zip, so a "missing" part number is
    routinely not missing at all.
    """

    name: str
    size: int
    part: str | None
    kind: str | None


@dataclass
class SurveyInventory:
    """Everything ``survey.py`` counts. Mutable; filled during the walk."""

    archives: list[ArchiveFact] = field(default_factory=list)
    loose_parts: list[LoosePart] = field(default_factory=list)
    part_numbers: set[str] = field(default_factory=set)

    ext_counts: Counter = field(default_factory=Counter)
    ext_bytes: Counter = field(default_factory=Counter)
    kind_counts: Counter = field(default_factory=Counter)
    kind_bytes: Counter = field(default_factory=Counter)
    area_counts: Counter = field(default_factory=Counter)

    # Members whose extension said "not media" but whose bytes said otherwise,
    # keyed by the *declared* extension.
    misnamed_counts: Counter = field(default_factory=Counter)
    misnamed_bytes: Counter = field(default_factory=Counter)

    # Exact photoTakenTime values, ISO 8601 to the second.
    timestamp_counts: Counter = field(default_factory=Counter)
    year_counts: Counter = field(default_factory=Counter)
    # Sidecars that parsed but yielded no usable capture time. Note that
    # metadata.parse_photo_metadata already discards timestamps outside
    # 1826-2100, so "before photography or in the future" arrives here rather
    # than as an out-of-range year -- the survey cannot distinguish those two
    # cases through that API, and says so rather than inventing a breakdown.
    sidecars_without_timestamp: int = 0

    descriptor_human: int = 0
    descriptor_machine: int = 0

    media_without_sidecar: int = 0
    orphan_sidecars: int = 0

    basename_collisions: int = 0
    basename_collision_members: int = 0

    unreadable_archives: int = 0


def find_distrusted_timestamps(
    counts: Mapping[str, int], threshold: int
) -> frozenset[str]:
    """Return timestamps shared by at least *threshold* files.

    A timestamp repeated to the exact second across many files is evidence of a
    stopped clock, not of a capture moment. A burst of shots can legitimately
    share a second; two hundred files cannot.

    A *threshold* of 0 or less disables the rule and returns nothing -- it must
    never be read as "distrust every date".
    """
    if threshold <= 0:
        return frozenset()
    return frozenset(ts for ts, n in counts.items() if n >= threshold)


def _missing_parts(part_numbers: set[str], loose: list[LoosePart]) -> list[str]:
    """Part numbers absent from the zip sequence and not covered by a loose file.

    The comparison is done on **integers**, deliberately. Padding both sides to
    a width taken from an arbitrary set element is not merely untidy: with
    mixed-width part numbers the chosen width varies between process runs
    (string hash randomization reorders set iteration), and a padded candidate
    compared against unpadded originals silently reports parts as missing that
    are present. A confident wrong "your archive set is incomplete" is exactly
    the failure this module's refuse-to-guess rule exists to prevent. Do not
    simplify this back to string padding.
    """
    if not part_numbers:
        return []

    def _as_int(value: str | None) -> int | None:
        return int(value) if value and value.isdigit() else None

    known = {n for n in (_as_int(p) for p in part_numbers) if n is not None}
    if not known:
        return []
    covered = known | {n for n in (_as_int(lp.part) for lp in loose) if n is not None}
    width = max(len(p) for p in part_numbers if p.isdigit())
    return [
        str(n).zfill(width)
        for n in range(min(known), max(known) + 1)
        if n not in covered
    ]


def build_report(inv: SurveyInventory, *, distrust_threshold: int) -> dict[str, Any]:
    """Build the report document from a collected inventory."""
    total_members = sum(inv.kind_counts.values())
    media = inv.kind_counts.get("image", 0) + inv.kind_counts.get("video", 0)
    recovered = sum(inv.misnamed_counts.values())
    described = inv.descriptor_human + inv.descriptor_machine

    distrusted = find_distrusted_timestamps(inv.timestamp_counts, distrust_threshold)
    distrusted_members = sum(inv.timestamp_counts[ts] for ts in distrusted)

    return {
        "archives": {
            "status": "empty" if not inv.archives else "ok",
            "count": len(inv.archives),
            "unreadable": inv.unreadable_archives,
            "bytes": sum(a.size for a in inv.archives),
            "missing_parts": _missing_parts(inv.part_numbers, inv.loose_parts),
            "loose_parts": len(inv.loose_parts),
            "loose_part_detail": [
                {"name": lp.name, "size": lp.size, "part": lp.part, "kind": lp.kind}
                for lp in inv.loose_parts
            ],
        },
        "inventory": {
            "members": total_members,
            "bytes": sum(inv.kind_bytes.values()),
            "by_kind": dict(inv.kind_counts),
            "bytes_by_kind": dict(inv.kind_bytes),
            "by_extension": dict(inv.ext_counts.most_common()),
            "bytes_by_extension": dict(inv.ext_bytes),
            "by_area": dict(inv.area_counts.most_common()),
        },
        "anomalies": {
            "misnamed_media": {
                "total": recovered,
                "bytes": sum(inv.misnamed_bytes.values()),
                "by_extension": dict(inv.misnamed_counts.most_common()),
                "note": (
                    "Extension says not-media; content signature says otherwise. "
                    "These are filed to .takeout-provenance/ today."
                ),
            },
            "descriptors": {
                "human_tier30": inv.descriptor_human,
                "machine_tier0": inv.descriptor_machine,
                "human_share": round(inv.descriptor_human / described, 4) if described else 0.0,
                "note": (
                    "A tier-30 descriptor is permanent: tiers.is_upgrade forbids "
                    "an AI subject from displacing it. Reported, not judged."
                ),
            },
            "distrusted_date_clusters": {
                "threshold": distrust_threshold,
                "clusters": sorted(distrusted),
                "members": distrusted_members,
                "note": (
                    "One photoTakenTime shared to the second by this many files "
                    "is a stopped clock, not a capture moment."
                ),
            },
            "media_without_sidecar": inv.media_without_sidecar,
            "orphan_sidecars": inv.orphan_sidecars,
            "sidecars_without_timestamp": {
                "count": inv.sidecars_without_timestamp,
                "note": (
                    "No usable photoTakenTime. metadata.parse_photo_metadata "
                    "already discards timestamps outside 1826-2100, so an "
                    "implausible date lands here rather than as a bad year. "
                    "The two causes are not distinguished, and are not guessed at."
                ),
            },
        },
        "projection": {
            "organized_estimate": media + recovered,
            "destination_bytes": sum(inv.kind_bytes.values()),
            "by_year": dict(sorted(inv.year_counts.items())),
            "duplicates": {
                "exact": None,
                "name_collision_upper_bound": inv.basename_collision_members,
                "distinct_colliding_names": inv.basename_collisions,
                "note": (
                    "An upper bound from filename collisions only. The exact "
                    "duplicate count requires hashing every byte and is "
                    "deliberately not guessed."
                ),
            },
        },
    }


def _gib(n: int) -> str:
    return f"{n / 2**30:,.2f} GiB"


def format_summary(report: Mapping[str, Any]) -> str:
    """Render the report as readable text for a terminal."""
    arc, inv = report["archives"], report["inventory"]
    an, proj = report["anomalies"], report["projection"]
    lines: list[str] = []
    add = lines.append

    add("ARCHIVE SET")
    add(f"  archives      {arc['count']:,}  ({_gib(arc['bytes'])})")
    add(f"  unreadable    {arc['unreadable']:,}")
    add(f"  loose parts   {arc['loose_parts']:,}")
    add(f"  missing parts {', '.join(arc['missing_parts']) or 'none'}")

    add("")
    add(f"INVENTORY  {inv['members']:,} members, {_gib(inv['bytes'])}")
    for kind, count in sorted(inv["by_kind"].items(), key=lambda kv: -kv[1]):
        add(f"  {kind:10} {count:>9,}  {_gib(inv['bytes_by_kind'].get(kind, 0))}")

    add("")
    add("ANOMALIES")
    mis = an["misnamed_media"]
    add(f"  misnamed media          {mis['total']:,}  ({_gib(mis['bytes'])})")
    for ext, count in list(mis["by_extension"].items())[:8]:
        add(f"      {ext:12} {count:>7,}")
    desc = an["descriptors"]
    add(
        f"  tier-30 descriptors     {desc['human_tier30']:,} "
        f"({desc['human_share']:.0%} of named media, permanent)"
    )
    clusters = an["distrusted_date_clusters"]
    add(
        f"  distrusted date clusters {len(clusters['clusters']):,} "
        f"covering {clusters['members']:,} files"
    )
    for ts in clusters["clusters"][:5]:
        add(f"      {ts}")
    add(f"  media without sidecar   {an['media_without_sidecar']:,}")
    add(f"  orphan sidecars         {an['orphan_sidecars']:,}")

    add("")
    add("PROJECTION")
    add(f"  organized estimate      {proj['organized_estimate']:,}")
    add(f"  destination bytes       {_gib(proj['destination_bytes'])}")
    dupes = proj["duplicates"]
    add(
        f"  duplicates              upper bound {dupes['name_collision_upper_bound']:,} "
        f"(exact count not computed)"
    )
    if proj["by_year"]:
        years = list(proj["by_year"])
        add(f"  year range              {years[0]} .. {years[-1]}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_takeout_report.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/report.py tests/test_takeout_report.py
git commit -m "feat: pure report logic for the takeout survey"
```

---

### Task 3: `takeout/survey.py` — the I/O

**Files:**
- Create: `imageharbor/takeout/survey.py`
- Test: `tests/test_takeout_survey.py`

**Interfaces:**
- Consumes: `report.SurveyInventory`, `report.ArchiveFact`, `report.LoosePart`; `content_type.sniff` / `content_type.HEAD_BYTES`; `archive.iter_members` / `archive.classify` / `archive.KIND_*`; `pairing.build_index` / `pairing.sidecar_for`; `metadata.parse_photo_metadata`; `descriptor.is_camera_generated`.
- Produces: `survey_archives(archives_dir: Path) -> report.SurveyInventory`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_survey.py`:

```python
"""The survey reads archives and never writes to them."""

import hashlib
import json
import zipfile

import pytest

from imageharbor.takeout import survey

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64


def _sidecar(title, taken="2019-07-04T12:33:11Z"):
    return json.dumps(
        {"title": title, "photoTakenTime": {"formatted": taken, "timestamp": "1562243591"}}
    ).encode()


def _archive(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_counts_members_by_kind(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/Photos from 2019/a.jpg": JPEG,
        "Takeout/Google Photos/Photos from 2019/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        "Takeout/Google Photos/Photos from 2019/b.mp4": MP4,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["image"] == 1
    assert inv.kind_counts["video"] == 1
    assert inv.kind_counts["metadata"] == 1


def test_misnamed_media_is_found_by_sniffing(tmp_path):
    """A .screen member holding JPEG bytes is a photograph, not a document."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/5427880241588018962.screen": JPEG,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.misnamed_counts[".screen"] == 1
    assert inv.kind_counts["other"] == 1


def test_recognized_extension_is_never_second_guessed(tmp_path):
    """A .jpg is trusted on its extension; its bytes are not read."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": b"this is not actually a jpeg",
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["image"] == 1
    assert inv.misnamed_counts == {}


def test_loose_part_filling_a_sequence_gap_is_identified(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/Google Photos/a.jpg": JPEG})
    _archive(tmp_path / "takeout-20260818T012414Z-2-003.zip", {"Takeout/Google Photos/c.jpg": JPEG})
    (tmp_path / "VID_20160529_175415-002.mp4").write_bytes(MP4)
    inv = survey.survey_archives(tmp_path)
    assert [lp.part for lp in inv.loose_parts] == ["002"]
    assert inv.loose_parts[0].kind == "video"


def test_timestamps_are_collected_for_clustering(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg", "1968-01-12T10:35:03Z"),
        "Takeout/Google Photos/b.jpg": JPEG,
        "Takeout/Google Photos/b.jpg.supplemental-metadata.json": _sidecar("b.jpg", "1968-01-12T10:35:03Z"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.timestamp_counts["1968-01-12T10:35:03"] == 2
    assert inv.year_counts["1968"] == 2


def test_descriptor_tiers_are_counted(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/IMG_1234.jpg": JPEG,          # camera-generated
        "Takeout/Google Photos/Scouts and Halloween 002.jpg": JPEG,  # human
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.descriptor_machine == 1
    assert inv.descriptor_human == 1


def test_media_without_a_sidecar_is_counted(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        "Takeout/Google Photos/lonely.jpg": JPEG,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.media_without_sidecar == 1


def test_orphan_sidecars_are_counted_across_the_whole_batch(tmp_path):
    """A sidecar is orphaned only if NO archive holds its media."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        "Takeout/Google Photos/ghost.jpg.supplemental-metadata.json": _sidecar("ghost.jpg"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.orphan_sidecars == 1


def test_a_sidecar_pairing_across_two_archives_is_not_an_orphan(tmp_path):
    """Google splits parts by size, so media and sidecar routinely separate."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
    })
    _archive(tmp_path / "takeout-20260818T012414Z-2-002.zip", {
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.orphan_sidecars == 0
    assert inv.media_without_sidecar == 0


def test_a_sidecar_with_no_usable_timestamp_is_counted(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": b'{"title": "a.jpg"}',
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.sidecars_without_timestamp == 1


def test_an_unreadable_archive_is_recorded_not_raised(tmp_path):
    (tmp_path / "takeout-20260818T012414Z-2-001.zip").write_bytes(b"not a zip at all")
    inv = survey.survey_archives(tmp_path)
    assert inv.unreadable_archives == 1
    assert inv.archives[0].error is not None


def test_survey_is_read_only(tmp_path):
    """The property that makes it safe to run against a live archive set."""
    paths = [
        _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
            "Takeout/Google Photos/a.jpg": JPEG,
            "Takeout/Google Photos/x.screen": JPEG,
            "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        }),
    ]
    loose = tmp_path / "VID_20160529_175415-002.mp4"
    loose.write_bytes(MP4)
    paths.append(loose)

    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    survey.survey_archives(tmp_path)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after


def test_empty_directory_yields_an_empty_inventory(tmp_path):
    inv = survey.survey_archives(tmp_path)
    assert inv.archives == []
    assert sum(inv.kind_counts.values()) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_takeout_survey.py -v`
Expected: FAIL — `ImportError: cannot import name 'survey' from 'imageharbor.takeout'`

- [ ] **Step 3: Write the implementation**

Create `imageharbor/takeout/survey.py`:

```python
"""Measure a Google Takeout archive set without changing anything.

Read-only by construction: archives are opened in mode ``'r'`` and nothing is
written to them, to a library, or to a catalog. That property is pinned by
``tests/test_takeout_survey.py::test_survey_is_read_only`` and is what makes the
command safe to run against a live archive set -- including one another process
is still downloading or verifying.

Two passes over the archives. The first reads central directories only and
builds the whole-batch pairing index; the second reopens each archive to sniff
members whose extension is unrecognized and to read the per-media JSON sidecars.
The index has to be global before any pairing question can be answered, because
Google's multi-part zips split by size across the file list -- a photo and its
``.json`` routinely land in different parts.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from pathlib import Path

from .. import content_type
from ..descriptor import is_camera_generated
from . import archive, metadata, pairing
from .report import ArchiveFact, LoosePart, SurveyInventory

logger = logging.getLogger(__name__)

# takeout-20260818T012414Z-2-001.zip
_ZIP_PART_RE = re.compile(
    r"^takeout-\d{8}T\d{6}Z-\d+-(?P<part>\d+)\.zip$", re.IGNORECASE
)
# A raw part carries its part number as a trailing "-NNN" before the extension:
# VID_20160529_175415-162.mp4
_LOOSE_PART_RE = re.compile(r"-(?P<part>\d{2,4})\.[A-Za-z0-9]+$")

_MEDIA_KINDS = (archive.KIND_IMAGE, archive.KIND_VIDEO)


def _extension(member_path: str) -> str:
    name = member_path.rpartition("/")[2]
    stem, dot, ext = name.rpartition(".")
    return f".{ext.lower()}" if dot and stem else ""


def _area(member_path: str) -> str:
    parts = member_path.split("/")
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def _stem(member_path: str) -> str:
    name = member_path.rpartition("/")[2]
    stem, dot, _ = name.rpartition(".")
    return stem if dot and stem else name


def survey_archives(archives_dir: Path) -> SurveyInventory:
    """Survey every archive and loose part in *archives_dir*."""
    inv = SurveyInventory()

    zips: list[Path] = []
    loose_files: list[Path] = []
    for entry in sorted(archives_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() == ".zip":
            zips.append(entry)
        else:
            loose_files.append(entry)

    # --- pass one: central directories only ------------------------------
    members_by_zip: dict[Path, list[archive.MemberInfo]] = {}
    for path in zips:
        match = _ZIP_PART_RE.match(path.name)
        if match:
            inv.part_numbers.add(match.group("part"))
        try:
            with zipfile.ZipFile(path) as zf:
                members = list(archive.iter_members(zf))
        except (zipfile.BadZipFile, OSError) as exc:
            inv.unreadable_archives += 1
            inv.archives.append(
                ArchiveFact(name=path.name, size=path.stat().st_size, members=0, error=str(exc))
            )
            logger.warning("survey: cannot read %s: %s", path.name, exc)
            continue
        members_by_zip[path] = members
        inv.archives.append(
            ArchiveFact(name=path.name, size=path.stat().st_size, members=len(members))
        )

    all_paths = [m.path for members in members_by_zip.values() for m in members]
    index = pairing.build_index(all_paths)

    basenames: Counter = Counter()
    # Every metadata member seen, and every one that some media member actually
    # paired to. The difference is the orphan set: a sidecar whose media is
    # nowhere in the batch. This has to be a whole-batch question -- a sidecar
    # is only orphaned if NO archive holds its media.
    metadata_paths: set[str] = set()
    paired_sidecars: set[str] = set()

    # --- pass two: sniff unknowns, read sidecars --------------------------
    for path, members in members_by_zip.items():
        with zipfile.ZipFile(path) as zf:
            for member in members:
                _record_member(
                    inv, zf, member, index, basenames, metadata_paths, paired_sidecars
                )

    inv.orphan_sidecars = len(metadata_paths - paired_sidecars)

    # --- loose parts ------------------------------------------------------
    for path in loose_files:
        match = _LOOSE_PART_RE.search(path.name)
        with path.open("rb") as handle:
            head = handle.read(content_type.HEAD_BYTES)
        inv.loose_parts.append(
            LoosePart(
                name=path.name,
                size=path.stat().st_size,
                part=match.group("part") if match else None,
                kind=content_type.sniff(head),
            )
        )

    inv.basename_collisions = sum(1 for n in basenames.values() if n > 1)
    inv.basename_collision_members = sum(n for n in basenames.values() if n > 1)
    return inv


def _record_member(
    inv: SurveyInventory,
    zf: zipfile.ZipFile,
    member: archive.MemberInfo,
    index: pairing.PairingIndex,
    basenames: Counter,
    metadata_paths: set[str],
    paired_sidecars: set[str],
) -> None:
    """Fold one member into the inventory."""
    ext = _extension(member.path)
    kind = member.kind

    inv.ext_counts[ext] += 1
    inv.ext_bytes[ext] += member.size
    inv.area_counts[_area(member.path)] += 1

    # An unrecognized extension is the only case worth reading bytes for. A
    # recognized one is never second-guessed -- that keeps the fast path free
    # and guarantees no existing classification changes.
    if kind == archive.KIND_OTHER:
        try:
            with zf.open(member.path) as handle:
                head = handle.read(content_type.HEAD_BYTES)
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            logger.debug("survey: cannot sniff %s: %s", member.path, exc)
            head = b""
        sniffed = content_type.sniff(head)
        if sniffed is not None:
            inv.misnamed_counts[ext] += 1
            inv.misnamed_bytes[ext] += member.size

    inv.kind_counts[kind] += 1
    inv.kind_bytes[kind] += member.size

    if kind in _MEDIA_KINDS:
        basenames[member.path.rpartition("/")[2]] += 1
        if is_camera_generated(_stem(member.path)):
            inv.descriptor_machine += 1
        else:
            inv.descriptor_human += 1
        sidecar = pairing.sidecar_for(member.path, index)
        if sidecar is None:
            inv.media_without_sidecar += 1
        else:
            paired_sidecars.add(sidecar)

    if kind == archive.KIND_METADATA:
        metadata_paths.add(member.path)
        try:
            raw = zf.read(member.path)
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            logger.debug("survey: cannot read sidecar %s: %s", member.path, exc)
            inv.sidecars_without_timestamp += 1
            return
        parsed = metadata.parse_photo_metadata(raw)
        if parsed.photo_taken_at is not None:
            inv.timestamp_counts[parsed.photo_taken_at.isoformat(timespec="seconds")] += 1
            inv.year_counts[str(parsed.photo_taken_at.year)] += 1
        else:
            inv.sidecars_without_timestamp += 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_takeout_survey.py -v`
Expected: PASS, all tests.

If `test_media_without_a_sidecar_is_counted` fails, check `pairing.sidecar_for`'s expectations against `tests/test_takeout_pairing.py` — the sidecar naming convention must match what that suite already pins. Adjust the fixture's member names to a convention that suite proves works; do not change `pairing.py`.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/survey.py tests/test_takeout_survey.py
git commit -m "feat: read-only archive survey over central directories and sidecars"
```

---

### Task 4: The `takeout survey` CLI verb

**Files:**
- Modify: `imageharbor/cli.py` — add a new command to the existing `takeout_cmd` group (the group is defined just above `@takeout_cmd.command(name="ingest")` at cli.py:666)
- Test: `tests/test_cli.py` — append

**Interfaces:**
- Consumes: `survey.survey_archives`, `report.build_report`, `report.format_summary`.
- Produces: the `imageharbor takeout survey` command.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`. Note this file already imports `json`, `zipfile`,
`Path`, `CliRunner`, and **`main`** (not `cli`) at module level — the Click group
is named `main` and registered via `main.add_command(takeout_cmd, name="takeout")`
at cli.py:794. Reuse those imports rather than adding new ones.

```python
_SURVEY_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01" + b"\x00" * 64


def test_takeout_survey_runs_standalone_and_prints_a_summary(tmp_path):
    """No catalog, no dest, no network -- an archive directory is enough."""
    archives = tmp_path / "archives"
    archives.mkdir()
    with zipfile.ZipFile(archives / "takeout-20260818T012414Z-2-001.zip", "w") as zf:
        zf.writestr("Takeout/Google Photos/Photos from 2019/a.jpg", _SURVEY_JPEG)
        zf.writestr("Takeout/Google Photos/Photos from 2019/x.screen", _SURVEY_JPEG)

    out_json = tmp_path / "survey.json"
    result = CliRunner().invoke(
        main, ["takeout", "survey", "--archives", str(archives), "--json", str(out_json)]
    )

    assert result.exit_code == 0, result.output
    assert "INVENTORY" in result.output
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["inventory"]["by_kind"]["image"] == 1
    assert doc["anomalies"]["misnamed_media"]["total"] == 1


def test_takeout_survey_requires_an_existing_archives_dir(tmp_path):
    result = CliRunner().invoke(
        main, ["takeout", "survey", "--archives", str(tmp_path / "nope")]
    )
    assert result.exit_code != 0


def test_takeout_survey_writes_nothing_when_json_is_omitted(tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    with zipfile.ZipFile(archives / "takeout-20260818T012414Z-2-001.zip", "w") as zf:
        zf.writestr("Takeout/Google Photos/a.jpg", _SURVEY_JPEG)

    before = sorted(p.name for p in archives.iterdir())
    result = CliRunner().invoke(main, ["takeout", "survey", "--archives", str(archives)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in archives.iterdir()) == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k takeout_survey -v`
Expected: FAIL — `Error: No such command 'survey'.`

- [ ] **Step 3: Add the command**

In `imageharbor/cli.py`, insert this immediately **before** the `@takeout_cmd.command(name="ingest")` decorator at line 666:

```python
@takeout_cmd.command(name="survey")
@click.option(
    "--archives",
    "archives_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory holding Google Takeout archives (read-only).",
)
@click.option(
    "--json",
    "json_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the machine-readable report document here.",
)
@click.option(
    "--distrust-threshold",
    default=25,
    show_default=True,
    type=click.IntRange(min=0),
    help=(
        "How many files must share one photoTakenTime, to the second, before "
        "that timestamp is reported as a stopped clock. 0 disables the check."
    ),
)
def takeout_survey(archives_dir: Path, json_path: Path | None, distrust_threshold: int) -> None:
    """Measure an archive set and report what ingestion would do with it.

    Read-only and standalone: no catalog, no destination, no AI backend, no
    network. Nothing is written except the optional --json document, so this is
    safe to run against archives another process is still downloading.
    """
    import json as _json

    from .takeout import report as takeout_report
    from .takeout import survey as takeout_survey_mod

    inventory = takeout_survey_mod.survey_archives(archives_dir)
    document = takeout_report.build_report(inventory, distrust_threshold=distrust_threshold)

    click.echo(takeout_report.format_summary(document))

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(_json.dumps(document, indent=2), encoding="utf-8")
        click.echo(f"\nReport written to {json_path}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k takeout_survey -v`
Expected: PASS, all three tests.

- [ ] **Step 5: Verify the command is discoverable**

Run: `uv run imageharbor takeout survey --help`
Expected: usage text listing `--archives`, `--json`, `--distrust-threshold`.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: PASS. No existing test may change behavior — this task adds a command and touches nothing else.

- [ ] **Step 7: Commit**

```bash
git add imageharbor/cli.py tests/test_cli.py
git commit -m "feat: add the standalone takeout survey command"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md` — the commands table, and the `takeout/` module description
- Modify: `README.md` — command list

**Interfaces:**
- Consumes: the finished command.
- Produces: nothing code-level.

- [ ] **Step 1: Add the command to CLAUDE.md's commands table**

In `CLAUDE.md`, in the `## Commands` table, add this row immediately after the `takeout status` row:

```markdown
| Survey an archive set before ingesting (read-only, standalone) | `uv run imageharbor takeout survey --archives DIR --json report.json` |
```

- [ ] **Step 2: Document the new modules in CLAUDE.md's Architecture section**

In `CLAUDE.md`, inside the **`takeout/`** bullet, change the sentence listing the modules from "Five modules" to "Seven modules" and append to that list:

```markdown
  `survey.py` (read-only measurement of an archive set -- two passes: central
  directories to build the whole-batch pairing index, then a reopen to sniff
  members whose extension is unrecognized and read per-media sidecars), and
  `report.py` (pure: turns a collected inventory into the report document,
  split from `survey.py` for the same reason `projections.py` is split from
  `stats.py`).
```

- [ ] **Step 3: Document `content_type.py` in CLAUDE.md's Architecture section**

In `CLAUDE.md`, add this bullet immediately after the **`discovery.py`** bullet:

```markdown
- **`content_type.py`** -- pure, I/O-free identification of media by magic
  bytes: `sniff(head) -> "image" | "video" | None` and
  `canonical_extension(head)`. **The extension stays the first rung**; only a
  file whose extension is unrecognized is read and sniffed, so no existing
  classification changes. It exists because a real Google Takeout export
  delivers thousands of genuine photographs under extensions like `.screen`,
  `.tile`, and `.tmp`. Currently called only by `takeout/survey.py` -- the
  pipeline does not consult it yet.
```

- [ ] **Step 4: Add the command to README.md**

In `README.md`, find the list of commands and add:

```markdown
- `imageharbor takeout survey --archives DIR` — measure an archive set and
  report what ingestion would do with it. Read-only and standalone: no catalog,
  no destination, no AI backend, no network.
```

- [ ] **Step 5: Verify the docs match reality**

Run: `uv run imageharbor takeout survey --help`
Confirm the flags in the docs match the flags the command actually exposes.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document the takeout survey command and content_type"
```

---

## Verification

After all five tasks, before reporting completion:

- [ ] Run the full suite: `uv run pytest` — all tests pass.
- [ ] Confirm the read-only property test is present and passing: `uv run pytest tests/test_takeout_survey.py::test_survey_is_read_only -v`
- [ ] Run the command against the real archive set and capture the output:

```bash
uv run imageharbor takeout survey --archives "D:/Users/Conrad/Documents/programming/Google_Takeout_Downloader/takeout" --json survey-report.json
```

Expected, from the measurements in the spec: 175 archives, 6 loose parts, no missing parts, ~150,024 members, ~4,008 misnamed media, a distrusted cluster at `1968-01-12T10:35:03` covering ~210 files.

**If the numbers disagree with the spec, the survey is wrong — investigate before reporting success.** The spec's figures were measured directly from this archive set.

Note that a sibling process may be deep-verifying these archives; the survey is read-only, so this is safe, but expect slower I/O while it runs.
