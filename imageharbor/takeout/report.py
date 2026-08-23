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

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

_PART_NUMBER_RE = re.compile(r"^\d+$")

# Mirrors archive.KIND_IMAGE / KIND_VIDEO by value. Named here rather than
# imported so this module stays free of any dependency on the rest of the
# package -- it is pure and I/O-free by design.
_MEDIA_KINDS = frozenset({"image", "video"})


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
    # The sniffed content kind: "image", "video", or None for "not media".
    # ONLY a media kind may cover a sequence gap -- see _missing_parts.
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
    # The same members keyed by their SNIFFED kind ("image"/"video") instead.
    # The projection needs this split: recovered stills and recovered video do
    # not enter the same pipeline, so summing them into one number is what made
    # the old `organized_estimate` describe a pipeline that does not exist.
    misnamed_kind_counts: Counter = field(default_factory=Counter)
    misnamed_kind_bytes: Counter = field(default_factory=Counter)

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
    # Loose files beside the archives that could not be opened or stat'd -- a
    # mid-download file under a Windows byte-range lock is the expected cause.
    # Counted so the number is visible rather than silently absent.
    unreadable_loose_files: int = 0
    # Non-zip files in the directory that did NOT sniff as media (checksums.txt,
    # a survey.json written by a previous run). Visible, but never a "part".
    non_archive_files: int = 0


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


def _missing_parts(part_numbers: set[str], loose: list[LoosePart]) -> list[str] | None:
    """Part numbers absent from the zip sequence and not covered by a loose file.

    Returns ``None`` -- never ``[]`` -- when no part number could be parsed
    from any archive at all. ``[]`` means "gap detection ran and found no
    gap"; ``None`` means "gap detection never ran", and the two must not read
    the same. An unrecognized or renamed naming scheme reported as "missing
    parts none" is the refuse-to-guess rule inverted: it tells an operator
    their set is complete on the strength of having learned nothing about it.

    Only a loose part whose ``kind`` is media may cover a gap. Google delivers
    an oversized *media* file as its own raw part; ``transfer-log-002.txt`` is
    not a part, and letting it into the covered set erases a genuinely absent
    part 002.

    Gap detection is done on parsed *integers*, never on zero-padded strings.
    ``part_numbers`` is an unordered ``set[str]``, so picking a padding width
    from an arbitrary element (e.g. ``next(iter(part_numbers))``) is
    non-deterministic across process runs -- Python's hash randomization
    changes set iteration order for the same input data -- and whichever
    width happened to win could silently mismatch the width the raw,
    unpadded strings in ``covered`` actually used, producing a wrong
    missing-parts list (false positives or false negatives) instead of an
    error. Do not "simplify" this back to string padding: format only the
    *output* strings, using a width derived deterministically (the max width
    seen among valid part-number strings), and never use that padded form
    for membership comparisons.
    """
    if not part_numbers:
        return None

    def _to_int(s: str) -> int | None:
        return int(s) if _PART_NUMBER_RE.match(s) else None

    numeric = sorted(n for n in (_to_int(p) for p in part_numbers) if n is not None)
    if not numeric:
        return None

    covered_ints = set(numeric)
    for lp in loose:
        if lp.part is not None and lp.kind in _MEDIA_KINDS:
            n = _to_int(lp.part)
            if n is not None:
                covered_ints.add(n)

    width = max(len(p) for p in part_numbers if _PART_NUMBER_RE.match(p))
    return [
        str(n).zfill(width)
        for n in range(numeric[0], numeric[-1] + 1)
        if n not in covered_ints
    ]


def build_report(inv: SurveyInventory, *, distrust_threshold: int) -> dict[str, Any]:
    """Build the report document from a collected inventory."""
    total_members = sum(inv.kind_counts.values())
    recovered = sum(inv.misnamed_counts.values())
    described = inv.descriptor_human + inv.descriptor_machine

    images = inv.kind_counts.get("image", 0)
    videos = inv.kind_counts.get("video", 0)
    image_bytes = inv.kind_bytes.get("image", 0)
    video_bytes = inv.kind_bytes.get("video", 0)
    sniffed_media = sum(inv.misnamed_kind_counts.values())
    sniffed_media_bytes = sum(inv.misnamed_kind_bytes.values())

    distrusted = find_distrusted_timestamps(inv.timestamp_counts, distrust_threshold)
    distrusted_members = sum(inv.timestamp_counts[ts] for ts in distrusted)

    missing = _missing_parts(inv.part_numbers, inv.loose_parts)

    if not inv.archives:
        status = "empty"
    elif inv.unreadable_archives or inv.unreadable_loose_files:
        status = "degraded"
    else:
        status = "ok"

    return {
        "archives": {
            "status": status,
            "count": len(inv.archives),
            "unreadable": inv.unreadable_archives,
            # WHICH archive failed, not just how many. On a 175-part export a
            # bare count is not actionable -- the operator cannot go look.
            "unreadable_detail": [
                {"name": a.name, "error": a.error}
                for a in inv.archives
                if a.error is not None
            ],
            "unreadable_loose_files": inv.unreadable_loose_files,
            "bytes": sum(a.size for a in inv.archives),
            # None means gap detection never ran. It is NOT the same as [].
            "missing_parts": missing,
            "part_numbering": "unrecognized" if missing is None else "recognized",
            "loose_parts": len(inv.loose_parts),
            "non_archive_files": inv.non_archive_files,
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
                "excluded_unrecognized_extension": recovered,
                "note": (
                    "A tier-30 descriptor is permanent: tiers.is_upgrade forbids "
                    "an AI subject from displacing it. Reported, not judged. "
                    "excluded_unrecognized_extension media are NOT in this tally: "
                    "their extension was unrecognized, so they were never "
                    "classified as media to tier. They carry 19-digit machine "
                    "names that descriptor.is_camera_generated does not match "
                    "(its bare-digits pattern is 9-13 digits), so they are the "
                    "files most likely to be wrongly pinned at tier 30 once "
                    "content sniffing lands. Counted here rather than guessed at."
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
            # What TODAY's ingest actually organizes: recognized images only.
            "organized_today": images,
            "bytes_today": image_bytes,
            # What it would organize once video ingestion and content sniffing
            # land: recognized images + recognized video + sniffed media.
            "organized_after_deferred_fixes": images + videos + sniffed_media,
            "bytes_after_deferred_fixes": image_bytes + video_bytes + sniffed_media_bytes,
            # Every member's bytes, media or not (.json, .txt, .html included).
            # Named unambiguously: this is NOT a destination-space figure.
            "archive_total_bytes": sum(inv.kind_bytes.values()),
            "note": (
                "organized_today counts recognized images only, because that "
                "is all the current ingest copies: video is enumerated and "
                "recorded as deferred with no bytes copied, and misnamed media "
                "(extension says document, content says image) is filed to "
                ".takeout-provenance/ rather than organized. "
                "organized_after_deferred_fixes adds both, and is what to size "
                "for once video ingestion and content sniffing land. "
                "archive_total_bytes is every member's bytes including JSON, "
                "text and HTML -- it is not a destination-space budget."
            ),
            "by_year": dict(sorted(inv.year_counts.items())),
            "duplicates": {
                "exact": None,
                # The tight bound: N copies of one name are N-1 duplicates, not
                # N. Summing whole colliding groups overstates it by the number
                # of distinct names involved.
                "name_collision_upper_bound": (
                    inv.basename_collision_members - inv.basename_collisions
                ),
                "colliding_members": inv.basename_collision_members,
                "distinct_colliding_names": inv.basename_collisions,
                "note": (
                    "An upper bound from filename collisions only: for each "
                    "colliding name, every copy but one. The exact duplicate "
                    "count requires hashing every byte and is deliberately "
                    "not guessed."
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
    add(f"  status        {arc['status']}")
    add(f"  archives      {arc['count']:,}  ({_gib(arc['bytes'])})")
    add(f"  unreadable    {arc['unreadable']:,} archives, "
        f"{arc['unreadable_loose_files']:,} loose files")
    for bad in arc["unreadable_detail"][:10]:
        add(f"      {bad['name']}: {bad['error']}")
    if len(arc["unreadable_detail"]) > 10:
        add(f"      ... and {len(arc['unreadable_detail']) - 10:,} more")
    others = arc["non_archive_files"]
    add(f"  loose parts   {arc['loose_parts']:,}  "
        f"({others:,} other non-archive file{'' if others == 1 else 's'} ignored)")
    # None is not []: gap detection never ran, so "none" would be a claim the
    # survey cannot support.
    if arc["missing_parts"] is None:
        add("  missing parts not determined (part numbering unrecognized)")
    else:
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
    add(
        f"  media not tiered        {desc['excluded_unrecognized_extension']:,} "
        f"(unrecognized extension; excluded from the descriptor tally above)"
    )

    add("")
    add("PROJECTION")
    add(
        f"  organized today         {proj['organized_today']:,} files, "
        f"{_gib(proj['bytes_today'])}   (images only; video deferred, "
        f"misnamed media -> provenance)"
    )
    add(
        f"  after deferred fixes    {proj['organized_after_deferred_fixes']:,} files, "
        f"{_gib(proj['bytes_after_deferred_fixes'])}   (+ video, + sniffed media)"
    )
    add(
        f"  archive total           {_gib(proj['archive_total_bytes'])}   "
        f"(all members incl. JSON/text/HTML; not a destination budget)"
    )
    dupes = proj["duplicates"]
    add(
        f"  duplicates              upper bound {dupes['name_collision_upper_bound']:,} "
        f"across {dupes['distinct_colliding_names']:,} colliding names "
        f"(exact count not computed)"
    )
    if proj["by_year"]:
        years = list(proj["by_year"])
        add(f"  year range              {years[0]} .. {years[-1]}")

    return "\n".join(lines)
