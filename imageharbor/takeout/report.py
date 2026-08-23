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
        return []

    def _to_int(s: str) -> int | None:
        return int(s) if _PART_NUMBER_RE.match(s) else None

    numeric = sorted(n for n in (_to_int(p) for p in part_numbers) if n is not None)
    if not numeric:
        return []

    covered_ints = set(numeric)
    for lp in loose:
        if lp.part is not None:
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
