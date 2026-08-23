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
