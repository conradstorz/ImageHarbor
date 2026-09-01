"""Preserve every non-media archive member verbatim, under the organized root.

The rule is deliberately uncurated: anything that is not an image or a video
is kept. Deciding which unknown file is worth keeping is exactly where "never
lose" degrades into "lose the thing nobody thought about" -- so Google's
169 KB HTML viewer is preserved for the same reason a Picasa face-tag file
would be, if one were present.

This is where data that cannot be attached to a photo lives -- a Picasa
face-tag file is the motivating case: it would carry no photo reference at
all, so it would be kept intact here so that a future export supplying a join
key could attach it retroactively. (No such file was present in the export
this module was verified against -- 175 archives scanned, none resembling
`contacts.xml` or any Picasa face-tag document; see
`imageharbor/faces/roster.py`. The uncurated-preservation rule above is
unaffected by that absence -- it doesn't depend on which specific unknown
files a given export happens to contain.)

Layout, keyed by ``archive_id`` (the SHA-256 of the zip's own bytes) so a
renamed or moved archive resolves to the same room::

    <organized_dir>/.takeout-provenance/<archive_id>/
        manifest.json
        albums/<folder>/Albums.json
        orphaned/<basename>
        <every other non-media member, at its member path>

``manifest.json`` records a digest per preserved document, keyed by member
path. A member whose digest already matches what the manifest has on record
is skipped entirely -- no read result is discarded, but nothing is written
either -- which is what makes re-preserving the same archive a no-op.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..hashing import compute_sha256_b64url_bytes
from . import archive

logger = logging.getLogger(__name__)

ROOM_NAME = ".takeout-provenance"

_ALBUMS_DIR = "albums"
_ORPHANED_DIR = "orphaned"
_MANIFEST_NAME = "manifest.json"

# Mirrors archive._safe_name: only genuinely filesystem-illegal characters are
# replaced. Applied per path COMPONENT (not the whole member path) so a
# preserved member keeps its real directory structure on disk.
_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def _safe_component(name: str) -> str:
    cleaned = _ILLEGAL_NAME_CHARS.sub("_", name).rstrip(" .")
    return cleaned or "member"


def _safe_relpath(member_path: str) -> Path:
    """A member path, sanitized component-by-component and traversal-safe.

    ``..``/``.`` components are dropped rather than sanitized-in-place: a zip
    is untrusted input, and a member path is never allowed to write outside
    the room it was handed.
    """
    parts = [p for p in member_path.split("/") if p not in ("", ".", "..")]
    if not parts:
        return Path("member")
    return Path(*[_safe_component(p) for p in parts])


def _room_dir(organized_dir: Path, archive_id: str) -> Path:
    return organized_dir / ROOM_NAME / archive_id


def manifest_path(organized_dir: Path, archive_id: str) -> Path:
    """Where *archive_id*'s manifest lives."""
    return _room_dir(organized_dir, archive_id) / _MANIFEST_NAME


def _stored_path(room: Path, member: "archive.MemberInfo", *, orphaned: bool) -> Path:
    basename = _safe_component(member.path.rpartition("/")[2])
    if orphaned:
        return room / _ORPHANED_DIR / basename
    if member.kind == archive.KIND_ALBUM:
        folder = member.path.rpartition("/")[0].rpartition("/")[2]
        folder_safe = _safe_component(folder) if folder else "_"
        return room / _ALBUMS_DIR / folder_safe / basename
    return room / _safe_relpath(member.path)


def _write_bytes(dest: Path, data: bytes) -> None:
    """Write *data* to *dest* atomically.

    The sole write call site in this module (member documents AND the
    manifest both funnel through here), so a test can monkeypatch this one
    function to simulate a write failure or count writes without touching
    every call site individually.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"documents": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("documents"), list):
            return data
    except Exception as exc:
        logger.warning(
            "Unreadable provenance manifest %s (%s); treating as empty", path, exc,
        )
    return {"documents": []}


def preserve(
    organized_dir: Path,
    identity: "archive.ArchiveIdentity",
    zf: zipfile.ZipFile,
    members: Iterable["archive.MemberInfo"],
    *,
    orphaned: Iterable[str] = (),
) -> int:
    """Write every non-media member of *members* verbatim into the provenance room.

    *orphaned* is the set of member paths -- computed by the caller, which
    alone holds the whole-batch pairing index -- that are media-JSON sidecars
    with no media member anywhere in the batch; those land under
    ``orphaned/`` instead of at their normal member path.

    Returns how many documents were newly written. A member whose digest
    already matches the manifest's recorded digest for that member path is
    skipped (no read of its bytes onto disk, and the manifest is left
    untouched if nothing else changed) -- that is what makes re-preserving
    the same archive a no-op. A failure reading or writing one member is
    logged and skipped; it never raises, and never fails the archive -- the
    zip itself is the original and remains intact regardless.
    """
    orphaned_set = set(orphaned)
    room = _room_dir(organized_dir, identity.archive_id)
    manifest_file = manifest_path(organized_dir, identity.archive_id)
    manifest = _load_manifest(manifest_file)
    documents_by_member: dict[str, dict] = {
        doc["member"]: doc for doc in manifest.get("documents", []) if "member" in doc
    }

    written = 0
    for member in members:
        if member.kind in (archive.KIND_IMAGE, archive.KIND_VIDEO):
            continue
        try:
            data = archive.read_member(zf, member.path)
            digest = compute_sha256_b64url_bytes(data)
        except Exception as exc:
            logger.warning(
                "Failed to read %s from %s for provenance; skipping: %s",
                member.path, identity.path.name, exc, exc_info=True,
            )
            continue

        prior = documents_by_member.get(member.path)
        if prior is not None and prior.get("digest") == digest:
            continue  # already preserved, byte-identical -- nothing to do

        is_orphaned = member.path in orphaned_set
        dest = _stored_path(room, member, orphaned=is_orphaned)
        try:
            _write_bytes(dest, data)
        except Exception as exc:
            logger.warning(
                "Failed to preserve %s from %s: %s",
                member.path, identity.path.name, exc, exc_info=True,
            )
            continue

        documents_by_member[member.path] = {
            "member": member.path,
            "digest": digest,
            "stored_as": dest.relative_to(room).as_posix(),
        }
        written += 1

    if written:
        manifest = {
            "archive": identity.path.name,
            "archive_id": identity.archive_id,
            "size": identity.size,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "documents": list(documents_by_member.values()),
        }
        try:
            _write_bytes(
                manifest_file,
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            )
        except Exception as exc:
            logger.warning(
                "Failed to write provenance manifest for %s: %s",
                identity.path.name, exc, exc_info=True,
            )

    return written
