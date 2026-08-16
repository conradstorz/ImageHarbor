"""Archive identity, enumeration, classification, and member extraction.

Archives are opened ``'r'`` only. Nothing here writes into, alongside, or in
place of an archive -- the zip IS the original, and originals are read-only.

Enumeration reads only the central directory, so surveying a 60 GB export
costs a seek, not a decompression pass. Extraction is per member, on demand,
into a staging directory the caller owns.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ..discovery import SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS
from ..hashing import compute_sha256_b64url

if TYPE_CHECKING:
    from ..catalog import Catalog

logger = logging.getLogger(__name__)

KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_METADATA = "metadata"
KIND_ALBUM = "album"
KIND_OTHER = "other"

# Album descriptors. AlbumArchive exports use Albums.json; Google Photos
# exports use metadata.json. Both are accepted.
_ALBUM_BASENAMES = frozenset({"albums.json", "metadata.json"})

# A path component named exactly "trash" (any case). "Trashy Album" is not a
# trash tree, so an endswith/contains test would be wrong here.
_TRASH_COMPONENT = "trash"

# Characters a Windows filesystem refuses. Member names in the real export
# carry Unicode and shell-hostile characters (● U+25CF, +, =, spaces,
# parentheses) -- all of which are legal on every supported filesystem and
# must survive untouched, because the staged file's NAME is evidence the date
# and descriptor resolvers read. Only genuinely illegal characters are
# replaced.
_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


@dataclass(frozen=True)
class MemberInfo:
    """One entry from an archive's central directory."""

    path: str    # member path inside the zip, verbatim
    size: int    # uncompressed size
    crc32: int   # stored for diagnostics; NEVER the sole basis for a skip
    kind: str


@dataclass(frozen=True)
class ArchiveIdentity:
    """Which archive this is, and where it was found."""

    archive_id: str   # SHA-256 b64url of the .zip's own bytes
    path: Path
    size: int
    mtime_ns: int


def classify(member_path: str) -> str:
    """Classify a member by extension and basename alone.

    Deliberately service-agnostic: the real export is ``Takeout/AlbumArchive/
    Hangouts/<album>/`` with no ``Google Photos/`` directory anywhere in it, so
    keying on a service path would classify nothing. ``discovery`` is the
    single source of truth for what counts as an image.
    """
    name = member_path.rpartition("/")[2]
    lower = name.lower()
    if lower.endswith(".json"):
        return KIND_ALBUM if lower in _ALBUM_BASENAMES else KIND_METADATA
    _, dot, ext = lower.rpartition(".")
    suffix = f".{ext}" if dot else ""
    if suffix in SUPPORTED_EXTENSIONS:
        return KIND_IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return KIND_VIDEO
    return KIND_OTHER


def is_trash(member_path: str) -> bool:
    """True if *member_path* lives under a Trash tree."""
    return any(
        part.lower() == _TRASH_COMPONENT for part in member_path.split("/")[:-1]
    )


def iter_members(zf: zipfile.ZipFile) -> Iterator[MemberInfo]:
    """Yield every file member of *zf*. Reads the central directory only."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        yield MemberInfo(
            path=info.filename,
            size=info.file_size,
            crc32=info.CRC,
            kind=classify(info.filename),
        )


def identify(path: Path, catalog: "Catalog") -> ArchiveIdentity:
    """Identify the archive at *path*, hashing it only when necessary.

    The ``(path, size, mtime_ns)`` fast path avoids re-hashing an archive we
    have already hashed at exactly that location and stat. It is never treated
    as proof of content: any change to any of the three falls through to the
    digest, which is what actually keys the archive.
    """
    stat = path.stat()
    row = catalog.takeout_archive_get_by_stat(str(path), stat.st_size, stat.st_mtime_ns)
    if row is not None:
        return ArchiveIdentity(
            archive_id=row["archive_id"],
            path=path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    logger.debug("Hashing archive %s (%d bytes)", path.name, stat.st_size)
    return ArchiveIdentity(
        archive_id=compute_sha256_b64url(path),
        path=path,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def read_member(zf: zipfile.ZipFile, member_path: str) -> bytes:
    """Return one member's bytes. Used for small JSON sidecars only."""
    with zf.open(member_path, "r") as fh:
        return fh.read()


def _safe_name(name: str) -> str:
    cleaned = _ILLEGAL_NAME_CHARS.sub("_", name).rstrip(" .")
    return cleaned or "member"


def extract_to(zf: zipfile.ZipFile, member: MemberInfo, staging_dir: Path) -> Path:
    """Stream *member* to a staging file and return its path.

    The staged file keeps the member's BASENAME, because that name is evidence:
    ``date_resolver``'s filename rung and ``descriptor``'s camera-pattern table
    both read it. Each member gets its own directory under *staging_dir*, so two
    members with the same basename in different archive directories cannot
    collide.

    ``zipfile`` verifies CRC32 on a full read, so a corrupted member raises here
    rather than yielding bad bytes into the library.

    The caller owns cleanup: pass the returned path to :func:`discard_staged`
    in a ``finally``.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    holder = Path(tempfile.mkdtemp(dir=str(staging_dir)))
    dest = holder / _safe_name(member.path.rpartition("/")[2])
    with zf.open(member.path, "r") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 65536)
    return dest


def discard_staged(staged: Path) -> None:
    """Remove a staged file and the private directory holding it.

    A leftover staging file after a kill is inert debris, not state: phase 2
    resumes from `takeout_members`, never from what is on the staging floor.
    """
    shutil.rmtree(staged.parent, ignore_errors=True)
