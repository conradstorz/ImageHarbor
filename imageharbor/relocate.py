"""Target-path computation and safe relocation within the organized tree.

Organized copies belong to ImageHarbor, so moving and renaming them is legal --
unlike originals, which are never touched.  A relocation is applied to the
filesystem *before* the catalog is updated; a crash in between is recoverable
because the file is content-addressed, so `resolve_organized_path` can find it
by digest and repair the row.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .date_resolver import ResolvedDate
from .discovery import SUPPORTED_EXTENSIONS
from .filename import build_filename
from .hashing import extract_digest_from_stem, verify_file

logger = logging.getLogger(__name__)


def target_path(
    organized_dir: Path,
    date: ResolvedDate,
    descriptor: str,
    sha256_b64url: str,
    extension: str,
) -> Path:
    """Where a file with these facts belongs, under *organized_dir*."""
    name = build_filename(date.date_str, descriptor, sha256_b64url, extension)
    return organized_dir / date.folder / name


def apply_relocation(old_path: Path, new_path: Path) -> None:
    """Move *old_path* to *new_path*.

    A destination that already holds byte-identical content is accepted (this
    is the state left by a crash between the rename and the catalog update);
    a destination holding *different* content raises rather than clobbering.
    """
    if old_path == new_path:
        return

    new_path.parent.mkdir(parents=True, exist_ok=True)

    if new_path.exists():
        digest = extract_digest_from_stem(new_path.stem)
        if digest and verify_file(new_path, digest):
            logger.debug("Destination already present and verified: %s", new_path)
            old_path.unlink(missing_ok=True)
            return
        raise FileExistsError(
            f"Refusing to overwrite {new_path}: it holds different content"
        )

    os.replace(old_path, new_path)
    logger.info("Relocated %s -> %s", old_path.name, new_path)


def find_by_digest(organized_dir: Path, sha256_b64url: str) -> Path | None:
    """Locate an IMAGE anywhere under *organized_dir* by its embedded digest.

    This is the self-healing path: content addressing means a file that moved
    is never actually lost.

    Restricted to supported image extensions deliberately. A JSON sidecar's
    name is its image's stem with ``.json`` substituted, so the sidecar's stem
    carries the SAME digest. Without this filter, an image that moved while its
    sidecar stayed behind yields two glob matches, and a stale sidecar can be
    returned as though it were the organized copy -- after which the catalog
    records a ``.json`` as the image's path. Results are sorted so the choice is
    deterministic rather than filesystem-order dependent.
    """
    for candidate in sorted(organized_dir.rglob(f"*{sha256_b64url}.*")):
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if candidate.is_file() and extract_digest_from_stem(candidate.stem) == sha256_b64url:
            return candidate
    return None


def resolve_organized_path(
    organized_dir: Path, recorded: Path, sha256_b64url: str
) -> Path | None:
    """Return where the file actually is, repairing a stale recorded path.

    Returns None if the file is genuinely missing from the organized tree.
    """
    if recorded.exists():
        return recorded
    found = find_by_digest(organized_dir, sha256_b64url)
    if found is not None:
        logger.info(
            "Catalog path was stale (%s); found by digest at %s", recorded, found
        )
    return found
