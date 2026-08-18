"""Cumulative JSON sidecar files.

A sidecar accretes across runs rather than being rewritten: the facts pass
writes identity, sources, date, descriptor, and EXIF; the enrichment pass later
adds classification.  Unknown keys are preserved, so a hand-written correction
survives every subsequent run.

The catalog remains the source of truth; a sidecar is a portable projection of
it that travels with the image.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sidecar_schema import SCHEMA_VERSION as SIDECAR_SCHEMA_VERSION
from .sidecar_schema import merge as merge_documents

logger = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    """Fallback for values ``json.dumps`` cannot serialize natively.

    Real EXIF carries raw ``bytes`` (ExifVersion, SceneType, MakerNote) and
    other exotic types. A bare ``default=str`` would not raise, but it writes
    Python repr syntax into the file -- ``"b'0230'"`` rather than ``"0230"`` --
    and a sidecar is meant to be a portable, human-readable projection.

    This deliberately mirrors ``catalog._json_default`` rather than importing
    it, so this module stays dependency-free apart from the standard library.
    Keep the two in sync.
    """
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).decode("utf-8", "replace")
    return str(o)


def sidecar_path_for(organized_path: Path) -> Path:
    """Return the sidecar path for *organized_path*.

    Built explicitly from ``.stem`` + ``".json"`` rather than
    ``with_suffix(".json")``. The two are equivalent in current CPython
    (``.stem``/``with_suffix`` both operate on only the final suffix, so
    neither truncates a stem containing dots) -- this form is preferred
    simply because it states outright what's being built (the sidecar's
    name), rather than relying on a reader recalling ``with_suffix``'s
    exact single-suffix semantics.
    """
    return organized_path.with_name(f"{organized_path.stem}.json")


def _quarantine(path: Path, reason: str) -> None:
    """Move an unreadable sidecar aside instead of writing over it.

    Returning {} for a corrupt file -- the previous behavior -- meant the next
    merge silently replaced whatever those bytes held. Under the never-lose
    rule that is the one unacceptable outcome, so the bytes are preserved
    under a timestamped name and a fresh sidecar is built beside them.
    """
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        path.replace(target)
        logger.warning("Unreadable sidecar %s (%s); preserved as %s", path, reason, target.name)
    except OSError as exc:
        logger.error("Could not quarantine %s (%s); leaving it untouched", path, exc)


def read_sidecar(organized_path: Path) -> dict[str, Any]:
    """Return the existing sidecar contents, or ``{}`` if absent.

    An unreadable sidecar is quarantined (see :func:`_quarantine`) and reported
    as empty, so the caller proceeds with a fresh document while the original
    bytes survive on disk.
    """
    path = sidecar_path_for(organized_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        _quarantine(path, str(exc))
        return {}
    if not isinstance(data, dict):
        _quarantine(path, "top-level value is not an object")
        return {}
    return data


def merge_sidecar(organized_path: Path, updates: dict[str, Any]) -> Path:
    """Merge *updates* into the sidecar for *organized_path* and write it back.

    Merge policy lives in :mod:`imageharbor.sidecar_schema`; this function owns
    only reading, the atomic write (temp file then ``os.replace``), and the
    quarantine of an unreadable file.
    """
    path = sidecar_path_for(organized_path)
    observed_at = datetime.now(tz=timezone.utc).isoformat()
    merged = merge_documents(read_sidecar(organized_path), updates, observed_at=observed_at)

    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
