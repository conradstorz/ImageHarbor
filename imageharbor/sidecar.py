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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION = 1


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


def read_sidecar(organized_path: Path) -> dict[str, Any]:
    """Return the existing sidecar contents, or ``{}`` if absent or unreadable.

    A corrupt sidecar is reported and treated as empty rather than raising: it
    must never block an image that is already copied, verified, and cataloged.
    """
    path = sidecar_path_for(organized_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable sidecar %s (%s); treating as empty", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base*, returning a new dict.

    Nested dicts merge key-by-key so a partial update never drops a sibling
    field.  Lists and scalars replace wholesale -- callers that own a list
    (``sources``, ``history``) pass the complete value.
    """
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def merge_sidecar(organized_path: Path, updates: dict[str, Any]) -> Path:
    """Merge *updates* into the sidecar for *organized_path* and write it back.

    The write is atomic (temp file in the same directory, then ``os.replace``)
    so an interrupted run cannot leave a half-written sidecar.
    """
    path = sidecar_path_for(organized_path)
    merged = _deep_merge(read_sidecar(organized_path), updates)
    merged["schema_version"] = SIDECAR_SCHEMA_VERSION

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
