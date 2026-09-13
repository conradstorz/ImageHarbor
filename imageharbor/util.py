"""Tiny stdlib-only helpers shared across the package.

A leaf module (no intra-package imports) so anything -- including
deliberately dependency-light modules like sidecar.py -- can import it
without creating a cycle. Consumers re-export under their old private names
(`from .util import now_iso as _now_iso`) so tests that monkeypatch
`catalog._now_iso` etc. keep working per-module.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def fsync_file(path: Path) -> None:
    """Flush *path*'s written bytes to stable storage.

    verify-after-copy reads back through the OS page cache, so without this
    the catalog could durably assert "copied, verified" for data that never
    reached the platter (power loss, not process crash). Directory-entry
    durability is deliberately out of scope: os.fsync on a directory fd is
    unsupported on Windows, and the file's content is the gap that matters.

    Opened ``"rb+"`` (read-write), not ``"rb"``: on Windows, ``os.fsync``
    (``_commit``) raises ``OSError: [Errno 9] Bad file descriptor`` against a
    handle opened read-only. ``"rb+"`` requires the file to already exist and
    never truncates it, so no bytes are at risk -- it only widens the handle
    enough for the flush to be valid on every platform this runs on.
    """
    with open(path, "rb+") as fh:
        os.fsync(fh.fileno())


def json_default(o: Any) -> Any:
    """Fallback for values ``json.dumps`` cannot serialize natively.

    Real EXIF carries raw ``bytes`` (ExifVersion, SceneType, MakerNote) and
    other exotic types. A bare ``default=str`` would not raise, but it writes
    Python repr syntax -- ``"b'0230'"`` rather than ``"0230"`` -- and both the
    catalog's JSON columns and the sidecar are meant to be portable,
    human-readable projections. Bytes become a lossy text form; anything else
    falls back to its string representation.
    """
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).decode("utf-8", "replace")
    return str(o)
