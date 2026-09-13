"""Tiny stdlib-only helpers shared across the package.

A leaf module (no intra-package imports) so anything -- including
deliberately dependency-light modules like sidecar.py -- can import it
without creating a cycle. Consumers re-export under their old private names
(`from .util import now_iso as _now_iso`) so tests that monkeypatch
`catalog._now_iso` etc. keep working per-module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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
