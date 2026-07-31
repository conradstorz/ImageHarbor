"""JSON sidecar writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_sidecar(organized_path: Path, metadata: dict[str, Any]) -> Path:
    """Write *metadata* as a JSON sidecar next to *organized_path*.

    The sidecar file has the same stem but a ``.json`` extension.
    Returns the path to the written sidecar.
    """
    sidecar_path = organized_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return sidecar_path
