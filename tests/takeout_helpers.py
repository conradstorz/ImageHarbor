"""Plain helper functions shared by Takeout test modules.

These are NOT pytest fixtures -- see `tests/conftest.py` for those. This
module exists so `tests/test_takeout_ingest.py` and
`tests/test_takeout_index_equivalence.py` can share the same synthetic-zip
builders without one test module importing from another (a real problem:
importing names from a sibling test module creates a hidden, order-sensitive
coupling between two files that are otherwise independent, and makes ruff's
unused-import check useless for the imported fixtures -- see the R1 Task 6
`# noqa: F401`/`F811` workarounds this module lets `test_takeout_index_
equivalence.py` remove).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

D = "Takeout/AlbumArchive/Hangouts/album"


def _jpeg(n: int) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes([n]) * 16 + b"\xff\xd9"


def _sidecar(title: str, seconds: int, people: tuple[str, ...] = ()) -> bytes:
    doc = {
        "title": title,
        "creationTime": {"timestampSeconds": str(seconds + 14836)},
        "photoTakenTime": {"timestampSeconds": str(seconds)},
        "geoData": {"latitude": 38.2768361, "longitude": -85.7357389},
    }
    if people:
        doc["people"] = [{"name": n} for n in people]
    return json.dumps(doc).encode()


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path
