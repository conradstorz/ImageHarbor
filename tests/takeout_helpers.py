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
import sqlite3
import zipfile
from pathlib import Path

D = "Takeout/AlbumArchive/Hangouts/album"

# Schema for a synthetic Takeout_Inventory pairing-index database, used by
# make_index() below. Moved here from tests/test_takeout_index_reader.py
# (R4 review Minor #9) so tests/test_takeout_index_equivalence.py doesn't
# import it from a sibling test module -- see the module docstring above.
_INDEX_SCHEMA = """
CREATE TABLE sidecar (
  id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, name TEXT NOT NULL,
  title TEXT, taken_at TEXT, lat REAL, lon REAL, device TEXT,
  trashed INTEGER, archived INTEGER, from_partner INTEGER,
  parse_error TEXT, role TEXT);
CREATE TABLE media (
  id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, area TEXT NOT NULL,
  folder TEXT NOT NULL, name TEXT NOT NULL, ext TEXT, size INTEGER,
  actual_type TEXT, sidecar_id INTEGER REFERENCES sidecar(id),
  rule TEXT NOT NULL, confidence TEXT NOT NULL);
CREATE TABLE archive (
  name TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
  members INTEGER NOT NULL, error TEXT);
CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def make_index(path, *, version="1", archives=(("part-1.zip", 100, 5, 2, None),),
               media=(), sidecars=()):
    con = sqlite3.connect(path)
    con.executescript(_INDEX_SCHEMA)
    for row in archives:
        con.execute("INSERT INTO archive VALUES (?,?,?,?,?)", row)
    for row in sidecars:
        con.execute("INSERT INTO sidecar (id, archive, path, name)"
                    " VALUES (?,?,?,?)", row)
    for row in media:
        con.execute("INSERT INTO media (archive, path, area, folder, name,"
                    " sidecar_id, rule, confidence) VALUES (?,?,?,?,?,?,?,?)", row)
    if version is not None:
        con.execute("INSERT INTO index_meta VALUES ('schema_version', ?)", (version,))
    con.commit()
    con.close()
    return path


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
