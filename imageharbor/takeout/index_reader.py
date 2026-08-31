"""Read a pairing index published by Takeout_Inventory.

The index answers what ImageHarbor's own `pairing.py` cannot: not merely which
sidecar describes a media file, but how far to trust it. An `-edited` copy
inherits its ORIGINAL's sidecar, whose title and location belong to a
different file.

Optional by design. Every failure here falls back to `pairing.py`, which is
always a correct answer -- so this module raises rather than repairs, and the
caller decides whether a given failure is fatal.

Loaded eagerly into memory: an index for a 388 GB export holds ~79k media rows
of short strings, which is a few tens of MB and removes any per-member query
from the ingest loop.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_MEDIA_COLUMNS = frozenset({
    "archive", "path", "sidecar_id", "rule", "confidence"})
_SIDECAR_COLUMNS = frozenset({"id", "path"})
_ARCHIVE_COLUMNS = frozenset({"name", "size", "mtime"})


class IndexUnusable(Exception):
    """The index is absent, unreadable, or of a shape this code does not know."""


@dataclass(frozen=True)
class IndexedPairing:
    sidecar: str | None
    confidence: str
    rule: str


@dataclass(frozen=True)
class IndexPairings:
    """Pairings read from a Takeout_Inventory index.

    Only archives whose name, size and mtime match the index are covered;
    every other member falls back to `pairing.py`.
    """

    path: Path
    covered: frozenset[str] = frozenset()
    uncovered: frozenset[str] = frozenset()
    pairings: Mapping[str, IndexedPairing] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path, archive_stats: Mapping[str, Any]) -> "IndexPairings":
        """Load *path*, verifying it against the archives actually on disk.

        `archive_stats` maps an archive's file name to its `os.stat_result`
        (anything with `st_size` and `st_mtime` works).
        """
        path = Path(path)
        if not path.is_file():
            raise IndexUnusable(f"no index at {path}")
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise IndexUnusable(f"cannot open {path}: {exc}") from exc
        try:
            cls._verify(con, path)
            covered, uncovered = cls._verify_archives(con, archive_stats)
            pairings = cls._read_pairings(con, covered)
        except sqlite3.Error as exc:
            raise IndexUnusable(f"{path} is not a readable index: {exc}") from exc
        finally:
            con.close()
        return cls(path=path, covered=frozenset(covered),
                   uncovered=frozenset(uncovered), pairings=pairings)

    @staticmethod
    def _verify(con: sqlite3.Connection, path: Path) -> None:
        """Version and column checks.

        The column check exists because this file and its producer live in
        different repositories: a schema change there must surface here as a
        clear error, never as a wrong answer.
        """
        try:
            row = con.execute(
                "SELECT value FROM index_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise IndexUnusable(
                f"{path} has no index_meta table; it predates schema versioning"
            ) from exc
        if row is None:
            raise IndexUnusable(f"{path} records no schema_version")
        try:
            found = int(row[0])
        except (TypeError, ValueError) as exc:
            raise IndexUnusable(f"{path} has a non-numeric schema_version") from exc
        if found > SCHEMA_VERSION:
            raise IndexUnusable(
                f"{path} is schema {found}; this build knows {SCHEMA_VERSION}")
        for table, expected in (("media", _MEDIA_COLUMNS),
                                ("sidecar", _SIDECAR_COLUMNS),
                                ("archive", _ARCHIVE_COLUMNS)):
            names = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            missing = expected - names
            if missing:
                raise IndexUnusable(
                    f"{path} table {table} is missing {sorted(missing)}")

    @staticmethod
    def _verify_archives(
        con: sqlite3.Connection, archive_stats: Mapping[str, Any],
    ) -> tuple[set[str], set[str]]:
        indexed = {
            name: (size, mtime)
            for name, size, mtime in con.execute(
                "SELECT name, size, mtime FROM archive")
        }
        covered: set[str] = set()
        uncovered: set[str] = set()
        for name, st in archive_stats.items():
            entry = indexed.get(name)
            if entry is not None and entry == (st.st_size, int(st.st_mtime)):
                covered.add(name)
            else:
                uncovered.add(name)
        return covered, uncovered

    @staticmethod
    def _read_pairings(
        con: sqlite3.Connection, covered: set[str],
    ) -> dict[str, IndexedPairing]:
        out: dict[str, IndexedPairing] = {}
        for m_path, m_archive, s_path, rule, confidence in con.execute(
            "SELECT m.path, m.archive, s.path, m.rule, m.confidence"
            " FROM media m LEFT JOIN sidecar s ON s.id = m.sidecar_id"
        ):
            if m_archive not in covered:
                continue
            out[m_path] = IndexedPairing(
                sidecar=s_path, confidence=confidence, rule=rule)
        return out

    def covers(self, archive_name: str) -> bool:
        return archive_name in self.covered

    def sidecar_for(self, member_path: str) -> IndexedPairing | None:
        """The pairing for *member_path*, or None if the index does not have it."""
        return self.pairings.get(member_path)
