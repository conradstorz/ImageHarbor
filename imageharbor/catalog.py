"""SQLite catalog for organized photo metadata."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256_b64url    TEXT    NOT NULL UNIQUE,
    original_path    TEXT    NOT NULL,
    organized_path   TEXT,
    pcs_version      TEXT    NOT NULL DEFAULT '1',
    pcs_primary      INTEGER NOT NULL DEFAULT 900,
    pcs_name         TEXT    NOT NULL DEFAULT 'miscellaneous',
    secondary_tags   TEXT    NOT NULL DEFAULT '[]',
    ai_caption       TEXT    NOT NULL DEFAULT '',
    objects          TEXT    NOT NULL DEFAULT '[]',
    ocr_text         TEXT    NOT NULL DEFAULT '',
    exif             TEXT    NOT NULL DEFAULT '{}',
    model_version    TEXT    NOT NULL DEFAULT 'unknown',
    processing_history TEXT  NOT NULL DEFAULT '[]',
    created_at       TEXT    NOT NULL,
    processed_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_sha256 ON photos(sha256_b64url);
CREATE INDEX IF NOT EXISTS idx_pcs_primary ON photos(pcs_primary);
CREATE INDEX IF NOT EXISTS idx_processed_at ON photos(processed_at);

CREATE TABLE IF NOT EXISTS source_seen (
    source_path   TEXT    PRIMARY KEY,
    size          INTEGER NOT NULL,
    mtime_ns      INTEGER NOT NULL,
    sha256_b64url TEXT,
    seen_at       TEXT    NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _json_default(o: Any) -> Any:
    """Fallback for values ``json.dumps`` cannot serialize natively.

    Real EXIF carries raw ``bytes`` (e.g. ExifVersion, SceneType, MakerNote)
    and other exotic types; without this a single odd metadata value would
    raise and fail the whole image. Bytes become a lossy text form; anything
    else falls back to its string representation.
    """
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).decode("utf-8", "replace")
    return str(o)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _from_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Malformed JSON in catalog field; returning raw text: %r", text)
        return text


class Catalog:
    """Thin wrapper around a SQLite database for the ImageHarbor catalog."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.debug("Catalog opened at %s", db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        *,
        sha256_b64url: str,
        original_path: str,
        organized_path: str | None = None,
        pcs_version: str = "1",
        pcs_primary: int = 900,
        pcs_name: str = "miscellaneous",
        secondary_tags: list[str] | None = None,
        ai_caption: str = "",
        objects: list[str] | None = None,
        ocr_text: str = "",
        exif: dict[str, Any] | None = None,
        model_version: str = "unknown",
        processing_history: list[dict] | None = None,
    ) -> int:
        """Insert or update a photo record. Returns the row id."""
        now = _now_iso()
        existing = self.get_by_sha256(sha256_b64url)

        if existing:
            history = _from_json(existing["processing_history"])
            if not isinstance(history, list):
                history = []
        else:
            history = []

        if processing_history:
            history.extend(processing_history)

        params = (
            sha256_b64url,
            # original_path is intentionally NOT in the ON CONFLICT DO UPDATE SET
            # list below, so on conflict the first-seen path wins (never updated).
            original_path,
            organized_path,
            pcs_version,
            pcs_primary,
            pcs_name,
            _json(secondary_tags or []),
            ai_caption,
            _json(objects or []),
            ocr_text,
            _json(exif or {}),
            model_version,
            _json(history),
            now,  # created_at (preserved on UPDATE: not in the ON CONFLICT SET list)
            now,  # processed_at
        )

        cursor = self._conn.execute(
            """
            INSERT INTO photos (
                sha256_b64url, original_path, organized_path,
                pcs_version, pcs_primary, pcs_name,
                secondary_tags, ai_caption, objects, ocr_text, exif,
                model_version, processing_history, created_at, processed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(sha256_b64url) DO UPDATE SET
                organized_path    = excluded.organized_path,
                pcs_version       = excluded.pcs_version,
                pcs_primary       = excluded.pcs_primary,
                pcs_name          = excluded.pcs_name,
                secondary_tags    = excluded.secondary_tags,
                ai_caption        = excluded.ai_caption,
                objects           = excluded.objects,
                ocr_text          = excluded.ocr_text,
                exif              = excluded.exif,
                model_version     = excluded.model_version,
                processing_history = excluded.processing_history,
                processed_at      = excluded.processed_at
            """,
            params,
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def mark_duplicate(self, sha256_b64url: str, duplicate_path: str) -> None:
        """Append a duplicate-detection event to the processing history."""
        row = self.get_by_sha256(sha256_b64url)
        if row is None:
            return
        history = _from_json(row["processing_history"])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "event": "duplicate_detected",
                "duplicate_path": duplicate_path,
                "at": _now_iso(),
            }
        )
        self._conn.execute(
            "UPDATE photos SET processing_history=? WHERE sha256_b64url=?",
            (_json(history), sha256_b64url),
        )
        self._conn.commit()

    def record_source_seen(
        self,
        source_path: str,
        size: int,
        mtime_ns: int,
        sha256_b64url: str | None = None,
    ) -> None:
        """Record (or update) that a source file was processed, keyed by path."""
        self._conn.execute(
            """
            INSERT INTO source_seen (source_path, size, mtime_ns, sha256_b64url, seen_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(source_path) DO UPDATE SET
                size          = excluded.size,
                mtime_ns      = excluded.mtime_ns,
                sha256_b64url = excluded.sha256_b64url,
                seen_at       = excluded.seen_at
            """,
            (source_path, size, mtime_ns, sha256_b64url, _now_iso()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_sha256(self, sha256_b64url: str) -> sqlite3.Row | None:
        cursor = self._conn.execute(
            "SELECT * FROM photos WHERE sha256_b64url=?", (sha256_b64url,)
        )
        return cursor.fetchone()

    def get_by_original_path(self, original_path: str) -> sqlite3.Row | None:
        cursor = self._conn.execute(
            "SELECT * FROM photos WHERE original_path=?", (original_path,)
        )
        return cursor.fetchone()

    def iter_all(self) -> Iterator[sqlite3.Row]:
        cursor = self._conn.execute("SELECT * FROM photos ORDER BY id")
        yield from cursor

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM photos")
        return cursor.fetchone()[0]

    def is_known(self, sha256_b64url: str) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM photos WHERE sha256_b64url=? LIMIT 1", (sha256_b64url,)
        )
        return cursor.fetchone() is not None

    def source_is_unchanged(self, source_path: str, size: int, mtime_ns: int) -> bool:
        """Return True if this source path was seen before with the same size
        and mtime (so it can be skipped without re-hashing)."""
        cur = self._conn.execute(
            "SELECT size, mtime_ns FROM source_seen WHERE source_path=?",
            (source_path,),
        )
        row = cur.fetchone()
        if row is None:
            return False
        return row["size"] == size and row["mtime_ns"] == mtime_ns

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
