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
    pcs_primary      TEXT    NOT NULL DEFAULT '900',
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

CREATE TABLE IF NOT EXISTS taxonomy (
    code         TEXT    PRIMARY KEY,
    parent_code  TEXT,
    label        TEXT    NOT NULL,
    folder_name  TEXT    NOT NULL,
    aliases      TEXT    NOT NULL DEFAULT '[]',
    alias_of     TEXT,
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_taxonomy_parent ON taxonomy(parent_code);

CREATE TABLE IF NOT EXISTS learned_concepts (
    subject     TEXT    PRIMARY KEY,
    class_code  TEXT    NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS failed_files (
    source_path     TEXT    PRIMARY KEY,
    size            INTEGER NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT    NOT NULL DEFAULT '',
    first_failed_at TEXT    NOT NULL,
    last_failed_at  TEXT    NOT NULL,
    quarantined     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sources (
    sha256_b64url TEXT    NOT NULL,
    source_path   TEXT    NOT NULL,
    size          INTEGER,
    mtime_ns      INTEGER,
    first_seen_at TEXT    NOT NULL,
    last_seen_at  TEXT    NOT NULL,
    PRIMARY KEY (sha256_b64url, source_path)
);
CREATE INDEX IF NOT EXISTS idx_sources_digest ON sources(sha256_b64url);
"""

# Columns added to `photos` after the original schema shipped. Applied
# additively on open so an existing catalog upgrades in place.
_ADDED_PHOTO_COLUMNS: tuple[tuple[str, str], ...] = (
    ("date_value", "TEXT"),
    ("date_tier", "INTEGER NOT NULL DEFAULT 0"),
    ("date_source", "TEXT NOT NULL DEFAULT 'none'"),
    ("descriptor_value", "TEXT NOT NULL DEFAULT ''"),
    ("descriptor_tier", "INTEGER NOT NULL DEFAULT 0"),
    ("descriptor_source", "TEXT NOT NULL DEFAULT 'none'"),
    ("scene", "TEXT NOT NULL DEFAULT ''"),
    ("enriched_at", "TEXT"),
)


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
        self._ensure_photo_columns()
        self._conn.commit()
        logger.debug("Catalog opened at %s", db_path)

    def _ensure_photo_columns(self) -> None:
        """Add post-1.0 columns to `photos` if this DB predates them."""
        existing = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(photos)")
        }
        for name, ddl in _ADDED_PHOTO_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE photos ADD COLUMN {name} {ddl}")
                logger.debug("Catalog upgraded: added photos.%s", name)

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
        pcs_primary: str = "900",
        pcs_name: str = "miscellaneous",
        secondary_tags: list[str] | None = None,
        ai_caption: str = "",
        objects: list[str] | None = None,
        ocr_text: str = "",
        exif: dict[str, Any] | None = None,
        model_version: str = "unknown",
        processing_history: list[dict] | None = None,
        date_value: str | None = None,
        date_tier: int = 0,
        date_source: str = "none",
        descriptor_value: str = "",
        descriptor_tier: int = 0,
        descriptor_source: str = "none",
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
            date_value,
            date_tier,
            date_source,
            descriptor_value,
            descriptor_tier,
            descriptor_source,
            now,  # created_at (preserved on UPDATE: not in the ON CONFLICT SET list)
            now,  # processed_at
        )

        cursor = self._conn.execute(
            """
            INSERT INTO photos (
                sha256_b64url, original_path, organized_path,
                pcs_version, pcs_primary, pcs_name,
                secondary_tags, ai_caption, objects, ocr_text, exif,
                model_version, processing_history,
                date_value, date_tier, date_source,
                descriptor_value, descriptor_tier, descriptor_source,
                created_at, processed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                date_value        = excluded.date_value,
                date_tier         = excluded.date_tier,
                date_source       = excluded.date_source,
                descriptor_value  = excluded.descriptor_value,
                descriptor_tier   = excluded.descriptor_tier,
                descriptor_source = excluded.descriptor_source,
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

    def record_source(
        self, sha256_b64url: str, source_path: str, size: int, mtime_ns: int
    ) -> None:
        """Record that *source_path* holds the bytes identified by the digest.

        One row per distinct source path: this is the many-to-one back-pointer
        set that replaces a single `original_path`. `first_seen_at` is written
        once and never updated.
        """
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO sources (
                sha256_b64url, source_path, size, mtime_ns, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256_b64url, source_path) DO UPDATE SET
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                last_seen_at = excluded.last_seen_at
            """,
            (sha256_b64url, source_path, size, mtime_ns, now, now),
        )
        self._conn.commit()

    def sources_for(self, sha256_b64url: str) -> list[sqlite3.Row]:
        """All known source paths for a digest, oldest first."""
        return list(
            self._conn.execute(
                "SELECT * FROM sources WHERE sha256_b64url = ? ORDER BY first_seen_at",
                (sha256_b64url,),
            )
        )

    def tiers_for(self, sha256_b64url: str) -> tuple[int, int]:
        """Return ``(date_tier, descriptor_tier)``; ``(0, 0)`` if unknown."""
        row = self._conn.execute(
            "SELECT date_tier, descriptor_tier FROM photos WHERE sha256_b64url = ?",
            (sha256_b64url,),
        ).fetchone()
        if row is None:
            return (0, 0)
        return (row["date_tier"] or 0, row["descriptor_tier"] or 0)

    def set_placement(
        self,
        sha256_b64url: str,
        *,
        organized_path: str,
        date_value: str | None,
        date_tier: int,
        date_source: str,
        descriptor_value: str,
        descriptor_tier: int,
        descriptor_source: str,
    ) -> None:
        """Record a new organized path and the tiers that justified it."""
        self._conn.execute(
            """
            UPDATE photos SET
                organized_path = ?, date_value = ?, date_tier = ?, date_source = ?,
                descriptor_value = ?, descriptor_tier = ?, descriptor_source = ?,
                processed_at = ?
            WHERE sha256_b64url = ?
            """,
            (
                organized_path, date_value, date_tier, date_source,
                descriptor_value, descriptor_tier, descriptor_source,
                _now_iso(), sha256_b64url,
            ),
        )
        self._conn.commit()

    def iter_unenriched(self, limit: int | None = None) -> list[sqlite3.Row]:
        """Rows the AI enrichment pass has not yet processed."""
        sql = (
            "SELECT * FROM photos WHERE enriched_at IS NULL "
            "AND organized_path IS NOT NULL ORDER BY id"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return list(self._conn.execute(sql, params))

    def mark_enriched(
        self,
        sha256_b64url: str,
        *,
        pcs_primary: str,
        pcs_name: str,
        secondary_tags: list[str],
        ai_caption: str,
        objects: list[str],
        ocr_text: str,
        model_version: str,
        scene: str = "",
    ) -> None:
        """Store the AI's perception and stamp the row as enriched."""
        self._conn.execute(
            """
            UPDATE photos SET
                pcs_primary = ?, pcs_name = ?, secondary_tags = ?, ai_caption = ?,
                objects = ?, ocr_text = ?, model_version = ?, scene = ?,
                enriched_at = ?
            WHERE sha256_b64url = ?
            """,
            (
                pcs_primary, pcs_name, _json(secondary_tags), ai_caption,
                _json(objects), ocr_text, model_version, scene,
                _now_iso(), sha256_b64url,
            ),
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
    # Taxonomy
    # ------------------------------------------------------------------

    def taxonomy_is_empty(self) -> bool:
        cur = self._conn.execute("SELECT 1 FROM taxonomy LIMIT 1")
        return cur.fetchone() is None

    def taxonomy_insert(
        self,
        code: str,
        parent_code: str | None,
        label: str,
        folder_name: str,
        aliases: list[str] | None = None,
        alias_of: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO taxonomy (code, parent_code, label, folder_name,
                                  aliases, alias_of, active, created_at)
            VALUES (?,?,?,?,?,?,1,?)
            ON CONFLICT(code) DO NOTHING
            """,
            (code, parent_code, label, folder_name, _json(aliases or []), alias_of, _now_iso()),
        )
        self._conn.commit()

    def taxonomy_get(self, code: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM taxonomy WHERE code=?", (code,))
        return cur.fetchone()

    def taxonomy_children(self, parent_code: str | None) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM taxonomy WHERE parent_code IS ? ORDER BY code", (parent_code,)
        )
        return cur.fetchall()

    def taxonomy_all(self) -> list[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM taxonomy WHERE active=1 ORDER BY code")
        return cur.fetchall()

    def taxonomy_set_alias(self, from_code: str, to_code: str) -> None:
        self._conn.execute(
            "UPDATE taxonomy SET alias_of=?, active=0 WHERE code=?", (to_code, from_code)
        )
        self._conn.commit()

    def taxonomy_set_aliases(self, code: str, aliases: list[str]) -> None:
        self._conn.execute(
            "UPDATE taxonomy SET aliases=? WHERE code=?", (_json(aliases), code)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Learned concepts
    # ------------------------------------------------------------------

    def learned_concept_get(self, subject: str) -> str | None:
        cur = self._conn.execute(
            "SELECT class_code FROM learned_concepts WHERE subject=?", (subject,)
        )
        row = cur.fetchone()
        return row["class_code"] if row else None

    def learned_concept_remember(self, subject: str, class_code: str) -> None:
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO learned_concepts (subject, class_code, hits, created_at, updated_at)
            VALUES (?,?,1,?,?)
            ON CONFLICT(subject) DO UPDATE SET
                class_code = excluded.class_code,
                hits       = hits + 1,
                updated_at = excluded.updated_at
            """,
            (subject, class_code, now, now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Failed files (poison-file tracking)
    # ------------------------------------------------------------------

    def record_file_failure(
        self, source_path: str, size: int, mtime_ns: int, error: str
    ) -> int:
        """Record a processing failure for a source file; return new fail_count.

        If the stored size/mtime differ from the incoming values the file has
        changed on disk, so the count resets to 1 and any quarantine is cleared.
        """
        now = _now_iso()
        row = self._conn.execute(
            "SELECT size, mtime_ns, fail_count FROM failed_files WHERE source_path=?",
            (source_path,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO failed_files
                    (source_path, size, mtime_ns, fail_count, last_error,
                     first_failed_at, last_failed_at, quarantined)
                VALUES (?,?,?,?,?,?,?,0)
                """,
                (source_path, size, mtime_ns, 1, error, now, now),
            )
            self._conn.commit()
            return 1
        if row["size"] != size or row["mtime_ns"] != mtime_ns:
            self._conn.execute(
                """
                UPDATE failed_files
                   SET size=?, mtime_ns=?, fail_count=1, last_error=?,
                       last_failed_at=?, quarantined=0
                 WHERE source_path=?
                """,
                (size, mtime_ns, error, now, source_path),
            )
            self._conn.commit()
            return 1
        new_count = row["fail_count"] + 1
        self._conn.execute(
            "UPDATE failed_files SET fail_count=?, last_error=?, last_failed_at=? "
            "WHERE source_path=?",
            (new_count, error, now, source_path),
        )
        self._conn.commit()
        return new_count

    def quarantine_file(self, source_path: str) -> None:
        self._conn.execute(
            "UPDATE failed_files SET quarantined=1 WHERE source_path=?", (source_path,)
        )
        self._conn.commit()

    def is_quarantined(self, source_path: str, size: int, mtime_ns: int) -> bool:
        row = self._conn.execute(
            "SELECT quarantined, size, mtime_ns FROM failed_files WHERE source_path=?",
            (source_path,),
        ).fetchone()
        if row is None:
            return False
        return bool(row["quarantined"]) and row["size"] == size and row["mtime_ns"] == mtime_ns

    def clear_file_failure(self, source_path: str) -> None:
        self._conn.execute(
            "DELETE FROM failed_files WHERE source_path=?", (source_path,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
