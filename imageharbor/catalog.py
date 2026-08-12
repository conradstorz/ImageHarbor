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
    pcs_primary      TEXT,
    pcs_name         TEXT,
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

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Bumped when a catalog schema change is significant enough that opening an
# old catalog in place would silently corrupt its data (see
# `Catalog._guard_legacy_catalog`). Stamped into `meta` on every catalog that
# opens successfully, so a future such change has something to compare against.
SCHEMA_VERSION = "2"


class LegacyCatalogError(Exception):
    """Raised when a pre-redesign (schema v1) catalog with real content is
    opened in place instead of being rebuilt.

    See `docs/rebuild.md`. The 2026-08-11 facts-first redesign added the
    `date_tier`/`descriptor_tier`/`enriched_at` columns and the `sources`
    table; `Catalog._ensure_photo_columns` makes an old database *open*
    cleanly by adding the missing columns, but every pre-existing row then
    reads as `date_tier=0` (Undated) and `enriched_at IS NULL` (never
    enriched). `iter_unenriched` would return the whole existing library, and
    `tiers.is_upgrade((0, 0), (0, 20))` is True for every one of those rows --
    so the very first `enrich` (or `watch`) pass against an old catalog opened
    in place would relocate the entire already-organized tree into
    `Undated/`. The fix is to rebuild the catalog against the same source
    into a fresh destination (`docs/rebuild.md`), not to open the old one.
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
        self._guard_legacy_catalog()
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

    def _guard_legacy_catalog(self) -> None:
        """Refuse to open a pre-redesign catalog in place; stamp a fresh one.

        Detection: `photos` has rows, `meta` has no `schema_version` row, and
        `sources` is empty. Every write path added by the facts-first
        redesign populates `sources` (the facts pass always calls
        `record_source`), so a catalog with photo rows but zero `sources`
        rows and no version stamp is unambiguously pre-redesign -- see
        `LegacyCatalogError` for what opening it in place would do.

        A brand-new catalog (no rows at all) and an empty legacy catalog (the
        `photos` table exists but nothing was ever written to it) are not a
        hazard either way -- there is nothing to misinterpret -- so both are
        stamped and opened normally. A catalog already stamped at the current
        version is a no-op.
        """
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is not None:
            return
        photos_count = self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        if photos_count > 0:
            sources_count = self._conn.execute(
                "SELECT COUNT(*) FROM sources"
            ).fetchone()[0]
            if sources_count == 0:
                raise LegacyCatalogError(
                    f"{self._db_path} looks like a pre-redesign ImageHarbor catalog "
                    "(it has photo rows but no 'sources' rows and no schema_version "
                    "stamp). Opening it in place would relocate the entire existing "
                    "organized tree into Undated/ on the next enrichment pass -- see "
                    "docs/rebuild.md for how to rebuild it safely into a fresh "
                    "destination instead."
                )
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()

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
        # NULL (not "900"/"miscellaneous") until the enrichment pass actually
        # classifies this row -- an unenriched row must not assert a
        # classification that was never made. `catalog list` renders these
        # as "—" for rows where enriched_at IS NULL. `mark_enriched`
        # always supplies real values once classification happens.
        pcs_primary: str | None = None,
        pcs_name: str | None = None,
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
        """Append a duplicate-detection event to the processing history.

        A no-op if the most recent entry already records this exact path as
        a duplicate. `process` is advertised as cheap to re-run, but the
        pipeline calls `mark_duplicate` on every re-encounter of a known
        digest -- without this guard, re-running `process` against an
        unchanged source tree would append an identical entry (only the
        timestamp differs) and rewrite the whole JSON blob on every single
        pass, forever, growing `processing_history` without bound.
        """
        row = self.get_by_sha256(sha256_b64url)
        if row is None:
            return
        history = _from_json(row["processing_history"])
        if not isinstance(history, list):
            history = []
        if (
            history
            and isinstance(history[-1], dict)
            and history[-1].get("event") == "duplicate_detected"
            and history[-1].get("duplicate_path") == duplicate_path
        ):
            return
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

        `size` and `mtime_ns` are likewise written once. The row is keyed by
        digest, so its content is fixed by definition -- size is a function of
        that content and cannot change, and an mtime that moves without the
        bytes moving is metadata noise (a touch, a backup tool, a CIFS
        remount). Overwriting the stats on re-observation would let that noise
        silently lift a quarantine, because `iter_unenriched` correlates the
        exclusion on exactly this `(path, size, mtime_ns)` triple. Only
        `last_seen_at` moves.
        """
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO sources (
                sha256_b64url, source_path, size, mtime_ns, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256_b64url, source_path) DO UPDATE SET
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

    def iter_unenriched(
        self, limit: int | None = None, offset: int = 0
    ) -> list[sqlite3.Row]:
        """Rows the AI enrichment pass has not yet processed.

        *offset* skips the first *offset* rows of the (fixed `ORDER BY p.id`)
        result. It exists so the watcher's half-open probe can rotate past a
        cluster of permanently-failing rows at the head of the queue instead
        of re-trying the same head cluster (and re-tripping the breaker) on
        every pass -- see `watcher.watch`'s probe-offset tracking. SQLite
        requires `LIMIT` before `OFFSET`, and `OFFSET` alone is not valid
        SQL, so when *limit* is `None` this uses the SQLite idiom `LIMIT -1
        OFFSET ?` (a negative `LIMIT` means "no limit").

        Excludes quarantined content. Quarantine means "stop asking the model
        about this one", so a quarantined file must leave the queue entirely --
        otherwise the poison file is re-described on every pass forever, which
        is the exact cost quarantine exists to eliminate.

        `failed_files` is keyed by source path while `photos` is keyed by
        digest, so the exclusion joins through `sources`. Content reachable from
        several paths is quarantined if ANY of them is: identical bytes fail
        identically, so one quarantined path condemns the content.

        The join also requires `(size, mtime_ns)` to match between
        `failed_files` and `sources` -- the same triple `is_quarantined()`
        already uses -- rather than matching on `source_path` alone.
        `sources` is keyed by `(digest, source_path)`, so editing the file at
        a quarantined path in place (replacing a poison photo with a fixed
        one under the same filename) inserts a NEW `sources` row for the new
        digest at that path. Without the stat correlation that new row would
        still join to the OLD `failed_files` record purely on path and stay
        excluded forever -- and nothing would ever lift it, because
        `record_file_failure`'s stale-stat reset only runs when the file is
        actually attempted, which the exclusion itself prevents. Per
        CLAUDE.md, a quarantined file must be skipped "until its bytes
        change"; the stat correlation is what makes that recovery path work:
        new bytes at the same path produce different size/mtime, so the join
        no longer matches and the new content re-enters the queue.

        `--reclassify` deliberately bypasses this (it walks `iter_all`): asking
        again is the whole point of an explicit re-do.
        """
        sql = (
            "SELECT * FROM photos p WHERE p.enriched_at IS NULL "
            "AND p.organized_path IS NOT NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM sources s"
            "  JOIN failed_files f ON f.source_path = s.source_path"
            "   AND f.size = s.size AND f.mtime_ns = s.mtime_ns"
            "  WHERE s.sha256_b64url = p.sha256_b64url AND f.quarantined = 1"
            ") ORDER BY p.id"
        )
        params: list[Any] = []
        if limit is not None or offset:
            sql += " LIMIT ?"
            params.append(limit if limit is not None else -1)
            if offset:
                sql += " OFFSET ?"
                params.append(offset)
        return list(self._conn.execute(sql, tuple(params)))

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
