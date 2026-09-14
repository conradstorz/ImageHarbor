"""SQLite storage for Google Takeout ingestion state.

Both tables (`takeout_archives`, `takeout_members`) are purely additive: no
existing row is reinterpreted and no existing column changes meaning, so
`SCHEMA_VERSION` stays "2" and `Catalog._guard_legacy_catalog` correctly does
not fire for a catalog that predates them.

`TakeoutStore` is composed onto `Catalog`'s existing connection and lock
(`Catalog.__init__` builds it as `self.takeout = TakeoutStore(conn=self._conn,
lock=self.lock)`) rather than opening a second connection -- see the
single-connection rationale in `catalog.py`'s `Catalog` docstring. This module
holds only the references it is given; it never calls `sqlite3.connect`
itself.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

from ..util import now_iso as _now_iso


class TakeoutStore:
    """Google Takeout archive/member bookkeeping on the shared catalog connection."""

    def __init__(self, *, conn: sqlite3.Connection, lock: threading.RLock) -> None:
        self._conn = conn
        self.lock = lock

    def archive_get(self, archive_id: str) -> sqlite3.Row | None:
        with self.lock:
            return self._conn.execute(
                "SELECT * FROM takeout_archives WHERE archive_id = ?", (archive_id,)
            ).fetchone()

    def archive_get_by_stat(
        self, last_path: str, size: int, mtime_ns: int
    ) -> sqlite3.Row | None:
        """The identity fast path: recognise an archive without hashing it.

        A match on (path, size, mtime_ns) is not proof of identical content, but
        it is never used as one: it only avoids re-hashing an archive we have
        already hashed at that exact path/size/mtime. Any change to any of the
        three falls through to the digest.
        """
        with self.lock:
            return self._conn.execute(
                """
                SELECT * FROM takeout_archives
                WHERE last_path = ? AND size = ? AND mtime_ns = ?
                """,
                (last_path, size, mtime_ns),
            ).fetchone()

    def archives_all(self) -> list[sqlite3.Row]:
        with self.lock:
            return list(
                self._conn.execute("SELECT * FROM takeout_archives ORDER BY last_path")
            )

    def archive_upsert(
        self,
        *,
        archive_id: str,
        last_path: str,
        size: int,
        mtime_ns: int,
        member_count: int = 0,
        status: str = "partial",
        last_error: str = "",
    ) -> None:
        """Record an archive, keyed by the digest of its own bytes.

        `last_path`, `mtime_ns`, `member_count`, `status`, `last_error`, and
        `last_seen_at` all move on conflict -- the same archive may be copied
        or re-downloaded elsewhere, re-surveyed, or re-tried -- but
        `archive_id` never does, so a renamed archive is recognised rather
        than re-ingested. `first_seen_at` is written once.
        """
        with self.lock:
            now = _now_iso()
            self._conn.execute(
                """
                INSERT INTO takeout_archives (
                    archive_id, last_path, size, mtime_ns, member_count, status,
                    last_error, first_seen_at, last_seen_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(archive_id) DO UPDATE SET
                    last_path    = excluded.last_path,
                    mtime_ns     = excluded.mtime_ns,
                    member_count = excluded.member_count,
                    status       = excluded.status,
                    last_error   = excluded.last_error,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    archive_id, last_path, size, mtime_ns, member_count, status,
                    last_error, now, now,
                ),
            )
            self._conn.commit()

    def archive_set_status(
        self, archive_id: str, status: str, last_error: str = ""
    ) -> None:
        with self.lock:
            self._conn.execute(
                """
                UPDATE takeout_archives
                SET status = ?, last_error = ?, last_seen_at = ?
                WHERE archive_id = ?
                """,
                (status, last_error, _now_iso(), archive_id),
            )
            self._conn.commit()

    def member_add(
        self,
        *,
        archive_id: str,
        member_path: str,
        kind: str,
        size: int,
        crc32: int,
        status: str,
    ) -> None:
        """Record a member seen in an archive's central directory.

        DO NOTHING on conflict, deliberately. `archive_id` is the digest of the
        archive's own bytes, so the same id implies the same central directory,
        which implies the same kind/size/crc at the same member path -- there is
        nothing to refresh. What there IS to protect is `status`: re-surveying
        an archive must never drag an already-ingested member back to 'pending'
        and re-extract it.
        """
        with self.lock:
            self._conn.execute(
                """
                INSERT INTO takeout_members (
                    archive_id, member_path, kind, size, crc32, status, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(archive_id, member_path) DO NOTHING
                """,
                (archive_id, member_path, kind, size, crc32, status, _now_iso()),
            )
            self._conn.commit()

    def member_set(
        self,
        archive_id: str,
        member_path: str,
        *,
        status: str,
        sha256_b64url: str | None = None,
        taken_at: str | None = None,
        sidecar_path: str | None = None,
        last_error: str = "",
    ) -> None:
        """Record the outcome of ingesting one member.

        This is a blind full-row overwrite: `sha256_b64url`, `taken_at`, and
        `sidecar_path` are always written, so a caller that omits any of them
        clobbers the stored value with NULL rather than leaving it untouched.
        There is no COALESCE-style "only overwrite what's given" behaviour
        here -- a caller must pass every field it wants preserved. This is
        acceptable because each member is finalized exactly once from
        'pending' (a straight-line write with nothing yet to preserve), and a
        'failed' member's retry re-supplies every field from scratch rather
        than layering onto the failed attempt. A future caller that updates a
        member twice with different field subsets -- e.g. finalizing a
        previously-'deferred' video to 'ingested' without re-passing an
        already-recorded `taken_at` -- would silently null it out; such a
        caller must re-read and re-pass the existing value itself.
        """
        with self.lock:
            self._conn.execute(
                """
                UPDATE takeout_members SET
                    status = ?, sha256_b64url = ?, taken_at = ?, sidecar_path = ?,
                    last_error = ?, updated_at = ?
                WHERE archive_id = ? AND member_path = ?
                """,
                (
                    status, sha256_b64url, taken_at, sidecar_path, last_error,
                    _now_iso(), archive_id, member_path,
                ),
            )
            self._conn.commit()

    def members_pending(self, archive_id: str) -> list[sqlite3.Row]:
        """Members still owed work: 'pending' (never tried) or 'failed' (retry).

        An ingest failure is a local filesystem or archive fault, not a backend
        outage, so a failed member is simply retried next run -- there is no
        quarantine ladder and no backoff here.
        """
        with self.lock:
            return list(
                self._conn.execute(
                    """
                    SELECT * FROM takeout_members
                    WHERE archive_id = ? AND status IN ('pending', 'failed')
                    ORDER BY member_path
                    """,
                    (archive_id,),
                )
            )

    def members_all(self, archive_id: str) -> list[sqlite3.Row]:
        with self.lock:
            return list(
                self._conn.execute(
                    "SELECT * FROM takeout_members WHERE archive_id = ? ORDER BY member_path",
                    (archive_id,),
                )
            )

    def members_unskip_trash(self, archive_id: str) -> int:
        """Restore each trash member to the status its KIND warrants; returns
        how many moved.

        Called only when --include-trash is passed, so a user who changes their
        mind is not blocked by the terminal status recorded on an earlier run.
        Only image/video rows are work items -- resetting a metadata or album
        row to 'pending' would strand it in a queue nothing drains (only
        `_ingest_archive` drains 'pending', and it only ever looks at
        image/video rows), so those kinds go back to their own terminal
        status ('parsed') instead, and anything else goes back to 'ignored'.
        """
        with self.lock:
            cur = self._conn.execute(
                """
                UPDATE takeout_members SET status = CASE
                    WHEN kind IN ('image', 'video')    THEN 'pending'
                    WHEN kind IN ('metadata', 'album') THEN 'parsed'
                    ELSE 'ignored'
                END,
                updated_at = ?
                WHERE archive_id = ? AND status = 'skipped_trash'
                """,
                (_now_iso(), archive_id),
            )
            self._conn.commit()
            return cur.rowcount

    def status_counts(self) -> dict[str, Any]:
        """Aggregates for `imageharbor takeout status`."""
        with self.lock:
            archives = {
                row["status"]: row["n"]
                for row in self._conn.execute(
                    "SELECT status, COUNT(*) AS n FROM takeout_archives GROUP BY status"
                )
            }
            members = {
                row["status"]: row["n"]
                for row in self._conn.execute(
                    "SELECT status, COUNT(*) AS n FROM takeout_members GROUP BY status"
                )
            }
            missing = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM takeout_members
                WHERE kind = 'image' AND status IN ('ingested', 'duplicate')
                  AND sidecar_path IS NULL
                """
            ).fetchone()["n"]
            return {"archives": archives, "members": members, "missing_metadata": missing}
