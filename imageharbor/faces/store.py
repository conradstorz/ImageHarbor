"""SQLite persistence for faces, clusters, and the human confirmation gate.

This is the only place a person's identity is ever written. `record_proposals`
writes machine guesses to `proposals` and nothing else; `confirm` and `merge`
are the only two methods that touch `clusters.person_id` -- see the module's
tests for the mutation-tested guarantee.

Mirrors `imageharbor.catalog.Catalog`'s connection setup (`check_same_thread`,
row factory, WAL, busy timeout, a lock around every write) so the two stores
behave identically when they later share one on-disk file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .attribute import Proposal
from .cluster import Cluster, FaceVector
from .decode import Detection
from .names import normalize

logger = logging.getLogger(__name__)

# The five tables from the design spec's "Catalog schema" section, plus one
# addition: `face_organized_paths`. The spec is silent on where
# `set_organized_path` stores its data, but that method is part of this
# module's required interface, and FaceStore must not reach into
# `catalog.Catalog`'s own `photos` table to write it -- that table is owned by
# `Catalog`, and (per Task 12's fixtures) a digest may have no `photos` row at
# all when `set_organized_path` is called. A dedicated table keeps that
# ownership boundary intact.
#
# The spec's `bbox_x, bbox_y, bbox_w, bbox_h INTEGER NOT NULL` shorthand is
# not valid multi-column SQL -- SQLite parses each comma-separated segment as
# its own column-def, so only `bbox_h` would actually get the `INTEGER NOT
# NULL` type/constraint and the other three would be typeless and nullable.
# Expanded below to four real columns, preserving the evident intent.
_FACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
  id             INTEGER PRIMARY KEY,
  sha256_b64url  TEXT    NOT NULL,      -- -> photos
  bbox_x         INTEGER NOT NULL,
  bbox_y         INTEGER NOT NULL,
  bbox_w         INTEGER NOT NULL,
  bbox_h         INTEGER NOT NULL,
  det_score      REAL    NOT NULL,
  landmarks      TEXT    NOT NULL,      -- JSON, 5 points
  detect_model   TEXT    NOT NULL,      -- provenance: placed these landmarks
  embed_model    TEXT,                  -- provenance: made this vector
  embedding      BLOB,                  -- float32, L2-normalized
  embedding_dim  INTEGER,
  cluster_id     INTEGER,               -- NULL = unclustered
  rejected       TEXT,                  -- quality-gate reason, NULL = kept
  detected_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS face_scan (                -- work queue + idempotence
  sha256_b64url  TEXT    NOT NULL,
  detect_model   TEXT    NOT NULL,
  face_count     INTEGER NOT NULL,
  scanned_at     TEXT    NOT NULL,
  sidecar_at     TEXT,                  -- last person-propagation write
  PRIMARY KEY (sha256_b64url, detect_model)
);

CREATE TABLE IF NOT EXISTS clusters (
  id           INTEGER PRIMARY KEY,
  embed_model  TEXT    NOT NULL,        -- never compared across models
  centroid     BLOB,
  face_count   INTEGER NOT NULL,
  person_id    INTEGER,                 -- NULL until a human confirms
  assigned_at  TEXT,
  created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS people (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  source      TEXT NOT NULL,            -- human | google_photos_people | picasa_roster
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
  cluster_id       INTEGER NOT NULL,
  name             TEXT    NOT NULL,
  support          INTEGER NOT NULL,    -- photos in cluster tagged this name
  total_tagged     INTEGER NOT NULL,    -- photos in cluster tagged anything
  score            REAL    NOT NULL,
  untagged_photos  INTEGER NOT NULL DEFAULT 0,  -- what confirming would newly name
  proposed_at      TEXT    NOT NULL,
  decided          TEXT,                -- NULL | confirmed | rejected
  decided_at       TEXT,
  PRIMARY KEY (cluster_id, name)
);

CREATE INDEX IF NOT EXISTS idx_faces_digest  ON faces(sha256_b64url);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_model   ON faces(embed_model);
CREATE INDEX IF NOT EXISTS idx_clusters_person ON clusters(person_id);

CREATE TABLE IF NOT EXISTS face_organized_paths (
  sha256_b64url   TEXT PRIMARY KEY,
  organized_path  TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class MalformedEmbeddingError(ValueError):
    """A stored embedding blob's byte length doesn't match its recorded dim."""


class FaceStore:
    """Owns the face tables in a Catalog's SQLite file.

    Face persistence is intentionally decoupled from `catalog.Catalog`: it
    reaches its own connection to the same `db_path` rather than being handed
    a live `Catalog`, and never reads or writes `Catalog`-owned tables.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        # RLock, not Lock: `record_scan` calls `is_scanned` while already
        # holding the lock. Matches `Catalog.__init__`'s actual choice (its
        # docstring explains why a plain Lock would deadlock there too),
        # despite this task's brief text saying "threading.Lock" -- Catalog's
        # real source uses RLock.
        self.lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.executescript(_FACE_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Scan (work queue + idempotence)
    # ------------------------------------------------------------------

    def is_scanned(self, digest: str, detect_model: str) -> bool:
        with self.lock:
            row = self._conn.execute(
                "SELECT 1 FROM face_scan WHERE sha256_b64url=? AND detect_model=?",
                (digest, detect_model),
            ).fetchone()
            return row is not None

    def record_scan(
        self,
        digest: str,
        detect_model: str,
        faces: Sequence[
            tuple[Detection, np.ndarray | None, str | None]
            | tuple[Detection, np.ndarray | None, str | None, str | None]
        ],
    ) -> list[int]:
        """Record one photo's detected faces. Idempotent on (digest, detect_model).

        A repeat call for an already-scanned (digest, detect_model) returns the
        existing face ids without writing anything -- this is what makes a
        re-run of the scan pass a no-op.
        """
        with self.lock:
            if self.is_scanned(digest, detect_model):
                rows = self._conn.execute(
                    "SELECT id FROM faces WHERE sha256_b64url=? AND detect_model=? ORDER BY id",
                    (digest, detect_model),
                ).fetchall()
                return [row["id"] for row in rows]

            now = _now_iso()
            ids: list[int] = []
            for entry in faces:
                if len(entry) == 3:
                    det, embedding, embed_model = entry
                    rejected_reason = None
                else:
                    det, embedding, embed_model, rejected_reason = entry

                embedding_blob: bytes | None = None
                embedding_dim: int | None = None
                if embedding is not None:
                    arr = np.asarray(embedding, dtype=np.float32)
                    embedding_blob = arr.tobytes()
                    embedding_dim = int(arr.shape[0])

                cursor = self._conn.execute(
                    """
                    INSERT INTO faces (
                        sha256_b64url, bbox_x, bbox_y, bbox_w, bbox_h, det_score,
                        landmarks, detect_model, embed_model, embedding,
                        embedding_dim, cluster_id, rejected, detected_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)
                    """,
                    (
                        digest, det.x, det.y, det.w, det.h, det.score,
                        json.dumps(det.landmarks), detect_model, embed_model,
                        embedding_blob, embedding_dim, rejected_reason, now,
                    ),
                )
                ids.append(cursor.lastrowid)  # type: ignore[arg-type]

            self._conn.execute(
                """
                INSERT INTO face_scan (sha256_b64url, detect_model, face_count, scanned_at, sidecar_at)
                VALUES (?,?,?,?,NULL)
                """,
                (digest, detect_model, len(faces), now),
            )
            self._conn.commit()
            return ids

    # ------------------------------------------------------------------
    # Vectors
    # ------------------------------------------------------------------

    def _decode_embedding(self, row: sqlite3.Row) -> np.ndarray:
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        dim = row["embedding_dim"]
        if dim is not None and vec.shape[0] != dim:
            raise MalformedEmbeddingError(
                f"face {row['id']}: embedding blob has {vec.shape[0]} float32 "
                f"values but embedding_dim={dim}"
            )
        return vec

    def iter_face_vectors(self, embed_model: str) -> Iterator[FaceVector]:
        """Every unrejected face vector for *embed_model*.

        Embeddings are never compared across models -- filtering here is what
        keeps a mixed-model batch from ever reaching `cluster.cluster_faces`.
        """
        with self.lock:
            rows = list(
                self._conn.execute(
                    """
                    SELECT id, embedding, embedding_dim FROM faces
                    WHERE embed_model=? AND rejected IS NULL AND embedding IS NOT NULL
                    ORDER BY id
                    """,
                    (embed_model,),
                )
            )
        for row in rows:
            yield FaceVector(
                face_id=row["id"],
                embedding=self._decode_embedding(row),
                embed_model=embed_model,
            )

    def anchors(
        self, embed_model: str, photo_names: Mapping[str, Sequence[str]]
    ) -> list[tuple[str, np.ndarray]]:
        """`(name, embedding)` for photos with exactly one unrejected face and
        exactly one distinct normalized name.

        "Exactly one unrejected face" is counted across the whole photo
        (any detector), not scoped to *embed_model* -- that axis only decides
        which vector comes back once a photo qualifies.
        """
        out: list[tuple[str, np.ndarray]] = []
        with self.lock:
            for digest in sorted(photo_names):
                names = {normalize(n) for n in photo_names[digest] if normalize(n)}
                if len(names) != 1:
                    continue
                rows = self._conn.execute(
                    """
                    SELECT id, embed_model, embedding, embedding_dim FROM faces
                    WHERE sha256_b64url=? AND rejected IS NULL
                    """,
                    (digest,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                row = rows[0]
                if row["embed_model"] != embed_model or row["embedding"] is None:
                    continue
                out.append((next(iter(names)), self._decode_embedding(row)))
        return out

    def anchor_face_ids(
        self, embed_model: str, photo_names: Mapping[str, Sequence[str]]
    ) -> dict[str, list[int]]:
        """`{name: [face_id, ...]}` for the same anchor photos `anchors()` selects.

        `anchors()` returns `(name, embedding)` pairs because that is all
        `calibrate.calibrate` ever needs. `cluster.Seed` needs face ids
        instead, to place into `cluster_faces`'s own `by_id` lookup -- Task
        12's brief said to build seeds "from `store.anchors(...)`", but that
        method's return tuple has no id in it, only a name and a decoded
        embedding. Rather than widen `anchors()`'s tuple (which would break
        every existing two-value unpacking of its result, including
        `test_anchors_are_single_face_single_name_photos`), this is a second
        method with the identical selection criteria: one unrejected face,
        one distinct normalized name, that face embedded by *embed_model*.
        """
        out: dict[str, list[int]] = {}
        with self.lock:
            for digest in sorted(photo_names):
                names = {normalize(n) for n in photo_names[digest] if normalize(n)}
                if len(names) != 1:
                    continue
                rows = self._conn.execute(
                    """
                    SELECT id, embed_model, embedding FROM faces
                    WHERE sha256_b64url=? AND rejected IS NULL
                    """,
                    (digest,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                row = rows[0]
                if row["embed_model"] != embed_model or row["embedding"] is None:
                    continue
                out.setdefault(next(iter(names)), []).append(row["id"])
        return out

    def unclustered_face_count(self, embed_model: str) -> int:
        """How many *embed_model* faces have no cluster yet.

        Read-only counting query, deliberately separate from
        `iter_face_vectors` (which decodes every embedding just to iterate
        them) -- `imageharbor/watcher.py`'s faces pass calls this every
        cycle to decide whether a whole-library `build_clusters` run is due,
        and a `COUNT(*)` that never touches the embedding BLOBs is what
        keeps that decision cheap enough to make every cycle instead of
        only when a recluster is already known to be needed.
        """
        with self.lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS n FROM faces
                WHERE embed_model=? AND rejected IS NULL AND embedding IS NOT NULL
                  AND cluster_id IS NULL
                """,
                (embed_model,),
            ).fetchone()
            return row["n"]

    def digests_by_cluster(self, embed_model: str) -> dict[int, list[str]]:
        """`{cluster_id: [digest, ...]}` for every clustered face under *embed_model*.

        Feeds `attribute.propose`'s `cluster_photos` argument. Task 12's
        brief said to build this "from the store" without naming a method --
        nothing needed a photo-level view of a cluster before proposal-
        building did, so there was no existing accessor for it.
        """
        with self.lock:
            rows = self._conn.execute(
                """
                SELECT cluster_id, sha256_b64url FROM faces
                WHERE embed_model=? AND cluster_id IS NOT NULL
                ORDER BY cluster_id, id
                """,
                (embed_model,),
            ).fetchall()
        out: dict[int, list[str]] = {}
        for row in rows:
            out.setdefault(row["cluster_id"], []).append(row["sha256_b64url"])
        return out

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------

    def cluster_ids(self) -> list[int]:
        with self.lock:
            return [
                row["id"]
                for row in self._conn.execute("SELECT id FROM clusters ORDER BY id")
            ]

    def person_for_cluster(self, cluster_id: int) -> int | None:
        with self.lock:
            row = self._conn.execute(
                "SELECT person_id FROM clusters WHERE id=?", (cluster_id,)
            ).fetchone()
            return row["person_id"] if row is not None else None

    def replace_clusters(self, embed_model: str, clusters: Sequence[Cluster]) -> None:
        """Rebuild every cluster for *embed_model* from a fresh clustering run.

        A recluster is a machine operation and must never discard a human
        decision: before the old rows are deleted, every confirmed cluster's
        face set is captured, and after the new rows are inserted, any new
        cluster whose face set intersects a captured one gets that person
        back.

        A new cluster's face set can intersect *more than one* distinct
        confirmed person -- the spec's own "Aging" section treats one person
        owning several clusters, with merge/split as first-class repairs, as
        the expected steady state, so a recluster run merging two different
        people's confirmed clusters into one new cluster is a plausible real
        event, not a corner case. When that happens, picking either person
        would manufacture a confirmation nobody made -- losing a confirmation
        is the tolerated, safe direction, but inventing one is not. So the
        new cluster's `person_id` is left NULL (it returns to the review
        queue for a human) and a warning names the conflicting people. A
        single confirmed cluster splitting into several new fragments is the
        opposite, safe case -- every fragment intersects only that one
        person, so each still inherits it.
        """
        with self.lock:
            confirmed = self._conn.execute(
                """
                SELECT id, person_id FROM clusters
                WHERE embed_model=? AND person_id IS NOT NULL
                ORDER BY id
                """,
                (embed_model,),
            ).fetchall()
            confirmed_by_faces: list[tuple[frozenset[int], int]] = []
            for row in confirmed:
                face_ids = frozenset(
                    r["id"]
                    for r in self._conn.execute(
                        "SELECT id FROM faces WHERE cluster_id=?", (row["id"],)
                    )
                )
                if face_ids:
                    confirmed_by_faces.append((face_ids, row["person_id"]))

            old_ids = [
                row["id"]
                for row in self._conn.execute(
                    "SELECT id FROM clusters WHERE embed_model=?", (embed_model,)
                )
            ]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                # Proposals reference cluster ids that are about to stop
                # existing; without this cleanup a plain (non-AUTOINCREMENT)
                # `INTEGER PRIMARY KEY` could later reuse a deleted id and
                # silently resurrect a stale, unrelated proposal row.
                self._conn.execute(
                    f"DELETE FROM proposals WHERE cluster_id IN ({placeholders})",
                    old_ids,
                )
                self._conn.execute(
                    f"DELETE FROM clusters WHERE id IN ({placeholders})", old_ids
                )
            self._conn.execute(
                "UPDATE faces SET cluster_id=NULL WHERE embed_model=?", (embed_model,)
            )

            now = _now_iso()
            for cluster in clusters:
                centroid_blob = np.asarray(cluster.centroid, dtype=np.float32).tobytes()
                cursor = self._conn.execute(
                    """
                    INSERT INTO clusters (embed_model, centroid, face_count, person_id, assigned_at, created_at)
                    VALUES (?,?,?,NULL,NULL,?)
                    """,
                    (embed_model, centroid_blob, len(cluster.face_ids), now),
                )
                new_id = cursor.lastrowid
                if cluster.face_ids:
                    placeholders = ",".join("?" * len(cluster.face_ids))
                    self._conn.execute(
                        f"UPDATE faces SET cluster_id=? WHERE id IN ({placeholders})",
                        (new_id, *cluster.face_ids),
                    )

                new_face_set = frozenset(cluster.face_ids)
                matched_people = {
                    person_id
                    for old_face_set, person_id in confirmed_by_faces
                    if new_face_set & old_face_set
                }
                if len(matched_people) == 1:
                    self._conn.execute(
                        "UPDATE clusters SET person_id=?, assigned_at=? WHERE id=?",
                        (next(iter(matched_people)), now, new_id),
                    )
                elif len(matched_people) > 1:
                    placeholders = ",".join("?" * len(matched_people))
                    names = [
                        r["name"]
                        for r in self._conn.execute(
                            f"SELECT name FROM people WHERE id IN ({placeholders}) ORDER BY name",
                            tuple(matched_people),
                        )
                    ]
                    logger.warning(
                        "recluster: new cluster %s merges faces from confirmed "
                        "people %s -- leaving unconfirmed for human review "
                        "rather than picking one",
                        new_id, names,
                    )

            self._conn.commit()

    def split(self, cluster_id: int, face_ids: Sequence[int]) -> int:
        """Move *face_ids* out of *cluster_id* into a new, unconfirmed cluster.

        Splitting off outliers doesn't retract the remaining cluster's
        confirmation -- only the split-off faces start over as unreviewed,
        since they're exactly the ones a human just judged to not belong.
        Returns the new cluster's id.

        Every id in *face_ids* must currently belong to *cluster_id*; this is
        the layer that actually mutates, so it refuses even when a caller
        bypasses `dashboard.people.split`'s own (earlier, HTTP-facing) copy
        of this check -- a face silently stripped out of another cluster's
        membership, with `clusters.face_count` left stale, is exactly the
        bug this guards against. Raises before touching anything.
        """
        with self.lock:
            row = self._conn.execute(
                "SELECT embed_model FROM clusters WHERE id=?", (cluster_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no such cluster: {cluster_id}")
            embed_model = row["embed_model"]

            moving = list(face_ids)
            now = _now_iso()
            remaining_rows = self._conn.execute(
                "SELECT id, embedding, embedding_dim FROM faces WHERE cluster_id=?",
                (cluster_id,),
            ).fetchall()
            moving_set = set(moving)
            owned_ids = {r["id"] for r in remaining_rows}
            foreign = sorted(moving_set - owned_ids)
            if foreign:
                raise ValueError(
                    f"face id(s) {foreign} do not belong to cluster {cluster_id}"
                )
            remaining = [r for r in remaining_rows if r["id"] not in moving_set]

            def _centroid(rows: Sequence[sqlite3.Row]) -> bytes | None:
                vecs = [self._decode_embedding(r) for r in rows if r["embedding"] is not None]
                if not vecs:
                    return None
                mean = np.mean(np.stack(vecs), axis=0)
                norm = np.linalg.norm(mean)
                if norm > 1e-12:
                    mean = mean / norm
                return mean.astype(np.float32).tobytes()

            self._conn.execute(
                "UPDATE clusters SET centroid=?, face_count=? WHERE id=?",
                (_centroid(remaining), len(remaining), cluster_id),
            )

            cursor = self._conn.execute(
                """
                INSERT INTO clusters (embed_model, centroid, face_count, person_id, assigned_at, created_at)
                VALUES (?,?,?,NULL,NULL,?)
                """,
                # len(moving_set), not len(moving): face_ids can repeat an id
                # (the caller's mistake, not ours to amplify), and a
                # duplicate can't produce a second row for `UPDATE ... WHERE
                # id IN (...)` to move -- face_count must track rows moved,
                # not ids requested.
                (embed_model, _centroid([r for r in remaining_rows if r["id"] in moving_set]), len(moving_set), now),
            )
            new_id = cursor.lastrowid
            if moving_set:
                placeholders = ",".join("?" * len(moving_set))
                self._conn.execute(
                    f"UPDATE faces SET cluster_id=? WHERE id IN ({placeholders})",
                    (new_id, *moving_set),
                )
            self._conn.commit()
            return new_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Proposals and the confirmation gate
    # ------------------------------------------------------------------

    def record_proposals(self, proposals: Sequence[Proposal]) -> None:
        """Write machine-proposed names. Never touches `clusters.person_id`."""
        with self.lock:
            now = _now_iso()
            for p in proposals:
                self._conn.execute(
                    """
                    INSERT INTO proposals (cluster_id, name, support, total_tagged, score, untagged_photos, proposed_at, decided, decided_at)
                    VALUES (?,?,?,?,?,?,?,NULL,NULL)
                    ON CONFLICT(cluster_id, name) DO UPDATE SET
                        support         = excluded.support,
                        total_tagged    = excluded.total_tagged,
                        score           = excluded.score,
                        untagged_photos = excluded.untagged_photos,
                        proposed_at     = excluded.proposed_at
                    """,
                    (p.cluster_id, normalize(p.name), p.support, p.total_tagged, p.score,
                     p.untagged_photos, now),
                )
            self._conn.commit()

    def proposals_for(self, cluster_id: int) -> list[dict[str, Any]]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT * FROM proposals WHERE cluster_id=? ORDER BY score DESC, name",
                (cluster_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def confirm(self, cluster_id: int, name: str) -> int:
        """Set a cluster's person. Along with `merge`, the only place that does.

        Normalizes *name*, creates the `people` row if this is a new name
        (`source='human'`), and stamps `assigned_at` -- the timestamp
        `iter_pending_sidecars` compares against `face_scan.sidecar_at`.

        Re-validates *cluster_id* here, inside the same lock acquisition that
        does the write -- mirroring `split`'s precedent. `dashboard.people`'s
        wrapper checks existence too, but releases `store.lock` between that
        check and this call; `replace_clusters` can run in the gap and, since
        `clusters.id` is a plain (non-AUTOINCREMENT) INTEGER PRIMARY KEY,
        can make a validated id stop resolving to anything at all by the
        time this runs. Without this check that case is a silent no-op --
        zero rows matched, but a success is still returned, and the
        `INSERT OR IGNORE` above would even leave a spurious unused `people`
        row behind.

        This does NOT close every shape of the race: if the recluster
        happens to recycle the id onto a *different*, currently real
        cluster (rather than leaving it unused), this check sees a row and
        proceeds -- there is no way to tell "the same cluster, still there"
        from "a new cluster that happens to reuse the old id" without a
        caller-supplied fingerprint of what it last saw, which no caller
        here provides. Known, reported residual gap, not an oversight;
        closing it needs cluster identity to survive a recluster (e.g.
        AUTOINCREMENT ids, or a version stamp threaded through the HTTP
        boundary) -- out of scope for this check.
        """
        with self.lock:
            if self._conn.execute(
                "SELECT 1 FROM clusters WHERE id=?", (cluster_id,)
            ).fetchone() is None:
                raise KeyError(f"no such cluster: {cluster_id}")
            clean = normalize(name)
            now = _now_iso()
            self._conn.execute(
                "INSERT OR IGNORE INTO people (name, source, created_at) VALUES (?,?,?)",
                (clean, "human", now),
            )
            person_id = self._conn.execute(
                "SELECT id FROM people WHERE name=?", (clean,)
            ).fetchone()["id"]
            self._conn.execute(
                "UPDATE clusters SET person_id=?, assigned_at=? WHERE id=?",
                (person_id, now, cluster_id),
            )
            self._conn.commit()
            return person_id

    def reject(self, cluster_id: int, name: str) -> None:
        """Record a rejection on the proposal row rather than deleting it."""
        with self.lock:
            now = _now_iso()
            self._conn.execute(
                "UPDATE proposals SET decided='rejected', decided_at=? WHERE cluster_id=? AND name=?",
                (now, cluster_id, normalize(name)),
            )
            self._conn.commit()

    def merge(self, person_id: int, cluster_ids: Sequence[int]) -> None:
        """Point several clusters at one already-confirmed person.

        Along with `confirm`, the only method that writes `clusters.person_id`.

        Re-validates every id in *cluster_ids* here, inside the same lock
        acquisition that does the write -- see `confirm`'s docstring for why
        (same residual gap applies: this catches an id that stopped
        resolving to anything, not one recycled onto different-but-real
        content). Checking all ids up front (rather than relying on the
        `UPDATE ... WHERE id IN (...)` matching fewer rows than requested)
        matters because a *partial* match is the dangerous case within what
        this check *does* cover: some ids in the batch can be stale while
        others are real, and a silent partial `UPDATE` would merge some
        clusters onto *person_id* while quietly dropping the rest with no
        signal to the caller that only part of the batch happened.
        """
        with self.lock:
            if not cluster_ids:
                return
            placeholders = ",".join("?" * len(cluster_ids))
            existing = {
                row["id"]
                for row in self._conn.execute(
                    f"SELECT id FROM clusters WHERE id IN ({placeholders})",
                    tuple(cluster_ids),
                )
            }
            missing = sorted(set(cluster_ids) - existing)
            if missing:
                raise KeyError(f"no such cluster(s): {missing}")
            now = _now_iso()
            self._conn.execute(
                f"UPDATE clusters SET person_id=?, assigned_at=? WHERE id IN ({placeholders})",
                (person_id, now, *cluster_ids),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Sidecar propagation
    # ------------------------------------------------------------------

    def iter_pending_sidecars(self) -> Iterator[tuple[str, list[str]]]:
        """`(digest, [name, ...])` for every photo whose sidecar is behind a
        confirmation -- `assigned_at` newer than `face_scan.sidecar_at`, or no
        sidecar write has ever happened for it."""
        with self.lock:
            rows = list(
                self._conn.execute(
                    """
                    SELECT f.sha256_b64url AS digest, p.name AS name
                      FROM faces f
                      JOIN clusters c ON c.id = f.cluster_id
                      JOIN people   p ON p.id = c.person_id
                      JOIN face_scan s ON s.sha256_b64url = f.sha256_b64url
                     WHERE c.person_id IS NOT NULL
                       AND (s.sidecar_at IS NULL OR c.assigned_at > s.sidecar_at)
                     ORDER BY f.sha256_b64url, p.name
                    """
                )
            )
        grouped: dict[str, list[str]] = {}
        for row in rows:
            names = grouped.setdefault(row["digest"], [])
            if row["name"] not in names:
                names.append(row["name"])
        yield from grouped.items()

    def mark_sidecar_written(self, digest: str, detect_model: str) -> None:
        with self.lock:
            self._conn.execute(
                "UPDATE face_scan SET sidecar_at=? WHERE sha256_b64url=? AND detect_model=?",
                (_now_iso(), digest, detect_model),
            )
            self._conn.commit()

    def set_organized_path(self, digest: str, path: str) -> None:
        with self.lock:
            self._conn.execute(
                """
                INSERT INTO face_organized_paths (sha256_b64url, organized_path)
                VALUES (?,?)
                ON CONFLICT(sha256_b64url) DO UPDATE SET organized_path=excluded.organized_path
                """,
                (digest, path),
            )
            self._conn.commit()

    def organized_path_for(self, digest: str) -> str | None:
        """Resolve *digest*'s organized path from two sources, in this order:

        1. `face_organized_paths` -- an explicit override written by
           `set_organized_path`. This table exists because that method is
           part of this module's required interface even for a digest with
           no `photos` row at all (Task 12's fixtures exercise exactly that),
           so it can't simply defer to `Catalog`.
        2. A read-only fallback to `Catalog`'s own `photos.organized_path`.
           Reading another module's table is not an ownership violation --
           only writing is (see this class's docstring). The fallback exists
           because Task 11's `scan()` reads `organized_path` straight from
           the catalog and never calls `set_organized_path`, so in
           production `face_organized_paths` stays empty forever; without
           this, sidecar propagation would never resolve a path. Do not
           "simplify" this to a single SELECT against `face_organized_paths`
           -- that would silently break production while every test (which
           seeds the table directly) kept passing.

        A `FaceStore` can be opened on a database a `Catalog` has never
        touched, in which case `photos` doesn't exist at all; that is caught
        and treated as "no fallback value" rather than raised.
        """
        with self.lock:
            row = self._conn.execute(
                "SELECT organized_path FROM face_organized_paths WHERE sha256_b64url=?",
                (digest,),
            ).fetchone()
            if row is not None:
                return row["organized_path"]
            try:
                row = self._conn.execute(
                    "SELECT organized_path FROM photos WHERE sha256_b64url=?",
                    (digest,),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
            return row["organized_path"] if row is not None else None

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------

    def add_person(self, name: str, source: str) -> int:
        with self.lock:
            clean = normalize(name)
            self._conn.execute(
                "INSERT OR IGNORE INTO people (name, source, created_at) VALUES (?,?,?)",
                (clean, source, _now_iso()),
            )
            row = self._conn.execute(
                "SELECT id FROM people WHERE name=?", (clean,)
            ).fetchone()
            self._conn.commit()
            return row["id"]

    def known_names(self) -> list[str]:
        with self.lock:
            return [
                row["name"]
                for row in self._conn.execute("SELECT name FROM people ORDER BY name")
            ]

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self.lock:
            def _count(sql: str) -> int:
                return self._conn.execute(sql).fetchone()["n"]

            return {
                "faces": _count("SELECT COUNT(*) AS n FROM faces"),
                "scanned": _count("SELECT COUNT(*) AS n FROM face_scan"),
                "clusters": _count("SELECT COUNT(*) AS n FROM clusters"),
                "people": _count("SELECT COUNT(*) AS n FROM people"),
                "unreviewed": _count(
                    "SELECT COUNT(*) AS n FROM clusters WHERE person_id IS NULL"
                ),
                "singletons": _count(
                    "SELECT COUNT(*) AS n FROM clusters WHERE face_count = 1"
                ),
            }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self.lock:
            self._conn.close()

    def __enter__(self) -> FaceStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
