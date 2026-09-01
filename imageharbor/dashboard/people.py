"""The People review API: the dashboard's only path to human-confirmed identity.

Every mutation here (`confirm`, `reject`, `merge`, `split`) is a thin,
validating wrapper around the matching `FaceStore` method. `FaceStore`'s own
docstring guarantees `confirm`/`merge` are the only two places that ever
*assign a new* person to a cluster (a recluster's `replace_clusters` only
ever restores one a human already confirmed) -- this module must never
become a place that assigns one either; it only turns a bad request into
`ValueError` (the same boundary contract
`dashboard/control.py`'s `set_override` uses) and delegates.

`confirm` in particular must not write sidecars: `FaceStore.confirm` only
stamps `clusters.person_id`/`assigned_at`. The next `faces` pass propagates
via `FaceStore.iter_pending_sidecars()` -- a 340-photo cluster's sidecar
merges (minutes, over a CIFS mount) must never happen inside this HTTP
handler.

`review_queue` and `crop_bytes` reach `store._conn`/`store.lock` directly for
queries `FaceStore` has no wrapper method for (per-cluster face ids, person
names/aggregates, a face id's digest). This is the same pattern
`dashboard/stats.py` already uses against `Catalog._conn` (see that module's
docstring, CRITICAL finding #2): no method exists for the query, so it runs
here, guarded by the same lock every `FaceStore` method takes internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from imageharbor.faces.names import case_variants, normalize

# FaceStore is only ever used in type annotations here (never instantiated
# or called), and this module has `from __future__ import annotations`
# above, so the annotation is never evaluated at runtime. Importing it at
# module scope would still require numpy (see imageharbor/faces/store.py's
# module-scope `import numpy as np`) even when the dashboard is running
# without faces enabled -- the exact bug this TYPE_CHECKING guard exists to
# avoid. See CLAUDE.md's "a missing extra degrades to one warning" invariant.
if TYPE_CHECKING:
    from imageharbor.faces.store import FaceStore


def _cluster_exists(store: FaceStore, cluster_id: Any) -> bool:
    # Deliberately unscoped: this validates an operator-supplied primary key
    # against the `clusters` table, not "does this id belong to the current
    # embed_model" -- confirm/reject/merge never carry a model in scope, and
    # a cluster id is unique across the whole table regardless of which
    # model produced it. See `FaceStore.cluster_ids`'s docstring.
    return cluster_id in store.cluster_ids()


# ---------------------------------------------------------------------------
# review_queue
# ---------------------------------------------------------------------------


def review_queue(store: FaceStore, *, include_singletons: bool = False) -> dict[str, Any]:
    """The operator's confirm-todo list, plus the people/case-variant context around it.

    `clusters` holds only *unreviewed* clusters (`person_id IS NULL`) -- a
    confirmed cluster needs no further action, so it is not queue noise to
    hide or show, and is left out entirely (its `person`/proposal state is
    already visible in `people`). Within that unreviewed set, a cluster with
    exactly one face is a singleton and is excluded by default. This
    resolves the design doc's open question ("hidden from review by default
    or merely sorted last?"): hidden, but never silently --
    `singletons_hidden` always reports how many exist, regardless of
    `include_singletons`, so hiding the noise never hides the *fact* of the
    noise.

    Shown clusters are ordered by `face_count` descending (confirming the
    biggest cluster first names the most photos per click), `cluster_id`
    ascending only as a deterministic tiebreaker.
    """
    with store.lock:
        conn = store._conn
        cluster_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, face_count, person_id FROM clusters ORDER BY id"
            )
        ]
        # LEFT JOIN, not JOIN: a person can outlive every cluster that once
        # named them -- `replace_clusters` can leave a confirmed person with
        # zero current clusters (the old cluster's faces didn't land in any
        # new one; see that method's docstring on a recluster that merges or
        # fragments a confirmed cluster's faces). An inner join would
        # silently drop that person from the roster entirely.
        # `COUNT(DISTINCT ...)` over an all-NULL group correctly reports 0,
        # not NULL.
        people_rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT p.id AS person_id, p.name AS name,
                       COUNT(DISTINCT c.id) AS cluster_count,
                       COUNT(DISTINCT f.sha256_b64url) AS photo_count
                FROM people p
                LEFT JOIN clusters c ON c.person_id = p.id
                LEFT JOIN faces f ON f.cluster_id = c.id
                GROUP BY p.id
                ORDER BY p.name
                """
            )
        ]
        # The case-variant "merge" button's whole reason for existing:
        # POST /api/people/merge takes cluster_ids, and without them here the
        # operator has no way to discover which ids to send short of typing
        # them in by hand. A second, separate query rather than folding this
        # into the GROUP BY above -- GROUP_CONCAT would hand back a string
        # that needs re-parsing into ints, and a person with zero clusters
        # (the LEFT JOIN case just above) would GROUP_CONCAT to NULL, which
        # is the same ambiguity this is trying to avoid. Ids only, no
        # per-cluster detail the page doesn't use.
        cluster_ids_by_person: dict[int, list[int]] = {}
        for row in conn.execute(
            "SELECT id, person_id FROM clusters WHERE person_id IS NOT NULL ORDER BY person_id, id"
        ):
            cluster_ids_by_person.setdefault(row["person_id"], []).append(row["id"])
        for row in people_rows:
            row["cluster_ids"] = cluster_ids_by_person.get(row["person_id"], [])

        unreviewed = [r for r in cluster_rows if r["person_id"] is None]
        singletons_hidden = sum(1 for r in unreviewed if r["face_count"] == 1)

        shown = [r for r in unreviewed if include_singletons or r["face_count"] != 1]
        shown.sort(key=lambda r: (-r["face_count"], r["id"]))

        clusters_out = []
        for r in shown:
            sample_face_ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM faces WHERE cluster_id=? ORDER BY id LIMIT 9",
                    (r["id"],),
                )
            ]
            clusters_out.append(
                {
                    "cluster_id": r["id"],
                    "face_count": r["face_count"],
                    # Always None: `shown` is drawn from `unreviewed` only,
                    # by construction (person_id IS NULL). Kept in the shape
                    # so a consumer isn't guessing whether the key exists.
                    "person": None,
                    "sample_face_ids": sample_face_ids,
                    "proposals": [
                        {
                            "name": p["name"],
                            "support": p["support"],
                            "total_tagged": p["total_tagged"],
                            "score": p["score"],
                            "untagged_photos": p["untagged_photos"],
                            "decided": p["decided"],
                        }
                        for p in store.proposals_for(r["id"])
                    ],
                }
            )

    return {
        "clusters": clusters_out,
        "people": people_rows,
        # Flag for whoever builds the UI on this: `case_variants`' grouping
        # key is a per-character-lowercased string (see its docstring), which
        # merges two strings that are visually identical but not the same
        # code points -- the Kelvin sign U+212A groups with plain 'K'. This
        # dict's *values* are always the real, original variant strings (safe
        # to display/act on); it is only the dict *key* that can be a string
        # nobody actually typed. A UI must not treat the key as a name.
        "case_variants": case_variants(store.known_names()),
        "singletons_hidden": singletons_hidden,
        "stats": store.stats(),
    }


# ---------------------------------------------------------------------------
# Mutations -- thin, validating wrappers around FaceStore
# ---------------------------------------------------------------------------


def confirm(store: FaceStore, cluster_id: int, name: str) -> dict[str, Any]:
    """Confirm *cluster_id* as *name*. See the module docstring: this is the
    only place outside `FaceStore` itself that this module calls to write
    identity, and it never does the sidecar propagation work synchronously.

    The `_cluster_exists` check below is the fast, friendly path -- it gives
    a normal bad request a `ValueError` -> HTTP 400 without ever touching
    `store.lock`. It is *not* the real guard: `store.lock` is released
    between that check and `store.confirm`, and a `replace_clusters` recycle
    landing in that window can make the validated id stop resolving to
    anything (or, less commonly, resolve to a different, currently-real
    cluster -- `store.confirm`'s docstring covers why only the first case is
    actually caught) by the time `store.confirm` runs. `store.confirm`
    re-checks existence itself, inside its own lock acquisition, and raises
    `KeyError` when the id no longer resolves -- deliberately left uncaught
    here, same as `split` below leaves `FaceStore.split`'s `KeyError`
    uncaught, so the two mutations fail the same way at the HTTP boundary.
    """
    clean = normalize(name)
    if not clean:
        raise ValueError("name must not be empty")
    if not _cluster_exists(store, cluster_id):
        raise ValueError(f"unknown cluster: {cluster_id!r}")
    person_id = store.confirm(cluster_id, clean)
    return {"cluster_id": cluster_id, "person_id": person_id, "name": clean}


def reject(store: FaceStore, cluster_id: int, name: str) -> dict[str, Any]:
    """Dismiss one proposed name for *cluster_id*. Recorded, never deleted --
    see `FaceStore.reject`."""
    clean = normalize(name)
    if not clean:
        raise ValueError("name must not be empty")
    if not _cluster_exists(store, cluster_id):
        raise ValueError(f"unknown cluster: {cluster_id!r}")
    # `FaceStore.reject`'s UPDATE is a silent no-op for a (cluster_id, name)
    # that never had a proposal -- checked here, not there, because
    # `proposals_for` already exists as the read-side of this exact table
    # and a 200 for "nothing happened" is a lie this HTTP boundary must not
    # tell.
    if not any(p["name"] == clean for p in store.proposals_for(cluster_id)):
        raise ValueError(f"no proposal {clean!r} on cluster {cluster_id!r}")
    store.reject(cluster_id, clean)
    return {"cluster_id": cluster_id, "name": clean, "decided": "rejected"}


def merge(store: FaceStore, person_id: int, cluster_ids: list[int]) -> dict[str, Any]:
    """Point several clusters at one already-confirmed person -- the aging repair.

    Same two-layer shape as `confirm`: the `_cluster_exists` loop below is
    the fast pre-lock check for a normal bad request, not the real guard --
    `store.merge` re-validates every id itself, inside its own lock
    acquisition, and raises `KeyError` (left uncaught here, matching `split`)
    naming any id that stopped resolving to anything in the window between
    this check and that call. See `store.merge`'s docstring for the residual
    gap this does not close.
    """
    if not cluster_ids:
        raise ValueError("cluster_ids must not be empty")
    with store.lock:
        row = store._conn.execute(
            "SELECT 1 FROM people WHERE id=?", (person_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"unknown person: {person_id!r}")
    unknown = [cid for cid in cluster_ids if not _cluster_exists(store, cid)]
    if unknown:
        raise ValueError(f"unknown cluster(s): {unknown!r}")
    store.merge(person_id, cluster_ids)
    return {"person_id": person_id, "cluster_ids": list(cluster_ids)}


def split(store: FaceStore, cluster_id: int, face_ids: list[int]) -> dict[str, Any]:
    """Move *face_ids* out of *cluster_id* into a new, unconfirmed cluster --
    the bad-cluster repair. See `FaceStore.split`.

    Every id in *face_ids* must currently belong to *cluster_id* -- without
    this, a caller (POST /api/people/split with a mismatched cluster_id/
    face_ids pair) can silently strip a face out of a *different*, possibly
    already-confirmed cluster's membership: `FaceStore.split`'s per-id
    `UPDATE ... WHERE id IN (...)` doesn't care which cluster a face id
    currently sits in, and the source cluster it actually came from never
    gets its `face_count` touched, so that count goes stale. Checked here,
    against the HTTP boundary, so a bad request comes back 400 with no
    mutation; `FaceStore.split` carries its own copy of this same check for
    a caller that reaches it directly, but a wrapper-mediated call is
    rejected here first, so only one of the two ever actually raises.
    """
    if not _cluster_exists(store, cluster_id):
        raise ValueError(f"unknown cluster: {cluster_id!r}")
    if not face_ids:
        raise ValueError("face_ids must not be empty")
    with store.lock:
        owned = {
            row["id"]
            for row in store._conn.execute(
                "SELECT id FROM faces WHERE cluster_id=?", (cluster_id,)
            )
        }
    foreign = sorted(set(face_ids) - owned)
    if foreign:
        raise ValueError(
            f"face id(s) {foreign} do not belong to cluster {cluster_id!r}"
        )
    new_cluster_id = store.split(cluster_id, face_ids)
    return {"cluster_id": cluster_id, "new_cluster_id": new_cluster_id}


# ---------------------------------------------------------------------------
# Crop cache
# ---------------------------------------------------------------------------


def crop_bytes(crop_dir: Path, face_id: int, *, store: FaceStore | None = None) -> bytes | None:
    """One face's aligned crop from the on-disk cache, or `None`.

    Never raises: a crop is a derived, deletable-at-any-time cache (see the
    task brief's "crop_bytes returns None for a missing crop"), so a missing
    file, an unscanned/rejected face id, or no *store* at all must degrade
    the page rather than break it.

    The cache is written by `imageharbor/faces/runner.py`'s `_scan_one` as
    `crop_dir/<digest[:2]>/<digest[2:4]>/<digest>-<i>.jpg` -- one file per
    KEPT (non-rejected) face of a photo, named by the photo's digest and that
    face's 0-based rank among the photo's kept faces, *not* by `faces.id`:
    the crop file is written before the DB insert that assigns that id, so
    the id can't be baked into the filename. Resolving a dashboard face id to
    that path therefore needs `store` -- one lookup for the digest, one
    derived index for the rank -- which is why `store` exists as a parameter
    here at all despite the produced interface being framed as
    `(crop_dir, face_id)`: a face id alone is not a path.
    """
    if store is None:
        return None
    with store.lock:
        row = store._conn.execute(
            "SELECT sha256_b64url FROM faces WHERE id=? AND rejected IS NULL",
            (face_id,),
        ).fetchone()
        if row is None:
            return None
        digest = row["sha256_b64url"]
        kept_ids = [
            r["id"]
            for r in store._conn.execute(
                "SELECT id FROM faces WHERE sha256_b64url=? AND rejected IS NULL ORDER BY id",
                (digest,),
            )
        ]
    try:
        index = kept_ids.index(face_id)
    except ValueError:
        return None
    path = Path(crop_dir) / digest[:2] / digest[2:4] / f"{digest}-{index}.jpg"
    try:
        return path.read_bytes()
    except OSError:
        return None
