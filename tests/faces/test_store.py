"""Face persistence: work queue, clusters, and the confirmation gate."""

import logging

import numpy as np
import pytest

from imageharbor.catalog import Catalog
from imageharbor.faces import cluster
from imageharbor.faces.attribute import Proposal
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()


def _det(x=0.0, score=0.9):
    return Detection(
        x=x, y=0.0, w=50.0, h=50.0, score=score,
        landmarks=((1.0, 1.0), (2.0, 1.0), (1.5, 2.0), (1.0, 3.0), (2.0, 3.0)),
    )


def _vec(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_recording_a_scan_makes_it_scanned(store):
    assert not store.is_scanned("digestA", "yunet")
    store.record_scan("digestA", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    assert store.is_scanned("digestA", "yunet")


def test_rescanning_the_same_photo_is_a_no_op(store):
    ids_a = store.record_scan("digestA", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    ids_b = store.record_scan("digestA", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    assert ids_a == ids_b
    assert store.stats()["faces"] == 1


def test_a_photo_with_no_faces_is_still_recorded_as_scanned(store):
    store.record_scan("empty", "yunet", [])
    assert store.is_scanned("empty", "yunet")
    assert store.stats()["faces"] == 0


def test_scan_is_keyed_on_the_detector(store):
    store.record_scan("digestA", "yunet", [])
    assert store.is_scanned("digestA", "yunet")
    assert not store.is_scanned("digestA", "scrfd")


def test_face_vectors_round_trip(store):
    store.record_scan("d", "yunet", [(_det(), _vec([0.6, 0.8, 0.0]), "auraface")])
    vectors = list(store.iter_face_vectors("auraface"))
    assert len(vectors) == 1
    assert np.allclose(vectors[0].embedding, _vec([0.6, 0.8, 0.0]), atol=1e-6)
    assert vectors[0].embed_model == "auraface"


def test_face_vectors_are_filtered_by_model(store):
    store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    assert list(store.iter_face_vectors("sface")) == []


def test_confirm_is_the_only_thing_that_sets_person_id(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]), seed_name=None)
    ])
    cid = store.cluster_ids()[0]
    store.record_proposals([Proposal(cid, "Emma", 3, 3, 1.0, 10)])
    assert store.person_for_cluster(cid) is None      # a proposal asserts nothing

    person_id = store.confirm(cid, "Emma")
    assert store.person_for_cluster(cid) == person_id


def test_rejecting_a_proposal_records_it_rather_than_deleting(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]), seed_name=None)
    ])
    cid = store.cluster_ids()[0]
    store.record_proposals([Proposal(cid, "Emma", 3, 3, 1.0, 10)])
    store.reject(cid, "Emma")
    assert store.proposals_for(cid)[0]["decided"] == "rejected"


def test_merge_points_several_clusters_at_one_person(store):
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
    ])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(ids[1],), centroid=_vec([0, 1, 0])),
    ])
    a, b = store.cluster_ids()
    person_id = store.confirm(a, "Emma")
    store.merge(person_id, [b])
    assert store.person_for_cluster(b) == person_id


def test_confirm_raises_for_a_cluster_id_recycled_away_by_a_racing_recluster(store):
    # The reviewer's finding: dashboard.people.confirm's own `_cluster_exists`
    # check runs under a *separate* `with store.lock:` block that releases
    # before `store.confirm` is called -- a `replace_clusters` (whole-library
    # recluster) can run in that gap. A recluster that produces fewer
    # clusters than before doesn't just drop the *content* of a validated
    # id -- it can make the id itself stop existing (this holds regardless
    # of AUTOINCREMENT: a shrinking id space just means nothing ever gets
    # inserted at that id again). Simulated here directly (no threads
    # needed): seed 3 single-face clusters, confirm
    # one of them elsewhere first (so a real `person_id` exists, matching how
    # `merge` is exercised below), then race a second `replace_clusters` that
    # excludes the third face's photo entirely -- its old cluster id is
    # provably gone by the time the stale `confirm(..)` call lands.
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
        (_det(x=400), _vec([0, 0, 1]), "auraface"),
    ])
    fx, fy, fz = ids
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(fx,), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(fy,), centroid=_vec([0, 1, 0])),
        cluster.Cluster(face_ids=(fz,), centroid=_vec([0, 0, 1])),
    ])
    _rx, _ry, rz = store.cluster_ids()
    # rz is the id the (imagined) HTTP wrapper just validated as existing.

    # The race: a whole-library recluster completes before the confirm call
    # actually lands, and this round doesn't reproduce fz's cluster at all
    # (e.g. its photo dropped out of the run) -- rz is now provably gone.
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(fx,), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(fy,), centroid=_vec([0, 1, 0])),
    ])
    assert rz not in store.cluster_ids()

    with pytest.raises(KeyError, match=str(rz)):
        store.confirm(rz, "Emma")

    # Nothing leaked from the failed attempt: no stray "Emma" person row
    # (the pre-fix code created one via INSERT OR IGNORE before ever
    # checking whether the cluster existed), and no cluster was mislabeled.
    assert store._conn.execute(
        "SELECT 1 FROM people WHERE name='Emma'"
    ).fetchone() is None
    for cid in store.cluster_ids():
        assert store.person_for_cluster(cid) is None


def test_merge_raises_naming_the_stale_id_and_leaves_the_valid_one_untouched(store):
    # Same race as confirm's, but for `merge`'s per-id loop: brief item 2's
    # exact worry is a *partial* write -- a stale id silently matching zero
    # rows while a real id in the same call matches and gets mutated, with
    # nothing telling the caller only half the batch actually happened.
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
        (_det(x=400), _vec([0, 0, 1]), "auraface"),
    ])
    fx, fy, fz = ids
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(fx,), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(fy,), centroid=_vec([0, 1, 0])),
        cluster.Cluster(face_ids=(fz,), centroid=_vec([0, 0, 1])),
    ])
    rx, _ry, rz = store.cluster_ids()
    person_id = store.confirm(rx, "Judy")  # a real, already-confirmed person

    # Race: fz's cluster (rz) drops out of the next recluster round entirely,
    # while fy's survives (under a freshly allocated, but currently valid,
    # id) -- exactly the "one id real, one id stale" batch item 2 describes.
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(fx,), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(fy,), centroid=_vec([0, 1, 0])),
    ])
    current_ids = store.cluster_ids()
    assert rz not in current_ids
    valid_cid = next(
        cid for cid in current_ids
        if store._conn.execute(
            "SELECT 1 FROM faces WHERE cluster_id=? AND id=?", (cid, fy)
        ).fetchone() is not None
    )

    with pytest.raises(KeyError, match=str(rz)):
        store.merge(person_id, [valid_cid, rz])

    # The whole batch must fail together -- valid_cid must not have been
    # quietly merged while rz was silently skipped.
    assert store.person_for_cluster(valid_cid) is None


def test_confirm_after_a_same_count_recluster_does_not_write_the_wrong_identity(store):
    # This is the review finding's own worked example, reproduced exactly:
    #
    #   cluster->faces before: {1: [1], 2: [2], 3: [3]}
    #   confirm(cluster_id=3, "Emma") -> HTTP 200
    #   cluster->faces after : {1: [3], 2: [2], 3: [1]}
    #   operator intended to name faces: [3]
    #   faces actually named           : [1]
    #   *** WRONG IDENTITY WRITTEN ***
    #
    # fix-task-2-report.md's "IMPORTANT" section reproduced this directly
    # against fabdc12's fix (an inside-the-lock existence check alone) and
    # confirmed it still happens: a recluster that keeps the *same* cluster
    # count just permutes which content lands on which id, so the recycled
    # id still exists -- the existence check is not enough to catch it.
    #
    # The `clusters.id INTEGER PRIMARY KEY AUTOINCREMENT` fix changes what
    # the *recycled* id actually is: SQLite never reuses a rowid for the
    # life of the table once AUTOINCREMENT is set, so a same-count recluster
    # gets brand-new ids (here, 4/5/6) instead of falling back onto the
    # freed 1/2/3. The stale id the operator captured (3) then resolves to
    # nothing at all, and `confirm` (per fabdc12) raises `KeyError` instead
    # of silently writing onto whatever now holds id 3.
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
        (_det(x=400), _vec([0, 0, 1]), "auraface"),
    ])
    face1, face2, face3 = ids
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(face1,), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(face2,), centroid=_vec([0, 1, 0])),
        cluster.Cluster(face_ids=(face3,), centroid=_vec([0, 0, 1])),
    ])
    cluster1, cluster2, cluster3 = store.cluster_ids()
    assert (cluster1, cluster2, cluster3) == (1, 2, 3)  # first-ever inserts

    # The operator, looking at the review queue, decides to confirm the
    # cluster holding face3 -- id 3 -- as "Emma".
    target_cluster_id = cluster3
    assert store._conn.execute(
        "SELECT 1 FROM faces WHERE cluster_id=? AND id=?", (target_cluster_id, face3)
    ).fetchone() is not None

    # The race: a whole-library recluster lands before the confirm call
    # does, permuting content across the *same* three singleton clusters
    # (face3 first this time, then face2, then face1) -- same count as
    # before, so under the old plain INTEGER PRIMARY KEY this would recycle
    # ids 1/2/3 right back, with id 3 now landing on face1 instead of face3.
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(face3,), centroid=_vec([0, 0, 1])),
        cluster.Cluster(face_ids=(face2,), centroid=_vec([0, 1, 0])),
        cluster.Cluster(face_ids=(face1,), centroid=_vec([1, 0, 0])),
    ])
    new_ids = store.cluster_ids()
    # AUTOINCREMENT: brand-new ids, never falling back onto the freed 1/2/3.
    assert target_cluster_id not in new_ids
    assert new_ids == [4, 5, 6]

    # The stale confirm call the operator's earlier click queued up: it
    # still names id 3, which no longer resolves to anything.
    with pytest.raises(KeyError, match=str(target_cluster_id)):
        store.confirm(target_cluster_id, "Emma")

    # Nothing was silently written onto whatever cluster now holds face1 --
    # the wrong-identity write the review finding demonstrated must not
    # have happened.
    for cid in new_ids:
        assert store.person_for_cluster(cid) is None
    assert store._conn.execute(
        "SELECT 1 FROM people WHERE name='Emma'"
    ).fetchone() is None


def test_split_rejects_a_face_id_not_in_the_cluster(store):
    # Defense-in-depth: FaceStore.split is the layer that actually mutates,
    # so it must refuse even when a caller bypasses dashboard.people's
    # wrapper-level validation.
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
    ])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(ids[1],), centroid=_vec([0, 1, 0])),
    ])
    cid_a, cid_b = store.cluster_ids()
    foreign_face = ids[0]  # belongs to cid_a, not cid_b

    with pytest.raises(ValueError, match=str(foreign_face)):
        store.split(cid_b, [foreign_face])

    # Nothing mutated.
    row = store._conn.execute(
        "SELECT cluster_id FROM faces WHERE id=?", (foreign_face,)
    ).fetchone()
    assert row["cluster_id"] == cid_a
    assert store._conn.execute(
        "SELECT face_count FROM clusters WHERE id=?", (cid_a,)
    ).fetchone()["face_count"] == 1


def test_split_with_duplicate_face_ids_does_not_inflate_the_new_face_count(store):
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([1, 0, 0]), "auraface"),
    ])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0])),
    ])
    cid = store.cluster_ids()[0]
    dup_id = ids[-1]

    new_id = store.split(cid, [dup_id, dup_id])

    real_count = store._conn.execute(
        "SELECT COUNT(*) AS n FROM faces WHERE cluster_id=?", (new_id,)
    ).fetchone()["n"]
    stored_face_count = store._conn.execute(
        "SELECT face_count FROM clusters WHERE id=?", (new_id,)
    ).fetchone()["face_count"]
    assert real_count == 1
    assert stored_face_count == real_count


def test_replacing_clusters_preserves_confirmed_people(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))
    ])
    cid = store.cluster_ids()[0]
    person_id = store.confirm(cid, "Emma")

    # A recluster must not silently discard a human decision.
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))
    ])
    new_cid = store.cluster_ids()[0]
    assert store.person_for_cluster(new_cid) == person_id


def test_recluster_merging_two_confirmed_people_leaves_the_cluster_unconfirmed(store, caplog):
    # Emma's confirmed cluster and Judy's confirmed cluster get reclustered
    # into a single new cluster containing both their faces. Picking either
    # person would manufacture a confirmation nobody made -- see Finding 1.
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
    ])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(ids[1],), centroid=_vec([0, 1, 0])),
    ])
    a, b = store.cluster_ids()
    store.confirm(a, "Emma")
    store.confirm(b, "Judy")

    with caplog.at_level(logging.WARNING):
        store.replace_clusters("auraface", [
            cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 1, 0])),
        ])

    new_cid = store.cluster_ids()[0]
    assert store.person_for_cluster(new_cid) is None
    assert any(
        "Emma" in record.getMessage() and "Judy" in record.getMessage()
        for record in caplog.records
    )


def test_recluster_splitting_one_confirmed_cluster_both_fragments_inherit_person(store):
    # The opposite case: one confirmed cluster fragments into two new
    # clusters. This only ever duplicates a real confirmation, so both
    # fragments must keep it -- this must keep working, not just Finding 1's
    # merge case.
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0.99, 0.01, 0]), "auraface"),
    ])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0])),
    ])
    cid = store.cluster_ids()[0]
    person_id = store.confirm(cid, "Emma")

    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(ids[1],), centroid=_vec([0.99, 0.01, 0])),
    ])

    new_ids = store.cluster_ids()
    assert len(new_ids) == 2
    assert store.person_for_cluster(new_ids[0]) == person_id
    assert store.person_for_cluster(new_ids[1]) == person_id


def test_pending_sidecars_lists_a_photo_after_confirmation(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))
    ])
    cid = store.cluster_ids()[0]
    store.confirm(cid, "Emma")

    pending = dict(store.iter_pending_sidecars())
    assert pending == {"d": ["Emma"]}

    store.mark_sidecar_written("d", "yunet")
    assert dict(store.iter_pending_sidecars()) == {}


def test_cluster_ids_is_unscoped_by_default(store):
    # `dashboard/people.py`'s `_cluster_exists` relies on this: it validates
    # an operator-supplied cluster id against the primary key, regardless of
    # which embed_model produced it.
    ids_a = store.record_scan("a", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    ids_b = store.record_scan("b", "yunet", [(_det(x=200), _vec([0, 1, 0]), "sface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids_a), centroid=_vec([1, 0, 0]))
    ])
    store.replace_clusters("sface", [
        cluster.Cluster(face_ids=tuple(ids_b), centroid=_vec([0, 1, 0]))
    ])
    assert len(store.cluster_ids()) == 2


def test_cluster_ids_scoped_by_embed_model_excludes_other_models(store):
    # Cross-model isolation, matching how `unclustered_face_count` and
    # `digests_by_cluster` are already scoped: a cluster built for one
    # embed_model must never appear in another model's `cluster_ids(...)`.
    # This is the exact bug the fix-round finding described -- watcher.py's
    # recluster gate calling the unscoped form let a cluster left behind by
    # a since-abandoned embed_model mask "no clusters yet" for the model
    # actually in use.
    ids_a = store.record_scan("a", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    ids_b = store.record_scan("b", "yunet", [(_det(x=200), _vec([0, 1, 0]), "sface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids_a), centroid=_vec([1, 0, 0]))
    ])
    store.replace_clusters("sface", [
        cluster.Cluster(face_ids=tuple(ids_b), centroid=_vec([0, 1, 0]))
    ])

    auraface_ids = store.cluster_ids("auraface")
    sface_ids = store.cluster_ids("sface")

    assert len(auraface_ids) == 1
    assert len(sface_ids) == 1
    assert auraface_ids != sface_ids
    # The discriminating assertion: without the WHERE clause, both scoped
    # calls degrade to the same unscoped, 2-element list.
    assert set(auraface_ids).isdisjoint(sface_ids)
    assert store.cluster_ids() == sorted(auraface_ids + sface_ids)


def test_anchors_are_single_face_single_name_photos(store):
    store.record_scan("one", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.record_scan("two", "yunet", [
        (_det(x=0), _vec([0, 1, 0]), "auraface"),
        (_det(x=200), _vec([0, 0, 1]), "auraface"),
    ])
    anchors = store.anchors("auraface", {"one": ["Emma"], "two": ["Judy"]})
    assert [n for n, _ in anchors] == ["Emma"]  # "two" has two faces, so it is not an anchor


def test_organized_path_prefers_explicit_override_over_photos_row(tmp_path):
    db = tmp_path / "catalog.db"
    cat = Catalog(db)
    cat.upsert(sha256_b64url="d", original_path="/orig/d.jpg", organized_path="/from/photos.jpg")
    cat.close()
    store = FaceStore(db)
    store.set_organized_path("d", "/from/override.jpg")
    assert store.organized_path_for("d") == "/from/override.jpg"
    store.close()


def test_organized_path_falls_back_to_photos_row_when_no_override_exists(tmp_path):
    db = tmp_path / "catalog.db"
    cat = Catalog(db)
    cat.upsert(sha256_b64url="d", original_path="/orig/d.jpg", organized_path="/from/photos.jpg")
    cat.close()
    store = FaceStore(db)
    assert store.organized_path_for("d") == "/from/photos.jpg"
    store.close()


def test_organized_path_is_none_when_neither_source_has_it(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    store = FaceStore(db)
    assert store.organized_path_for("missing") is None
    store.close()


def test_organized_path_is_none_when_photos_table_does_not_exist(tmp_path):
    # A FaceStore opened on a database a Catalog has never touched: no
    # `photos` table at all. Must not raise.
    db = tmp_path / "faces_only.db"
    store = FaceStore(db)
    assert store.organized_path_for("d") is None
    store.close()
