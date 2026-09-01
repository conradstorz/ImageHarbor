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
