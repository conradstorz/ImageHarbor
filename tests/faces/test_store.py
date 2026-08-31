"""Face persistence: work queue, clusters, and the confirmation gate."""

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
