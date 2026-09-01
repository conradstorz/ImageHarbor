"""The People review API."""

import numpy as np
import pytest

from imageharbor.catalog import Catalog
from imageharbor.dashboard import people
from imageharbor.faces import cluster
from imageharbor.faces.attribute import Proposal
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


def _det():
    return Detection(x=0.0, y=0.0, w=50.0, h=50.0, score=0.9,
                     landmarks=((1.0, 1.0), (2.0, 1.0), (1.5, 2.0),
                                (1.0, 3.0), (2.0, 3.0)))


def _v(vals):
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()


def _one_cluster(store, faces=2, digest_prefix="d"):
    ids = []
    for i in range(faces):
        ids += store.record_scan(f"{digest_prefix}{i}", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_v([1, 0, 0]))
    ])
    return store.cluster_ids()[0]


# ---------------------------------------------------------------------------
# review_queue -- prescribed tests
# ---------------------------------------------------------------------------


def test_review_queue_reports_the_payoff_number(store):
    cid = _one_cluster(store)
    store.record_proposals([Proposal(cid, "Emma", 14, 15, 14 / 15, 340)])
    queue = people.review_queue(store)
    entry = queue["clusters"][0]
    assert entry["proposals"][0]["name"] == "Emma"
    assert entry["proposals"][0]["untagged_photos"] == 340


def test_singletons_are_hidden_but_counted(store):
    _one_cluster(store, faces=1)
    queue = people.review_queue(store)
    assert queue["clusters"] == []
    assert queue["singletons_hidden"] == 1

    shown = people.review_queue(store, include_singletons=True)
    assert len(shown["clusters"]) == 1


def test_confirm_assigns_a_person(store):
    cid = _one_cluster(store)
    result = people.confirm(store, cid, "Emma")
    assert result["person_id"] == store.person_for_cluster(cid)


def test_confirm_normalizes_whitespace(store):
    cid = _one_cluster(store)
    people.confirm(store, cid, "  Emma  ")
    assert people.review_queue(store, include_singletons=True)["people"][0]["name"] == "Emma"


def test_confirm_rejects_an_empty_name(store):
    cid = _one_cluster(store)
    with pytest.raises(ValueError, match="name"):
        people.confirm(store, cid, "   ")


def test_confirm_rejects_an_unknown_cluster(store):
    with pytest.raises(ValueError, match="cluster"):
        people.confirm(store, 9999, "Emma")


def test_case_variants_are_surfaced_as_suggestions_not_applied(store):
    cid_a = _one_cluster(store)
    people.confirm(store, cid_a, "pete storz")
    ids = store.record_scan("z", "yunet", [(_det(), _v([0, 1, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_v([0, 1, 0]))
    ])
    people.confirm(store, store.cluster_ids()[-1], "Pete Storz")

    queue = people.review_queue(store, include_singletons=True)
    assert queue["case_variants"] == {"pete storz": ["Pete Storz", "pete storz"]}
    # Both still exist separately. Nothing was merged.
    assert len(queue["people"]) == 2


def test_crop_bytes_returns_none_for_a_missing_crop(tmp_path):
    assert people.crop_bytes(tmp_path, 12345) is None


# ---------------------------------------------------------------------------
# review_queue -- ordering and confirmed-cluster exclusion
# (mutation guard: dropping the face_count-descending sort must fail this)
# ---------------------------------------------------------------------------


def test_review_queue_orders_by_face_count_descending(store):
    # `replace_clusters` rebuilds *every* cluster for one embed_model in a
    # single call -- calling `_one_cluster` three times in a row would wipe
    # the earlier two, not build three coexisting clusters. All three must
    # be handed to one `replace_clusters` call.
    def _faces(prefix, n):
        ids = []
        for i in range(n):
            ids += store.record_scan(f"{prefix}{i}", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
        return ids

    small_ids = _faces("a", 2)
    big_ids = _faces("b", 5)
    medium_ids = _faces("c", 3)
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(small_ids), centroid=_v([1, 0, 0])),
        cluster.Cluster(face_ids=tuple(big_ids), centroid=_v([1, 0, 0])),
        cluster.Cluster(face_ids=tuple(medium_ids), centroid=_v([1, 0, 0])),
    ])

    queue = people.review_queue(store)
    assert [c["face_count"] for c in queue["clusters"]] == [5, 3, 2]


def test_confirmed_clusters_are_not_in_the_queue(store):
    cid = _one_cluster(store)
    people.confirm(store, cid, "Emma")
    queue = people.review_queue(store, include_singletons=True)
    assert queue["clusters"] == []
    # But the person is still visible via the `people` roster.
    assert queue["people"][0]["name"] == "Emma"
    assert queue["people"][0]["cluster_count"] == 1


def test_sample_face_ids_capped_at_nine(store):
    cid = _one_cluster(store, faces=12)
    queue = people.review_queue(store, include_singletons=True)
    entry = next(c for c in queue["clusters"] if c["cluster_id"] == cid)
    assert len(entry["sample_face_ids"]) == 9
    assert entry["sample_face_ids"] == sorted(entry["sample_face_ids"])


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


def test_reject_marks_the_proposal_decided(store):
    cid = _one_cluster(store)
    store.record_proposals([Proposal(cid, "Emma", 14, 15, 14 / 15, 340)])
    people.reject(store, cid, "Emma")
    proposals = store.proposals_for(cid)
    assert proposals[0]["decided"] == "rejected"


def test_reject_rejects_an_empty_name(store):
    cid = _one_cluster(store)
    with pytest.raises(ValueError, match="name"):
        people.reject(store, cid, "  ")


def test_reject_rejects_an_unknown_cluster(store):
    with pytest.raises(ValueError, match="cluster"):
        people.reject(store, 9999, "Emma")


def test_reject_rejects_a_name_with_no_matching_proposal(store):
    # A known cluster but a name that was never proposed on it -- the
    # store-level UPDATE is a silent no-op here, so the wrapper must be the
    # one to notice nothing happened.
    cid = _one_cluster(store)
    store.record_proposals([Proposal(cid, "Emma", 14, 15, 14 / 15, 340)])
    with pytest.raises(ValueError, match="Nobody"):
        people.reject(store, cid, "Nobody")
    # The real proposal is untouched by the failed call.
    assert store.proposals_for(cid)[0]["decided"] is None


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_points_clusters_at_one_person(store):
    # Both clusters must come from the same `replace_clusters` call -- see
    # the note in test_review_queue_orders_by_face_count_descending.
    ids_a = store.record_scan("a0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    ids_b = store.record_scan("b0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids_a), centroid=_v([1, 0, 0])),
        cluster.Cluster(face_ids=tuple(ids_b), centroid=_v([1, 0, 0])),
    ])
    cid_a, cid_b = store.cluster_ids()
    person_id = store.confirm(cid_a, "Emma")

    result = people.merge(store, person_id, [cid_b])
    assert result["person_id"] == person_id
    assert store.person_for_cluster(cid_b) == person_id


def test_merge_rejects_empty_cluster_ids(store):
    cid_a = _one_cluster(store)
    person_id = store.confirm(cid_a, "Emma")
    with pytest.raises(ValueError, match="cluster_ids"):
        people.merge(store, person_id, [])


def test_merge_rejects_an_unknown_person(store):
    cid_b = _one_cluster(store)
    with pytest.raises(ValueError, match="person"):
        people.merge(store, 9999, [cid_b])


def test_merge_rejects_an_unknown_cluster(store):
    cid_a = _one_cluster(store)
    person_id = store.confirm(cid_a, "Emma")
    with pytest.raises(ValueError, match="cluster"):
        people.merge(store, person_id, [9999])


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_moves_faces_into_a_new_cluster(store):
    cid = _one_cluster(store, faces=3)
    face_ids = sorted(
        r["id"] for r in store._conn.execute(
            "SELECT id FROM faces WHERE cluster_id=?", (cid,)
        )
    )
    result = people.split(store, cid, [face_ids[-1]])
    new_id = result["new_cluster_id"]
    assert new_id != cid
    assert store.person_for_cluster(new_id) is None


def test_split_rejects_an_unknown_cluster(store):
    with pytest.raises(ValueError, match="cluster"):
        people.split(store, 9999, [1])


def test_split_rejects_empty_face_ids(store):
    cid = _one_cluster(store)
    with pytest.raises(ValueError, match="face_ids"):
        people.split(store, cid, [])


def _two_clusters(store):
    """Two coexisting, unconfirmed clusters -- both from one `replace_clusters`
    call, per the note in test_review_queue_orders_by_face_count_descending."""
    ids_a = store.record_scan("a0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    ids_b = store.record_scan("b0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids_a), centroid=_v([1, 0, 0])),
        cluster.Cluster(face_ids=tuple(ids_b), centroid=_v([1, 0, 0])),
    ])
    cid_a, cid_b = store.cluster_ids()
    return cid_a, cid_b, ids_a[0]


def test_split_rejects_a_face_from_another_cluster(store):
    cid_a, cid_b, foreign_face = _two_clusters(store)

    with pytest.raises(ValueError, match=str(foreign_face)):
        people.split(store, cid_b, [foreign_face])

    # Nothing mutated: the face is still where it started, and cluster A's
    # face_count still matches its real membership.
    row = store._conn.execute(
        "SELECT cluster_id FROM faces WHERE id=?", (foreign_face,)
    ).fetchone()
    assert row["cluster_id"] == cid_a
    assert store._conn.execute(
        "SELECT face_count FROM clusters WHERE id=?", (cid_a,)
    ).fetchone()["face_count"] == 1


def test_split_rejects_a_face_from_a_confirmed_cluster(store):
    # The reviewer's exact repro: confirm cluster A as "Emma", then try to
    # split one of A's faces out of unrelated cluster B.
    cid_a, cid_b, stolen_face = _two_clusters(store)
    people.confirm(store, cid_a, "Emma")

    with pytest.raises(ValueError, match=str(stolen_face)):
        people.split(store, cid_b, [stolen_face])

    assert store.person_for_cluster(cid_a) is not None
    row = store._conn.execute(
        "SELECT cluster_id FROM faces WHERE id=?", (stolen_face,)
    ).fetchone()
    assert row["cluster_id"] == cid_a
    assert store._conn.execute(
        "SELECT face_count FROM clusters WHERE id=?", (cid_a,)
    ).fetchone()["face_count"] == 1


def test_split_face_count_matches_real_membership_on_both_clusters(store):
    cid = _one_cluster(store, faces=3)
    face_ids = sorted(
        r["id"] for r in store._conn.execute(
            "SELECT id FROM faces WHERE cluster_id=?", (cid,)
        )
    )
    result = people.split(store, cid, [face_ids[-1]])
    new_id = result["new_cluster_id"]

    def _real_count(cluster_id):
        return store._conn.execute(
            "SELECT COUNT(*) AS n FROM faces WHERE cluster_id=?", (cluster_id,)
        ).fetchone()["n"]

    src_face_count = store._conn.execute(
        "SELECT face_count FROM clusters WHERE id=?", (cid,)
    ).fetchone()["face_count"]
    new_face_count = store._conn.execute(
        "SELECT face_count FROM clusters WHERE id=?", (new_id,)
    ).fetchone()["face_count"]
    assert src_face_count == _real_count(cid) == 2
    assert new_face_count == _real_count(new_id) == 1


# ---------------------------------------------------------------------------
# crop_bytes -- real file resolution
# ---------------------------------------------------------------------------


def test_crop_bytes_reads_a_real_crop(store, tmp_path):
    digest = "abcdef0123456789"
    ids = store.record_scan(digest, "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    face_id = ids[0]

    photo_dir = tmp_path / digest[:2] / digest[2:4]
    photo_dir.mkdir(parents=True)
    (photo_dir / f"{digest}-0.jpg").write_bytes(b"fake-jpeg-bytes")

    assert people.crop_bytes(tmp_path, face_id, store=store) == b"fake-jpeg-bytes"


def test_crop_bytes_returns_none_for_a_rejected_face(store, tmp_path):
    ids = store.record_scan(
        "rejdigest", "yunet", [(_det(), None, None, "too_small")]
    )
    assert people.crop_bytes(tmp_path, ids[0], store=store) is None


def test_crop_bytes_returns_none_for_an_unknown_face_id_with_a_store(store, tmp_path):
    assert people.crop_bytes(tmp_path, 999999, store=store) is None


def test_crop_bytes_returns_none_when_the_file_is_missing_but_the_face_exists(store, tmp_path):
    digest = "ffeeddccbbaa0011"
    ids = store.record_scan(digest, "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    # No file written under tmp_path -- the DB row exists, the cache doesn't.
    assert people.crop_bytes(tmp_path, ids[0], store=store) is None
