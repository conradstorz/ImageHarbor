"""Clustering, proposal, and sidecar propagation wired to the store."""

import json
import math

import click
import numpy as np
import pytest
from PIL import Image

from imageharbor.catalog import Catalog
from imageharbor.faces import runner
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


def _det(x=10.0):
    return Detection(x=x, y=10.0, w=50.0, h=50.0, score=0.9,
                     landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0),
                                (22.0, 42.0), (38.0, 42.0)))


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


def test_similar_faces_cluster_and_get_a_proposal(store):
    for i in range(3):
        store.record_scan(f"d{i}", "yunet", [(_det(), _v([1, 0.01 * i, 0]), "auraface")])
    names = {"d0": ["Emma"], "d1": ["Emma"]}

    made = runner.build_clusters(store, names, embed_model="auraface",
                                 threshold=0.5, min_score=0.6, min_support=2)
    assert made == 1
    cid = store.cluster_ids()[0]
    props = store.proposals_for(cid)
    assert props[0]["name"] == "Emma"
    assert props[0]["support"] == 2
    assert props[0]["untagged_photos"] == 1     # d2 is the gap being filled


def test_a_proposal_never_sets_a_person(store):
    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    runner.build_clusters(store, {"d0": ["Emma"]}, embed_model="auraface",
                          threshold=0.5, min_score=0.5, min_support=1)
    cid = store.cluster_ids()[0]
    assert store.person_for_cluster(cid) is None


def test_measure_threshold_uses_single_face_single_name_photos(store):
    rng = np.random.default_rng(0)
    for i in range(12):
        base = np.array([1.0, 0.0, 0.0]) if i < 6 else np.array([0.0, 1.0, 0.0])
        v = base + rng.normal(0, 0.02, 3)
        store.record_scan(f"d{i}", "yunet", [(_det(), _v(v), "auraface")])
    names = {f"d{i}": ["Emma" if i < 6 else "Judy"] for i in range(12)}

    result = runner.measure_threshold(store, names, embed_model="auraface",
                                      target_precision=0.99)
    assert 0.0 < result.threshold < 1.0
    assert result.precision >= 0.99


def test_measure_threshold_needs_at_least_two_names(store):
    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    with pytest.raises(click.ClickException):
        runner.measure_threshold(store, {"d0": ["Emma"]}, embed_model="auraface",
                                 target_precision=0.99)


def test_propagation_writes_a_confirmed_name_into_the_sidecar(store, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    photo = dest / "photo.jpg"
    Image.new("RGB", (50, 50)).save(photo)
    sidecar = photo.with_suffix(".json")
    sidecar.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.set_organized_path("d0", str(photo))
    runner.build_clusters(store, {}, embed_model="auraface",
                          threshold=0.5, min_score=0.6, min_support=1)
    store.confirm(store.cluster_ids()[0], "Emma")

    written = runner.propagate_sidecars(store, dest, "yunet")
    assert written == 1

    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    entry = [p for p in doc["people"] if p["source"] == "imageharbor_faces"]
    assert entry and entry[0]["name"] == "Emma"


def test_propagation_is_idempotent(store, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    photo = dest / "photo.jpg"
    Image.new("RGB", (50, 50)).save(photo)
    photo.with_suffix(".json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.set_organized_path("d0", str(photo))
    runner.build_clusters(store, {}, embed_model="auraface",
                          threshold=0.5, min_score=0.6, min_support=1)
    store.confirm(store.cluster_ids()[0], "Emma")

    runner.propagate_sidecars(store, dest, "yunet")
    first = photo.with_suffix(".json").read_bytes()
    assert runner.propagate_sidecars(store, dest, "yunet") == 0
    assert photo.with_suffix(".json").read_bytes() == first


def test_propagation_advances_confirmed_at_without_growing_history(store, tmp_path):
    # The brief's idempotence test only proves the SECOND call is a no-op
    # (store-level gating short-circuits it entirely). This proves the merge
    # itself -- not just the gate -- is idempotent: force two REAL writes with
    # two different confirmed_at stamps (by re-confirming between them, which
    # bumps clusters.assigned_at past face_scan.sidecar_at again) and check
    # the document is stable except confirmed_at advancing in place, with no
    # `history` list appearing on the entry.
    dest = tmp_path / "dest"
    dest.mkdir()
    photo = dest / "photo.jpg"
    Image.new("RGB", (50, 50)).save(photo)
    photo.with_suffix(".json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.set_organized_path("d0", str(photo))
    runner.build_clusters(store, {}, embed_model="auraface",
                          threshold=0.5, min_score=0.6, min_support=1)
    cid = store.cluster_ids()[0]

    store.confirm(cid, "Emma")
    assert runner.propagate_sidecars(store, dest, "yunet") == 1
    first_doc = json.loads(photo.with_suffix(".json").read_text(encoding="utf-8"))

    store.confirm(cid, "Emma")  # bumps assigned_at -- a genuinely new observation
    assert runner.propagate_sidecars(store, dest, "yunet") == 1
    second_doc = json.loads(photo.with_suffix(".json").read_text(encoding="utf-8"))

    first_entries = [p for p in first_doc["people"] if p["source"] == "imageharbor_faces"]
    second_entries = [p for p in second_doc["people"] if p["source"] == "imageharbor_faces"]
    assert len(first_entries) == 1
    assert len(second_entries) == 1
    assert "history" not in second_entries[0]
    assert first_entries[0]["confirmed_at"] != second_entries[0]["confirmed_at"]
    # Everything but confirmed_at is byte-for-byte the same entry.
    stable = {**second_entries[0], "confirmed_at": first_entries[0]["confirmed_at"]}
    assert stable == first_entries[0]


def test_build_clusters_sorts_face_vectors_before_clustering(store, monkeypatch):
    # Threshold=0.9, three unit vectors at 0 deg / th deg / 2*th deg where
    # th = arccos(0.95): adjacent pairs cos to 0.95 (>= threshold), but the
    # two endpoints cos to 0.805 (< threshold). Processed in ascending
    # face-id order, face 0 and face 1 join first and face 2 ends up alone;
    # processed in descending order, face 2 and face 1 join first and face 0
    # ends up alone. If build_clusters stopped re-sorting the vectors it
    # receives from the store, patching the store to hand them back in
    # descending order would flip which pair clusters together -- this is
    # the guard the brief asked for in case no store-backed test notices,
    # since FaceStore.iter_face_vectors's own `ORDER BY id` already hides the
    # mutation from every store-backed test in this file.
    th = math.degrees(math.acos(0.95))

    def _vec2(deg):
        r = math.radians(deg)
        return _v([math.cos(r), math.sin(r)])

    store.record_scan("d0", "yunet", [(_det(), _vec2(0), "auraface")])
    store.record_scan("d1", "yunet", [(_det(), _vec2(th), "auraface")])
    store.record_scan("d2", "yunet", [(_det(), _vec2(2 * th), "auraface")])

    real_iter = store.iter_face_vectors
    monkeypatch.setattr(
        store, "iter_face_vectors",
        lambda embed_model: reversed(list(real_iter(embed_model))),
    )

    runner.build_clusters(store, {}, embed_model="auraface",
                          threshold=0.9, min_score=0.0, min_support=1)

    groups = store.digests_by_cluster("auraface")
    membership = {digest: frozenset(digests) for digests in groups.values() for digest in digests}
    assert membership["d0"] == membership["d1"]
    assert membership["d2"] != membership["d0"]


def test_google_names_reads_people_tagged_by_google(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    photo = dest / "photo.jpg"
    Image.new("RGB", (10, 10)).save(photo)
    photo.with_suffix(".json").write_text(json.dumps({
        "schema_version": 2,
        "identity": {"sha256_b64url": "digestA"},
        "people": [
            {"name": "Emma", "source": "google_photos_people"},
            {"name": "Judy", "source": "human"},
        ],
    }), encoding="utf-8")

    assert runner.google_names(dest) == {"digestA": ["Emma"]}


def test_google_names_ignores_sidecars_without_a_resolvable_digest(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "stray.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (dest / "corrupt.json").write_text("{not json", encoding="utf-8")
    assert runner.google_names(dest) == {}
