"""Clustering on synthetic vectors. No model, no database."""

import numpy as np
import pytest

from imageharbor.faces import cluster


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _fv(face_id, v, model="auraface"):
    return cluster.FaceVector(face_id=face_id, embedding=_unit(v), embed_model=model)


def test_identical_vectors_form_one_cluster():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [1, 0, 0]), _fv(3, [1, 0, 0])]
    out = cluster.cluster_faces(faces, threshold=0.5)
    assert len(out) == 1
    assert out[0].face_ids == (1, 2, 3)


def test_orthogonal_vectors_form_separate_clusters():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [0, 1, 0]), _fv(3, [0, 0, 1])]
    out = cluster.cluster_faces(faces, threshold=0.5)
    assert len(out) == 3


def test_threshold_boundary_is_inclusive():
    # cos 60 degrees = 0.5 exactly.
    a, b = _fv(1, [1, 0, 0]), _fv(2, [0.5, np.sqrt(3) / 2, 0])
    assert len(cluster.cluster_faces([a, b], threshold=0.5)) == 1
    assert len(cluster.cluster_faces([a, b], threshold=0.51)) == 2


def test_mixing_embed_models_raises():
    faces = [_fv(1, [1, 0, 0], "auraface"), _fv(2, [1, 0, 0], "sface")]
    with pytest.raises(cluster.MixedModelError, match="auraface"):
        cluster.cluster_faces(faces, threshold=0.5)


def test_seeds_are_placed_before_unseeded_faces():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [1, 0, 0]), _fv(3, [0, 1, 0])]
    seeds = [cluster.Seed(name="Emma", face_ids=(3,))]
    out = cluster.cluster_faces(faces, threshold=0.5, seeds=seeds)
    assert out[0].seed_name == "Emma"
    assert out[0].face_ids == (3,)
    assert out[1].seed_name is None


def test_one_seed_name_may_produce_several_clusters():
    # Aging: the same person, two life stages, not mutually similar.
    faces = [_fv(1, [1, 0, 0]), _fv(2, [0, 1, 0])]
    seeds = [cluster.Seed(name="Emma", face_ids=(1, 2))]
    out = cluster.cluster_faces(faces, threshold=0.9, seeds=seeds)
    assert len(out) == 2
    assert {c.seed_name for c in out} == {"Emma"}


def test_seed_isolation_prevents_merging_different_people():
    # The invariant this module exists for: two different people must never
    # merge just because their embeddings are close. Phase A restricts each
    # seed's comparisons to that seed's own clusters (`accumulators[start:]`);
    # mutating that to search all accumulators would merge Judy into Emma's
    # cluster here, since their embeddings are identical.
    faces = [_fv(1, [1, 0, 0]), _fv(2, [1, 0, 0])]
    seeds = [
        cluster.Seed(name="Emma", face_ids=(1,)),
        cluster.Seed(name="Judy", face_ids=(2,)),
    ]
    out = cluster.cluster_faces(faces, threshold=0.5, seeds=seeds)
    assert len(out) == 2
    by_name = {c.seed_name: c.face_ids for c in out}
    assert by_name == {"Emma": (1,), "Judy": (2,)}


def test_is_deterministic_for_the_same_input_order():
    faces = [_fv(i, [np.cos(i), np.sin(i), 0]) for i in range(20)]
    a = cluster.cluster_faces(faces, threshold=0.8)
    b = cluster.cluster_faces(faces, threshold=0.8)
    assert [c.face_ids for c in a] == [c.face_ids for c in b]


def test_centroids_are_unit_length():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [0.9, 0.1, 0])]
    for c in cluster.cluster_faces(faces, threshold=0.5):
        assert np.linalg.norm(c.centroid) == pytest.approx(1.0, abs=1e-5)


def test_empty_input_returns_no_clusters():
    assert cluster.cluster_faces([], threshold=0.5) == []
