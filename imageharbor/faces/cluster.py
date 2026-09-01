"""Group face embeddings into clusters. Pure: no I/O, no model, no database.

Faces are compared against cluster **centroids**, not against each other.
Pairwise over ~150,000 faces is ~11 billion comparisons; against a few thousand
centroids it is one chunked matmul.

Two phases. Anchors -- faces whose person is known from a Google tag -- are
placed first, so the clusters that matter exist before any guessing begins.
Everything else is then assigned to its nearest centroid above threshold.

Phase B is order-dependent, and that is this module's known weakness. It is
contained by placing seeds first, by the caller supplying a deterministic order
(digest order), and by merge/split in the review UI being the actual repair.
Callers must not shuffle: the same input order must give the same output, and
that is pinned by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceVector:
    face_id: int
    embedding: np.ndarray  # L2-normalized
    embed_model: str


@dataclass(frozen=True)
class Seed:
    name: str
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class Cluster:
    face_ids: tuple[int, ...]
    centroid: np.ndarray
    seed_name: str | None = None


class MixedModelError(ValueError):
    """Embeddings from different models were compared.

    Vectors from two models share a coordinate space only by coincidence. A
    comparison across them returns a plausible number that means nothing, which
    is worse than an error, so this raises.
    """


class _Accumulator:
    """A cluster under construction, tracking a running normalized centroid."""

    __slots__ = ("face_ids", "_sum", "seed_name")

    def __init__(self, face_id: int, vector: np.ndarray, seed_name: str | None) -> None:
        self.face_ids = [face_id]
        self._sum = vector.astype(np.float64).copy()
        self.seed_name = seed_name

    def add(self, face_id: int, vector: np.ndarray) -> None:
        self.face_ids.append(face_id)
        self._sum += vector

    @property
    def centroid(self) -> np.ndarray:
        norm = np.linalg.norm(self._sum)
        if norm < 1e-12:  # pragma: no cover - only if vectors cancel exactly
            return self._sum.astype(np.float32)
        return (self._sum / norm).astype(np.float32)

    def freeze(self) -> Cluster:
        return Cluster(
            face_ids=tuple(self.face_ids),
            centroid=self.centroid,
            seed_name=self.seed_name,
        )


def cluster_faces(
    faces: list[FaceVector],
    *,
    threshold: float,
    seeds: list[Seed] | tuple[Seed, ...] = (),
) -> list[Cluster]:
    """Assign faces to clusters. Seeded faces first, then the rest in order."""
    if not faces:
        return []

    models = {f.embed_model for f in faces}
    if len(models) > 1:
        raise MixedModelError(
            f"cannot cluster across embedding models: {sorted(models)}"
        )

    by_id = {f.face_id: f for f in faces}
    accumulators: list[_Accumulator] = []

    def _best_match(
        candidates: list[_Accumulator], embedding: np.ndarray
    ) -> _Accumulator | None:
        """The candidate closest to `embedding`, if it clears `threshold`."""
        if not candidates:
            return None
        centroids = np.stack([a.centroid for a in candidates])
        sims = centroids @ embedding
        best = int(np.argmax(sims))
        return candidates[best] if float(sims[best]) >= threshold else None

    # Phase A: seeds, grouped by name so one name may yield several clusters.
    seeded: set[int] = set()
    for seed in seeds:
        start = len(accumulators)
        for face_id in seed.face_ids:
            face = by_id.get(face_id)
            if face is None or face_id in seeded:
                continue
            seeded.add(face_id)
            # Only compare against this seed's own clusters: two different
            # people must never be merged just because they look alike.
            match = _best_match(accumulators[start:], face.embedding)
            if match is not None:
                match.add(face_id, face.embedding)
                continue
            accumulators.append(_Accumulator(face_id, face.embedding, seed.name))

    # Phase B: everything else, in the caller's order.
    for face in faces:
        if face.face_id in seeded:
            continue
        match = _best_match(accumulators, face.embedding)
        if match is not None:
            match.add(face.face_id, face.embedding)
        else:
            accumulators.append(_Accumulator(face.face_id, face.embedding, None))

    return [a.freeze() for a in accumulators]
