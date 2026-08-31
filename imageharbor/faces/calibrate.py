"""Measure the clustering threshold from the library's own labelled data.

A photo with exactly one detected face and exactly one Google name is an
unambiguous (face, name) pair. This library has 5,670 photos carrying exactly
one name, so the threshold can be *measured* rather than copied out of another
project's README.

Precision here is over pairs: of all anchor pairs at or above a threshold, the
fraction that really are the same person. The chosen threshold is the lowest one
meeting the target, which maximizes recall subject to that precision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    threshold: float
    precision: float
    recall: float
    curve: tuple[tuple[float, float, float], ...]  # (threshold, precision, recall)


def calibrate(
    anchors: Sequence[tuple[str, np.ndarray]],
    *,
    target_precision: float = 0.99,
    max_anchors: int = 4000,
    steps: int = 200,
) -> Calibration:
    """Pick the lowest threshold reaching `target_precision` on anchor pairs."""
    names = [n for n, _ in anchors]
    if len(set(names)) < 2:
        raise ValueError("calibration needs anchors for at least two names")

    if len(anchors) > max_anchors:
        # Deterministic subsample: a seeded generator, so a re-run of calibrate
        # on the same library returns the same threshold.
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(anchors), size=max_anchors, replace=False))
        anchors = [anchors[i] for i in keep]
        names = [n for n, _ in anchors]

    matrix = np.stack([v for _, v in anchors]).astype(np.float32)
    sims = matrix @ matrix.T
    labels = np.array(names)
    same = labels[:, None] == labels[None, :]

    upper = np.triu_indices(len(anchors), k=1)
    pair_sims = sims[upper]
    pair_same = same[upper]

    total_same = int(pair_same.sum())
    if total_same == 0:
        raise ValueError("calibration needs at least one same-name anchor pair")

    curve: list[tuple[float, float, float]] = []
    chosen: tuple[float, float, float] | None = None
    for t in np.linspace(0.0, 1.0, steps, dtype=np.float32):
        selected = pair_sims >= t
        n_selected = int(selected.sum())
        if n_selected == 0:
            continue
        tp = int((selected & pair_same).sum())
        precision = tp / n_selected
        recall = tp / total_same
        curve.append((float(t), precision, recall))
        if chosen is None and precision >= target_precision:
            chosen = (float(t), precision, recall)

    if chosen is None:
        # Nothing reaches the target; return the most precise point measured so
        # the operator sees the real ceiling instead of a fabricated threshold.
        best = max(curve, key=lambda c: (c[1], c[0]))
        chosen = best

    return Calibration(
        threshold=chosen[0],
        precision=chosen[1],
        recall=chosen[2],
        curve=tuple(curve),
    )
