"""Threshold calibration from labelled anchors."""

import numpy as np
import pytest

from imageharbor.faces import calibrate


def _anchors(rng, names=("Emma", "Judy", "Pete"), per=12, spread=0.02):
    out = []
    # A one-hot basis needs at least one dimension per name; 8 is only a floor
    # so the default 3-name tests keep their original (unchanged) vectors.
    dim = max(len(names), 8)
    for i, name in enumerate(names):
        base = np.zeros(dim, dtype=np.float32)
        base[i] = 1.0
        for _ in range(per):
            v = base + rng.normal(0, spread, dim).astype(np.float32)
            out.append((name, v / np.linalg.norm(v)))
    return out


def test_well_separated_anchors_yield_a_high_precision_threshold():
    rng = np.random.default_rng(0)
    result = calibrate.calibrate(_anchors(rng), target_precision=0.99)
    assert 0.0 < result.threshold < 1.0
    assert result.precision >= 0.99
    assert result.recall > 0.5


def test_the_threshold_separates_the_synthetic_groups():
    rng = np.random.default_rng(1)
    anchors = _anchors(rng)
    result = calibrate.calibrate(anchors, target_precision=0.99)
    same = np.dot(anchors[0][1], anchors[1][1])
    diff = np.dot(anchors[0][1], anchors[-1][1])
    assert diff < result.threshold <= same


def test_curve_is_returned_and_ordered():
    rng = np.random.default_rng(2)
    result = calibrate.calibrate(_anchors(rng), target_precision=0.99)
    thresholds = [t for t, _, _ in result.curve]
    assert thresholds == sorted(thresholds)
    assert len(result.curve) > 1


def test_too_few_anchors_raises():
    with pytest.raises(ValueError, match="at least two names"):
        calibrate.calibrate([("Emma", np.array([1.0, 0.0], dtype=np.float32))])


def test_is_deterministic_under_subsampling():
    rng = np.random.default_rng(3)
    anchors = _anchors(rng, names=tuple(f"P{i}" for i in range(10)), per=30)
    a = calibrate.calibrate(anchors, max_anchors=50)
    b = calibrate.calibrate(anchors, max_anchors=50)
    assert a.threshold == b.threshold


def _two_pair_anchors():
    # Two same-name pairs (A at sim 0.9, B at sim 0.7) with zero similarity
    # across names -- built in 4D so the A-pair and B-pair occupy disjoint
    # subspaces and every cross pair is exactly 0.
    return [
        ("A", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("A", np.array([0.9, np.sqrt(1 - 0.9**2), 0.0, 0.0], dtype=np.float32)),
        ("B", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        ("B", np.array([0.0, 0.0, 0.7, np.sqrt(1 - 0.7**2)], dtype=np.float32)),
    ]


def test_unreachable_target_precision_falls_back_to_best_recall():
    # With this anchor set, precision is 1.0 across the whole threshold range
    # up to 0.9 -- low thresholds keep both pairs (recall 1.0), high
    # thresholds keep only the A-pair (recall 0.5). An unreachable
    # target_precision forces the fallback; among tied-precision points it
    # must pick the lowest threshold, matching the primary scan's own bias
    # toward recall -- not the highest threshold, which is the worst point on
    # the plateau.
    result = calibrate.calibrate(_two_pair_anchors(), target_precision=1.5)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_self_pairs_are_excluded_from_the_curve():
    # A face compared to itself has similarity 1.0 and is trivially
    # "same-name" -- including it (np.triu_indices with k=0 instead of k=1)
    # inflates both precision and recall. Just above the B-pair's similarity
    # (0.7), only the A-pair (0.9) remains a genuine same-name match: 1 of 2
    # same-name pairs, so recall is 0.5. If self-pairs leaked in, the 4
    # self-pairs (always selected, always "same") would push recall to
    # 5/6 instead.
    result = calibrate.calibrate(_two_pair_anchors(), target_precision=0.99)
    point = next(c for c in result.curve if c[0] > 0.7)
    _, precision, recall = point
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
