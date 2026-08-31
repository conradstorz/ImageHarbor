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
