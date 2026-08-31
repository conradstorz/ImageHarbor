"""YuNet output decoding and NMS, on synthetic tensors. No model required."""

import numpy as np
import pytest

from imageharbor.faces import decode


def test_nms_keeps_the_highest_scoring_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert decode.nms(boxes, scores, 0.3) == [0]


def test_nms_keeps_both_when_they_do_not_overlap():
    boxes = np.array([[0, 0, 50, 50], [500, 500, 50, 50]], dtype=np.float32)
    scores = np.array([0.7, 0.9], dtype=np.float32)
    assert sorted(decode.nms(boxes, scores, 0.3)) == [0, 1]


def test_nms_returns_indices_in_descending_score_order():
    boxes = np.array([[0, 0, 10, 10], [500, 0, 10, 10], [0, 500, 10, 10]], dtype=np.float32)
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert decode.nms(boxes, scores, 0.3) == [1, 2, 0]


def test_nms_on_empty_input():
    empty = np.zeros((0, 4), dtype=np.float32)
    assert decode.nms(empty, np.zeros((0,), dtype=np.float32), 0.3) == []


def _synthetic_outputs(size=(640, 640), hot=None):
    """Build YuNet-shaped outputs with at most one confident cell.

    YuNet emits, per stride in (8, 16, 32): cls (N,1), obj (N,1), bbox (N,4)
    and kps (N,10), where N = (size/stride)**2 in row-major order.
    """
    out = []
    for stride in (8, 16, 32):
        gw, gh = size[0] // stride, size[1] // stride
        n = gw * gh
        cls = np.zeros((n, 1), dtype=np.float32)
        obj = np.zeros((n, 1), dtype=np.float32)
        bbox = np.zeros((n, 4), dtype=np.float32)
        kps = np.zeros((n, 10), dtype=np.float32)
        if hot is not None and hot[0] == stride:
            idx = hot[1]
            cls[idx, 0] = 1.0
            obj[idx, 0] = 1.0
            # bbox is (dx, dy, log-w, log-h) relative to the cell, in strides.
            bbox[idx] = [0.0, 0.0, np.log(4.0), np.log(4.0)]
            # Five landmarks, all offset one stride right and down of the cell.
            kps[idx] = [1.0, 1.0] * 5
        out.extend([cls, obj, bbox, kps])
    return out


def test_decode_returns_nothing_when_every_score_is_zero():
    assert decode.decode_yunet(_synthetic_outputs(), (640, 640), 0.5, 0.3) == []


def test_decode_places_a_detection_at_the_hot_cell():
    # Stride 8, grid 80x80. Cell index 81 is row 1, column 1 -> centre (8, 8).
    outs = _synthetic_outputs(hot=(8, 81))
    dets = decode.decode_yunet(outs, (640, 640), 0.5, 0.3)
    assert len(dets) == 1
    d = dets[0]
    # Width and height are exp(log 4) * stride = 32.
    assert d.w == pytest.approx(32.0)
    assert d.h == pytest.approx(32.0)
    # The box is centred on the cell, so its top-left is centre - size/2.
    assert d.x == pytest.approx(8.0 - 16.0)
    assert d.y == pytest.approx(8.0 - 16.0)
    assert d.score == pytest.approx(1.0)
    assert len(d.landmarks) == 5
    # Each landmark is one stride right and down of the cell centre.
    assert d.landmarks[0] == pytest.approx((16.0, 16.0))


def test_decode_respects_the_score_threshold():
    outs = _synthetic_outputs(hot=(8, 81))
    outs[0][81, 0] = 0.1  # cls for stride 8 -> score becomes sqrt(0.1 * 1.0)
    assert decode.decode_yunet(outs, (640, 640), 0.9, 0.3) == []
    assert len(decode.decode_yunet(outs, (640, 640), 0.2, 0.3)) == 1


def test_decode_is_deterministic():
    outs = _synthetic_outputs(hot=(16, 100))
    a = decode.decode_yunet(outs, (640, 640), 0.5, 0.3)
    b = decode.decode_yunet(_synthetic_outputs(hot=(16, 100)), (640, 640), 0.5, 0.3)
    assert [(d.x, d.y, d.w, d.h, d.score) for d in a] == [
        (d.x, d.y, d.w, d.h, d.score) for d in b
    ]
