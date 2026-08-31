"""YuNet output decoding and NMS, on synthetic tensors. No model required.

One test at the bottom is the exception: it runs the real ONNX artifact, when
present, as the check that this module's assumptions about output order and
rank actually match the exported graph rather than a hand-built draft of it.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from imageharbor.faces import decode, models


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

    Mirrors the real exported graph exactly: twelve tensors, type-major then
    stride-major -- all three `cls_{8,16,32}`, then all three `obj`, then all
    three `bbox`, then all three `kps` -- each shaped `(1, N, C)` with a
    leading batch axis of 1, where N = (size/stride)**2 in row-major order.
    """
    strides = (8, 16, 32)
    cls_out, obj_out, bbox_out, kps_out = [], [], [], []
    for stride in strides:
        gw, gh = size[0] // stride, size[1] // stride
        n = gw * gh
        cls = np.zeros((1, n, 1), dtype=np.float32)
        obj = np.zeros((1, n, 1), dtype=np.float32)
        bbox = np.zeros((1, n, 4), dtype=np.float32)
        kps = np.zeros((1, n, 10), dtype=np.float32)
        if hot is not None and hot[0] == stride:
            idx = hot[1]
            cls[0, idx, 0] = 1.0
            obj[0, idx, 0] = 1.0
            # bbox is (dx, dy, log-w, log-h) relative to the cell, in strides.
            bbox[0, idx] = [0.0, 0.0, np.log(4.0), np.log(4.0)]
            # Five landmarks, all offset one stride right and down of the cell.
            kps[0, idx] = [1.0, 1.0] * 5
        cls_out.append(cls)
        obj_out.append(obj)
        bbox_out.append(bbox)
        kps_out.append(kps)
    return cls_out + obj_out + bbox_out + kps_out


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
    outs[0][0, 81, 0] = 0.1  # cls for stride 8 -> score becomes sqrt(0.1 * 1.0)
    assert decode.decode_yunet(outs, (640, 640), 0.9, 0.3) == []
    assert len(decode.decode_yunet(outs, (640, 640), 0.2, 0.3)) == 1


def test_decode_is_deterministic():
    outs = _synthetic_outputs(hot=(16, 100))
    a = decode.decode_yunet(outs, (640, 640), 0.5, 0.3)
    b = decode.decode_yunet(_synthetic_outputs(hot=(16, 100)), (640, 640), 0.5, 0.3)
    assert [(d.x, d.y, d.w, d.h, d.score) for d in a] == [
        (d.x, d.y, d.w, d.h, d.score) for d in b
    ]


def test_decode_yunet_against_the_real_model_on_a_blank_image():
    """The real proof: run the actual exported graph, not a synthetic stand-in.

    `_synthetic_outputs` above encodes the same beliefs about output order and
    rank as `decode_yunet` does -- agreement between the two proves nothing
    about the real artifact. This test is the thing that would have caught
    both defects: it feeds the decoder onnxruntime's real output list, in
    onnxruntime's real order and rank, unmodified.
    """
    model_dir = os.environ.get("IMAGEHARBOR_FACE_MODEL_DIR")
    if not model_dir:
        pytest.skip("IMAGEHARBOR_FACE_MODEL_DIR not set")

    info = models.get("yunet")
    weights = Path(model_dir) / info.filename
    if not weights.exists():
        pytest.skip(f"weights not found: {weights}")

    onnxruntime = pytest.importorskip("onnxruntime")

    width, height = info.input_size
    # Mid-grey, constant-fill image: channel order is moot when every channel
    # holds the same value, but build the blob the way a real BGR frame would
    # be built, per the yunet entry's declared contract (BGR, mean 0.0, std
    # 1.0, NCHW), rather than relying on the coincidence.
    hwc = np.full((height, width, 3), 128.0, dtype=np.float32)
    assert info.channel_order == "BGR"
    blob = ((hwc - info.mean) / info.std).transpose(2, 0, 1)[np.newaxis, ...]
    blob = blob.astype(np.float32)

    session = onnxruntime.InferenceSession(str(weights))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    detections = decode.decode_yunet(outputs, (width, height), 0.6, 0.3)
    assert isinstance(detections, list)
    # A blank frame has no face in it. If this starts failing with a handful
    # of low-confidence boxes, that is real information about the decoder or
    # the model, not a reason to raise the threshold until it goes quiet.
    assert detections == [], (
        f"blank image produced {len(detections)} spurious detection(s): "
        f"{[(round(d.score, 4), d.w, d.h) for d in detections]}"
    )
