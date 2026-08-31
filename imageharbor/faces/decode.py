"""Decode YuNet's raw ONNX outputs into detections. Pure: no I/O, no session.

This is the fiddliest logic in the face pipeline, so it lives here, separated
from the ONNX session in `detect.py`, and is tested against synthetic tensors
with no model present.

YuNet emits four tensors per stride in (8, 16, 32), each with one row per grid
cell in row-major order:

    cls  (N, 1)   classification logit, already sigmoid-ed by the graph
    obj  (N, 1)   objectness, already sigmoid-ed
    bbox (N, 4)   (dx, dy, log w, log h), offsets in stride units
    kps  (N, 10)  five (dx, dy) landmark offsets, in stride units

The confidence of a cell is ``sqrt(cls * obj)`` -- the geometric mean, which is
what the reference implementation uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

STRIDES: tuple[int, ...] = (8, 16, 32)


@dataclass(frozen=True)
class Detection:
    """One detected face in input-image pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: tuple[tuple[float, float], ...]  # 5 points: eyes, nose, mouth


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best score first.

    `boxes` is (N, 4) as (x, y, w, h).
    """
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= threshold]
    return keep


def decode_yunet(
    outputs: Sequence[np.ndarray],
    input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    """Turn YuNet's raw outputs into detections in input-image coordinates."""
    width, height = input_size
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    kps: list[np.ndarray] = []

    for si, stride in enumerate(STRIDES):
        cls, obj, bbox, kp = outputs[si * 4 : si * 4 + 4]
        gw, gh = width // stride, height // stride

        # Cell centres in row-major order, matching the graph's flattening.
        cols = np.tile(np.arange(gw, dtype=np.float32), gh) * stride
        rows = np.repeat(np.arange(gh, dtype=np.float32), gw) * stride

        conf = np.sqrt(
            np.clip(cls[:, 0], 0.0, None) * np.clip(obj[:, 0], 0.0, None)
        )
        hot = conf >= score_threshold
        if not np.any(hot):
            continue

        cx = cols[hot] + bbox[hot, 0] * stride
        cy = rows[hot] + bbox[hot, 1] * stride
        bw = np.exp(bbox[hot, 2]) * stride
        bh = np.exp(bbox[hot, 3]) * stride

        boxes.append(np.stack([cx - bw / 2.0, cy - bh / 2.0, bw, bh], axis=1))
        scores.append(conf[hot])

        pts = kp[hot].reshape(-1, 5, 2) * stride
        pts[:, :, 0] += cols[hot][:, None]
        pts[:, :, 1] += rows[hot][:, None]
        kps.append(pts)

    if not boxes:
        return []

    all_boxes = np.concatenate(boxes).astype(np.float32)
    all_scores = np.concatenate(scores).astype(np.float32)
    all_kps = np.concatenate(kps).astype(np.float32)

    return [
        Detection(
            x=float(all_boxes[i, 0]),
            y=float(all_boxes[i, 1]),
            w=float(all_boxes[i, 2]),
            h=float(all_boxes[i, 3]),
            score=float(all_scores[i]),
            landmarks=tuple(
                (float(p[0]), float(p[1])) for p in all_kps[i]
            ),
        )
        for i in nms(all_boxes, all_scores, nms_threshold)
    ]
