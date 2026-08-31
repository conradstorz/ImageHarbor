"""Warp a detected face onto the ArcFace 5-point template. Pure geometry.

The template is the standard InsightFace ArcFace destination for a 112x112
crop. Every ArcFace-family embedder -- AuraFace included -- expects its input
aligned to it, so this is a contract, not a preference.

No OpenCV. Pillow's `Image.transform(..., AFFINE, ...)` does the resampling,
which keeps a 60 MB vision dependency out of a project whose entire runtime
dependency list is Pillow and Click.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

# InsightFace's canonical 5-point destination for a 112x112 crop:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


class DegenerateLandmarks(ValueError):
    """Landmarks that cannot define a similarity transform.

    Collinear or coincident points make the estimate rank-deficient. Raising
    here means the caller rejects that face, which is correct: a face whose
    landmarks collapse to a line is not a usable face, and warping it anyway
    produces a crop that embeds to noise.
    """


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (scale, rotation, translation) mapping src->dst.

    The Umeyama estimate. Returns a 3x3 homogeneous matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    src_var = (src_demean**2).sum() / n
    if src_var < 1e-12:
        raise DegenerateLandmarks("landmarks are coincident")

    cov = dst_demean.T @ src_demean / n
    u, s, vt = np.linalg.svd(cov)

    if np.linalg.matrix_rank(cov) < 2:
        raise DegenerateLandmarks("landmarks are collinear")

    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0

    rotation = u @ np.diag(d) @ vt
    scale = float(s @ d) / src_var

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rotation * scale
    matrix[:2, 2] = dst_mean - (rotation * scale) @ src_mean
    return matrix


def align_crop(
    image: Image.Image,
    landmarks: Sequence[tuple[float, float]],
    size: tuple[int, int] = (112, 112),
) -> Image.Image:
    """Warp `image` so `landmarks` land on the ArcFace template."""
    if len(landmarks) != 5:
        raise DegenerateLandmarks(f"expected 5 landmarks, got {len(landmarks)}")

    scale = np.array([size[0] / 112.0, size[1] / 112.0])
    forward = similarity_transform(
        np.asarray(landmarks, dtype=np.float64), ARCFACE_TEMPLATE * scale
    )

    # Pillow's AFFINE data is the OUTPUT -> INPUT mapping, so the inverse of the
    # transform we just estimated. Passing `forward` here yields a warp that
    # looks plausible and is wrong.
    inverse = np.linalg.inv(forward)
    data = (
        inverse[0, 0], inverse[0, 1], inverse[0, 2],
        inverse[1, 0], inverse[1, 1], inverse[1, 2],
    )
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return rgb.transform(size, Image.AFFINE, data, resample=Image.BILINEAR)
