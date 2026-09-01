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
    """Landmarks that cannot define a stable similarity transform.

    Coincident points, and points that are exactly or *nearly* collinear,
    leave the covariance matrix rank-deficient or so ill-conditioned that
    the estimate is numerically unstable. Raising here means the caller
    rejects that face, which is correct: a face whose landmarks collapse
    toward a line is not a usable face, and warping it anyway produces a
    crop that embeds to noise. See `similarity_transform` for how "nearly"
    is measured and why.
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

    # `np.linalg.matrix_rank(cov) < 2` (the original check) only catches
    # *exactly* rank-deficient input -- its default tolerance is essentially
    # machine precision, so landmarks that are merely near-collinear (a real
    # detector on an extreme profile, motion blur, or partial occlusion)
    # sail through and produce a wild, physically nonsensical scale instead
    # of a rejection. Test the conditioning of `cov` directly with the
    # singular values `svd` already computed above, rather than paying for
    # a second, redundant SVD inside matrix_rank.
    #
    # Measured smallest/largest singular-value ratios (see
    # tests/faces/test_align.py and docs/superpowers/plans/
    # 2026-08-31-face-recognition.md Task 5 for the full sweep):
    #   - ArcFace template onto itself (typical frontal face):        0.63
    #   - template + 1.5px detector jitter:                           0.62
    #   - rotated/scaled template (existing regression cases):        0.63
    #   - three-quarter profile, eyes/nose compressed toward one side: 0.40
    #   - extreme near-edge-on profile (80% compression):             0.29
    #   - reviewer's landmarks 1e-4 off a perfect line:            1.15e-6
    #   - exactly collinear:                                       7.0e-17
    # Realistic geometry -- including a hard profile -- never drops below
    # ~0.2; the pathological near-collinear case sits around 1e-6, five to
    # six orders of magnitude lower. 1e-3 sits comfortably in that gap: it
    # is ~1000x above the pathological ratio and ~200x below the most
    # extreme realistic one measured, so it rejects unstable fits without
    # discarding usable faces.
    # Guard the division: s[0] == 0 only if `cov` itself is the zero matrix
    # (e.g. `dst` is coincident, since `src` coincidence is already excluded
    # above), which is degenerate regardless of the ratio -- `0/0` is `nan`
    # and `nan < 1e-3` is silently False, so this must be checked first
    # rather than folded into the ratio comparison.
    if s[0] < 1e-12 or s[-1] / s[0] < 1e-3:
        raise DegenerateLandmarks("landmarks are collinear or too close to it")

    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    # `sign(det(cov))` and `sign(det(u) * det(vt))` are always equal once
    # the conditioning guard above has passed -- both encode the same
    # reflection via the same (now well-conditioned) `cov`, so this branch
    # currently never flips `d` a second time. Kept anyway: reflection
    # handling here is subtle, and a future reader who loosens or removes
    # the guard above may need this check to still be doing real work.
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
