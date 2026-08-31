"""Landmark alignment onto the ArcFace template. Pure geometry, no model."""

import numpy as np
import pytest
from PIL import Image

from imageharbor.faces import align


def test_template_is_five_points_in_a_112_box():
    assert align.ARCFACE_TEMPLATE.shape == (5, 2)
    assert align.ARCFACE_TEMPLATE.min() > 0
    assert align.ARCFACE_TEMPLATE.max() < 112


def test_identity_when_source_is_already_the_template():
    t = align.similarity_transform(align.ARCFACE_TEMPLATE, align.ARCFACE_TEMPLATE)
    assert np.allclose(t, np.eye(3), atol=1e-6)


def test_recovers_a_known_scale_and_translation():
    src = align.ARCFACE_TEMPLATE * 2.0 + np.array([30.0, 40.0])
    t = align.similarity_transform(src, align.ARCFACE_TEMPLATE)
    homogeneous = np.hstack([src, np.ones((5, 1))])
    mapped = (t @ homogeneous.T).T[:, :2]
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-4)


def test_recovers_a_known_rotation():
    theta = np.deg2rad(30.0)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = align.ARCFACE_TEMPLATE @ r.T
    t = align.similarity_transform(src, align.ARCFACE_TEMPLATE)
    homogeneous = np.hstack([src, np.ones((5, 1))])
    mapped = (t @ homogeneous.T).T[:, :2]
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-4)


def test_collinear_landmarks_raise_rather_than_returning_a_bad_warp():
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    with pytest.raises(align.DegenerateLandmarks):
        align.similarity_transform(collinear, align.ARCFACE_TEMPLATE)


def test_identical_landmarks_raise():
    same = np.zeros((5, 2))
    with pytest.raises(align.DegenerateLandmarks):
        align.similarity_transform(same, align.ARCFACE_TEMPLATE)


def test_near_collinear_landmarks_raise():
    # Reviewer-reported gap: points 1e-4 off a perfect line passed the old
    # exact-rank check (np.linalg.matrix_rank's default tolerance is
    # essentially machine precision) and produced a similarity transform
    # with scale ~12.6 -- physically nonsensical for a face fit. The
    # conditioning check must catch this near-degenerate case too.
    near_collinear = np.array(
        [[0.0, 0.0], [1.0, 1.0001], [2.0, 2.0], [3.0, 3.0001], [4.0, 4.0]]
    )
    with pytest.raises(align.DegenerateLandmarks):
        align.similarity_transform(near_collinear, align.ARCFACE_TEMPLATE)


def test_arcface_template_onto_itself_does_not_raise():
    # Guard against over-rejection: the template is a perfectly typical
    # frontal face and must never be flagged degenerate.
    t = align.similarity_transform(align.ARCFACE_TEMPLATE, align.ARCFACE_TEMPLATE)
    assert np.allclose(t, np.eye(3), atol=1e-6)


def test_rotated_and_scaled_template_does_not_raise():
    # Guard against over-rejection on realistic, well-conditioned geometry.
    theta = np.deg2rad(30.0)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = align.ARCFACE_TEMPLATE @ r.T * 2.0 + np.array([30.0, 40.0])
    t = align.similarity_transform(src, align.ARCFACE_TEMPLATE)
    homogeneous = np.hstack([src, np.ones((5, 1))])
    mapped = (t @ homogeneous.T).T[:, :2]
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-4)


def test_align_crop_returns_the_requested_size():
    img = Image.new("RGB", (400, 400), (128, 64, 32))
    landmarks = [(150.0, 160.0), (250.0, 160.0), (200.0, 210.0), (160.0, 270.0), (240.0, 270.0)]
    out = align.align_crop(img, landmarks)
    assert out.size == (112, 112)
    assert out.mode == "RGB"


def test_align_crop_puts_the_eye_where_the_template_says():
    # A face drawn so its landmarks are the template scaled by 2 and shifted:
    # after alignment the left eye must land on the template's left eye.
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    src = align.ARCFACE_TEMPLATE * 2.0 + np.array([50.0, 50.0])
    # Mark a small block around the left-eye landmark, not a single pixel:
    # align_crop downsamples this synthetic face by ~2x, and a lone marked
    # pixel lands off-grid under bilinear resampling, diluting to ~63/255
    # (verified numerically) even for a correctly-inverted transform -- a
    # false negative, not evidence of a bug. A block survives the same
    # downsampling with full brightness while still landing at (0, 0) if the
    # transform is inverted (a deliberately un-inverted transform was checked
    # to place this exact block entirely off-canvas -- window max 0).
    mx, my = int(src[0][0]), int(src[0][1])
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            img.putpixel((mx + dx, my + dy), (255, 255, 255))
    out = align.align_crop(img, [tuple(p) for p in src])
    ex, ey = align.ARCFACE_TEMPLATE[0]
    window = [
        out.getpixel((x, y))[0]
        for x in range(int(ex) - 2, int(ex) + 3)
        for y in range(int(ey) - 2, int(ey) + 3)
    ]
    assert max(window) > 100
