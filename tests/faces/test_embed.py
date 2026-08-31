"""Embedder integration. Skips without weights; fails on a broken runtime."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")

from imageharbor.faces import embed, models  # noqa: E402

MODEL_DIR = Path(os.environ["IMAGEHARBOR_FACE_MODEL_DIR"]) if os.environ.get(
    "IMAGEHARBOR_FACE_MODEL_DIR"
) else None

needs_weights = pytest.mark.skipif(
    MODEL_DIR is None
    or not (MODEL_DIR / models.EMBEDDERS["auraface"].filename).exists(),
    reason="set IMAGEHARBOR_FACE_MODEL_DIR to a directory holding the weights",
)

LANDMARKS = [(150.0, 160.0), (250.0, 160.0), (200.0, 210.0), (160.0, 270.0), (240.0, 270.0)]


@needs_weights
def test_embedding_has_the_declared_dimension():
    e = embed.Embedder(MODEL_DIR)
    v = e.embed(Image.new("RGB", (400, 400), (120, 100, 90)), LANDMARKS)
    assert v.shape == (e.dim,)
    assert e.dim == 512


@needs_weights
def test_embedding_is_l2_normalized_at_production():
    e = embed.Embedder(MODEL_DIR)
    v = e.embed(Image.new("RGB", (400, 400), (120, 100, 90)), LANDMARKS)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


@needs_weights
def test_the_same_image_embeds_identically():
    e = embed.Embedder(MODEL_DIR)
    img = Image.new("RGB", (400, 400), (120, 100, 90))
    assert np.allclose(e.embed(img, LANDMARKS), e.embed(img, LANDMARKS), atol=1e-6)


@needs_weights
def test_a_real_face_is_closer_to_itself_rotated_than_to_a_blank():
    fixture = Path(__file__).parent.parent / "fixtures" / "one_face.jpg"
    img = Image.open(fixture).convert("RGB")
    e = embed.Embedder(MODEL_DIR)
    from imageharbor.faces import detect

    d = detect.Detector(MODEL_DIR).detect(img)[0]
    a = e.embed(img, d.landmarks)
    b = e.embed(img.rotate(4, expand=False), d.landmarks)
    blank = e.embed(Image.new("RGB", img.size, (255, 255, 255)), d.landmarks)
    assert float(a @ b) > float(a @ blank)


@needs_weights
def test_correct_alignment_is_more_self_similar_than_gross_misalignment():
    # The brief's rotation test is a weak discriminator: almost any vector
    # arrangement satisfies "closer to itself than to a blank". This is a
    # stronger check of the alignment contract specifically. Build the
    # correctly-aligned pair the same way the rotation test does -- the real
    # detected landmarks against the original image and against a slightly
    # rotated copy, which keeps the crop close to correctly aligned to the
    # ArcFace template both times -- and compare its self-similarity against
    # the same original crop embedded with its landmarks shifted by a large
    # offset (eyes/nose/mouth mapped to the wrong place in the template, so
    # the warp is grossly misaligned). If alignment is doing real work, the
    # correctly-aligned pair should agree with each other far more than the
    # original agrees with its own misaligned crop.
    fixture = Path(__file__).parent.parent / "fixtures" / "one_face.jpg"
    img = Image.open(fixture).convert("RGB")
    e = embed.Embedder(MODEL_DIR)
    from imageharbor.faces import detect

    d = detect.Detector(MODEL_DIR).detect(img)[0]
    correct = d.landmarks
    misaligned = tuple((x + 60.0, y - 60.0) for x, y in d.landmarks)

    a = e.embed(img, correct)
    b = e.embed(img.rotate(4, expand=False), correct)
    misaligned_embedding = e.embed(img, misaligned)

    correctly_aligned_similarity = float(a @ b)
    misaligned_similarity = float(a @ misaligned_embedding)
    assert correctly_aligned_similarity > misaligned_similarity
