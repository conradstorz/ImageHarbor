"""Detector integration. Skips without weights; fails on a broken runtime."""

import os
from pathlib import Path

import pytest
from PIL import Image

onnxruntime = pytest.importorskip("onnxruntime")

from imageharbor.faces import detect, models  # noqa: E402

MODEL_DIR = Path(os.environ.get("IMAGEHARBOR_FACE_MODEL_DIR", "")) if os.environ.get(
    "IMAGEHARBOR_FACE_MODEL_DIR"
) else None


def _weights_present():
    if MODEL_DIR is None:
        return False
    return (MODEL_DIR / models.DETECTORS["yunet"].filename).exists()


needs_weights = pytest.mark.skipif(
    not _weights_present(),
    reason="set IMAGEHARBOR_FACE_MODEL_DIR to a directory holding the weights",
)


@needs_weights
def test_detector_loads_and_reports_its_model():
    d = detect.Detector(MODEL_DIR)
    assert d.model_name == "yunet"


@needs_weights
def test_a_blank_image_yields_no_faces():
    d = detect.Detector(MODEL_DIR)
    assert d.detect(Image.new("RGB", (640, 640), (10, 10, 10))) == []


@needs_weights
def test_a_real_photograph_yields_a_plausible_face():
    # tests/fixtures/one_face.jpg is a single-face photograph committed in Step 6.
    fixture = Path(__file__).parent.parent / "fixtures" / "one_face.jpg"
    img = Image.open(fixture)
    faces = detect.Detector(MODEL_DIR).detect(img)
    assert len(faces) == 1
    f = faces[0]
    assert f.score > 0.6
    # The box must sit inside the image and be a plausible fraction of it.
    assert 0 <= f.x < img.width and 0 <= f.y < img.height
    assert 0.02 < (f.w * f.h) / (img.width * img.height) < 0.9
    assert len(f.landmarks) == 5
    # Eyes above mouth: the cheapest check that the landmarks are not scrambled.
    assert f.landmarks[0][1] < f.landmarks[3][1]
