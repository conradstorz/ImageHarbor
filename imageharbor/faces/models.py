"""Registry of face models. Pure: no I/O, no imports from the package.

This module exists because **channel order and normalization are not present in
an ONNX graph**. Getting the input shape wrong raises immediately. Getting the
channel order wrong loads, runs, and returns plausible output that is quietly
worse -- which is far more expensive, because nothing fails. Those fields are
declared per model and never inferred.

Filenames are disambiguated by publisher for the same reason: InsightFace's
antelopev2 pack and fal's AuraFace both ship a file named `glintr100.onnx`, and
they are different models. A name match is not an artifact match.

Both defaults are permissively licensed on purpose. InsightFace's ArcFace
weights are not redistributable, and ImageHarbor must not acquire a non-free
artifact dependency by default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """An ONNX artifact plus the preprocessing contract its graph omits."""

    name: str
    kind: str                    # "detector" | "embedder"
    filename: str                # local name; disambiguated by publisher
    url: str
    sha256: str | None           # pinned in Task 9 after a verified download
    input_size: tuple[int, int]  # (width, height)
    channel_order: str           # "RGB" | "BGR"
    mean: float
    std: float
    licence: str
    embedding_dim: int | None = None


DETECTORS: dict[str, ModelInfo] = {
    "yunet": ModelInfo(
        name="yunet",
        kind="detector",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256=None,
        input_size=(640, 640),
        # YuNet consumes raw 0-255 BGR: OpenCV's FaceDetectorYN builds its blob
        # with every blobFromImage default, and those defaults are BGR with no
        # scaling. Verified against OpenCV's source, not inferred.
        channel_order="BGR",
        mean=0.0,
        std=1.0,
        licence="MIT",
    ),
}

EMBEDDERS: dict[str, ModelInfo] = {
    "auraface": ModelInfo(
        name="auraface",
        kind="embedder",
        filename="auraface_v1_glintr100.onnx",
        url="https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
        sha256=None,
        input_size=(112, 112),
        # Every embedding model on the aligned crop takes RGB, normalized to
        # roughly [-1, 1] by (x - 127.5) / 128.
        channel_order="RGB",
        mean=127.5,
        std=128.0,
        licence="Apache-2.0",
        embedding_dim=512,
    ),
}

DEFAULT_DETECTOR = "yunet"
DEFAULT_EMBEDDER = "auraface"


def get(name: str) -> ModelInfo:
    """Look up a model by name across both registries."""
    if name in DETECTORS:
        return DETECTORS[name]
    if name in EMBEDDERS:
        return EMBEDDERS[name]
    raise KeyError(f"unknown face model: {name!r}")
