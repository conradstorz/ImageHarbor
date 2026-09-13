"""Structural types for the runner/store seam.

Protocols, not ABCs: the test suite's FakeDetector/FakeEmbedder satisfy
them by shape without importing anything, and detect.py/embed.py (the only
onnxruntime importers) satisfy them without this module ever importing
onnxruntime.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .decode import Detection


class DetectorLike(Protocol):
    model_name: str

    def detect(self, img: Image.Image) -> list[Detection]: ...


class EmbedderLike(Protocol):
    model_name: str
    dim: int

    def embed_batch(self, crops: Sequence[Image.Image]) -> np.ndarray: ...
