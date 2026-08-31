"""Turn an aligned face crop into a vector. I/O only; the warp is in `align`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from . import models
from .align import align_crop
from .download import ensure


class Embedder:
    """A loaded embedding session. Not thread-safe; construct one per worker."""

    def __init__(self, model_dir: Path, name: str = models.DEFAULT_EMBEDDER) -> None:
        import onnxruntime as ort

        self._info = models.EMBEDDERS[name]
        self.model_name = name
        self.dim = self._info.embedding_dim
        path = ensure(self._info, Path(model_dir))
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def _blob(self, crops: Sequence[Image.Image]) -> np.ndarray:
        arrays = []
        for crop in crops:
            a = np.asarray(crop, dtype=np.float32)
            if self._info.channel_order == "BGR":  # pragma: no cover - RGB today
                a = a[:, :, ::-1]
            arrays.append((a - self._info.mean) / self._info.std)
        # Crops stack to (N, H, W, C); the model wants NCHW.
        stacked = np.stack(arrays).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(stacked, dtype=np.float32)

    def embed_batch(self, crops: Sequence[Image.Image]) -> np.ndarray:
        """Embed pre-aligned 112x112 crops. Returns (N, dim), L2-normalized."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        raw = self._session.run(None, {self._input: self._blob(crops)})[0]
        vectors = np.asarray(raw, dtype=np.float32)
        # Normalized here, at the point of production, so cosine and Euclidean
        # stay equivalent for every consumer downstream (cluster.py's centroid
        # math, calibrate.py's cosine similarities).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def embed(
        self, image: Image.Image, landmarks: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        """Align one face out of `image` and embed it."""
        crop = align_crop(image, landmarks, self._info.input_size)
        return self.embed_batch([crop])[0]
