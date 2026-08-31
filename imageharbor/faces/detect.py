"""Run YuNet over an image. I/O only: the decode lives in `decode.py`."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import models
from .decode import Detection, decode_yunet
from .download import ensure

logger = logging.getLogger(__name__)


class Detector:
    """A loaded YuNet session. Not thread-safe; construct one per worker."""

    def __init__(self, model_dir: Path, name: str = models.DEFAULT_DETECTOR) -> None:
        import onnxruntime as ort

        self._info = models.DETECTORS[name]
        self.model_name = name
        path = ensure(self._info, Path(model_dir))
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def detect(
        self,
        image: Image.Image,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> list[Detection]:
        """Detect faces, returning boxes in `image`'s own pixel coordinates."""
        width, height = self._info.input_size
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        scale_x = rgb.width / width
        scale_y = rgb.height / height

        resized = rgb.resize((width, height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        if self._info.channel_order == "BGR":
            array = array[:, :, ::-1]
        array = (array - self._info.mean) / self._info.std
        blob = np.ascontiguousarray(array.transpose(2, 0, 1)[None])

        outputs = self._session.run(None, {self._input: blob})
        detections = decode_yunet(
            outputs, (width, height), score_threshold, nms_threshold
        )

        # Back to the source image's coordinates.
        return [
            Detection(
                x=d.x * scale_x,
                y=d.y * scale_y,
                w=d.w * scale_x,
                h=d.h * scale_y,
                score=d.score,
                landmarks=tuple(
                    (px * scale_x, py * scale_y) for px, py in d.landmarks
                ),
            )
            for d in detections
        ]
