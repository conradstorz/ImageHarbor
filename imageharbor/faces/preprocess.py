"""Build an ONNX input blob from images, per a model's declared contract.

Pure: numpy and Pillow only, no session, no filesystem, no network.

This lives apart from `detect` and `embed` because it is the one step of the
inference path that can be wrong while nothing raises. A wrong input *shape*
fails immediately and loudly. A wrong *channel order* or normalization loads,
runs, and returns plausible output that is merely worse -- which surfaces much
later as bad face clusters, with nothing pointing back here.

That is why `models.py` declares those fields instead of inferring them, and
why this function is reachable without loading a 261 MB artifact: the contract
can then be asserted directly, rather than inferred from a statistical wobble
in the embeddings it produces.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from .models import ModelInfo


def build_blob(images: Sequence[Image.Image], info: ModelInfo) -> np.ndarray:
    """Return an NCHW float32 blob for `images`, per `info`'s contract.

    Images are converted to RGB, resized to `info.input_size` if they are not
    already that size, reordered to `info.channel_order`, normalized by
    `(pixel - info.mean) / info.std`, and stacked NHWC -> NCHW.
    """
    width, height = info.input_size
    if not images:
        return np.zeros((0, 3, height, width), dtype=np.float32)

    if info.channel_order not in ("RGB", "BGR"):
        raise ValueError(
            f"unknown channel order {info.channel_order!r} for model {info.name!r}"
        )

    arrays = []
    for image in images:
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32)
        if info.channel_order == "BGR":
            # Pillow always decodes to RGB, so a BGR model needs the channel
            # axis reversed. Getting this backwards does not raise.
            array = array[:, :, ::-1]
        arrays.append((array - info.mean) / info.std)

    stacked = np.stack(arrays).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(stacked, dtype=np.float32)
