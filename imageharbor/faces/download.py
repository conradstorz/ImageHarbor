"""Fetch and verify model artifacts.

An unverified artifact is never used. Two publishers ship different models under
the same filename -- InsightFace's antelopev2 pack and fal's AuraFace both call
theirs `glintr100.onnx` -- so a name match is not an artifact match, and only a
checksum settles it.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from .models import ModelInfo

logger = logging.getLogger(__name__)

_CHUNK = 1 << 16


class ChecksumMismatch(RuntimeError):
    """An artifact does not match its pinned digest, or has no pin at all."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(info: ModelInfo, model_dir: Path) -> Path:
    """Return a verified local path for `info`, downloading it if absent."""
    if not info.sha256:
        raise ChecksumMismatch(
            f"{info.filename}: no pinned checksum, refusing to run an "
            "unverified model"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / info.filename

    if not target.exists():
        logger.info("downloading face model %s from %s", info.name, info.url)
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(info.url, tmp)  # noqa: S310 - pinned URL
        tmp.replace(target)

    actual = _sha256(target)
    if actual != info.sha256:
        raise ChecksumMismatch(
            f"{info.filename}: expected {info.sha256}, got {actual}"
        )
    return target
