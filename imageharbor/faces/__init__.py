"""Face detection, embedding, clustering, and name proposal.

The pure modules in this package (`names`, `decode`, `align`, `cluster`,
`attribute`, `calibrate`) import nothing from the rest of ImageHarbor and touch
no filesystem, so the whole core is testable without a byte of model weights.
Only `detect` and `embed` import onnxruntime, and only when a model is actually
run -- importing this package must never require the optional `faces` extra.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivial availability probe
    import onnxruntime  # noqa: F401

    HAS_ONNX = True
except Exception:  # pragma: no cover
    # Catch all exceptions: missing onnxruntime (ImportError), broken
    # installations with ABI mismatches (RuntimeError, OSError), or any other
    # failure during import. All failures to import mean we cannot run a model,
    # so they all answer the same question: HAS_ONNX = False.
    logger.debug("onnxruntime unavailable; faces functionality disabled", exc_info=True)
    HAS_ONNX = False

__all__ = ["HAS_ONNX"]
