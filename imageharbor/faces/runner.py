"""The per-photo detect-and-embed pass. Resumable at one-photo granularity.

This pass makes no AI-backend call and touches no network. A failure here is a
filesystem or image-decode fault, never evidence about a backend, so it must
never reach the circuit breaker -- that is reserved for
`AIClassifier.describe()`. A corrupt or unreadable photo is instead recorded
through `Catalog.record_file_failure`, the same write the enrichment pass uses
for `failed_files` (see `imageharbor/watcher.py`'s `_reconcile_poison`), and
counted in `ScanResult.errors`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image

from .align import DegenerateLandmarks, align_crop

logger = logging.getLogger(__name__)

# On a 12 MP JPEG, Image.draft(...) before Image.load() downscales in the DCT
# domain and skips most of the decode -- decode, not inference, dominates this
# loop, and this one call is what fits a ~77,000-photo pass into an overnight
# CPU run. Do not remove it, and never touch pixels before it runs.
DECODE_SIZE = (640, 640)


@dataclass(frozen=True)
class QualityGate:
    """Thresholds below which a face embeds to noise rather than a person."""

    min_score: float = 0.6
    min_box: int = 32


@dataclass
class ScanResult:
    scanned: int = 0
    faces: int = 0
    rejected: int = 0
    errors: int = 0


def _work_queue(
    catalog, store, detect_model: str
) -> Iterator[tuple[str, str]]:
    """Yield `(digest, organized_path)` for organized photos not yet scanned.

    Every organized photo comes from `catalog.iter_all()` (there is no
    enrichment-status filter here -- face scanning is independent of the AI
    enrichment pass, unlike `Catalog.iter_unenriched`). A digest already
    scanned by *detect_model* is skipped so a re-run of `scan()` is a no-op:
    this is the loop's half of the idempotence contract, alongside
    `FaceStore.record_scan`'s own guard.
    """
    for row in catalog.iter_all():
        organized_path = row["organized_path"]
        if organized_path is None:
            continue
        digest = row["sha256_b64url"]
        if store.is_scanned(digest, detect_model):
            continue
        yield digest, organized_path


def _scan_one(
    path: Path,
    detector,
    embedder,
    gate: QualityGate,
    crop_dir: Path,
    digest: str,
    store,
) -> tuple[int, int]:
    """Detect, gate, align, and embed one photo's faces. Returns (kept, rejected)."""
    img = Image.open(path)
    img.draft("RGB", DECODE_SIZE)
    img.load()

    detections = detector.detect(img)

    records: list[tuple] = []
    kept_detections = []
    for det in detections:
        if det.score < gate.min_score:
            records.append((det, None, None, "low_score"))
        elif min(det.w, det.h) < gate.min_box:
            records.append((det, None, None, "too_small"))
        else:
            kept_detections.append(det)

    crops = []
    aligned_detections = []
    for det in kept_detections:
        try:
            crops.append(align_crop(img, det.landmarks))
        except DegenerateLandmarks:
            records.append((det, None, None, "degenerate_landmarks"))
        else:
            aligned_detections.append(det)

    # One `embed_batch` call per photo for every kept crop, not one per face.
    if crops:
        embeddings = embedder.embed_batch(crops)
    else:
        embeddings = np.zeros((0, embedder.dim), dtype=np.float32)

    photo_dir = crop_dir / digest[:2] / digest[2:4]
    if crops:
        photo_dir.mkdir(parents=True, exist_ok=True)
    for i, (det, crop, embedding) in enumerate(
        zip(aligned_detections, crops, embeddings)
    ):
        crop.save(photo_dir / f"{digest}-{i}.jpg", quality=85)
        records.append((det, embedding, embedder.model_name, None))

    store.record_scan(digest, detector.model_name, records)
    return len(aligned_detections), len(records) - len(aligned_detections)


def scan(
    catalog,
    store,
    detector,
    embedder,
    crop_dir: Path,
    *,
    gate: QualityGate,
    limit: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ScanResult:
    """Detect and embed every organized image not yet scanned by `detector`."""
    result = ScanResult()
    crop_dir = Path(crop_dir)

    for digest, organized_path in _work_queue(catalog, store, detector.model_name):
        if should_stop is not None and should_stop():
            break
        if limit is not None and result.scanned >= limit:
            break

        path = Path(organized_path)
        try:
            kept, rejected = _scan_one(
                path, detector, embedder, gate, crop_dir, digest, store
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
            logger.warning("face scan failed for %s: %s", digest, exc)
            try:
                st = path.stat()
                size, mtime_ns = st.st_size, st.st_mtime_ns
            except OSError:
                size, mtime_ns = 0, 0
            catalog.record_file_failure(str(path), size, mtime_ns, f"[faces] {exc}")
            result.errors += 1
            continue

        result.scanned += 1
        result.faces += kept
        result.rejected += rejected

    return result
