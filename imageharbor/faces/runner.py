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

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import click
import numpy as np
from PIL import Image

from ..sidecar import merge_sidecar
from . import attribute, calibrate, cluster
from .align import DegenerateLandmarks, align_crop

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

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


def build_clusters(
    store,
    photo_names: Mapping[str, Sequence[str]],
    *,
    embed_model: str,
    threshold: float,
    min_score: float,
    min_support: int,
) -> int:
    """Re-cluster every face for *embed_model*, seeded from Google anchors,
    and record fresh name proposals.

    Never writes `clusters.person_id` -- that happens only when a human
    confirms a cluster on the dashboard (`FaceStore.confirm`/`merge`). This
    function only calls `replace_clusters` (a machine reclustering, which on
    its own reattaches an already-confirmed cluster's *existing* person but
    never assigns a new one) and `record_proposals` (which the store's own
    docstring guarantees never touches `person_id`).
    """
    # Sorted explicitly rather than trusted from the store: `cluster_faces`
    # is order-dependent by design (see cluster.py's module docstring), so a
    # re-cluster of the same faces must always visit them in the same order
    # regardless of what order the caller's iterator happens to yield.
    vectors = sorted(store.iter_face_vectors(embed_model), key=lambda f: f.face_id)

    anchor_ids = store.anchor_face_ids(embed_model, photo_names)
    seeds = [
        cluster.Seed(name=name, face_ids=tuple(anchor_ids[name]))
        for name in sorted(anchor_ids)
    ]

    clusters = cluster.cluster_faces(vectors, threshold=threshold, seeds=seeds)
    store.replace_clusters(embed_model, clusters)

    cluster_photos = store.digests_by_cluster(embed_model)
    proposals = attribute.propose(
        cluster_photos, photo_names, min_score=min_score, min_support=min_support
    )
    store.record_proposals(proposals)
    return len(clusters)


def measure_threshold(
    store,
    photo_names: Mapping[str, Sequence[str]],
    *,
    embed_model: str,
    target_precision: float,
) -> calibrate.Calibration:
    """Calibrate the clustering threshold from the library's own anchors."""
    anchors = store.anchors(embed_model, photo_names)
    distinct_names = {name for name, _ in anchors}
    if len(distinct_names) < 2:
        # calibrate.calibrate raises ValueError for the same condition, but
        # only after trying to np.stack an anchor list that may have zero or
        # one rows -- a much less legible failure for a CLI user than a
        # message that names the actual shortfall.
        raise click.ClickException(
            "calibration needs anchor photos (exactly one detected face, "
            "exactly one Google-tagged name) for at least two distinct "
            f"people; found {len(distinct_names)}. Tag more photos in "
            "Google Photos and re-run `faces scan` first."
        )
    return calibrate.calibrate(anchors, target_precision=target_precision)


def propagate_sidecars(store, dest: Path, detect_model: str) -> int:
    """Write every confirmed cluster's name into its photos' sidecars.

    Idempotent: `iter_pending_sidecars` only yields a digest whose
    confirmation (`clusters.assigned_at`) is newer than its last sidecar
    write (`face_scan.sidecar_at`), and even a forced re-write merges to a
    byte-identical document -- `confirmed_at` is a registered annotation
    field (`sidecar_schema._ANNOTATION_FIELDS`), so it advances in place on
    a `(name, source)` match instead of growing a `history` list.
    """
    written = 0
    for digest, names in store.iter_pending_sidecars():
        organized_path = store.organized_path_for(digest)
        if organized_path is None:
            # No known path to write to -- nothing this pass can do about it,
            # and it must not be counted as written or it will never be
            # retried once a path becomes known.
            continue
        updates = {
            "people": [
                {
                    "name": name,
                    "source": "imageharbor_faces",
                    "confirmed_at": _now_iso(),
                }
                for name in names
            ]
        }
        merge_sidecar(Path(organized_path), updates)
        store.mark_sidecar_written(digest, detect_model)
        written += 1
    return written


def google_names(dest: Path) -> dict[str, list[str]]:
    """`{digest: [name, ...]}` from every sidecar's Google-tagged people.

    Walks *dest* for sidecars directly rather than going through a `Catalog`
    handle -- this is shared by the `cluster` and `calibrate` CLI
    subcommands (Task 13), which only take `--dest`. Each sidecar carries its
    own digest in `identity.sha256_b64url` (written by the facts pass and by
    `backfill.py`); a sidecar missing that field, unreadable, or carrying no
    `google_photos_people` entry contributes nothing rather than raising --
    a name lookup for clustering must not abort a whole run over one bad
    file.
    """
    out: dict[str, list[str]] = {}
    for sidecar_path in sorted(Path(dest).rglob("*.json")):
        try:
            doc = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        identity = doc.get("identity")
        digest = identity.get("sha256_b64url") if isinstance(identity, dict) else None
        if not digest:
            continue
        names = [
            person.get("name")
            for person in doc.get("people", ())
            if isinstance(person, dict)
            and person.get("source") == "google_photos_people"
            and person.get("name")
        ]
        if names:
            out[digest] = names
    return out
