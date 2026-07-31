"""Main photo-organization processing pipeline."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ai_classifier import AIClassifier, PhotoClassification, StubClassifier
from .catalog import Catalog
from .discovery import discover_images
from .exif_reader import read_exif
from .filename import generate_filename, normalize_descriptor
from .hashing import compute_sha256_b64url, verify_file
from .pcs import parent_folder_name, resolve_code, sub_folder_name
from .sidecar import write_sidecar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ProcessResult:
    """Outcome of processing a single image."""

    source_path: Path
    sha256_b64url: str
    status: str          # "copied" | "duplicate" | "skipped" | "error"
    organized_path: Path | None = None
    error: str = ""


@dataclass
class PipelineStats:
    """Aggregated statistics for a pipeline run."""

    total: int = 0
    copied: int = 0
    duplicates: int = 0
    skipped: int = 0
    errors: int = 0
    results: list[ProcessResult] = field(default_factory=list)

    def record(self, result: ProcessResult) -> None:
        self.total += 1
        self.results.append(result)
        if result.status == "copied":
            self.copied += 1
        elif result.status == "duplicate":
            self.duplicates += 1
        elif result.status == "skipped":
            self.skipped += 1
        else:
            self.errors += 1


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Deterministic, resumable photo-organization pipeline.

    Parameters
    ----------
    source_dir:
        Read-only directory containing original photos.
    organized_dir:
        Root of the organized library tree.
    catalog:
        Open :class:`~imageharbor.catalog.Catalog` instance.
    classifier:
        :class:`~imageharbor.ai_classifier.AIClassifier` implementation.
        Defaults to :class:`~imageharbor.ai_classifier.StubClassifier`.
    duplicates_dir:
        Where to copy duplicate images. If None, duplicates are skipped.
    write_sidecars:
        Whether to write a JSON sidecar alongside each organized image.
    dry_run:
        When True, no files are written and the catalog is not updated.
    """

    def __init__(
        self,
        source_dir: Path,
        organized_dir: Path,
        catalog: Catalog,
        classifier: AIClassifier | None = None,
        duplicates_dir: Path | None = None,
        write_sidecars: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_dir = source_dir
        self.organized_dir = organized_dir
        self.catalog = catalog
        self.classifier: AIClassifier = classifier or StubClassifier()
        self.duplicates_dir = duplicates_dir
        self.write_sidecars = write_sidecars
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, recursive: bool = True) -> PipelineStats:
        """Process all images under :attr:`source_dir`.

        Returns a :class:`PipelineStats` summary.
        """
        stats = PipelineStats()
        for image_path in discover_images(self.source_dir, recursive=recursive):
            result = self._process_one(image_path)
            stats.record(result)
            _log_result(result)
        return stats

    def process_file(self, image_path: Path) -> ProcessResult:
        """Process a single image file and return its result."""
        result = self._process_one(image_path)
        _log_result(result)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_one(self, source_path: Path) -> ProcessResult:
        try:
            return self._do_process(source_path)
        except Exception as exc:
            logger.exception("Unexpected error processing %s", source_path)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url="",
                status="error",
                error=str(exc),
            )

    def _do_process(self, source_path: Path) -> ProcessResult:
        # Step 1: hash original
        sha256_b64url = compute_sha256_b64url(source_path)

        # Step 2: duplicate detection
        if self.catalog.is_known(sha256_b64url):
            existing = self.catalog.get_by_sha256(sha256_b64url)
            logger.info("Duplicate detected: %s (matches %s)", source_path, existing["organized_path"] if existing else "?")
            if not self.dry_run:
                self.catalog.mark_duplicate(sha256_b64url, str(source_path))
                if self.duplicates_dir:
                    self._copy_to_duplicates(source_path, sha256_b64url)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="duplicate",
            )

        # Step 3: EXIF
        exif_data = read_exif(source_path)

        # Step 4: AI classification
        classification = self.classifier.classify(source_path, exif_data)

        # Step 5: resolve PCS code
        pcs_code = resolve_code(classification.pcs_code)

        # Step 6: generate filename
        descriptor = normalize_descriptor(classification.descriptor)
        extension = source_path.suffix.lstrip(".").lower()
        filename = generate_filename(pcs_code, descriptor, sha256_b64url, extension)

        # Step 7: determine output path
        organized_path = (
            self.organized_dir
            / parent_folder_name(pcs_code)
            / sub_folder_name(pcs_code)
            / filename
        )

        if self.dry_run:
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="copied",
                organized_path=organized_path,
            )

        # Step 8: copy
        organized_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(organized_path))

        # Step 9: verify
        if not verify_file(organized_path, sha256_b64url):
            organized_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Integrity check failed after copying {source_path} -> {organized_path}"
            )

        # Step 10: catalog
        self._update_catalog(
            source_path=source_path,
            organized_path=organized_path,
            sha256_b64url=sha256_b64url,
            classification=classification,
            pcs_code=pcs_code,
            exif_data=exif_data,
        )

        # Step 11: optional sidecar
        if self.write_sidecars:
            self._write_sidecar(organized_path, source_path, sha256_b64url, classification, exif_data)

        return ProcessResult(
            source_path=source_path,
            sha256_b64url=sha256_b64url,
            status="copied",
            organized_path=organized_path,
        )

    def _update_catalog(
        self,
        *,
        source_path: Path,
        organized_path: Path,
        sha256_b64url: str,
        classification: PhotoClassification,
        pcs_code: int,
        exif_data: dict[str, Any],
    ) -> None:
        from .pcs import PCS_CATEGORIES

        cat = PCS_CATEGORIES.get(pcs_code)
        pcs_name = cat.name if cat else "miscellaneous"

        self.catalog.upsert(
            sha256_b64url=sha256_b64url,
            original_path=str(source_path),
            organized_path=str(organized_path),
            pcs_version=classification.pcs_version,
            pcs_primary=pcs_code,
            pcs_name=pcs_name,
            secondary_tags=classification.secondary_tags,
            ai_caption=classification.caption,
            objects=classification.objects,
            ocr_text=classification.ocr_text,
            exif=exif_data,
            model_version=classification.model_version,
            processing_history=[
                {
                    "event": "processed",
                    "source": str(source_path),
                    "destination": str(organized_path),
                }
            ],
        )

    def _copy_to_duplicates(self, source_path: Path, sha256_b64url: str) -> None:
        assert self.duplicates_dir is not None
        self.duplicates_dir.mkdir(parents=True, exist_ok=True)
        dest = self.duplicates_dir / f"{sha256_b64url[:8]}_{source_path.name}"
        shutil.copy2(str(source_path), str(dest))

    def _write_sidecar(
        self,
        organized_path: Path,
        source_path: Path,
        sha256_b64url: str,
        classification: PhotoClassification,
        exif_data: dict[str, Any],
    ) -> None:
        from .filename import parse_filename

        parsed = parse_filename(organized_path.name)
        metadata: dict[str, Any] = {
            "sha256_b64url": sha256_b64url,
            "original_path": str(source_path),
            "organized_path": str(organized_path),
            "pcs_code": parsed["pcs_code"] if parsed else classification.pcs_code,
            "descriptor": parsed["descriptor"] if parsed else classification.descriptor,
            "caption": classification.caption,
            "objects": classification.objects,
            "secondary_tags": classification.secondary_tags,
            "ocr_text": classification.ocr_text,
            "model_version": classification.model_version,
            "exif": exif_data,
        }
        write_sidecar(organized_path, metadata)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_result(result: ProcessResult) -> None:
    if result.status == "copied":
        logger.info("Copied  %s -> %s", result.source_path.name, result.organized_path)
    elif result.status == "duplicate":
        logger.info("Dup     %s [%s]", result.source_path.name, result.sha256_b64url[:8])
    elif result.status == "error":
        logger.error("Error   %s: %s", result.source_path.name, result.error)
    else:
        logger.info("Skip    %s", result.source_path.name)
