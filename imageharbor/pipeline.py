"""Main photo-organization processing pipeline."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import concept_map
from .ai_classifier import AIClassifier, ContentDescription, StubClassifier
from .catalog import Catalog
from .discovery import discover_images
from .exif_reader import read_exif
from .filename import generate_filename, normalize_descriptor
from .hashing import compute_sha256_b64url, verify_file
from .sidecar import write_sidecar
from .taxonomy import Taxonomy

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
        self.taxonomy = Taxonomy(catalog)
        self.duplicates_dir = duplicates_dir
        self.write_sidecars = write_sidecars
        self.dry_run = dry_run
        # Tracks content digests seen during a dry run, so intra-run duplicates
        # are detected even though the catalog is never written in dry_run mode.
        self._dry_run_seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, recursive: bool = True) -> PipelineStats:
        """Process all images under :attr:`source_dir`.

        Returns a :class:`PipelineStats` summary.
        """
        stats = PipelineStats()
        self._dry_run_seen.clear()
        # A dry run must perform ZERO taxonomy writes: skip seeding entirely.
        if not self.dry_run:
            self.taxonomy.ensure_seeded()
        for image_path in discover_images(self.source_dir, recursive=recursive):
            result = self._process_one(image_path)
            stats.record(result)
            _log_result(result)
        return stats

    def process_file(self, image_path: Path) -> ProcessResult:
        """Process a single image file and return its result."""
        if not self.dry_run:
            self.taxonomy.ensure_seeded()
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

        # Step 2: duplicate detection. During a dry run the catalog is never
        # written, so also treat a digest already seen earlier in this same dry
        # run as a duplicate (intra-run dedup).
        if self.catalog.is_known(sha256_b64url) or (
            self.dry_run and sha256_b64url in self._dry_run_seen
        ):
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

        # Dry-run short-circuit: report the file as "copied" WITHOUT touching the
        # taxonomy or invoking the AI classifier. This must happen before EXIF/
        # classify/resolve so a dry run performs zero taxonomy writes and zero AI
        # calls. Record the digest for intra-run dedup (a later identical-content
        # file in the same dry run is reported as a duplicate, not "copied").
        if self.dry_run:
            self._dry_run_seen.add(sha256_b64url)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="copied",
                organized_path=None,
            )

        # Step 3: EXIF
        exif_data = read_exif(source_path)

        # Step 4: Perception — the AI only describes the image.
        content = self.classifier.describe(source_path, exif_data)

        # Step 5: Organization — our code decides the class (concept-map first,
        # AI fallback). A learned/seed hit is deterministic and network-free; a
        # genuine miss asks the AI to pick a class and memoizes the answer so the
        # next identical subject is a deterministic hit.
        cls = concept_map.class_for(
            content.primary_subject, content.objects, content.scene, self.catalog
        )
        if cls is None:
            cls = self.classifier.pick_class(content, self._classes())
            concept_map.remember(self.catalog, content.primary_subject, cls)

        # Step 6: resolve class -> code; primary_subject is the level-2 label
        # (dedup + optional AI adjudication).
        pcs_code = self.taxonomy.resolve_or_create(
            cls, content.primary_subject, adjudicator=self.classifier.adjudicate
        )
        node = self.taxonomy.get(pcs_code)
        pcs_name = node.label if node else content.primary_subject

        # Step 7: generate filename (pcs_code is a string, e.g. "330" or "540~1")
        descriptor = normalize_descriptor(content.primary_subject)
        extension = source_path.suffix.lstrip(".").lower()
        filename = generate_filename(pcs_code, descriptor, sha256_b64url, extension)

        # Step 8: determine output path from the taxonomy folder tree
        organized_path = (
            self.organized_dir / self.taxonomy.folder_path(pcs_code) / filename
        )

        # Step 8: copy
        organized_path.parent.mkdir(parents=True, exist_ok=True)
        if organized_path.exists() and verify_file(organized_path, sha256_b64url):
            # Resumed/idempotent run: the destination already exists and its
            # bytes verify against the digest. Skip the copy and re-verify and
            # proceed straight to cataloging. This closes the crash-after-copy-
            # before-catalog resume gap and avoids blindly overwriting.
            logger.debug(
                "Destination already present and verified, skipping copy: %s",
                organized_path,
            )
        else:
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
            content=content,
            pcs_code=pcs_code,
            pcs_name=pcs_name,
            exif_data=exif_data,
        )

        # Step 11: optional sidecar. A sidecar-write failure must NOT fail an
        # image that is already copied, verified, and catalogued; log and move
        # on so the result stays "copied".
        if self.write_sidecars:
            try:
                self._write_sidecar(organized_path, source_path, sha256_b64url, pcs_code, descriptor, content, exif_data)
            except Exception:
                logger.warning(
                    "Failed to write sidecar for %s; image is organized and catalogued",
                    organized_path,
                    exc_info=True,
                )

        return ProcessResult(
            source_path=source_path,
            sha256_b64url=sha256_b64url,
            status="copied",
            organized_path=organized_path,
        )

    def _classes(self) -> list[tuple[str, str]]:
        """The 9 fixed top-level classes, as (code, label) pairs."""
        return [(n.code, n.label) for n in self.taxonomy.children(None)]

    def _update_catalog(
        self,
        *,
        source_path: Path,
        organized_path: Path,
        sha256_b64url: str,
        content: ContentDescription,
        pcs_code: str,
        pcs_name: str,
        exif_data: dict[str, Any],
    ) -> None:
        self.catalog.upsert(
            sha256_b64url=sha256_b64url,
            original_path=str(source_path),
            organized_path=str(organized_path),
            pcs_primary=pcs_code,
            pcs_name=pcs_name,
            secondary_tags=content.tags,
            ai_caption=content.caption,
            objects=content.objects,
            ocr_text=content.ocr_text,
            exif=exif_data,
            model_version=content.model_version,
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
        # Use the FULL digest (not an 8-char prefix) to avoid collisions where
        # a shared prefix plus identical basename would silently overwrite a
        # different image. Identical content => identical bytes => harmless.
        dest = self.duplicates_dir / f"{sha256_b64url}_{source_path.name}"
        shutil.copy2(str(source_path), str(dest))

    def _write_sidecar(
        self,
        organized_path: Path,
        source_path: Path,
        sha256_b64url: str,
        pcs_code: str,
        descriptor: str,
        content: ContentDescription,
        exif_data: dict[str, Any],
    ) -> None:
        from .filename import parse_filename

        parsed = parse_filename(organized_path.name)
        metadata: dict[str, Any] = {
            "sha256_b64url": sha256_b64url,
            "original_path": str(source_path),
            "organized_path": str(organized_path),
            "pcs_code": parsed["pcs_code"] if parsed else pcs_code,
            "descriptor": parsed["descriptor"] if parsed else descriptor,
            "caption": content.caption,
            "objects": content.objects,
            "secondary_tags": content.tags,
            "ocr_text": content.ocr_text,
            "model_version": content.model_version,
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
