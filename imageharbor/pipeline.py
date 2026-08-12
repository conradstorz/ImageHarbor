"""Main photo-organization processing pipeline.

This is the **facts pass**: it hashes, dedups, reads EXIF, resolves a date and
descriptor from facts alone (never from AI perception), copies, verifies, and
catalogs. It makes NO AI calls at all -- a run with the AI backend
permanently offline is a finished run, not a degraded one. A separate
enrichment pass adds AI descriptions later.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import tiers
from .catalog import Catalog
from .date_resolver import resolve_date
from .descriptor import resolve_descriptor
from .discovery import discover_images
from .enrich import _date_from_row
from .exif_reader import read_exif
from .hashing import compute_sha256_b64url, verify_file
from .relocate import apply_relocation, resolve_organized_path, target_path
from .sidecar import merge_sidecar, sidecar_path_for

if TYPE_CHECKING:
    from .date_resolver import ResolvedDate
    from .descriptor import ResolvedDescriptor

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
    """Deterministic, resumable photo-organization pipeline -- the facts pass.

    Parameters
    ----------
    source_dir:
        Read-only directory containing original photos.
    organized_dir:
        Root of the organized library tree.
    catalog:
        Open :class:`~imageharbor.catalog.Catalog` instance.
    duplicates_dir:
        Where to copy duplicate images. If None, duplicates are skipped.
    write_sidecars:
        Whether to write a JSON sidecar alongside each organized image.
    dry_run:
        When True, no files are written and the catalog is not updated.

    This pass makes no AI calls: it decides placement (date) and naming
    (descriptor) purely from EXIF and the original filename. A separate
    enrichment pass adds AI-derived descriptions later.
    """

    def __init__(
        self,
        source_dir: Path,
        organized_dir: Path,
        catalog: Catalog,
        duplicates_dir: Path | None = None,
        write_sidecars: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_dir = source_dir
        self.organized_dir = organized_dir
        self.catalog = catalog
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

        This pass never calls an AI backend, so there is no breaker to feed and
        no systemic-outage abort: it runs at disk speed and completes.
        """
        stats = PipelineStats()
        self._dry_run_seen.clear()
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
        stat = source_path.stat()

        # Step 2: duplicate detection. A duplicate still records a back-pointer
        # -- the same bytes reachable from another path is information, not
        # noise, and a better-named path can upgrade the file on a later pass.
        if self.catalog.is_known(sha256_b64url) or (
            self.dry_run and sha256_b64url in self._dry_run_seen
        ):
            if not self.dry_run:
                self.catalog.mark_duplicate(sha256_b64url, str(source_path))
                self.catalog.record_source(
                    sha256_b64url, str(source_path), stat.st_size, stat.st_mtime_ns
                )
                self._maybe_upgrade_from_duplicate(source_path, sha256_b64url)
                if self.duplicates_dir:
                    self._copy_to_duplicates(source_path, sha256_b64url)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="duplicate",
            )

        if self.dry_run:
            self._dry_run_seen.add(sha256_b64url)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="copied",
                organized_path=None,
            )

        # Step 3: EXIF (best effort; returns {} rather than raising)
        exif_data = read_exif(source_path)

        # Step 4: facts -- date decides the folder, descriptor decides the name.
        date = resolve_date(source_path, exif_data)
        descriptor = resolve_descriptor(source_path)

        # Step 5: destination
        extension = source_path.suffix.lstrip(".").lower()
        organized_path = target_path(
            self.organized_dir, date, descriptor.value, sha256_b64url, extension
        )

        # Step 6: copy
        organized_path.parent.mkdir(parents=True, exist_ok=True)
        if organized_path.exists() and verify_file(organized_path, sha256_b64url):
            logger.debug(
                "Destination already present and verified, skipping copy: %s",
                organized_path,
            )
        else:
            shutil.copy2(str(source_path), str(organized_path))

            # Step 7: verify before anything is recorded
            if not verify_file(organized_path, sha256_b64url):
                organized_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Integrity check failed after copying {source_path} -> {organized_path}"
                )

        # Step 8: catalog
        self.catalog.upsert(
            sha256_b64url=sha256_b64url,
            original_path=str(source_path),
            organized_path=str(organized_path),
            exif=exif_data,
            date_value=date.date_str,
            date_tier=date.tier,
            date_source=date.source,
            descriptor_value=descriptor.value,
            descriptor_tier=descriptor.tier,
            descriptor_source=descriptor.source,
            processing_history=[
                {
                    "event": "facts",
                    "source": str(source_path),
                    "destination": str(organized_path),
                }
            ],
        )
        self.catalog.record_source(
            sha256_b64url, str(source_path), stat.st_size, stat.st_mtime_ns
        )

        # Step 9: optional sidecar. A sidecar failure must never fail an image
        # that is already copied, verified, and catalogued.
        if self.write_sidecars:
            try:
                self._write_sidecar(
                    organized_path, sha256_b64url, stat.st_size, extension,
                    date, descriptor, exif_data,
                )
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

    def _maybe_upgrade_from_duplicate(
        self, source_path: Path, sha256_b64url: str
    ) -> None:
        """Re-evaluate a known file's tiers against a newly-seen source path.

        Identical bytes mean identical EXIF, but not identical filenames: the
        same photo found at a better-named path can supply a date or a
        descriptor the first copy lacked.
        """
        row = self.catalog.get_by_sha256(sha256_b64url)
        if row is None or not row["organized_path"]:
            return

        date = resolve_date(source_path, {})
        descriptor = resolve_descriptor(source_path)
        old = (row["date_tier"] or 0, row["descriptor_tier"] or 0)
        new = (max(old[0], date.tier), max(old[1], descriptor.tier))
        if not tiers.is_upgrade(old, new):
            return

        recorded = Path(row["organized_path"])
        actual = resolve_organized_path(self.organized_dir, recorded, sha256_b64url)
        if actual is None:
            logger.warning("Cannot upgrade %s: organized file missing", sha256_b64url)
            return

        best_date = date if date.tier >= old[0] else _date_from_row(row)
        best_descriptor = (
            descriptor.value if descriptor.tier >= old[1] else (row["descriptor_value"] or "")
        )
        proposed = target_path(
            self.organized_dir, best_date, best_descriptor, sha256_b64url,
            actual.suffix.lstrip(".").lower(),
        )
        try:
            apply_relocation(actual, proposed)
        except OSError as exc:
            logger.warning("Upgrade rename failed for %s: %s", actual.name, exc)
            return

        old_sidecar = sidecar_path_for(actual)
        if old_sidecar.exists():
            old_sidecar.replace(sidecar_path_for(proposed))

        self.catalog.set_placement(
            sha256_b64url,
            organized_path=str(proposed),
            date_value=best_date.date_str,
            date_tier=best_date.tier,
            date_source=best_date.source,
            descriptor_value=best_descriptor,
            descriptor_tier=new[1],
            descriptor_source=(
                descriptor.source if descriptor.tier >= old[1]
                else (row["descriptor_source"] or "none")
            ),
        )
        logger.info("Upgraded %s from a better-named duplicate", proposed.name)

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
        sha256_b64url: str,
        size: int,
        extension: str,
        date: "ResolvedDate",
        descriptor: "ResolvedDescriptor",
        exif_data: dict[str, Any],
    ) -> None:
        sources = [
            {
                "path": row["source_path"],
                "first_seen": row["first_seen_at"],
                "last_seen": row["last_seen_at"],
            }
            for row in self.catalog.sources_for(sha256_b64url)
        ]
        merge_sidecar(
            organized_path,
            {
                "identity": {
                    "sha256_b64url": sha256_b64url,
                    "size": size,
                    "ext": extension,
                },
                "sources": sources,
                "date": {
                    "value": date.value.isoformat() if date.value else None,
                    "tier": date.tier,
                    "source": date.source,
                },
                "descriptor": {
                    "value": descriptor.value,
                    "tier": descriptor.tier,
                    "source": descriptor.source,
                },
                "exif": exif_data,
            },
        )


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
