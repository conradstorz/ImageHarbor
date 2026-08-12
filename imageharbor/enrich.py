"""The AI enrichment pass.

Runs after the facts pass, independently and resumably.  It reads the
*organized copy* rather than the source: the bytes are verified identical, so
enrichment works when the source volume is unmounted.

Enrichment can only ever improve a file.  It writes classification to the
catalog and sidecar unconditionally, but renames the file only when
:func:`~imageharbor.tiers.is_upgrade` says the result is strictly better --
so an AI subject can never displace a human-authored filename, and a repeated
run is a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import concept_map, tiers
from .ai_classifier import AIClassifier
from .catalog import Catalog
from .date_resolver import ResolvedDate
from .filename import normalize_descriptor
from .relocate import apply_relocation, resolve_organized_path, target_path
from .sidecar import merge_sidecar, sidecar_path_for
from .taxonomy import Taxonomy

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class EnrichStats:
    """Aggregated statistics for an enrichment pass."""

    total: int = 0
    enriched: int = 0
    renamed: int = 0
    errors: int = 0
    aborted: bool = False
    # Digests that failed this pass. The watcher feeds these to poison-file
    # reconciliation: with no AI in the facts pass, this is now the ONLY source
    # of the per-file failure signal quarantine depends on.
    failed: list[str] = field(default_factory=list)


def _date_from_row(row) -> ResolvedDate:
    """Rebuild a ResolvedDate from stored catalog columns."""
    raw = row["date_value"]
    value = None
    if raw:
        try:
            value = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            logger.warning("Unparseable stored date %r; treating as undated", raw)
    tier = row["date_tier"] or tiers.DATE_NONE
    return ResolvedDate(
        value=value,
        tier=tier,
        source=row["date_source"] or tiers.DATE_SOURCE_NAMES[tiers.DATE_NONE],
    )


def enrich_library(
    catalog: Catalog,
    organized_dir: Path,
    classifier: AIClassifier,
    *,
    write_sidecars: bool = False,
    breaker: CircuitBreaker | None = None,
    limit: int | None = None,
    reclassify: bool = False,
) -> EnrichStats:
    """Describe and classify organized images that have not been enriched yet.

    When a *breaker* is supplied, a systemic run of failures trips it and
    aborts the pass -- continuing would only churn a dead backend.
    """
    stats = EnrichStats()
    taxonomy = Taxonomy(catalog)
    taxonomy.ensure_seeded()

    rows = list(catalog.iter_all()) if reclassify else catalog.iter_unenriched(limit)
    if reclassify and limit is not None:
        rows = rows[:limit]

    classes = [(n.code, n.label) for n in taxonomy.children(None)]

    for row in rows:
        stats.total += 1
        digest = row["sha256_b64url"]
        recorded = Path(row["organized_path"])

        actual = resolve_organized_path(organized_dir, recorded, digest)
        if actual is None:
            logger.error("Organized file missing for %s (%s)", digest, recorded)
            stats.errors += 1
            stats.failed.append(digest)
            continue

        try:
            content = classifier.describe(actual, {})
        except Exception as exc:
            logger.warning("Enrichment failed for %s: %s", actual.name, exc)
            stats.errors += 1
            stats.failed.append(digest)
            if breaker is not None:
                breaker.record_failure()
                if breaker.is_open():
                    logger.error(
                        "AI backend appears down — aborting enrichment after "
                        "%d consecutive failures",
                        breaker.trip_threshold,
                    )
                    stats.aborted = True
                    break
            continue

        if breaker is not None:
            breaker.record_success()

        # Organization: our code picks the class; the AI is only a fallback.
        cls = concept_map.class_for(
            content.primary_subject, content.objects, content.scene, catalog
        )
        if cls is None:
            cls = classifier.pick_class(content, classes)
            concept_map.remember(catalog, content.primary_subject, cls)

        pcs_code = taxonomy.resolve_or_create(
            cls, content.primary_subject, adjudicator=classifier.adjudicate
        )
        node = taxonomy.get(pcs_code)
        pcs_name = node.label if node else content.primary_subject

        catalog.mark_enriched(
            digest,
            pcs_primary=pcs_code,
            pcs_name=pcs_name,
            secondary_tags=content.tags,
            ai_caption=content.caption,
            objects=content.objects,
            ocr_text=content.ocr_text,
            model_version=content.model_version,
            scene=content.scene,
        )
        stats.enriched += 1

        # Naming: only if strictly better.
        date = _date_from_row(row)
        old = (date.tier, row["descriptor_tier"] or tiers.DESC_NONE)
        new = (date.tier, tiers.DESC_AI_SUBJECT)
        final_path = actual

        if tiers.is_upgrade(old, new):
            descriptor = normalize_descriptor(content.primary_subject)
            proposed = target_path(
                organized_dir, date, descriptor, digest, actual.suffix.lstrip(".").lower()
            )
            try:
                # Filesystem first, catalog second: a crash in between is
                # recovered by digest lookup on the next pass.
                apply_relocation(actual, proposed)
                catalog.set_placement(
                    digest,
                    organized_path=str(proposed),
                    date_value=date.date_str,
                    date_tier=date.tier,
                    date_source=date.source,
                    descriptor_value=descriptor,
                    descriptor_tier=tiers.DESC_AI_SUBJECT,
                    descriptor_source=tiers.DESC_SOURCE_NAMES[tiers.DESC_AI_SUBJECT],
                )
                final_path = proposed
                stats.renamed += 1

                # Carry the sidecar along with the file it describes.
                old_sidecar = sidecar_path_for(actual)
                if old_sidecar.exists():
                    old_sidecar.replace(sidecar_path_for(proposed))
            except OSError as exc:
                logger.warning("Rename failed for %s: %s", actual.name, exc)
        elif str(actual) != row["organized_path"]:
            # Self-healed a stale path without otherwise changing anything.
            catalog.set_placement(
                digest,
                organized_path=str(actual),
                date_value=date.date_str,
                date_tier=date.tier,
                date_source=date.source,
                descriptor_value=row["descriptor_value"] or "",
                descriptor_tier=row["descriptor_tier"] or tiers.DESC_NONE,
                descriptor_source=row["descriptor_source"] or "none",
            )

        if write_sidecars:
            try:
                merge_sidecar(
                    final_path,
                    {
                        "classification": {
                            "pcs_code": pcs_code,
                            "folder_path": taxonomy.folder_path(pcs_code),
                            "primary_subject": content.primary_subject,
                            "scene": content.scene,
                            "caption": content.caption,
                            "objects": content.objects,
                            "tags": content.tags,
                            "ocr_text": content.ocr_text,
                            "model_version": content.model_version,
                        }
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to update sidecar for %s", final_path, exc_info=True
                )

    return stats
