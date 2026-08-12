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
from pathlib import Path
from typing import TYPE_CHECKING

from . import concept_map, tiers
from .ai_classifier import AIClassifier
from .catalog import Catalog
from .date_resolver import date_from_row
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

    if reclassify:
        # iter_all has no organized_path filter, unlike iter_unenriched -- whose
        # guard exists precisely because Path(None) raises TypeError. --reclassify
        # walks the WHOLE catalog, so it must re-apply that guard itself rather
        # than rely on today's single insert path always populating the column.
        rows = [r for r in catalog.iter_all() if r["organized_path"]]
        if limit is not None:
            rows = rows[:limit]
    else:
        rows = catalog.iter_unenriched(limit)

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

        # Everything below is LOCAL work -- taxonomy, catalog, filesystem --
        # not a backend call, so it must never feed the breaker. It is also
        # isolated per row: the queue is ordered by id, and a row that raises
        # here is marked neither enriched nor failed, so an escaping
        # exception would crash on the same row every subsequent pass and
        # permanently block every row behind it (mirrors
        # Pipeline._process_one, which wraps its whole per-file body for the
        # same reason). An escape would also bypass stats.failed entirely,
        # silently disabling the poison-file quarantine that consumes it.
        try:
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
            date = date_from_row(row)
            old = (date.tier, row["descriptor_tier"] or tiers.DESC_NONE)
            new = (date.tier, tiers.DESC_AI_SUBJECT)
            final_path = actual

            if tiers.is_upgrade(old, new):
                descriptor = normalize_descriptor(content.primary_subject)
                proposed = target_path(
                    organized_dir, date, descriptor, digest,
                    actual.suffix.lstrip(".").lower(),
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
                except OSError as exc:
                    logger.warning("Rename failed for %s: %s", actual.name, exc)
                else:
                    # Carry the sidecar along with the file it describes.
                    # Kept out of the block above so a sidecar-only failure
                    # is never misattributed as a rename failure.
                    try:
                        old_sidecar = sidecar_path_for(actual)
                        if old_sidecar.exists():
                            old_sidecar.replace(sidecar_path_for(proposed))
                    except OSError as exc:
                        logger.warning(
                            "Sidecar carry failed for %s: %s", actual.name, exc
                        )
            elif str(actual) != row["organized_path"]:
                # Self-healed a stale path without otherwise changing anything.
                # The sidecar must follow the file here too. Without this, a
                # file whose descriptor tier already blocks an AI rename (a
                # human filename) but which was relocated externally would
                # leave its sidecar orphaned at the old path -- and the merge
                # below would then build a fresh one at the new location from
                # an empty base, silently dropping the facts pass's
                # identity/sources/date/descriptor data.
                old_sidecar = sidecar_path_for(recorded)
                new_sidecar = sidecar_path_for(actual)
                if old_sidecar.exists() and old_sidecar != new_sidecar:
                    old_sidecar.replace(new_sidecar)
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
        except Exception as exc:
            logger.exception(
                "Post-perception enrichment failed for %s: %s", actual.name, exc
            )
            stats.errors += 1
            stats.failed.append(digest)
            continue

    return stats
