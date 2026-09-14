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
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from . import concept_map, tiers
from .ai_classifier import AIClassifier, ContentDescription
from .catalog import Catalog
from .date_resolver import date_from_row
from .filename import normalize_descriptor
from .relocate import apply_relocation, resolve_organized_path, target_path
from .sidecar import merge_sidecar, sidecar_path_for
from .taxonomy import Taxonomy

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# What a per-row helper (`_describe_row`/`_apply_enrichment`) tells the loop
# in `enrich_library` to do next, replacing an implicit `continue`/`break`
# with an explicit value the loop itself acts on: "ok" falls through to
# whatever comes after this row's step, "failed_continue" means this row is
# done (the loop's `for` naturally moves on), and "aborted" means the
# breaker just tripped -- the loop must set `stats.aborted = True` and
# `break`.
RowOutcome = Literal["ok", "failed_continue", "aborted"]


@dataclass
class EnrichStats:
    """Aggregated statistics for an enrichment pass."""

    total: int = 0
    enriched: int = 0
    renamed: int = 0
    errors: int = 0
    aborted: bool = False
    # Digests that failed this pass, split by WHY they failed -- only a
    # classifier.describe() failure is AI-perception evidence. `errors`
    # counts both; poison-file quarantine (fed by the watcher) must consume
    # only `ai_failed` -- an I/O failure (a missing organized file, a local
    # catalog/filesystem error after perception already succeeded) says
    # nothing about the AI backend or the ORIGINAL source file, and must
    # never count toward quarantining that original.
    ai_failed: list[str] = field(default_factory=list)
    io_failed: list[str] = field(default_factory=list)


def _describe_row(
    classifier: AIClassifier,
    actual: Path,
    digest: str,
    *,
    breaker: "CircuitBreaker | None",
    stats: EnrichStats,
) -> tuple[RowOutcome, ContentDescription | None]:
    """Perceive one organized image, feeding AI-perception evidence to *breaker*.

    `classifier.describe()` is the only backend call this function makes, so
    a failure here is unambiguous AI-perception evidence -- the only reason
    this function ever touches *breaker*. Returns ``("ok", content)`` on
    success; on failure, ``("aborted", None)`` once that failure trips the
    breaker (the caller must `break`), else ``("failed_continue", None)``
    (the caller must move on to the next row).
    """
    try:
        content = classifier.describe(actual, {})
    except Exception as exc:
        logger.warning("Enrichment failed for %s: %s", actual.name, exc)
        stats.errors += 1
        stats.ai_failed.append(digest)
        if breaker is not None:
            breaker.record_failure()
            if breaker.is_open():
                logger.error(
                    "AI backend appears down — aborting enrichment after "
                    "%d consecutive failures",
                    breaker.trip_threshold,
                )
                return "aborted", None
        return "failed_continue", None
    return "ok", content


def _apply_enrichment(
    *,
    catalog: Catalog,
    taxonomy: Taxonomy,
    classes: list[tuple[str, str]],
    organized_dir: Path,
    digest: str,
    actual: Path,
    recorded: Path,
    row: sqlite3.Row,
    content: ContentDescription,
    classifier: AIClassifier,
    breaker: "CircuitBreaker | None",
    stats: EnrichStats,
    write_sidecars: bool,
) -> RowOutcome:
    """Everything for one row that runs AFTER perception has already succeeded.

    This is LOCAL work -- taxonomy, catalog, filesystem -- not a backend
    call, so a failure anywhere in most of this function must never feed
    *breaker*; it is handled as I/O evidence (``stats.io_failed``, never
    ``ai_failed`` / quarantine) by the single outer try/except that wraps
    the whole block. This isolation is per row: the queue is ordered by id,
    and a row that raises here is marked neither enriched nor failed, so an
    escaping exception would crash on the same row every subsequent pass and
    permanently block every row behind it (mirrors
    `Pipeline._process_one`, which wraps its whole per-file body for the
    same reason). An escape would also bypass ``stats.io_failed`` entirely,
    silently dropping this row from the pass's failure accounting.
    ``class_for()`` and ``remember()`` do real SQLite I/O and must stay
    inside this same outer try for exactly that reason.

    The ``pick_class`` fallback just below is the one exception within this
    block: it is a real backend chat call, exactly like ``describe()`` in
    `_describe_row`, so it keeps its own NESTED try/except that handles a
    failure there as AI evidence instead -- ``ai_failed``, feeds *breaker*,
    aborts the pass on trip -- rather than letting the outer except's
    ``io_failed`` handling swallow it. ``adjudicate()`` is different and is
    NOT special-cased here: ``taxonomy.resolve_or_create`` catches it
    internally and degrades to minting a new code.

    Returns ``"ok"`` on success, ``"failed_continue"`` for any failure that
    should simply move on to the next row, or ``"aborted"`` once a
    ``pick_class`` failure trips *breaker* (the caller must `break`).
    """
    try:
        cls = concept_map.class_for(
            content.primary_subject, content.objects, content.scene, catalog
        )
        if cls is None:
            # The text-only fallback is a BACKEND call on a real classifier
            # (OpenAIClassifier.pick_class -> chat.completions) -- its
            # failure is AI evidence, handled exactly like a describe()
            # failure.
            try:
                cls = classifier.pick_class(content, classes)
            except Exception as exc:
                logger.warning(
                    "Class fallback failed for %s: %s", actual.name, exc
                )
                stats.errors += 1
                stats.ai_failed.append(digest)
                if breaker is not None:
                    breaker.record_failure()
                    if breaker.is_open():
                        logger.error(
                            "AI backend appears down — aborting enrichment "
                            "after repeated failures in the class fallback"
                        )
                        return "aborted"
                return "failed_continue"

            # Recorded here -- once every backend call this row needed
            # (describe, and now pick_class) has actually succeeded --
            # and deliberately BEFORE concept_map.remember() just below
            # (R3 review Minor #5): record_success() keys purely to
            # backend evidence, so a later remember() failure (local
            # SQLite I/O, handled as io_failed by the outer except
            # below) can never suppress recording backend success that
            # already happened.
            if breaker is not None:
                breaker.record_success()
            concept_map.remember(catalog, content.primary_subject, cls)
        else:
            # The concept map already knew this subject -- no backend
            # call beyond describe() was needed for this row -- so
            # success is recorded here instead, on the identical
            # evidence rule: purely backend, before any further local
            # work. (Recording success right after describe() instead
            # of here would reset the breaker's consecutive-failure
            # counter on every row even when that row's pick_class call
            # then fails, making a run of pick_class-only failures
            # unable to ever reach trip_threshold -- which is why this
            # isn't hoisted up to right after describe() either.)
            if breaker is not None:
                breaker.record_success()

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
            updates: dict = {
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
            }
            if final_path is not actual:
                # The rename fired: record the descriptor the filename
                # now carries, so sidecar, filename, and catalog agree.
                # Deferred-known-issues #9.
                updates["descriptor"] = {
                    "value": descriptor,
                    "tier": tiers.DESC_AI_SUBJECT,
                    "source": tiers.DESC_SOURCE_NAMES[tiers.DESC_AI_SUBJECT],
                }
            try:
                merge_sidecar(final_path, updates)
            except Exception:
                logger.warning(
                    "Failed to update sidecar for %s", final_path, exc_info=True
                )
    except Exception as exc:
        logger.exception(
            "Post-perception enrichment failed for %s: %s", actual.name, exc
        )
        stats.errors += 1
        stats.io_failed.append(digest)
        return "failed_continue"

    return "ok"


def enrich_library(
    catalog: Catalog,
    organized_dir: Path,
    classifier: AIClassifier,
    *,
    write_sidecars: bool = False,
    breaker: CircuitBreaker | None = None,
    limit: int | None = None,
    offset: int = 0,
    reclassify: bool = False,
    pause_check: Callable[[], bool] | None = None,
) -> EnrichStats:
    """Describe and classify organized images that have not been enriched yet.

    When a *breaker* is supplied, a systemic run of failures trips it and
    aborts the pass -- continuing would only churn a dead backend. *offset*
    is passed straight through to `catalog.iter_unenriched`; the watcher uses
    it to rotate a half-open probe past a stuck head cluster (see
    `watcher.watch`).

    *pause_check*, when given, is consulted BEFORE each row -- never in the
    middle of a row's describe/classify/catalog/rename sequence -- so a
    pause always stops between photos, mirroring the facts pass's guarantee
    in `Pipeline.run`.
    """
    stats = EnrichStats()
    taxonomy = Taxonomy(catalog)
    taxonomy.ensure_seeded()

    if reclassify:
        # iter_all has no organized_path filter, unlike iter_unenriched -- whose
        # guard exists precisely because Path(None) raises TypeError. --reclassify
        # walks the WHOLE catalog, so it must re-apply that guard itself rather
        # than rely on today's single insert path always populating the column.
        #
        # offset is deliberately ignored here: it exists only to let the
        # watcher's half-open probe skip past a poison cluster stuck at the
        # head of iter_unenriched's queue. --reclassify walks the whole
        # catalog on an explicit, one-off user request rather than a
        # recurring probe, so there is no stuck-cluster/livelock condition
        # for an offset to route around.
        rows = [r for r in catalog.iter_all() if r["organized_path"]]
        if limit is not None:
            rows = rows[:limit]
    else:
        rows = catalog.iter_unenriched(limit, offset)

    classes = [(n.code, n.label) for n in taxonomy.children(None)]

    for row in rows:
        if pause_check is not None and pause_check():
            logger.info("Paused after %d row(s); stopping cleanly", stats.total)
            break
        stats.total += 1
        digest = row["sha256_b64url"]
        recorded = Path(row["organized_path"])

        actual = resolve_organized_path(organized_dir, recorded, digest)
        if actual is None:
            logger.error("Organized file missing for %s (%s)", digest, recorded)
            stats.errors += 1
            stats.io_failed.append(digest)
            continue

        outcome, content = _describe_row(
            classifier, actual, digest, breaker=breaker, stats=stats
        )
        if outcome == "aborted":
            stats.aborted = True
            break
        if outcome == "failed_continue":
            continue
        assert content is not None  # "ok" always carries content

        outcome = _apply_enrichment(
            catalog=catalog,
            taxonomy=taxonomy,
            classes=classes,
            organized_dir=organized_dir,
            digest=digest,
            actual=actual,
            recorded=recorded,
            row=row,
            content=content,
            classifier=classifier,
            breaker=breaker,
            stats=stats,
            write_sidecars=write_sidecars,
        )
        if outcome == "aborted":
            stats.aborted = True
            break

    return stats
