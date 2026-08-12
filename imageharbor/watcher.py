"""Continuous polling watcher for ImageHarbor.

Rescans the source on an interval and organizes new/changed files (the facts
phase), then describes and classifies whatever the AI backend can reach (the
enrichment phase). The catalog's source_seen cache lets the facts phase skip
unchanged files without re-hashing them (cheap os.stat instead of a full
network read) -- this is why the facts phase always goes through `run_pass`
rather than `Pipeline.run()` directly. Filesystem event watching is
deliberately not used: inotify does not work reliably over SMB/CIFS mounts.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .ai_classifier import AIClassifier
from .catalog import Catalog
from .circuit_breaker import CircuitBreaker
from .discovery import discover_images
from .enrich import EnrichStats, enrich_library
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class WatchStats:
    passes: int = 0
    # Facts-phase counts.
    processed: int = 0
    skipped_unchanged: int = 0
    errors: int = 0
    # Enrichment-phase counts (0 whenever enrichment did not run this pass,
    # e.g. no classifier configured or the breaker was OPEN).
    enriched: int = 0
    renamed: int = 0
    enrich_errors: int = 0
    # Quarantine actions taken while reconciling the enrichment phase's
    # per-digest failures (see `run_once`) -- never the facts phase, which can
    # no longer produce an AI-caused failure.
    quarantined: int = 0


def _copy_to_quarantine(quarantine_dir: Path, source_path: str) -> None:
    """Copy a quarantined ORIGINAL into quarantine_dir (originals stay read-only).

    Named by a hash of the source PATH (the failure result carries no digest),
    so distinct paths never collide; identical path => identical bytes.
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    dest = quarantine_dir / f"{prefix}_{Path(source_path).name}"
    shutil.copy2(source_path, str(dest))


def run_pass(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    recursive: bool = True,
) -> WatchStats:
    """Process new/changed files once (the facts phase only).

    Unchanged files (per the source_seen cache) are skipped without hashing.
    This pass makes no AI calls -- it never consults a circuit breaker and
    never feeds one. Every error it can return is an I/O error (a permissions
    problem, an unreadable file, a full disk), and feeding those into the AI
    breaker would let a filesystem fault masquerade as a backend outage.
    """
    stats = WatchStats()
    try:
        for path in discover_images(source, recursive=recursive):
            try:
                st = path.stat()
            except OSError:
                # File vanished/changed between discovery and stat (common on a
                # networked mount). Count it and move on rather than crashing.
                logger.warning(
                    "Could not stat %s; skipping this pass", path, exc_info=True
                )
                stats.errors += 1
                continue
            if catalog.source_is_unchanged(str(path), st.st_size, st.st_mtime_ns):
                stats.skipped_unchanged += 1
                continue
            result = pipeline.process_file(path)
            if result.status in ("copied", "duplicate"):
                # Only record success so a transient error is retried next pass.
                catalog.record_source_seen(
                    str(path), st.st_size, st.st_mtime_ns, result.sha256_b64url
                )
                stats.processed += 1
            elif result.status == "error":
                stats.errors += 1
            # any other status (e.g. a future "skipped") is neither counted as an
            # error nor recorded as seen
    except OSError:
        # The source root vanished or the mount dropped mid-pass (discover_images
        # itself raises here on a flaky network mount). Count it and end the pass
        # rather than crashing the watcher; the next pass retries.
        logger.warning("Source unavailable: %s; skipping this pass", source, exc_info=True)
        stats.errors += 1

    return stats


def _reconcile_poison(
    *,
    catalog: Catalog,
    failed_buffer: list[tuple[str, int, int, str]],
    pass_had_success: bool,
    tripped: bool,
    poison_max_fails: int,
    quarantine_dir: Optional[Path],
    stats: WatchStats,
) -> None:
    """Decide whether this pass's failures count toward poison-quarantine.

    - Breaker tripped this pass  -> systemic outage: discard (never counts).
    - Pass had >=1 success       -> backend proven up: count each failure; a
      file reaching poison_max_fails is quarantined (and optionally copied).
    - Neither                    -> health unknowable: discard (conservative).
    """
    if tripped or not pass_had_success or not failed_buffer:
        return
    for source_path, size, mtime_ns, error in failed_buffer:
        count = catalog.record_file_failure(source_path, size, mtime_ns, error)
        if count >= poison_max_fails:
            catalog.quarantine_file(source_path)
            stats.quarantined += 1
            logger.warning(
                "Quarantined poison file after %d failures: %s (%s)",
                count,
                source_path,
                error,
            )
            if quarantine_dir is not None:
                try:
                    _copy_to_quarantine(quarantine_dir, source_path)
                except OSError:
                    logger.warning(
                        "Failed to copy quarantined file %s to %s; still marked "
                        "quarantined in the catalog",
                        source_path,
                        quarantine_dir,
                        exc_info=True,
                    )


def _failed_buffer_from_digests(
    catalog: Catalog, digests: list[str], error: str
) -> list[tuple[str, int, int, str]]:
    """Map enrichment-failed digests back to (source_path, size, mtime_ns, error).

    `EnrichStats.failed` carries digests (enrichment reads the organized copy,
    not the original), but poison-quarantine bookkeeping (`failed_files`) is
    keyed by the ORIGINAL source path -- so each digest is resolved back to
    every known source it was seen at via `catalog.sources_for`. A digest with
    no recorded source (should not happen) is skipped rather than crashing the
    pass.
    """
    buffer: list[tuple[str, int, int, str]] = []
    for digest in digests:
        sources = catalog.sources_for(digest)
        if not sources:
            logger.warning(
                "No known source path for failed digest %s; cannot reconcile "
                "poison-quarantine for it",
                digest,
            )
            continue
        for row in sources:
            buffer.append((row["source_path"], row["size"], row["mtime_ns"], error))
    return buffer


def run_once(
    source: Path,
    dest: Path,
    catalog: Catalog,
    *,
    classifier: Optional[AIClassifier],
    breaker: Optional[CircuitBreaker] = None,
    pipeline: Optional[Pipeline] = None,
    duplicates_dir: Optional[Path] = None,
    write_sidecars: bool = False,
    recursive: bool = True,
    enrich_enabled: bool = True,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
    offset: int = 0,
) -> tuple[WatchStats, Optional[EnrichStats]]:
    """One full sweep: the facts phase, then the enrichment phase.

    The facts leg MUST go through `run_pass`, not `Pipeline.run()`: `run_pass`
    consults `catalog.source_is_unchanged` (a cheap os.stat) and only
    re-hashes new or changed files, while `Pipeline.run()` re-hashes every
    file it walks -- a full read of the whole library on every pass. Over the
    CIFS NAS mount this watcher exists to serve, that is exactly the cost this
    module was written to avoid.

    The facts phase never consults the breaker -- it makes no AI calls, so a
    dead backend has no bearing on whether the library can be organized. Only
    the enrichment phase is skipped while the breaker is OPEN.

    Poison-file quarantine is driven entirely by the enrichment phase's
    per-digest failures (`EnrichStats.failed`); the facts phase can no longer
    produce an AI-caused failure, so it never feeds quarantine.

    *offset* is forwarded to `enrich_library`/`catalog.iter_unenriched`
    unchanged. `watch` owns the actual probe-offset bookkeeping (advance on
    abort, reset on a clean or empty pass) across repeated calls to this
    function; a caller driving `run_once` directly (as the poison tests do)
    is responsible for its own offset if it wants the same rotating-probe
    behaviour.

    If *pipeline* is not supplied, one is constructed from *source*/*dest*
    (and *duplicates_dir*/*write_sidecars*) for this call.
    """
    if pipeline is None:
        pipeline = Pipeline(
            source, dest, catalog,
            duplicates_dir=duplicates_dir,
            write_sidecars=write_sidecars,
        )

    facts = run_pass(pipeline=pipeline, catalog=catalog, source=source, recursive=recursive)

    enrich_stats: Optional[EnrichStats] = None
    if enrich_enabled and classifier is not None:
        if breaker is not None and breaker.is_open():
            logger.info("Breaker open — skipping the enrichment phase this pass")
        else:
            enrich_stats = enrich_library(
                catalog, dest, classifier,
                write_sidecars=write_sidecars,
                breaker=breaker,
                offset=offset,
            )
            failed_buffer = _failed_buffer_from_digests(
                catalog, enrich_stats.failed, "enrichment failed"
            )
            _reconcile_poison(
                catalog=catalog,
                failed_buffer=failed_buffer,
                pass_had_success=enrich_stats.enriched > 0,
                tripped=enrich_stats.aborted,
                poison_max_fails=poison_max_fails,
                quarantine_dir=quarantine_dir,
                stats=facts,
            )

    return facts, enrich_stats


def watch(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    interval: float,
    recursive: bool = True,
    stop_event: threading.Event | None = None,
    sleep: Callable[[float], bool] | None = None,
    classifier: Optional[AIClassifier] = None,
    breaker: Optional[CircuitBreaker] = None,
    enrich_enabled: bool = True,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
) -> WatchStats:
    """Run passes until stop_event is set. An immediate first pass runs before
    the first sleep. ``sleep`` defaults to ``stop_event.wait`` so a signal
    interrupts the wait promptly. When the breaker is OPEN, the between-pass
    wait is the breaker's remaining backoff instead of ``interval``; once it
    elapses the next pass runs as a half-open probe.

    Each pass is a full facts-then-enrichment sweep (`run_once`): the facts
    phase always runs; the enrichment phase (and therefore the breaker) is
    skipped while the breaker is OPEN.

    This loop owns a rotating probe *offset* into `iter_unenriched`'s
    (fixed `ORDER BY p.id`) queue. Without it, a cluster of files that
    always fail AI perception can settle at the head of the queue and
    livelock enrichment forever: every pass immediately re-hits the same
    cluster, trips the breaker, aborts, and `_reconcile_poison` discards the
    whole pass's failures (a tripped pass never counts toward
    poison-quarantine) -- so `fail_count` never climbs to the quarantine
    threshold and no other file is ever reached either. Advancing the
    offset after an abort makes the next half-open probe skip past a whole
    cluster (`max(1, breaker.trip_threshold)` rows) instead of re-probing
    the same head row, so it can eventually land on a working file
    elsewhere in the queue, close the breaker, and let the pass proceed far
    enough that later poison files fail during a HEALTHY pass -- the only
    condition under which quarantine can fire. The offset resets to 0
    whenever a pass completes without aborting (including the `total == 0`
    case, where an offset that has run past the end of the current queue
    would otherwise leave the breaker half-open forever with nothing left
    to probe).
    """
    stop_event = stop_event or threading.Event()
    if sleep is None:
        sleep = stop_event.wait  # interruptible sleep
    wstats = WatchStats()
    probe_offset = 0
    while not stop_event.is_set():
        if breaker is not None and breaker.is_open():
            wait = breaker.seconds_until_probe()
            if wait > 0:
                sleep(wait)
                continue
            breaker.begin_probe()
        facts, enrich_stats = run_once(
            pipeline.source_dir,
            pipeline.organized_dir,
            catalog,
            classifier=classifier,
            breaker=breaker,
            pipeline=pipeline,
            write_sidecars=pipeline.write_sidecars,
            recursive=recursive,
            enrich_enabled=enrich_enabled,
            poison_max_fails=poison_max_fails,
            quarantine_dir=quarantine_dir,
            offset=probe_offset,
        )
        if enrich_stats is not None:
            if enrich_stats.aborted:
                # Skip past a whole cluster rather than crawling it one row
                # at a time -- see the offset explanation in this
                # function's docstring.
                step = max(1, breaker.trip_threshold) if breaker is not None else 1
                probe_offset += step
            else:
                # Covers both a clean completion AND total == 0 (an
                # offset that outran the queue): either way there is
                # nothing gained by staying non-zero.
                probe_offset = 0
        wstats.passes += 1
        wstats.processed += facts.processed
        wstats.skipped_unchanged += facts.skipped_unchanged
        wstats.errors += facts.errors
        wstats.quarantined += facts.quarantined
        if enrich_stats is not None:
            wstats.enriched += enrich_stats.enriched
            wstats.renamed += enrich_stats.renamed
            wstats.enrich_errors += enrich_stats.errors
        logger.info(
            "watch pass %d: facts[processed=%d skipped=%d errors=%d] "
            "enrich[enriched=%d renamed=%d errors=%d aborted=%s] quarantined=%d",
            wstats.passes,
            facts.processed,
            facts.skipped_unchanged,
            facts.errors,
            enrich_stats.enriched if enrich_stats is not None else 0,
            enrich_stats.renamed if enrich_stats is not None else 0,
            enrich_stats.errors if enrich_stats is not None else 0,
            enrich_stats.aborted if enrich_stats is not None else False,
            facts.quarantined,
        )
        if stop_event.is_set():
            break
        # If the breaker tripped this pass it is now OPEN; let the OPEN branch at
        # the top of the loop govern the wait (the remaining backoff) instead of
        # sleeping the full poll interval. Otherwise a short early backoff
        # (60/120/240s) would be floored at `interval`, delaying the quick
        # recovery probe the exponential schedule intends. No time is lost either
        # way — seconds_until_probe() is absolute-clock based.
        if breaker is not None and breaker.is_open():
            continue
        sleep(interval)
    return wstats
