"""Continuous polling watcher for ImageHarbor.

Rescans the source on an interval and processes new/changed files, using the
catalog's source_seen cache to skip unchanged files without re-hashing them
(cheap os.stat instead of a full network read). Filesystem event watching is
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

from .catalog import Catalog
from .circuit_breaker import CircuitBreaker
from .discovery import discover_images
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class WatchStats:
    passes: int = 0
    processed: int = 0
    skipped_unchanged: int = 0
    errors: int = 0
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
    breaker: Optional[CircuitBreaker] = None,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
) -> WatchStats:
    """Process new/changed files once. Unchanged files (per the source_seen
    cache) are skipped without hashing. When a breaker is supplied, a systemic
    run of AI failures trips it and aborts the pass early."""
    stats = WatchStats()
    pass_had_success = False
    failed_buffer: list[tuple[str, int, int, str]] = []
    tripped = False
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
            if breaker is not None and catalog.is_quarantined(
                str(path), st.st_size, st.st_mtime_ns
            ):
                # Poison file, already quarantined and unchanged: skip silently.
                stats.skipped_unchanged += 1
                continue
            result = pipeline.process_file(path)
            if result.status in ("copied", "duplicate"):
                # Only record success so a transient error is retried next pass.
                catalog.record_source_seen(
                    str(path), st.st_size, st.st_mtime_ns, result.sha256_b64url
                )
                if breaker is not None:
                    catalog.clear_file_failure(str(path))
                    breaker.record_success()
                pass_had_success = True
                stats.processed += 1
            elif result.status == "error":
                stats.errors += 1
                if breaker is not None:
                    failed_buffer.append(
                        (str(path), st.st_size, st.st_mtime_ns, result.error)
                    )
                    breaker.record_failure()
                    if breaker.is_open():
                        tripped = True
                        logger.warning(
                            "AI backend appears down (%d consecutive failures) "
                            "— backing off %.0fs",
                            breaker.trip_threshold,
                            breaker.seconds_until_probe(),
                        )
                        break
            # any other status (e.g. a future "skipped") is neither counted as an
            # error nor recorded as seen
    except OSError:
        # The source root vanished or the mount dropped mid-pass (discover_images
        # itself raises here on a flaky network mount). Count it and end the pass
        # rather than crashing the watcher; the next pass retries.
        logger.warning("Source unavailable: %s; skipping this pass", source, exc_info=True)
        stats.errors += 1

    # --- poison reconciliation (fully implemented in Task 4) ---
    _reconcile_poison(
        catalog=catalog,
        failed_buffer=failed_buffer,
        pass_had_success=pass_had_success,
        tripped=tripped,
        poison_max_fails=poison_max_fails,
        quarantine_dir=quarantine_dir,
        stats=stats,
    )
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


def watch(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    interval: float,
    recursive: bool = True,
    stop_event: threading.Event | None = None,
    sleep: Callable[[float], bool] | None = None,
    breaker: Optional[CircuitBreaker] = None,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
) -> WatchStats:
    """Run passes until stop_event is set. An immediate first pass runs before
    the first sleep. ``sleep`` defaults to ``stop_event.wait`` so a signal
    interrupts the wait promptly. When the breaker is OPEN, the between-pass
    wait is the breaker's remaining backoff instead of ``interval``; once it
    elapses the next pass runs as a half-open probe."""
    stop_event = stop_event or threading.Event()
    if sleep is None:
        sleep = stop_event.wait  # interruptible sleep
    wstats = WatchStats()
    while not stop_event.is_set():
        if breaker is not None and breaker.is_open():
            wait = breaker.seconds_until_probe()
            if wait > 0:
                sleep(wait)
                continue
            breaker.begin_probe()
        pass_stats = run_pass(
            pipeline=pipeline,
            catalog=catalog,
            source=source,
            recursive=recursive,
            breaker=breaker,
            poison_max_fails=poison_max_fails,
            quarantine_dir=quarantine_dir,
        )
        wstats.passes += 1
        wstats.processed += pass_stats.processed
        wstats.skipped_unchanged += pass_stats.skipped_unchanged
        wstats.errors += pass_stats.errors
        wstats.quarantined += pass_stats.quarantined
        logger.info(
            "watch pass %d: processed=%d skipped=%d errors=%d quarantined=%d",
            wstats.passes,
            pass_stats.processed,
            pass_stats.skipped_unchanged,
            pass_stats.errors,
            pass_stats.quarantined,
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
