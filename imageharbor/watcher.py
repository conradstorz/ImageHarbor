"""Continuous polling watcher for ImageHarbor.

Rescans the source on an interval and processes new/changed files, using the
catalog's source_seen cache to skip unchanged files without re-hashing them
(cheap os.stat instead of a full network read). Filesystem event watching is
deliberately not used: inotify does not work reliably over SMB/CIFS mounts.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .catalog import Catalog
from .discovery import discover_images
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class WatchStats:
    passes: int = 0
    processed: int = 0
    skipped_unchanged: int = 0
    errors: int = 0


def run_pass(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    recursive: bool = True,
) -> WatchStats:
    """Process new/changed files once. Unchanged files (per the source_seen
    cache) are skipped without hashing."""
    stats = WatchStats()
    for path in discover_images(source, recursive=recursive):
        try:
            st = path.stat()
        except OSError:
            # File vanished/changed between discovery and stat (common on a
            # networked mount). Count it and move on rather than crashing.
            logger.warning("Could not stat %s; skipping this pass", path, exc_info=True)
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
    return stats


def watch(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    interval: float,
    recursive: bool = True,
    stop_event: threading.Event | None = None,
    sleep: Callable[[float], bool] | None = None,
) -> WatchStats:
    """Run passes until stop_event is set. An immediate first pass runs before
    the first sleep. ``sleep`` defaults to ``stop_event.wait`` so a signal
    interrupts the wait promptly."""
    stop_event = stop_event or threading.Event()
    if sleep is None:
        sleep = stop_event.wait  # interruptible sleep
    wstats = WatchStats()
    while not stop_event.is_set():
        pass_stats = run_pass(
            pipeline=pipeline, catalog=catalog, source=source, recursive=recursive
        )
        wstats.passes += 1
        wstats.processed += pass_stats.processed
        wstats.skipped_unchanged += pass_stats.skipped_unchanged
        wstats.errors += pass_stats.errors
        logger.info(
            "watch pass %d: processed=%d skipped=%d errors=%d",
            wstats.passes,
            pass_stats.processed,
            pass_stats.skipped_unchanged,
            pass_stats.errors,
        )
        if stop_event.is_set():
            break
        sleep(interval)
    return wstats
