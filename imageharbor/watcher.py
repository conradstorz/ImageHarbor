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
import math
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .ai_classifier import AIClassifier
from .catalog import Catalog
from .circuit_breaker import CircuitBreaker
from .discovery import discover_images
from .enrich import EnrichStats, enrich_library
from .pipeline import Pipeline

if TYPE_CHECKING:
    from .dashboard.control import ControlPlane
    from .faces.detect import Detector
    from .faces.embed import Embedder
    from .faces.runner import QualityGate
    from .faces.store import FaceStore

logger = logging.getLogger(__name__)

# Above this many unclustered faces, `watch()`'s third pass reclusters the
# whole library even without a fresh cluster-triggering event -- see
# `FacesConfig.recluster_threshold` and the faces-pass block in `watch()`.
DEFAULT_FACE_RECLUSTER_THRESHOLD = 500

# How many consecutive ABORTED enrichment passes (see `watch()`) trigger a
# single diagnostic warning that enrichment is making no progress. ~10 passes
# at the 900s backoff cap is roughly 2.5 hours -- long enough that a couple of
# slow probes against a flaky-but-recovering backend don't false-positive,
# short enough that a genuinely stuck watcher gets flagged well within a day.
CONSECUTIVE_ABORT_WARNING_THRESHOLD = 10

# A defensive ceiling on any single `sleep()` call this loop makes, in
# seconds (~1 year). `control.py` validates `interval` with `math.isfinite`
# at both write time (`set_override`) and read time (`_parse_interval`), so
# a non-finite value should never reach here -- but `_safe_sleep` below is a
# second, independent guard: belt-and-braces, not the primary fix. Without
# it, a value that somehow slips past the store (a future caller that
# doesn't go through ControlPlane, a hand-rolled test double, `math.inf`
# passed directly) would reach `Event.wait(math.inf)` and raise
# `OverflowError` -- an unhandled exception in the middle of the watch loop,
# taking the whole watcher down. A year is far longer than any real poll
# interval and still short enough that a wedged watcher recovers within the
# process's lifetime rather than sleeping until the process is manually
# restarted anyway.
_MAX_SLEEP_SECONDS = 365 * 24 * 3600.0


def _safe_sleep(sleep: Callable[[float], bool | None], seconds: float) -> None:
    """Call *sleep(seconds)* defensively -- a bad value must never crash the loop.

    A non-finite (`inf`/`-inf`/`nan`) or negative *seconds* is replaced with
    `_MAX_SLEEP_SECONDS` before the call. `Event.wait` (the default `sleep`)
    additionally raises `OverflowError` for a `float` timeout too large for
    the platform's `time_t` even when it IS finite -- e.g. a stored interval
    of "1e400" parsed by `float()` -- so the call itself is also wrapped
    rather than trusting the pre-check alone.
    """
    if not math.isfinite(seconds) or seconds < 0:
        logger.warning(
            "watch(): refusing to sleep(%r); using %.0fs instead",
            seconds, _MAX_SLEEP_SECONDS,
        )
        seconds = _MAX_SLEEP_SECONDS
    else:
        seconds = min(seconds, _MAX_SLEEP_SECONDS)
    try:
        sleep(seconds)
    except OverflowError:
        logger.warning(
            "watch(): sleep(%r) raised OverflowError; using %.0fs instead",
            seconds, _MAX_SLEEP_SECONDS,
        )
        sleep(_MAX_SLEEP_SECONDS)


@dataclass
class WatchStats:
    passes: int = 0
    # Facts-phase counts. `processed` is the pre-existing combined counter
    # (copied + duplicate, kept for backward compatibility with every
    # existing caller/test); `copied`/`duplicates` are the same two cases
    # split out, added so `runs.copied`/`runs.duplicates` (see
    # `Catalog.run_finish`) can be populated without guessing -- the design
    # doc's "Now" panel example ("12 copied - 3 duplicates - 0 errors")
    # requires the split, which `processed` alone cannot provide.
    processed: int = 0
    copied: int = 0
    duplicates: int = 0
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
    # Faces-phase counts (0 whenever the faces pass did not run this pass,
    # e.g. faces disabled, the 'faces' extra not installed, or no FacesConfig
    # wired in). A face-scan error is recorded in `failed_files` (see
    # `faces.runner.scan`) and counted here, but -- like every other error in
    # this module's facts/faces phases -- it never reaches the circuit
    # breaker, which is reserved for `AIClassifier.describe()` failures.
    faces_scanned: int = 0
    faces_found: int = 0
    faces_rejected: int = 0
    faces_errors: int = 0


def _copy_to_quarantine(quarantine_dir: Path, source_path: str) -> None:
    """Copy a quarantined ORIGINAL into quarantine_dir (originals stay read-only).

    Named by a hash of the source PATH (the failure result carries no digest),
    so distinct paths never collide; identical path => identical bytes.
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    dest = quarantine_dir / f"{prefix}_{Path(source_path).name}"
    shutil.copy2(source_path, str(dest))


def faces_available() -> bool:
    """True when the optional 'faces' extra (onnxruntime) actually imports.

    Reads `faces_pkg.HAS_ONNX` through a fresh `from . import faces as
    faces_pkg` at call time rather than a name bound once at this module's
    own import time -- the same reasoning as `cli.py`'s `_require_onnx`:
    a test's `monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)` patches the
    attribute on the `imageharbor.faces` module object, and only a lookup
    performed after that patch (not a value captured before it) will ever
    see it.
    """
    from . import faces as faces_pkg

    return faces_pkg.HAS_ONNX


@dataclass
class FacesConfig:
    """Everything the faces pass needs that is fixed for the life of one `watch()` run.

    Built once by the caller (`cli.py`'s `watch` command) and handed to
    `watch()` unchanged -- `Detector`/`Embedder` each load an ONNX session,
    which is expensive enough that constructing a fresh pair every cycle
    would be its own performance bug, exactly like `classifier`/`breaker`
    above. The on/off *decision* is a separate, per-cycle concern (see
    `watch()`'s `faces_enabled`/`control.faces_enabled`) -- this object only
    carries what a cycle needs once it has already decided to run.

    `cluster_threshold` has no sensible default: it must be measured by
    `imageharbor faces calibrate` against this library's own anchor photos,
    never guessed (see `docs/deploy-docker.md`'s calibrate-then-cluster
    ordering). Leaving it `None` still lets the pass scan and propagate
    every cycle -- only whole-library clustering is skipped, with one
    warning on the first cycle that would otherwise have clustered (see
    `watch()`).
    """

    store: "FaceStore"
    detector: "Detector"
    embedder: "Embedder"
    crop_dir: Path
    dest: Path
    gate: "QualityGate"
    cluster_threshold: float | None = None
    cluster_min_score: float = 0.6
    cluster_min_support: int = 2
    recluster_threshold: int = DEFAULT_FACE_RECLUSTER_THRESHOLD


def run_pass(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    recursive: bool = True,
    pause_check: Optional[Callable[[], bool]] = None,
    stats: Optional[WatchStats] = None,
) -> WatchStats:
    """Process new/changed files once (the facts phase only).

    Unchanged files (per the source_seen cache) are skipped without hashing.
    This pass makes no AI calls -- it never consults a circuit breaker and
    never feeds one. Every error it can return is an I/O error (a permissions
    problem, an unreadable file, a full disk), and feeding those into the AI
    breaker would let a filesystem fault masquerade as a backend outage.

    *pause_check*, when given, is consulted BEFORE each file -- mirroring
    `Pipeline.run`'s own guarantee -- so a pause always stops between files,
    never mid-copy.

    *stats*, when given, is mutated in place and also returned, instead of a
    fresh `WatchStats` being created. This lets a caller (`run_once`) keep a
    reference to the counts reached so far even if this function raises
    partway through an iteration (e.g. a catalog write failure) -- an
    exception here loses nothing already recorded in the shared object,
    which is what makes "the counts recorded so far" in a crashed pass's
    `runs` row possible.
    """
    if stats is None:
        stats = WatchStats()
    try:
        for path in discover_images(source, recursive=recursive):
            if pause_check is not None and pause_check():
                logger.info(
                    "Paused after %d file(s) this pass; stopping cleanly",
                    stats.processed,
                )
                break
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
                if result.status == "copied":
                    stats.copied += 1
                else:
                    stats.duplicates += 1
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

    Known, deliberate limitation: quarantine requires a HEALTHY pass (>=1
    success, not tripped). If the poison files remaining in `iter_unenriched`
    ever constitute the ENTIRE unenriched queue (no describable file left
    anywhere to probe into), no pass can ever satisfy that condition for
    them: a full pass trips before any success can land, and a half-open
    probe re-opens on its first (poison) failure before a success can land
    either. Those files are therefore structurally un-quarantinable in that
    state -- and this is intentional, not a bug: an all-poison remaining
    queue is information-theoretically indistinguishable from a real backend
    outage, and quarantining anyway would risk condemning an entire library
    during a real outage, which is exactly what the "tripped -> discard" and
    "no success -> discard" rules above exist to prevent. The cost is
    bounded: `watch()`'s rotating probe offset still attempts one half-open
    probe per backoff interval (capped at `breaker.backoff_cap`, 900s by
    default), so as soon as even one describable file reappears in the
    queue (a new photo arrives, a poison file's bytes change, etc.) normal
    quarantine accounting resumes. See also `watch()`'s
    CONSECUTIVE_ABORT_WARNING_THRESHOLD, which logs a diagnostic (not an
    error, and not a quarantine action) if this state persists.
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

    `EnrichStats.ai_failed` carries digests (enrichment reads the organized
    copy, not the original), but poison-quarantine bookkeeping
    (`failed_files`) is keyed by the ORIGINAL source path -- so each digest is
    resolved back to every known source it was seen at via
    `catalog.sources_for`. A digest with no recorded source (should not
    happen) is skipped rather than crashing the pass. Callers MUST pass only
    `ai_failed`, never `io_failed` -- an I/O failure (a missing organized
    file, a local catalog/filesystem error) is not AI-perception evidence
    about the original source file.
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
    pause_check: Optional[Callable[[], bool]] = None,
) -> tuple[WatchStats, Optional[EnrichStats]]:
    """One full sweep: the facts phase, then the enrichment phase.

    Each phase that actually runs writes exactly one `runs` row
    (`Catalog.run_start`/`run_finish`) -- 'facts' always, 'enrich' only when
    the enrichment phase is entered below (not when it is skipped because
    enrichment is disabled, no classifier is configured, or the breaker is
    OPEN: a row should mean a pass actually happened). `run_finish` is
    always reached via a `finally`, so a phase that raises still closes its
    row -- with whatever counts had accumulated in the *shared* `WatchStats`/
    `EnrichStats` object before the crash (see `run_pass`'s `stats` param),
    plus one additional error for the crash itself -- rather than leaving
    `ended_at` NULL forever (which the dashboard reads as "this process
    died"). The exception itself is NOT swallowed here: it re-raises after
    the row is closed, so a facts-phase crash also skips the enrichment
    phase this sweep (same as before this feature: a badly broken source
    tree should not go on to also attempt enrichment). `watch()` is what
    catches it, logs it, and moves on to the next pass -- see its docstring.

    *pause_check* is forwarded to both `run_pass` and `enrich_library`; each
    is consulted BEFORE that phase's next file/row, never mid-item. Whether
    a given phase's row is recorded `paused=1` is decided by re-reading
    *pause_check* right after that phase returns (or raises) -- since a
    phase's own pause_check is what caused it to stop early, if it is still
    true immediately afterward, this pass ended because of a pause.

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
    per-digest AI-perception failures (`EnrichStats.ai_failed`); the facts
    phase can no longer produce an AI-caused failure, so it never feeds
    quarantine, and neither does `EnrichStats.io_failed` (a missing organized
    file, a local catalog/filesystem error after perception already
    succeeded) -- only a `classifier.describe()` failure is evidence about
    the AI backend or the original source file.

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

    def _breaker_state() -> str:
        # Read fresh at each call site rather than once: the enrichment
        # phase below can trip/close the breaker while it runs, so "the
        # breaker state at pass end" must be read AFTER that phase, not
        # before. Lower-case to match `BreakerState.value` (see
        # `circuit_breaker.py`) and `dashboard/stats.py`'s own use of it --
        # NOT the schema's upper-case SQL `DEFAULT 'CLOSED'`, which is never
        # actually written by application code (this function always
        # supplies an explicit value).
        return breaker.state.value if breaker is not None else "closed"

    def _paused_now() -> bool:
        return bool(pause_check()) if pause_check is not None else False

    # -- facts phase -------------------------------------------------------
    facts = WatchStats()
    facts_run_id = catalog.run_start("facts")
    facts_crashed = False
    try:
        run_pass(
            pipeline=pipeline,
            catalog=catalog,
            source=source,
            recursive=recursive,
            pause_check=pause_check,
            stats=facts,
        )
    except Exception:
        facts_crashed = True
        raise
    finally:
        if facts_crashed:
            facts.errors += 1
        catalog.run_finish(
            facts_run_id,
            scanned=facts.processed + facts.skipped_unchanged + facts.errors,
            copied=facts.copied,
            duplicates=facts.duplicates,
            errors=facts.errors,
            enriched=0,
            enrich_failed=0,
            breaker_state=_breaker_state(),
            paused=_paused_now(),
        )

    # -- enrichment phase ----------------------------------------------------
    enrich_stats: Optional[EnrichStats] = None
    if enrich_enabled and classifier is not None:
        if breaker is not None and breaker.is_open():
            logger.info("Breaker open — skipping the enrichment phase this pass")
        else:
            enrich_run_id = catalog.run_start("enrich")
            enrich_crashed = False
            try:
                enrich_stats = enrich_library(
                    catalog, dest, classifier,
                    write_sidecars=write_sidecars,
                    breaker=breaker,
                    offset=offset,
                    pause_check=pause_check,
                )
            except Exception:
                enrich_crashed = True
                raise
            finally:
                # `enrich_library` returns nothing on a crash (unlike
                # `run_pass`, it owns its own EnrichStats internally rather
                # than accepting one to mutate -- out of scope for this
                # module to change), so the counts reached so far are
                # unknowable here; only the crash itself is recorded.
                row_stats = enrich_stats if enrich_stats is not None else EnrichStats()
                row_errors = row_stats.errors + (1 if enrich_crashed else 0)
                # IMPORTANT finding #7 (2026-08-19 whole-branch review): this
                # used to pass `row_errors` into BOTH `errors` and
                # `enrich_failed`, so the dashboard history panel's 24h error
                # figure (which sums `errors` across runs -- see
                # `dashboard/stats.py`'s `_window_summary`) counted every
                # enrichment failure twice: once as itself (`enrich_failed`)
                # and once again as if it were a facts-phase error
                # (`errors`). `errors` on a `runs` row means "facts-phase
                # errors"; an 'enrich'-kind row has no facts phase at all, so
                # it is always 0 here. `enrich_failed` alone carries every
                # AI-perception and post-perception failure this pass hit
                # (`EnrichStats.errors`, which already sums `ai_failed` +
                # `io_failed`, plus the crash-in-flight count) -- see
                # `EnrichStats`'s own fields in enrich.py.
                catalog.run_finish(
                    enrich_run_id,
                    scanned=row_stats.total,
                    copied=0,
                    duplicates=0,
                    errors=0,
                    enriched=row_stats.enriched,
                    enrich_failed=row_errors,
                    breaker_state=_breaker_state(),
                    paused=_paused_now(),
                )

            failed_buffer = _failed_buffer_from_digests(
                catalog, enrich_stats.ai_failed, "enrichment failed"
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
    interval: float,
    recursive: bool = True,
    stop_event: threading.Event | None = None,
    sleep: Callable[[float], bool] | None = None,
    classifier: Optional[AIClassifier] = None,
    breaker: Optional[CircuitBreaker] = None,
    enrich_enabled: bool = True,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
    control: ControlPlane | None = None,
    faces_enabled: bool = False,
    face_config: FacesConfig | None = None,
) -> WatchStats:
    """Run passes until stop_event is set. An immediate first pass runs before
    the first sleep. ``sleep`` defaults to ``stop_event.wait`` so a signal
    interrupts the wait promptly. When the breaker is OPEN, the between-pass
    wait is the breaker's remaining backoff instead of ``interval``; once it
    elapses the next pass runs as a half-open probe.

    ``control``, when given, is a live :class:`~imageharbor.dashboard.control.
    ControlPlane` consulted FRESH on every iteration for four dials:
    ``control.pause_check()``, ``control.interval``, ``control.enrich_enabled``,
    and ``control.faces_enabled``. This loop runs once and lives for the life of
    the container, so anything captured as a plain *value* at call time would
    be frozen at startup -- a dashboard edit would update the UI, persist to
    the database, and never actually change behavior until a restart. Reading
    the object on each pass instead of its values up front is what makes the
    dashboard's dials live rather than decorative. When ``control`` is
    already paused BEFORE a pass starts, this loop sleeps the interval and
    runs NO pass at all, rather than starting one and immediately breaking
    out. ``control.pause_check`` is also forwarded into ``run_once`` (and
    from there into ``run_pass``/``enrich_library``), so a pause that lands
    WHILE a pass is running still takes effect between files/rows, never
    mid-item (see ``Pipeline.run``'s and ``enrich_library``'s own
    ``pause_check`` guarantee) -- that pass's ``runs`` row is then recorded
    ``paused=1`` (see ``run_once``). When ``control`` is ``None``, the
    ``interval``/``enrich_enabled`` parameters below are used exactly as
    before this option existed, and no ``pause_check`` is forwarded at all.

    Each pass is a full facts-then-enrichment sweep (`run_once`): the facts
    phase always runs; the enrichment phase (and therefore the breaker) is
    skipped while the breaker is OPEN.

    A third, independent pass runs after enrichment when faces are enabled
    (``control.faces_enabled`` if ``control`` is given, else the
    ``faces_enabled`` parameter) AND `faces_available()` is true AND
    *face_config* is not ``None``. It is `faces.runner.scan` (per-photo,
    resumable, `should_stop` wired to the same ``pause_check`` the facts and
    enrichment phases use -- so a pause still stops between photos, never
    mid-photo) followed by `faces.runner.propagate_sidecars` (also
    per-photo, and cheap: it only visits digests a confirmation has newly
    outrun). `faces.runner.build_clusters` -- a whole-library reclustering
    -- is deliberately NOT part of that every-cycle work: it only runs when
    `FaceStore.unclustered_face_count` exceeds `face_config.recluster_
    threshold` or no cluster exists yet, exactly mirroring why `watch` never
    passes `--recluster` to a bare `faces cluster` invocation either. A face
    failure is recorded into `failed_files` by `runner.scan` itself (see its
    docstring) and never reaches *breaker* -- that circuit is reserved for
    `AIClassifier.describe()`. If `faces_available()` is false while faces
    are enabled, one warning is logged on the first such cycle only (never
    per cycle -- a permanently-missing optional extra must not flood the log
    for days) and the pass is skipped; organizing and enrichment continue
    unaffected. The same one-warning treatment applies when clustering is
    due but `face_config.cluster_threshold` is still `None` -- shipped empty
    in `docker-compose.yml` on purpose, because the threshold cannot be
    honestly chosen before `imageharbor faces calibrate` has measured it
    against this library's own embeddings.

    The source tree to walk is taken from ``pipeline.source_dir`` and is
    deliberately NOT a separate parameter. `run_once` uses its *source*
    argument only for discovery, while the *pipeline* governs hashing,
    placement and copying; accepting both here would let a caller point
    discovery at one tree and the pipeline at another, and the mismatch
    would be silent. Deriving one from the other makes that unrepresentable.

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

    This does not guarantee every poison file is eventually quarantined: if
    the files that keep tripping the breaker constitute the ENTIRE
    remaining unenriched queue, no pass can ever be both non-tripped and
    contain a success, so quarantine can never fire for them -- see the
    "Known, deliberate limitation" note on `_reconcile_poison`. That state
    is bounded in cost (one probe per backoff interval) but could in
    principle persist indefinitely if the AI backend never recovers and no
    new describable file ever arrives; this loop logs one diagnostic
    warning (see `CONSECUTIVE_ABORT_WARNING_THRESHOLD`) if that happens, so
    it is visible in the logs rather than silent.
    """
    stop_event = stop_event or threading.Event()
    if sleep is None:
        sleep = stop_event.wait  # interruptible sleep
    wstats = WatchStats()
    probe_offset = 0
    consecutive_aborted_passes = 0
    # Latched True the first time each condition is hit, and never reset for
    # the life of this `watch()` call -- both are permanent-until-fixed
    # states (the extra stays uninstalled, or the threshold stays unset,
    # until an operator acts), so warning again every cycle would flood the
    # log for as long as that stays true instead of saying it once. See the
    # faces-pass block below and this function's own docstring.
    faces_unavailable_warned = False
    faces_no_threshold_warned = False
    while not stop_event.is_set():
        # Read the pause flag fresh every iteration -- see the docstring
        # above for why a value captured once at startup would silently
        # defeat this dial. A paused watcher sleeps the (also freshly-read)
        # interval and runs NO pass at all, rather than starting one and
        # immediately breaking out.
        if control is not None and control.pause_check():
            _safe_sleep(sleep, control.interval)
            continue
        if breaker is not None and breaker.is_open():
            wait = breaker.seconds_until_probe()
            if wait > 0:
                _safe_sleep(sleep, wait)
                continue
            breaker.begin_probe()
        # Same freshness requirement as the interval: read on every pass, not
        # once at call time.
        pass_enrich_enabled = (
            control.enrich_enabled if control is not None else enrich_enabled
        )
        try:
            facts, enrich_stats = run_once(
                pipeline.source_dir,
                pipeline.organized_dir,
                catalog,
                classifier=classifier,
                breaker=breaker,
                pipeline=pipeline,
                write_sidecars=pipeline.write_sidecars,
                recursive=recursive,
                enrich_enabled=pass_enrich_enabled,
                poison_max_fails=poison_max_fails,
                quarantine_dir=quarantine_dir,
                offset=probe_offset,
                pause_check=control.pause_check if control is not None else None,
            )
        except Exception:
            # `run_once` already closed this pass's `runs` row(s) in a
            # `finally` (with the counts reached plus one for the crash
            # itself) before re-raising -- see its docstring. Catching here
            # is what keeps a single bad pass from taking the whole watch
            # loop down with it: the dashboard server thread is independent
            # of this loop regardless, but a dead loop would still freeze
            # every *future* pass's data (a new `current_run` would never
            # start), which is exactly what an operator watching the page
            # during an incident must not see.
            logger.exception(
                "watch pass %d crashed; its run row was still closed with "
                "the counts reached, and the loop continues to the next pass",
                wstats.passes + 1,
            )
            wstats.passes += 1
            wstats.errors += 1
            if stop_event.is_set():
                break
            current_interval = control.interval if control is not None else interval
            _safe_sleep(sleep, current_interval)
            continue
        if enrich_stats is not None:
            if enrich_stats.aborted:
                # Skip past a whole cluster rather than crawling it one row
                # at a time -- see the offset explanation in this
                # function's docstring.
                step = max(1, breaker.trip_threshold) if breaker is not None else 1
                probe_offset += step
                consecutive_aborted_passes += 1
                if consecutive_aborted_passes == CONSECUTIVE_ABORT_WARNING_THRESHOLD:
                    # Log exactly once per threshold crossing -- not on every
                    # subsequent pass while it stays >= threshold, and again
                    # if it later resets and climbs back up. Read-only
                    # diagnostic query, separate from the probe's own
                    # offset-based iter_unenriched call above.
                    head = catalog.iter_unenriched(limit=5, offset=0)
                    head_digests = [row["sha256_b64url"] for row in head]
                    logger.warning(
                        "Enrichment has made no progress across %d consecutive "
                        "aborted passes; the AI backend may be down, or every "
                        "remaining file may be undescribable (see "
                        "_reconcile_poison's 'Known, deliberate limitation' "
                        "note). Files currently at the head of the queue: %s",
                        consecutive_aborted_passes,
                        head_digests,
                    )
            else:
                # Covers both a clean completion AND total == 0 (an
                # offset that outran the queue): either way there is
                # nothing gained by staying non-zero.
                probe_offset = 0
                consecutive_aborted_passes = 0
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

        # -- faces pass (third, independent pass) --------------------------
        # Read fresh every iteration, same as pause/interval/enrich_enabled
        # above: a dashboard toggle must take effect on the very next cycle,
        # not stay frozen at whatever `watch()` was started with.
        pass_faces_enabled = (
            control.faces_enabled if control is not None else faces_enabled
        )
        if pass_faces_enabled and face_config is not None:
            # A face-pass crash (as opposed to a per-photo failure, which
            # `runner.scan` already catches and records into `failed_files`
            # itself) must not take the whole watch loop down with it --
            # same "one bad phase doesn't kill the process" rule `run_once`'s
            # own try/except enforces for facts/enrichment above. Never
            # touches *breaker*: that circuit is reserved for
            # `AIClassifier.describe()` failures, and nothing in this block
            # references it.
            try:
                if not faces_available():
                    if not faces_unavailable_warned:
                        logger.warning(
                            "faces enabled but the 'faces' extra is not "
                            "installed (uv sync --extra faces); organizing "
                            "and enrichment continue, the faces pass is "
                            "skipped until it is",
                        )
                        faces_unavailable_warned = True
                else:
                    from .faces import runner as face_runner

                    face_result = face_runner.scan(
                        catalog,
                        face_config.store,
                        face_config.detector,
                        face_config.embedder,
                        face_config.crop_dir,
                        gate=face_config.gate,
                        should_stop=(
                            control.pause_check if control is not None else None
                        ),
                    )
                    face_runner.propagate_sidecars(
                        face_config.store,
                        face_config.dest,
                        face_config.detector.model_name,
                    )

                    embed_model = face_config.embedder.model_name
                    unclustered = face_config.store.unclustered_face_count(embed_model)
                    recluster_due = (
                        unclustered > face_config.recluster_threshold
                        or not face_config.store.cluster_ids()
                    )
                    if recluster_due:
                        if face_config.cluster_threshold is None:
                            if not faces_no_threshold_warned:
                                logger.warning(
                                    "faces: %d unclustered face(s) but no "
                                    "cluster threshold is configured -- run "
                                    "`imageharbor faces calibrate` and set "
                                    "IMAGEHARBOR_FACE_THRESHOLD; clustering "
                                    "is skipped until it is (scanning and "
                                    "sidecar propagation continue)",
                                    unclustered,
                                )
                                faces_no_threshold_warned = True
                        else:
                            photo_names = face_runner.google_names(face_config.dest)
                            face_runner.build_clusters(
                                face_config.store,
                                photo_names,
                                embed_model=embed_model,
                                threshold=face_config.cluster_threshold,
                                min_score=face_config.cluster_min_score,
                                min_support=face_config.cluster_min_support,
                            )

                    wstats.faces_scanned += face_result.scanned
                    wstats.faces_found += face_result.faces
                    wstats.faces_rejected += face_result.rejected
                    wstats.faces_errors += face_result.errors
                    logger.info(
                        "faces pass %d: scanned=%d faces=%d rejected=%d errors=%d",
                        wstats.passes,
                        face_result.scanned,
                        face_result.faces,
                        face_result.rejected,
                        face_result.errors,
                    )
            except Exception:
                logger.exception(
                    "watch pass %d: faces pass crashed; organizing and "
                    "enrichment are unaffected and the loop continues to "
                    "the next pass",
                    wstats.passes,
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
        # Read fresh, same as the pause/enrich_enabled reads above: a
        # dashboard interval change must take effect on THIS sleep, not
        # whatever value `watch()` happened to be called with.
        current_interval = control.interval if control is not None else interval
        _safe_sleep(sleep, current_interval)
    return wstats
