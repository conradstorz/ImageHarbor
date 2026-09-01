"""Collect the catalog into the ``/api/stats`` document.

``collect(catalog, control, *, breaker=None, now=None, face_store=None)`` is
the single entry point: it composes one small, independent function per
section (``now``, ``library``, ``evidence``, ``queues``, ``history``,
``projection``, ``overrides``, ``faces``) and returns them all as one
plain-dict JSON document, per the design's "one document rather than several
endpoints" rule (see
``docs/superpowers/specs/2026-08-19-dashboard-design.md``, "HTTP surface").

Accepted inconsistency, and the lock that is NOT a fix for it
----------------------------------------------------------------
Two different concurrency properties are in play in this module, and they
must not be conflated. (They once were: an earlier version of this
docstring told a future maintainer to remove the very lock that fixes
property 2 below, in the name of preserving property 1. That would have
reintroduced a measured Critical defect. The two are separate changes with
opposite verdicts -- read both before touching either.)

1. **No consistent snapshot across sections -- still true, still
   deliberate, still correct not to fix.** Every section here is a live SQL
   query against a catalog the watcher may be actively writing to, and none
   of them run inside a shared transaction with each other. A page that
   shows a photo counted in one number and not yet in another for one poll
   interval is a far better trade than blocking the writer for the
   duration of a whole page render. See the design doc's "Accepted
   inconsistency" section. **Do not** wrap `collect()` (or any subset of
   its sections) in a transaction to try to make the sections agree with
   each other -- that remains the wrong fix, for this reason, today as much
   as ever.

2. **Thread-safety on the shared connection -- required, not optional, and
   already fixed below; do not undo it.** The dashboard's HTTP server
   (`daemon_threads = True`) and the watcher loop share one
   `sqlite3.Connection`. `check_same_thread=False` *permits* concurrent
   access to it from multiple threads; it does not make that access safe.
   Measured under realistic concurrent load (dashboard polling + watcher
   writing, 25s): 55 exceptions out of the writer -- `cannot commit - no
   transaction is active`, `SystemError: error return without exception
   set` out of `Catalog.run_finish` -- and a file copied and verified but
   never catalogued. `_library_section`, `_evidence_section`, and
   `_queues_section` below run ad hoc aggregate SQL directly against
   `catalog._conn` (there is no `Catalog` wrapper method for that query),
   so each acquires `catalog.lock` -- the same `threading.RLock` every
   other guarded `Catalog` method takes internally -- around its query
   block. **Do not remove these `with catalog.lock:` blocks**; doing so
   reintroduces the defect measured above. See `catalog.py`'s class
   docstring (CRITICAL finding #2, 2026-08-19 whole-branch review) for the
   full account.

   This lock is not the transaction/snapshot described in point 1, and
   solves a different problem: it only serializes *access to the
   connection object* for the duration of one section's queries -- it does
   not hold the writer off for the whole page render, and it creates no
   consistent view across sections (`_library_section` and
   `_evidence_section` each take and release the lock separately, so the
   watcher is free to write between them). A future maintainer who notices
   the lock and reads it as "oh, so we DO serialize for consistency after
   all, let me extend it to a snapshot" would be making the same mistake as
   the maintainer who would have removed it -- both come from merging
   these two properties into one. They are not one property. Keep them
   separate: no transaction (point 1), but yes lock (point 2).

A failing section must not fail the document
----------------------------------------------
Each section function is wrapped by ``_safe``: if it raises, the exception is
logged and the section's value in the returned document is ``None``. A
dashboard missing one panel is better than a dashboard that 500s -- the
operator still learns everything else, and the page is most wanted when
something is already wrong. ``None`` is reserved for exactly this "the query
could not be answered" case; a genuinely empty/zero result (e.g. a brand-new
catalog) must never come back as ``None`` -- see the per-section docstrings
for how each one keeps those two cases distinct (the lesson from Task 1:
``None``, ``0``, and an empty collection must never do duty for each other).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from imageharbor import tiers
from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.dashboard import projections
from imageharbor.dashboard.control import ControlPlane
from imageharbor.faces.store import FaceStore

logger = logging.getLogger(__name__)

# How many `runs` rows the history section pulls. Generous relative to a
# 5-minute poll interval (that's ~288 passes/day), so a 24h/30d window
# summary computed from this slice is rarely truncated in practice; the
# per-poll `runs` table growth is small so pulling this many rows is cheap.
_HISTORY_RUN_LIMIT = 500

# Runs shown in the raw `history.runs` list -- deliberately smaller than
# _HISTORY_RUN_LIMIT since it is meant for a UI table, not aggregation.
_HISTORY_RUNS_DISPLAYED = 50


def _safe(name: str, fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception(
            "dashboard stats: %r section raised; reporting it as unavailable", name
        )
        return None


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _delta_seconds(later: datetime, earlier: datetime) -> float | None:
    """(later - earlier) in seconds, or None if the two cannot be compared.

    Mirrors ``dashboard/projections.py``'s own tz-awareness guard: `datetime`
    refuses to subtract an aware value from a naive one, and guessing a
    timezone to make the subtraction succeed would be inventing evidence, the
    same mistake the date ladder refuses to make.
    """
    if (later.tzinfo is not None) != (earlier.tzinfo is not None):
        return None
    return (later - earlier).total_seconds()


# ---------------------------------------------------------------------------
# now
# ---------------------------------------------------------------------------


def _now_section(
    catalog: Catalog,
    control: ControlPlane,
    breaker: CircuitBreaker | None,
    now: datetime,
) -> dict:
    """Running/paused/phase/breaker/next-pass, per the design's "Now" panel.

    ``current_run`` is the most recent run row with ``ended_at IS NULL``
    THAT THIS PROCESS ITSELF STARTED (``Catalog.run_started_by_this_process``
    -- see its docstring); ``last_run`` is the most recent row that finished.
    Both are ``None`` -- distinctly, not ``{}`` -- when no such row exists
    yet (e.g. a brand-new catalog that has never run a pass).

    IMPORTANT finding #4 (2026-08-19 whole-branch review): ``current_run``
    used to be simply "the most recent row with ``ended_at IS NULL``",
    including one a crash left behind. `docker-compose.yml` has no
    `stop_grace_period`, so Docker's 10s default can SIGKILL a facts pass
    mid-copy over a large CIFS mount -- and `run_once`'s `finally` (which
    always closes a row, even on an in-process exception) never runs at all
    when the whole process is killed, leaving that row open forever. After a
    restart, the NEW process reading that same catalog would previously
    treat that orphaned row as "the current in-flight pass" -- and if
    `paused` also happened to be persisted `True` (so the new process's
    `watch()` loop never starts a pass at all, per `ControlPlane`'s
    restart-durable pause), the page was PINNED at "PAUSING…" forever,
    because nothing this process ever does can make that stale row's
    `ended_at` stop being NULL. Restricting `current_run` to a row this
    process itself started (tracked in-memory since `Catalog.__init__`,
    never persisted) fixes that: a genuinely live pass is always this
    process's own, so `current_run` can never be a stale orphan. Any OTHER
    unfinished row is now explicitly surfaced as ``crashed_runs`` below,
    instead of the two "unfinished_runs() is implemented and tested but
    called from nowhere in production" and "the documented crash signal
    never appears" complaints this finding also raised.
    """
    recent = catalog.recent_runs(limit=5)
    current_run: dict | None = None
    last_run: dict | None = None
    for row in recent:
        if (
            row["ended_at"] is None
            and current_run is None
            and catalog.run_started_by_this_process(row["id"])
        ):
            current_run = dict(row)
        elif row["ended_at"] is not None and last_run is None:
            last_run = dict(row)
        if current_run is not None and last_run is not None:
            break

    # Any unfinished row that is not the (now correctly identified)
    # in-flight pass is a died-mid-pass leftover from a previous process --
    # see the docstring above. `unfinished_runs()` is unlimited (unlike the
    # `recent_runs(limit=5)` window above), because a stale row can be
    # arbitrarily old once the watcher has been paused since the crash.
    current_id = current_run["id"] if current_run is not None else None
    crashed_runs = [dict(r) for r in catalog.unfinished_runs() if r["id"] != current_id]

    paused = control.paused
    if paused:
        # Pause takes effect at the next file boundary (see the design's
        # "Pause semantics"), so a still-in-flight run while paused IS the
        # "finishing up" state the UI calls PAUSING. `current_run` is now
        # never a stale orphan (see above), so this can no longer stay
        # "pausing" forever after a restart.
        state = "pausing" if current_run is not None else "paused"
    else:
        state = "running"

    phase = current_run["kind"] if current_run is not None else None

    if breaker is None:
        # Distinct from a real state: no breaker was wired in at all (e.g.
        # the facts-only CLI path, which never touches AI or the breaker).
        breaker_info = {"state": None, "enabled": None, "seconds_until_probe": None}
    else:
        breaker_info = {
            "state": breaker.state.value,
            "enabled": breaker.enabled,
            "seconds_until_probe": (
                breaker.seconds_until_probe() if breaker.is_open() else 0.0
            ),
        }

    next_pass_seconds: float | None = None
    if not paused and last_run is not None:
        last_end = _parse_iso(last_run.get("ended_at"))
        if last_end is not None:
            elapsed = _delta_seconds(now, last_end)
            if elapsed is not None:
                next_pass_seconds = max(0.0, control.interval - elapsed)

    return {
        "state": state,
        "paused": paused,
        "phase": phase,
        "current_run": current_run,
        "last_run": last_run,
        "crashed_runs": crashed_runs,
        "breaker": breaker_info,
        "next_pass_seconds": next_pass_seconds,
        "interval": control.interval,
    }


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------


def _library_section(catalog: Catalog) -> dict:
    """Photos, bytes, source paths, date range, Undated count, dedup savings.

    Restricted throughout to ``organized_path IS NOT NULL``: per the facts
    pass (``pipeline.py``), a catalog row only ever exists after its copy was
    written and verified, so this is a defensive filter against a legacy or
    hand-edited row rather than a real code path -- but it keeps this
    section's definition of "in the library" identical to `evidence`'s and
    `queues`'s.

    Digest-level aggregates (bytes, dedup count/savings) are computed from
    `sources`, grouped by digest, because `photos` itself carries no size
    column -- size is recorded once per encountered source path. Content
    addressing guarantees every source row for one digest has the same size,
    so ``MIN(size)`` per digest reads the library's actual (deduplicated)
    byte total, not a sum across duplicate copies.

    CRITICAL finding #2 (2026-08-19 whole-branch review): this section
    reaches ``catalog._conn`` directly for aggregate SQL that has no
    ``Catalog`` wrapper method, so the whole block runs under
    ``catalog.lock`` -- the same lock every guarded ``Catalog`` method takes
    -- rather than racing the watcher's writes on the shared connection.
    ``catalog.count()`` below is itself lock-guarded; ``catalog.lock`` is an
    ``RLock`` precisely so a guarded method can be called from inside a
    block that already holds it, from the same thread, without deadlocking.
    """
    with catalog.lock:
        conn = catalog._conn
        total_photos = catalog.count()

        rows = conn.execute(
            """
            SELECT p.sha256_b64url AS digest, COUNT(s.source_path) AS n, MIN(s.size) AS sz
            FROM photos p
            JOIN sources s ON s.sha256_b64url = p.sha256_b64url
            WHERE p.organized_path IS NOT NULL
            GROUP BY p.sha256_b64url
            """
        ).fetchall()
        total_bytes = sum((r["sz"] or 0) for r in rows)
        duplicates_collapsed = sum(max(0, r["n"] - 1) for r in rows)
        bytes_saved = sum(max(0, r["n"] - 1) * (r["sz"] or 0) for r in rows)

        distinct_source_paths = conn.execute(
            "SELECT COUNT(DISTINCT source_path) AS n FROM sources"
        ).fetchone()["n"]

        date_range = conn.execute(
            "SELECT MIN(date_value) AS mn, MAX(date_value) AS mx FROM photos "
            "WHERE organized_path IS NOT NULL AND date_value IS NOT NULL"
        ).fetchone()

        undated_count = conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE organized_path IS NOT NULL "
            "AND date_tier = ?",
            (tiers.DATE_NONE,),
        ).fetchone()["n"]

        enriched_count = conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE organized_path IS NOT NULL "
            "AND enriched_at IS NOT NULL"
        ).fetchone()["n"]
        unenriched_count = conn.execute(
            "SELECT COUNT(*) AS n FROM photos WHERE organized_path IS NOT NULL "
            "AND enriched_at IS NULL"
        ).fetchone()["n"]

    return {
        "total_photos": total_photos,
        "total_bytes": total_bytes,
        "distinct_source_paths": distinct_source_paths,
        "date_range": {"earliest": date_range["mn"], "latest": date_range["mx"]},
        "undated_count": undated_count,
        "duplicates_collapsed": duplicates_collapsed,
        "bytes_saved": bytes_saved,
        "enriched_count": enriched_count,
        "unenriched_count": unenriched_count,
    }


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


def _evidence_section(catalog: Catalog) -> dict:
    """Date-tier and descriptor-tier distributions -- straight from the columns.

    Every tier value from `tiers.DATE_SOURCE_NAMES`/`DESC_SOURCE_NAMES` is
    always present in the output, highest tier first, with an explicit 0
    count rather than a missing key when nothing currently sits at that tier
    -- an absent key would be indistinguishable from "this section could not
    be queried", which `collect()` already represents a different way (the
    whole section is `None`).

    CRITICAL finding #2: raw ``catalog._conn`` access, same as
    `_library_section` above -- guarded by ``catalog.lock`` for the same
    reason.
    """
    with catalog.lock:
        conn = catalog._conn
        date_counts = {
            row["date_tier"]: row["n"]
            for row in conn.execute(
                "SELECT date_tier, COUNT(*) AS n FROM photos "
                "WHERE organized_path IS NOT NULL GROUP BY date_tier"
            )
        }
        descriptor_counts = {
            row["descriptor_tier"]: row["n"]
            for row in conn.execute(
                "SELECT descriptor_tier, COUNT(*) AS n FROM photos "
                "WHERE organized_path IS NOT NULL GROUP BY descriptor_tier"
            )
        }

    date_tiers = [
        {"tier": t, "source": tiers.DATE_SOURCE_NAMES[t], "count": date_counts.get(t, 0)}
        for t in sorted(tiers.DATE_SOURCE_NAMES, reverse=True)
    ]
    descriptor_tiers = [
        {
            "tier": t,
            "source": tiers.DESC_SOURCE_NAMES[t],
            "count": descriptor_counts.get(t, 0),
        }
        for t in sorted(tiers.DESC_SOURCE_NAMES, reverse=True)
    ]
    return {"date_tiers": date_tiers, "descriptor_tiers": descriptor_tiers}


# ---------------------------------------------------------------------------
# queues
# ---------------------------------------------------------------------------


def _queues_section(catalog: Catalog, unenriched_count: int | None) -> dict:
    """Unenriched, quarantined (with reasons), failed-but-not-yet-quarantined,
    and pending Takeout members.

    ``unenriched_count`` is passed in, already computed once by `collect()`
    via `Catalog.count_unenriched()` -- see that method's docstring (IMPORTANT
    finding #5, 2026-08-19 whole-branch review) for why this section no
    longer runs its own `len(catalog.iter_unenriched())` (a `SELECT *` that
    fetched and discarded every row just to count them, duplicated with
    `_projection_section`'s identical call) and why the value can be `None`
    (the precomputing call itself failed) rather than crashing this section.

    This module's raw-SQL queries below acquire ``catalog.lock`` around the
    whole block (see CRITICAL finding #2): `dashboard/stats.py` reaches
    `catalog._conn` directly for aggregate SQL that has no `Catalog` wrapper
    method, so it must take the same lock every guarded `Catalog` method
    takes internally, or these queries would race the watcher's writes on
    the shared connection exactly like the wrapped methods used to before
    the lock existed.
    """
    with catalog.lock:
        conn = catalog._conn
        quarantined_rows = conn.execute(
            "SELECT source_path, last_error, fail_count, first_failed_at, last_failed_at "
            "FROM failed_files WHERE quarantined = 1 ORDER BY last_failed_at DESC"
        ).fetchall()
        quarantined = [dict(r) for r in quarantined_rows]

        failed_active_rows = conn.execute(
            "SELECT source_path, last_error, fail_count, first_failed_at, last_failed_at "
            "FROM failed_files WHERE quarantined = 0 ORDER BY last_failed_at DESC"
        ).fetchall()
        failed_active = [dict(r) for r in failed_active_rows]

    takeout = catalog.takeout_status_counts()
    takeout_pending = takeout.get("members", {}).get("pending", 0)

    return {
        "unenriched_count": unenriched_count,
        "quarantined_count": len(quarantined),
        "quarantined": quarantined,
        "failed_active_count": len(failed_active),
        "failed_active": failed_active,
        "takeout_pending": takeout_pending,
        "takeout": takeout,
    }


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def _window_summary(runs: list[dict], now: datetime, window_seconds: float) -> dict | None:
    """Aggregate *runs* whose `started_at` falls within *window_seconds* of *now*.

    Returns ``None`` -- not a dict of zeros -- when *now* is not a usable
    `datetime`: an unreadable clock means every age in the window is
    unreadable too, which is a different fact from "zero passes ran in this
    window" and must not collapse into it.
    """
    if not isinstance(now, datetime):
        return None
    passes = copied = duplicates = errors = enriched = enrich_failed = 0
    for run in runs:
        started = _parse_iso(run.get("started_at"))
        if started is None:
            continue
        age = _delta_seconds(now, started)
        if age is None or not (0 <= age <= window_seconds):
            continue
        passes += 1
        copied += int(run.get("copied") or 0)
        duplicates += int(run.get("duplicates") or 0)
        errors += int(run.get("errors") or 0)
        enriched += int(run.get("enriched") or 0)
        enrich_failed += int(run.get("enrich_failed") or 0)
    return {
        "passes": passes,
        "copied": copied,
        "duplicates": duplicates,
        "errors": errors,
        "enriched": enriched,
        "enrich_failed": enrich_failed,
    }


def _history_section(catalog: Catalog, now: datetime) -> dict:
    """Recent passes and 24h/30d throughput summaries, from the `runs` table."""
    runs = [dict(r) for r in catalog.recent_runs(limit=_HISTORY_RUN_LIMIT)]
    return {
        "runs": runs[:_HISTORY_RUNS_DISPLAYED],
        "last_24h": _window_summary(runs, now, 86400.0),
        "last_30d": _window_summary(runs, now, 30 * 86400.0),
    }


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


def _projection_section(
    catalog: Catalog,
    control: ControlPlane,
    breaker: CircuitBreaker | None,
    now: datetime,
    unenriched_count: int | None,
) -> dict:
    """Wrap `dashboard/projections.project` with this catalog's live inputs.

    ``unenriched_count`` is passed in from `collect()` (IMPORTANT finding
    #5): both this section and `_queues_section` need the same backlog
    count, and each independently running `len(catalog.iter_unenriched())`
    -- a `SELECT *` fetching and discarding every row just to count them --
    measured 0.32s per call at 15k photos, linear, TWICE per `/api/stats`
    poll, on the connection the watcher writes through. `collect()` now
    computes it once via `Catalog.count_unenriched()` (a real `COUNT(*)`)
    and passes the result to both. `None` here means that single
    precomputation failed -- `projections.project`'s own `_parse_backlog`
    already treats an unreadable backlog as `STATUS_UNKNOWN` rather than
    crashing (see its docstring), so this section stays exactly as safe as
    it was when it queried independently -- it just no longer queries
    independently. A previously-failing `queues` section can therefore no
    longer take `projection` down with it (nor vice versa): both read the
    same precomputed, already-`_safe`-guarded value, so one query failure
    degrades both sections identically instead of one crashing.

    `stale_after_seconds` is deliberately NOT `projections.DEFAULT_STALE_AFTER_SECONDS`
    (one day): that default assumes at most about one pass a day is normal,
    which is backwards for this system -- a watcher polling every few minutes
    produces many passes per day, so waiting a full day of silence before
    calling the history "stale" would hide a watcher that has been dead for
    hours. Instead we derive the window from the actual poll cadence: four
    missed passes' worth of silence, floored at one hour so a short interval
    (e.g. 30s in a test or a very chatty deployment) doesn't flag ordinary
    single-pass jitter as staleness.
    """
    # `projections._parse_run` requires `isinstance(run, Mapping)` to accept a
    # row as readable -- and `sqlite3.Row` deliberately does NOT satisfy
    # `Mapping` (it supports index/name lookup but not the full Mapping
    # protocol), so every row would silently fail that isinstance check and
    # be treated as unparseable if passed through as-is. That is exactly the
    # "valid value silently treated as unreadable" failure mode this
    # project's CLAUDE.md warns about, so the rows are converted to plain
    # dicts here before reaching `project()`.
    #
    # IMPORTANT finding #3 (2026-08-19 whole-branch review): filtered to
    # `kind == "enrich"` ONLY. `run_once` (watcher.py) writes one 'facts' row
    # and, when it runs, one 'enrich' row per pass -- and a 'facts' row
    # ALWAYS has `enriched == 0` (the facts phase makes no AI calls and never
    # enriches anything), so mixing the two kinds into one rate sample feeds
    # `projections.project` a valid-looking `0.0/hr` sample for every facts
    # pass under a second, which drags `statistics.median_low` straight to
    # zero -- reported "stalled" on every healthy deployment with a
    # sub-second facts pass (i.e. most real ones), even while enrichment is
    # actively working through a real backlog at a real rate. A facts pass
    # is not a slow enrichment pass; it is a different measurement of a
    # different phase, and averaging them together does not produce a
    # meaningful rate, it destroys the only one that exists.
    runs = [
        dict(r) for r in catalog.recent_runs(limit=_HISTORY_RUN_LIMIT) if r["kind"] == "enrich"
    ]
    breaker_open = breaker.is_open() if breaker is not None else False
    stale_after_seconds = max(4 * control.interval, 3600.0)

    result = projections.project(
        runs,
        unenriched_count,
        breaker_open=breaker_open,
        paused=control.paused,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    return {
        "backlog": result.backlog,
        "rate_per_hour": result.rate_per_hour,
        "eta_seconds": result.eta_seconds,
        "status": result.status,
        "reason": result.reason,
    }


# ---------------------------------------------------------------------------
# faces
# ---------------------------------------------------------------------------


def _faces_section(face_store: FaceStore | None) -> dict:
    """`FaceStore.stats()` under a `"wired"` flag, per the design's People panel.

    Unlike every other section, ``face_store`` being absent is not a query
    failure to hide behind ``_safe``'s ``None`` -- it is the ordinary state
    of a deployment that never enabled `IMAGEHARBOR_FACES` (`cli.py`'s
    `watch` only constructs a `FaceStore` when faces are enabled and the
    `faces` extra is importable). Collapsing that into `None` would make it
    indistinguishable from "the faces query itself raised", which
    `_safe` already reports the same way for every other section -- so this
    function always returns a real dict, with `"wired": False` and the count
    fields explicitly `None` when there is no store to query, mirroring
    `_now_section`'s `breaker is None` handling above.
    """
    if face_store is None:
        return {
            "wired": False,
            "faces": None,
            "scanned": None,
            "clusters": None,
            "people": None,
            "unreviewed": None,
            "singletons": None,
        }
    return {"wired": True, **face_store.stats()}


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------


def _overrides_section(control: ControlPlane) -> dict:
    return control.overrides()


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def collect(
    catalog: Catalog,
    control: ControlPlane,
    *,
    breaker: CircuitBreaker | None = None,
    now: datetime | None = None,
    face_store: FaceStore | None = None,
) -> dict:
    """Build the whole `/api/stats` document.

    `now=None` (the default) means "use the real clock"; any other non-clock
    value that reaches an individual section is that section's own problem to
    detect and refuse (see e.g. `_window_summary`'s `now` check) rather than
    something silently coerced here, since a caller-supplied `now` is exactly
    what makes the projection and history windows testable.

    ``face_store``, when given, is queried for the `"faces"` section
    (`FaceStore.stats()`). It is passed in rather than opened here because a
    `FaceStore` is only constructed by a caller that actually enabled and
    can run the faces pass (see `cli.py`'s `watch`) -- `collect()` itself
    must not decide whether faces are wired up, only report what it is
    handed. Unlike every other section, `face_store=None` is a real,
    expected state (faces never enabled), not a query failure -- see
    `_faces_section`'s docstring for how it stays distinct from `_safe`'s
    `None`.

    Every section is independently wrapped by `_safe`: a query that raises
    logs and reports that section as `None` rather than failing the whole
    document. See the module docstring for why no *transaction* spans these
    sections (deliberate) versus why three of them *do* take `catalog.lock`
    around their own raw-SQL block (required for thread-safety, a different
    property).

    IMPORTANT finding #5 (2026-08-19 whole-branch review): the unenriched
    backlog count is computed exactly ONCE here (via the guarded, `_safe`-
    wrapped `Catalog.count_unenriched()`, a real `COUNT(*)`) and handed to
    both `_queues_section` and `_projection_section`, which previously each
    ran their own `len(catalog.iter_unenriched())` -- a `SELECT *` fetching
    and discarding every row just to count them. `None` (the precomputation
    itself failed) is a legitimate value for both sections to receive; see
    each section's own docstring for how it stays safe on that input.
    """
    resolved_now = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    unenriched_count = _safe("unenriched_count", catalog.count_unenriched)

    return {
        "now": _safe("now", _now_section, catalog, control, breaker, resolved_now),
        "library": _safe("library", _library_section, catalog),
        "evidence": _safe("evidence", _evidence_section, catalog),
        "queues": _safe("queues", _queues_section, catalog, unenriched_count),
        "history": _safe("history", _history_section, catalog, resolved_now),
        "projection": _safe(
            "projection", _projection_section, catalog, control, breaker,
            resolved_now, unenriched_count,
        ),
        "overrides": _safe("overrides", _overrides_section, control),
        "faces": _safe("faces", _faces_section, face_store),
    }
