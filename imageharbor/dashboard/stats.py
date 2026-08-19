"""Collect the catalog into the ``/api/stats`` document.

``collect(catalog, control, *, breaker=None, now=None)`` is the single entry
point: it composes one small, independent function per section (``now``,
``library``, ``evidence``, ``queues``, ``history``, ``projection``,
``overrides``) and returns them all as one plain-dict JSON document, per the
design's "one document rather than several endpoints" rule (see
``docs/superpowers/specs/2026-08-19-dashboard-design.md``, "HTTP surface").

Accepted inconsistency
-----------------------
Every section here is a live SQL query against a catalog the watcher may be
actively writing to. **This module deliberately takes no lock and opens no
transaction to obtain a consistent snapshot** -- a page that shows a photo
counted in one number and not yet in another for one poll interval is a far
better trade than blocking the writer to render a status page. See the
design doc's "Accepted inconsistency" section. Do not "fix" this with a
transaction or a lock.

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

    ``current_run`` is the most recent run row with ``ended_at IS NULL`` (a
    pass in flight, or one a crash left behind); ``last_run`` is the most
    recent row that finished. Both are ``None`` -- distinctly, not ``{}`` --
    when no such row exists yet (e.g. a brand-new catalog that has never run
    a pass).
    """
    recent = catalog.recent_runs(limit=5)
    current_run: dict | None = None
    last_run: dict | None = None
    for row in recent:
        if row["ended_at"] is None and current_run is None:
            current_run = dict(row)
        elif row["ended_at"] is not None and last_run is None:
            last_run = dict(row)
        if current_run is not None and last_run is not None:
            break

    paused = control.paused
    if paused:
        # Pause takes effect at the next file boundary (see the design's
        # "Pause semantics"), so a still-in-flight run while paused IS the
        # "finishing up" state the UI calls PAUSING.
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
    """
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
    """
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


def _queues_section(catalog: Catalog) -> dict:
    """Unenriched, quarantined (with reasons), failed-but-not-yet-quarantined,
    and pending Takeout members.

    ``unenriched_count`` reuses ``Catalog.iter_unenriched`` rather than a
    hand-duplicated COUNT query, deliberately: that method's exclusion join
    (quarantine correlated on `(source_path, size, mtime_ns)`, see its own
    docstring) is exactly the definition of "still in the queue", and
    duplicating it here would risk the same kind of silent drift the
    project's CLAUDE.md already warns about for the digest-parsing logic.
    """
    conn = catalog._conn
    unenriched_count = len(catalog.iter_unenriched())

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
) -> dict:
    """Wrap `dashboard/projections.project` with this catalog's live inputs.

    The backlog is recomputed here (not read from an already-collected
    `queues` section) so this section can succeed or fail entirely
    independently of `queues` -- one failing section must never take another
    down with it.

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
    backlog = len(catalog.iter_unenriched())
    # `projections._parse_run` requires `isinstance(run, Mapping)` to accept a
    # row as readable -- and `sqlite3.Row` deliberately does NOT satisfy
    # `Mapping` (it supports index/name lookup but not the full Mapping
    # protocol), so every row would silently fail that isinstance check and
    # be treated as unparseable if passed through as-is. That is exactly the
    # "valid value silently treated as unreadable" failure mode this
    # project's CLAUDE.md warns about, so the rows are converted to plain
    # dicts here before reaching `project()`.
    runs = [dict(r) for r in catalog.recent_runs(limit=_HISTORY_RUN_LIMIT)]
    breaker_open = breaker.is_open() if breaker is not None else False
    stale_after_seconds = max(4 * control.interval, 3600.0)

    result = projections.project(
        runs,
        backlog,
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
) -> dict:
    """Build the whole `/api/stats` document.

    `now=None` (the default) means "use the real clock"; any other non-clock
    value that reaches an individual section is that section's own problem to
    detect and refuse (see e.g. `_window_summary`'s `now` check) rather than
    something silently coerced here, since a caller-supplied `now` is exactly
    what makes the projection and history windows testable.

    Every section is independently wrapped by `_safe`: a query that raises
    logs and reports that section as `None` rather than failing the whole
    document. See the module docstring for the reasoning and for why no lock
    or transaction is taken here.
    """
    resolved_now = now if isinstance(now, datetime) else datetime.now(timezone.utc)

    return {
        "now": _safe("now", _now_section, catalog, control, breaker, resolved_now),
        "library": _safe("library", _library_section, catalog),
        "evidence": _safe("evidence", _evidence_section, catalog),
        "queues": _safe("queues", _queues_section, catalog),
        "history": _safe("history", _history_section, catalog, resolved_now),
        "projection": _safe(
            "projection", _projection_section, catalog, control, breaker, resolved_now
        ),
        "overrides": _safe("overrides", _overrides_section, control),
    }
