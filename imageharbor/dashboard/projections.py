"""Project when outstanding work will finish -- or refuse to.

Pure: sequences in, a dataclass out, no clock of its own and no database. The
caller supplies `now`, which is what makes every case here testable.

The rule this module exists to enforce is that a confident wrong answer is
worse than no answer. A dashboard that says "done in 4 hours" while the AI
backend is unreachable has told the operator something false; saying "stalled"
tells them something true and actionable. This is the same instinct that puts a
photo in `Undated/` rather than guessing a year.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

STATUS_PROJECTED = "projected"
STATUS_STALLED = "stalled"
STATUS_COMPLETE = "complete"
STATUS_UNKNOWN = "unknown"

# How many recent passes inform the rate. Small enough to reflect current
# conditions, large enough that one slow pass does not dominate.
RECENT_PASSES = 10

# Two samples is the minimum worth calling a trend. One pass is an anecdote.
MIN_SAMPLES = 2

# If the most recent *completed* pass ended longer ago than this, the recent
# history is not "recent" -- it's an artifact of a wedged watcher, and this
# module has no way to know whether the world has changed since. This is the
# module's own defense: it does not trust the caller's paused/breaker_open
# flags, because a watcher that is wedged badly enough to go silent for a day
# is exactly the kind of watcher whose bookkeeping might also be stale or
# wrong. The default (one day) is deliberately conservative -- a real
# operational system produces many passes per day, so a full day of silence
# is already an anomaly worth surfacing as "stalled" rather than projecting
# from ancient evidence. The caller (dashboard/stats.py) is expected to pass
# a value derived from its poll interval rather than relying on this default
# in production.
DEFAULT_STALE_AFTER_SECONDS = 86400.0

# If the projected ETA is further out than this, the number is not useful to
# an operator -- it is a way of saying "the backlog is not clearing at the
# current rate" while looking like a real estimate. Default: 90 days.
DEFAULT_MAX_HORIZON_SECONDS = 90 * 86400.0


@dataclass(frozen=True)
class Projection:
    backlog: int
    rate_per_hour: float | None
    eta_seconds: float | None
    status: str
    reason: str


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class _ParsedRun:
    """A run row with its timestamps parsed exactly once.

    Parsing `started_at`/`ended_at` here -- and nowhere else -- is what keeps
    the recency sort and the rate computation from disagreeing about what a
    row's timestamp means.
    """

    start: datetime
    end: datetime | None
    enriched_raw: Any


def _parse_run(run: Any) -> _ParsedRun | None:
    """Parse one row's timestamps, or None if it cannot be placed in time.

    A row whose `started_at` does not parse cannot be ordered chronologically
    at all, so it is excluded here -- before sorting -- rather than merely
    scored low. Excluding it here is what stops a malformed row from evicting
    a real pass from the recency window (it was never in the window to begin
    with) or fabricating a fake ordering (it never influences it).
    """
    if not isinstance(run, Mapping):
        return None
    start = _parse(run.get("started_at"))
    if start is None:
        return None
    end = _parse(run.get("ended_at"))
    return _ParsedRun(start=start, end=end, enriched_raw=run.get("enriched"))


def _safe_delta_seconds(later: datetime, earlier: datetime) -> float | None:
    """(later - earlier) in seconds, or None if they cannot be compared.

    `datetime` refuses to subtract an aware value from a naive one. Rather
    than guess a timezone to make the two comparable -- which would be the
    same class of error as inventing a capture date -- we treat the pair as
    unreadable and let the caller decide that means "not evidence".
    """
    if (later.tzinfo is not None) != (earlier.tzinfo is not None):
        return None
    return (later - earlier).total_seconds()


def _rate(parsed: _ParsedRun) -> float | None:
    """Photos per hour for one completed pass, or None if it cannot be read.

    A pass still in flight (`ended_at` NULL/unparseable) is excluded rather
    than treated as zero-length -- an unfinished pass is not evidence of a
    rate. So is a row a crashed pass left behind, which is the same shape.
    A row whose two timestamps disagree on timezone-awareness is likewise
    excluded: it cannot be read, so it is not evidence either.
    """
    if parsed.end is None:
        return None
    hours = _safe_delta_seconds(parsed.end, parsed.start)
    if hours is None:
        return None
    hours /= 3600.0
    if hours <= 0:
        return None
    try:
        enriched = int(parsed.enriched_raw or 0)
    except (TypeError, ValueError):
        return None
    return enriched / hours


def _format_age(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = seconds / 3600.0
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24.0:.1f} days"


def project(
    runs: Sequence[Any],
    backlog: int,
    *,
    breaker_open: bool,
    paused: bool,
    now: datetime,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    max_horizon_seconds: float = DEFAULT_MAX_HORIZON_SECONDS,
) -> Projection:
    """Project completion of *backlog* from the *runs* history.

    Returns a stalled or unknown status rather than a number whenever the
    evidence does not support one. Never raises.

    `stale_after_seconds` is this module's own defense against a wedged
    watcher: if the most recent *completed* pass ended longer ago than this,
    the history is not projected from, regardless of what `paused` or
    `breaker_open` claim -- those flags are exactly the bookkeeping that can
    be stale or wrong when the watcher itself has stopped updating anything.
    The caller (`dashboard/stats.py`) should pass a value derived from its
    poll interval rather than relying on the default in production.

    `max_horizon_seconds` caps how far out an ETA is allowed to look before
    it is reported as "stalled" instead: an ETA measured in centuries is not
    a useful estimate, it is a statement that the backlog is not clearing at
    the current rate, and it should read as one.
    """
    try:
        backlog = int(backlog)
    except (TypeError, ValueError):
        backlog = 0

    if backlog <= 0:
        return Projection(0, None, None, STATUS_COMPLETE, "nothing outstanding")

    # Refusals first: state beats history. A rate measured before the backend
    # went down says nothing about a system that is not currently working.
    if paused:
        return Projection(backlog, None, None, STATUS_STALLED, "paused")
    if breaker_open:
        return Projection(backlog, None, None, STATUS_STALLED,
                          "backend unreachable (breaker OPEN)")

    # Parse once, sort by the parsed value, and drop anything that cannot be
    # placed in time -- it never enters the recency window at all.
    parsed = [p for p in (_parse_run(r) for r in runs) if p is not None]
    parsed.sort(key=lambda p: p.start, reverse=True)
    window = parsed[:RECENT_PASSES]

    rated = [(p, _rate(p)) for p in window]
    rates = [r for _, r in rated if r is not None]

    if len(rates) < MIN_SAMPLES:
        return Projection(backlog, None, None, STATUS_UNKNOWN,
                          "not enough completed passes to estimate a rate")

    # Independent staleness defense: a history can be entirely well-formed
    # and still be too old to say anything about the present.
    ends = [p.end for p, r in rated if r is not None and p.end is not None]
    ages = [age for age in (_safe_delta_seconds(now, end) for end in ends) if age is not None]
    if ages:
        most_recent_age = min(ages)
        if most_recent_age > stale_after_seconds:
            return Projection(
                backlog, None, None, STATUS_STALLED,
                f"most recent completed pass ended {_format_age(most_recent_age)} ago",
            )

    # median_low, not median: with an even sample count `median` would
    # interpolate between the two middle samples, synthesizing a rate no
    # pass actually achieved. This module reports evidence, not arithmetic
    # invention -- and the low sample is also the conservative choice, which
    # is the right bias for a number someone is about to act on.
    rate = statistics.median_low(rates)
    if rate <= 0:
        return Projection(backlog, 0.0, None, STATUS_STALLED,
                          "no progress in recent passes")

    eta_seconds = (backlog / rate) * 3600.0
    if eta_seconds > max_horizon_seconds:
        # Still surface the rate -- the operator can see the system is
        # working, just not fast enough for the ETA to mean anything.
        return Projection(backlog, rate, None, STATUS_STALLED,
                          "backlog will not clear at the current rate")

    return Projection(
        backlog=backlog,
        rate_per_hour=rate,
        eta_seconds=eta_seconds,
        status=STATUS_PROJECTED,
        reason=f"median of {len(rates)} recent passes",
    )
