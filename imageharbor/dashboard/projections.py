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


def _rate(run: Any) -> float | None:
    """Photos per hour for one completed pass, or None if it cannot be read.

    A pass still in flight (`ended_at` NULL) is excluded rather than treated as
    zero-length -- an unfinished pass is not evidence of a rate. So is a row a
    crashed pass left behind, which is the same shape.
    """
    if not isinstance(run, Mapping):
        return None
    start, end = _parse(run.get("started_at")), _parse(run.get("ended_at"))
    if start is None or end is None:
        return None
    hours = (end - start).total_seconds() / 3600.0
    if hours <= 0:
        return None
    try:
        enriched = int(run.get("enriched") or 0)
    except (TypeError, ValueError):
        return None
    return enriched / hours


def project(
    runs: Sequence[Any],
    backlog: int,
    *,
    breaker_open: bool,
    paused: bool,
    now: datetime,
) -> Projection:
    """Project completion of *backlog* from the *runs* history.

    Returns a stalled or unknown status rather than a number whenever the
    evidence does not support one. Never raises.
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

    ordered = [r for r in runs if isinstance(r, Mapping)]
    ordered.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
    rates = [r for r in (_rate(run) for run in ordered[:RECENT_PASSES]) if r is not None]

    if len(rates) < MIN_SAMPLES:
        return Projection(backlog, None, None, STATUS_UNKNOWN,
                          "not enough completed passes to estimate a rate")

    rate = statistics.median(rates)
    if rate <= 0:
        return Projection(backlog, 0.0, None, STATUS_STALLED,
                          "no progress in recent passes")

    return Projection(
        backlog=backlog,
        rate_per_hour=rate,
        eta_seconds=(backlog / rate) * 3600.0,
        status=STATUS_PROJECTED,
        reason=f"median of {len(rates)} recent passes",
    )
