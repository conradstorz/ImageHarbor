"""Tests for the dashboard's projection logic.

"When will this finish" is the dashboard's most confident-sounding claim and
the easiest one to get quietly wrong, so it is a pure function with a table of
histories rather than something inferred from a live system.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from imageharbor.dashboard.projections import (
    STATUS_COMPLETE,
    STATUS_PROJECTED,
    STATUS_STALLED,
    STATUS_UNKNOWN,
    project,
)

NOW = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)


def _run(minutes_ago: float, duration_min: float, enriched: int) -> dict:
    start = NOW - timedelta(minutes=minutes_ago)
    return {
        "started_at": start.isoformat(),
        "ended_at": (start + timedelta(minutes=duration_min)).isoformat(),
        "enriched": enriched,
    }


def _steady(n: int = 10, enriched: int = 10, duration_min: float = 10.0) -> list[dict]:
    """n passes, each enriching `enriched` photos in `duration_min` -> 60/hr."""
    return [_run(minutes_ago=(i + 1) * 15, duration_min=duration_min, enriched=enriched)
            for i in range(n)]


# --- the refusals: these matter more than the arithmetic ------------------


def test_paused_never_projects() -> None:
    """A paused system has no rate, whatever its history says."""
    p = project(_steady(), backlog=500, breaker_open=False, paused=True, now=NOW)
    assert p.status == STATUS_STALLED
    assert p.eta_seconds is None
    assert "paused" in p.reason.lower()


def test_an_open_breaker_never_projects() -> None:
    """OPEN means the AI backend is unreachable and enrichment is being skipped."""
    p = project(_steady(), backlog=500, breaker_open=True, paused=False, now=NOW)
    assert p.status == STATUS_STALLED
    assert p.eta_seconds is None
    assert "backend" in p.reason.lower()


def test_no_recent_progress_is_stalled_not_infinite() -> None:
    """Ten passes that enriched nothing is a rate of zero, not a huge ETA."""
    runs = [_run((i + 1) * 15, 10.0, 0) for i in range(10)]
    p = project(runs, backlog=500, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_STALLED
    assert p.eta_seconds is None


def test_an_empty_history_is_unknown_not_a_crash() -> None:
    p = project([], backlog=500, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_UNKNOWN
    assert p.eta_seconds is None


def test_a_single_pass_is_not_a_trend() -> None:
    """One sample is not a rate. Two is the minimum worth extrapolating."""
    p = project([_run(15, 10.0, 10)], backlog=500, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_UNKNOWN


def test_an_empty_backlog_is_complete_not_an_eta_of_zero() -> None:
    p = project(_steady(), backlog=0, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_COMPLETE
    assert p.eta_seconds is None


# --- the arithmetic ------------------------------------------------------


def test_a_steady_rate_projects() -> None:
    """10 photos per 10 minutes = 60/hour; 120 backlog = 2 hours."""
    p = project(_steady(), backlog=120, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_PROJECTED
    assert p.rate_per_hour == pytest.approx(60.0)
    assert p.eta_seconds == pytest.approx(2 * 3600)


def test_the_median_ignores_one_wild_pass() -> None:
    """A lifetime average would be dragged by the outlier; a median is not.

    This is the rule that keeps a backlog burned down at 3am from claiming the
    same rate is available now.
    """
    runs = _steady(n=9)
    runs.append(_run(minutes_ago=200, duration_min=1.0, enriched=10_000))  # absurd
    p = project(runs, backlog=120, breaker_open=False, paused=False, now=NOW)
    assert p.rate_per_hour == pytest.approx(60.0)


def test_only_recent_passes_count() -> None:
    """Old fast passes must not prop up a currently-slow system."""
    recent = [_run((i + 1) * 15, 60.0, 10) for i in range(10)]   # 10/hr
    ancient = [_run(10_000 + i * 15, 1.0, 100) for i in range(20)]  # 6000/hr
    p = project(recent + ancient, backlog=100, breaker_open=False, paused=False, now=NOW)
    assert p.rate_per_hour == pytest.approx(10.0)


# --- totality ------------------------------------------------------------


@pytest.mark.parametrize(
    "runs",
    [
        [{"started_at": "nonsense", "ended_at": "also nonsense", "enriched": 5}],
        [{"started_at": NOW.isoformat(), "ended_at": None, "enriched": 5}],       # in flight
        [{"started_at": NOW.isoformat(), "ended_at": NOW.isoformat(), "enriched": 5}],  # zero duration
        [{}],
        [None],
    ],
)
def test_malformed_runs_never_raise(runs) -> None:
    """A dashboard must not crash on a row a crashed pass left behind."""
    p = project(runs, backlog=10, breaker_open=False, paused=False, now=NOW)
    assert p.status in {STATUS_UNKNOWN, STATUS_STALLED, STATUS_PROJECTED}
