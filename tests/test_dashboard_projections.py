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


# --- the five defects fixed here -------------------------------------------


def test_a_timezone_mismatch_within_a_row_does_not_raise() -> None:
    """A row we cannot read in time is not evidence -- and must not crash the page."""
    naive_end = (NOW.replace(tzinfo=None) + timedelta(minutes=10)).isoformat()
    row = {"started_at": NOW.isoformat(), "ended_at": naive_end, "enriched": 10}
    p = project([row, row], backlog=100, breaker_open=False, paused=False, now=NOW)
    assert p.status in {STATUS_UNKNOWN, STATUS_STALLED}


def test_unparseable_rows_cannot_evict_real_passes_from_the_window() -> None:
    """The window is chosen chronologically, not by how the timestamp text sorts.

    With a string sort, rows beginning with a letter sort ahead of every ISO
    date and starve the window of real passes -- or, with other junk, promote
    fake ones and produce a confidently absurd rate.
    """
    healthy = _steady(n=10)
    junk = [{"started_at": f"zzz-{i}", "ended_at": f"zzz-{i}", "enriched": 1} for i in range(15)]
    p = project(healthy + junk, backlog=120, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_PROJECTED
    assert p.rate_per_hour == pytest.approx(60.0)


def test_a_stale_history_does_not_project() -> None:
    """Flags can be wrong; a wedged watcher may never trip the breaker.

    This is the module's independent defense, which is why it does not simply
    trust the caller's paused/breaker_open bookkeeping.
    """
    stale = [_run(minutes_ago=60 * 24 * 60 + (i + 1) * 15, duration_min=10.0, enriched=10)
             for i in range(10)]
    p = project(stale, backlog=500, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_STALLED
    assert p.eta_seconds is None


def test_the_rate_is_always_one_a_pass_actually_achieved() -> None:
    """An interpolated median invents a rate nobody ever ran at.

    Pinned to `== 10.0`, not `in {10.0, 1000.0}`: the code comments justify
    `median_low` specifically as the conservative choice, and a test that
    accepts either direction does not pin that reasoning -- it would still
    pass if `median_low` were swapped for `median_high`.
    """
    bimodal = ([_run((i + 1) * 15, 60.0, 10) for i in range(5)] +      # 10/hr
               [_run((i + 6) * 15, 6.0, 100) for i in range(5)])       # 1000/hr
    p = project(bimodal, backlog=100, breaker_open=False, paused=False, now=NOW)
    assert p.rate_per_hour == 10.0


def test_an_absurd_eta_is_reported_as_stalled() -> None:
    """1,141 years is a statement that work is not progressing, not an estimate."""
    slow = [_run((i + 1) * 15, 600.0, 1) for i in range(10)]   # 0.1/hr
    p = project(slow, backlog=1_000_000, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_STALLED
    assert p.rate_per_hour is not None     # the operator can still see the rate
    assert p.eta_seconds is None


# --- the third round of defects ---------------------------------------------


@pytest.mark.parametrize("bad_backlog", [None, "abc", float("nan"), -5])
def test_an_unreadable_backlog_is_unknown_not_complete(bad_backlog) -> None:
    """"Cannot read the backlog" must never be reported as "nothing left".

    COMPLETE tells the operator there is nothing to watch; UNKNOWN tells them
    the dashboard cannot currently say. Those are very different claims, and
    only a genuinely-parsed 0 earns COMPLETE.
    """
    p = project(_steady(), backlog=bad_backlog, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_UNKNOWN
    assert p.status != STATUS_COMPLETE


def test_a_genuine_zero_backlog_is_still_complete() -> None:
    p = project(_steady(), backlog=0, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_COMPLETE


def test_a_stale_history_does_not_project_with_a_naive_now() -> None:
    """Guards the staleness check's second failure vector.

    The first fix made an unreadable-age list stalled/unknown when `now` was
    aware and every row's `end` was naive (or vice versa). But when *every*
    row mismatches, `ages` ends up empty either way -- and the original bug
    was `if ages:` silently skipping the whole check on an empty list, which
    let a 60-day-old history read as fresh again. This test pins that an
    empty `ages` list refuses (STATUS_UNKNOWN), it does not fall through to
    STATUS_PROJECTED, with a *naive* `now` against aware row timestamps --
    the mirror image of the aware-`now`-vs-naive-rows case.
    """
    naive_now = NOW.replace(tzinfo=None)
    stale = [_run(minutes_ago=60 * 24 * 60 + (i + 1) * 15, duration_min=10.0, enriched=10)
             for i in range(10)]
    p = project(stale, backlog=500, breaker_open=False, paused=False, now=naive_now)
    assert p.status != STATUS_PROJECTED
    assert p.status == STATUS_UNKNOWN
    assert p.eta_seconds is None


def test_sub_second_passes_do_not_produce_a_rate() -> None:
    """Two 1-millisecond passes must not be read as an 18,000,000/hr rate."""
    fast = [_run(minutes_ago=15, duration_min=1.0 / 60000.0, enriched=5),
            _run(minutes_ago=30, duration_min=1.0 / 60000.0, enriched=5)]
    p = project(fast, backlog=100, breaker_open=False, paused=False, now=NOW)
    assert p.status != STATUS_PROJECTED
    assert p.status == STATUS_UNKNOWN


def test_runs_of_none_does_not_raise() -> None:
    """`runs=None` is exactly what a failed DB query yields."""
    p = project(None, backlog=500, breaker_open=False, paused=False, now=NOW)
    assert p.status == STATUS_UNKNOWN


@pytest.mark.parametrize("bad_now", ["2026-08-19T12:00:00Z", None, 12345])
def test_a_non_datetime_now_does_not_raise(bad_now) -> None:
    """A bad `now` disables the staleness check rather than crashing the page.

    With a real rated history present, staleness cannot be established, so
    the call must refuse (STATUS_UNKNOWN) rather than raise or silently
    project as if the history were fresh.
    """
    p = project(_steady(), backlog=500, breaker_open=False, paused=False, now=bad_now)
    assert p.status == STATUS_UNKNOWN
