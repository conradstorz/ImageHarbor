"""Tests for the AI-backend circuit breaker state machine."""
from __future__ import annotations

from imageharbor.circuit_breaker import BreakerState, CircuitBreaker


def _breaker(clock, **kw):
    kw.setdefault("trip_threshold", 3)
    kw.setdefault("backoff_base", 60.0)
    kw.setdefault("backoff_cap", 900.0)
    return CircuitBreaker(now=lambda: clock[0], **kw)


def test_starts_closed():
    assert _breaker([0.0]).state is BreakerState.CLOSED


def test_trips_open_after_threshold_consecutive_failures():
    b = _breaker([0.0])
    b.record_failure()
    b.record_failure()
    assert b.state is BreakerState.CLOSED  # 2 < 3
    b.record_failure()
    assert b.state is BreakerState.OPEN     # 3 == threshold


def test_success_resets_consecutive_counter():
    b = _breaker([0.0])
    b.record_failure()
    b.record_failure()
    b.record_success()      # counter back to 0
    b.record_failure()
    b.record_failure()
    assert b.state is BreakerState.CLOSED   # only 2 in a row


def test_open_to_half_open_only_after_backoff_elapses():
    clock = [1000.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()  # OPEN at t=1000, backoff=60
    assert b.seconds_until_probe() == 60.0
    b.begin_probe()
    assert b.state is BreakerState.OPEN     # backoff not elapsed -> no transition
    clock[0] = 1060.0
    assert b.seconds_until_probe() == 0.0
    b.begin_probe()
    assert b.state is BreakerState.HALF_OPEN


def test_half_open_success_closes_and_resets_backoff():
    clock = [1000.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock[0] = 1060.0
    b.begin_probe()
    b.record_success()
    assert b.state is BreakerState.CLOSED
    assert b.current_backoff == 60.0


def test_half_open_failure_reopens_with_doubled_capped_backoff():
    clock = [0.0]
    b = _breaker(clock, backoff_base=60.0, backoff_cap=200.0)
    for _ in range(3):
        b.record_failure()          # OPEN, backoff 60
    clock[0] = 60.0
    b.begin_probe()
    b.record_failure()              # reopen, backoff 120
    assert b.state is BreakerState.OPEN
    assert b.current_backoff == 120.0
    clock[0] = 180.0
    b.begin_probe()
    b.record_failure()              # reopen, 240 -> capped at 200
    assert b.current_backoff == 200.0


def test_threshold_zero_disables_breaker():
    b = _breaker([0.0], trip_threshold=0)
    assert b.enabled is False
    for _ in range(50):
        b.record_failure()
    assert b.state is BreakerState.CLOSED
