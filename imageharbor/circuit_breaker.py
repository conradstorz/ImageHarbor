"""A minimal three-state circuit breaker for the AI backend.

Pure state machine: no I/O, no knowledge of files/HTTP/catalog. The watcher (or
the one-shot pipeline) feeds it success/failure signals and reads its state to
decide whether to back off. Time is injected (``now``) so backoff is testable
without real sleeps.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        trip_threshold: int = 5,
        backoff_base: float = 60.0,
        backoff_multiplier: float = 2.0,
        backoff_cap: float = 900.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.trip_threshold = trip_threshold
        self.backoff_base = backoff_base
        self.backoff_multiplier = backoff_multiplier
        self.backoff_cap = backoff_cap
        self._now = now
        self._state = BreakerState.CLOSED
        self._consecutive = 0
        self._backoff = backoff_base
        self._opened_at = 0.0

    @property
    def enabled(self) -> bool:
        return self.trip_threshold > 0

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def current_backoff(self) -> float:
        return self._backoff

    def is_open(self) -> bool:
        return self._state is BreakerState.OPEN

    def is_half_open(self) -> bool:
        return self._state is BreakerState.HALF_OPEN

    def record_success(self) -> None:
        if self._state is BreakerState.HALF_OPEN:
            self._state = BreakerState.CLOSED
            self._backoff = self.backoff_base
        self._consecutive = 0

    def record_failure(self) -> None:
        if not self.enabled:
            return
        if self._state is BreakerState.HALF_OPEN:
            self._open(reopen=True)
            return
        self._consecutive += 1
        if self._consecutive >= self.trip_threshold:
            self._open(reopen=False)

    def _open(self, *, reopen: bool) -> None:
        if reopen:
            self._backoff = min(self._backoff * self.backoff_multiplier, self.backoff_cap)
        else:
            self._backoff = self.backoff_base
        self._state = BreakerState.OPEN
        self._opened_at = self._now()
        self._consecutive = 0

    def seconds_until_probe(self) -> float:
        if self._state is not BreakerState.OPEN:
            return 0.0
        return max(0.0, self._opened_at + self._backoff - self._now())

    def begin_probe(self) -> None:
        if self._state is BreakerState.OPEN and self.seconds_until_probe() <= 0.0:
            self._state = BreakerState.HALF_OPEN
