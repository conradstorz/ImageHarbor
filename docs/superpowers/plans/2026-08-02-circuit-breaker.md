# AI-Backend Circuit Breaker + Poison-File Quarantine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the watcher detect a systemic AI-backend outage and back off instead of hammering it, recover automatically, and quarantine files that persistently fail while the backend is healthy.

**Architecture:** A new pure `CircuitBreaker` state machine (CLOSED/OPEN/HALF_OPEN) lives in `imageharbor/circuit_breaker.py`. The watcher drives it: on 5 consecutive AI failures it aborts the pass, exponential-backs-off (60s→×2→900s), then probes with one real image. A new catalog `failed_files` table tracks per-file failures; a file that fails 5 *healthy* passes is quarantined (skipped thereafter, optionally copied to a dir). A consecutive-vs-isolated rule ensures a backend outage never mis-quarantines good files. The one-shot `process` command reuses the breaker for detection-only early-abort.

**Tech Stack:** Python 3.12, `uv`, pytest, Click, SQLite (stdlib `sqlite3`). No new dependencies.

## Global Constraints

- Codes/paths untouched — this feature is orchestration-layer only; do NOT modify `ai_classifier.py`, `taxonomy.py`, `pcs.py`, `concept_map.py`, `hashing.py`, or `filename.py`.
- Preserve existing invariants: no data loss (failed files never entered the catalog before and still don't), copy→verify→catalog ordering, read-only originals (quarantine COPIES, never moves), catalog stays local.
- Backward compatibility: `run_pass`/`watch` must behave EXACTLY as today when called without a breaker (`breaker=None`), so all existing `tests/test_watcher.py` tests pass unchanged.
- Use `uv` for everything: `uv run pytest`, never pip/venv. Do NOT chain shell commands with `&&` (run as separate calls).
- Defaults (all configurable): `trip_threshold=5`, `backoff_base=60.0`, `backoff_multiplier=2.0`, `backoff_cap=900.0`, `poison_max_fails=5`. `trip_threshold=0` disables the breaker.
- No exception-type classification: any `ProcessResult.status == "error"` is "a failure".

---

## File Structure

- **Create** `imageharbor/circuit_breaker.py` — pure `CircuitBreaker` state machine + `BreakerState` enum. No I/O.
- **Modify** `imageharbor/catalog.py` — add `failed_files` table to `_SCHEMA` + 4 methods.
- **Modify** `imageharbor/watcher.py` — `WatchStats.quarantined`; breaker + poison logic in `run_pass`; backoff/half-open loop in `watch`; `_copy_to_quarantine` helper.
- **Modify** `imageharbor/pipeline.py` — optional `breaker` param on `Pipeline.run` (detection-only early-abort).
- **Modify** `imageharbor/cli.py` — new `watch` options + breaker construction; `process` breaker option + abort message.
- **Modify** `CLAUDE.md` — document the new module, table, and watcher behavior.
- **Create** `tests/test_circuit_breaker.py`, `tests/test_poison.py`; **extend** `tests/test_catalog.py`, `tests/test_watcher.py`, `tests/test_cli.py`.

---

## Task 1: `CircuitBreaker` pure state machine

**Files:**
- Create: `imageharbor/circuit_breaker.py`
- Test: `tests/test_circuit_breaker.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `class BreakerState(Enum)` with members `CLOSED`, `OPEN`, `HALF_OPEN`.
  - `class CircuitBreaker` with:
    - `__init__(self, trip_threshold=5, backoff_base=60.0, backoff_multiplier=2.0, backoff_cap=900.0, now: Callable[[], float] = time.monotonic)`
    - `record_success() -> None`, `record_failure() -> None`
    - properties `state -> BreakerState`, `enabled -> bool`, `current_backoff -> float`
    - `is_open() -> bool`, `is_half_open() -> bool`
    - `seconds_until_probe() -> float`, `begin_probe() -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_circuit_breaker.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_circuit_breaker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.circuit_breaker'`.

- [ ] **Step 3: Implement `imageharbor/circuit_breaker.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_circuit_breaker.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add imageharbor/circuit_breaker.py tests/test_circuit_breaker.py
git commit -m "feat: add CircuitBreaker state machine for AI-backend outages"
```

---

## Task 2: Catalog `failed_files` table + methods

**Files:**
- Modify: `imageharbor/catalog.py` (add table to `_SCHEMA`; add 4 methods)
- Test: `tests/test_catalog.py` (extend)

**Interfaces:**
- Consumes: existing `Catalog`, `_now_iso`.
- Produces on `Catalog`:
  - `record_file_failure(source_path: str, size: int, mtime_ns: int, error: str) -> int` — upsert; returns new `fail_count`. If stored size/mtime differ from the incoming values, resets `fail_count` to 1 and `quarantined` to 0.
  - `quarantine_file(source_path: str) -> None`
  - `is_quarantined(source_path: str, size: int, mtime_ns: int) -> bool` — True only if a row exists with `quarantined=1` AND matching size/mtime.
  - `clear_file_failure(source_path: str) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
# ---------------------------------------------------------------------------
# failed_files (poison-file tracking)
# ---------------------------------------------------------------------------


def test_record_file_failure_increments(catalog: Catalog) -> None:
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 1
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 2
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 3


def test_changed_file_resets_fail_count_and_quarantine(catalog: Catalog) -> None:
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    catalog.quarantine_file("/src/a.jpg")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is True
    # File changed (new size/mtime): count resets, quarantine cleared.
    assert catalog.record_file_failure("/src/a.jpg", 200, 222, "boom") == 1
    assert catalog.is_quarantined("/src/a.jpg", 200, 222) is False


def test_is_quarantined_requires_flag_and_matching_stat(catalog: Catalog) -> None:
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is False  # not flagged yet
    catalog.quarantine_file("/src/a.jpg")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is True
    assert catalog.is_quarantined("/src/a.jpg", 999, 111) is False  # size differs
    assert catalog.is_quarantined("/src/missing.jpg", 100, 111) is False  # no row


def test_clear_file_failure_removes_row(catalog: Catalog) -> None:
    catalog.record_file_failure("/src/a.jpg", 100, 111, "boom")
    catalog.quarantine_file("/src/a.jpg")
    catalog.clear_file_failure("/src/a.jpg")
    assert catalog.is_quarantined("/src/a.jpg", 100, 111) is False
    assert catalog.record_file_failure("/src/a.jpg", 100, 111, "boom") == 1  # fresh row
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_catalog.py -k "failure or quarantin" -v`
Expected: FAIL — `AttributeError: 'Catalog' object has no attribute 'record_file_failure'`.

- [ ] **Step 3: Add the table to `_SCHEMA`**

In `imageharbor/catalog.py`, inside the `_SCHEMA` string, after the `learned_concepts` table definition (before the closing `"""`), add:

```sql

CREATE TABLE IF NOT EXISTS failed_files (
    source_path     TEXT    PRIMARY KEY,
    size            INTEGER NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT    NOT NULL DEFAULT '',
    first_failed_at TEXT    NOT NULL,
    last_failed_at  TEXT    NOT NULL,
    quarantined     INTEGER NOT NULL DEFAULT 0
);
```

- [ ] **Step 4: Add the methods**

In `imageharbor/catalog.py`, add these methods to the `Catalog` class (place them after `learned_concept_remember`, before the `# Lifecycle` section):

```python
    # ------------------------------------------------------------------
    # Failed files (poison-file tracking)
    # ------------------------------------------------------------------

    def record_file_failure(
        self, source_path: str, size: int, mtime_ns: int, error: str
    ) -> int:
        """Record a processing failure for a source file; return new fail_count.

        If the stored size/mtime differ from the incoming values the file has
        changed on disk, so the count resets to 1 and any quarantine is cleared.
        """
        now = _now_iso()
        row = self._conn.execute(
            "SELECT size, mtime_ns, fail_count FROM failed_files WHERE source_path=?",
            (source_path,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO failed_files
                    (source_path, size, mtime_ns, fail_count, last_error,
                     first_failed_at, last_failed_at, quarantined)
                VALUES (?,?,?,?,?,?,?,0)
                """,
                (source_path, size, mtime_ns, 1, error, now, now),
            )
            self._conn.commit()
            return 1
        if row["size"] != size or row["mtime_ns"] != mtime_ns:
            self._conn.execute(
                """
                UPDATE failed_files
                   SET size=?, mtime_ns=?, fail_count=1, last_error=?,
                       last_failed_at=?, quarantined=0
                 WHERE source_path=?
                """,
                (size, mtime_ns, error, now, source_path),
            )
            self._conn.commit()
            return 1
        new_count = row["fail_count"] + 1
        self._conn.execute(
            "UPDATE failed_files SET fail_count=?, last_error=?, last_failed_at=? "
            "WHERE source_path=?",
            (new_count, error, now, source_path),
        )
        self._conn.commit()
        return new_count

    def quarantine_file(self, source_path: str) -> None:
        self._conn.execute(
            "UPDATE failed_files SET quarantined=1 WHERE source_path=?", (source_path,)
        )
        self._conn.commit()

    def is_quarantined(self, source_path: str, size: int, mtime_ns: int) -> bool:
        row = self._conn.execute(
            "SELECT quarantined, size, mtime_ns FROM failed_files WHERE source_path=?",
            (source_path,),
        ).fetchone()
        if row is None:
            return False
        return bool(row["quarantined"]) and row["size"] == size and row["mtime_ns"] == mtime_ns

    def clear_file_failure(self, source_path: str) -> None:
        self._conn.execute(
            "DELETE FROM failed_files WHERE source_path=?", (source_path,)
        )
        self._conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -v`
Expected: PASS (existing catalog tests + 4 new).

- [ ] **Step 6: Commit**

```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: add failed_files table + poison-tracking methods to catalog"
```

---

## Task 3: Watcher — breaker trip / abort / backoff / half-open

**Files:**
- Modify: `imageharbor/watcher.py`
- Test: `tests/test_watcher.py` (extend)

**Interfaces:**
- Consumes: `CircuitBreaker` (Task 1); `Catalog.is_quarantined`/`record_file_failure`/`quarantine_file`/`clear_file_failure` (Task 2, used fully in Task 4 — this task wires the breaker only).
- Produces:
  - `WatchStats` gains `quarantined: int = 0`.
  - `run_pass(*, pipeline, catalog, source, recursive=True, breaker=None, poison_max_fails=5, quarantine_dir=None) -> WatchStats`
  - `watch(*, pipeline, catalog, source, interval, recursive=True, stop_event=None, sleep=None, breaker=None, poison_max_fails=5, quarantine_dir=None) -> WatchStats`

Note: when `breaker is None`, `run_pass`/`watch` behave exactly as before (legacy callers/tests).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_watcher.py` (add imports at top: `from imageharbor.circuit_breaker import CircuitBreaker`):

```python
class _AlwaysFails:
    """Classifier whose describe() always raises — simulates a dead backend."""
    def describe(self, image_path, exif_data):
        raise RuntimeError("backend down")
    def adjudicate(self, label, candidates):
        return None
    def pick_class(self, content, classes):
        return "900"


def _src_with(tmp_path: Path, n: int) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        _make_jpeg(src / f"img_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")
    return src


def test_run_pass_aborts_remaining_files_when_breaker_trips(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 5)
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AlwaysFails())
    breaker = CircuitBreaker(trip_threshold=2, now=lambda: 0.0)
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker)
    # 2 failures trip the breaker; the pass aborts before the other 3 files.
    assert stats.errors == 2
    assert breaker.is_open()


def test_run_pass_half_open_failure_tries_only_one_file(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 5)
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AlwaysFails())
    clock = [0.0]
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=60.0, now=lambda: clock[0])
    breaker.record_failure(); breaker.record_failure()   # OPEN
    clock[0] = 60.0
    breaker.begin_probe()                                  # HALF_OPEN
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker)
    assert stats.errors == 1          # only the probe file was tried
    assert breaker.is_open()          # probe failed -> reopened


def test_run_pass_half_open_success_resumes_full_pass(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 3)
    pipeline = Pipeline(src, organized_dir, catalog)   # StubClassifier: all succeed
    clock = [0.0]
    breaker = CircuitBreaker(trip_threshold=2, backoff_base=60.0, now=lambda: clock[0])
    breaker.record_failure(); breaker.record_failure()
    clock[0] = 60.0
    breaker.begin_probe()                               # HALF_OPEN
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker)
    assert stats.processed == 3                         # probe closed it, rest ran
    assert breaker.state.name == "CLOSED"


def test_watch_sleeps_breaker_backoff_when_open(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = _src_with(tmp_path, 1)
    pipeline = Pipeline(src, organized_dir, catalog)
    clock = [1000.0]
    breaker = CircuitBreaker(trip_threshold=1, backoff_base=60.0, now=lambda: clock[0])
    breaker.record_failure()          # OPEN at t=1000, backoff=60
    stop = threading.Event()
    slept: list[float] = []

    def _sleep(d: float) -> bool:
        slept.append(d)
        stop.set()                    # exit after first sleep
        return True

    watch(pipeline=pipeline, catalog=catalog, source=src, interval=300.0,
          stop_event=stop, sleep=_sleep, breaker=breaker)
    assert slept and abs(slept[0] - 60.0) < 1.0   # slept the backoff, not the interval
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_watcher.py -k "breaker or half_open" -v`
Expected: FAIL — `run_pass() got an unexpected keyword argument 'breaker'`.

- [ ] **Step 3: Implement the breaker wiring in `watcher.py`**

Add imports near the top of `imageharbor/watcher.py`:

```python
import hashlib
import shutil
```

and add to the existing typing import line: `from typing import Callable, Optional`. Import the breaker type for annotations:

```python
from .circuit_breaker import CircuitBreaker
```

Add the `quarantined` field to `WatchStats`:

```python
@dataclass
class WatchStats:
    passes: int = 0
    processed: int = 0
    skipped_unchanged: int = 0
    errors: int = 0
    quarantined: int = 0
```

Add this module-level helper (used in Task 4; define it now):

```python
def _copy_to_quarantine(quarantine_dir: Path, source_path: str) -> None:
    """Copy a quarantined ORIGINAL into quarantine_dir (originals stay read-only).

    Named by a hash of the source PATH (the failure result carries no digest),
    so distinct paths never collide; identical path => identical bytes.
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    prefix = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    dest = quarantine_dir / f"{prefix}_{Path(source_path).name}"
    shutil.copy2(source_path, str(dest))
```

Replace the body of `run_pass` with the breaker-aware version (poison reconciliation added in Task 4 is marked with a comment placeholder for now):

```python
def run_pass(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    recursive: bool = True,
    breaker: Optional[CircuitBreaker] = None,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
) -> WatchStats:
    """Process new/changed files once. Unchanged files (per the source_seen
    cache) are skipped without hashing. When a breaker is supplied, a systemic
    run of AI failures trips it and aborts the pass early."""
    stats = WatchStats()
    pass_had_success = False
    failed_buffer: list[tuple[str, int, int, str]] = []
    tripped = False
    try:
        for path in discover_images(source, recursive=recursive):
            try:
                st = path.stat()
            except OSError:
                logger.warning(
                    "Could not stat %s; skipping this pass", path, exc_info=True
                )
                stats.errors += 1
                continue
            if catalog.source_is_unchanged(str(path), st.st_size, st.st_mtime_ns):
                stats.skipped_unchanged += 1
                continue
            if breaker is not None and catalog.is_quarantined(
                str(path), st.st_size, st.st_mtime_ns
            ):
                # Poison file, already quarantined and unchanged: skip silently.
                stats.skipped_unchanged += 1
                continue
            result = pipeline.process_file(path)
            if result.status in ("copied", "duplicate"):
                catalog.record_source_seen(
                    str(path), st.st_size, st.st_mtime_ns, result.sha256_b64url
                )
                if breaker is not None:
                    catalog.clear_file_failure(str(path))
                    breaker.record_success()
                pass_had_success = True
                stats.processed += 1
            elif result.status == "error":
                stats.errors += 1
                if breaker is not None:
                    failed_buffer.append(
                        (str(path), st.st_size, st.st_mtime_ns, result.error)
                    )
                    breaker.record_failure()
                    if breaker.is_open():
                        tripped = True
                        logger.warning(
                            "AI backend appears down (%d consecutive failures) "
                            "— backing off %.0fs",
                            breaker.trip_threshold,
                            breaker.seconds_until_probe(),
                        )
                        break
    except OSError:
        logger.warning(
            "Source unavailable: %s; skipping this pass", source, exc_info=True
        )
        stats.errors += 1

    # --- poison reconciliation (fully implemented in Task 4) ---
    _reconcile_poison(
        catalog=catalog,
        failed_buffer=failed_buffer,
        pass_had_success=pass_had_success,
        tripped=tripped,
        poison_max_fails=poison_max_fails,
        quarantine_dir=quarantine_dir,
        stats=stats,
    )
    return stats


def _reconcile_poison(**_kw) -> None:
    """Placeholder — implemented in Task 4."""
    return None
```

Replace the `watch` function with the backoff-aware version:

```python
def watch(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    interval: float,
    recursive: bool = True,
    stop_event: threading.Event | None = None,
    sleep: Callable[[float], bool] | None = None,
    breaker: Optional[CircuitBreaker] = None,
    poison_max_fails: int = 5,
    quarantine_dir: Optional[Path] = None,
) -> WatchStats:
    """Run passes until stop_event is set. An immediate first pass runs before
    the first sleep. When the breaker is OPEN, the between-pass wait is the
    breaker's remaining backoff instead of ``interval``; once it elapses the
    next pass runs as a half-open probe."""
    stop_event = stop_event or threading.Event()
    if sleep is None:
        sleep = stop_event.wait  # interruptible sleep
    wstats = WatchStats()
    while not stop_event.is_set():
        if breaker is not None and breaker.is_open():
            wait = breaker.seconds_until_probe()
            if wait > 0:
                sleep(wait)
                continue
            breaker.begin_probe()
        pass_stats = run_pass(
            pipeline=pipeline,
            catalog=catalog,
            source=source,
            recursive=recursive,
            breaker=breaker,
            poison_max_fails=poison_max_fails,
            quarantine_dir=quarantine_dir,
        )
        wstats.passes += 1
        wstats.processed += pass_stats.processed
        wstats.skipped_unchanged += pass_stats.skipped_unchanged
        wstats.errors += pass_stats.errors
        wstats.quarantined += pass_stats.quarantined
        logger.info(
            "watch pass %d: processed=%d skipped=%d errors=%d quarantined=%d",
            wstats.passes,
            pass_stats.processed,
            pass_stats.skipped_unchanged,
            pass_stats.errors,
            pass_stats.quarantined,
        )
        if stop_event.is_set():
            break
        sleep(interval)
    return wstats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_watcher.py -v`
Expected: PASS — existing watcher tests (breaker=None path unchanged) + 4 new breaker tests.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/watcher.py tests/test_watcher.py
git commit -m "feat: circuit breaker trip/backoff/half-open in the watcher"
```

---

## Task 4: Watcher — poison reconciliation + quarantine

**Files:**
- Modify: `imageharbor/watcher.py` (replace the `_reconcile_poison` placeholder)
- Test: `tests/test_poison.py` (create)

**Interfaces:**
- Consumes: `failed_buffer`/`pass_had_success`/`tripped` from `run_pass` (Task 3); `Catalog.record_file_failure`/`quarantine_file` (Task 2); `_copy_to_quarantine` (Task 3).
- Produces: `_reconcile_poison(*, catalog, failed_buffer, pass_had_success, tripped, poison_max_fails, quarantine_dir, stats) -> None` — increments per-file counts for isolated failures and quarantines those reaching the threshold; increments `stats.quarantined`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_poison.py`:

```python
"""Tests for poison-file quarantine in the watcher."""
from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import run_pass


def _make_jpeg(path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9") -> Path:
    path.write_bytes(content)
    return path


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


class _FailsFor:
    """StubClassifier-like backend that raises only for named files."""
    def __init__(self, bad_names: set[str]) -> None:
        self._bad = bad_names
    def describe(self, image_path, exif_data):
        if image_path.name in self._bad:
            raise RuntimeError("cannot decode")
        from imageharbor.ai_classifier import StubClassifier
        return StubClassifier().describe(image_path, exif_data)
    def adjudicate(self, label, candidates):
        return None
    def pick_class(self, content, classes):
        return "900"


def _fresh_breaker() -> CircuitBreaker:
    # High threshold so a single poison file never trips it in these tests.
    return CircuitBreaker(trip_threshold=100, now=lambda: 0.0)


def test_poison_file_quarantined_after_k_healthy_passes(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "good.jpg")
    _make_jpeg(src / "bad.jpg", b"\xff\xd8\xff\xe0" + b"\x07" * 16 + b"\xff\xd9")
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_FailsFor({"bad.jpg"}))
    breaker = _fresh_breaker()

    for _ in range(4):
        # 'good.jpg' succeeds each pass -> pass_had_success -> 'bad.jpg' counts.
        # Touch mtime so 'good.jpg' is re-seen? No: good is recorded seen after
        # pass 1, but bad.jpg is retried every pass (never recorded). We need a
        # success in EVERY pass, so re-create good.jpg unseen each pass:
        _make_jpeg(src / f"good_{_}.jpg", b"\xff\xd8\xff\xe0" + bytes([_ + 1]) * 16 + b"\xff\xd9")
        run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
                 poison_max_fails=5)
        assert catalog.is_quarantined(str(src / "bad.jpg"),
                                      (src / "bad.jpg").stat().st_size,
                                      (src / "bad.jpg").stat().st_mtime_ns) is False

    _make_jpeg(src / "good_final.jpg", b"\xff\xd8\xff\xe0" + b"\x63" * 16 + b"\xff\xd9")
    run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
             poison_max_fails=5)   # 5th healthy-pass failure -> quarantine
    bad = src / "bad.jpg"
    assert catalog.is_quarantined(str(bad), bad.stat().st_size, bad.stat().st_mtime_ns)


def test_systemic_outage_does_not_quarantine(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "a.jpg")
    _make_jpeg(src / "b.jpg", b"\xff\xd8\xff\xe0" + b"\x02" * 16 + b"\xff\xd9")

    class _AllFail:
        def describe(self, image_path, exif_data):
            raise RuntimeError("backend down")
        def adjudicate(self, label, candidates):
            return None
        def pick_class(self, content, classes):
            return "900"

    pipeline = Pipeline(src, organized_dir, catalog, classifier=_AllFail())
    breaker = CircuitBreaker(trip_threshold=2, now=lambda: 0.0)
    for _ in range(10):
        run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
                 poison_max_fails=1)
    # Every pass tripped (all fail) -> failures are systemic -> NOTHING quarantined.
    a = src / "a.jpg"
    assert catalog.is_quarantined(str(a), a.stat().st_size, a.stat().st_mtime_ns) is False


def test_quarantine_copies_to_dir_when_set(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _make_jpeg(src / "bad.jpg", b"\xff\xd8\xff\xe0" + b"\x09" * 16 + b"\xff\xd9")
    qdir = tmp_path / "quarantine"
    pipeline = Pipeline(src, organized_dir, catalog, classifier=_FailsFor({"bad.jpg"}))
    breaker = _fresh_breaker()
    for i in range(3):
        _make_jpeg(src / f"good_{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i + 1]) * 16 + b"\xff\xd9")
        run_pass(pipeline=pipeline, catalog=catalog, source=src, breaker=breaker,
                 poison_max_fails=3, quarantine_dir=qdir)
    copied = list(qdir.glob("*_bad.jpg"))
    assert len(copied) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_poison.py -v`
Expected: FAIL — quarantine never happens (`_reconcile_poison` is a no-op placeholder), so the `is_quarantined(...)` assertion is False when it should be True.

- [ ] **Step 3: Implement `_reconcile_poison`**

In `imageharbor/watcher.py`, replace the placeholder `_reconcile_poison` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_poison.py -v`
Expected: PASS (3 tests).

Then the full suite: `uv run pytest -q`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add imageharbor/watcher.py tests/test_poison.py
git commit -m "feat: poison-file quarantine with systemic-vs-isolated reconciliation"
```

---

## Task 5: CLI wiring (`watch` flags + `process` detection-only)

**Files:**
- Modify: `imageharbor/pipeline.py` (optional `breaker` param on `run`)
- Modify: `imageharbor/cli.py` (options + construction on both commands)
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `CircuitBreaker` (Task 1); `watch(..., breaker, poison_max_fails, quarantine_dir)` (Task 3).
- Produces:
  - `Pipeline.run(recursive=True, breaker: Optional[CircuitBreaker] = None) -> PipelineStats` — on trip, logs an abort line and stops early.
  - `watch`/`process` commands gain the flags from the design's config table.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_pipeline_run_aborts_on_breaker_trip(tmp_path):
    from imageharbor.catalog import Catalog
    from imageharbor.circuit_breaker import CircuitBreaker
    from imageharbor.pipeline import Pipeline

    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        (src / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")

    class _AllFail:
        def describe(self, image_path, exif_data):
            raise RuntimeError("down")
        def adjudicate(self, label, candidates):
            return None
        def pick_class(self, content, classes):
            return "900"

    with Catalog(tmp_path / "cat.db") as cat:
        pipeline = Pipeline(src, tmp_path / "org", cat, classifier=_AllFail())
        breaker = CircuitBreaker(trip_threshold=2, now=lambda: 0.0)
        stats = pipeline.run(breaker=breaker)
    assert breaker.is_open()
    assert stats.errors == 2          # aborted after the trip, 3 files untried


def test_process_command_aborts_and_reports_when_backend_down(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from imageharbor.cli import main

    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        (src / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")

    # Force the stub classifier to fail so the breaker trips.
    from imageharbor.ai_classifier import StubClassifier

    def _boom(self, image_path, exif_data):
        raise RuntimeError("down")

    monkeypatch.setattr(StubClassifier, "describe", _boom)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["process", "--source", str(src), "--dest", str(tmp_path / "org"),
         "--breaker-threshold", "2"],
    )
    assert result.exit_code == 1
    assert "backend appears down" in result.output.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "breaker or backend_down" -v`
Expected: FAIL — `Pipeline.run() got an unexpected keyword argument 'breaker'` / missing `--breaker-threshold` option.

- [ ] **Step 3: Add `breaker` support to `Pipeline.run`**

In `imageharbor/pipeline.py`, add the import at the top (with the other `from .` imports):

```python
from .circuit_breaker import CircuitBreaker
```

Replace `Pipeline.run` with:

```python
    def run(
        self, recursive: bool = True, breaker: "CircuitBreaker | None" = None
    ) -> PipelineStats:
        """Process all images under :attr:`source_dir`.

        When a *breaker* is supplied, each result feeds it; if it trips (a
        systemic run of AI failures) the run aborts early — the one-shot command
        has no backoff/retry loop, so continuing would just churn a dead backend.
        Returns a :class:`PipelineStats` summary.
        """
        stats = PipelineStats()
        self._dry_run_seen.clear()
        if not self.dry_run:
            self.taxonomy.ensure_seeded()
        for image_path in discover_images(self.source_dir, recursive=recursive):
            result = self._process_one(image_path)
            stats.record(result)
            _log_result(result)
            if breaker is not None:
                if result.status == "error":
                    breaker.record_failure()
                elif result.status in ("copied", "duplicate"):
                    breaker.record_success()
                if breaker.is_open():
                    logger.error(
                        "AI backend appears down — aborted after %d consecutive "
                        "failures (%d processed)",
                        breaker.trip_threshold,
                        stats.copied + stats.duplicates,
                    )
                    break
        return stats
```

- [ ] **Step 4: Add the shared breaker-option helper + wire `process`**

In `imageharbor/cli.py`, add after `_build_classifier`:

```python
def _build_breaker(threshold: int, backoff: float, backoff_cap: float):
    from .circuit_breaker import CircuitBreaker

    return CircuitBreaker(
        trip_threshold=threshold, backoff_base=backoff, backoff_cap=backoff_cap
    )
```

On the `process` command, add these options (below `--no-recursive`):

```python
@click.option(
    "--breaker-threshold",
    envvar="IMAGEHARBOR_BREAKER_THRESHOLD",
    default=5,
    show_default=True,
    type=int,
    help="Consecutive AI failures before aborting (0 disables).",
)
```

Change the `process` signature to accept `breaker_threshold: int` (add the parameter), and inside `process`, replace the `stats = pipeline.run(recursive=not no_recursive)` line with:

```python
        breaker = _build_breaker(breaker_threshold, 60.0, 900.0)
        stats = pipeline.run(recursive=not no_recursive, breaker=breaker)

    if breaker.is_open():
        click.echo(
            f"AI backend appears down — aborted after {breaker.trip_threshold} "
            f"consecutive failures ({stats.copied + stats.duplicates} processed).",
            err=True,
        )
        sys.exit(1)
```

Note: the `with Catalog(...)` block ends before this check — dedent the check so it runs after the block (mirroring the existing summary/`sys.exit` at the end of `process`). Keep the existing summary echo before the new check.

- [ ] **Step 5: Wire the `watch` command**

On the `watch` command, add these options (below `--no-recursive`):

```python
@click.option(
    "--breaker-threshold",
    envvar="IMAGEHARBOR_BREAKER_THRESHOLD",
    default=5,
    show_default=True,
    type=int,
    help="Consecutive AI failures before the breaker trips (0 disables).",
)
@click.option(
    "--breaker-backoff",
    envvar="IMAGEHARBOR_BREAKER_BACKOFF",
    default=60.0,
    show_default=True,
    type=float,
    help="Base backoff seconds after the breaker trips.",
)
@click.option(
    "--breaker-backoff-cap",
    envvar="IMAGEHARBOR_BREAKER_BACKOFF_CAP",
    default=900.0,
    show_default=True,
    type=float,
    help="Maximum backoff seconds.",
)
@click.option(
    "--poison-max-fails",
    envvar="IMAGEHARBOR_POISON_MAX_FAILS",
    default=5,
    show_default=True,
    type=int,
    help="Healthy-pass failures before a file is quarantined.",
)
@click.option(
    "--quarantine-dir",
    "quarantine_dir",
    envvar="IMAGEHARBOR_QUARANTINE",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="If set, copy quarantined originals here.",
)
```

Add the matching parameters to the `watch` signature: `breaker_threshold: int, breaker_backoff: float, breaker_backoff_cap: float, poison_max_fails: int, quarantine_dir: Path | None`.

Inside `watch`, build the breaker and pass the new args into `_watcher.watch(...)`:

```python
        breaker = _build_breaker(breaker_threshold, breaker_backoff, breaker_backoff_cap)
        stats = _watcher.watch(
            pipeline=pipeline,
            catalog=catalog,
            source=source,
            interval=interval,
            recursive=not no_recursive,
            stop_event=stop_event,
            breaker=breaker,
            poison_max_fails=poison_max_fails,
            quarantine_dir=quarantine_dir,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing CLI tests + 2 new).

Then: `uv run pytest -q`
Expected: PASS (whole suite).

- [ ] **Step 7: Commit**

```bash
git add imageharbor/pipeline.py imageharbor/cli.py tests/test_cli.py
git commit -m "feat: wire circuit breaker into watch (backoff/quarantine) and process (abort)"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the architecture section**

In `CLAUDE.md`, under "Module responsibilities", add a bullet for the new module (after the `watcher`/`pipeline` bullets — place it after `concept_map.py`):

```markdown
- **`circuit_breaker.py`** — a pure three-state (`CLOSED`/`OPEN`/`HALF_OPEN`)
  circuit breaker with no I/O. The `watch` command drives it: after
  `--breaker-threshold` (default 5) consecutive AI failures it trips, the pass
  aborts, and the watcher backs off (`--breaker-backoff` 60s → ×2 →
  `--breaker-backoff-cap` 900s) before a half-open probe (one real image)
  re-tests the backend. `process` reuses it for detection-only early-abort.
  `trip_threshold=0` disables it. This is orchestration-layer only — the
  classifier is untouched.
```

Under the `catalog.py` bullet, add a sentence about the new table:

```markdown
  A fourth `failed_files` table (`source_path`, `size`, `mtime_ns`, `fail_count`,
  `last_error`, `quarantined`, timestamps) backs poison-file quarantine: a file
  that fails `--poison-max-fails` (default 5) *healthy* passes is quarantined and
  skipped thereafter (until its bytes change). Failures during a breaker-tripped
  outage never count, so a backend outage cannot mis-quarantine good files.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document circuit breaker + poison-file quarantine"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** every spec section maps to a task — CircuitBreaker (§Architecture.1 → Task 1); `failed_files` table + methods (§Architecture.2 → Task 2); watcher trip/backoff/half-open (§Architecture.3 → Task 3); poison reconciliation + quarantine copy (§Architecture.3–4 → Task 4); config surface + `process` detection-only (§Configuration, §Architecture.5 → Task 5); docs/invariants (→ Task 6). Error-handling edge cases (§Error handling) are covered by tests in Tasks 3–4 (mount-drop untouched, quarantine-copy failure guarded, disabled breaker via threshold 0, changed-file reset).

**Placeholder scan:** no TBD/TODO in shipped code. The `_reconcile_poison` no-op in Task 3 is an explicit, intentional stub replaced in Task 4 (its call site is present from Task 3 so the module imports cleanly).

**Type consistency:** `CircuitBreaker` signatures, `WatchStats.quarantined`, `run_pass`/`watch`/`Pipeline.run` keyword params, and the `failed_files` method names are identical across Tasks 1–6. The failure tuple `(source_path, size, mtime_ns, error)` is produced in Task 3 and consumed with the same shape in Task 4.

**Deferred:** none — the approved scope (breaker + poison quarantine) is fully covered.
```
