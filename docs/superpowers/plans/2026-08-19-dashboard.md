# Operational Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the long-running watcher a web page reporting what it has done, what it is doing, and when it will finish — plus pause/resume, poll interval, and an AI-enrichment toggle.

**Architecture:** One process, one container. `watch` starts a stdlib `http.server` on a daemon thread beside the existing loop, sharing `stop_event` and an in-memory control object. Projection logic is a pure module with no I/O so "when will this finish" can be table-tested. Two additive catalog tables record passes and settings.

**Tech Stack:** Python 3 stdlib (`http.server`, `threading`, `sqlite3`), Click, pytest, `uv`. **No new runtime dependencies.**

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-19-dashboard-design.md`. Every task's requirements implicitly include this section.

- **A dashboard failure must never stop the watcher.** Port in use, server thread raising, stats query failing — all log a warning and let organizing continue. The observability layer is subordinate to the work, exactly as a sidecar failure never fails an image that is already copied, verified, and catalogued.
- **Pause takes effect between photos, never mid-photo**, in both the facts and enrichment phases. Copy → verify → catalog is atomic per photo; deliberately interrupting it would be perverse.
- **Pause survives a restart**, persisted as a `paused` key in `settings`.
- **Projections refuse to guess.** Breaker `OPEN`, paused, or no recent progress → a stalled status, never a number. Same instinct as `Undated/` over a fabricated year.
- **Median of recent passes, never a lifetime average.**
- **No new runtime dependencies.** Stdlib only; the page is one self-contained HTML file with no CDN and no build step.
- **`SCHEMA_VERSION` stays `"2"`.** Both new tables are additive.
- **Reverting an override DELETES the settings row**, so a later compose change is picked up rather than shadowed by a stale copy.
- **Stats may be momentarily inconsistent** — accepted deliberately; a page that blocks the writer to get a consistent snapshot is worse.
- Python is managed with `uv` (`uv run pytest`); never pip or venv. Do not chain shell commands with `&&`.
- **Add to existing test files; never rewrite one.** Run `git diff` before committing and confirm every `-` line is intended.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `imageharbor/dashboard/__init__.py` | create | Package marker |
| `imageharbor/dashboard/projections.py` | create | **Pure.** Run history + backlog → ETA or a stalled status |
| `imageharbor/dashboard/control.py` | create | Pause flag, settings, override precedence |
| `imageharbor/dashboard/stats.py` | create | Catalog → the numbers on the page |
| `imageharbor/dashboard/server.py` | create | stdlib HTTP server, routes, page delivery |
| `imageharbor/dashboard/index.html` | create | The page — self-contained, polls `/api/stats` |
| `imageharbor/catalog.py` | modify | `runs` + `settings` tables and accessors; `PRAGMA busy_timeout` |
| `imageharbor/pipeline.py` | modify | `pause_check` honored between photos |
| `imageharbor/enrich.py` | modify | `pause_check` honored between rows |
| `imageharbor/watcher.py` | modify | Record runs; honor pause; start/stop the server thread |
| `imageharbor/cli.py` | modify | `--dashboard-port` / `--no-dashboard` |
| `docker-compose.yml`, `Dockerfile` | modify | Publish the port; healthcheck |
| `tests/test_dashboard_projections.py` | create | The pure logic, exhaustively |
| `tests/test_dashboard_control.py` | create | Pause semantics, override precedence |
| `tests/test_dashboard_stats.py` | create | Counts against a real catalog |
| `tests/test_dashboard_server.py` | create | Routes through the handler, no socket |
| `tests/test_catalog.py`, `tests/test_watcher.py`, `tests/test_cli.py` | modify | Tables, pause plumbing, flags |

---

### Task 1: `dashboard/projections.py` — the pure projection

The claim most likely to be quietly wrong, so it gets no I/O.

**Files:**
- Create: `imageharbor/dashboard/__init__.py`, `imageharbor/dashboard/projections.py`
- Test: `tests/test_dashboard_projections.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Projection` frozen dataclass: `backlog: int`, `rate_per_hour: float | None`, `eta_seconds: float | None`, `status: str`, `reason: str`
  - `project(runs, backlog, *, breaker_open, paused, now) -> Projection`
  - `STATUS_COMPLETE`/`STATUS_PROJECTED`/`STATUS_STALLED`/`STATUS_UNKNOWN` constants
  - `runs` is a sequence of mappings with `started_at`, `ended_at`, `enriched` (ISO strings; extra keys ignored)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_projections.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_dashboard_projections.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.dashboard'`.

- [ ] **Step 3: Write the package marker**

Create `imageharbor/dashboard/__init__.py`:

```python
"""Operational dashboard: reporting, projection, and a small control gateway.

Served in-process by `watch` on a daemon thread. The split here mirrors the
rest of the project: the module most likely to be wrong (`projections`) has no
I/O and is table-tested, while `server` owns the sockets.

Nothing in this package may stop the watcher. A dashboard that cannot start,
cannot render, or cannot query still leaves photos being organized -- the
observability layer is subordinate to the work.
"""
```

- [ ] **Step 4: Write `projections.py`**

Create `imageharbor/dashboard/projections.py`:

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_dashboard_projections.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/dashboard/ tests/test_dashboard_projections.py
git commit -m "feat: pure projection logic that refuses to guess"
```

---

### Task 2: Catalog — `runs`, `settings`, and a busy timeout

**Files:**
- Modify: `imageharbor/catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Produces, on `Catalog`:
  - `run_start(kind: str) -> int` — insert and return the row id
  - `run_finish(run_id: int, *, scanned, copied, duplicates, errors, enriched, enrich_failed, breaker_state, paused) -> None`
  - `recent_runs(limit: int = 50) -> list[sqlite3.Row]` — newest first
  - `unfinished_runs() -> list[sqlite3.Row]` — `ended_at IS NULL`; evidence a pass died
  - `setting_get(key) -> str | None`, `setting_set(key, value) -> None`, `setting_delete(key) -> None`, `settings_all() -> dict[str, str]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_catalog.py` — cover: a started run has `ended_at` NULL and appears in `unfinished_runs()`; finishing it populates the counters and removes it from that list; `recent_runs` is newest-first and respects `limit`; settings round-trip; `setting_delete` removes the row entirely (**not** writes an empty string — that distinction is what lets a compose change be picked up); `SCHEMA_VERSION` is still `"2"` and a catalog written before these tables reopens without raising.

- [ ] **Step 2: Run to verify they fail**

Expected: `AttributeError: 'Catalog' object has no attribute 'run_start'`.

- [ ] **Step 3: Add the tables to `_SCHEMA`**

Append the `runs` and `settings` DDL from the spec's Data model section, verbatim including the comment above `settings` explaining the key set and the delete-not-blank rule.

- [ ] **Step 4: Add the busy timeout**

In `Catalog.__init__`, immediately after the WAL pragma:

```python
        # The dashboard writes settings rows from its own connection while the
        # watcher writes photo rows from this one. Without a busy timeout, any
        # overlap surfaces as an opaque `database is locked` abort instead of a
        # brief wait -- and a settings write must never be able to fail a pass.
        self._conn.execute("PRAGMA busy_timeout=5000;")
```

- [ ] **Step 5: Add the accessors**

Follow the file's existing idioms — `_now_iso()`, `execute` then `commit`, `sqlite3.Row` returns. Place them in a clearly-commented section before the Taxonomy section.

- [ ] **Step 6: Run the suite and commit**

```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: runs and settings tables, and a catalog busy timeout"
```

---

### Task 3: `dashboard/control.py` — pause and override precedence

**Files:**
- Create: `imageharbor/dashboard/control.py`
- Test: `tests/test_dashboard_control.py`

**Interfaces:**
- Consumes: Task 2's settings accessors.
- Produces:
  - `ControlPlane(catalog, *, env_interval: float, env_enrich: bool)`
  - `.paused -> bool`, `.set_paused(bool)`
  - `.interval -> float`, `.enrich_enabled -> bool`
  - `.set_override(key, value)`, `.revert(key)`
  - `.overrides() -> dict[str, dict]` — per key: `{"value", "env_value", "overridden"}`, for the UI's "⚠ overriding …" line
  - `.pause_check() -> bool` — the callable handed to the passes; returns True when they should stop

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_control.py` covering:

```python
def test_an_override_wins_over_the_env_value(tmp_path) -> None: ...
def test_reverting_restores_the_env_value(tmp_path) -> None: ...
def test_reverting_deletes_the_row_so_a_later_config_change_is_seen(tmp_path) -> None:
    """The trap this avoids: edit docker-compose, restart, nothing changes.

    If revert wrote the env value into the table instead of deleting the row,
    the stored copy would shadow every future config change silently.
    """
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_override("interval", 120)
    control.revert("interval")
    assert catalog.setting_get("interval") is None
    # a later "config change" is now visible
    assert ControlPlane(catalog, env_interval=600, env_enrich=True).interval == 600


def test_pause_survives_a_restart(tmp_path) -> None:
    """A container that comes back running after a deliberate pause is the kind
    of surprise that makes an operator stop trusting the button."""
    ControlPlane(catalog, env_interval=300, env_enrich=True).set_paused(True)
    assert ControlPlane(catalog, env_interval=300, env_enrich=True).paused is True


def test_overrides_report_both_values_for_the_ui(tmp_path) -> None: ...
def test_an_unparseable_stored_value_falls_back_to_env(tmp_path) -> None:
    """A hand-edited settings row must not break startup."""
```

- [ ] **Step 2: Run to verify they fail, then write `control.py`**

The module docstring must state the override rule and why revert deletes:

```python
"""Runtime control: the pause flag and the settings that override config.

Precedence is deliberately simple and deliberately visible. An env var supplies
the value at first start; a dashboard change writes a row that wins from then
on. The failure mode that creates is obvious and nasty -- edit
docker-compose.yml, restart, and nothing changes because a stored override from
months ago is silently winning -- so `overrides()` reports both values and the
UI shows the conflict wherever it is in effect.

Reverting DELETES the row rather than writing the env value into it. Writing it
back would look identical today and shadow every future config change.
"""
```

`pause_check()` reads the in-memory flag, not the database — it is called between every photo and must not be a query.

- [ ] **Step 3: Run the suite and commit**

```bash
git add imageharbor/dashboard/control.py tests/test_dashboard_control.py
git commit -m "feat: pause flag and visible override precedence"
```

---

### Task 4: Pause plumbing — between photos, never mid-photo

The subtle task. Everything else is reporting.

**Files:**
- Modify: `imageharbor/pipeline.py`, `imageharbor/enrich.py`, `imageharbor/watcher.py`
- Test: `tests/test_pipeline.py`, `tests/test_enrich.py`, `tests/test_watcher.py`

**Interfaces:**
- `Pipeline.run(recursive=True, *, pause_check: Callable[[], bool] | None = None) -> PipelineStats`
- `enrich_library(..., pause_check: Callable[[], bool] | None = None)`
- `watcher.watch(..., control: ControlPlane | None = None)`

All keyword-only with `None` defaults, so every existing caller behaves identically.

**`watch` must take the control object, not its values.** `watch()` is called
once and loops for the life of the container, so an `interval` or
`enrich_enabled` passed as a value is frozen at startup and no dashboard change
could ever take effect — which would silently defeat two of the three dials
while appearing to work in the UI. The loop therefore reads `control.interval`,
`control.enrich_enabled`, and `control.pause_check()` **on each iteration**.

When `control` is `None`, the existing `interval` and `enrich_enabled`
parameters are used exactly as today, so every current caller and test is
unaffected.

**Test this explicitly**: change the interval mid-loop through the control
object and assert the next sleep uses the new value, not the startup one. It is
the difference between a working dial and a dial that lies.

- [ ] **Step 1: Write the failing tests**

```python
def test_pause_stops_between_photos_never_mid_photo(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The guarantee is copy -> verify -> catalog as an atomic unit per photo.

    Pausing must therefore leave the in-flight photo complete and the next one
    untouched -- never a half-copied file, which is the state the crash-recovery
    machinery exists to survive rather than something to induce deliberately.
    """
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        _make_jpeg(src / f"photo{i}.jpg", b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")

    seen = 0

    def _pause_after_two() -> bool:
        nonlocal seen
        seen += 1
        return seen > 2

    stats = Pipeline(src, organized_dir, catalog).run(pause_check=_pause_after_two)

    organized = list(organized_dir.rglob("*.jpg"))
    assert len(organized) == 2                       # both complete
    assert all(verify_pcs_file(p) for p in organized)  # neither half-written
    assert catalog.count() == 2                       # and both catalogued


def test_no_pause_check_processes_everything(source_dir, organized_dir, catalog) -> None:
    """The default path is unchanged for every existing caller."""
    assert Pipeline(source_dir, organized_dir, catalog).run().copied == 2
```

Plus the enrichment equivalent, and a watcher test asserting a paused loop sleeps instead of running a pass.

- [ ] **Step 2: Run to verify they fail**

Expected: `TypeError: run() got an unexpected keyword argument 'pause_check'`.

- [ ] **Step 3: Thread `pause_check` through**

In `Pipeline.run`, check **before** starting each photo, never between the copy and the catalog write:

```python
        for image_path in discover_images(self.source_dir, recursive=recursive):
            if pause_check is not None and pause_check():
                logger.info("Paused after %d photo(s); stopping cleanly", stats.total)
                break
            result = self._process_one(image_path)
```

Do the same in `enrich_library`'s row loop, and in `watcher.watch`'s loop — where a paused watcher sleeps the interval and runs no pass at all rather than starting one and immediately breaking out.

- [ ] **Step 4: Run the full suite and commit**

```bash
git add imageharbor/pipeline.py imageharbor/enrich.py imageharbor/watcher.py tests/
git commit -m "feat: pause between photos in both passes"
```

---

### Task 5: `dashboard/stats.py` — the numbers

**Files:**
- Create: `imageharbor/dashboard/stats.py`
- Test: `tests/test_dashboard_stats.py`

**Interfaces:**
- `collect(catalog, control, *, breaker=None, now=None) -> dict` — the full `/api/stats` document

Sections: `now` (running/paused/phase/breaker/next-pass), `library`, `evidence` (date and descriptor tier distributions), `queues`, `history` (from `recent_runs`), `projection` (from Task 1), `overrides` (from Task 3).

- [ ] **Step 1: Write the failing tests**

Cover: an **empty catalog** produces a complete document with zeros rather than raising (the first thing a fresh container renders); tier distributions match a catalog built by the real pipeline; unenriched and quarantined counts are correct; and the projection block is present and stalled when paused.

- [ ] **Step 2: Run to verify they fail, then write `stats.py`**

One function per section, composed by `collect`. Each section is a small SQL query — no ORM, no caching. If a section raises, log and return that section as `None` rather than failing the whole document; a dashboard missing one panel is better than a dashboard that 500s.

- [ ] **Step 3: Run the suite and commit**

```bash
git add imageharbor/dashboard/stats.py tests/test_dashboard_stats.py
git commit -m "feat: dashboard stats collection"
```

---

### Task 6: `dashboard/server.py` and the page

**Files:**
- Create: `imageharbor/dashboard/server.py`, `imageharbor/dashboard/index.html`
- Test: `tests/test_dashboard_server.py`

**Interfaces:**
- `make_handler(catalog, control, *, breaker=None) -> type[BaseHTTPRequestHandler]`
- `serve(catalog, control, *, port, breaker=None, stop_event) -> threading.Thread | None` — returns None and logs a warning if the port cannot be bound

Routes exactly as the spec lists them.

- [ ] **Step 1: Write the failing tests**

Exercise the handler directly, no socket. Cover: `/api/stats` on an empty catalog returns 200 and valid JSON; `POST /api/pause` flips the flag; a **malformed JSON body returns 400 rather than raising**; an unknown path returns 404; `/healthz` returns 200; and `serve()` on an **already-bound port returns None and does not raise** — the constraint that keeps a dashboard failure from stopping the watcher.

- [ ] **Step 2: Run to verify they fail, then write the server**

`ThreadingHTTPServer` with `daemon_threads = True`. The handler reads `Content-Length`, rejects an oversized body, and parses JSON defensively. Serve `index.html` from the package directory.

**The page** is one file: a header strip, the panels from the spec, and a `setInterval` poll of `/api/stats`. No framework, no CDN, no build step. Keep the markup boring — this is an operational page, and the values are the content.

- [ ] **Step 3: Run the suite and commit**

```bash
git add imageharbor/dashboard/server.py imageharbor/dashboard/index.html tests/test_dashboard_server.py
git commit -m "feat: dashboard HTTP server and page"
```

---

### Task 7: The watcher records its passes

**Files:**
- Modify: `imageharbor/watcher.py`
- Test: `tests/test_watcher.py`

Without this, history and projections have nothing to read.

- [ ] **Step 1: Write the failing tests**

A completed pass writes a `runs` row with `ended_at` and the counts from `WatchStats`; a pass that raises still closes its row (the counts recorded so far, plus the error count) rather than leaving it open forever; a pass ended by a pause records `paused=1`; and the breaker state at pass end is recorded.

Also: **the server thread survives a failed pass and the page still reports.**
That is the moment an operator most wants the dashboard, so a pass raising must
not take the page down with it.

- [ ] **Step 2: Run to verify they fail, then implement**

`run_start` at the top of each pass, `run_finish` in a `finally` so a crash cannot leave a row open. An open row is meaningful — it is how the page reports that the previous run died — but only for a process that actually died, not for one that raised and continued.

- [ ] **Step 3: Run the suite and commit**

```bash
git add imageharbor/watcher.py tests/test_watcher.py
git commit -m "feat: record each pass in the runs table"
```

---

### Task 8: CLI and Docker

**Files:**
- Modify: `imageharbor/cli.py`, `docker-compose.yml`, `Dockerfile`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`watch --no-dashboard` starts no server; `--dashboard-port` is accepted; and — the important one — **`watch` with the port already bound still runs**, logging a warning. Mutation-test that last one by letting the bind error propagate and confirming the test fails.

- [ ] **Step 2: Implement**

`--dashboard-port` (default `8080`) and `--no-dashboard`. Build the `ControlPlane` from the env-derived values, call `serve(...)`, and pass `control.pause_check` and `control.interval`/`control.enrich_enabled` into `watch`.

Publish the port in `docker-compose.yml` and add the healthcheck from the spec.

- [ ] **Step 3: Run the suite and commit**

```bash
git add imageharbor/cli.py docker-compose.yml Dockerfile tests/test_cli.py
git commit -m "feat: dashboard flags and container wiring"
```

---

### Task 9: Documentation and live verification

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/deploy-docker.md`, the spec

- [ ] **Step 1: Update the docs**

`CLAUDE.md`: a `dashboard/` module entry stating the never-stop-the-watcher rule, the pause-between-photos semantics, and that projections refuse to guess; the two new tables under `catalog.py`; the `busy_timeout` addition and why. `README.md` and `docs/deploy-docker.md`: how to reach the page and what the controls do. Mark the spec implemented.

- [ ] **Step 2: Verify live**

Run the container (or `watch` locally against a scratch source), then report **real output**:

- the page renders with a real library's numbers;
- `curl /api/stats` returns a complete document;
- **pause takes effect and the in-flight photo completes** — pause during a pass over many files, then confirm every organized file passes `imageharbor verify`;
- pause survives a restart;
- an override shows the ⚠ line and reverting restores the env value;
- the projection reads `stalled` when the AI backend is unreachable;
- **`watch` still organizes photos with the port already bound.**

- [ ] **Step 3: Full suite and commit**

```bash
git add CLAUDE.md README.md docs/
git commit -m "docs: document the dashboard and control gateway"
```

---

## A note on the test steps

Tasks 1, 3, and 4 carry full test code because they hold the logic most likely
to be wrong. Tasks 2 and 5–8 describe their tests as explicit assertion lists
rather than code, because they are mechanical against interfaces this plan
already fixes. Write those tests from the stated assertions — each one names
what must hold. **If an assertion is ambiguous, ask rather than inventing one**;
on the previous project a placeholder test in a plan was filled in by guesswork
and the guess was wrong.

## Notes for the implementer

**The dashboard is subordinate.** Every failure mode in this plan resolves the same way: log it and keep organizing photos. If you find yourself writing a code path where a dashboard problem stops a pass, you have the priority inverted.

**Pause is between photos, and that is not a limitation.** Copy → verify → catalog is atomic per photo. A test that pauses and finds a half-copied file has found a bug, not an edge case.

**Projections are the one place to be pessimistic.** Every ambiguity resolves toward `unknown` or `stalled`. The dashboard is read by someone deciding whether to intervene; a confident wrong ETA sends them away when they should have looked.
