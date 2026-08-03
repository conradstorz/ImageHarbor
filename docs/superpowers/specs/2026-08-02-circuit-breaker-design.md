# Design: AI-backend circuit breaker + poison-file quarantine

**Date:** 2026-08-02
**Status:** Approved (brainstorming)
**Scope:** `imageharbor/circuit_breaker.py` (new), `imageharbor/watcher.py`,
`imageharbor/catalog.py`, `imageharbor/cli.py`, `imageharbor/pipeline.py` (minor),
plus tests.

## Motivation

During the first large live run (hpz440 → Jetson `qwen2.5vl:3b`), the Jetson GPU
went out of memory (`cudaMalloc failed`) after ~50 minutes. The watcher had no
notion of a *systemic* backend outage: it kept walking the entire ~4162-file
library every pass, making one failing AI call per file (~30 s each, plus the
OpenAI SDK's own internal retries), producing ~1 hour of pure churn and ~190
tracebacks — while hammering an already-wounded GPU, which can *prevent* it from
recovering.

The existing pipeline is already resilient in the ways that matter most: failed
files never enter the catalog, so no data is lost and every failure is retried on
the next watch pass (`watcher.py` records `source_seen` only for `copied`/
`duplicate`). This design adds the missing piece — **detecting a systemic outage
and backing off** — plus a related fix: a genuinely un-processable *poison file*
is currently retried forever, every pass.

## Goals

1. Detect a systemic AI-backend outage and stop hammering it (circuit breaker).
2. Recover automatically once the backend is healthy again (half-open probe).
3. Quarantine a file that persistently fails **while the backend is healthy**, so
   it stops churning every pass.
4. Never mis-quarantine a good file because of a backend outage.
5. Keep all behavior configurable and opt-outable; preserve existing invariants
   (no data loss, resumability, read-only originals, catalog-not-on-NAS).

## Non-goals

- Classifying exception *types* (connection vs. 500 vs. timeout). The
  consecutive-vs-isolated logic below separates systemic from file-specific
  failures without fragile error-string matching, and stays correct across
  backends.
- Persisting breaker state across process restarts (intentionally in-memory).
- Filesystem-event watching, alerting/metrics endpoints, or a dashboard.

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Trip action | Abort the current pass, exponential backoff, then probe |
| Recovery | Half-open: process exactly **one real image** as the probe |
| Scope | Circuit breaker **and** poison-file quarantine |
| Trip threshold | 5 consecutive AI failures |
| Backoff | base 60 s, ×2, cap 900 s (15 min) |
| Poison threshold | 5 healthy-pass failures before quarantine |
| Disable switch | `--breaker-threshold 0` = pure pass-through |
| `process` command | Detection-only early-abort (no backoff, no quarantine) |

## Architecture

### 1. `CircuitBreaker` — pure state machine (new `imageharbor/circuit_breaker.py`)

A small, I/O-free three-state breaker. It knows nothing about files, HTTP, or the
catalog; the watcher drives it. Time is injected (a `now: Callable[[], float]`
clock, defaulting to `time.monotonic`) so backoff is unit-testable with zero real
sleeps.

```
States:  CLOSED  ── trip_threshold consecutive failures ──▶  OPEN
         OPEN    ── backoff elapsed (begin_probe) ─────────▶  HALF_OPEN
         HALF_OPEN ── 1 success ──▶ CLOSED  |  ── 1 failure ──▶ OPEN (backoff ×2, capped)
```

Public surface (all pure; only mutate internal state):

- `__init__(trip_threshold=5, backoff_base=60.0, backoff_multiplier=2.0,
  backoff_cap=900.0, now=time.monotonic)`
- `record_success()` — CLOSED: reset consecutive counter. HALF_OPEN: → CLOSED,
  reset backoff to base.
- `record_failure()` — CLOSED: counter++; at `trip_threshold` → OPEN, set
  `opened_at = now()`, current backoff = base. HALF_OPEN: → OPEN, backoff =
  `min(backoff * multiplier, cap)`, `opened_at = now()`.
- `state` / `is_open()` / `is_half_open()`
- `seconds_until_probe()` — `max(0, opened_at + current_backoff - now())`; used by
  the watcher to size its sleep. Meaningful only when OPEN.
- `begin_probe()` — OPEN→HALF_OPEN; call once `seconds_until_probe() == 0`.

**Disable:** `trip_threshold <= 0` makes `record_failure()` a no-op and the breaker
never leaves CLOSED — a pure pass-through preserving pre-change behavior.

### 2. Catalog: `failed_files` table (new)

```sql
CREATE TABLE IF NOT EXISTS failed_files (
    source_path     TEXT PRIMARY KEY,
    size            INTEGER NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT NOT NULL DEFAULT '',
    first_failed_at TEXT NOT NULL,
    last_failed_at  TEXT NOT NULL,
    quarantined     INTEGER NOT NULL DEFAULT 0
);
```

New `Catalog` methods:

- `record_file_failure(source_path, size, mtime_ns, error) -> int` — upsert;
  increments `fail_count`, updates `last_error`/`last_failed_at`; if the stored
  size/mtime differ from the incoming values, **reset** `fail_count` to 1 and
  `quarantined` to 0 (the file changed — treat as fresh). Returns the new
  `fail_count`.
- `quarantine_file(source_path)` — set `quarantined = 1`.
- `is_quarantined(source_path, size, mtime_ns) -> bool` — True only if a row exists
  with `quarantined = 1` **and** matching size/mtime (a changed file is not
  considered quarantined).
- `clear_file_failure(source_path)` — delete the row (called when a file finally
  succeeds, so its history doesn't linger).

The table is created in `_SCHEMA` alongside the others; existing catalogs pick it
up via `CREATE TABLE IF NOT EXISTS` on next open (no migration needed).

### 3. Watcher integration (`watcher.py`)

`watch()` constructs one `CircuitBreaker` from config and passes it (plus the
poison config) into each `run_pass`.

**Between passes.** Sleep `breaker.seconds_until_probe()` when the breaker is OPEN,
otherwise the normal `interval`. Still via the injectable `sleep` (default
`stop_event.wait`), so SIGINT/SIGTERM stay responsive during a long backoff.

**Within a pass (`run_pass`).** Maintain two per-pass locals: `pass_had_success`
(bool) and `failed_buffer: list[(path, size, mtime_ns, error)]`. For each unchanged-
skip-surviving file:

1. If breaker is OPEN (tripped earlier this pass) → **break** (abort remaining
   files; they retry next pass, unrecorded).
2. If the file `is_quarantined(path, size, mtime_ns)` → skip (count as
   `skipped_unchanged`-style skip; not an error).
3. If breaker `is_half_open()` **and** we have already spent the single probe this
   pass → break (defer the rest until the probe result is known next pass). The
   probe is the *first eligible* file processed while HALF_OPEN.
4. Call `pipeline.process_file(path)`:
   - `copied` / `duplicate` → `breaker.record_success()`; `pass_had_success =
     True`; `catalog.record_source_seen(...)`; `catalog.clear_file_failure(path)`;
     `stats.processed += 1`.
   - `error` → `breaker.record_failure()`; append to `failed_buffer`;
     `stats.errors += 1`. If the breaker just transitioned to OPEN, log **one**
     line `AI backend appears down (N consecutive failures) — backing off Xs` and
     **break** the pass.

**Half-open bookkeeping.** At pass start, if breaker `is_open()` and
`seconds_until_probe() == 0`, call `begin_probe()` (→ HALF_OPEN). The first file
processed in a HALF_OPEN pass is the probe; its outcome (via `record_success`/
`record_failure`) closes or re-opens the breaker.

**Poison reconciliation (pass end).** Decide whether buffered failures count
toward quarantine:

- Breaker tripped during this pass → **discard** `failed_buffer` (systemic; never
  counts against files).
- Else `pass_had_success` is True → the backend was demonstrably up, so for each
  buffered file call `record_file_failure(...)`; if the returned `fail_count >=
  poison_max_fails`, `quarantine_file(path)`, log a WARNING, and — if a quarantine
  dir is configured — copy the original there.
- Else (no success, no trip — health unknowable) → **discard** (conservative).

This is the systemic-vs-isolated rule that prevents a flaky backend from
quarantining good files.

### 4. Quarantine copy

Mirrors the existing `--duplicates` pattern in `pipeline.py`. When
`--quarantine-dir` is set, copy the failed **original** (read-only preserved) to
`<quarantine-dir>/<sha-or-safe-name>_<basename>`. A copy failure is logged as a
warning and does **not** crash the watcher or prevent the catalog quarantine mark
(same principle as the existing sidecar-write guard). When unset, quarantine is
catalog + WARNING log only.

The error `ProcessResult` carries an empty `sha256_b64url` (`pipeline._process_one`
returns `""` on any exception), so the copy name derives from a filesystem-safe
rendering of `source_path` as the prefix rather than the digest (collisions are
harmless — identical path ⇒ same file).

### 5. `process` (one-shot) — detection-only

`process` runs `pipeline.run()` once; backoff/half-open are meaningless without a
loop. It reuses `CircuitBreaker` for **detection only**: `pipeline.run` (or a thin
wrapper) feeds each result to the breaker; on trip it stops early and the command
exits non-zero with `AI backend appears down — aborted after N consecutive
failures (M processed)`. No backoff, no half-open, no poison quarantine (single-
shot has no cross-pass history to judge a "healthy pass"). `--breaker-threshold 0`
disables this too.

## Configuration

All defaults preserve a sensible balanced policy; all are opt-outable. Added to
`watch`, and the breaker-threshold subset shared with `process`.

| CLI flag | Env var | Default | Meaning |
|---|---|---|---|
| `--breaker-threshold` | `IMAGEHARBOR_BREAKER_THRESHOLD` | `5` | Consecutive AI failures to trip (0 = disabled) |
| `--breaker-backoff` | `IMAGEHARBOR_BREAKER_BACKOFF` | `60` | Base backoff seconds |
| `--breaker-backoff-cap` | `IMAGEHARBOR_BREAKER_BACKOFF_CAP` | `900` | Max backoff seconds |
| `--poison-max-fails` | `IMAGEHARBOR_POISON_MAX_FAILS` | `5` | Healthy-pass failures before quarantine |
| `--quarantine-dir` | `IMAGEHARBOR_QUARANTINE` | *(none)* | If set, copy quarantined originals here |

Backoff multiplier is a fixed constant (`2.0`) — not exposed.

## Error handling & edge cases

- **Failure definition:** any `ProcessResult.status == "error"`. No exception-type
  classification.
- **Mount drops / source vanish:** unchanged — `run_pass` already catches `OSError`
  and ends the pass. These are **not** fed to the breaker (not AI failures), so a
  flaky NAS won't trip the AI breaker.
- **Probe file is itself poison:** a failing HALF_OPEN probe re-opens the breaker
  (backoff ×2) even if only one file was bad. Accepted: the next probe (after a
  longer backoff) likely draws a different file, and the poison file still accrues
  `fail_count` only under the healthy-pass rule. We accept a possible one-cycle
  extra backoff rather than add probe-selection logic (YAGNI).
- **Breaker is per-process / in-memory:** resets on container restart, which
  correctly forces an immediate re-probe after a redeploy.
- **Quarantine copy failure:** logged; the catalog quarantine mark still applies;
  the watcher does not crash.
- **`--quarantine-dir` unset:** quarantine still happens (catalog + WARNING);
  nothing is copied.
- **Changed file after quarantine:** new size/mtime resets `fail_count` and clears
  `quarantined`, giving a fixed/replaced file a fresh chance.

## Testing strategy

TDD; all offline with `StubClassifier` (or a stub whose `describe()` raises on
demand). No network, no real sleeps (inject clock/`sleep`).

- **`tests/test_circuit_breaker.py`** (new, pure unit): CLOSED→OPEN at threshold;
  counter resets on success below threshold; OPEN→HALF_OPEN only after backoff
  elapses; HALF_OPEN success→CLOSED (backoff reset to base); HALF_OPEN
  failure→OPEN with doubled, capped backoff; `threshold=0` disables.
- **`tests/test_watcher.py`** (extend): pass aborts after threshold (remaining
  files untried); OPEN pass sleeps the backoff; HALF_OPEN processes exactly one
  file; auto-resume after the stub "recovers"; systemic failures never increment
  `fail_count`.
- **`tests/test_poison.py`** (new): isolated failures across K healthy passes →
  quarantined; quarantined + unchanged file skipped next pass; changed file resets
  and retries; copies to `--quarantine-dir` when set; failures during a tripped
  pass do **not** count.
- **`tests/test_catalog.py`** (extend): `failed_files` CRUD — increment, get,
  quarantine flag, changed-file reset, clear-on-success.

## Invariants preserved

- No data loss; failed files never entered the catalog before, and still don't.
- Copy → verify → catalog ordering untouched.
- Originals read-only (quarantine copies, never moves).
- Catalog stays local (breaker/poison state is in the same local SQLite, never the
  NAS).
- Classifier still only perceives; the breaker lives in the *orchestration* layer
  (watcher/cli), not in `ai_classifier.py`.
