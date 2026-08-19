# Operational dashboard and control gateway — design

**Date:** 2026-08-19
**Status:** implemented (2026-08-19) — see "Departures from this design" below
**Extends:** [`2026-07-31-dockerized-watcher-design.md`](2026-07-31-dockerized-watcher-design.md)

## Goal

Give the long-running watcher a web page that answers *what has it done, what
is it doing, and when will it finish* — and a small set of controls that change
its behaviour without editing a compose file or restarting a container.

Three things it must do:

1. **Report** — library stats, evidence quality, work queues, and a history of
   passes.
2. **Project** — when the outstanding work will be done, or say plainly that it
   is stalled.
3. **Control** — pause and resume, adjust the poll interval, and turn the AI
   enrichment pass on or off.

## Non-goals

- **Authentication.** The page binds to all interfaces with no login. The blast
  radius is deliberately small: nothing it exposes can delete, move, or reveal a
  photo, and the worst an intruder achieves is pausing a photo organizer. If the
  host ever leaves a trusted network this decision must be revisited.
- **Full runtime reconfiguration.** Source and destination paths, tier
  thresholds, breaker settings, and the AI backend URL stay in compose. Some
  cannot safely change mid-run; others could restructure a library from a web
  page with no confirmation.
- **A consistent snapshot per render.** See "Accepted inconsistency" below.
- **Triggering work.** No "run now", no "re-organize", no "re-enrich". The
  watcher's schedule is its own.
- **Historical charts beyond the `runs` table.** Per-photo timelines are a
  catalog query, not a dashboard feature.

## Ground truth

Measured against the current tree at `daaff10`.

**What already exists and is reused:**

- `watcher.watch()` loops on a `threading.Event` named `stop_event`, and its
  sleep is `stop_event.wait` — an already-interruptible wait. A pause hooks the
  same mechanism rather than inventing one.
- `docker-compose.yml` runs `command: watch` with `restart: unless-stopped` and
  env-var configuration.
- `Catalog` opens SQLite in WAL mode (`catalog.py:201`) with
  `check_same_thread=False`, so concurrent readers are already supported.
- The catalog carries `created_at`, `processed_at`, and `enriched_at` per photo,
  plus `date_tier`/`descriptor_tier`, `failed_files`, and
  `takeout_status_counts()`.

**What is missing:**

- **A record of passes.** The catalog knows about photos, not about runs. There
  is no way to ask "how many passes ran yesterday" or "what rate is enrichment
  achieving", which is precisely what history and projections need.
- **An explicit `PRAGMA busy_timeout`** — though **not** the gap it was claimed
  to be. This spec originally asserted that without it, contention surfaces as
  an opaque `database is locked` abort, and the 2026-08-18 whole-branch review
  said the same. Both were wrong: CPython's `sqlite3.connect()` applies
  `timeout=5.0` by default, which *is* `busy_timeout=5000`, so a contended write
  already waits five seconds. Measured 2026-08-19 — a default connection and an
  ImageHarbor `Catalog` both report `busy_timeout = 5000`.

  The pragma is still added, for one honest reason: it pins the value at the
  point of use, so a future `connect(timeout=0)` cannot silently remove the wait
  (measured: that call yields `busy_timeout = 0`). It is belt-and-braces and
  self-documenting, not a bug fix, and no test can distinguish its presence from
  its absence — which the implementation reported rather than inventing a test
  that would pass either way.
- **Runtime dependencies for a web server.** The project runs on Pillow and
  Click alone.

## Architecture

One process, one container. `watch` starts an HTTP server on a daemon thread
beside the existing loop; the two share `stop_event` and an in-memory control
object.

The alternative — a separate dashboard process reading the catalog — was
rejected. `watch` is already the single writer, so keeping the control plane
inside it makes "paused" authoritative rather than eventually-consistent, and a
pause becomes a flag the loop checks rather than a message between processes.

The accepted cost: if the watcher process dies, the page dies with it. Mitigated
by keeping the server thread alive across a *failed pass* and reporting the
failure, which is the case that actually matters — a crashed pass is when an
operator most wants the page.

### Modules

A new `imageharbor/dashboard/` package, split so the part most likely to be
wrong has no I/O — the same discipline that made `sidecar_schema.py` and
`takeout/pairing.py` exhaustively testable.

| Module | Responsibility | I/O |
|---|---|---|
| `projections.py` | recent run rates + backlog → ETA, or an honest "stalled" | **no** |
| `stats.py` | catalog → the numbers on the page | reads |
| `control.py` | pause flag, settings, override precedence | reads/writes `settings` |
| `server.py` | stdlib `http.server`, routing, the page | yes |

`projections.py` is pure because "when will this finish" is the claim most
likely to be quietly wrong. It needs table tests over run histories — empty,
single-pass, stalled, wildly variable — not a mocked database.

### No new dependencies

The page is served by `http.server` from the standard library and is a single
self-contained HTML file: no framework, no build step, no CDN. A NAS box should
not need internet access to render its own dashboard.

This is a deliberate trade. A framework would be more comfortable to write; two
runtime dependencies is a property of this project worth more than that comfort
for a read-mostly page with four POST endpoints and one user. If the surface
grows past that, swapping in a framework is contained to `server.py`.

## What the dashboard shows

### Now

```
● RUNNING          pass 1,284 · facts phase · 00:42 elapsed
next pass in 4:18                            breaker CLOSED
last pass: 12 copied · 3 duplicates · 0 errors · 8 enriched
```

Breaker state is load-bearing, not decoration. `OPEN` means the AI backend is
unreachable, the enrichment phase is being skipped, and every enrichment
projection below it is meaningless until it closes.

### Library

Photos, total bytes, distinct source paths, date range covered, count in
`Undated/`, duplicates collapsed with bytes saved, enriched vs unenriched.

### Evidence quality

The view no generic job monitor would have, and the most useful one here:

| Date tier | | Descriptor tier |
|---|---|---|
| EXIF original (40) | | Human filename (30) |
| External sidecar (30) | | AI subject (20) |
| EXIF other (20) | | None (0) |
| Filename pattern (10) | | |
| **Undated (0)** | | |

It answers *how well do I know my own library*, and maps directly onto work
still available: photos at descriptor tier 0 are ones enrichment can still name;
photos at date tier 0 are ones only better evidence can place.

### Queues

Unenriched, quarantined (with reasons), failed files, Takeout members pending.

### History

From the `runs` table: throughput per hour over 24h and 30d, passes and their
outcomes, errors over time, and a breaker-state timeline showing when the AI
backend was reachable.

### Projections

```
Enrichment backlog    457 photos
Recent rate           38/hour  (median, last 10 passes)
Projected complete    ~12 hours  (tomorrow ~09:40)
```

Two rules:

- **Median of recent passes, never a lifetime average.** A backlog burned down
  at 3am tells you nothing about the rate now.
- **Refuse to guess.** When the breaker is `OPEN` or the system is paused, show
  `stalled — backend unreachable` rather than a number. A confident wrong ETA is
  worse than no ETA; this is the same instinct as `Undated/` over a fabricated
  year.

## Control plane

### Pause semantics

**Pause takes effect at the next file boundary, never mid-file.** This is the
only correct behaviour: the system's guarantee is copy → verify → catalog as an
atomic unit per photo, and interrupting between those steps is what the
crash-recovery machinery exists to survive. Deliberately inducing it would be
perverse.

The loop checks the flag between files and between phases. The UI shows the real
state:

```
● RUNNING   [ Pause ]
◐ PAUSING…  finishing photo 47 of 112
○ PAUSED    [ Resume ]   paused 14m ago
```

**Pause survives a restart**, persisted as a `paused` key in the `settings`
table like any other override. A container that comes back running after being
deliberately paused is the kind of surprise that makes an operator stop trusting
the button.

**Pause applies to both phases**, between photos in the facts pass and between
rows in the enrichment pass. Neither is interrupted mid-photo.

### The dials

| Control | Effect | Takes effect |
|---|---|---|
| Pause / Resume | Loop stops between files | Next file boundary |
| Poll interval | Seconds between passes | Next sleep |
| AI enrichment on/off | Skips the enrichment phase | Next pass |

The AI toggle is more than convenience. The facts pass organizes and needs no
backend; turning enrichment off keeps organizing at full speed while the Jetson
is down or busy, without touching compose.

### Override precedence, made visible

Settings live in a `settings` table. Env vars supply the value at first start; a
dashboard change writes a stored override that wins from then on.

That creates an obvious and nasty failure mode: edit `docker-compose.yml`,
restart, and nothing changes, because a stored override from months ago is
silently winning. So an override is displayed wherever it is in effect, with a
one-click revert:

```
Poll interval   [ 120 ] sec   [Apply]
                ⚠ overriding IMAGEHARBOR_INTERVAL=300   [ Revert to config ]
```

## Data model

Two additive tables. `SCHEMA_VERSION` stays `"2"`: no existing row is
reinterpreted and no existing column changes meaning, so
`Catalog._guard_legacy_catalog` correctly does not fire.

```sql
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT    NOT NULL,          -- 'facts' | 'enrich'
    started_at    TEXT    NOT NULL,
    ended_at      TEXT,                      -- NULL while in flight
    scanned       INTEGER NOT NULL DEFAULT 0,
    copied        INTEGER NOT NULL DEFAULT 0,
    duplicates    INTEGER NOT NULL DEFAULT 0,
    errors        INTEGER NOT NULL DEFAULT 0,
    enriched      INTEGER NOT NULL DEFAULT 0,
    enrich_failed INTEGER NOT NULL DEFAULT 0,
    breaker_state TEXT    NOT NULL DEFAULT 'CLOSED',
    paused        INTEGER NOT NULL DEFAULT 0  -- pass ended because of a pause
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

-- Keys: 'paused' ('0'|'1'), 'interval' (seconds), 'enrich' ('0'|'1').
-- A key present here overrides the env-var value; absent means "follow config".
-- Reverting an override DELETES the row rather than writing the env value, so
-- a later compose change is picked up instead of being shadowed by a stale copy.
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

A row is inserted when a pass starts and updated when it ends, so a pass
interrupted by a crash leaves `ended_at` NULL — which is itself the signal that
the previous run died, and is what the page reports.

## Concurrency

The dashboard **reads through its own connection**. WAL already supports
concurrent readers, so a slow page render can never block the watcher.

For settings it **writes** — a single-row upsert with no relationship to any
file operation. This is a deliberate exception to the project's
single-writer-per-catalog discipline and is stated rather than smuggled in. The
discipline exists to stop two ImageHarbor processes racing on photo rows and
relocations; a settings row participates in neither.

It is made safe by adding `PRAGMA busy_timeout` to `Catalog.__init__`. Without
it, any contention surfaces as an opaque `database is locked` abort rather than
a brief wait. This closes a gap the 2026-08-18 review already recorded.

### Accepted inconsistency

Stats are derived from live SQL over a catalog the watcher is actively writing,
so a page rendered mid-pass can show a photo counted in one number and not yet
in another. **This is accepted deliberately**: a momentarily inconsistent page is
better than one that blocks the writer to obtain a consistent snapshot. Numbers
converge within one poll interval, and no decision the dashboard offers depends
on two counters agreeing.

## HTTP surface

```
GET  /                        the page (single self-contained HTML file)
GET  /api/stats               everything on it, one JSON document
POST /api/pause               {"paused": true|false}
POST /api/settings            {"interval": 120} | {"enrich": false}
POST /api/settings/revert     {"key": "interval"} -- drop an override
GET  /healthz                 liveness, for Docker's healthcheck
```

`/api/stats` is one document rather than several endpoints so the page cannot
render a mix of two different moments.

## CLI and Docker

`watch` gains `--dashboard-port` (default `8080`) and `--no-dashboard`. The
server thread is a daemon sharing `stop_event`, so `docker stop` shuts down
cleanly.

**A dashboard failure must never stop the watcher.** If the port is already in
use, or the server thread raises, `watch` logs a warning and carries on
organizing photos with no dashboard. This is the same reasoning that keeps a
sidecar failure from failing an image that is already copied, verified, and
catalogued: the observability layer is subordinate to the work. The inverse --
a container that refuses to organize photos because a status page could not
bind a port -- would be an absurd trade.

Because the dashboard is on by default, this matters outside Docker too: a
local `imageharbor watch` on a machine already using port 8080 must degrade to
a warning, not an abort.

```yaml
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')"]
      interval: 30s
      timeout: 5s
      retries: 3
```

## Testing

**`tests/test_dashboard_projections.py`** — pure, and the most valuable file
here.

- An empty run history yields no projection, not a division by zero.
- A single pass yields no rate — one sample is not a trend.
- Wildly variable passes use the median, and a table of histories pins the
  result.
- **Breaker `OPEN` or paused yields `stalled`, never a number.** Mutation-test
  this: make it extrapolate anyway and confirm the test fails.
- A backlog of zero yields "complete", not an ETA of zero.

**`tests/test_dashboard_control.py`**

- **Pause stops between files, never mid-file.** Drive a fake pass of N photos,
  pause partway, assert the in-flight photo completed and photo N+1 never
  started.
- **A restart comes back paused.**
- An override wins over the env value, is reported as overriding, and reverting
  restores the env value.

**`tests/test_dashboard_stats.py`** — counts against a catalog built by the real
pipeline, including tier distributions and an empty library.

**`tests/test_dashboard_server.py`** — routes exercised through the handler
directly, no socket. Includes: `/api/stats` on an empty catalog, a POST with a
malformed body returning 400 rather than raising, and an unknown path returning
404.

**`tests/test_watcher.py`**

- The server thread survives a failed pass and the page still reports.
- `watch --no-dashboard` behaves exactly as today.
- **A dashboard that cannot start does not stop the watcher.** Bind the port
  first, then start `watch`, and assert photos are still organized and only a
  warning was logged. Mutation-test it: let the bind error propagate and confirm
  the test fails.

## Departures from this design

Two deliberate departures from what this spec originally described, both
confirmed against the shipped code (`imageharbor/watcher.py`,
`imageharbor/dashboard/`, `imageharbor/catalog.py`) rather than assumed:

- **`watch()` takes the `ControlPlane` object, not `interval`/
  `enrich_enabled` values.** This spec's "The dials" table implies the three
  controls are read like ordinary config. In the implementation,
  `watcher.watch(..., control=control)` receives the live `ControlPlane`
  instance itself and re-reads `control.pause_check()`, `control.interval`,
  and `control.enrich_enabled` fresh on *every* loop iteration — never a
  value captured once. `watch()` loops for the life of the container, so a
  plain float/bool parameter would freeze at process start: the dashboard
  would accept an edit, persist it, and show it as active, but two of the
  three dials (interval, enrichment toggle) would silently never actually
  change runtime behavior until a restart. The pre-existing `interval`/
  `enrich_enabled` parameters are kept on `watch()`, used only when
  `control=None` (e.g. tests, or a future non-dashboard caller), and behave
  exactly as they did before this feature.
- **The `PRAGMA busy_timeout` was never actually the gap this spec first
  claimed.** Already corrected in "Ground truth" and "Concurrency" above,
  confirmed still accurate against the shipped `Catalog.__init__`
  (`imageharbor/catalog.py`): CPython's `sqlite3.connect()` applies
  `timeout=5.0` by default, which *is* `busy_timeout=5000`, so contention
  between the watcher's photo writes and the dashboard's settings writes
  already waited five seconds before this pragma was added. The pragma is
  kept as an explicit pin at the point of use — belt-and-braces against a
  future `connect(timeout=0)` silently removing that wait, not a bug fix.

Everything else in this spec — the module split, the two additive tables, the
five HTTP routes, the never-stop-the-watcher rule, pause-between-photos
semantics in both passes, and projections refusing to guess — matches the
shipped implementation as verified during Task 9 (live watcher + dashboard
testing against a real organized library).

## Accepted limitations

- **The page dies with the process.** By design (see Architecture). A watchdog
  outside the container is the answer if this ever matters.
- **No authentication.** See Non-goals.
- **Projections assume the recent past predicts the near future.** A backlog
  whose photos are unusually large or whose AI responses are unusually slow will
  finish later than projected. The median-of-recent-passes rule limits this;
  it does not eliminate it.
- **Settings are global, not per-pass.** Changing the interval mid-sleep takes
  effect on the following sleep, not the current one.
