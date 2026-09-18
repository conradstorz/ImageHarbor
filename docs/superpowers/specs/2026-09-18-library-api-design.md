# Library API and `--json` contract

**Date:** 2026-09-18
**Status:** approved design, not yet implemented.
Sibling of nas-ingest's `library-api` work (its PR #1, merged 2026-09-17),
built for the same consumer: `organize-my-life`, whose spec
(`docs/superpowers/specs/2026-09-17-organize-my-life-design.md` in that repo)
names "ImageHarbor as a second tool" as its next slice.

## Why this exists

ImageHarbor's orchestrators are already importable — `Pipeline`,
`enrich_library`, `Catalog`, `backfill_sidecars`, `ingest_archives` — but
nothing about them is promised. `imageharbor/__init__.py` exports only
`__version__`; the `*Stats` dataclasses have no serialisation; every CLI
command prints prose and exits 1 for both "some files failed" and "the run
could not proceed". A caller that wants to drive ImageHarbor from another
program today must either parse prose or bind to internals that change
whenever the pipeline does.

`organize-my-life` runs its tools as subprocesses behind a `Tool` protocol
(argv in, JSON report and exit code out) and will later want ImageHarbor's
stats document as a panel. nas-ingest already offers both an import and a
`--json` subprocess contract with the same shape. This spec gives ImageHarbor
the same two doors so the consumer can treat both tools alike.

**Licensing note.** ImageHarbor is AGPL-3.0-or-later; nas-ingest and
organize-my-life are MIT. Importing ImageHarbor in-process places the
importing program under AGPL; the subprocess boundary does not. Both doors are
offered; the README states this once so the choice is informed. The owner of
all three repos is the same person, so this is a fact to record, not a
blocker.

## Scope

**In.** A facade module `imageharbor/api.py` with `process()`, `enrich()`,
`verify()`, `stats()`; frozen report dataclasses with `to_dict()`; an
exception hierarchy; `--json` on the `process`, `enrich`, `verify` CLI
commands; nas-ingest-aligned exit codes; tests, README, CLAUDE.md.

**Out.** `takeout ingest`, `sidecar backfill`, `watch`, and every `faces`
command stay CLI-only. Their orchestrators remain importable but unpromised.
No `stats` CLI subcommand. No change to how any file is hashed, copied,
verified, placed, named, or cataloged. No new dependency.

## Public surface (`imageharbor/api.py`)

Re-exported from `imageharbor/__init__.py` alongside `__version__`:

```python
from imageharbor import (
    process, enrich, verify, stats,
    AIConfig,
    ProcessReport, EnrichReport, VerifyReport,
    ProcessRow, EnrichFailureRow, VerifyRow,
    COPIED, DUPLICATE, SKIPPED, ERROR,      # process outcomes
    AI, IO,                                 # enrich failure reasons
    OK, FAILED,                             # verify outcomes
    ImageHarborError, ConfigError, Aborted,
    __version__,
)
```

### Functions

```python
def process(source, dest, *, catalog=None, duplicates_dir=None,
            sidecar=True, recursive=True, dry_run=False) -> ProcessReport

def enrich(dest, *, catalog=None, ai=AIConfig(), sidecar=True,
           breaker_threshold=5, limit=None, reclassify=False) -> EnrichReport

def verify(path) -> VerifyReport

def stats(catalog) -> dict
```

- Path arguments accept `str` or `Path`. `catalog=None` means
  `dest / "catalog.db"`, exactly as the CLI defaults today.
- `process` applies the same dest-inside-source guard the CLI applies
  (`cli._guard_dest_not_inside_source` moves to `api.py`; the CLI calls the
  facade). `dry_run=True` uses an in-memory catalog and writes nothing, as
  today.
- `enrich` builds the classifier from `ai` and a `CircuitBreaker` from
  `breaker_threshold` (backoff values are irrelevant to a single pass and stay
  at the CLI's current 60/900). `breaker_threshold=0` disables the breaker, as
  today.
- `verify` walks a file or directory the way the CLI command does now: only
  `SUPPORTED_EXTENSIONS`, only stems with an extractable digest.
- `stats(catalog)` opens the catalog read-only for the call, builds a
  `ControlPlane` over it, and returns `dashboard.stats.collect(...)` with no
  breaker and no face store — the same document `watch` serves at
  `/api/stats`, minus the sections that need a live process. It is
  import-only; a subprocess caller uses the dashboard.
- Every function opens and closes its own `Catalog`. The caller manages no
  lifecycle.
- All four are synchronous and block for the pass's duration. Like the CLI,
  they assume one writer per catalog at a time.

### `AIConfig`

```python
@dataclass(frozen=True)
class AIConfig:
    backend: str = "stub"          # "stub" | "openai"
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 60.0
    api_key: str | None = None
```

Replaces the five loose AI parameters. `cli._build_classifier` becomes
`api._build_classifier(ai: AIConfig)`; a missing `openai` extra raises
`ConfigError`, which the CLI edge turns into `click.ClickException` so the
message is unchanged.

### Exceptions

```
ImageHarborError(Exception)
├── ConfigError     dest inside source; unknown backend; missing extra;
│                   catalog missing/unreadable; path does not exist
└── Aborted         the breaker tripped and the enrichment pass stopped
```

A per-file problem never raises from the facade. It becomes an `ERROR` row
(or a `FAILED` verify row). This is nas-ingest's contract and is what lets a
caller distinguish "the run finished, some files are bad" from "the run did
not finish".

## Report types

All frozen dataclasses. Each has `.ok: bool`, `.counts: dict[str, int]`,
`.rows`, and `.to_dict() -> dict`. `to_dict()` is the `--json` document:
every `Path` becomes `str`, every row a dict, nothing else transformed.
`started`/`finished` are ISO-8601 UTC strings from `util.now_iso`.

```
ProcessReport
  source, dest, catalog: str      dry_run: bool
  started, finished: str
  counts: {COPIED, DUPLICATE, SKIPPED, ERROR, TOTAL}
  rows: list[ProcessRow]          one per discovered file
  ok = counts[ERROR] == 0

ProcessRow
  source_path: str   outcome: COPIED|DUPLICATE|SKIPPED|ERROR
  dest_path: str | None   digest: str   detail: str

EnrichReport
  dest, catalog, ai_backend: str
  started, finished: str          aborted: bool
  counts: {ENRICHED, RENAMED, ERROR, TOTAL}
  rows: list[EnrichFailureRow]    failures only
  ok = not aborted and counts[ERROR] == 0

EnrichFailureRow
  digest: str   dest_path: str | None   reason: AI|IO   detail: str

VerifyReport
  path: str   started, finished: str
  counts: {OK, FAILED, SKIPPED}
  rows: list[VerifyRow]           every file that was actually checked
  ok = counts[FAILED] == 0 and counts[OK] + counts[FAILED] > 0

VerifyRow
  path: str   outcome: OK|FAILED   digest: str
```

`ProcessRow` is built from the existing `pipeline.ProcessResult`
(`status` → outcome constant, `error` → `detail`). `VerifyRow` is built in
the facade from `hashing.extract_digest_from_stem` and `verify_pcs_file`.

`EnrichFailureRow` needs evidence `EnrichStats` does not carry today: the
organized path and the error text per failure, not just the digest.
`EnrichStats` gains `failures: list[EnrichFailure]` (`digest`, `organized_path`,
`reason`, `detail`), populated at the same two sites that append to
`ai_failed`/`io_failed`. Those two lists are **unchanged** — `watcher.
_reconcile_poison` reads them and must keep consuming only `ai_failed`.

Under `--json` with exit 2 (see below) the document additionally carries
`"error": "<message>"`; on exit 0/1 that key is absent.

## CLI changes (`imageharbor/cli.py`)

`process`, `enrich`, `verify` each gain `--json` (a bare flag). The command
body becomes: parse flags → build `AIConfig` where relevant → call the `api`
function → render. Rendering is either today's prose (unchanged text) or
`json.dumps(report.to_dict())` on stdout. Diagnostics — the breaker-abort
notice, `ClickException` messages — go to stderr in both modes, so under
`--json` stdout is exactly one JSON document.

Exit codes, aligned with nas-ingest so `organize-my-life` can share one
status mapping:

| code | meaning |
|------|---------|
| 0 | `report.ok` |
| 1 | finished, but the report has `ERROR` rows (`verify`: `FAILED` rows) |
| 2 | `ConfigError` or `Aborted`: the run did not proceed or did not finish. Under `--json` a document with `"error"` set is still printed when a report exists (an `Aborted` enrich has one; a `ConfigError` before any work does not). |

Two behaviour changes to the prose CLI, both recorded in `CHANGELOG.md`:

- `enrich` on a breaker trip exits **2**, not 1.
- `verify` with nothing verifiable exits **2**, not 1.

Everything else — text, defaults, flag names, the `--sidecar/--no-sidecar`
default of on — is unchanged.

## Data flow

```
cli.process ─┐
             ├─► api.process ─► Catalog ─► Pipeline.run ─► PipelineStats
import       ┘                                              │
                                    ProcessReport ◄── rows from .results

cli.enrich ──┐
             ├─► api.enrich ─► _build_classifier(AIConfig) ─► CircuitBreaker
import       ┘                 └─► Catalog ─► enrich_library ─► EnrichStats
                                    EnrichReport ◄── .failures / .aborted
                                    raise Aborted if stats.aborted

cli.verify ──┐
             ├─► api.verify ─► walk ─► extract_digest_from_stem / verify_pcs_file
import       ┘                 VerifyReport

import ─────────► api.stats ─► Catalog (ro) ─► ControlPlane ─► dashboard.stats.collect
```

Note on `Aborted` versus `EnrichReport.aborted`: the facade **raises**
`Aborted` when the breaker trips, carrying the partial report as
`exc.report`, so an import caller gets both the signal and the evidence. The
CLI catches it, prints the report (under `--json`) and the stderr notice, and
exits 2.

## Error handling

- `ConfigError` is raised before any catalog is opened or file touched.
- `Aborted` is raised after the catalog is closed; the partial `EnrichReport`
  is complete for the rows that ran.
- Unexpected exceptions from inside a pass are not wrapped. They are bugs
  and should surface as such, matching the CLI today.
- `stats()` inherits `collect()`'s `_safe()` posture: a failing section is
  `None` in the document, never an exception.

## Testing

- **`tests/test_api.py`** — each facade function against a tmp library with
  `StubClassifier`:
  - `process`: rows match the files discovered; a duplicate produces a
    `DUPLICATE` row; an unreadable file produces an `ERROR` row and no
    exception; `dry_run` writes nothing; dest inside source raises
    `ConfigError`; `to_dict()` round-trips through `json.dumps`/`json.loads`.
  - `enrich`: enriched count and rename count; a classifier whose `describe`
    raises on every call trips the breaker → `Aborted` with `exc.report.
    aborted` and the AI-reason failure rows; a missing organized file yields
    an `IO` failure row and no abort; `ai_failed`/`io_failed` on
    `EnrichStats` still populated identically (guards the watcher's poison
    accounting).
  - `verify`: `OK`/`FAILED`/`SKIPPED` counts; a corrupted byte flips a row to
    `FAILED`; an empty dir gives `ok == False` with zero checked.
  - `stats`: returns a dict with the `collect()` top-level keys on a fresh
    catalog.
- **`tests/test_cli.py`** additions — `--json` stdout parses and equals the
  report's `to_dict()` for all three commands; exit codes 0, 1, 2 pinned
  with one test each; stderr, not stdout, carries the abort notice under
  `--json`; the two exit-code moves asserted explicitly.
- **`tests/test_packaging.py`** — the built wheel imports every name in the
  public list above, as nas-ingest's packaging test does.
- Existing tests keep passing unchanged, apart from any that pinned the two
  moved exit codes.

## Documentation

- **README** — new "Calling from Python" and "Calling as a Subprocess"
  sections mirroring nas-ingest's, the three JSON shapes, the exit-code
  table, and the one-paragraph AGPL note.
- **CLAUDE.md** — an `api.py` bullet under Architecture; a line in Commands
  for `--json`; an invariant-adjacent note that the `--json` document and
  exit codes are a stable contract consumed by `organize-my-life` and must
  only change additively.
- **CHANGELOG.md** — the new public names and the two exit-code changes.

## Non-goals, stated so they stay out

- No async API, no progress callbacks, no cancellation. `organize-my-life`'s
  job cancellation is its own undesigned slice and depends on what a safe
  stop means per tool.
- No configuration file. ImageHarbor's inputs are flags and env vars today;
  `AIConfig` is the only grouping introduced.
- `to_dict()` is additive-only from here: new keys may appear, existing keys
  keep their meaning and type.
