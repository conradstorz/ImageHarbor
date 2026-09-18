# Library API and `--json` Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give ImageHarbor a promised import surface (`imageharbor.process/enrich/verify/stats`, report dataclasses with `to_dict()`, an `ImageHarborError` hierarchy) and a `--json` subprocess contract with nas-ingest-aligned exit codes, so `organize-my-life` can drive it either way.

**Architecture:** One new facade module, `imageharbor/api.py`, wraps the existing orchestrators (`Pipeline`, `enrich_library`, `hashing.verify_pcs_file`, `dashboard.stats.collect`) and builds frozen report dataclasses from their `*Stats` results. `cli.py`'s `process`/`enrich`/`verify` commands shrink to flag parsing plus render-prose-or-JSON over the facade. Internals (`PipelineStats`, `EnrichStats`) stay private; only `EnrichStats` gains one additive field so failures carry a path and reason.

**Tech Stack:** Python 3.10+, click, sqlite3, pytest, `uv`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-18-library-api-design.md` — read it first.

## Global Constraints

- Use `uv` for ALL Python work; never pip. Never chain shell commands with `&&` — one command per Bash call. (User global CLAUDE.md.)
- Branch: `feat/library-api` off `release/v5-structural` (the spec is committed there as f95cc15). One commit per task.
- Before every commit: `uv run pytest -q` (record the baseline in Task 1, 0 failures thereafter), `uv run ruff check .`, `uv run mypy imageharbor` — all clean.
- **No change to how any file is hashed, copied, verified, placed, named, or cataloged.** The facade only wraps.
- `EnrichStats.ai_failed` / `io_failed` are **unchanged** in type and population — `watcher._reconcile_poison` reads them. The new `failures` list is additive.
- Prose CLI text is unchanged except the two sanctioned exit-code moves (Task 7): `enrich` breaker abort → exit 2; `verify` with nothing verifiable → exit 2.
- Under `--json`, stdout is exactly one JSON document; every diagnostic goes to stderr.
- `to_dict()` keys are additive-only once shipped. Use the exact key names in this plan.
- CLAUDE.md is updated in the same commit as any task that changes a documented module boundary (Task 1 moves two `cli.py` helpers; Task 9 documents the rest).
- Outcome/reason constants are the uppercase strings `"COPIED"`, `"DUPLICATE"`, `"SKIPPED"`, `"ERROR"`, `"ENRICHED"`, `"RENAMED"`, `"TOTAL"`, `"AI"`, `"IO"`, `"OK"`, `"FAILED"` — nas-ingest's convention.

---

### Task 1: `api.py` skeleton — exceptions, constants, `AIConfig`, moved helpers

**Files:**
- Create: `imageharbor/api.py`, `tests/test_api.py`
- Modify: `imageharbor/cli.py:48-98` (remove `_build_classifier` and `_guard_dest_not_inside_source`; import from `api`), `imageharbor/cli.py:166,309,597,602,944` (call sites), `tests/test_cli.py:654-660` (`test_build_classifier_stub_default`), `CLAUDE.md` (cli.py bullet)

**Interfaces:**
- Produces, in `imageharbor/api.py`:
  - `class ImageHarborError(Exception)`, `class ConfigError(ImageHarborError)`, `class Aborted(ImageHarborError)` with `__init__(self, message: str, report: "EnrichReport | None" = None)` storing `self.report`.
  - Constants `COPIED, DUPLICATE, SKIPPED, ERROR, ENRICHED, RENAMED, TOTAL, AI, IO, OK, FAILED` (each equal to its own name as a `str`).
  - `@dataclass(frozen=True) class AIConfig: backend: str = "stub"; base_url: str | None = None; model: str = "gpt-4o-mini"; timeout: float = 60.0; api_key: str | None = None`.
  - `def _build_classifier(ai: AIConfig) -> AIClassifier` — raises `ConfigError` on an unknown backend or a missing `openai` extra.
  - `def _guard_dest_not_inside_source(source: Path, dest: Path) -> None` — raises `ConfigError` with the exact message `cli.py` raises today.
- Produces in `cli.py`: `class _ConfigFailure(click.ClickException)` with `exit_code = 2` — the CLI edge for every `api.ConfigError`. Every CLI call site wraps its api call in `try: ... except ConfigError as exc: raise _ConfigFailure(str(exc)) from exc`. Later tasks rely on this pattern. (Exit 2 for a config error is the spec's table; the existing tests only assert `!= 0` for these paths.)

- [ ] **Step 1: Branch and baseline**

```bash
git checkout release/v5-structural
git checkout -b feat/library-api
uv run pytest -q
```
Record the `N passed / M skipped` line in your report; that is the baseline every later task must not regress.

- [ ] **Step 2: Failing tests**

`tests/test_api.py`:

```python
"""Tests for the public facade in imageharbor.api."""

from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor import api
from imageharbor.ai_classifier import StubClassifier


def test_exception_hierarchy():
    assert issubclass(api.ConfigError, api.ImageHarborError)
    assert issubclass(api.Aborted, api.ImageHarborError)
    assert api.Aborted("x").report is None


def test_outcome_constants_are_their_own_names():
    for name in ("COPIED", "DUPLICATE", "SKIPPED", "ERROR", "ENRICHED",
                 "RENAMED", "TOTAL", "AI", "IO", "OK", "FAILED"):
        assert getattr(api, name) == name


def test_aiconfig_defaults_match_the_cli():
    cfg = api.AIConfig()
    assert (cfg.backend, cfg.base_url, cfg.model, cfg.timeout, cfg.api_key) == (
        "stub", None, "gpt-4o-mini", 60.0, None
    )


def test_build_classifier_stub_default():
    assert isinstance(api._build_classifier(api.AIConfig()), StubClassifier)


def test_build_classifier_unknown_backend_is_a_config_error():
    with pytest.raises(api.ConfigError):
        api._build_classifier(api.AIConfig(backend="nope"))


def test_guard_rejects_dest_inside_source(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    with pytest.raises(api.ConfigError) as ei:
        api._guard_dest_not_inside_source(src, src / "d")
    assert "--dest" in str(ei.value) and "--source" in str(ei.value)


def test_guard_allows_sibling_dest(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    api._guard_dest_not_inside_source(src, tmp_path / "d")  # no raise
```

Run: `uv run pytest tests/test_api.py -q` → FAIL, `ModuleNotFoundError: imageharbor.api`.

- [ ] **Step 3: Create `imageharbor/api.py`**

```python
"""Public library facade: one call per pass, one report per call.

Everything the CLI does beyond flag parsing for `process`, `enrich`, and
`verify` lives here, so a Python caller and a shell caller get identical
behaviour. See docs/superpowers/specs/2026-09-18-library-api-design.md.

Contract, mirrored from nas-ingest: a per-file problem never raises -- it
becomes an ERROR row; ConfigError means the run could not start; Aborted
means it did not finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ai_classifier import AIClassifier

# Outcome and reason constants -- their own names, so a JSON consumer and a
# Python consumer read the same string.
COPIED = "COPIED"
DUPLICATE = "DUPLICATE"
SKIPPED = "SKIPPED"
ERROR = "ERROR"
ENRICHED = "ENRICHED"
RENAMED = "RENAMED"
TOTAL = "TOTAL"
AI = "AI"
IO = "IO"
OK = "OK"
FAILED = "FAILED"


class ImageHarborError(Exception):
    """Base of every exception the facade raises on purpose."""


class ConfigError(ImageHarborError):
    """The run could not start: bad paths, unknown backend, missing extra."""


class Aborted(ImageHarborError):
    """The enrichment pass stopped early (circuit breaker tripped).

    ``report`` holds the partial EnrichReport for the rows that did run.
    """

    def __init__(self, message: str, report: "EnrichReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class AIConfig:
    """Which perception backend `enrich` talks to. Defaults match the CLI."""

    backend: str = "stub"
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 60.0
    api_key: str | None = None


def _build_classifier(ai: AIConfig) -> AIClassifier:
    """Construct the classifier for ``ai.backend``.

    Raises ConfigError for an unknown backend or when the optional `openai`
    extra is not installed."""
    backend = ai.backend.lower()
    if backend == "openai":
        from .ai_classifier import OpenAIClassifier

        try:
            return OpenAIClassifier(
                api_key=ai.api_key, model=ai.model, base_url=ai.base_url,
                timeout=ai.timeout,
            )
        except ImportError as exc:
            raise ConfigError(str(exc)) from exc
    if backend == "stub":
        from .ai_classifier import StubClassifier

        return StubClassifier()
    raise ConfigError(f"unknown AI backend: {ai.backend!r} (expected 'stub' or 'openai')")


def _guard_dest_not_inside_source(source: Path, dest: Path) -> None:
    """Refuse to run with dest nested inside source.

    `enrich` and the duplicate-upgrade path (`pipeline._maybe_upgrade_from_
    duplicate`) RENAME files under dest. If dest is a subdirectory of
    source, those renames would write into the source tree -- directly
    violating "originals are read-only", the invariant the whole project is
    built on. Only meaningful when source is a directory; a single source
    FILE cannot contain a dest directory.
    """
    if not source.is_dir():
        return
    source_resolved = source.resolve()
    dest_resolved = dest.resolve()
    if dest_resolved == source_resolved or source_resolved in dest_resolved.parents:
        raise ConfigError(
            f"--dest ({dest}) is inside --source ({source}). Renames performed "
            "by enrich/watch would then write into the read-only source tree. "
            "Choose a --dest that is not nested inside --source."
        )
```

(`EnrichReport` is defined in Task 4; the forward reference in `Aborted` is a string, so this imports fine now.)

- [ ] **Step 4: Rewire `cli.py`**

Delete `cli._build_classifier` (lines 48-68) and `cli._guard_dest_not_inside_source` (lines 79-98). Keep `_build_breaker`. Add to the imports:

```python
from .api import AIConfig, ConfigError, _build_classifier, _guard_dest_not_inside_source
```

Add, where `_build_classifier` used to be:

```python
class _ConfigFailure(click.ClickException):
    """An api.ConfigError at the CLI edge: same message format, exit code 2
    (nas-ingest's convention -- the run could not start)."""

    exit_code = 2
```

At each of the three guard call sites (`process` line 166, `watch` line 597, `takeout_ingest` line 944) replace the bare call with:

```python
    try:
        _guard_dest_not_inside_source(source, dest)   # archives_dir, dest in takeout_ingest
    except ConfigError as exc:
        raise _ConfigFailure(str(exc)) from exc
```

At the two classifier call sites (`enrich` line 309, `watch` line 602) replace with:

```python
    try:
        classifier = _build_classifier(
            AIConfig(backend=ai_backend, base_url=ai_base_url, model=ai_model,
                     timeout=ai_timeout, api_key=openai_key)
        )
    except ConfigError as exc:
        raise _ConfigFailure(str(exc)) from exc
```

- [ ] **Step 5: Update the one CLI test that reached into the helper**

Replace `tests/test_cli.py::test_build_classifier_stub_default` (lines 654-660) with:

```python
def test_build_classifier_stub_default() -> None:
    from imageharbor.ai_classifier import StubClassifier
    from imageharbor.api import AIConfig, _build_classifier

    clf = _build_classifier(AIConfig())
    assert isinstance(clf, StubClassifier)
```

- [ ] **Step 6: Run**

```bash
uv run pytest tests/test_api.py tests/test_cli.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
Expected: baseline count + 7 new, 0 failures; ruff and mypy clean.

- [ ] **Step 7: CLAUDE.md** — in the `cli.py` bullet under Architecture, after the first sentence, add: "`_build_classifier` and `_guard_dest_not_inside_source` live in `api.py` (the public facade — see its bullet) and raise `api.ConfigError`; `cli.py` converts that to `_ConfigFailure` (a `ClickException` with `exit_code = 2`) at each call site." Add a stub `api.py` bullet immediately before `cli.py`'s: "**`api.py`** — the public library facade (`process`, `enrich`, `verify`, `stats`, report dataclasses, `ImageHarborError`/`ConfigError`/`Aborted`, `AIConfig`). Filled in by the library-API work; see `docs/superpowers/specs/2026-09-18-library-api-design.md`."

- [ ] **Step 8: Commit**

```bash
git add imageharbor/api.py imageharbor/cli.py tests/test_api.py tests/test_cli.py CLAUDE.md
git commit -m "feat(api): facade skeleton — exceptions, AIConfig, moved CLI helpers"
```

---

### Task 2: `EnrichStats.failures` — per-failure path and reason

**Files:**
- Modify: `imageharbor/enrich.py:47-63` (dataclass), `:85-88`, `:160-163`, `:313-316`, `:379-383` (the four failure sites)
- Test: `tests/test_enrich_failures.py` (new)

**Interfaces:**
- Produces: `@dataclass(frozen=True) class EnrichFailure: digest: str; organized_path: Path | None; reason: str  # "AI" | "IO"; detail: str` in `imageharbor/enrich.py`, and `EnrichStats.failures: list[EnrichFailure]` (default empty). Every site that appends to `ai_failed` also appends an `EnrichFailure(reason="AI")`; every site that appends to `io_failed` appends one with `reason="IO"`. `ai_failed`/`io_failed` untouched.

- [ ] **Step 1: Failing tests**

`tests/test_enrich_failures.py`:

```python
"""EnrichStats.failures carries path + reason for every failed row, in
lockstep with the existing ai_failed / io_failed digest lists."""

from __future__ import annotations

from pathlib import Path

from imageharbor.ai_classifier import StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrich import EnrichFailure, enrich_library
from imageharbor.pipeline import Pipeline


def _organize(tmp_path: Path, n: int = 3) -> tuple[Path, Path]:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        (src / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]) * 16 + b"\xff\xd9")
    dest = tmp_path / "org"
    with Catalog(dest / "catalog.db") as catalog:
        Pipeline(source_dir=src, organized_dir=dest, catalog=catalog).run()
    return src, dest


class _Boom(StubClassifier):
    def describe(self, image_path, exif_data):
        raise RuntimeError("down")


def test_describe_failure_is_an_ai_failure_with_path_and_detail(tmp_path):
    _, dest = _organize(tmp_path, n=1)
    with Catalog(dest / "catalog.db") as catalog:
        stats = enrich_library(catalog, dest, _Boom())
    assert stats.ai_failed and not stats.io_failed
    assert len(stats.failures) == 1
    f = stats.failures[0]
    assert isinstance(f, EnrichFailure)
    assert f.digest == stats.ai_failed[0]
    assert f.reason == "AI"
    assert f.organized_path is not None and f.organized_path.exists()
    assert "down" in f.detail


def test_missing_organized_file_is_an_io_failure_with_no_path(tmp_path):
    _, dest = _organize(tmp_path, n=1)
    with Catalog(dest / "catalog.db") as catalog:
        row = next(iter(catalog.iter_unenriched(None, 0)))
        Path(row["organized_path"]).unlink()
        stats = enrich_library(catalog, dest, StubClassifier())
    assert stats.io_failed and not stats.ai_failed
    assert len(stats.failures) == 1
    f = stats.failures[0]
    assert f.digest == stats.io_failed[0]
    assert f.reason == "IO"
    assert f.organized_path is None
    assert "missing" in f.detail.lower()


def test_failures_and_digest_lists_stay_in_lockstep(tmp_path):
    _, dest = _organize(tmp_path, n=3)
    with Catalog(dest / "catalog.db") as catalog:
        stats = enrich_library(catalog, dest, _Boom())
    assert [f.digest for f in stats.failures] == stats.ai_failed + stats.io_failed
    assert stats.errors == len(stats.failures)
```

Run: `uv run pytest tests/test_enrich_failures.py -q` → FAIL, `ImportError: cannot import name 'EnrichFailure'`.

- [ ] **Step 2: Implement**

In `imageharbor/enrich.py`, above `EnrichStats`:

```python
@dataclass(frozen=True)
class EnrichFailure:
    """One failed row, with WHY it failed.

    ``reason`` is "AI" (a classifier backend call raised -- the same evidence
    that feeds the breaker and poison quarantine) or "IO" (local work after
    perception, or a missing organized file). ``organized_path`` is None when
    the file could not be located at all.
    """

    digest: str
    organized_path: Path | None
    reason: str
    detail: str
```

Add to `EnrichStats`, after `io_failed`:

```python
    # Every failure above, with its path and reason, for report consumers
    # (`api.EnrichReport`). Additive: ai_failed/io_failed are unchanged and
    # remain what the watcher's poison accounting reads.
    failures: list[EnrichFailure] = field(default_factory=list)
```

At each of the four sites, immediately after the existing `stats.<list>.append(digest)` line, add one line:

| site | add after |
|---|---|
| `_describe_row`, line ~88 | `stats.failures.append(EnrichFailure(digest, actual, "AI", str(exc)))` |
| `_apply_enrichment` pick_class except, line ~163 | `stats.failures.append(EnrichFailure(digest, actual, "AI", str(exc)))` |
| `_apply_enrichment` outer except, line ~316 | `stats.failures.append(EnrichFailure(digest, actual, "IO", str(exc)))` |
| `enrich_library` missing-file branch, line ~383 | `stats.failures.append(EnrichFailure(digest, None, "IO", f"Organized file missing: {recorded}"))` |

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_enrich_failures.py tests/test_enrich.py tests/test_poison.py tests/test_watcher.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
Expected: all pass; the poison/watcher tests prove `ai_failed`/`io_failed` behaviour is unchanged.

- [ ] **Step 4: Commit**

```bash
git add imageharbor/enrich.py tests/test_enrich_failures.py
git commit -m "feat(enrich): EnrichStats.failures carries path and AI/IO reason per failed row"
```

---

### Task 3: Report dataclasses and `api.process()`

**Files:**
- Modify: `imageharbor/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces in `api.py`:
  - `_path_str(p: Path | str | None) -> str | None`.
  - `@dataclass(frozen=True) class ProcessRow: source_path: str; outcome: str; dest_path: str | None; digest: str; detail: str` with `to_dict()` (= `dataclasses.asdict`).
  - `@dataclass(frozen=True) class ProcessReport: source: str; dest: str; catalog: str; dry_run: bool; started: str; finished: str; counts: dict[str, int]; rows: tuple[ProcessRow, ...]` with `ok` property and `to_dict()` returning `{"source", "dest", "catalog", "dry_run", "started", "finished", "counts", "rows": [row.to_dict() ...]}`.
  - `def process(source, dest, *, catalog=None, duplicates_dir=None, sidecar=True, recursive=True, dry_run=False) -> ProcessReport`.
- Consumes: Task 1's constants/exceptions/guard; `pipeline.Pipeline`, `pipeline.ProcessResult`; `catalog.Catalog`; `util.now_iso`.

- [ ] **Step 1: Failing tests** (append to `tests/test_api.py`)

```python
import json

from imageharbor.hashing import compute_sha256_b64url


def _jpeg(path: Path, fill: int = 0) -> Path:
    path.write_bytes(b"\xff\xd8\xff\xe0" + bytes([fill]) * 16 + b"\xff\xd9")
    return path


def _source(tmp_path: Path, n: int = 2) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        _jpeg(src / f"img_{i}.jpg", i)
    return src


def test_process_reports_one_row_per_file_and_copies(tmp_path: Path):
    src = _source(tmp_path)
    report = api.process(src, tmp_path / "org")
    assert report.ok
    assert report.counts == {api.COPIED: 2, api.DUPLICATE: 0, api.SKIPPED: 0, api.ERROR: 0, api.TOTAL: 2}
    assert len(report.rows) == 2
    for row in report.rows:
        assert row.outcome == api.COPIED
        assert row.dest_path is not None and Path(row.dest_path).exists()
        assert row.digest == compute_sha256_b64url(Path(row.source_path))
        assert row.detail == ""
    assert report.catalog == str(tmp_path / "org" / "catalog.db")
    assert report.dry_run is False
    assert report.started <= report.finished


def test_process_second_run_reports_duplicates(tmp_path: Path):
    src = _source(tmp_path)
    api.process(src, tmp_path / "org")
    report = api.process(src, tmp_path / "org")
    assert report.ok
    assert report.counts[api.DUPLICATE] == 2 and report.counts[api.COPIED] == 0
    assert {r.outcome for r in report.rows} == {api.DUPLICATE}


def test_process_unreadable_file_is_an_error_row_not_an_exception(tmp_path: Path, monkeypatch):
    src = _source(tmp_path, n=2)
    from imageharbor import pipeline as pipeline_mod

    real = pipeline_mod.compute_sha256_b64url

    def _boom(path):
        if path.name == "img_1.jpg":
            raise OSError("unreadable")
        return real(path)

    monkeypatch.setattr(pipeline_mod, "compute_sha256_b64url", _boom)
    report = api.process(src, tmp_path / "org")
    assert not report.ok
    assert report.counts[api.ERROR] == 1 and report.counts[api.COPIED] == 1
    bad = [r for r in report.rows if r.outcome == api.ERROR]
    assert len(bad) == 1 and "unreadable" in bad[0].detail and bad[0].dest_path is None


def test_process_dry_run_writes_nothing(tmp_path: Path):
    src = _source(tmp_path)
    dest = tmp_path / "org"
    report = api.process(src, dest, dry_run=True)
    assert report.dry_run is True
    assert not dest.exists()
    assert report.counts[api.TOTAL] == 2


def test_process_dest_inside_source_is_a_config_error(tmp_path: Path):
    src = _source(tmp_path)
    with pytest.raises(api.ConfigError):
        api.process(src, src / "org")
    assert not (src / "org").exists()


def test_process_missing_source_is_a_config_error(tmp_path: Path):
    with pytest.raises(api.ConfigError):
        api.process(tmp_path / "nope", tmp_path / "org")


def test_process_accepts_str_paths_and_custom_catalog(tmp_path: Path):
    src = _source(tmp_path)
    cat = tmp_path / "elsewhere" / "cat.db"
    report = api.process(str(src), str(tmp_path / "org"), catalog=str(cat))
    assert report.ok and cat.exists() and report.catalog == str(cat)


def test_process_report_to_dict_round_trips_json(tmp_path: Path):
    report = api.process(_source(tmp_path), tmp_path / "org")
    doc = json.loads(json.dumps(report.to_dict()))
    assert set(doc) == {"source", "dest", "catalog", "dry_run", "started", "finished", "counts", "rows"}
    assert doc["counts"] == report.counts
    assert doc["rows"][0] == {
        "source_path": report.rows[0].source_path, "outcome": "COPIED",
        "dest_path": report.rows[0].dest_path, "digest": report.rows[0].digest, "detail": "",
    }
```

Run: `uv run pytest tests/test_api.py -q -k process` → FAIL, `AttributeError: module 'imageharbor.api' has no attribute 'process'`.

(`pipeline.py` does `from .hashing import compute_sha256_b64url` at line 28, so patching the name on `imageharbor.pipeline` is the right target.)

- [ ] **Step 2: Implement** (append to `api.py`; add `import dataclasses`, `from typing import Any`, `from .catalog import Catalog`, `from .pipeline import Pipeline, ProcessResult`, `from .util import now_iso` at the top)

```python
def _path_str(p: "Path | str | None") -> str | None:
    return None if p is None else str(p)


_PROCESS_OUTCOMES = {"copied": COPIED, "duplicate": DUPLICATE, "skipped": SKIPPED, "error": ERROR}


@dataclass(frozen=True)
class ProcessRow:
    source_path: str
    outcome: str          # COPIED | DUPLICATE | SKIPPED | ERROR
    dest_path: str | None
    digest: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_result(cls, r: ProcessResult) -> "ProcessRow":
        return cls(
            source_path=str(r.source_path),
            outcome=_PROCESS_OUTCOMES.get(r.status, ERROR),
            dest_path=_path_str(r.organized_path),
            digest=r.sha256_b64url,
            detail=r.error,
        )


@dataclass(frozen=True)
class ProcessReport:
    source: str
    dest: str
    catalog: str
    dry_run: bool
    started: str
    finished: str
    counts: dict[str, int]
    rows: tuple[ProcessRow, ...]

    @property
    def ok(self) -> bool:
        return self.counts.get(ERROR, 0) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dest": self.dest,
            "catalog": self.catalog,
            "dry_run": self.dry_run,
            "started": self.started,
            "finished": self.finished,
            "counts": dict(self.counts),
            "rows": [r.to_dict() for r in self.rows],
        }


def process(
    source: "Path | str",
    dest: "Path | str",
    *,
    catalog: "Path | str | None" = None,
    duplicates_dir: "Path | str | None" = None,
    sidecar: bool = True,
    recursive: bool = True,
    dry_run: bool = False,
) -> ProcessReport:
    """The facts pass: hash, dedup, EXIF, place, copy, verify, catalog.

    No AI call, no network. Raises ConfigError before touching disk if
    ``source`` does not exist or ``dest`` is inside ``source``. A per-file
    problem is an ERROR row, never an exception.
    """
    source_p = Path(source)
    dest_p = Path(dest)
    if not source_p.exists():
        raise ConfigError(f"--source ({source_p}) does not exist")
    _guard_dest_not_inside_source(source_p, dest_p)
    catalog_p = Path(catalog) if catalog is not None else dest_p / "catalog.db"

    started = now_iso()
    if not dry_run:
        dest_p.mkdir(parents=True, exist_ok=True)
    catalog_target = Path(":memory:") if dry_run else catalog_p
    with Catalog(catalog_target) as cat:
        stats = Pipeline(
            source_dir=source_p,
            organized_dir=dest_p,
            catalog=cat,
            duplicates_dir=Path(duplicates_dir) if duplicates_dir is not None else None,
            write_sidecars=sidecar,
            dry_run=dry_run,
        ).run(recursive=recursive)
    finished = now_iso()

    rows = tuple(ProcessRow.from_result(r) for r in stats.results)
    counts = {
        COPIED: stats.copied,
        DUPLICATE: stats.duplicates,
        SKIPPED: stats.skipped,
        ERROR: stats.errors,
        TOTAL: stats.total,
    }
    return ProcessReport(
        source=str(source_p), dest=str(dest_p), catalog=str(catalog_p),
        dry_run=dry_run, started=started, finished=finished,
        counts=counts, rows=rows,
    )
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_api.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
Expected: all pass. If `test_process_unreadable_file_is_an_error_row_not_an_exception` fails because the error row's `dest_path` is not `None`, read `Pipeline._process_one`'s except branch and match what it records — the assertion should follow the pipeline, not the other way round; report what you changed.

- [ ] **Step 4: Commit**

```bash
git add imageharbor/api.py tests/test_api.py
git commit -m "feat(api): process() facade and ProcessReport"
```

---

### Task 4: `api.enrich()` and `EnrichReport`, raising `Aborted`

**Files:**
- Modify: `imageharbor/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces in `api.py`:
  - `@dataclass(frozen=True) class EnrichFailureRow: digest: str; dest_path: str | None; reason: str; detail: str` with `to_dict()`.
  - `@dataclass(frozen=True) class EnrichReport: dest: str; catalog: str; ai_backend: str; started: str; finished: str; aborted: bool; counts: dict[str, int]; rows: tuple[EnrichFailureRow, ...]` with `ok` (`not aborted and counts[ERROR] == 0`) and `to_dict()` keys `{"dest", "catalog", "ai_backend", "started", "finished", "aborted", "counts", "rows"}`.
  - `def enrich(dest, *, catalog=None, ai=AIConfig(), sidecar=True, breaker_threshold=5, limit=None, reclassify=False) -> EnrichReport`; raises `Aborted(msg, report)` when `stats.aborted`.
- Consumes: Task 2's `EnrichStats.failures`; Task 1's `_build_classifier`, `AIConfig`, `Aborted`; `enrich.enrich_library`; `circuit_breaker.CircuitBreaker`.

- [ ] **Step 1: Failing tests** (append to `tests/test_api.py`)

```python
class _Boom(StubClassifier):
    def describe(self, image_path, exif_data):
        raise RuntimeError("down")


def _organized(tmp_path: Path, n: int = 3) -> Path:
    dest = tmp_path / "org"
    api.process(_source(tmp_path, n), dest)
    return dest


def test_enrich_happy_path_counts_and_no_rows(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    report = api.enrich(dest)
    assert report.ok and not report.aborted
    assert report.ai_backend == "stub"
    assert report.counts[api.ENRICHED] == 2 and report.counts[api.TOTAL] == 2
    assert report.counts[api.ERROR] == 0 and api.RENAMED in report.counts
    assert report.rows == ()
    assert report.catalog == str(dest / "catalog.db")


def test_enrich_is_idempotent_second_run_has_nothing_to_do(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    api.enrich(dest)
    again = api.enrich(dest)
    assert again.ok and again.counts[api.TOTAL] == 0


def test_enrich_breaker_trip_raises_aborted_with_partial_report(tmp_path: Path, monkeypatch):
    dest = _organized(tmp_path, 4)
    monkeypatch.setattr(api, "_build_classifier", lambda ai: _Boom())
    with pytest.raises(api.Aborted) as ei:
        api.enrich(dest, breaker_threshold=2)
    report = ei.value.report
    assert report is not None and report.aborted and not report.ok
    assert report.counts[api.ERROR] == 2
    assert len(report.rows) == 2
    assert all(r.reason == api.AI and "down" in r.detail for r in report.rows)
    assert all(r.dest_path is not None for r in report.rows)


def test_enrich_breaker_disabled_runs_every_row_and_returns(tmp_path: Path, monkeypatch):
    dest = _organized(tmp_path, 3)
    monkeypatch.setattr(api, "_build_classifier", lambda ai: _Boom())
    report = api.enrich(dest, breaker_threshold=0)
    assert not report.aborted and not report.ok
    assert report.counts[api.ERROR] == 3 and len(report.rows) == 3


def test_enrich_missing_organized_file_is_an_io_row(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    victim = next(p for p in dest.rglob("*.jpg"))
    victim.unlink()
    report = api.enrich(dest)
    assert not report.ok and not report.aborted
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.reason == api.IO and row.dest_path is None and "missing" in row.detail.lower()


def test_enrich_missing_catalog_is_a_config_error(tmp_path: Path):
    with pytest.raises(api.ConfigError):
        api.enrich(tmp_path / "nowhere")


def test_enrich_unknown_backend_is_a_config_error(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    with pytest.raises(api.ConfigError):
        api.enrich(dest, ai=api.AIConfig(backend="nope"))


def test_enrich_report_to_dict_round_trips_json(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    victim = next(p for p in dest.rglob("*.jpg"))
    victim.unlink()
    doc = json.loads(json.dumps(api.enrich(dest).to_dict()))
    assert set(doc) == {"dest", "catalog", "ai_backend", "started", "finished", "aborted", "counts", "rows"}
    assert doc["rows"][0]["reason"] == "IO" and doc["rows"][0]["dest_path"] is None
```

Run: `uv run pytest tests/test_api.py -q -k enrich` → FAIL, no attribute `enrich`.

- [ ] **Step 2: Implement** (append to `api.py`; add `from .circuit_breaker import CircuitBreaker`, `from .enrich import EnrichStats, enrich_library`)

```python
@dataclass(frozen=True)
class EnrichFailureRow:
    digest: str
    dest_path: str | None
    reason: str           # AI | IO
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EnrichReport:
    dest: str
    catalog: str
    ai_backend: str
    started: str
    finished: str
    aborted: bool
    counts: dict[str, int]
    rows: tuple[EnrichFailureRow, ...]   # failures only

    @property
    def ok(self) -> bool:
        return not self.aborted and self.counts.get(ERROR, 0) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dest": self.dest,
            "catalog": self.catalog,
            "ai_backend": self.ai_backend,
            "started": self.started,
            "finished": self.finished,
            "aborted": self.aborted,
            "counts": dict(self.counts),
            "rows": [r.to_dict() for r in self.rows],
        }


def _enrich_report(
    stats: EnrichStats, *, dest: Path, catalog: Path, ai_backend: str,
    started: str, finished: str,
) -> EnrichReport:
    rows = tuple(
        EnrichFailureRow(
            digest=f.digest, dest_path=_path_str(f.organized_path),
            reason=f.reason, detail=f.detail,
        )
        for f in stats.failures
    )
    counts = {
        ENRICHED: stats.enriched,
        RENAMED: stats.renamed,
        ERROR: stats.errors,
        TOTAL: stats.total,
    }
    return EnrichReport(
        dest=str(dest), catalog=str(catalog), ai_backend=ai_backend,
        started=started, finished=finished, aborted=stats.aborted,
        counts=counts, rows=rows,
    )


def enrich(
    dest: "Path | str",
    *,
    catalog: "Path | str | None" = None,
    ai: AIConfig = AIConfig(),
    sidecar: bool = True,
    breaker_threshold: int = 5,
    limit: int | None = None,
    reclassify: bool = False,
) -> EnrichReport:
    """The enrichment pass over an already-organized library.

    Raises ConfigError if the catalog does not exist or the backend cannot be
    built; raises Aborted (carrying the partial report) when the circuit
    breaker trips. ``breaker_threshold=0`` disables the breaker. A per-row
    failure is a row in ``report.rows``, never an exception.
    """
    dest_p = Path(dest)
    catalog_p = Path(catalog) if catalog is not None else dest_p / "catalog.db"
    if not catalog_p.is_file():
        raise ConfigError(f"catalog not found: {catalog_p}")
    classifier = _build_classifier(ai)
    breaker = CircuitBreaker(trip_threshold=breaker_threshold, backoff_base=60.0, backoff_cap=900.0)

    started = now_iso()
    with Catalog(catalog_p) as cat:
        stats = enrich_library(
            cat, dest_p, classifier,
            write_sidecars=sidecar, breaker=breaker, limit=limit, reclassify=reclassify,
        )
    finished = now_iso()

    report = _enrich_report(
        stats, dest=dest_p, catalog=catalog_p, ai_backend=ai.backend.lower(),
        started=started, finished=finished,
    )
    if stats.aborted:
        raise Aborted(
            f"AI backend appears down — aborted after {breaker.trip_threshold} "
            "consecutive failures.",
            report,
        )
    return report
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_api.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
Expected: all pass. Note: `test_enrich_is_idempotent_second_run_has_nothing_to_do` depends on `StubClassifier` producing a deterministic answer so every row gets `enriched_at` stamped — it does today (`tests/test_monotonicity.py` relies on it).

- [ ] **Step 4: Commit**

```bash
git add imageharbor/api.py tests/test_api.py
git commit -m "feat(api): enrich() facade, EnrichReport, Aborted carries the partial report"
```

---

### Task 5: `api.verify()` and `VerifyReport`

**Files:**
- Modify: `imageharbor/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class VerifyRow: path: str; outcome: str; digest: str` with `to_dict()`; `@dataclass(frozen=True) class VerifyReport: path: str; started: str; finished: str; counts: dict[str, int]; rows: tuple[VerifyRow, ...]` with `ok` (`counts[FAILED] == 0 and counts[OK] + counts[FAILED] > 0`) and `to_dict()` keys `{"path", "started", "finished", "counts", "rows"}`; `def verify(path) -> VerifyReport`; raises `ConfigError` if `path` does not exist.
- Consumes: `hashing.extract_digest_from_stem`, `hashing.verify_pcs_file`, `discovery.SUPPORTED_EXTENSIONS`.

- [ ] **Step 1: Failing tests** (append to `tests/test_api.py`)

```python
def test_verify_organized_dir_all_ok(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    report = api.verify(dest)
    assert report.ok
    assert report.counts == {api.OK: 2, api.FAILED: 0, api.SKIPPED: 0}
    assert len(report.rows) == 2 and {r.outcome for r in report.rows} == {api.OK}
    assert all(len(r.digest) == 43 for r in report.rows)
    assert report.path == str(dest)


def test_verify_corrupted_file_is_a_failed_row(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    victim = next(p for p in dest.rglob("*.jpg"))
    victim.write_bytes(b"\xff\xd8corrupt\xff\xd9")
    report = api.verify(dest)
    assert not report.ok
    assert report.counts[api.FAILED] == 1 and report.counts[api.OK] == 1
    bad = [r for r in report.rows if r.outcome == api.FAILED]
    assert bad[0].path == str(victim)


def test_verify_skips_non_image_and_undigested_files(tmp_path: Path):
    d = tmp_path / "d"
    d.mkdir()
    _jpeg(d / "just_a_photo.jpg")           # supported ext, no digest
    (d / "notes.txt").write_text("hi")      # unsupported ext
    report = api.verify(d)
    assert not report.ok                    # nothing verifiable
    assert report.counts == {api.OK: 0, api.FAILED: 0, api.SKIPPED: 2}
    assert report.rows == ()


def test_verify_single_file(tmp_path: Path):
    dest = _organized(tmp_path, 1)
    f = next(p for p in dest.rglob("*.jpg"))
    report = api.verify(str(f))
    assert report.ok and report.counts[api.OK] == 1 and report.rows[0].path == str(f)


def test_verify_missing_path_is_a_config_error(tmp_path: Path):
    with pytest.raises(api.ConfigError):
        api.verify(tmp_path / "nope")


def test_verify_report_to_dict_round_trips_json(tmp_path: Path):
    doc = json.loads(json.dumps(api.verify(_organized(tmp_path, 1)).to_dict()))
    assert set(doc) == {"path", "started", "finished", "counts", "rows"}
    assert doc["rows"][0]["outcome"] == "OK"
```

Run: `uv run pytest tests/test_api.py -q -k verify` → FAIL, no attribute `verify`.

- [ ] **Step 2: Implement** (append to `api.py`; add `from .discovery import SUPPORTED_EXTENSIONS`, `from .hashing import extract_digest_from_stem, verify_pcs_file`)

```python
@dataclass(frozen=True)
class VerifyRow:
    path: str
    outcome: str          # OK | FAILED
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class VerifyReport:
    path: str
    started: str
    finished: str
    counts: dict[str, int]
    rows: tuple[VerifyRow, ...]   # every file actually checked

    @property
    def ok(self) -> bool:
        checked = self.counts.get(OK, 0) + self.counts.get(FAILED, 0)
        return self.counts.get(FAILED, 0) == 0 and checked > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "started": self.started,
            "finished": self.finished,
            "counts": dict(self.counts),
            "rows": [r.to_dict() for r in self.rows],
        }


def verify(path: "Path | str") -> VerifyReport:
    """Re-hash every organized image under ``path`` (a file or directory)
    and compare it with the digest embedded in its own filename.

    Files with an unsupported extension or no extractable digest are
    SKIPPED, not checked. ``report.ok`` is False when nothing was checked.
    Raises ConfigError if ``path`` does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"path does not exist: {target}")
    started = now_iso()
    candidates = [target] if target.is_file() else sorted(
        p for p in target.rglob("*") if p.is_file()
    )
    rows: list[VerifyRow] = []
    skipped = 0
    for p in candidates:
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue
        digest = extract_digest_from_stem(p.stem)
        if digest is None:
            skipped += 1
            continue
        rows.append(VerifyRow(path=str(p), outcome=OK if verify_pcs_file(p) else FAILED, digest=digest))
    finished = now_iso()
    counts = {
        OK: sum(1 for r in rows if r.outcome == OK),
        FAILED: sum(1 for r in rows if r.outcome == FAILED),
        SKIPPED: skipped,
    }
    return VerifyReport(path=str(target), started=started, finished=finished,
                        counts=counts, rows=tuple(rows))
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_api.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```

- [ ] **Step 4: Commit**

```bash
git add imageharbor/api.py tests/test_api.py
git commit -m "feat(api): verify() facade and VerifyReport"
```

---

### Task 6: `api.stats()`

**Files:**
- Modify: `imageharbor/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `def stats(catalog) -> dict` — opens `Catalog(catalog)`, builds `ControlPlane(cat, env_interval=0.0, env_enrich=False)`, returns `dashboard.stats.collect(cat, control)`. Raises `ConfigError` if the catalog file does not exist.
- Consumes: `dashboard.stats.collect(catalog, control, *, breaker=None, now=None, face_store=None)`, `dashboard.control.ControlPlane(catalog, *, env_interval, env_enrich, env_faces=False)`.

- [ ] **Step 1: Failing tests** (append to `tests/test_api.py`)

```python
def test_stats_returns_the_dashboard_document(tmp_path: Path):
    dest = _organized(tmp_path, 2)
    doc = api.stats(dest / "catalog.db")
    assert {"now", "library", "evidence", "queues", "history", "projection"} <= set(doc)
    assert doc["library"] is not None
    json.dumps(doc)   # must be serialisable as-is


def test_stats_accepts_str_and_reflects_the_library(tmp_path: Path):
    dest = _organized(tmp_path, 3)
    before = api.stats(str(dest / "catalog.db"))
    api.enrich(dest)
    after = api.stats(str(dest / "catalog.db"))
    assert before != after   # enrichment changed at least one section


def test_stats_missing_catalog_is_a_config_error(tmp_path: Path):
    with pytest.raises(api.ConfigError):
        api.stats(tmp_path / "nope.db")
```

Run: `uv run pytest tests/test_api.py -q -k stats` → FAIL.

- [ ] **Step 2: Implement** (append to `api.py`)

```python
def stats(catalog: "Path | str") -> dict[str, Any]:
    """The dashboard's ``/api/stats`` document for a library, without a
    running `watch`.

    Sections that need a live process (breaker state, the current run) read
    as they would for an idle watcher. A failing section is ``None`` in the
    document, never an exception -- `dashboard.stats.collect`'s own posture.
    Raises ConfigError if the catalog file does not exist.
    """
    from .dashboard.control import ControlPlane
    from .dashboard.stats import collect

    catalog_p = Path(catalog)
    if not catalog_p.is_file():
        raise ConfigError(f"catalog not found: {catalog_p}")
    with Catalog(catalog_p) as cat:
        control = ControlPlane(cat, env_interval=0.0, env_enrich=False)
        return collect(cat, control)
```

(Local imports: `dashboard/stats.py` imports `imageharbor.catalog` at module scope, and `api.py` is imported from `imageharbor/__init__.py` in Task 8 — importing the dashboard package at `api.py`'s module scope would pull it into every `import imageharbor`. Keep these two imports inside the function.)

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_api.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
If `collect` raises on `env_interval=0.0` (check `_now_section`/`projections` for a division), pass `env_interval=300.0` instead and note it in the docstring — the value is only used to compute a "next pass" estimate that has no meaning without a watcher.

- [ ] **Step 4: Commit**

```bash
git add imageharbor/api.py tests/test_api.py
git commit -m "feat(api): stats() returns the dashboard document without a running watch"
```

---

### Task 7: CLI `--json` and exit codes for `process`, `enrich`, `verify`

**Files:**
- Modify: `imageharbor/cli.py` (`process` 106-205, `enrich` 207-343, `verify` 767-821 by pre-Task-1 numbering; re-locate with `grep -n "^def process\|^def enrich\|^def verify" imageharbor/cli.py`)
- Modify: `tests/test_cli.py:442-453` (`test_verify_non_pcs_file_skipped`), `tests/test_cli.py:1230-1262` (`test_enrich_command_aborts_and_reports_when_backend_down`)
- Test: `tests/test_cli_json.py` (new)

**Interfaces:**
- Consumes: Tasks 3-5's `api.process/enrich/verify`, `ProcessReport/EnrichReport/VerifyReport.to_dict()`, `api.ConfigError`, `api.Aborted`.
- Produces: `--json` flag on the three commands; exit codes 0 / 1 / 2 per the spec table; private helper `_emit_json(doc: dict) -> None` in `cli.py` (`click.echo(json.dumps(doc), nl=True)`).

- [ ] **Step 1: Failing tests**

`tests/test_cli_json.py`:

```python
"""The --json subprocess contract on process / enrich / verify, and the
nas-ingest-aligned exit codes 0 / 1 / 2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from imageharbor import api
from imageharbor.cli import main


def _jpeg(path: Path, fill: int = 0) -> Path:
    path.write_bytes(b"\xff\xd8\xff\xe0" + bytes([fill]) * 16 + b"\xff\xd9")
    return path


def _source(tmp_path: Path, n: int = 2) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(n):
        _jpeg(src / f"img_{i}.jpg", i)
    return src


@pytest.fixture()
def runner() -> CliRunner:
    # click >= 8.2 (we pin 8.4.x): .stdout and .stderr are always separate;
    # .output is the interleaved view.
    return CliRunner()


def _doc(result) -> dict:
    """stdout must be exactly one JSON document."""
    return json.loads(result.stdout)


# --- process -----------------------------------------------------------------

def test_process_json_is_the_report_to_dict(runner, tmp_path):
    src = _source(tmp_path)
    dest = tmp_path / "org"
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest), "--json"])
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert set(doc) == {"source", "dest", "catalog", "dry_run", "started", "finished", "counts", "rows"}
    assert doc["counts"]["COPIED"] == 2 and len(doc["rows"]) == 2
    assert "Done." not in result.stdout


def test_process_json_exit_1_when_a_row_is_an_error(runner, tmp_path, monkeypatch):
    src = _source(tmp_path, 2)
    from imageharbor import pipeline as pipeline_mod

    real = pipeline_mod.compute_sha256_b64url
    monkeypatch.setattr(
        pipeline_mod, "compute_sha256_b64url",
        lambda p: (_ for _ in ()).throw(OSError("unreadable")) if p.name == "img_1.jpg" else real(p),
    )
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(tmp_path / "org"), "--json"])
    assert result.exit_code == 1
    doc = _doc(result)
    assert doc["counts"]["ERROR"] == 1


def test_process_json_exit_2_on_config_error_prints_no_document(runner, tmp_path):
    src = _source(tmp_path)
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(src / "org"), "--json"])
    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "--dest" in result.stderr


def test_process_prose_is_unchanged_without_json(runner, tmp_path):
    src = _source(tmp_path)
    result = runner.invoke(main, ["process", "--source", str(src), "--dest", str(tmp_path / "org")])
    assert result.exit_code == 0
    assert result.stdout.startswith("Done. Total=2  Copied=2  Duplicates=0  Errors=0")


# --- enrich ------------------------------------------------------------------

def test_enrich_json_is_the_report_to_dict(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path), dest)
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--json"])
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert set(doc) == {"dest", "catalog", "ai_backend", "started", "finished", "aborted", "counts", "rows"}
    assert doc["counts"]["ENRICHED"] == 2 and doc["aborted"] is False


def test_enrich_json_breaker_trip_exits_2_with_document_and_error(runner, tmp_path, monkeypatch):
    dest = tmp_path / "org"
    api.process(_source(tmp_path, 4), dest)
    from imageharbor.ai_classifier import StubClassifier

    monkeypatch.setattr(StubClassifier, "describe", lambda self, p, e: (_ for _ in ()).throw(RuntimeError("down")))
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--breaker-threshold", "2", "--json"])
    assert result.exit_code == 2
    doc = _doc(result)
    assert doc["aborted"] is True
    assert doc["counts"]["ERROR"] == 2 and len(doc["rows"]) == 2
    assert "backend appears down" in doc["error"].lower()
    assert "backend appears down" in result.stderr.lower()


def test_enrich_json_io_failure_exits_1(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path, 1), dest)
    next(dest.rglob("*.jpg")).unlink()
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--json"])
    assert result.exit_code == 1
    doc = _doc(result)
    assert doc["rows"][0]["reason"] == "IO" and "error" not in doc


def test_enrich_prose_breaker_trip_now_exits_2(runner, tmp_path, monkeypatch):
    dest = tmp_path / "org"
    api.process(_source(tmp_path, 4), dest)
    from imageharbor.ai_classifier import StubClassifier

    monkeypatch.setattr(StubClassifier, "describe", lambda self, p, e: (_ for _ in ()).throw(RuntimeError("down")))
    result = runner.invoke(main, ["enrich", "--dest", str(dest), "--breaker-threshold", "2"])
    assert result.exit_code == 2
    assert "backend appears down" in result.stderr.lower()


# --- verify ------------------------------------------------------------------

def test_verify_json_is_the_report_to_dict(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path), dest)
    result = runner.invoke(main, ["verify", str(dest), "--json"])
    assert result.exit_code == 0, result.stderr
    doc = _doc(result)
    assert set(doc) == {"path", "started", "finished", "counts", "rows"}
    assert doc["counts"] == {"OK": 2, "FAILED": 0, "SKIPPED": 0}
    assert not result.stdout.startswith("OK ")


def test_verify_json_failed_row_exits_1(runner, tmp_path):
    dest = tmp_path / "org"
    api.process(_source(tmp_path), dest)
    next(dest.rglob("*.jpg")).write_bytes(b"\xff\xd8bad\xff\xd9")
    result = runner.invoke(main, ["verify", str(dest), "--json"])
    assert result.exit_code == 1
    assert _doc(result)["counts"]["FAILED"] == 1


def test_verify_nothing_verifiable_exits_2_in_both_modes(runner, tmp_path):
    plain = _jpeg(tmp_path / "just_a_photo.jpg")
    prose = runner.invoke(main, ["verify", str(plain)])
    assert prose.exit_code == 2
    assert "No organized image files" in prose.stderr
    js = runner.invoke(main, ["verify", str(plain), "--json"])
    assert js.exit_code == 2
    doc = _doc(js)
    assert doc["counts"]["SKIPPED"] == 1 and "No organized image files" in doc["error"]
```

Also update two existing tests in `tests/test_cli.py`:

- `test_verify_non_pcs_file_skipped` (line 442): change `assert result.exit_code != 0, result.output` to `assert result.exit_code == 2, result.output`. The default `CliRunner` mixes stderr into `.output`, so the message assertions still hold.
- `test_enrich_command_aborts_and_reports_when_backend_down` (line 1230): change `assert result.exit_code == 1` to `assert result.exit_code == 2`.

Run: `uv run pytest tests/test_cli_json.py -q` → FAIL, `no such option: --json`.

- [ ] **Step 2: Implement in `cli.py`**

Add imports: `import json`, and extend the `api` import to `from .api import AIConfig, Aborted, ConfigError, _build_classifier, _guard_dest_not_inside_source` plus `from . import api`. Remove the now-unused `from .catalog import Catalog`, `from .enrich import enrich_library`, `from .hashing import extract_digest_from_stem, verify_pcs_file`, `from .pipeline import Pipeline` **only if** nothing else in `cli.py` uses them (`watch`, `takeout_ingest`, `catalog list/get` still use `Catalog` and `Pipeline` — check with grep; keep what's used).

Add a helper near the top:

```python
def _emit_json(doc: dict) -> None:
    """Under --json, stdout is exactly one document; everything else is stderr."""
    click.echo(json.dumps(doc))
```

Add the option to all three commands, immediately above the `def`:

```python
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print the run report as one JSON document on stdout (exit 0 ok, 1 error rows, 2 aborted/config).",
)
```

Replace the body of `process` with:

```python
def process(source, dest, catalog_path, duplicates_dir, sidecar, dry_run, no_recursive, as_json) -> None:
    """Discover, hash, copy and catalog photos from SOURCE to DEST.

    This is the facts pass: it makes no AI calls and requires no AI backend
    to be configured. Run `enrich` afterwards to describe and classify the
    organized copies.
    """
    try:
        report = api.process(
            source, dest, catalog=catalog_path, duplicates_dir=duplicates_dir,
            sidecar=sidecar, recursive=not no_recursive, dry_run=dry_run,
        )
    except ConfigError as exc:
        raise _ConfigFailure(str(exc)) from exc

    if as_json:
        _emit_json(report.to_dict())
    else:
        if dry_run:
            click.echo("[DRY-RUN] No files were written.")
        c = report.counts
        click.echo(
            f"Done. Total={c[api.TOTAL]}  Copied={c[api.COPIED]}  "
            f"Duplicates={c[api.DUPLICATE]}  Errors={c[api.ERROR]}"
        )
    if not report.ok:
        sys.exit(1)
```

Use Task 1's `_ConfigFailure` (exit 2) at every `except ConfigError` site. Check `tests/test_cli.py` for any test that pins exit code 1 on a config failure (`grep -n "exit_code == 1" tests/test_cli.py`) and change it to 2, noting each in your report.

Replace the body of `enrich` with:

```python
def enrich(dest, catalog_path, sidecar, ai_backend, ai_base_url, ai_model, ai_timeout,
           openai_key, breaker_threshold, limit, reclassify, as_json) -> None:
    """Describe and classify already-organized images in DEST.

    Reads the organized copies, so the original source volume need not be
    mounted. Safe to interrupt and re-run: a file is only ever renamed when
    the result is strictly better.
    """
    ai = AIConfig(backend=ai_backend, base_url=ai_base_url, model=ai_model,
                  timeout=ai_timeout, api_key=openai_key)
    error: str | None = None
    try:
        report = api.enrich(
            dest, catalog=catalog_path, ai=ai, sidecar=sidecar,
            breaker_threshold=breaker_threshold, limit=limit, reclassify=reclassify,
        )
    except ConfigError as exc:
        raise _ConfigFailure(str(exc)) from exc
    except Aborted as exc:
        report = exc.report
        error = str(exc)
        assert report is not None  # api.enrich always attaches one

    if as_json:
        doc = report.to_dict()
        if error is not None:
            doc["error"] = error
        _emit_json(doc)
    else:
        c = report.counts
        click.echo(
            f"Enriched={c[api.ENRICHED]}  Renamed={c[api.RENAMED]}  "
            f"Errors={c[api.ERROR]}  Total={c[api.TOTAL]}"
        )
    if error is not None:
        click.echo(error, err=True)
        sys.exit(2)
    if not report.ok:
        sys.exit(1)
```

Note the existing `enrich` currently opens the catalog even when the file doesn't exist (creating an empty one). `api.enrich` raises `ConfigError` instead. `tests/test_cli.py::test_enrich_accepts_limit_and_reclassify` invokes `enrich` on an empty `dest` with no catalog and expects exit 0 — update that test to run `process` first on a one-file source (copy the pattern from `test_enrich_command_exists_and_reports` directly above it), and note the behaviour change in the CHANGELOG entry (Task 9): "`enrich` on a dest with no catalog is now a config error (exit 2) instead of silently creating an empty catalog."

Replace the body of `verify` with:

```python
def verify(path: Path, as_json: bool) -> None:
    """Verify organized-file integrity for PATH (file or directory).

    Every organized file embeds its SHA-256 digest in its filename (the last
    43 characters of the stem, Base64url-encoded); this re-hashes each file's
    content and confirms it still matches the digest embedded in its name.
    """
    try:
        report = api.verify(path)
    except ConfigError as exc:
        raise _ConfigFailure(str(exc)) from exc

    c = report.counts
    checked = c[api.OK] + c[api.FAILED]
    nothing_msg = "No organized image files (with an embedded digest) found to verify."

    if as_json:
        doc = report.to_dict()
        if checked == 0:
            doc["error"] = nothing_msg
        _emit_json(doc)
    else:
        for row in report.rows:
            if row.outcome == api.OK:
                click.echo(f"OK   {row.path}")
            else:
                click.echo(f"FAIL {row.path}", err=True)
        click.echo(
            f"\nVerified {checked} organized image(s) "
            f"({c[api.SKIPPED]} non-image/no-digest skipped): {c[api.OK]} OK, {c[api.FAILED]} FAILED"
        )
    if checked == 0:
        click.echo(nothing_msg, err=True)
        sys.exit(2)
    if c[api.FAILED]:
        sys.exit(1)
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_cli_json.py tests/test_cli.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
Expected: all pass. Every other `tests/test_cli.py` assertion on prose output must be untouched — if one breaks, the prose changed; fix the code, not the test.

- [ ] **Step 4: Commit**

```bash
git add imageharbor/cli.py tests/test_cli.py tests/test_cli_json.py
git commit -m "feat(cli): --json on process/enrich/verify; exit 2 for config errors and breaker aborts"
```

---

### Task 8: Public exports and packaging test

**Files:**
- Modify: `imageharbor/__init__.py`, `tests/test_packaging.py`

**Interfaces:**
- Produces: `from imageharbor import process, enrich, verify, stats, AIConfig, ProcessReport, EnrichReport, VerifyReport, ProcessRow, EnrichFailureRow, VerifyRow, COPIED, DUPLICATE, SKIPPED, ERROR, ENRICHED, RENAMED, TOTAL, AI, IO, OK, FAILED, ImageHarborError, ConfigError, Aborted, __version__` all resolve; `__all__` lists exactly these.

- [ ] **Step 1: Failing tests** (append to `tests/test_packaging.py`)

```python
PUBLIC_API = [
    "process", "enrich", "verify", "stats",
    "AIConfig",
    "ProcessReport", "EnrichReport", "VerifyReport",
    "ProcessRow", "EnrichFailureRow", "VerifyRow",
    "COPIED", "DUPLICATE", "SKIPPED", "ERROR", "ENRICHED", "RENAMED", "TOTAL",
    "AI", "IO", "OK", "FAILED",
    "ImageHarborError", "ConfigError", "Aborted",
    "__version__",
]


def test_the_package_exports_the_documented_public_api():
    import imageharbor

    assert set(imageharbor.__all__) == set(PUBLIC_API)
    for name in PUBLIC_API:
        assert hasattr(imageharbor, name), name


def test_the_public_api_imports_from_the_built_wheel(wheel_built_without_git: Path, tmp_path: Path):
    """The wheel, installed into a scratch venv with no dev extras, must
    expose every public name -- what `uv add git+…` gives a consumer."""
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True, text=True, timeout=120)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(venv), str(wheel_built_without_git)],
        check=True, capture_output=True, text=True, timeout=600,
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    code = "import imageharbor as ih; " + "; ".join(f"ih.{n}" for n in PUBLIC_API) + "; print('ok')"
    proc = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
```

Add `import os` to the test module's imports if absent. Run: `uv run pytest tests/test_packaging.py -q -k public_api` → FAIL on `__all__`.

- [ ] **Step 2: Implement `imageharbor/__init__.py`** — keep the licence docstring and `__version__` block; append:

```python
from .api import (  # noqa: E402  -- __version__ must resolve first
    AI,
    COPIED,
    DUPLICATE,
    ENRICHED,
    ERROR,
    FAILED,
    IO,
    OK,
    RENAMED,
    SKIPPED,
    TOTAL,
    Aborted,
    AIConfig,
    ConfigError,
    EnrichFailureRow,
    EnrichReport,
    ImageHarborError,
    ProcessReport,
    ProcessRow,
    VerifyReport,
    VerifyRow,
    enrich,
    process,
    stats,
    verify,
)

__all__ = [
    "process", "enrich", "verify", "stats",
    "AIConfig",
    "ProcessReport", "EnrichReport", "VerifyReport",
    "ProcessRow", "EnrichFailureRow", "VerifyRow",
    "COPIED", "DUPLICATE", "SKIPPED", "ERROR", "ENRICHED", "RENAMED", "TOTAL",
    "AI", "IO", "OK", "FAILED",
    "ImageHarborError", "ConfigError", "Aborted",
    "__version__",
]
```

Then check for an import cycle: `uv run python -c "import imageharbor.api"` and `uv run python -c "import imageharbor.cli"` must both succeed. `api.py` imports `.ai_classifier`, `.catalog`, `.pipeline`, `.enrich`, `.circuit_breaker`, `.discovery`, `.hashing`, `.util` — none of those import `imageharbor` as a package attribute at module scope (`dashboard/stats.py` does `from imageharbor import tiers`, which is why Task 6 kept the dashboard imports local). If a cycle appears, make the offending `api.py` import local to the function that needs it, the same way.

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_packaging.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```

- [ ] **Step 4: Commit**

```bash
git add imageharbor/__init__.py tests/test_packaging.py
git commit -m "feat: export the public library API from the imageharbor package"
```

---

### Task 9: Documentation — README, CLAUDE.md, CHANGELOG

**Files:**
- Modify: `README.md` (new sections after "Main commands", line ~131-150), `CLAUDE.md` (Commands table, `api.py` bullet, `cli.py` bullet, a new Critical invariants entry), `CHANGELOG.md` (new top entry)

**Interfaces:** none produced; consumes the exact names/shapes from Tasks 3-8.

- [ ] **Step 1: README** — insert after the "Main commands" section:

````markdown
## Calling from Python

ImageHarbor is importable. Install it from git:

```bash
uv add git+https://github.com/conradstorz/ImageHarbor
```

```python
from imageharbor import process, enrich, verify, stats, AIConfig, ImageHarborError, Aborted, ERROR

try:
    report = process("/photos/incoming", "/photos/organized")
    if not report.ok:
        for row in report.rows:
            if row.outcome == ERROR:
                print("failed:", row.source_path, row.detail)
    enriched = enrich("/photos/organized", ai=AIConfig(backend="openai", base_url="http://jetson:11434/v1", model="llava"))
except Aborted as exc:          # breaker tripped; exc.report is the partial report
    print("backend down after", exc.report.counts)
except ImageHarborError as exc:  # ConfigError: bad paths, unknown backend, missing extra
    raise SystemExit(str(exc))

print(verify("/photos/organized").counts)
print(stats("/photos/organized/catalog.db")["library"])
```

Every call opens and closes its own catalog, blocks for the pass, and never
raises for a single bad file — that becomes an `ERROR` row. `report.to_dict()`
is the same document `--json` prints. `stats()` returns the dashboard's
`/api/stats` document without a running `watch`.

Only these names are promised: `process`, `enrich`, `verify`, `stats`,
`AIConfig`, the three `*Report` and three `*Row` types, the outcome constants,
and `ImageHarborError`/`ConfigError`/`Aborted`. Everything else in the package
is importable but may change.

**Licence note.** ImageHarbor is AGPL-3.0-or-later. Importing it in-process
places the importing program under the AGPL; running it as a subprocess (below)
does not.

## Calling as a subprocess

`process`, `enrich`, and `verify` take `--json`: stdout is exactly one JSON
document (the report's `to_dict()`), diagnostics go to stderr, and the exit
code follows nas-ingest's convention:

| exit | meaning |
|------|---------|
| 0 | ok |
| 1 | finished, but the report has `ERROR` rows (`verify`: `FAILED` rows) |
| 2 | could not start (config error) or did not finish (breaker abort). A document with an `"error"` key is still printed when a report exists. |

```python
import json, subprocess
r = subprocess.run(["imageharbor", "process", "--source", src, "--dest", dest, "--json"],
                   capture_output=True, text=True)
if r.returncode in (0, 1):
    report = json.loads(r.stdout)
    print(report["counts"])
```

Report shapes (keys are additive-only from here):

```
process: {source, dest, catalog, dry_run, started, finished,
          counts: {COPIED, DUPLICATE, SKIPPED, ERROR, TOTAL},
          rows: [{source_path, outcome, dest_path, digest, detail}]}
enrich:  {dest, catalog, ai_backend, started, finished, aborted,
          counts: {ENRICHED, RENAMED, ERROR, TOTAL},
          rows: [{digest, dest_path, reason: "AI"|"IO", detail}]}   # failures only
verify:  {path, started, finished,
          counts: {OK, FAILED, SKIPPED},
          rows: [{path, outcome: "OK"|"FAILED", digest}]}
```
````

- [ ] **Step 2: CLAUDE.md**
  - Commands table: add a row `| Same, as one JSON document on stdout (exit 0/1/2) | `uv run imageharbor process … --json` (also `enrich`, `verify`) |` after the "Re-verify integrity" row.
  - Replace the Task 1 stub `api.py` bullet with: "**`api.py`** — the public library facade and the only promised import surface: `process()`, `enrich()`, `verify()`, `stats()`, frozen `ProcessReport`/`EnrichReport`/`VerifyReport` (+ row types) with `to_dict()`, `AIConfig`, and `ImageHarborError` → `ConfigError` (could not start) / `Aborted` (breaker tripped; carries the partial `EnrichReport` as `.report`). It only *wraps* the orchestrators — no placement, naming, hashing, or catalog logic lives here. A per-file problem is a row, never an exception. `stats()` imports the dashboard package *inside* the function so `import imageharbor` stays light. Re-exported from `imageharbor/__init__.py`; `__all__` there is the contract. Spec: `docs/superpowers/specs/2026-09-18-library-api-design.md`."
  - `cli.py` bullet: add "`process`/`enrich`/`verify` are thin renderers over `api.*`: `--json` prints `report.to_dict()` as the only stdout line; exit 0 = ok, 1 = ERROR/FAILED rows, 2 = `ConfigError` or `Aborted` (`_ConfigFailure` is `ClickException` with `exit_code = 2`)."
  - Critical invariants, new bullet at the end: "**The `--json` document and exit codes are a stable contract.** `organize-my-life` consumes them as a subprocess tool. `to_dict()` keys may be added, never removed or retyped; the outcome constants and the 0/1/2 exit mapping do not change. The same goes for every name in `imageharbor.__all__`."

- [ ] **Step 3: CHANGELOG** — new entry above `[1.3.0]`:

```markdown
## [1.4.0] — unreleased

Library API and `--json` contract, for `organize-my-life`:

- `from imageharbor import process, enrich, verify, stats` — a promised
  facade over the two passes, returning frozen reports with `to_dict()`.
  `ImageHarborError` → `ConfigError` / `Aborted` (the latter carries the
  partial `EnrichReport`). `AIConfig` replaces five loose AI parameters.
- `--json` on `process`, `enrich`, `verify`: one document on stdout,
  diagnostics on stderr.
- Exit codes now follow nas-ingest: 0 ok, 1 ERROR/FAILED rows, 2 could not
  start or did not finish. **Behaviour change:** config errors (dest inside
  source, unknown backend, missing `openai` extra), `enrich` on a breaker
  trip, and `verify` with nothing verifiable all exit 2 (were 1); `enrich`
  on a dest with no catalog is a config error (exit 2) instead of silently
  creating an empty catalog.
- `EnrichStats.failures` records path and AI/IO reason per failed row
  (additive; `ai_failed`/`io_failed` unchanged).
```

- [ ] **Step 4: Verify docs claims against code** — run the README's Python snippet's *import line* and the subprocess snippet shape:

```bash
uv run python -c "from imageharbor import process, enrich, verify, stats, AIConfig, ImageHarborError, Aborted, ERROR; print('ok')"
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md CHANGELOG.md
git commit -m "docs: library API, --json contract, exit codes; CLAUDE.md api.py bullet and contract invariant"
```

---

### Task 10: Whole-branch review and hand-off

**Files:** none new.

- [ ] **Step 1: Full gate**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
```
Expected: baseline + all new tests, 0 failures.

- [ ] **Step 2: Contract smoke test end-to-end** on a scratch tree (under the session scratchpad, not the repo):

```bash
uv run imageharbor process --source <scratch>/src --dest <scratch>/org --json
uv run imageharbor enrich --dest <scratch>/org --json
uv run imageharbor verify <scratch>/org --json
```
Each must print exactly one line of JSON and exit 0; pipe each through `python -c "import json,sys; json.load(sys.stdin)"` to prove it parses. Record the three exit codes in your report.

- [ ] **Step 3: Grep for leftovers**

```bash
grep -rn "_build_classifier(\"\|_build_classifier('" imageharbor tests
grep -rn "exit_code != 0" tests/test_cli.py
```
The first must return nothing (no old positional-signature calls). For the second, each hit should be one where the exact code genuinely doesn't matter; tighten any that now has a defined code.

- [ ] **Step 4:** Invoke `superpowers:finishing-a-development-branch` for the merge/PR decision. The PR body must mention the three behaviour changes from the CHANGELOG and link the spec.
