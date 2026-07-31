# Dockerized Watcher + OpenAI-Compatible AI Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy ImageHarbor as a containerized continuous polling watcher that organizes a NAS photo library, classifying images via a self-hosted OpenAI-compatible AI server (Jetson).

**Architecture:** Generalize the existing `OpenAIClassifier` to any OpenAI-compatible endpoint (configurable base URL + model + timeout). Add an `imageharbor watch` command: a poll loop that discovers source files, skips unchanged ones via a local "seen" cache (avoiding re-hashing over the network), and runs the existing pipeline on new/changed files. Package as a Docker image + compose file that bind-mounts a read-only NAS source and a read-write NAS dest, keeping the SQLite catalog on a local volume.

**Tech Stack:** Python 3.10+, Click, Pillow, the `openai` SDK (optional extra), SQLite, Docker / docker-compose. Managed with `uv`.

## Global Constraints

- Python floor: `requires-python = ">=3.10"`. Do not use newer-only syntax.
- Runtime deps limited to `Pillow` + `click`; `openai` only via the `openai` extra (imported lazily). No new third-party runtime deps.
- Container base image: `python:3.12-slim`, amd64, runs as a **non-root** user.
- Originals are read-only; the organized copy is verified before it is catalogued; filenames stay self-verifying (SHA-256 embedded). Preserve all existing invariants.
- The catalog (`catalog.db`) is SQLite and must live on a **local** volume, never the NAS mount.
- Tests must be **offline and deterministic** — no network, no real `openai` package required, no real Docker build in the test suite.
- Local dev/test commands use `uv` (e.g. `uv run pytest`). Do not chain shell commands with `&&` (run them as separate commands).
- Match existing code/test style (plain pytest functions or `Test*` classes, `tmp_path`, minimal JPEG byte blobs, section-comment headers).

---

## File Structure

**Modify:**
- `imageharbor/ai_classifier.py` — `OpenAIClassifier.__init__` gains `base_url`, `timeout`, and an api-key placeholder.
- `imageharbor/catalog.py` — new `source_seen` table + `source_is_unchanged` / `record_source_seen` methods.
- `imageharbor/cli.py` — a `_build_classifier` helper, new AI options on `process`, and a new `watch` command with signal handling.
- `tests/test_ai_classifier.py`, `tests/test_catalog.py`, `tests/test_cli.py` — new tests.
- `CLAUDE.md` — commands table gains `watch` + Docker rows.

**Create:**
- `imageharbor/watcher.py` — `run_pass()` and `watch()` (loop core, no CLI/signal coupling).
- `tests/test_watcher.py` — watcher unit tests.
- `Dockerfile`, `docker-compose.yml` — packaging.
- `docs/deploy-docker.md` — deployment guide.

---

## Task 1: Generalize `OpenAIClassifier` to any OpenAI-compatible endpoint

**Files:**
- Modify: `imageharbor/ai_classifier.py` (the `OpenAIClassifier.__init__` around lines 177-189)
- Test: `tests/test_ai_classifier.py`

**Interfaces:**
- Produces: `OpenAIClassifier(api_key: str | None = None, model: str = "gpt-4o-mini", base_url: str | None = None, timeout: float = 60.0)`. `.classify()` signature unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ai_classifier.py` (near the other OpenAI tests; it uses the same `sys`/`types` injection pattern already present in that file):

```python
def test_openai_classifier_passes_base_url_model_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("openai")
    fake.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)

    clf = OpenAIClassifier(
        api_key=None,
        model="llava",
        base_url="http://jetson.local:11434/v1",
        timeout=30.0,
    )

    assert captured["base_url"] == "http://jetson.local:11434/v1"
    assert captured["timeout"] == 30.0
    assert captured["api_key"] == "not-needed"  # placeholder when none supplied
    assert clf._model == "llava"
    assert clf.MODEL_VERSION == "llava"
```

Ensure `import types` and `import sys` are present at the top of the test file (add if missing).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_classifier.py::test_openai_classifier_passes_base_url_model_timeout -v`
Expected: FAIL — `OpenAIClassifier.__init__` does not accept `base_url` (TypeError).

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/ai_classifier.py`, replace the `OpenAIClassifier.__init__` body:

```python
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAIClassifier. "
                "Install it with: pip install imageharbor[openai]"
            ) from exc

        self._openai = _openai
        # Local OpenAI-compatible servers (e.g. Ollama on the Jetson) usually
        # ignore the API key, but the SDK requires a non-empty value, so fall
        # back to a placeholder when none is supplied.
        self._client = _openai.OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout,
        )
        self._model = model
        self.MODEL_VERSION = model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ai_classifier.py -q`
Expected: PASS (all existing OpenAI tests plus the new one).

- [ ] **Step 5: Commit**

```bash
git add imageharbor/ai_classifier.py tests/test_ai_classifier.py
git commit -m "feat: make OpenAIClassifier target any OpenAI-compatible endpoint"
```

---

## Task 2: Add the `source_seen` cache to the catalog

**Files:**
- Modify: `imageharbor/catalog.py` (the `_SCHEMA` string ~lines 14-37; add methods in the read/write sections)
- Test: `tests/test_catalog.py`

**Interfaces:**
- Produces:
  - `Catalog.source_is_unchanged(source_path: str, size: int, mtime_ns: int) -> bool`
  - `Catalog.record_source_seen(source_path: str, size: int, mtime_ns: int, sha256_b64url: str | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_catalog.py`:

```python
def test_source_seen_unknown_is_not_unchanged(catalog: Catalog) -> None:
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 111) is False


def test_source_seen_roundtrip_unchanged(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111, "A" * 43)
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 111) is True


def test_source_seen_changed_size_is_not_unchanged(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111)
    assert catalog.source_is_unchanged("/src/a.jpg", 200, 111) is False


def test_source_seen_changed_mtime_is_not_unchanged(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111)
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 222) is False


def test_source_seen_upsert_updates(catalog: Catalog) -> None:
    catalog.record_source_seen("/src/a.jpg", 100, 111)
    catalog.record_source_seen("/src/a.jpg", 100, 999)  # file changed
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 999) is True
    assert catalog.source_is_unchanged("/src/a.jpg", 100, 111) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_catalog.py -k source_seen -v`
Expected: FAIL — `Catalog` has no attribute `source_is_unchanged`.

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/catalog.py`, append to the `_SCHEMA` string (before the closing `"""`, after the existing indexes):

```sql

CREATE TABLE IF NOT EXISTS source_seen (
    source_path   TEXT    PRIMARY KEY,
    size          INTEGER NOT NULL,
    mtime_ns      INTEGER NOT NULL,
    sha256_b64url TEXT,
    seen_at       TEXT    NOT NULL
);
```

Then add these methods to the `Catalog` class (in the Read/Write sections):

```python
    def source_is_unchanged(self, source_path: str, size: int, mtime_ns: int) -> bool:
        """Return True if this source path was seen before with the same size
        and mtime (so it can be skipped without re-hashing)."""
        cur = self._conn.execute(
            "SELECT size, mtime_ns FROM source_seen WHERE source_path=?",
            (source_path,),
        )
        row = cur.fetchone()
        if row is None:
            return False
        return row["size"] == size and row["mtime_ns"] == mtime_ns

    def record_source_seen(
        self,
        source_path: str,
        size: int,
        mtime_ns: int,
        sha256_b64url: str | None = None,
    ) -> None:
        """Record (or update) that a source file was processed, keyed by path."""
        self._conn.execute(
            """
            INSERT INTO source_seen (source_path, size, mtime_ns, sha256_b64url, seen_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(source_path) DO UPDATE SET
                size          = excluded.size,
                mtime_ns      = excluded.mtime_ns,
                sha256_b64url = excluded.sha256_b64url,
                seen_at       = excluded.seen_at
            """,
            (source_path, size, mtime_ns, sha256_b64url, _now_iso()),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: add source_seen cache to catalog for watch dedup"
```

---

## Task 3: Watcher core (`run_pass` + `watch` loop)

**Files:**
- Create: `imageharbor/watcher.py`
- Test: `tests/test_watcher.py`

**Interfaces:**
- Consumes: `Pipeline.process_file(path) -> ProcessResult` (existing); `Catalog.source_is_unchanged` / `record_source_seen` (Task 2); `discover_images(source, recursive)` (existing).
- Produces:
  - `WatchStats` dataclass with `passes: int`, `processed: int`, `skipped_unchanged: int`, `errors: int`.
  - `run_pass(*, pipeline: Pipeline, catalog: Catalog, source: Path, recursive: bool = True) -> WatchStats`
  - `watch(*, pipeline: Pipeline, catalog: Catalog, source: Path, interval: float, recursive: bool = True, stop_event: threading.Event | None = None, sleep: Callable[[float], bool] | None = None) -> WatchStats`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_watcher.py`:

```python
"""Tests for the continuous polling watcher."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.pipeline import Pipeline
from imageharbor.watcher import WatchStats, run_pass, watch


def _make_jpeg(path: Path, content: bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9") -> Path:
    path.write_bytes(content)
    return path


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    _make_jpeg(src / "beach_photo.jpg")
    _make_jpeg(src / "mountain_view.jpg", b"\xff\xd8\xff\xe0" + b"\x01" * 16 + b"\xff\xd9")
    return src


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


def test_run_pass_processes_new_files(source_dir: Path, organized_dir: Path, catalog: Catalog) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)
    assert stats.processed == 2
    assert stats.skipped_unchanged == 0
    assert stats.errors == 0


def test_run_pass_second_pass_skips_unchanged_without_hashing(
    source_dir: Path, organized_dir: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    # On the second pass nothing changed: process_file must NOT be called.
    calls = []
    real_process = pipeline.process_file

    def _spy(path):
        calls.append(path)
        return real_process(path)

    monkeypatch.setattr(pipeline, "process_file", _spy)
    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    assert calls == []  # unchanged files never re-processed / re-hashed
    assert stats.skipped_unchanged == 2
    assert stats.processed == 0


def test_run_pass_reprocesses_changed_file(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)

    # Change one file's bytes (new content -> new size/mtime).
    target = source_dir / "beach_photo.jpg"
    _make_jpeg(target, b"\xff\xd8\xff\xe0" + b"\x02" * 40 + b"\xff\xd9")

    stats = run_pass(pipeline=pipeline, catalog=catalog, source=source_dir)
    assert stats.processed == 1
    assert stats.skipped_unchanged == 1


def test_watch_runs_one_pass_then_stops(source_dir: Path, organized_dir: Path, catalog: Catalog) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stop = threading.Event()

    # Fake sleep sets the stop event so the loop exits after exactly one pass.
    def _sleep(_interval: float) -> bool:
        stop.set()
        return True

    wstats = watch(
        pipeline=pipeline,
        catalog=catalog,
        source=source_dir,
        interval=1.0,
        stop_event=stop,
        sleep=_sleep,
    )
    assert wstats.passes == 1


def test_watch_exits_immediately_if_stop_already_set(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stop = threading.Event()
    stop.set()
    wstats = watch(pipeline=pipeline, catalog=catalog, source=source_dir, interval=1.0, stop_event=stop)
    assert wstats.passes == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_watcher.py -q`
Expected: FAIL — `imageharbor.watcher` does not exist (ImportError).

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/watcher.py`:

```python
"""Continuous polling watcher for ImageHarbor.

Rescans the source on an interval and processes new/changed files, using the
catalog's source_seen cache to skip unchanged files without re-hashing them
(cheap os.stat instead of a full network read). Filesystem event watching is
deliberately not used: inotify does not work reliably over SMB/CIFS mounts.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .catalog import Catalog
from .discovery import discover_images
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class WatchStats:
    passes: int = 0
    processed: int = 0
    skipped_unchanged: int = 0
    errors: int = 0


def run_pass(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    recursive: bool = True,
) -> WatchStats:
    """Process new/changed files once. Unchanged files (per the source_seen
    cache) are skipped without hashing."""
    stats = WatchStats()
    for path in discover_images(source, recursive=recursive):
        st = path.stat()
        if catalog.source_is_unchanged(str(path), st.st_size, st.st_mtime_ns):
            stats.skipped_unchanged += 1
            continue
        result = pipeline.process_file(path)
        if result.status in ("copied", "duplicate"):
            # Only record success so a transient error is retried next pass.
            catalog.record_source_seen(
                str(path), st.st_size, st.st_mtime_ns, result.sha256_b64url
            )
            stats.processed += 1
        else:
            stats.errors += 1
    return stats


def watch(
    *,
    pipeline: Pipeline,
    catalog: Catalog,
    source: Path,
    interval: float,
    recursive: bool = True,
    stop_event: threading.Event | None = None,
    sleep: Callable[[float], bool] | None = None,
) -> WatchStats:
    """Run passes until stop_event is set. An immediate first pass runs before
    the first sleep. ``sleep`` defaults to ``stop_event.wait`` so a signal
    interrupts the wait promptly."""
    stop_event = stop_event or threading.Event()
    if sleep is None:
        sleep = stop_event.wait  # interruptible sleep
    wstats = WatchStats()
    while not stop_event.is_set():
        pass_stats = run_pass(
            pipeline=pipeline, catalog=catalog, source=source, recursive=recursive
        )
        wstats.passes += 1
        wstats.processed += pass_stats.processed
        wstats.skipped_unchanged += pass_stats.skipped_unchanged
        wstats.errors += pass_stats.errors
        logger.info(
            "watch pass %d: processed=%d skipped=%d errors=%d",
            wstats.passes,
            pass_stats.processed,
            pass_stats.skipped_unchanged,
            pass_stats.errors,
        )
        if stop_event.is_set():
            break
        sleep(interval)
    return wstats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_watcher.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/watcher.py tests/test_watcher.py
git commit -m "feat: add polling watcher core with seen-cache dedup"
```

---

## Task 4: CLI — classifier helper, AI options on `process`, and the `watch` command

**Files:**
- Modify: `imageharbor/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `OpenAIClassifier(...)` (Task 1); `imageharbor.watcher.watch` (Task 3); `Pipeline`, `Catalog` (existing).
- Produces: a `watch` CLI command; a private `_build_classifier(ai_backend, api_key, base_url, model, timeout)` helper.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
def test_build_classifier_stub_default() -> None:
    from imageharbor.cli import _build_classifier
    from imageharbor.ai_classifier import StubClassifier

    clf = _build_classifier("stub", None, None, "gpt-4o-mini", 60.0)
    assert isinstance(clf, StubClassifier)


def test_cli_watch_wires_args(monkeypatch, tmp_path):
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9")
    dest = tmp_path / "dest"

    captured = {}

    def _fake_watch(**kwargs):
        captured.update(kwargs)
        return WatchStats(passes=1)

    monkeypatch.setattr(_watcher, "watch", _fake_watch)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["watch", "--source", str(src), "--dest", str(dest), "--interval", "5"],
    )
    assert result.exit_code == 0, result.output
    assert captured["interval"] == 5.0
    assert captured["source"] == src
```

(`CliRunner` and `main` are already imported at the top of `tests/test_cli.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "build_classifier or watch" -v`
Expected: FAIL — `_build_classifier` / `watch` command do not exist.

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/cli.py`, add the helper (after the imports, before the `process` command):

```python
def _build_classifier(
    ai_backend: str,
    api_key: str | None,
    base_url: str | None,
    model: str,
    timeout: float,
):
    """Construct the AI classifier for the chosen backend. Raises a clean
    ClickException if the optional 'openai' package is missing."""
    if ai_backend == "openai":
        from .ai_classifier import OpenAIClassifier

        try:
            return OpenAIClassifier(
                api_key=api_key, model=model, base_url=base_url, timeout=timeout
            )
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc
    from .ai_classifier import StubClassifier

    return StubClassifier()
```

Refactor the existing `process` command's classifier construction (the `if ai_backend == "openai": ... else: ...` block) to a single call, and add the new shared AI options. Replace the classifier-building block in `process` with:

```python
    classifier = _build_classifier(ai_backend, openai_key, ai_base_url, ai_model, ai_timeout)
```

and add these options to `process` (alongside the existing `--ai` / `--openai-key`):

```python
@click.option("--ai-base-url", envvar="IMAGEHARBOR_AI_BASE_URL", default=None,
              help="Base URL of an OpenAI-compatible server (e.g. a local Jetson).")
@click.option("--ai-model", envvar="IMAGEHARBOR_AI_MODEL", default="gpt-4o-mini",
              show_default=True, help="Model name for the AI backend.")
@click.option("--ai-timeout", envvar="IMAGEHARBOR_AI_TIMEOUT", default=60.0,
              show_default=True, type=float, help="AI request timeout (seconds).")
```

Update the `process` function signature to accept `ai_base_url: str | None, ai_model: str, ai_timeout: float`, and change its `--openai-key` option to read either env var:

```python
@click.option("--openai-key", "openai_key", default=None,
              envvar=["IMAGEHARBOR_AI_API_KEY", "OPENAI_API_KEY"],
              help="API key (or set IMAGEHARBOR_AI_API_KEY / OPENAI_API_KEY).")
```

Then add the new `watch` command (after `process`):

```python
@main.command()
@click.option("--source", envvar="IMAGEHARBOR_SOURCE", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
              help="Read-only source directory (or single image file).")
@click.option("--dest", envvar="IMAGEHARBOR_DEST", required=True,
              type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              help="Root directory for the organized library.")
@click.option("--catalog", "catalog_path", envvar="IMAGEHARBOR_CATALOG", default=None,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.")
@click.option("--interval", envvar="IMAGEHARBOR_INTERVAL", default=300.0, show_default=True,
              type=float, help="Seconds between watch passes.")
@click.option("--duplicates", "duplicates_dir", envvar="IMAGEHARBOR_DUPLICATES", default=None,
              type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              help="Directory to copy duplicates into.")
@click.option("--sidecar/--no-sidecar", envvar="IMAGEHARBOR_SIDECAR", default=False,
              show_default=True, help="Write a JSON sidecar alongside each organized image.")
@click.option("--ai", "ai_backend", envvar="IMAGEHARBOR_AI", default="stub", show_default=True,
              type=click.Choice(["stub", "openai"], case_sensitive=False),
              help="AI classification backend.")
@click.option("--ai-base-url", envvar="IMAGEHARBOR_AI_BASE_URL", default=None,
              help="Base URL of an OpenAI-compatible server (e.g. a local Jetson).")
@click.option("--ai-model", envvar="IMAGEHARBOR_AI_MODEL", default="gpt-4o-mini",
              show_default=True, help="Model name for the AI backend.")
@click.option("--ai-timeout", envvar="IMAGEHARBOR_AI_TIMEOUT", default=60.0,
              show_default=True, type=float, help="AI request timeout (seconds).")
@click.option("--openai-key", "openai_key", default=None,
              envvar=["IMAGEHARBOR_AI_API_KEY", "OPENAI_API_KEY"],
              help="API key (or set IMAGEHARBOR_AI_API_KEY / OPENAI_API_KEY).")
@click.option("--no-recursive", is_flag=True, default=False,
              help="Do not recurse into sub-directories.")
def watch(
    source: Path,
    dest: Path,
    catalog_path: Path | None,
    interval: float,
    duplicates_dir: Path | None,
    sidecar: bool,
    ai_backend: str,
    ai_base_url: str | None,
    ai_model: str,
    ai_timeout: float,
    openai_key: str | None,
    no_recursive: bool,
) -> None:
    """Continuously watch SOURCE and organize new/changed photos into DEST."""
    import signal
    import threading

    from . import watcher as _watcher

    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    classifier = _build_classifier(ai_backend, openai_key, ai_base_url, ai_model, ai_timeout)
    dest.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    def _handle(signum, _frame):
        click.echo(f"Received signal {signum}; shutting down after current pass.")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    with Catalog(catalog_path) as catalog:
        pipeline = Pipeline(
            source_dir=source,
            organized_dir=dest,
            catalog=catalog,
            classifier=classifier,
            duplicates_dir=duplicates_dir,
            write_sidecars=sidecar,
        )
        click.echo(f"Watching {source} -> {dest} every {interval:.0f}s (Ctrl-C to stop).")
        stats = _watcher.watch(
            pipeline=pipeline,
            catalog=catalog,
            source=source,
            interval=interval,
            recursive=not no_recursive,
            stop_event=stop_event,
        )

    click.echo(
        f"Stopped after {stats.passes} pass(es). "
        f"Processed={stats.processed} Skipped={stats.skipped_unchanged} Errors={stats.errors}"
    )
```

Note: the `watch` test monkeypatches `imageharbor.watcher.watch`, so the loop returns immediately; real signals are never delivered during the test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -q`
Expected: PASS (existing `process` tests still pass with the new default AI options; new tests pass).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — all tests green.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/cli.py tests/test_cli.py
git commit -m "feat: add 'watch' command and shared AI backend options"
```

---

## Task 5: Docker packaging

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Interfaces:** none (packaging only). Deliverable is a buildable image whose default command is `imageharbor watch`.

- [ ] **Step 1: Write the `Dockerfile`**

Create `Dockerfile`:

```dockerfile
# ImageHarbor watcher image (amd64).
FROM python:3.12-slim

# Non-root runtime user.
RUN useradd --create-home --uid 1000 harbor

WORKDIR /app

# Install the package with the OpenAI-compatible classifier extra.
COPY pyproject.toml README.md ./
COPY imageharbor ./imageharbor
RUN pip install --no-cache-dir ".[openai]"

# Default mount points (see docker-compose.yml).
ENV IMAGEHARBOR_SOURCE=/data/source \
    IMAGEHARBOR_DEST=/data/dest \
    IMAGEHARBOR_CATALOG=/data/catalog/catalog.db

USER harbor

ENTRYPOINT ["imageharbor"]
CMD ["watch"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

Create `docker-compose.yml`:

```yaml
# ImageHarbor continuous watcher.
#
# Prerequisites on the Docker host (NOT in this file):
#   - Mount the NAS source share read-only, e.g. at /mnt/nas/photos
#   - Mount the NAS organized share read-write, e.g. at /mnt/nas/photos-organized
#     (via /etc/fstab or autofs; NAS credentials live there, not here.)
services:
  imageharbor:
    build: .
    image: imageharbor:latest
    command: watch
    environment:
      IMAGEHARBOR_SOURCE: /data/source
      IMAGEHARBOR_DEST: /data/dest
      IMAGEHARBOR_CATALOG: /data/catalog/catalog.db
      IMAGEHARBOR_INTERVAL: "300"
      IMAGEHARBOR_AI: openai
      IMAGEHARBOR_AI_BASE_URL: http://jetson.local:11434/v1
      IMAGEHARBOR_AI_MODEL: llava
      IMAGEHARBOR_AI_API_KEY: not-needed
      IMAGEHARBOR_AI_TIMEOUT: "60"
    volumes:
      - /mnt/nas/photos:/data/source:ro          # read-only source
      - /mnt/nas/photos-organized:/data/dest     # read-write organized library
      - imageharbor-catalog:/data/catalog        # local catalog volume (NOT on NAS)
    restart: unless-stopped

volumes:
  imageharbor-catalog:
```

- [ ] **Step 3: Verify the image builds (manual smoke test)**

Run: `docker build -t imageharbor:latest .`
Expected: build succeeds; the final image installs `imageharbor` with the `openai` extra.

Run: `docker run --rm imageharbor:latest --help`
Expected: prints the CLI help listing `catalog`, `process`, `verify`, and `watch`.

Run: `docker run --rm imageharbor:latest watch --help`
Expected: prints the `watch` options including `--interval`, `--ai-base-url`, `--ai-model`.

(If Docker is unavailable in the working environment, note that and defer these three runs to the deploy host.)

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add Dockerfile and compose for the watcher service"
```

---

## Task 6: Deployment docs + CLAUDE.md

**Files:**
- Create: `docs/deploy-docker.md`
- Modify: `CLAUDE.md` (Commands table)

**Interfaces:** none (documentation).

- [ ] **Step 1: Write `docs/deploy-docker.md`**

Create `docs/deploy-docker.md`:

```markdown
# Deploying ImageHarbor as a Docker watcher

ImageHarbor can run as a continuous watcher container on an always-on amd64
Linux host, organizing a NAS photo library and classifying images with a
self-hosted OpenAI-compatible AI server (e.g. a Jetson running Ollama).

## 1. Mount the NAS on the host

The container uses **host bind-mounts**, so mount the NAS shares on the Docker
host first (credentials stay here, never in the container). Example CIFS entries
in `/etc/fstab`:

```
//DS220plus/photos          /mnt/nas/photos            cifs  ro,credentials=/etc/nas.cred,uid=1000,iocharset=utf8  0 0
//DS220plus/photos-organized /mnt/nas/photos-organized  cifs  rw,credentials=/etc/nas.cred,uid=1000,iocharset=utf8  0 0
```

`/etc/nas.cred` holds `username=` / `password=` (chmod 600). Adjust share names
to your NAS. NFS works equally well.

## 2. Point at your AI server

Edit `docker-compose.yml` environment:

- `IMAGEHARBOR_AI: openai`
- `IMAGEHARBOR_AI_BASE_URL`: your server's OpenAI-compatible endpoint
  (e.g. `http://jetson.local:11434/v1` for Ollama).
- `IMAGEHARBOR_AI_MODEL`: a vision model available on that server (e.g. `llava`).
- `IMAGEHARBOR_AI_API_KEY`: usually `not-needed` for local servers.

To run without AI (filename-keyword stub), set `IMAGEHARBOR_AI: stub`.

## 3. Build and run

```
docker compose build
docker compose up -d
docker compose logs -f
```

Each pass logs `watch pass N: processed=.. skipped=.. errors=..`. The catalog is
kept on the local `imageharbor-catalog` volume; organized copies are written to
the NAS. Originals are never modified.

## 4. Smoke test

```
docker run --rm imageharbor:latest --help
docker run --rm imageharbor:latest watch --help
```

Then verify integrity of the organized library at any time:

```
docker compose run --rm imageharbor verify /data/dest
```

## Notes

- Watching is **poll-based** (default 300s via `IMAGEHARBOR_INTERVAL`), because
  filesystem events (inotify) are unreliable over SMB/CIFS. Unchanged files are
  skipped without re-reading them, using a local seen-cache.
- Keep only one watcher instance per catalog (the catalog is single-writer).
```

- [ ] **Step 2: Update `CLAUDE.md` commands table**

In `CLAUDE.md`, add these rows to the Commands table (after the `verify` row):

```markdown
| Watch a library continuously | `uv run imageharbor watch --source SRC --dest DEST` |
| Build the Docker image | `docker build -t imageharbor:latest .` |
| Run the watcher (compose) | `docker compose up -d` (see `docs/deploy-docker.md`) |
```

- [ ] **Step 3: Commit**

```bash
git add docs/deploy-docker.md CLAUDE.md
git commit -m "docs: add Docker deployment guide and watch/docker commands"
```

---

## Self-Review

**Spec coverage:**
- §3 generalized classifier → Task 1. ✓
- §4.1 watch behavior + §4.3 signals → Tasks 3 (core) + 4 (signal wiring). ✓
- §4.2 seen-source-files cache → Task 2 (storage) + Task 3 (use). ✓
- §5 Docker packaging → Task 5. ✓
- §6 config env vars → Task 4 (`envvar=` on every option) + Task 5 (compose). ✓
- §7 testing → tests in Tasks 1-4. ✓
- §8 docs → Task 6. ✓
- §10 acceptance criteria → covered by Tasks 1-5 and the full-suite run in Task 4 Step 5.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `WatchStats` fields (`passes`, `processed`, `skipped_unchanged`, `errors`) are defined in Task 3 and consumed identically in Task 4's summary line. `_build_classifier(ai_backend, api_key, base_url, model, timeout)` is defined and called with matching argument order in Task 4. `OpenAIClassifier(api_key, model, base_url, timeout)` matches Task 1. `run_pass`/`watch` keyword-only signatures match their call sites. ✓

**Deferred (per spec §9):** true inotify watching, running on the NAS, human "Review/" workflow, multi-instance watchers — intentionally not planned.
