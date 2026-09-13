# R1 "Foundation" (v1.0.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship ImageHarbor's first official release (v1.0.0) with CI as the merge gate, tag-driven versioning that auto-increments on every merge to `main`, lockfile-faithful Docker CD to GHCR, and the codebase cleanups from the 2026-09-12 quality review.

**Architecture:** Cleanups and lint/type enforcement land first (Tasks 1–10), then the CI workflow that runs them (Task 11), then tag-derived versioning via `setuptools-scm` (Task 12), docs/changelog/metadata (Task 13), the lockfile Docker build (Task 14), and the release-automation workflow (Task 15). Task 16 tags `v1.0.0` manually; every merge after that auto-tags. Roadmap context: `docs/quality-roadmap.md`, Release 1.

**Tech Stack:** Python >= 3.10, uv, setuptools + setuptools-scm, pytest, ruff, mypy, GitHub Actions, Docker/GHCR.

## Global Constraints

- Use `uv` for ALL Python work: `uv sync`, `uv run pytest`, `uv run python`. Never `pip install`, `python -m venv`, or activate scripts. (User global CLAUDE.md.)
- Never chain shell commands with `&&` — the permission system blocks chained commands. One command per Bash call. (User global CLAUDE.md.)
- Do all work on branch `release/v1-foundation` off `main`; one commit per task, message prefixes matching repo history (`fix:`, `feat:`, `docs:`, `chore:`, `test:`, `ci:`).
- After ANY edit to `pyproject.toml`, run `uv lock` and include the updated `uv.lock` in that task's commit — the Docker build (Task 14) and CI use `--frozen`.
- The full suite must pass before every commit: `uv run pytest` → expect `1192 passed` (count grows/shrinks slightly as tasks add/remove tests; zero failures always).
- Do not touch the critical invariants in `CLAUDE.md` ("Critical invariants — do not break these"). Nothing in this plan changes pipeline, sidecar, tier, or breaker behavior.
- Do not reformat the codebase wholesale. Ruff runs lint-only in R1 (no `ruff format`).

---

### Task 1: Untrack compiled artifacts and agent state

**Files:**
- Delete from git index (keep nothing): `imageharbor/__pycache__/*.pyc` (11 files)
- Delete from git index (keep on disk): `.superpowers/sdd/progress.md`
- Modify: `.gitignore`

**Interfaces:** none — pure repo hygiene.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b release/v1-foundation
```

- [ ] **Step 2: Untrack the stale .pyc files**

```bash
git rm -r --cached imageharbor/__pycache__
```

Expected: 11 `rm 'imageharbor/__pycache__/...'` lines. (`__pycache__/` is already in `.gitignore:2`; these predate the rule.)

- [ ] **Step 3: Untrack progress.md, keep task reports**

`CLAUDE.md` references `.superpowers/sdd/task-8-report.md`, so reports stay tracked. Only the 59 KB workflow-state file goes:

```bash
git rm --cached .superpowers/sdd/progress.md
```

- [ ] **Step 4: Ignore progress.md going forward**

Append to `.gitignore` (after the `.corpus-tmp/` block):

```gitignore

# Agent workflow state (task reports under .superpowers/sdd/ stay tracked --
# CLAUDE.md cites them; the mutable progress ledger does not belong in git)
.superpowers/sdd/progress.md
```

- [ ] **Step 5: Verify clean status semantics**

```bash
git status --short
```

Expected: the 12 staged deletions, the `.gitignore` modification, nothing else surprising. `imageharbor/__pycache__/` and `.superpowers/sdd/progress.md` must still exist on disk.

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: untrack stale .pyc files and agent progress ledger"
```

---

### Task 2: Evict the personal Takeout zip from the package and the image

**Files:**
- Move: `imageharbor/takeout-20230618T004316Z-001.zip` → `D:\Users\Conrad\Documents\programming\ImageHarbor-local-data\` (outside the repo)
- Modify: `.dockerignore`

**Interfaces:** none. The file is git-ignored (`.gitignore:38`), so this is a filesystem move plus a Docker-context rule; no git change for the zip itself.

- [ ] **Step 1: Move the zip out of the repo**

```powershell
New-Item -ItemType Directory -Force "D:\Users\Conrad\Documents\programming\ImageHarbor-local-data"
Move-Item "D:\Users\Conrad\Documents\programming\ImageHarbor\imageharbor\takeout-20230618T004316Z-001.zip" "D:\Users\Conrad\Documents\programming\ImageHarbor-local-data\"
```

- [ ] **Step 2: Close the Docker-context hole for good**

Append to `.dockerignore`:

```
*.zip
*.tgz
uv.lock.bak
```

(The zip only reached images because `Dockerfile` does `COPY imageharbor ./imageharbor` and `.dockerignore` had no archive rule — a future stray zip must not repeat this.)

- [ ] **Step 3: Run the suite** — the takeout tests synthesize their own zips; nothing references this file.

```bash
uv run pytest tests/test_takeout_ingest.py tests/test_takeout_survey.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add .dockerignore
git commit -m "chore: keep archives out of the Docker build context"
```

---

### Task 3: Fix the stale dashboard port in deploy-docker.md

**Files:**
- Modify: `docs/deploy-docker.md:151`

- [ ] **Step 1: Fix the reference.** Line 151 points the People panel at `http://<docker-host>:8080/`; the published port is 8087 (line 62 of the same file, and commit `c350484`). Change `8080` → `8087` on line 151 only — the `8080` occurrences describing the *in-container* default are correct and stay.

- [ ] **Step 2: Verify no other stale host-port references**

```bash
grep -n "8080" docs/deploy-docker.md
```

Expected: remaining hits all describe the container-internal port/`--dashboard-port` default, not a host URL.

- [ ] **Step 3: Commit**

```bash
git add docs/deploy-docker.md
git commit -m "docs: point the People panel at the published 8087 port"
```

---

### Task 4: Remove dead code

**Files:**
- Modify: `imageharbor/taxonomy.py` (delete `snapshot_text`, lines ~257+)
- Modify: `imageharbor/circuit_breaker.py` (delete `is_half_open`, lines 55–56)
- Modify: `tests/test_taxonomy.py` (delete the tests that exist only to exercise `snapshot_text`, around lines 177 and 208–209)

**Interfaces:** removals only. Explicitly KEEP: `pcs.resolve_code` and `ai_classifier._build_pcs_list` (documented retention — CLAUDE.md calls `resolve_code` "retained for legacy/tooling use"; deferred-issues #7 accepted both).

- [ ] **Step 1: Confirm both are dead**

```bash
grep -rn "snapshot_text" imageharbor tests
grep -rn "is_half_open" imageharbor tests
```

Expected: `snapshot_text` — only its definition and `tests/test_taxonomy.py`; `is_half_open` — only its definition. If ANY other caller appears, stop and report instead of deleting.

- [ ] **Step 2: Delete `CircuitBreaker.is_half_open`** (`circuit_breaker.py:55-56`) — the two-line property and nothing else.

- [ ] **Step 3: Delete `Taxonomy.snapshot_text`** — the whole method. Then delete the `snapshot_text` tests found in Step 1 from `tests/test_taxonomy.py` (the test functions in their entirety, not just assertions).

- [ ] **Step 4: Run the affected suites**

```bash
uv run pytest tests/test_taxonomy.py tests/test_circuit_breaker.py -q
```

Expected: all pass, count reduced by the deleted tests.

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/taxonomy.py imageharbor/circuit_breaker.py tests/test_taxonomy.py
git commit -m "chore: remove dead snapshot_text and is_half_open"
```

---

### Task 5: Dedupe `_now_iso` and `_json_default` into one leaf module

**Files:**
- Create: `imageharbor/util.py`
- Create: `tests/test_util.py`
- Modify: `imageharbor/catalog.py:191-205`, `imageharbor/sidecar.py:27-41`, `imageharbor/faces/store.py:131-132`, `imageharbor/faces/runner.py:33-34`

**Interfaces:**
- Produces: `imageharbor.util.now_iso() -> str` and `imageharbor.util.json_default(o: Any) -> Any`.
- Constraint: tests monkeypatch `catalog._now_iso` (e.g. `tests/test_catalog.py:747-758`) — every consuming module MUST keep a module-level `_now_iso`/`_json_default` name via aliased import so per-module patching keeps working.

The three `_now_iso` bodies and two `_json_default` bodies are byte-identical (verified: `catalog.py:191-205`, `sidecar.py:27-41`, `store.py:131-132`, `runner.py:33-34`).

- [ ] **Step 1: Write the failing test** — `tests/test_util.py`:

```python
"""imageharbor.util is the single home of the tiny helpers that used to be
copy-pasted across catalog.py, sidecar.py, and faces/ with a "keep the two
in sync" comment -- the class of hand-maintained invariant this codebase
otherwise refuses to have."""

from datetime import datetime

from imageharbor.util import json_default, now_iso


def test_now_iso_is_utc_aware_isoformat():
    value = now_iso()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_json_default_decodes_bytes_as_text_not_repr():
    # ExifVersion-style payload: must become "0230", never "b'0230'".
    assert json_default(b"0230") == "0230"


def test_json_default_replaces_undecodable_bytes_rather_than_raising():
    assert json_default(b"\xff\xfe") == "\ufffd\ufffd"


def test_json_default_falls_back_to_str_for_exotic_types():
    assert json_default(3.5) == "3.5"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/test_util.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.util'`.

- [ ] **Step 3: Create `imageharbor/util.py`** (stdlib-only leaf, like `tiers.py`):

```python
"""Tiny stdlib-only helpers shared across the package.

A leaf module (no intra-package imports) so anything -- including
deliberately dependency-light modules like sidecar.py -- can import it
without creating a cycle. Consumers re-export under their old private names
(`from .util import now_iso as _now_iso`) so tests that monkeypatch
`catalog._now_iso` etc. keep working per-module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def json_default(o: Any) -> Any:
    """Fallback for values ``json.dumps`` cannot serialize natively.

    Real EXIF carries raw ``bytes`` (ExifVersion, SceneType, MakerNote) and
    other exotic types. A bare ``default=str`` would not raise, but it writes
    Python repr syntax -- ``"b'0230'"`` rather than ``"0230"`` -- and both the
    catalog's JSON columns and the sidecar are meant to be portable,
    human-readable projections. Bytes become a lossy text form; anything else
    falls back to its string representation.
    """
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).decode("utf-8", "replace")
    return str(o)
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/test_util.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Switch the four consumers to aliased imports.** In each file, delete the local function definition (and, in `sidecar.py`, its "Keep the two in sync" docstring) and add to the imports:

`imageharbor/catalog.py` (replaces lines 191–205):
```python
from .util import json_default as _json_default
from .util import now_iso as _now_iso
```

`imageharbor/sidecar.py` (replaces lines 27–41):
```python
from .util import json_default as _json_default
```

`imageharbor/faces/store.py` (replaces lines 131–132) and `imageharbor/faces/runner.py` (replaces lines 33–34):
```python
from ..util import now_iso as _now_iso
```

Remove any `datetime`/`timezone` imports each file no longer uses (check with `grep -n "datetime" <file>` — `catalog.py` and others may still use them elsewhere; only remove if unused).

- [ ] **Step 6: Full suite** — this is what proves the monkeypatch surface survived:

```bash
uv run pytest -q
```

Expected: 0 failures (in particular `tests/test_catalog.py` and `tests/faces/` all green).

- [ ] **Step 7: Commit**

```bash
git add imageharbor/util.py tests/test_util.py imageharbor/catalog.py imageharbor/sidecar.py imageharbor/faces/store.py imageharbor/faces/runner.py
git commit -m "refactor: single home for now_iso and json_default"
```

---

### Task 6: Ruff — config and a clean lint

**Files:**
- Modify: `pyproject.toml` (add `[tool.ruff]` sections; add `ruff` to the `dev` extra)
- Modify: whatever `ruff check` flags (discovery-based; scope bounded by the rule selection below)

**Interfaces:**
- Produces: `uv run ruff check .` exits 0 — Task 11's CI job depends on this.

- [ ] **Step 1: Add ruff to the dev extra** in `pyproject.toml`:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    # (existing numpy comment and entry stay exactly as they are)
    "numpy>=1.24",
    "ruff>=0.6",
    "mypy>=1.11",
]
```

(mypy added here too so Task 7 doesn't touch this list again.)

- [ ] **Step 2: Add the ruff config** to `pyproject.toml`:

```toml
# Lint only in R1 -- no `ruff format`, no line-length rules: a wholesale
# reformat of 15k LOC is deliberate churn we are not buying in the same
# release that introduces CI. Rules: pyflakes + pycodestyle errors + import
# order + bugbear.
[tool.ruff]
target-version = "py310"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B"]
```

- [ ] **Step 3: Lock and install**

```bash
uv lock
uv sync --extra dev
```

- [ ] **Step 4: Run, autofix, then fix the remainder by hand**

```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff check .
```

For anything `--fix` can't handle: fix real defects properly; for a rule that is wrong for a specific line (e.g. a deliberate `B` pattern with a documented reason), add a targeted `# noqa: <RULE>` with a trailing reason — never a file-level or blanket suppression. If a rule produces >50 findings of one code, stop and add that single code to an `ignore = [...]` list with a comment, rather than churning — report it.

- [ ] **Step 5: Full suite**

```bash
uv run pytest -q
```

Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: adopt ruff (lint-only) and fix what it flags"
```

---

### Task 7: Mypy — config and a clean run

**Files:**
- Modify: `pyproject.toml` (add `[tool.mypy]`)
- Modify: whatever `mypy` flags, including the plain-`sqlite3.Row` params at `backfill.py:68,153`, `pipeline.py:600`, `taxonomy.py:67`, `takeout/ingest.py:804,832,1019` (annotate as `sqlite3.Row`)

**Interfaces:**
- Produces: `uv run mypy imageharbor` exits 0 — Task 11's CI job depends on this. (The faces `catalog/store/detector/embedder` params are Task 8's job, not this one's.)

- [ ] **Step 1: Add the mypy config** to `pyproject.toml`:

```toml
# Non-strict on purpose: the codebase is 99% return-annotated, so this
# baseline is cheap to keep green; ratchet toward strict in later releases.
[tool.mypy]
files = ["imageharbor"]
python_version = "3.10"
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = ["onnxruntime.*", "openai.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Lock, install, run**

```bash
uv lock
uv sync --extra dev
uv run mypy imageharbor
```

- [ ] **Step 3: Fix what it reports.** Annotate the seven `row` params above as `row: sqlite3.Row` (adding `import sqlite3` where missing). For other errors: fix genuinely, or `# type: ignore[<code>]` with a trailing reason for a true false positive. Escape hatch if one module has deep, structural complaints (>25 errors): add a per-module override with an explanatory comment and report it:

```toml
[[tool.mypy.overrides]]
module = "imageharbor.<module>"
ignore_errors = true  # R1 baseline: <one-line reason>; tighten in R5
```

- [ ] **Step 4: Verify clean, then full suite**

```bash
uv run mypy imageharbor
uv run pytest -q
```

Expected: `Success: no issues found`, then 0 test failures.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: adopt mypy baseline and annotate row params"
```

---

### Task 8: Type the faces runner/store seam

**Files:**
- Create: `imageharbor/faces/interfaces.py`
- Modify: `imageharbor/faces/runner.py` (`scan`, `_scan_one`, `build_clusters`, `measure_threshold`, `propagate_sidecars` signatures; the `records: list[tuple]` protocol)
- Modify: `imageharbor/faces/store.py` (`record_scan` signature + unpacking)
- Modify: every test constructing the record tuple or a fake detector/embedder (find via grep; they satisfy the Protocols structurally — no test-double class changes needed)

**Interfaces:**
- Produces: `imageharbor.faces.interfaces.DetectorLike` / `EmbedderLike` (Protocols) and `imageharbor.faces.store.ScannedFace` (frozen dataclass). `FaceStore.record_scan(digest: str, detect_model: str, faces: Sequence[ScannedFace]) -> list[int]`.
- Consumes: `Detection` from its defining faces module (verify with `grep -rn "class Detection" imageharbor/faces` — expected `decode.py`).

- [ ] **Step 1: Create `imageharbor/faces/interfaces.py`:**

```python
"""Structural types for the runner/store seam.

Protocols, not ABCs: the test suite's FakeDetector/FakeEmbedder satisfy
them by shape without importing anything, and detect.py/embed.py (the only
onnxruntime importers) satisfy them without this module ever importing
onnxruntime.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .decode import Detection


class DetectorLike(Protocol):
    model_name: str

    def detect(self, img: Image.Image) -> list[Detection]: ...


class EmbedderLike(Protocol):
    model_name: str
    dim: int

    def embed_batch(self, crops: Sequence[Image.Image]) -> np.ndarray: ...
```

(If `grep` shows `Detection` lives elsewhere, or the real `Detector`/`Embedder` attribute/method names differ from `model_name`/`detect`/`dim`/`embed_batch` as read from `runner.py:100-137`, match the real names — the Protocol mirrors existing code, never renames it.)

- [ ] **Step 2: Write the failing test** — add to `tests/faces/test_store.py`:

```python
def test_record_scan_takes_scanned_face_records(store):
    from imageharbor.faces.store import ScannedFace

    det = _detection()  # reuse this file's existing Detection helper/fixture
    ids = store.record_scan(
        "d" * 43,
        "yunet-test",
        [ScannedFace(detection=det, embedding=None, embed_model=None,
                     reject_reason="low_score")],
    )
    assert len(ids) == 1
```

(Adapt the `_detection()` helper name to whatever this test file already uses to build a `Detection` — read the file first; do not invent a second helper.)

- [ ] **Step 3: Run it to verify it fails**

```bash
uv run pytest tests/faces/test_store.py -q
```

Expected: FAIL — `ImportError: cannot import name 'ScannedFace'`.

- [ ] **Step 4: Add `ScannedFace` to `store.py` and cut `record_scan` over.** Near the top of `imageharbor/faces/store.py`:

```python
@dataclass(frozen=True)
class ScannedFace:
    """One detected face as the scan hands it to the store.

    Replaces the positional (detection, embedding, embed_model,
    reject_reason) tuple -- a 4-slot protocol nobody could read at a call
    site. A rejected face has embedding=None and a reject_reason; a kept
    face has an embedding, its embed_model, and reject_reason=None.
    """

    detection: Detection
    embedding: np.ndarray | None
    embed_model: str | None
    reject_reason: str | None = None
```

Change `record_scan`'s signature (`store.py:174-182`) to `faces: Sequence[ScannedFace]` and replace its tuple unpacking with attribute access (`face.detection`, `face.embedding`, `face.embed_model`, `face.reject_reason`). The SQL and everything below the unpacking stays byte-identical. Drop the old 3-tuple/4-tuple union entirely — one shape, no compatibility layer.

- [ ] **Step 5: Cut `runner.py` over.** The four `records.append((...))` sites (`runner.py:106,108,118,135`) become:

```python
records.append(ScannedFace(det, None, None, "low_score"))
records.append(ScannedFace(det, None, None, "too_small"))
records.append(ScannedFace(det, None, None, "degenerate_landmarks"))
records.append(ScannedFace(det, embedding, embedder.model_name, None))
```

with `records: list[ScannedFace] = []` (was `list[tuple]`, `runner.py:102`) and the import `from .store import FaceStore, ScannedFace`. Annotate the untyped collaborator params using the new Protocols plus the real types:

```python
def scan(
    catalog: Catalog,
    store: FaceStore,
    detector: DetectorLike,
    embedder: EmbedderLike,
    crop_dir: Path,
    *,
    gate: QualityGate,
    limit: int | None = None,
    ...
```

Same pattern for `_scan_one`, and `store: FaceStore` on `build_clusters`, `measure_threshold`, `propagate_sidecars` (`runner.py:185,227,251`). Import `Catalog` under `if TYPE_CHECKING:` if a circular import appears (mirror how `dashboard/server.py:42-51` guards `FaceStore`).

- [ ] **Step 6: Update every remaining tuple call site**

```bash
grep -rn "record_scan(" imageharbor tests
```

Convert each test's tuple literals to `ScannedFace(...)` with the same four values (3-tuples gain `reject_reason=None` implicitly via the default — but only if the third slot was the *model*; read each site, the old 3-tuple form was `(detection, embedding, embed_model)`).

- [ ] **Step 7: Run faces suite, then mypy, then full suite**

```bash
uv run pytest tests/faces -q
uv run mypy imageharbor
uv run pytest -q
```

Expected: all green, mypy clean.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: typed protocols and ScannedFace record for the faces seam"
```

---

### Task 9: Make ResourceWarning a failure and fix the connection leak

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`)
- Modify: whichever test/fixture leaks (the run currently emits `ResourceWarning: unclosed database` attributed to `imageharbor/catalog.py:336` during `tests/test_catalog.py:747`)

**Interfaces:** none new. `Catalog` already has `close()`/`__enter__`/`__exit__` (`catalog.py:1301-1308`) — the fix is using them, not adding them.

- [ ] **Step 1: Turn the warning into an error.** In `pyproject.toml`'s `[tool.pytest.ini_options]`:

```toml
filterwarnings = [
    # A leaked sqlite connection is a latent bug, not a log line. Catalog
    # has close()/context-manager support; every opener must use it.
    "error::ResourceWarning",
]
```

- [ ] **Step 2: Lock, then run the suite to surface every leak**

```bash
uv lock
uv run pytest -q
```

Expected: FAILURES wherever a `Catalog` (or `FaceStore`) is opened and never closed — at minimum around `tests/test_catalog.py:747`.

- [ ] **Step 3: Fix each leak at its source.** For a fixture: `yield` then `close()` (or `with Catalog(...) as catalog: yield catalog`). For an inline open in a test body: wrap in `with`. If a leak traces into *source* code (a code path that opens a `Catalog` and abandons it), fix the source — that is exactly what this filter exists to catch. Do not suppress the filter for a test; the only acceptable per-test `filterwarnings` mark is for a warning a test *deliberately provokes*, with a comment.

Note: `tests/test_takeout_index_equivalence.py` emits a deliberate `UserWarning` in fallback mode — that is not a `ResourceWarning` and is unaffected.

- [ ] **Step 4: Verify green**

```bash
uv run pytest -q
```

Expected: 0 failures, and the `ResourceWarning` no longer appears even as a warning.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: fail on ResourceWarning and close every catalog opener"
```

---

### Task 10: Make the sibling-oracle fallback loud on demand

**Files:**
- Modify: `tests/test_takeout_index_equivalence.py`

**Interfaces:**
- Produces: env contract `IMAGEHARBOR_REQUIRE_SIBLING_ORACLE=1` → the module hard-asserts `INDEX_SOURCE == "sibling_writer"`. Unset → visible skip (never a silent weakening). Task 13's CONTRIBUTING.md documents the variable.

Background: on any machine without `D:\...\Takeout_Inventory\takeout_inventory.py`, the module silently downgrades to `_LITERAL_PAIRS` — a transcription of ImageHarbor's own rules — announced only by a `UserWarning` (`test_takeout_index_equivalence.py:108-118`). The existing warning stays; this adds an enforceable check.

- [ ] **Step 1: Add the test** (after the `INDEX_SOURCE` block, ~line 119):

```python
def test_the_sibling_oracle_actually_loaded_where_required():
    """On a machine that HAS the sibling checkout (Conrad's box; any env
    that sets the flag), the differential tests must run against the real
    oracle -- a silent fallback there means every equivalence test below
    is checking ImageHarbor against itself. Elsewhere this skips VISIBLY,
    which is the honest summary-line for "the oracle is absent"."""
    import os

    if os.environ.get("IMAGEHARBOR_REQUIRE_SIBLING_ORACLE") != "1":
        pytest.skip(
            "IMAGEHARBOR_REQUIRE_SIBLING_ORACLE not set; oracle degradation "
            f"is permitted here (INDEX_SOURCE={INDEX_SOURCE})"
        )
    assert INDEX_SOURCE == "sibling_writer", (
        "IMAGEHARBOR_REQUIRE_SIBLING_ORACLE=1 but the sibling writer did not "
        f"load from {_SIBLING_PATH} -- the differential tests in this module "
        "just ran in near-tautological literal_schema mode. Fix the sibling "
        "checkout (or unset the flag if the machine legitimately lacks it)."
    )
```

Add `import pytest` at module top if not already imported (check — the file's existing tests likely import it).

- [ ] **Step 2: Verify both modes on this machine** (the sibling checkout exists here):

```powershell
uv run pytest tests/test_takeout_index_equivalence.py -q
$env:IMAGEHARBOR_REQUIRE_SIBLING_ORACLE = "1"; uv run pytest tests/test_takeout_index_equivalence.py -q; Remove-Item Env:IMAGEHARBOR_REQUIRE_SIBLING_ORACLE
```

Expected: first run — the new test SKIPS, rest pass; second run — everything passes (oracle loads on this machine). If the second run FAILS the new assert, the sibling checkout has the known import problem described in the module docstring — fix that first; do not weaken the test.

- [ ] **Step 3: Set the flag permanently on this machine** so local runs enforce it (document-only step; the user's shell profile is theirs — note it in the task report for Conrad to add):

```powershell
[Environment]::SetEnvironmentVariable("IMAGEHARBOR_REQUIRE_SIBLING_ORACLE", "1", "User")
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_takeout_index_equivalence.py
git commit -m "test: hard-assert the sibling oracle where it must exist"
```

---

### Task 11: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: a workflow named exactly `CI` — Task 15's `workflow_run` trigger matches on this string. Jobs: `lint` (ruff + mypy), `test` (pytest matrix Windows + Ubuntu). The suite has NEVER run on POSIX — expect and fix Linux failures in this task.

- [ ] **Step 1: Create `.github/workflows/ci.yml`:**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # setuptools-scm (Task 12) derives the version from tags
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.13"
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run mypy imageharbor

  test:
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            python: "3.10"   # requires-python floor
          - os: ubuntu-latest
            python: "3.13"
          - os: windows-latest
            python: "3.13"   # the only platform the suite has run on to date
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python }}
      - run: uv sync --extra dev
      - run: uv run pytest
```

- [ ] **Step 2: Push the branch and open a draft PR** so CI runs:

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint + typecheck + test matrix on Windows and Ubuntu"
git push -u origin release/v1-foundation
gh pr create --draft --title "R1: v1.0.0 foundation release" --body "Implements docs/superpowers/plans/2026-09-12-r1-foundation-release.md (docs/quality-roadmap.md Release 1). CI, tag-driven versioning, lockfile Docker build, cleanups."
```

- [ ] **Step 3: Watch the run**

```bash
gh run watch
```

Expected outcomes and responses:
- `lint` green (Tasks 6–7 made it so locally).
- `windows-latest` green (matches local).
- `ubuntu-latest` MAY fail — first POSIX run ever. Debug each failure with the systematic-debugging skill. Known-risk areas: case-sensitive filesystems, path-separator string assertions in tests, Windows-specific file-locking assumptions (deleting an open SQLite file succeeds on Linux, fails on Windows — a test relying on the *failure* would break), and `test_takeout_index_equivalence` (must show its documented `UserWarning` + 1 skip, not a failure). Fix root causes; never mark tests Windows-only to get green unless the behavior is genuinely platform-defined (then use `pytest.mark.skipif(os.name != "nt", ...)` with a reason string).

- [ ] **Step 4: Iterate until all three matrix legs and lint are green.** Commit fixes as `fix: <what> on POSIX` (separate commits per root cause).

---

### Task 12: Tag-derived versioning (setuptools-scm)

**Files:**
- Modify: `pyproject.toml` (`[build-system]`, `[project]`, new `[tool.setuptools_scm]`)
- Modify: `imageharbor/__init__.py:19`
- Modify (if needed): `tests/test_packaging.py`
- Create: `tests/test_version.py`

**Interfaces:**
- Produces: version derived from git tags matching `v*`; `imageharbor.__version__` reads installed-dist metadata; `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMAGEHARBOR` env var overrides (Task 14's Docker build sets it, because the build context has no `.git`); `fallback_version = "0.0.0"` covers git-less builds with no override (the `tests/test_packaging.py` git-less wheel build).
- `cli.py:12,27` keeps importing `__version__` unchanged.

- [ ] **Step 1: Write the failing test** — `tests/test_version.py`:

```python
"""The version has exactly one source: git tags, via setuptools-scm at
build/install time, surfaced through importlib.metadata. A literal version
string anywhere in the tree is the bug this file exists to prevent -- the
0.1.0 era had two (pyproject.toml and __init__.py) with nothing binding
them."""

import re
from pathlib import Path

import imageharbor


def test_version_is_a_real_pep440_string():
    assert re.match(r"^\d+\.\d+", imageharbor.__version__) or \
        imageharbor.__version__.startswith("0.0.0")


def test_no_hardcoded_version_literals_remain():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'version = "0.1.0"' not in text
    assert 'dynamic = ["version"]' in text
    init = Path(imageharbor.__file__).read_text(encoding="utf-8")
    assert '__version__ = "0' not in init
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_version.py -q
```

Expected: `test_no_hardcoded_version_literals_remain` FAILS (both literals exist today).

- [ ] **Step 3: Rewire `pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools>=77", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "imageharbor"
dynamic = ["version"]
# ... (everything else in [project] unchanged for now; Task 13 adds metadata)

[tool.setuptools_scm]
# A build with no .git and no SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMAGEHARBOR
# (tests/test_packaging.py's git-less wheel; a bare source snapshot) gets a
# recognizable non-version rather than an error.
fallback_version = "0.0.0"
```

Delete the `version = "0.1.0"` line.

- [ ] **Step 4: Rewire `imageharbor/__init__.py:19`** (replace the literal):

```python
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("imageharbor")
except PackageNotFoundError:  # source tree with no installed dist
    __version__ = "0.0.0+uninstalled"
```

- [ ] **Step 5: Lock, reinstall, verify end to end**

```bash
uv lock
uv sync --extra dev
uv run imageharbor --version
uv run pytest tests/test_version.py tests/test_packaging.py -q
```

Expected: `--version` prints a scm-derived dev version (e.g. `0.1.dev348+g<sha>` — no `v*` tag exists yet, so scm counts from zero; correct until Task 16 tags). `test_packaging.py`'s git-less wheel build must pass with the fallback — if it asserts anything about the wheel's version string, update that assertion to expect `0.0.0` with a comment referencing `fallback_version`.

- [ ] **Step 6: Full suite, then commit**

```bash
uv run pytest -q
git add -A
git commit -m "feat: derive the version from git tags via setuptools-scm"
```

---

### Task 13: CHANGELOG, package metadata, README onboarding, CONTRIBUTING

**Files:**
- Create: `CHANGELOG.md`, `CONTRIBUTING.md`
- Modify: `pyproject.toml` (`[project]` metadata), `README.md`

**Interfaces:**
- Produces: the `[minor]` / `[major]` commit-subject tokens (documented here, implemented in Task 15); `CONTRIBUTING.md` documents `IMAGEHARBOR_REQUIRE_SIBLING_ORACLE` (Task 10).

- [ ] **Step 1: Fill `[project]` metadata** in `pyproject.toml` (setuptools>=77 from Task 12 supports the SPDX string form):

```toml
license = "AGPL-3.0-or-later"
license-files = ["LICENSE"]
authors = [{ name = "Conrad Storz", email = "conradstorz@gmail.com" }]
keywords = ["photos", "photo-organization", "exif", "sha256", "google-takeout", "deduplication"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Multimedia :: Graphics",
]

[project.urls]
Homepage = "https://github.com/conradstorz/ImageHarbor"
Changelog = "https://github.com/conradstorz/ImageHarbor/blob/main/CHANGELOG.md"
```

Remove the old `license = { text = "AGPL-3.0-or-later" }` table form. (Verify the repo slug with `git remote get-url origin` and use the real one.)

- [ ] **Step 2: Create `CHANGELOG.md`:**

```markdown
# Changelog

Notable changes, [Keep a Changelog](https://keepachangelog.com/) style.
Versioning: every merge to `main` is released — patch bump by default; a
`[minor]` or `[major]` token in the merge-commit subject overrides. Tags are
the single source of version truth (`setuptools-scm`); per-release notes are
generated on each GitHub Release. This file curates the human summary for
notable releases only — not every patch.

## [1.0.0] — 2026-09-12

First official release. The tool has been in daily production use since
2026-08; this release establishes the engineering floor around it:

- CI (ruff, mypy, pytest on Windows + Ubuntu) as the merge gate.
- Tag-derived versioning; automatic release on every merge to `main`.
- Docker image built from `uv.lock` (ships exactly what was tested), with
  the AGPL licence text included.
- Codebase cleanups from the 2026-09-12 quality review (`docs/quality-roadmap.md`).
```

(Set the date to the actual tag date at Task 16 if it differs.)

- [ ] **Step 3: Add onboarding to `README.md`** — insert after the `## Main commands` section (line ~127; keep all existing content):

```markdown
## Install

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/conradstorz/ImageHarbor
cd ImageHarbor
uv sync --extra dev                      # core + test tooling
uv sync --extra dev --extra openai --extra faces   # + AI classifier + face recognition
uv run imageharbor --help
```

## Running the tests

```bash
uv run pytest            # full suite, no network, no AI backend needed
uv run ruff check .      # lint
uv run mypy imageharbor  # types
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the CI gate and release process.
```

Also add the missing rows to the `## Main commands` table: `catalog list` / `catalog get` (`uv run imageharbor catalog list --catalog DEST/catalog.db`) and `faces models download` — copy wording from `CLAUDE.md`'s command table.

- [ ] **Step 4: Create `CONTRIBUTING.md`:**

```markdown
# Contributing

## Environment

`uv` only — no pip, no manual venvs. `uv sync --extra dev` installs
everything the suite needs (`--extra openai --extra faces` for the optional
backends). Run things with `uv run <cmd>`.

## The gate

Every PR and every push to `main` runs CI: `ruff check .`,
`mypy imageharbor`, and the full pytest suite on Windows and Ubuntu. Green
CI is the merge condition; there is no other gate.

## Releases and versioning

Every merge to `main` with green CI is automatically tagged and released —
patch bump by default. Put `[minor]` or `[major]` in the merge-commit
subject to bump those instead. Never hand-edit a version string; tags are
the only source (`setuptools-scm`).

## Test conventions

- Tests are deterministic: no network, no sleeps, injected clocks, seeded
  randomness. Keep it that way.
- `-m corpus` tests run against a real Takeout export on disk — opt-in,
  never in CI.
- `tests/test_takeout_index_equivalence.py` checks against the sibling
  Takeout_Inventory repo when present. On a machine that has that checkout,
  set `IMAGEHARBOR_REQUIRE_SIBLING_ORACLE=1` so a silent fallback to the
  weaker literal-schema mode fails loudly instead.
- `ResourceWarning` is an error: close every `Catalog`/`FaceStore` you open
  (both are context managers).

## Invariants

Read `CLAUDE.md`'s "Critical invariants — do not break these" before
touching the pipeline, sidecar merge, tier system, or circuit breaker.
```

- [ ] **Step 5: Lock, full suite, commit**

```bash
uv lock
uv run pytest -q
git add pyproject.toml uv.lock CHANGELOG.md CONTRIBUTING.md README.md
git commit -m "docs: onboarding, changelog, contributing, package metadata"
```

---

### Task 14: Docker builds from the lockfile and carries the licence

**Files:**
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMAGEHARBOR` (Task 12) via the `IMAGEHARBOR_VERSION` build arg — the Docker context has no `.git`, so without it the image version would be the `0.0.0` fallback.
- Produces: the `IMAGEHARBOR_VERSION` build-arg contract Task 15's release workflow fills with the tag.

- [ ] **Step 1: Rewrite the install section of `Dockerfile`.** Keep lines 1–7 (base image, `useradd`, `WORKDIR`) and everything from `ENV IMAGEHARBOR_SOURCE=...` (line 20) down unchanged. Replace lines 9–17 (the comment + `COPY` + `pip install`) with:

```dockerfile
# Build from the lockfile so the container ships exactly the dependency set
# the suite was tested against -- `pip install ".[...]"` resolved fresh
# against loose floors on every build. uv is copied from its official image
# (pinned); --frozen refuses a stale lock instead of silently re-resolving.
# 'faces' adds ~261 MB of model weights on the FIRST `faces scan`/`watch`
# run (see imageharbor/faces/download.py) -- not at build time, so this
# layer stays small; docker-compose.yml's `imageharbor-models` volume stops
# that download from repeating on every container recreate.
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# The build context has no .git, so setuptools-scm cannot derive a version;
# the release workflow passes the tag here. A local `docker build` without
# the arg gets 0.0.0 -- visibly a non-release.
ARG IMAGEHARBOR_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMAGEHARBOR=${IMAGEHARBOR_VERSION} \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# LICENSE ships in the image: this is an AGPL network service; the conveyed
# artifact must carry the licence text (README.md "Licence" section).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY imageharbor ./imageharbor
RUN uv sync --frozen --no-dev --no-editable --extra openai --extra faces
ENV PATH="/opt/venv/bin:${PATH}"
```

- [ ] **Step 2: Build locally to prove it**

```bash
docker build -t imageharbor:lockfile-test --build-arg IMAGEHARBOR_VERSION=1.0.0-test .
```

Expected: build succeeds; the `uv sync --frozen` layer must NOT print any "resolving" against the network beyond fetching the locked wheels. If `--frozen` fails with a lock/manifest mismatch, run `uv lock` locally, commit the lock, rebuild — do not drop `--frozen`.

- [ ] **Step 3: Smoke-test the image**

```bash
docker run --rm imageharbor:lockfile-test --version
docker run --rm --entrypoint ls imageharbor:lockfile-test /app/LICENSE
```

Expected: `imageharbor, version 1.0.0-test` (or `1.0.0.test0` after PEP 440 normalization — either is fine); `/app/LICENSE` listed.

- [ ] **Step 4: Verify the personal-zip class of leak is closed**

```bash
docker run --rm --entrypoint find imageharbor:lockfile-test /app -name "*.zip"
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "feat(deploy): build the image from uv.lock and ship the licence"
```

---

### Task 15: Release automation — auto-tag, GitHub Release, GHCR push

**Files:**
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: the `CI` workflow name (Task 11), the `IMAGEHARBOR_VERSION` build arg (Task 14), the `[minor]`/`[major]` tokens (Task 13).
- Design constraint: tag + Docker push live in ONE workflow because tags created with the default `GITHUB_TOKEN` do not trigger other workflows — a separate on-tag docker workflow would silently never run.
- Guard: if no `v*` tag exists yet, the workflow no-ops — automation begins only after Task 16's manual `v1.0.0`.

- [ ] **Step 1: Confirm the GHCR image path.** Run `git remote get-url origin`; the image is `ghcr.io/<owner-lowercase>/imageharbor`. The YAML below assumes `conradstorz` — substitute the real owner if it differs.

- [ ] **Step 2: Create `.github/workflows/release.yml`:**

```yaml
name: Release

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [main]

permissions:
  contents: write
  packages: write

concurrency:
  group: release
  cancel-in-progress: false

jobs:
  tag:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.bump.outputs.tag }}
      version: ${{ steps.bump.outputs.version }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ github.event.workflow_run.head_sha }}
      - id: bump
        shell: bash
        run: |
          last=$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)
          if [ -z "$last" ]; then
            echo "No v* tag yet; releases begin after v1.0.0 is tagged manually."
            exit 0
          fi
          if [ -n "$(git tag --points-at HEAD --list 'v*')" ]; then
            echo "HEAD is already tagged; nothing to do."
            exit 0
          fi
          subject=$(git log -1 --pretty=%s)
          IFS=. read -r maj min pat <<<"${last#v}"
          case "$subject" in
            *"[major]"*) tag="v$((maj+1)).0.0" ;;
            *"[minor]"*) tag="v${maj}.$((min+1)).0" ;;
            *)           tag="v${maj}.${min}.$((pat+1))" ;;
          esac
          echo "tag=${tag}"           >> "$GITHUB_OUTPUT"
          echo "version=${tag#v}"     >> "$GITHUB_OUTPUT"
      - if: steps.bump.outputs.tag != ''
        env:
          GH_TOKEN: ${{ github.token }}
        run: >
          gh release create "${{ steps.bump.outputs.tag }}"
          --generate-notes
          --target "${{ github.event.workflow_run.head_sha }}"

  docker:
    needs: tag
    if: ${{ needs.tag.outputs.tag != '' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ needs.tag.outputs.tag }}
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ github.token }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          build-args: |
            IMAGEHARBOR_VERSION=${{ needs.tag.outputs.version }}
          tags: |
            ghcr.io/conradstorz/imageharbor:${{ needs.tag.outputs.tag }}
            ghcr.io/conradstorz/imageharbor:latest
```

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/release.yml
git commit -m "ci: auto-tag, release, and publish the image on every green merge"
git push
```

- [ ] **Step 4: Verify the no-tag guard.** After CI on this push completes, the Release workflow fires on `main`? No — the branch filter is `main`, so it does NOT run for this branch push. Nothing to observe yet; the guard is exercised in Task 16. Confirm only that the workflow file parses (the Actions tab lists "Release" without a syntax error).

---

### Task 16: Ship it — merge, tag v1.0.0, prove the automation

This task is interactive with GitHub and ends the release. It is the exit-criteria check from `docs/quality-roadmap.md` R1.

- [ ] **Step 1: Final local verification (verification-before-completion)**

```bash
uv run ruff check .
uv run mypy imageharbor
uv run pytest
```

All green, with output shown in the task report.

- [ ] **Step 2: Mark the PR ready and get it green**

```bash
gh pr ready
gh run watch
```

All CI legs green on the PR.

- [ ] **Step 3: Merge** (merge commit, subject WITHOUT `[minor]`/`[major]` — the first tag is manual):

```bash
gh pr merge --merge
```

Wait for CI on `main` to complete. The Release workflow will run and must log "No v* tag yet" and exit without tagging — verify in the Actions log (this proves the guard).

- [ ] **Step 4: Tag v1.0.0 manually** on the merge commit:

```bash
git checkout main
git pull
gh release create v1.0.0 --generate-notes --title "v1.0.0 — Foundation" --target main
```

Note: this tag was NOT created by the workflow, so it also does not trigger the docker job. Push the first image by hand once:

```bash
git pull --tags
docker build -t ghcr.io/conradstorz/imageharbor:v1.0.0 --build-arg IMAGEHARBOR_VERSION=1.0.0 .
docker tag ghcr.io/conradstorz/imageharbor:v1.0.0 ghcr.io/conradstorz/imageharbor:latest
docker push ghcr.io/conradstorz/imageharbor:v1.0.0
docker push ghcr.io/conradstorz/imageharbor:latest
```

(Requires `docker login ghcr.io` with a PAT that has `write:packages` — ask Conrad to run the login if credentials are not already present.)

- [ ] **Step 5: Prove the exit criterion — the NEXT merge auto-tags v1.0.1.** The roadmap's R1 exit criterion is automation with no human action. Verify it with the next real merge to `main` (any small change — R2's first PR qualifies): after its CI goes green, the Release workflow must create `v1.0.1` + a GitHub Release + push `ghcr.io/conradstorz/imageharbor:v1.0.1` and `:latest`, unattended. Record the run URL in the task report. If a suitable merge isn't imminent, a one-line docs PR (e.g. updating `CHANGELOG.md`'s 1.0.0 date) is an honest test vehicle.

- [ ] **Step 6: Update `uv run imageharbor --version` sanity**

```bash
uv sync --extra dev
uv run imageharbor --version
```

Expected: `imageharbor, version 1.0.0` (exact tag; scm now has a real tag to derive from).

---

## Self-Review Notes

- **Roadmap R1 coverage:** 1a cleanups → Tasks 1–5; 1b lint/type → Tasks 6–8; 1c CI (incl. ResourceWarning filter and oracle flag) → Tasks 9–11; 1d versioning/release → Tasks 12, 15, 16; 1e Docker CD → Tasks 14–15; 1f onboarding → Task 13. No R1 roadmap item is unassigned.
- **Discovery-bounded steps** (ruff/mypy fixes, POSIX CI failures, tuple call-site conversion) carry explicit search commands, decision rules, and escape hatches instead of pretending the findings are known in advance.
- **Known interactions handled:** setuptools-scm × git-less wheel test (`fallback_version`, Task 12); setuptools-scm × Docker context (`PRETEND_VERSION`, Task 14); `GITHUB_TOKEN` tags not triggering workflows (single Release workflow, Task 15); monkeypatched `_now_iso` (aliased imports, Task 5); `workflow_run` name coupling (Tasks 11↔15).
