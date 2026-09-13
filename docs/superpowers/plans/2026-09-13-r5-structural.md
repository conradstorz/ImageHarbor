# R5 "Structural Debt" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pay down the quality review's structural findings — the `Catalog` god object, the eight dashboard `_conn` reach-ins, the three oversized functions, `projections.py`'s absent/unreadable/zero conflation (deferred #10), the `click` import in `faces/runner.py`, comment archaeology — plus the dashboard's AI/IO failure-split display (deferred from R3) and small carried minors. Ships as v1.3.0.

**Architecture:** Refactors are behavior-preserving with the 1267-test suite as the net; each task runs the full suite plus its domain's focused files. The `Catalog` split is by COMPOSITION on the same connection and lock (no second connection — that decision is documented in catalog.py and stands): sub-stores hold references to the catalog's `_conn`/`lock`, `Catalog` exposes them as attributes, and every call site migrates in the same task — no deprecated delegating shims. The dashboard gets a guarded read-only query method instead of eight bespoke aggregate wrappers. Order puts the two store extractions first (they simplify everything after), the UI change second-to-last, ship last.

**Tech Stack:** Python, sqlite3, pytest; no new dependencies.

## Global Constraints

- Use `uv` for ALL Python work; never pip. Never chain shell commands with `&&` — one command per Bash call. (User global CLAUDE.md.)
- Branch: `release/v5-structural` off current `main`; one commit per task.
- Before every commit: `uv run pytest -q` (baseline 1267 passed / 20 skipped, 0 failures), `uv run ruff check .`, `uv run mypy imageharbor` — all clean.
- **Single-connection + single-lock invariant holds everywhere:** every sub-store and query method uses the catalog's (or FaceStore's) existing `_conn` under its existing `lock`. No new `sqlite3.connect` anywhere.
- **Behavior-preserving except** Task 7's sanctioned micro-changes and Task 8's UI addition. SQL statements move verbatim unless a task says otherwise. SCHEMA_VERSION stays `"2"`; no DDL changes.
- CLAUDE.md is updated IN THE SAME COMMIT as any task that changes a documented module boundary or removes a documented method (the R1-R3 lesson: no window where code and contract disagree).
- CLAUDE.md critical invariants untouched: tier gating, sidecar merge, breaker scoping, copy→fsync→verify→catalog, faces-never-rename, name-identity-exact.

---

### Task 1: Extract `TakeoutStore`

**Files:**
- Create: `imageharbor/takeout/store.py`, `tests/test_takeout_store.py`
- Modify: `imageharbor/catalog.py` (remove the `takeout_*` methods and construct the store), every `catalog.takeout_*` call site (`grep -rn "\.takeout_" imageharbor tests` — expect takeout/ingest.py, takeout/survey.py, cli.py's `takeout status`, dashboard/stats.py's queue section if it uses them, and their tests), `CLAUDE.md` (catalog.py + takeout/ bullets)

**Interfaces:**
- Produces: `class TakeoutStore` in `imageharbor/takeout/store.py`, constructed by `Catalog.__init__` as `self.takeout = TakeoutStore(conn=self._conn, lock=self.lock)` (keyword-only). Methods are the current `takeout_*` catalog methods with the prefix dropped: `archive_get/archive_upsert/... member_add/member_set/member_get/... status_counts` — enumerate the real set with `grep -n "def takeout_" imageharbor/catalog.py` and map 1:1. SQL and docstrings (including `takeout_member_set`'s blind-overwrite landmine docstring) move VERBATIM. Every method takes the lock exactly as before.
- Call sites become `catalog.takeout.member_add(...)` etc. NO delegating `takeout_*` methods remain on `Catalog` (clean cut; grep proves zero remaining).

- [ ] **Step 1: Branch setup** — `git checkout main`; `git pull`; `git checkout -b release/v5-structural`; commit the plan doc (`docs: R5 structural implementation plan`).
- [ ] **Step 2: Enumerate** — the `takeout_*` method list and every call site (both greps above); record the table in your report.
- [ ] **Step 3: Failing test first** — `tests/test_takeout_store.py`:

```python
def test_catalog_composes_a_takeout_store_on_the_same_connection(catalog):
    from imageharbor.takeout.store import TakeoutStore
    assert isinstance(catalog.takeout, TakeoutStore)
    # same connection + same lock object -- the single-writer invariant:
    assert catalog.takeout._conn is catalog._conn
    assert catalog.takeout.lock is catalog.lock

def test_no_takeout_methods_remain_on_catalog(catalog):
    assert not [m for m in dir(catalog) if m.startswith("takeout_")]
```

- [ ] **Step 4: Extract.** Move methods verbatim (prefix dropped), wire construction, migrate every call site mechanically. Do NOT reorder or "improve" SQL.
- [ ] **Step 5: Migrate the takeout tests** that called `catalog.takeout_*` (rename calls only; assertions untouched).
- [ ] **Step 6: CLAUDE.md** — catalog.py bullet: takeout tables now owned by `takeout/store.py`'s `TakeoutStore`, composed on the same conn/lock (`catalog.takeout`); add a sentence to the takeout/ section listing store.py as the eighth module.
- [ ] **Step 7: Verify + commit** — focused: `uv run pytest tests/test_takeout_store.py tests/test_takeout_ingest.py tests/test_takeout_survey.py tests/test_catalog.py -q`; then full suite/ruff/mypy; `git commit -m "refactor(catalog): extract TakeoutStore onto the shared connection"`.

---

### Task 2: Extract `TaxonomyStore` into `taxonomy.py`

**Files:**
- Modify: `imageharbor/taxonomy.py` (gains `class TaxonomyStore`; `Taxonomy` consumes it), `imageharbor/catalog.py` (remove `taxonomy_*` methods; construct `self.taxonomy_store = TaxonomyStore(conn=..., lock=...)`), call sites (`grep -rn "\.taxonomy_" imageharbor tests` — expect taxonomy.py itself, possibly enrich.py/cli.py/dashboard, and tests), `CLAUDE.md`
- Test: `tests/test_taxonomy.py` (+ a composition test mirroring Task 1's)

**Interfaces:**
- Produces: `TaxonomyStore` co-located in `taxonomy.py` (its only real consumer is `Taxonomy` — co-location beats a new module). Methods = current `taxonomy_*` catalog methods, prefix dropped, SQL verbatim, lock discipline identical. `Taxonomy.__init__` signature is unchanged externally (still takes the catalog; internally it uses `catalog.taxonomy_store`). Import-cycle note: `catalog.py` must NOT import `taxonomy.py` at module top if that creates a cycle (taxonomy.py imports from catalog? check) — if a cycle appears, construct the store lazily in `Catalog.__init__` via a local import, with a comment; if no cycle, top-level import is fine.
- Same clean-cut rule: zero `taxonomy_*` methods remain on `Catalog`.

Steps mirror Task 1 exactly (enumerate → failing composition test → extract verbatim → migrate call sites and tests → CLAUDE.md (catalog.py + taxonomy.py bullets) → focused suites (`tests/test_taxonomy.py tests/test_enrich.py tests/test_catalog.py`) → full verify → commit `refactor(catalog): extract TaxonomyStore, co-located with its consumer`).

---

### Task 3: Guarded read API; retire the dashboard `_conn` reach-ins

**Files:**
- Modify: `imageharbor/catalog.py` + `imageharbor/faces/store.py` (each gains `run_select`), `imageharbor/dashboard/stats.py` (3 reach-ins), `imageharbor/dashboard/people.py` (5 reach-ins), `CLAUDE.md`
- Test: `tests/test_catalog.py`, `tests/faces/test_store.py`

**Interfaces:**
- Produces, on BOTH stores:

```python
def run_select(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    """Execute a read-only SELECT under the store's lock.

    The dashboard's aggregate queries are bespoke enough that wrapping each
    as a named method would just re-bloat the store; this is the sanctioned
    read-side door. SELECT-only is enforced, not assumed.
    """
    if not sql.lstrip().upper().startswith("SELECT"):
        raise ValueError("run_select only runs SELECT statements")
    with self.lock:
        return self._conn.execute(sql, params).fetchall()
```

- Consumes: every `catalog._conn.execute(...)`/`store._conn.execute(...)` in `dashboard/` becomes `catalog.run_select(...)`/`store.run_select(...)` with the SAME SQL; the now-redundant explicit `with catalog.lock:` wrappers around those blocks are removed (the method takes the lock). Multi-statement blocks that interleave several SELECTs under one lock hold: splitting into per-query lock acquisitions is acceptable ONLY if no block depends on cross-query snapshot consistency — read each block and judge; where a block genuinely needs one consistent snapshot, keep a `with catalog.lock:` around several `run_select` calls (RLock reentrancy makes that safe) and say so in a comment.
- The R4 stats-lock mutation-guard test must still pass unmodified (it asserts behavior, not implementation).
- Grep exit criterion: `grep -rn "_conn" imageharbor/dashboard/` → zero hits.

Steps: failing tests for `run_select` (returns rows; rejects `UPDATE ...` with ValueError; works reentrantly inside `with catalog.lock:`) → implement on both stores → migrate the 8 reach-ins → CLAUDE.md (catalog.py `Catalog.lock` paragraph's mention of stats.py's direct-`_conn` sections gets updated to name `run_select`) → focused (`tests/test_dashboard_stats.py tests/test_dashboard_people.py tests/test_dashboard_people_http.py tests/test_catalog.py tests/faces/test_store.py`) → full verify → commit `refactor(dashboard): guarded run_select retires the _conn reach-ins`.

---

### Task 4: Decompose `_survey` (takeout/ingest.py)

**Files:**
- Modify: `imageharbor/takeout/ingest.py`
- Test: existing suites only (behavior-preserving; no new tests required, no test edits allowed)

**Interfaces:**
- Produces: `_survey` reduced to a readable orchestrator calling extracted private methods on `_Ingestor`. Mandatory extractions (names indicative): `_survey_pass_one_recorded_members(...)` (the complete-archives-from-catalog pass), `_survey_enumerate_archive(...)` (per-archive identity/corrupt/enumeration handling), `_survey_second_pass_late_sidecars(...)` (the `# SECOND PASS` block — the plan's original tell). Each helper gets a docstring stating its contract; `self.stats` mutation stays exactly where it is today (the "PURE: never touches self.stats" annotations must remain true — do not move counting across helper boundaries).
- Behavior-identical: the whole-batch pairing index construction order, `pending`-reset semantics, and error handling move verbatim inside helpers.

Steps: read `_survey` end-to-end → extract mechanically (cut/paste bodies, thread parameters/return values; no logic edits) → focused `uv run pytest tests/test_takeout_survey.py tests/test_takeout_ingest.py tests/test_takeout_index_equivalence.py -q` → full verify → commit `refactor(takeout): decompose _survey into named passes`.

---

### Task 5: Decompose `watch()` and `enrich_library()`

**Files:**
- Modify: `imageharbor/watcher.py`, `imageharbor/enrich.py`, `CLAUDE.md` (only if a documented seam moves — the faces-pass description references watch()'s inline block)
- Test: existing suites only; no test edits allowed

**Interfaces:**
- `watcher.py`: the inline faces pass (~100 lines with its own try/except and two warning latches) becomes `_run_faces_pass(...)` — the warning-latch state (log-once-for-the-life-of-the-run flags) must survive extraction across cycles, so latches stay owned by `watch()`'s scope and pass in/out explicitly (or live on a small `_FacesPassState` dataclass created before the loop). The never-references-breaker property of the faces block must remain grep-verifiable (`grep -n breaker` inside the new function → zero hits).
- `enrich.py`: `enrich_library`'s loop body decomposes into `_describe_row(...)` (perception + breaker feeding, both failure blocks) and `_apply_enrichment(...)` (the LOCAL block: class resolution incl. fallback+success recording, catalog write, tier-gated rename, sidecar merge). The two-tier try/except semantics from R3/R4 (per-row isolation, break-vs-continue, record_success placement) move VERBATIM — the extracted functions return explicit outcomes (`continue`/`break` decisions become return values the loop acts on; e.g. an enum or `("aborted" | "failed" | "ok")`).
- The R3/R4 behavioral pins (`test_record_success_fires_before_remember`, per-row isolation tests, poison suite) are the safety net — they must pass unmodified.

Steps: read both functions in full → extract watcher first, run `tests/test_watcher.py tests/faces/test_watch_faces.py tests/test_poison.py -q` → extract enrich, run `tests/test_enrich.py tests/test_monotonicity.py tests/test_poison.py -q` → CLAUDE.md if needed → full verify → commit `refactor: name the watch faces pass and the enrich row phases`.

---

### Task 6: `projections.py` `Unreadable` sentinel (deferred #10)

**Files:**
- Modify: `imageharbor/dashboard/projections.py`, `imageharbor/dashboard/stats.py` (the `backlog` hand-off), `CLAUDE.md` (the "conflates readability with meaning — known, not-yet-fixed gap" paragraph becomes a "fixed in R5" note or is deleted in favor of describing the sentinel)
- Test: `tests/test_dashboard_projections.py`

**Interfaces:**
- Produces: a module-level sentinel:

```python
class _UnreadableType:
    """A parse site read SOMETHING and could not interpret it -- distinct
    from None (absent) and 0 (genuinely zero). Deferred issue #10: eight
    defects in three review rounds were all one of these three meanings
    standing in for another."""
    __slots__ = ()
    def __repr__(self) -> str: return "UNREADABLE"

UNREADABLE = _UnreadableType()
```

Every internal parse helper (`_parse_iso`-equivalents, duration/rate computations) returns `T | None | _UnreadableType` where the three meanings genuinely differ, and `project()`'s decision points branch on all three explicitly. `stats.py`'s `backlog` parameter becomes `int | None` (not `Any` — R1 review Minor) and passes through honestly.
- **External behavior:** statuses/ETAs for well-formed inputs are byte-identical (the existing test file is the oracle — it must pass unmodified EXCEPT tests whose very subject was the conflation; each such change must be listed in the report with the old-vs-new meaning). New tests: for at least three previously-conflated sites, a triple (`absent`, `unreadable`, `zero`) test showing the three inputs now take distinct paths (unreadable → `unknown` with a reason; zero → a real computation; absent → whatever the field's documented absence semantics are).

Steps: read projections.py + its test file fully → introduce the sentinel and thread it through one parse site at a time, running the projection tests after each → stats.py `int | None` → new triple tests → CLAUDE.md → full verify → commit `refactor(projections): UNREADABLE sentinel ends the absent/unreadable/zero conflation`.

---

### Task 7: Small cleanups — click, record_success seam, annotations, archaeology

**Files:**
- Modify: `imageharbor/faces/runner.py` + `imageharbor/cli.py`; `imageharbor/enrich.py`; `tests/conftest.py`; `imageharbor/watcher.py` + `imageharbor/catalog.py` + `imageharbor/dashboard/stats.py` (comment trims); `CLAUDE.md` if it cites any trimmed text
- Test: `tests/faces/test_runner.py`, `tests/test_enrich.py`

**Interfaces & items:**
1. **click out of faces/runner.py:** define `class ModelsUnavailableError(RuntimeError)` in the faces package (wherever the `click.ClickException` is raised today — find it: `grep -n click imageharbor/faces/runner.py`); runner raises it; `cli.py` catches and re-raises as `click.ClickException` with the same message. Test: the runner-level test asserts the domain exception; a cli-level test (if one covered the old path) asserts the message survived.
2. **record_success seam (R4 review Minor #6):** hoist the duplicated `breaker.record_success()` to ONE call site after the class-resolution if/else, moving `concept_map.remember` BELOW it — preserving the R4-pinned ordering (success before remember) and the R3-pinned failure paths. This interacts with Task 5's extraction — do Task 7 AFTER Task 5 and apply the hoist inside the extracted `_apply_enrichment` (or wherever the two call sites landed). `test_record_success_fires_before_remember` and the poison suite must pass unmodified.
3. **conftest annotations (R4 review Minor #5):** generator fixtures get `Iterator[...]` return annotations (`from collections.abc import Iterator`).
4. **Comment archaeology:** rewrite the dated review-narrative comments at `watcher.py` (~511-524, "IMPORTANT finding #7 (2026-08-19 whole-branch review)..."), `catalog.py` (~225-238, "CRITICAL finding #2..."), and `stats.py` (~355) into present-tense statements of the CONSTRAINT each protects (what must stay true and why), dropping the review-round narration. The compose file's finding #4 comment is load-bearing ops context — do NOT touch it. If CLAUDE.md quotes any trimmed text, update it in step.

Steps: items in order 1→4, focused tests after each → full verify → commit `chore: domain exception for faces CLI, single success seam, comment archaeology trim`.

---

### Task 8: Dashboard shows the AI/IO failure split

**Files:**
- Modify: `imageharbor/dashboard/index.html`
- Test: `tests/test_dashboard_server.py` only if it asserts page content (check); otherwise manual curl verification

**Interfaces:**
- Consumes: `/api/stats` history rows and window summaries already carry `enrich_ai_failed`/`enrich_io_failed` (R3 Task 6). Render them: in the pass-history table, the enrich-failed cell becomes `N (ai A / io B)` when either is nonzero; in the 24h/30d summary, add the same split in parentheses. Degrade gracefully when the keys are absent/zero (pre-upgrade rows): show the plain total exactly as today.
- Style: match the page's existing table/JS conventions (it renders named keys — extend the render functions, no framework).

Steps: read the render functions for the history table and window summaries → implement → verify with a live scratch watcher + curl (`/api/stats` fields present) and a browser-less DOM sanity grep (the new labels appear in the served page) → full suite (unchanged expectations) → commit `feat(dashboard): surface the AI vs I/O enrichment failure split`.

---

### Task 9: Ship v1.3.0 and deploy

- [ ] Final local verification (ruff, mypy, full pytest — tail in report).
- [ ] Push; `gh pr create --title "R5: structural debt paydown" --body "Implements docs/superpowers/plans/2026-09-13-r5-structural.md (quality-roadmap Release 5)."`; CI (5 required legs) + Faces ONNX workflow both green.
- [ ] Merge: `gh pr merge <PR#> --merge --subject "Merge R5 structural debt paydown [minor]"` → automation cuts v1.3.0 + Release + GHCR. Record run URLs/conclusions.
- [ ] Deploy hpz440 (proven path; compose pull/up; healthz on-host + from here; logs clean; digest match). Browse-level check: the dashboard page renders and the history table shows the new split formatting (curl the page, grep the new label).
- [ ] Report: including a before/after `wc -l` for catalog.py and the three decomposed functions' new sizes, and the final `grep -rn "_conn" imageharbor/dashboard/` (empty) as the reach-in exit criterion.

---

## Self-Review Notes

- Roadmap R5 coverage: Catalog split (T1-T2, takeout+taxonomy per the roadmap's "first" scoping; runs/settings deliberately stay — they're small and load-bearing for the dashboard), read-side API (T3), three oversized functions (T4-T5), click removal (T7.1), Unreadable sentinel (T6), archaeology trim (T7.4). Deferred-from-earlier: failure-split UI (T8), record_success seam (T7.2), conftest annotations (T7.3).
- Order rationale: T5 before T7 (the seam hoist lands in the extracted function); T1-T3 before T4-T5 (smaller call-site churn once stores exist); T8 late (UI depends on nothing else); ship last.
- The clean-cut (no delegating shims) decision trades a bigger diff for zero API ambiguity; the suite plus the composition tests (same conn, same lock) are the guard.
- T3's snapshot-consistency judgment call is explicitly flagged rather than assumed — the reviewer should check each converted block's lock granularity.
