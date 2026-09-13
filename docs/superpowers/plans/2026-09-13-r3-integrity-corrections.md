# R3 "Integrity & Failure-Handling Corrections" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the quality review's integrity findings — the `pick_class` breaker leak, the missing fsync before verification, model-download wedging, staging debris — plus two deferred-known-issues items (#9 sidecar descriptor staleness, #13 `enrich_failed` conflation), shipping as v1.2.0.

**Architecture:** Six independent corrections, each with its own tests, ordered so the breaker-semantics change (which edits a documented invariant) lands first with its CLAUDE.md update in the same commit. The ship task merges with `[minor]` and deploys to hpz440 via the GHCR pull path R2 established.

**Tech Stack:** Python stdlib (os.fsync, shutil), pytest, existing enrich/pipeline/catalog seams.

## Global Constraints

- Use `uv` for ALL Python work; never pip. Never chain shell commands with `&&` — one command per Bash call. (User global CLAUDE.md.)
- Branch: `release/v3-integrity` off current `main`; one commit per task; repo-style commit prefixes.
- Before every commit: `uv run pytest -q` (baseline 1237 passed / 19-or-20 skipped, 0 failures), `uv run ruff check .`, `uv run mypy imageharbor` — all clean.
- `SCHEMA_VERSION` stays `"2"`; any new `runs` columns are additive with defaults, upgrading an existing catalog in place (hpz440's live catalog must open unchanged).
- Sidecar changes go through `sidecar_schema.merge` only; a repeated run must stay byte-identical (idempotence invariant). No new annotation keys.
- The tier system is untouched: no new rename/move paths; `tiers.is_upgrade` still gates everything it gated.

---

### Task 1: `pick_class` failures are AI evidence

**Files:**
- Modify: `imageharbor/enrich.py` (the fallback at :157-159 and the LOCAL-work comment at :141-151)
- Modify: `CLAUDE.md` (the "Non-AI failures must never feed the circuit breaker" invariant, and `enrich.py`'s module bullet)
- Test: `tests/test_enrich.py` (locate the file actually covering `enrich_library` — `grep -rln "enrich_library" tests/`)

**Interfaces:**
- Produces: a `classifier.pick_class()` exception inside `enrich_library` is recorded as `stats.ai_failed` (not `io_failed`), feeds `breaker.record_failure()`, and aborts the pass when the breaker opens — identical handling to a `describe()` failure. `classifier.adjudicate` stays as-is (already caught inside `taxonomy.resolve_or_create`, degrading to minting a new code — document, don't change).

- [ ] **Step 1: Write the failing test.** Build on the file's existing fakes (a classifier whose `describe` succeeds is already there):

```python
def test_a_pick_class_failure_is_ai_evidence_and_feeds_the_breaker(...):
    """OpenAIClassifier.pick_class is a network call; when the backend dies
    between describe() and a concept-map miss, the old code recorded
    io_failed (which never feeds the breaker or quarantine) and churned the
    whole queue against a dead backend."""

    class DescribesButCannotPick(StubClassifier):  # or the file's fake base
        def pick_class(self, content, classes):
            raise RuntimeError("backend died mid-pass")

    # Arrange: N rows whose subject misses the concept map (unique nonsense
    # subjects so class_for returns None), breaker threshold 3.
    breaker = CircuitBreaker(trip_threshold=3, backoff_base=60)
    stats = enrich_library(..., classifier=DescribesButCannotPick(), breaker=breaker, ...)

    assert stats.ai_failed        # not io_failed
    assert not stats.io_failed
    assert stats.aborted          # breaker opened and the pass stopped
    assert breaker.is_open()
```

Also a companion: `test_a_concept_map_hit_never_calls_pick_class` if not already pinned (a subject already in `learned_concepts` must enrich fine with the same broken `pick_class` — proves the fallback is the only new breaker path).

- [ ] **Step 2: Run to verify failure** — `uv run pytest <file> -q`: first test FAILS (today the exception lands in the outer LOCAL `except`, `io_failed`, no abort).

- [ ] **Step 3: Implement.** In `enrich.py`, inside the LOCAL `try`, isolate the fallback (replacing :157-159):

```python
            if cls is None:
                # The text-only fallback is a BACKEND call on a real
                # classifier (OpenAIClassifier.pick_class -> chat.completions)
                # -- its failure is AI evidence, handled exactly like a
                # describe() failure. adjudicate() is different: taxonomy
                # catches it internally and degrades to minting a new code.
                try:
                    cls = classifier.pick_class(content, classes)
                except Exception as exc:
                    logger.warning(
                        "Class fallback failed for %s: %s", actual.name, exc
                    )
                    stats.errors += 1
                    stats.ai_failed.append(digest)
                    if breaker is not None:
                        breaker.record_failure()
                        if breaker.is_open():
                            logger.error(
                                "AI backend appears down — aborting enrichment"
                                " after repeated failures in the class fallback"
                            )
                            stats.aborted = True
                    continue
                concept_map.remember(catalog, content.primary_subject, cls)
```

then, after the `continue`-carrying loop body, ensure the `stats.aborted` case breaks the loop — mirror how the `describe()` failure block breaks (`if stats.aborted: break` after the `continue` cannot work inside one statement; structure it exactly like :126-135: record, check `is_open()`, set `aborted`, `break`; a non-open failure `continue`s). Note the subtlety: `continue` vs `break` must match the describe-block's behavior — open → `break`, else `continue`.

Update the LOCAL-work comment (:141-151) to say "everything below EXCEPT the pick_class fallback"; move `breaker.record_success()` consideration NOT needed (success stays keyed to describe).

- [ ] **Step 4: Update CLAUDE.md** in the same commit: the invariant bullet "Non-AI failures must never feed the circuit breaker" changes from "Only a failure raised by `AIClassifier.describe()`" to "Only a failure raised by the classifier's backend-facing calls — `describe()` and the `pick_class` fallback — feeds the breaker (`adjudicate` is caught inside `taxonomy.resolve_or_create` and degrades to minting)". Update the `enrich.py` module bullet's breaker sentence to match.

- [ ] **Step 5: Suite, lint, types, commit**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/enrich.py CLAUDE.md tests/<file>
git commit -m "fix(enrich): pick_class failures are backend evidence and feed the breaker"
```

---

### Task 2: fsync before the catalog can claim "verified"

**Files:**
- Modify: `imageharbor/pipeline.py` (Step 6/7 block, :308-362), `imageharbor/sidecar.py` (atomic write, ~:135-143), `imageharbor/takeout/provenance.py` (`_write_bytes`, :99-110)
- Create: nothing new — the helper lives in `imageharbor/util.py`
- Test: `tests/test_util.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `imageharbor.util.fsync_file(path: Path) -> None` — opens `rb`, `os.fsync(fh.fileno())`; raises OSError upward (callers decide). Called: (a) in `pipeline.py` after the copy/replace and BEFORE `verify_file`, on every path that wrote bytes (both `shutil.copy2` sites and the `os.replace` site — a moved staging file was never synced either); (b) in `sidecar.py` on the tmp file before `os.replace`; (c) in `provenance._write_bytes` on the tmp before `tmp.replace`. Directory-entry fsync is deliberately NOT attempted — `os.fsync` on a directory fd is unsupported on Windows; the file-content sync is the gap the review named. State this in the helper's docstring.

- [ ] **Step 1: Write the failing tests.**

`tests/test_util.py`:
```python
def test_fsync_file_syncs_and_propagates_oserror(tmp_path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    called = {}
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: called.setdefault("fd", fd))
    fsync_file(f)
    assert "fd" in called
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk gone")))
    with pytest.raises(OSError):
        fsync_file(f)
```

`tests/test_pipeline.py` (behavioral pin — the ordering, not the syscall):
```python
def test_the_copy_is_fsynced_before_it_is_verified(tmp_path, monkeypatch, ...):
    """Power-loss gap: verify_file reads the page cache, so without an
    fsync the catalog can durably record 'verified' for bytes that never
    reached disk. Pin the order: fsync happens, and happens before verify."""
    events = []
    monkeypatch.setattr(pipeline_module, "fsync_file",
                        lambda p: events.append(("fsync", Path(p).name)))
    real_verify = pipeline_module.verify_file
    monkeypatch.setattr(pipeline_module, "verify_file",
                        lambda p, d: events.append(("verify", Path(p).name)) or real_verify(p, d))
    # run one file through Pipeline.process_file (reuse this file's fixtures)
    ...
    fsync_i = events.index(next(e for e in events if e[0] == "fsync"))
    verify_i = events.index(next(e for e in events if e[0] == "verify" and events.index(e) > fsync_i))
    assert fsync_i < verify_i
```
(Adapt to the file's fixture idioms; the assertion of order is the spec. Note the destination-already-verified fast path at pipeline.py:310 performs no write and needs no fsync — don't break it.)

- [ ] **Step 2: Run to verify failure** — `fsync_file` doesn't exist → ImportError.

- [ ] **Step 3: Implement.** `util.py` gains:

```python
def fsync_file(path: Path) -> None:
    """Flush *path*'s written bytes to stable storage.

    verify-after-copy reads back through the OS page cache, so without this
    the catalog could durably assert "copied, verified" for data that never
    reached the platter (power loss, not process crash). Directory-entry
    durability is deliberately out of scope: os.fsync on a directory fd is
    unsupported on Windows, and the file's content is the gap that matters.
    """
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())
```
(`from pathlib import Path` + `import os` — util stays stdlib-only.) Wire the three call sites; in `pipeline.py` place ONE call immediately before the Step-7 `verify_file`, inside the `else:` branch that wrote bytes (covers copy2, replace, and the EXDEV fallback alike). In `sidecar.py` and `provenance._write_bytes`, sync the tmp file after writing, before the replace.

- [ ] **Step 4: Idempotence check** — sidecar writes now fsync; re-run the sidecar suite and the property test:

```bash
uv run pytest tests/test_sidecar.py tests/test_sidecar_schema.py tests/test_takeout_provenance.py tests/test_pipeline.py tests/test_util.py -q
```

- [ ] **Step 5: Full suite, lint, types, commit**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/util.py imageharbor/pipeline.py imageharbor/sidecar.py imageharbor/takeout/provenance.py tests/test_util.py tests/test_pipeline.py
git commit -m "fix: fsync written bytes before verification and atomic replaces"
```

---

### Task 3: Model download recovery

**Files:**
- Modify: `imageharbor/faces/download.py`
- Test: `tests/faces/test_download.py` (exists — extend)

**Interfaces:**
- Produces: on checksum mismatch of an EXISTING file, the bad artifact is deleted before raising, and the message says a re-run will re-download; the `.part` temp is removed on any download failure (`try/finally`).

- [ ] **Step 1: Write the failing tests:**

```python
def test_a_mismatched_artifact_is_deleted_so_the_next_run_can_recover(tmp_path):
    # Old behavior: the bad file stayed at target, and the `if not
    # target.exists()` gate meant every subsequent run re-hashed the same
    # bad bytes and re-failed, forever, with no hint.
    target = tmp_path / INFO.filename
    target.write_bytes(b"corrupt")
    with pytest.raises(ChecksumMismatch) as exc:
        ensure(INFO, tmp_path)
    assert not target.exists()
    assert "re-run" in str(exc.value).lower() or "re-download" in str(exc.value).lower()


def test_a_failed_download_leaves_no_part_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "imageharbor.faces.download.urllib.request.urlretrieve",
        lambda url, tmp: (_ for _ in ()).throw(OSError("network died")),
    )
    with pytest.raises(OSError):
        ensure(INFO, tmp_path)
    assert not list(tmp_path.glob("*.part"))
```
(`INFO`: reuse the test file's existing pinned `ModelInfo` fixture/constant.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** In `ensure()`:

```python
    if not target.exists():
        logger.info("downloading face model %s from %s", info.name, info.url)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(info.url, tmp)  # noqa: S310 - pinned URL
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)

    actual = _sha256(target)
    if actual != info.sha256:
        # Delete the bad artifact: leaving it wedges every future run behind
        # the `not target.exists()` gate with the same failure and no exit.
        target.unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"{info.filename}: expected {info.sha256}, got {actual}; the "
            "artifact was removed -- re-run to re-download"
        )
```

- [ ] **Step 4: Suite, lint, types, commit**

```bash
uv run pytest tests/faces/test_download.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/faces/download.py tests/faces/test_download.py
git commit -m "fix(faces): unwedge checksum mismatches and clean up failed downloads"
```

---

### Task 4: Sweep staging debris at ingest start

**Files:**
- Modify: `imageharbor/takeout/ingest.py` (`run()`, the `staging_dir.mkdir` at :1115)
- Test: `tests/test_takeout_ingest.py`

**Interfaces:**
- Produces: `run()` removes any pre-existing `<dest>/.takeout-staging/` contents before creating it fresh. Safe by the module's own reasoning (`archive.discard_staged`'s docstring: leftovers are "inert debris, not state" — phase 2 resumes from `takeout_members`, never from the staging floor). Single-writer deployment means nothing else owns the directory at `run()` start.

- [ ] **Step 1: Write the failing test:**

```python
def test_stale_staging_debris_is_swept_at_run_start(dirs, catalog):
    # A kill -9 mid-extract leaves full-size images in mkdtemp dirs under
    # <dest>/.takeout-staging -- INSIDE the library the user backs up.
    # Repeated kills silently accumulate real space. run() owns the floor.
    staging = dirs.dest / ".takeout-staging"
    debris = staging / "tmpOLD" / "leftover.jpg"
    debris.parent.mkdir(parents=True)
    debris.write_bytes(b"\xff\xd8\xff\xe0old bytes")
    ingest_archives(...)  # this file's minimal one-archive invocation
    assert not debris.exists()
    assert not debris.parent.exists()
    assert staging.exists()  # recreated fresh for the run
```
(Reuse this file's `dirs`/`catalog` fixtures and its smallest working `ingest_archives` call.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** Replace `self.staging_dir.mkdir(parents=True, exist_ok=True)` at ingest.py:1115 with:

```python
        # Sweep debris from a previous killed run before creating the floor
        # fresh. Safe: staging is never resume state (phase 2 resumes from
        # takeout_members -- see archive.discard_staged), and the single-
        # writer deployment means nothing else owns this directory now.
        # ignore_errors: a locked leftover must degrade to wasted space,
        # never abort the ingest that would supersede it.
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
```
(`shutil` is already imported in ingest.py — verify, else add.)

- [ ] **Step 4: Suite, lint, types, commit**

```bash
uv run pytest tests/test_takeout_ingest.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/takeout/ingest.py tests/test_takeout_ingest.py
git commit -m "fix(takeout): sweep stale staging debris at ingest start"
```

---

### Task 5: A renamed file's sidecar agrees with its own filename (deferred #9)

**Files:**
- Modify: `imageharbor/enrich.py` (the sidecar merge at :246-263)
- Test: the enrich test file from Task 1, plus a monotonicity-style idempotence assertion

**Interfaces:**
- Produces: when (and only when) the tier-gated rename fired this row, the `merge_sidecar` update dict ALSO carries a `descriptor` block:

```python
{"value": descriptor, "tier": tiers.DESC_AI_SUBJECT,
 "source": tiers.DESC_SOURCE_NAMES[tiers.DESC_AI_SUBJECT]}
```

so the sidecar stops contradicting the filename and catalog after an AI rename (observed live on hpz440: file renamed to `boys_gGB9...jpg` while its sidecar still said `descriptor {"value": "", "tier": 0}`). `sidecar_schema.merge`'s tiered handling relocates the superseded facts-pass block into `history[]` — no schema change, no new annotation keys.

- [ ] **Step 1: Write the failing test:**

```python
def test_a_renamed_files_sidecar_agrees_with_its_own_filename(...):
    # Camera-named file (DESC_NONE) -> AI subject renames it. Deferred
    # issue #9: enrich merged only `classification`, so the sidecar kept
    # the facts-pass descriptor and contradicted the filename it sits
    # beside -- self-contradictory in exactly the case the tier system
    # exists to make legible.
    # ...run facts pass on IMG_1234.jpg, then enrich_library with the stub...
    renamed = <the renamed organized file>
    doc = json.loads(sidecar_path_for(renamed).read_text(encoding="utf-8"))
    assert doc["descriptor"]["tier"] == tiers.DESC_AI_SUBJECT
    assert doc["descriptor"]["value"] in renamed.name
    # the facts-pass block is history, not lost (never-lose rule):
    assert any(h.get("tier") == tiers.DESC_NONE
               for h in doc["descriptor"].get("history", []))


def test_re_enriching_leaves_the_descriptor_sidecar_byte_identical(...):
    # idempotence: run enrich twice (second via --reclassify semantics or a
    # fresh enrich over the already-enriched row), sidecar bytes unchanged.
```

- [ ] **Step 2: Run to verify failure** (first test: sidecar still holds the tier-0 block as current).

- [ ] **Step 3: Implement.** In `enrich.py`, build the update dict once, adding the descriptor block only on the renamed path:

```python
            if write_sidecars:
                updates: dict = {"classification": {...as today...}}
                if final_path is not actual:
                    # The rename fired: record the descriptor the filename
                    # now carries, so sidecar, filename, and catalog agree.
                    # Deferred-known-issues #9.
                    updates["descriptor"] = {
                        "value": descriptor,
                        "tier": tiers.DESC_AI_SUBJECT,
                        "source": tiers.DESC_SOURCE_NAMES[tiers.DESC_AI_SUBJECT],
                    }
                try:
                    merge_sidecar(final_path, updates)
                ...
```
Careful with the identity check: `final_path` is reassigned to `proposed` only on a successful rename (:206) — `final_path is not actual` is exactly "the rename fired"; verify `descriptor` (the normalized string) is still in scope there.

- [ ] **Step 4: Suite (monotonicity + sidecar suites especially), lint, types, commit**

```bash
uv run pytest tests/test_monotonicity.py tests/test_sidecar_schema.py <enrich test file> -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/enrich.py tests/<file>
git commit -m "fix(enrich): merge the descriptor block when a rename fires (deferred #9)"
```

---

### Task 6: Split `enrich_failed` into AI vs I/O counts (deferred #13)

**Files:**
- Modify: `imageharbor/catalog.py` (runs DDL comment, additive columns, `run_finish`), `imageharbor/watcher.py` (both `run_finish` call sites, :472-482 and :525-535), `imageharbor/dashboard/stats.py` (`_window_summary`, :460-481)
- Test: `tests/test_catalog.py`, `tests/test_watcher.py`, `tests/test_dashboard_stats.py`

**Interfaces:**
- Produces: additive `runs` columns `enrich_ai_failed INTEGER NOT NULL DEFAULT 0` and `enrich_io_failed INTEGER NOT NULL DEFAULT 0` (existing catalogs upgrade in place via the same additive-column mechanism `_ensure_photo_columns` uses — add `_ensure_run_columns` beside it, called from `__init__`, mirroring its style). `run_finish` gains keyword params `enrich_ai_failed: int = 0, enrich_io_failed: int = 0`; `enrich_failed` REMAINS and stays the total (backward compat — the dashboard page and projections read it today). `_window_summary` adds both new keys to its dict. The circuit breaker itself always kept this distinction; only the `runs` row lost it.

- [ ] **Step 1: Write the failing tests:**

```python
# tests/test_catalog.py
def test_run_finish_records_the_ai_io_failure_split(catalog):
    run_id = catalog.run_start("enrich")
    catalog.run_finish(run_id, scanned=5, copied=0, duplicates=0, errors=0,
                       enriched=2, enrich_failed=3, breaker_state="CLOSED",
                       paused=False, enrich_ai_failed=2, enrich_io_failed=1)
    row = catalog.recent_runs(1)[0]
    assert row["enrich_ai_failed"] == 2
    assert row["enrich_io_failed"] == 1

def test_an_existing_catalog_gains_the_run_columns_in_place(tmp_path):
    # open, close, strip columns? -- can't strip; instead simulate: create a
    # catalog with current code, drop the two columns via raw SQL is not
    # possible in SQLite <3.35 style -- so pin the mechanism instead:
    # _ensure_run_columns is idempotent (open the same catalog twice).
    ...
```
(For the migration test, mirror however `_ensure_photo_columns` is currently tested — find with `grep -n "_ensure_photo_columns" tests/test_catalog.py` — and follow that pattern; if it's tested by opening a hand-built legacy schema, do the same with a `runs` table lacking the two columns.)

```python
# tests/test_watcher.py — extend the existing runs-row assertions:
# after a watch cycle with a failing classifier, the enrich run row's
# enrich_ai_failed reflects len(ai_failed), enrich_io_failed the remainder,
# and enrich_failed still equals their sum plus any crash count.
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.**
  - `catalog.py`: add the two columns to the `runs` DDL (new catalogs) AND `_ensure_run_columns` (existing catalogs; `ALTER TABLE runs ADD COLUMN ... DEFAULT 0` guarded by a PRAGMA table_info check, mirroring `_ensure_photo_columns`); extend `run_finish`'s signature + UPDATE statement.
  - `watcher.py` facts-phase call (:472-482): pass `enrich_ai_failed=0, enrich_io_failed=0`. Enrich-phase call (:525-535): `enrich_ai_failed=len(enrich_stats.ai_failed)` and `enrich_io_failed=row_errors - len(enrich_stats.ai_failed)` (the crash-in-flight increment counts as I/O — it is local-work evidence by definition; note this in a comment).
  - `stats.py` `_window_summary`: accumulate + emit `enrich_ai_failed` / `enrich_io_failed` alongside the existing keys (rows from pre-upgrade history lack the columns → `run.get(...) or 0` handles it, same as every other field).
  - The dashboard page (`index.html`) is deliberately NOT changed — the JSON now carries the split; surfacing it in the UI is R5's dashboard work. Say so in the commit body.

- [ ] **Step 4: Suite, lint, types, commit**

```bash
uv run pytest tests/test_catalog.py tests/test_watcher.py tests/test_dashboard_stats.py -q
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/catalog.py imageharbor/watcher.py imageharbor/dashboard/stats.py tests/
git commit -m "feat(catalog): split enrich failures into AI vs I/O counts (deferred #13)"
```

---

### Task 7: Ship v1.2.0 and deploy

Interactive with GitHub + hpz440. Mirrors R2's Task 9, which worked end-to-end.

- [ ] **Step 1: Final local verification** — `uv run ruff check .`; `uv run mypy imageharbor`; `uv run pytest` (tail in report).

- [ ] **Step 2: PR + CI (five legs)**

```bash
git push -u origin release/v3-integrity
gh pr create --title "R3: integrity and failure-handling corrections" --body "Implements docs/superpowers/plans/2026-09-13-r3-integrity-corrections.md (quality-roadmap Release 3): pick_class breaker evidence, fsync-before-verify, model-download recovery, staging sweep, sidecar descriptor agreement (deferred #9), enrich failure split (deferred #13)."
```

Watch the PR's CI run to completion — lint, test×3, docker all green.

- [ ] **Step 3: Merge with a minor bump**

```bash
gh pr merge <PR#> --merge --subject "Merge R3 integrity corrections [minor]"
```

Watch CI on main, then the Release workflow: tag `v1.2.0` + Release + GHCR `v1.2.0`/`:latest`. Record run URL + job conclusions.

- [ ] **Step 4: Deploy to hpz440** (the R2 path, proven): `ssh claude@hpz440`, then in `/home/claude/imageharbor`: `git pull`, `docker compose pull`, `docker compose up -d`. Verify on host: `curl -fsS http://<tailnet-ip>:8087/healthz` → ok; container logs show the new version; the catalog opens cleanly (the additive `runs` columns migrate in place — check the watch log for a clean first pass, and `docker compose logs --tail 50` for any `OperationalError`). From this machine: `curl -s -o NUL -w "%{http_code}" http://100.69.239.123:8087/healthz -H "Host: 100.69.239.123"` → 200.

- [ ] **Step 5: Report** — versions, run URLs, deploy evidence; note that fsync adds per-file latency on the CIFS mount (expected, accepted — integrity over throughput; if the next overnight pass shows unacceptable slowdown, `IMAGEHARBOR`-level tuning is a future decision, not a rollback trigger).

---

## Self-Review Notes

- Roadmap R3 coverage: breaker leak (T1), fsync (T2), download recovery (T3), staging sweep (T4), deferred #9 (T5), deferred #13 (T6), ship+deploy (T7). Complete.
- T1 changes a documented CLAUDE.md invariant — the doc edit rides the same commit so no window exists where code and contract disagree.
- T2's helper lives in `util.py` (stdlib-only leaf) so `sidecar.py` and `provenance.py` can share it without new coupling; the already-verified fast path (pipeline.py:310) is explicitly exempted.
- T5 keeps every byte through `sidecar_schema.merge` — the superseded block goes to `history[]`, satisfying never-lose, and the idempotence test guards the annotation-registry trap.
- T6 is purely additive on `runs` (SCHEMA_VERSION unchanged); `enrich_failed` retained as the total so `projections.py` and the page keep working untouched.
