# Final whole-branch review fixes — feature/face-recognition

## Context

The face-recognition branch (17 tasks, `.superpowers/sdd/progress.md`) passed
its final whole-branch review with findings that must be fixed before merge.
This plan fixes them, one task per finding (or a small coherent group),
**in the order given** — the two Criticals first.

Working directory for every task: `D:/Users/Conrad/Documents/programming/ImageHarbor-worktrees/face-recognition`
(isolated git worktree, branch `feature/face-recognition`). Do NOT touch
`D:/Users/Conrad/Documents/programming/ImageHarbor` — a different session is
actively editing that checkout.

## Global constraints (every task)

- `uv run pytest`, `uv run python` — never `pip install`, never a bare `python`/`pytest`.
- Do not chain shell commands with `&&`; issue separate commands.
- Comments explain *why*, not *what*. No per-file licence header.
- Every fix needs a test that **fails against the current code** (RED) and
  passes after the fix (GREEN). Paste real terminal output — banners,
  timings, filler and all. Do not retype or tidy it, and do not fabricate it.
  Two tasks on this branch already shipped fabricated RED evidence — this is
  a hard requirement, not a formality.
- Run the full suite at the end of the task:
  `uv run pytest` (baseline: 1144 passed, 11 skipped) and, if
  `IMAGEHARBOR_FACE_MODEL_DIR` is set in the environment, also with weights
  (baseline: 1154 passed, 1 skipped). Report both.
- Do NOT touch anything the review listed as "ship as is" or "follow-up":
  the Kelvin display, `MixedModelError`'s base, the `det(u)*det(vt)` branch,
  `split`'s local `_centroid`, `google_names` normalization, `measure_threshold`'s
  guard, `face_organized_paths`, `clusters.centroid`, the doc-drift list. Those
  are follow-up work, not this fix.
- One commit per task (or per finding within a combined task), with a
  message describing the *why*.

---

## Task 1: CRITICAL — the optional extra is now mandatory

`imageharbor/cli.py`'s `watch` command imports `dashboard.server` at
function top (`from .dashboard import server as dashboard_server`, around
line 565), unconditionally, **before** the `--no-dashboard` check further
down (around line 655). `dashboard/server.py` imports `dashboard.people` and
`dashboard.stats` at module scope; both of those import
`from imageharbor.faces.store import FaceStore` at module scope
(`imageharbor/dashboard/people.py:30`, `imageharbor/dashboard/stats.py:88`);
`imageharbor/faces/store.py:24` does `import numpy as np` at module scope.
numpy ships only in the `faces` extra. Net effect: `watch --no-faces
--no-dashboard` raises `ImportError: No module named 'numpy'` when numpy is
not installed — even though both faces and the dashboard were explicitly
disabled.

This contradicts `imageharbor/faces/__init__.py`'s module docstring ("only
`detect` and `embed` import onnxruntime... importing this package must never
require the optional `faces` extra"), `cli.py`'s `--faces` help text ("with
it missing, this only logs one warning and keeps organizing"), and
CLAUDE.md's stated invariant that a missing extra degrades to one warning.

`tests/faces/test_extra.py` only blocks `onnxruntime` at the meta-path, never
`numpy`, so it cannot see this failure mode.

**Fix:**

1. In `imageharbor/dashboard/people.py` and `imageharbor/dashboard/stats.py`,
   both of which already have `from __future__ import annotations` (so type
   annotations are never evaluated at runtime), move
   `from imageharbor.faces.store import FaceStore` under `typing.TYPE_CHECKING`.
   Check first that neither module uses `FaceStore` anywhere except type
   annotations (it does not, as of this writing — confirm with `grep -n
   FaceStore` in both files before moving the import, since a runtime use
   would need a different fix, e.g. a function-local import at the call site).
2. Extend `tests/faces/test_extra.py` to also block `numpy` at the same
   `builtins.__import__` patch point used for onnxruntime in
   `test_package_imports_when_onnxruntime_raises_non_import_error`, and add a
   new test that `watch --no-faces --no-dashboard` (via
   `click.testing.CliRunner`, same pattern as `tests/test_cli.py` or
   `tests/faces/test_faces_cli.py`) starts and shuts down cleanly with both
   `numpy` and `onnxruntime` blocked. The watcher only needs to run one pass
   and stop — look at how existing `watch` CLI tests bound the run (a
   `stop_event`-like mechanism, `--once`, or monkeypatching `_watcher.watch`)
   rather than letting it loop forever.
3. Fix the mid-function `import builtins` in
   `test_package_imports_when_onnxruntime_raises_non_import_error` (currently
   inside the test body) — move it to the module's top-level imports with the
   rest.
4. RED must be demonstrated against the **current, unmodified** `cli.py` and
   `people.py`/`stats.py` — show the new test failing with the numpy
   ImportError before making the `TYPE_CHECKING` change, then show it passing
   after.

Test files: `tests/faces/test_extra.py`. Covering command: `uv run pytest
tests/faces/test_extra.py -v`, then the full suite per Global constraints.

---

## Task 2: CRITICAL — a recluster mid-confirm writes the wrong identity

`imageharbor/dashboard/people.py`'s `confirm` (around line 167-176) and
`merge` (around line 200-213) each call `_cluster_exists(store, cluster_id)`
(or, for `merge`, per id), which takes `store.lock`, queries, and **releases**
it, and only afterward call `store.confirm`/`store.merge`
(`imageharbor/faces/store.py:614`, `:648`), which take the lock again. Between
those two lock acquisitions, `FaceStore.replace_clusters`
(`store.py:393`) can run a whole-library recluster: it holds `store.lock` for
the entire rebuild, and `DELETE FROM clusters WHERE id IN (...)` followed by
fresh `INSERT`s means `clusters.id` (a plain `INTEGER PRIMARY KEY`, not
`AUTOINCREMENT`) gets **recycled**. A cluster id validated as existing by
`_cluster_exists` before the recluster can refer to a completely different
cluster's faces by the time `store.confirm`/`store.merge` actually run.

Reproduced by the reviewer:
```
cluster->faces before: {1: [1], 2: [2], 3: [3]}
confirm(cluster_id=3, "Emma") -> HTTP 200
cluster->faces after : {1: [3], 2: [2], 3: [1]}
operator intended to name faces: [3]
faces actually named           : [1]
*** WRONG IDENTITY WRITTEN ***
```

`FaceStore.split` (`store.py:506`) is immune to this exact race because it
re-validates cluster existence **inside its own lock acquisition**, right
before it mutates — that is the precedent to follow. `merge`'s current
existence check for the *person* id (`SELECT 1 FROM people WHERE id=?`,
`people.py` around line 202-208) has the identical bug: it also runs under a
separate `with store.lock:` block that releases before `store.merge` is
called, though `people` rows are not deleted by `replace_clusters` so that
particular check is not what caused the reproduced failure — the cluster-id
checks are.

**Fix:**

1. Give `FaceStore.confirm` (`store.py:614`) an inside-the-lock existence
   check: before the `UPDATE clusters SET person_id=...`, or as part of the
   same locked block, verify the `cluster_id` still exists (`SELECT 1 FROM
   clusters WHERE id=?`) and raise (e.g. `KeyError(f"no such cluster:
   {cluster_id}")`) rather than silently performing a no-op `UPDATE` that
   matches zero rows.
2. Give `FaceStore.merge` (`store.py:648`) the same inside-the-lock check for
   every id in `cluster_ids` — raise naming the unknown id(s) rather than
   silently matching zero rows for a stale/recycled id while matching real
   rows for the others.
3. `dashboard/people.py`'s `confirm`/`merge` still keep their existing
   pre-lock `_cluster_exists`/person-id checks (those give a fast, friendly
   `ValueError` -> HTTP 400 for the common case of a genuinely bad request);
   the store-level check is the correctness backstop for the race, matching
   how `split`'s two-layer check already works (HTTP-facing check for the
   nice error, store-level check as the real guard). Decide whether
   `people.py`'s `confirm`/`merge` need to catch the new store-level
   exception and convert it to `ValueError` for HTTP-boundary consistency —
   look at how `people.py`'s `split` wrapper and `FaceStore.split` interact
   (`split` in `store.py` raises `KeyError`, and check what, if anything,
   `people.py` does with it) and follow the same pattern for `confirm`/`merge`.
4. Add a test that reproduces the race and **fails against the current
   code**. A real thread race is flaky by nature here (the window is a single
   Python function call); prefer a deterministic simulation of the
   interleaving — e.g. monkeypatch or subclass to call `replace_clusters` (or
   directly perform the delete/recycle/insert `replace_clusters` does)
   between `_cluster_exists`'s lock release and `store.confirm`'s lock
   acquisition, or simplest: call `store.confirm`/`store.merge` directly with
   a cluster id that used to exist but was deleted+recycled by a prior
   `replace_clusters` call in the test, and assert it raises rather than
   silently writing to the wrong (recycled) cluster. Model the reproduction
   in the finding above: seed 3 single-face clusters, confirm one is
   subsequently deleted and its id reused, and assert the write goes to the
   right cluster / raises rather than silently mislabeling a recycled one.

Test files: `tests/faces/test_store.py`, `tests/faces/test_dashboard_people.py`.
Covering command: `uv run pytest tests/faces/test_store.py
tests/faces/test_dashboard_people.py -v`, then the full suite.

---

## Task 3: IMPORTANT — the crop-rank contract has no test on either side

`imageharbor/faces/runner.py`'s `_scan_one` (around lines 100-130) names crop
files `<digest>-<i>.jpg`, where `i` is the 0-based rank among **successfully
aligned** (kept, non-degenerate) faces, assigned via `enumerate(zip(
aligned_detections, crops, embeddings))` — this loop runs *after* an earlier
loop that already appended gate-rejected records (`low_score`, `too_small`)
and a second loop that appended `degenerate_landmarks` records, so those
non-kept records occupy earlier positions in the `records` list that is later
passed to `store.record_scan`, which inserts faces in list order (and hence
in ascending `id` order, confirmed via `record_scan`'s `INSERT` loop at
`store.py:160`).

`imageharbor/dashboard/people.py`'s `crop_bytes` (around lines 258-298)
re-derives that same rank by querying `SELECT id FROM faces WHERE
sha256_b64url=? AND rejected IS NULL ORDER BY id` and finding the target
`face_id`'s index in that list — i.e., rank among non-rejected faces for that
digest, ordered by id. Because gate-rejected and degenerate-landmark faces
are inserted (with a non-NULL `rejected` reason) *before* the kept ones in
id order, filtering to `rejected IS NULL` before computing the index is what
makes the two ranks agree.

The shipped code is correct, but two discriminating mutations both pass all
239 existing tests:
- appending kept faces in **reverse** of their crop rank in `runner.py`'s
  final loop (e.g. iterating `reversed(list(enumerate(...)))` while still
  saving `f"{digest}-{i}.jpg"` with the original `i`, or reversing the
  insertion order passed to `record_scan` so ids no longer ascend in crop-rank
  order)
- dropping `AND rejected IS NULL` from the rank query in `people.py`'s
  `crop_bytes` (around line 290)

Either mutation makes `crop_bytes(face_id)` return a **different face's**
crop while the dashboard still labels it with `face_id`'s row — an operator
looking at face A's photo while being asked to confirm/reject/name face B's
cluster, silently.

**Fix:** add a test that scans a real (or the project's existing synthetic
fixture — check `tests/faces/test_runner.py` and `tests/faces/test_detect.py`
for the multi-face / drawn-face fixture already used elsewhere in this
branch, e.g. Task 9's committed Pillow-drawn face fixture) multi-face photo
where **at least one face is rejected by the quality gate** (a low-score or
too-small synthetic detection is enough — look at how `QualityGate` and its
thresholds are constructed in existing tests to force a gate rejection
deterministically), runs it through `runner._scan_one` (or `runner.scan`)
against a real `FaceStore` and a real crop directory, then calls
`dashboard.people.crop_bytes(crop_dir, face_id, store=store)` for a kept
face's id and asserts the bytes returned match that face's own saved crop
file on disk (read the file directly and compare, or compare against the
crop the runner is known to have written for that face's rank) — not some
other face's crop. Confirm both mutations described above make this new test
fail:
1. Temporarily reverse the kept-face append order in `runner.py`'s final
   loop (or otherwise decouple `i` from insertion order) and show the test
   fails.
2. Revert that, then temporarily drop `AND rejected IS NULL` from the rank
   query in `people.py`'s `crop_bytes` and show the test fails.
Revert both temporary mutations before committing — the shipped code does
not change; only the test is new.

Test file: `tests/faces/test_runner.py` or a new `tests/faces/test_crop_rank.py`
if that reads cleaner — your call, but keep it near the existing runner/people
tests rather than inventing a new top-level test directory.
Covering command: `uv run pytest tests/faces/test_runner.py
tests/faces/test_dashboard_people.py -v` (adjust to wherever the test lands),
then the full suite.

---

## Task 4: IMPORTANT — the recluster gate spins forever on a zero-cluster library

`imageharbor/watcher.py` (around lines 836-838) computes:
```python
recluster_due = (
    unclustered > face_config.recluster_threshold
    or not face_config.store.cluster_ids()
)
```
`store.cluster_ids()` (`imageharbor/faces/store.py:379`) returns **every**
cluster id in the `clusters` table, with no `embed_model` filter — unlike
`unclustered_face_count(embed_model)` a few lines above it, which is scoped.
When no face has ever successfully clustered (e.g. every detected face in the
library was rejected by the quality gate, or the library is fresh), `not
store.cluster_ids()` is `True` on every single watch cycle, forever —
`recluster_due` never goes false no matter what `build_clusters` does (it has
nothing to build from, so it correctly keeps producing zero clusters). Each
cycle then re-runs `google_names(dest)`, a full `rglob("*.json")` over the
organized tree (find the exact call site around this block in `watcher.py`)
— reproduced against a library whose only face was rejected by the quality
gate: `recluster_due=True` every cycle, and at 77,000 sidecars that `rglob`
becomes a real, repeated cost on every watch interval, forever, on a fresh
deployment or a landscape-heavy library with few or no faces.

**Fix:**

1. Change the gate so it requires **both** "no clusters exist" **and** "there
   is at least one clusterable (unclustered) face" — i.e. don't treat a
   library with zero clusterable faces as due for reclustering. Something
   like:
   ```python
   recluster_due = unclustered > face_config.recluster_threshold or (
       unclustered > 0 and not face_config.store.cluster_ids()
   )
   ```
   (adjust to whatever reads clearest against the existing variable names —
   the requirement is: no spurious `recluster_due=True` when `unclustered ==
   0`).
2. While in `store.cluster_ids()` (`store.py:379-384`), scope it by
   `embed_model` the same way `unclustered_face_count` and other
   cluster-scoped methods already are — this was flagged in the review as a
   Minor deferred from Task 16 but elevated to fix-before-merge here. Check
   every call site of `cluster_ids()` (there are a few across
   `cli.py`/`watcher.py`/tests) and update them to pass the relevant
   `embed_model` — likely `face_config.embedder.model_name` at each call
   site, matching how `unclustered_face_count(embed_model)` is already
   called nearby.

**Fix must be verified through the real watcher wiring**, not just a direct
`store.cluster_ids()` unit test — see `tests/faces/test_watch_faces.py` for
the existing pattern of driving `watcher.watch`/`run_once` with a real
`FaceStore` and asserting on what gets called (e.g. mock/spy on
`google_names` or on `build_clusters` to assert it does NOT get invoked
every single cycle when there are zero clusterable faces).

Add a test that reproduces the spin (fails against current code): a
`FaceStore` with zero clusters and zero unclustered faces (or a face that was
scanned but has no embedding because it was gate-rejected) run through
several watch cycles, asserting the recluster path is NOT re-triggered every
cycle. Also add/verify a `cluster_ids(embed_model)` test in
`tests/faces/test_store.py` confirming cross-model isolation, matching how
`unclustered_face_count` is already tested there.

Test files: `imageharbor/watcher.py`, `imageharbor/faces/store.py`, call
sites in `imageharbor/cli.py`; tests in `tests/faces/test_watch_faces.py`,
`tests/faces/test_store.py`, plus any test currently calling
`cluster_ids()` without an argument (grep for `cluster_ids(` across `tests/`
and update call sites, since this is a signature change).
Covering command: `uv run pytest tests/faces/test_watch_faces.py
tests/faces/test_store.py -v`, then the full suite.

---

## Task 5: IMPORTANT — the `detect_model` threading is untested

`imageharbor/watcher.py` (around line 830, inside the faces-pass block)
calls:
```python
face_runner.propagate_sidecars(
    face_config.store,
    face_config.dest,
    face_config.detector.model_name,
)
```
Mutating `face_config.detector.model_name` to `face_config.embedder.model_name`
here passes all **290** currently-relevant tests. The consequence in
production: `propagate_sidecars` (`imageharbor/faces/runner.py:246`) passes
whatever `detect_model` it's given straight through to
`store.mark_sidecar_written(digest, detect_model)`
(`imageharbor/faces/store.py:694`), whose `UPDATE ... WHERE detect_model=?`
would then match **zero rows** (since `face_scan.detect_model` is stamped
with the detector's model name, not the embedder's, back in
`record_scan`/`_scan_one`) — `face_scan.sidecar_at` never advances,
`iter_pending_sidecars`/whatever downstream consumer relies on `sidecar_at`
keeps returning the same already-handled photos forever, and since
`merge_sidecar`/the sidecar write always executes regardless of whether the
bookkeeping row advanced, this becomes a **permanent full-library sidecar
rewrite every single watch cycle** — a silent, severe, ongoing cost with zero
test signal.

**Fix:** add a test that asserts this wiring is correct — and correctness
here means "a second pass through the real `watch`/faces-pass code path is a
no-op" (no repeated sidecar writes, `sidecar_at` genuinely advances) — **run
through the actual watcher wiring** (`imageharbor.watcher.watch` or
`run_once`, whatever the existing `tests/faces/test_watch_faces.py` pattern
already drives), not through a direct call like
`runner.propagate_sidecars(store, dest, "yunet")` that bypasses the exact
line in `watcher.py` under test — a direct call can't detect this class of
wiring bug because it hand-supplies the correct model name instead of
sourcing it from `face_config.detector.model_name` the way the real code
does.

Concretely: run two full watch cycles (or two calls to whatever the faces
pass function is that `watch()` invokes per cycle) against a photo that gets
a face detected, embedded, and a sidecar written on cycle 1; assert cycle 2
performs no additional sidecar write / that `face_scan.sidecar_at` does not
change between cycle 1's end and cycle 2's end. Confirm this test fails if
`face_config.detector.model_name` is swapped for `face_config.embedder.model_name`
at the `propagate_sidecars` call site (temporarily make that mutation, show
RED, revert it, show GREEN — do not ship the mutation).

Test file: `tests/faces/test_watch_faces.py`.
Covering command: `uv run pytest tests/faces/test_watch_faces.py -v`, then
the full suite.

---

## Task 6: IMPORTANT — a documented invariant is false

`imageharbor/faces/store.py` module docstring (lines 4-5) states: "This is
the only place a person's identity is ever written. `record_proposals`
writes machine guesses to `proposals` and nothing else; `confirm` and
`merge` are the only two methods that touch `clusters.person_id`". `merge`'s
own docstring (around line 651) repeats: "Along with `confirm`, the only
method that writes `clusters.person_id`." `CLAUDE.md` (around lines 611-612
and 756-758) makes the same absolute claim twice. `imageharbor/dashboard/people.py`'s
module docstring (lines 5-6) also states it as fact.

This is false: `replace_clusters` (`store.py:393`, see its own accurate
docstring at lines 396-414) also writes `clusters.person_id` — when a new
cluster's face set intersects exactly one previously-confirmed person's face
set, it restores that `person_id` (`store.py` around lines 484-487). The
*behavior* is correct and already well-documented at `replace_clusters`
itself (lines 396-414) and in `runner.py`'s `cluster()` docstring (around
line 192, which already correctly distinguishes "never *assigns a new*
[person]" from `replace_clusters` "reattach[ing] an already-confirmed
cluster's *existing* person") — only the absolute wording elsewhere in the
codebase is wrong.

**Fix:** reword every place that states the false absolute to the accurate
distinction: "only `confirm`/`merge` **assign a new** person to a cluster;
`replace_clusters` only ever **restores an existing** one (never invents
one — see its own docstring)." Update:
1. `imageharbor/faces/store.py` module docstring, lines 4-5.
2. `imageharbor/faces/store.py`, `merge`'s docstring, around line 651.
3. `CLAUDE.md`, around line 611-612 (the store.py architecture bullet).
4. `CLAUDE.md`, around lines 756-758 (the "No identity is written without
   human confirmation" bullet) — keep the surrounding "no identity without
   human confirmation" claim, which remains true (`replace_clusters` never
   invents a person_id, it only restores one a human already confirmed);
   only fix the "only two methods" wording.
5. `imageharbor/dashboard/people.py` module docstring, lines 5-6.
Do NOT touch `runner.py`'s `cluster()` docstring (already correct) or the
historical plan/spec documents under `docs/superpowers/plans/` and
`docs/superpowers/specs/` (not load-bearing, out of scope).

No test is needed for a docstring/comment wording fix — this task is a
documentation-only change. Still run the full suite at the end to confirm
nothing else was disturbed, and note in the commit message which invariant
was corrected and why (link back to `replace_clusters`' own accurate
docstring rather than re-explaining).

---

## Task 7: Elevated Minors — also fix before merge

Two small, independent items, both flagged in the review as fix-before-merge
Minors. One commit per item is fine, or one combined commit — controller's
call once you see the diff size.

### 7a. `_faces_model_dir` env-var-over-`--dest` precedence is untested

`imageharbor/cli.py`'s `_faces_model_dir` (around line 958) resolves, in
order: explicit `--model-dir`/`model_dir` argument, then
`$IMAGEHARBOR_FACE_MODEL_DIR`, then `<dest>/.faces-models`, else raises. A
mutation of that resolution order (e.g. swapping the env-var check and the
`--dest` fallback, or checking `dest` before `env`) currently survives all 23
existing CLI tests.

**Fix:** add a three-line test in whichever CLI test file already covers
`_faces_model_dir` or the `watch`/`models download` commands (check
`tests/test_cli.py` and `tests/faces/test_faces_cli.py` for existing
coverage of this function first — it may already be partially tested and
just missing this one ordering case) that sets
`IMAGEHARBOR_FACE_MODEL_DIR` in the environment (via `monkeypatch.setenv`),
passes a `dest` path, and asserts the env var wins over the `dest` fallback
(`resolved != dest / ".faces-models"`), and a second assertion that an
explicit `model_dir` argument still wins over the env var. Confirm this
fails against a deliberately reordered `_faces_model_dir` (temporarily swap
the env-var and dest-fallback checks, show RED, revert, show GREEN).

### 7b. The faces dashboard override is dead code with a lying comment

`imageharbor/dashboard/server.py`'s `_SETTINGS_KEYS = ("interval",
"enrich")` (line 60) does not include `"faces"`, so `POST /api/settings
{"faces": ...}` is rejected as an unknown key before it ever reaches
`ControlPlane.set_override` — meaning `control.faces_enabled` can never
differ from `env_faces` in practice, even though `ControlPlane` itself
(`imageharbor/dashboard/control.py`) already fully implements `_resolve_faces`,
the `faces_enabled` property, `_FACES_KEY` handling in `set_override`, and
`"faces"` in `overrides()` — the storage and resolution layer is complete;
only the HTTP boundary is missing the key. Meanwhile
`imageharbor/watcher.py` (around lines 790-795) has a comment claiming "a
dashboard toggle must take effect on the very next cycle" for the faces
pass, which is not true today since the toggle is unreachable via the API.

Wiring is preferable here since the storage layer is already complete and
correct — this is not a design gap, just a one-line omission at the HTTP
boundary. **Fix:**
1. Add `"faces"` to `_SETTINGS_KEYS` in `imageharbor/dashboard/server.py`.
2. Add validation for it in `_handle_settings` alongside the existing
   `"enrich" in body` boolean check (mirror that exact pattern — `"faces" in
   body and not isinstance(body["faces"], bool)` -> `errors["faces"] =
   "faces must be a boolean"`).
3. Confirm (read, don't guess) that `control.set_override("faces", value)`
   already round-trips correctly via `ControlPlane.set_override`'s existing
   `elif key == _FACES_KEY` branch — it should, since that branch already
   exists; if it doesn't, that's a separate bug to flag, not silently work
   around.
4. Add a test in `tests/test_dashboard_server.py` (matching the existing
   `POST /api/settings {"enrich": ...}` test pattern) that `POST
   /api/settings {"faces": true}` (or `false`) is accepted (200, not the
   "unknown setting(s)" 400) and that a subsequent read (`GET
   /api/settings` or `control.faces_enabled`) reflects it. This test must
   fail against the current `_SETTINGS_KEYS` tuple before the fix.
5. Since the toggle is now genuinely reachable, `watcher.py`'s comment
   about taking effect "on the very next cycle" is now true — leave it as
   is (no comment change needed once wired; only fix the comment if you end
   up choosing the delete-the-stanzas path instead, which is not the chosen
   path here).

Test files: `tests/test_dashboard_server.py`, plus whichever CLI test file
task 7a lands in.
Covering command: `uv run pytest tests/test_dashboard_server.py
tests/test_cli.py tests/faces/test_faces_cli.py -v`, then the full suite.

---

## Verification (final, after all 7 tasks)

- Every fix has a test that failed against the current code (RED) and passes
  after the fix (GREEN) — pasted verbatim, not retyped.
- Full suite without weights: `uv run pytest` — compare against baseline
  1144 passed, 11 skipped.
- Full suite with weights (if `IMAGEHARBOR_FACE_MODEL_DIR` is set in this
  environment): compare against baseline 1154 passed, 1 skipped.
- `git status --short` clean.
- One commit per finding, or a small number of coherent commits — not one
  giant commit.
