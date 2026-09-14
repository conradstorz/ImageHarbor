# Quality Roadmap

Addresses every finding from the 2026-09-12 full-project quality review
(four independent review passes: architecture, tests, hygiene, robustness &
security; suite at the time: 1,192 passed / 19 skipped / 93% coverage).
Sequenced as five releases. Release 1 comes first by design: it establishes
the CI/CD gate and the versioning system that every later release ships
through.

**Process:** each release is executed as its own implementation plan under
`docs/superpowers/plans/` (written with the `writing-plans` skill, executed
subagent-driven). This document owns ordering and scope, not step-level
detail.

**Versioning policy (established in Release 1, governs everything after):**

- Version is derived from git tags via `setuptools-scm` — single source of
  truth, no hand-edited version strings anywhere.
- **Every merge to `main` produces a release.** A CI workflow tags the next
  patch version automatically after the test/lint/type gate passes; a
  `[minor]` or `[major]` token in the merge-commit subject overrides the
  bump size. Small and large revisions alike get a real, immutable version.
- Between tags, local builds get PEP 440 dev versions
  (`1.0.1.dev3+g<sha>`) for free from `setuptools-scm` — no collisions, no
  ambiguity about what's deployed.
- Each tag publishes a GitHub Release with generated notes and a Docker
  image tagged `vX.Y.Z` + `latest`.

---

## Release 1 — v1.0.0 "Foundation" (CI/CD, versioning, cleanups)

Shipped: v1.0.0 / v1.0.1 (2026-09-13)

The first official release. v1.0.0 rather than v0.2.0: the tool is in daily
production use on hpz440 and the review graded the engineering A− — the
0.1.0 label was the fiction, not the maturity.

### 1a. Codebase cleanups (land before the first tag)

- `git rm -r --cached imageharbor/__pycache__` (11 stale tracked `.pyc`).
- Move `imageharbor/takeout-20230618T004316Z-001.zip` (79 MB personal data)
  out of the package directory; add `*.zip` to `.dockerignore`.
- Fix `docs/deploy-docker.md:151` — People panel port 8080 → 8087.
- Remove dead code: `Taxonomy.snapshot_text` (test-only),
  `CircuitBreaker.is_half_open` (zero references). Deferred-issues item #7
  (`ai_classifier._build_pcs_list`, `pcs.resolve_code`) is reviewed at the
  same time — keep or cut with one decision.
- Dedupe the three `_now_iso()` copies and two `_json_default()` copies
  into one shared helper each (removes two "keep in sync" comments).
- Decide `.superpowers/sdd/progress.md` (59 KB tracked agent state):
  git-ignore or delete.

### 1b. Lint + type enforcement

- Add `[tool.ruff]` (lint + format) and `[tool.mypy]` to `pyproject.toml`;
  fix what they flag. The codebase is at 99% return-annotation coverage —
  this is one config block plus a modest cleanup, not a campaign.
- Type the 17 unannotated parameters (clustered in `faces/runner.py` and
  `sqlite3.Row` params); add a `Protocol` for detector/embedder and a
  frozen dataclass for the `(det, embedding, model, reject_reason)` record
  tuple.

### 1c. CI

- `.github/workflows/ci.yml`: `uv sync --extra dev` → ruff → mypy →
  `uv run pytest`, on a matrix of **Windows + Ubuntu** (the suite has never
  run on POSIX — the review flagged this) across the Python versions
  `pyproject.toml` declares.
- Add `filterwarnings = ["error::ResourceWarning"]` to pytest config — the
  suite already surfaces an unclosed-database warning at
  `catalog.py:336`; make it a failure and fix the leak.
- Set the env flag that makes
  `tests/test_takeout_index_equivalence.py` **fail** rather than silently
  degrade when the sibling-repo oracle is absent — on Conrad's machine and
  in CI the oracle must be real or the test must say it isn't.

### 1d. Versioning + release automation

- `dynamic = ["version"]` + `setuptools-scm`; delete the version literal
  from `pyproject.toml` and have `imageharbor/__init__.py` read
  `importlib.metadata.version("imageharbor")`.
- `release.yml`: on push to `main` with CI green → compute bump (patch
  default, `[minor]`/`[major]` override) → tag → GitHub Release with
  generated notes.
- Start `CHANGELOG.md` (Keep-a-Changelog format); the release workflow
  appends the generated notes per tag.
- Fill package metadata: `authors`, `classifiers`, `keywords`,
  `[project.urls]`, and the SPDX string form
  `license = "AGPL-3.0-or-later"`.

### 1e. CD (Docker)

- Dockerfile builds from the lockfile (`COPY uv.lock` +
  `uv sync --frozen --extra openai --extra faces`) instead of
  `pip install` against loose floors — production ships what was tested.
- `COPY LICENSE` into the image (AGPL compliance for a network-served app).
- `docker.yml`: build and push to GHCR on every tag, `vX.Y.Z` + `latest`.

### 1f. Onboarding docs

- README gains **Install / Quickstart / Running the tests** (lifted from
  CLAUDE.md), the missing commands (`catalog list/get`,
  `faces models download`), and a short `CONTRIBUTING.md` (uv-only policy,
  how CI gates merges, the versioning tokens).

**Exit criterion:** `v1.0.0` tag exists; CI is the merge gate; the next
merge after it auto-tags `v1.0.1` with no human action.

---

## Release 2 — Security hardening (High + Med-High findings)

Shipped: v1.1.0

- **Dashboard exposure (High).** Bind `127.0.0.1` by default; add
  `--dashboard-host` / `IMAGEHARBOR_DASHBOARD_HOST` (compose sets it to
  `0.0.0.0` explicitly, inside a container that publishes to the tailnet
  only). Shared-secret token (env-provided) required on every `POST`
  route; `Host`-header allowlist; `timeout = 10` on the handler class
  (closes the slowloris/unbounded-thread DoS). Document the threat model
  in `docs/deploy-docker.md` — currently the only risky decision in the
  project with zero written rationale.
- **Zip-slip via backslash, Windows (Med-High).** Add `\\` to
  `_ILLEGAL_NAME_CHARS` in both `takeout/archive.py` and
  `takeout/provenance.py`; add a
  `dest.resolve().is_relative_to(room.resolve())` containment assertion
  before every write; regression tests using the verified
  `Takeout/..\..\..\pwned.txt` member shape.
- Update the hpz440 deployment after release (env var + compose change).

**Exit criterion:** an unauthenticated `POST /api/pause` from a non-listed
host returns 401/403; the traversal member lands sanitized inside the room
in a test that fails on the old code.

---

## Release 3 — Integrity & failure-handling corrections

Shipped: v1.2.0

- **`pick_class` breaker leak.** Wrap `classifier.pick_class()` (and audit
  `adjudicate`'s path) in `enrich.py` so a backend failure there is
  classified as AI evidence and feeds the breaker — today it lands in
  `io_failed` and a dead backend churns the whole queue.
- **fsync before verify.** `os.fsync` the copied file before
  `verify_file` in `pipeline.py`; same for sidecar and provenance writes
  or an explicit documented exemption. Closes the power-loss gap where the
  catalog durably asserts "verified" for uncommitted pages.
- **Model download recovery.** On `ChecksumMismatch`, unlink the bad
  artifact before raising (today it wedges permanently); `.part` cleanup
  in a `finally`.
- **Staging sweep.** Clear `.takeout-staging/` leftovers at `ingest.run()`
  start — orphans from a `kill -9` currently accumulate inside the
  backed-up library.
- **Sidecar descriptor staleness** (deferred-issues #9): the enrichment
  pass merges a `descriptor` update alongside `classification` whenever
  `is_upgrade` fires a rename; test that a renamed file's sidecar agrees
  with its own filename.
- **`enrich_failed` counter split** (deferred-issues #13): record
  `ai_failed` and `io_failed` separately in the `runs` row so the
  dashboard can tell an outage from a filesystem fault.

---

## Release 4 — Test-debt paydown

Shipped: v1.2.1

- Add `tests/conftest.py`; consolidate the duplicated fixtures (`catalog`
  ×14, `organized_dir` ×7, `store`/`source_dir`/`face_store` ×4) and
  remove the cross-module fixture imports.
- Test `exif_reader.py:149–182` — the entire GPS DMS→decimal walk,
  hemisphere signing, rational coercion, and both silent `except` arms
  (68% coverage today; classic quiet-shipping territory).
- Test `watcher.py`'s faces-pass crash handler (815–890) — the
  survives-a-crash guarantee is proven for the other two passes only.
- ONNX boundary (0% on `detect.py`/`embed.py`): a weights-cached CI lane
  (models are already checksum-pinned) or recorded-tensor fixtures, so a
  shape/dtype/output-order mismatch against real weights can't pass.
- Replace the tautological
  `tests/faces/test_runner.py::test_rejected_face_reasons_are_distinguishable`
  with one that can actually detect a reason collapse.
- `stats.py` lock regression test (deferred-issues #11): drive
  `stats.collect()` under writer load so deleting the three
  `with catalog.lock:` sites fails a test.

---

## Release 5 — Structural debt

Shipping as v1.3.0

Largest-risk refactors last, behind the CI gate the earlier releases built.

- **Split `Catalog`** (~60 methods, 8 domains) along the ownership lines
  `FaceStore` already demonstrates: extract takeout persistence and
  taxonomy persistence first (cleanly separable per the review), then
  runs/settings if it still earns it.
- **Read-side aggregation API** on the stores, eliminating the eight
  documented `catalog._conn` / `store._conn` reach-ins from `dashboard/`.
- **Extract the three oversized functions:** `_survey` (322 lines — the
  `# SECOND PASS` block becomes its own function), `watch` (268 — the
  inline faces pass becomes `_run_faces_pass`), `enrich_library` (208).
- Remove `click` from `faces/runner.py` (domain exception translated at
  the CLI boundary).
- **`projections.py` `Unreadable` sentinel** (deferred-issues #10, called
  the highest-value follow-up there): distinct from real `None`/`0` at
  every parse site, turning the next conflation defect into a type error.
- Trim changelog-style comment archaeology (`watcher.py:511–524`,
  `catalog.py:225–238`, etc.) now that `CHANGELOG.md` exists to hold it.

---

## Tracked but not scheduled

From the deferred-known-issues list, unchanged in status by this roadmap
(deliberate deferrals, not oversights): #2 non-deterministic-backend
orphans, #3 `_getexif()` deprecation, #4 lossy singularization,
#5 adjudicate candidate growth, #6 `Taxonomy` single-writer, #8
learned-concepts poisoning (wants dashboard tooling), #12 `runs` table
retention. Also accepted by the review: the `apply_relocation` TOCTOU
(single-writer deployment) and the all-poison-queue quarantine boundary.
