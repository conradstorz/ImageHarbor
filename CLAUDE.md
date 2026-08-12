# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ImageHarbor is

A deterministic, resumable CLI that organizes a photo library in **two passes**: a
fast, AI-free **facts pass** (`process`) that organizes every photo from EXIF and
filename evidence alone, and a resumable **enrichment pass** (`enrich`) that adds
AI description/classification afterward. Its three verbs — **Classify. Verify.
Preserve.** — still describe the system, but classification is now optional and
deferred:

- **Classify** — the AI backend only *perceives* the image (subject/scene/objects/
  caption/tags) during `enrich`; the organizer (`concept_map.py` + `taxonomy.py`)
  decides the PCS class/code from that perception. The PCS code lives in the
  catalog and the JSON sidecar, **not** in the folder path or the filename — see
  "Critical invariants" below.
- **Verify** — every file is content-addressed by SHA-256; the digest is embedded
  in the filename so any file can later be re-verified against its own name.
- **Preserve** — originals are treated as read-only. Files are *copied* (never
  moved/modified) into the organized tree, verified after copy, and recorded in a
  SQLite catalog. `process` alone (no `enrich` ever run) already produces a
  complete, verified, organized library — a permanently offline AI backend
  degrades nothing.

See [`docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`](docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md)
for the full design rationale behind this split.

## Commands

Python project managed with `uv` (see global CLAUDE.md — do not use pip/venv directly).

| Task | Command |
|------|---------|
| Install deps + dev tools | `uv sync --extra dev` |
| Add OpenAI classifier support | `uv sync --extra dev --extra openai` |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_pipeline.py` |
| Run one test | `uv run pytest tests/test_pcs.py::test_resolve_code_known` |
| Coverage | `uv run pytest --cov=imageharbor` |
| Run the CLI | `uv run imageharbor --help` |
| Organize a library (facts pass, no AI) | `uv run imageharbor process --source SRC --dest DEST` |
| Describe/classify the organized copies | `uv run imageharbor enrich --dest DEST --ai openai` |
| Re-verify integrity | `uv run imageharbor verify DEST` |
| Watch a library continuously (both passes) | `uv run imageharbor watch --source SRC --dest DEST` |
| Build the Docker image | `docker build -t imageharbor:latest .` |
| Run the watcher (compose) | `docker compose up -d` (see `docs/deploy-docker.md`) |
| Query the catalog | `uv run imageharbor catalog list --catalog DEST/catalog.db` |

`process` takes no `--ai`/`--ai-*`/`--breaker-*`/`--poison-*` flags and makes no
network call — those flags live on `enrich` (and on `watch`, which drives both
passes). There is no linter/formatter configured. `pyproject.toml` is the single
source of truth for deps, extras, pytest config, and the `imageharbor` entry point.

## Architecture

Single package `imageharbor/`, orchestrated by **two passes** instead of one linear
pipeline. Each pass has its own spine.

**Facts pass** (`pipeline.Pipeline._do_process`, driven by `imageharbor process`) —
no AI, no network call:

`hash → dedup (+ back-pointer, + duplicate upgrade) → EXIF → resolve date →
resolve descriptor → target path → copy → verify → catalog → sidecar`

**Enrichment pass** (`enrich.enrich_library`, driven by `imageharbor enrich`) —
reads the organized copy, resumable, AI-dependent:

`unenriched rows → describe → concept_map/pick_class → taxonomy resolve →
catalog → tier-gated rename → sidecar`

The facts pass is what makes a full library organize possible without an AI
backend at all: hashing, dedup, EXIF, and copying are pure local computation, and
placement/naming are decided from facts (EXIF + original filename) rather than
from AI perception. The enrichment pass adds description/classification later,
and can only ever *improve* a file — see the monotonicity invariant under
"Critical invariants" below for the rule that guarantees this.

Module responsibilities:

- **`pipeline.py`** — the facts-pass orchestrator above. It makes **no AI calls**.
  After hashing, it checks for a duplicate (recording a `sources` back-pointer and
  possibly upgrading the existing file's placement/name — see
  `_maybe_upgrade_from_duplicate`), reads EXIF, resolves a date
  (`date_resolver.resolve_date`) and a descriptor (`descriptor.resolve_descriptor`)
  purely from facts, computes the destination path (`relocate.target_path`), copies,
  verifies, upserts the catalog, records the source, and optionally merges a
  sidecar. Owns the `PipelineStats`/`ProcessResult` result types. If a post-copy
  integrity check fails, the copy is deleted and an error is raised — nothing
  enters the catalog unverified.
- **`tiers.py`** — pure, I/O-free module defining the two independent quality
  ladders (`DATE_*`, `DESC_*`) and the single predicate `is_upgrade(old, new)` that
  governs every rename in the system: a proposed `(date_tier, descriptor_tier)`
  must be strictly better in at least one dimension and worse in neither. Equal
  tiers are a no-op — this is what makes a repeated run idempotent.
- **`date_resolver.py`** — resolves a `ResolvedDate` (`value`, `tier`, `source`,
  plus derived `.date_str`/`.folder`) from EXIF first (`DateTimeOriginal` >
  `DateTimeDigitized`/`DateTime`), then from a date pattern in the original
  filename, else `Undated/`. **File mtime is never consulted** — it is deliberately
  absent from the ladder because it records when a file was copied, not when a
  photo was taken. `date_from_row` rebuilds a `ResolvedDate` from stored catalog
  columns so both passes can compare a freshly-resolved date against the one on
  record.
- **`descriptor.py`** — resolves a `ResolvedDescriptor` from the original
  filename's stem: a stem that doesn't match a `CAMERA_PATTERNS` entry (IMG_1234,
  DSC0042, Screenshot_…, WhatsApp Image …, etc.) is human-authored and gets
  `DESC_HUMAN_FILENAME` (tier 30); a camera-generated stem gets `DESC_NONE` (tier
  0) and waits for the AI enrichment pass to fill it at `DESC_AI_SUBJECT` (tier
  20).
- **`relocate.py`** — target-path computation (`target_path`) and safe
  in-tree relocation (`apply_relocation`, filesystem first then caller updates the
  catalog) plus digest-based self-healing (`find_by_digest`,
  `resolve_organized_path`) for a stale/missing recorded path — a moved or
  crash-interrupted file is never actually lost because it's content-addressed.
- **`enrich.py`** — the enrichment-pass orchestrator above (`enrich_library`).
  Iterates `catalog.iter_unenriched()` (or, with `--reclassify`, every enriched
  row), calls `classifier.describe()`, then the same
  `concept_map.class_for`/`pick_class`/`taxonomy.resolve_or_create` chain the old
  single pipeline used, writes classification to the catalog and sidecar
  **unconditionally**, but renames/moves the file **only when
  `tiers.is_upgrade` says so** — an AI subject (tier 20) can never displace a
  human-authored filename (tier 30). Drives the circuit breaker (an AI-perception
  failure feeds it; everything after `describe()` is local work and must not).
  Enrichment failure leaves a file at its current tier, which is always valid: the
  facts pass already gave it a real name and a real home.
- **`pcs.py`** — **seed data + helpers only**: `PCS_CATEGORIES` defines the 9 fixed
  top-level classes (100–900) and their original sub-codes, used once to seed the
  catalog `taxonomy` table on first run (`Taxonomy.ensure_seeded`). `resolve_code`
  (int → int, unknown → 900) is retained for legacy/tooling use, but nothing in
  the pipeline calls it — code assignment lives in `taxonomy.py`, and folder-path
  resolution for the organized tree lives in `date_resolver.ResolvedDate.folder`
  (see "Placement and naming"). `pcs.parent_folder_name`/`sub_folder_name` were
  **removed** and were never replaced by an equivalent — PCS no longer decides any
  filesystem path.
- **`taxonomy.py`** — the self-extending PCS taxonomy: a `Taxonomy` class backed by
  the catalog `taxonomy` table (append-only, never renumbered). The 9 top-level
  classes are fixed; growth happens beneath them. Codes are **strings** matching
  `^\d+(~\d+)*$` — plain integers for the common case (e.g. `"330"`), with a `~N`
  suffix minted when a parent's normal integer slots are exhausted or a leaf needs a
  child of its own (**never a dot**). `resolve_or_create(top_parent, label,
  sub_parent=None, adjudicator=None)` is how the pipeline turns a class + label pair
  into a code: it normalizes the label, checks the target parent's children for an
  exact/alias match, and — only when there's no exact match and an `adjudicator` is
  supplied — asks the AI classifier's `adjudicate(label, candidates)` whether the
  label is a synonym of one of the parent's *existing* children (semantic match, not
  a string-similarity/fuzzy-ratio gate — true synonyms are often string-dissimilar,
  e.g. "festivities" vs. "holidays", so a text-distance pre-filter would make the
  adjudicator unreachable for the case it exists to handle). A match records an alias
  and reuses the code; otherwise a new code is minted. `merge(from_code, to_code)`
  aliases one code onto another after the fact. `folder_path(code)` still walks
  `parent_code` links to build the slash-joined classification path, but that path
  **no longer decides where a file lives on disk** — `enrich.py` writes it into
  the sidecar's `classification.folder_path` field only, as a human-readable
  record of the PCS tree the file was filed under. Actual placement comes from
  `date_resolver.ResolvedDate.folder`. `snapshot_text()` renders the current
  taxonomy for the classifier prompt. `taxonomy.py` itself was not touched by the
  facts/enrichment split — `enrich.py` calls `resolve_or_create(class,
  primary_subject)` with a fixed top-level class and **no `sub_parent`**, so in
  practice the taxonomy is effectively **two levels** (fixed class →
  `primary_subject` sub-category); the `sub_parent`/`~N`-under-a-leaf machinery
  still exists but its call sites are currently unused.
- **`ai_classifier.py`** — perception only. `AIClassifier` ABC with two
  implementations chosen by the `--ai` flag: `StubClassifier` (default;
  deterministic, no network — derives a subject/tags from filename keywords, used
  by all tests) and `OpenAIClassifier` (optional, gated behind the `openai` extra
  and imported lazily). `describe(image_path, exif_data) -> ContentDescription`
  (`primary_subject`, `scene`, `objects`, `caption`, `tags`, `ocr_text`,
  `model_version`) is the only required method — **the classifier never picks a
  class or a PCS code; it only reports what it sees.** `PhotoClassification` and
  `classify()` are gone. Two more ABC methods support the organizer: `pick_class
  (content, classes) -> str` is a **text-only fallback** `enrich.py` calls only
  when `concept_map.class_for` misses (default: `"900"`; `OpenAIClassifier` asks
  the model to choose among the 9 fixed classes), and `adjudicate(label,
  candidates) -> str | None` (default: no match) lets a real-model backend decide
  whether a proposed label is a synonym of an existing sibling category —
  `OpenAIClassifier` implements it as a follow-up chat call. Add new backends by
  subclassing `AIClassifier` and wiring them in `cli.py`.
  **Design intent (important):** this abstraction exists so the *AI server doing the
  work is swappable*, not just the vendor. The project was inspired by a self-hosted
  AI server (a Jetson Orin Nano on the local network), but nothing is hard-wired to
  it — a local/Jetson HTTP backend is an expected future implementation that does not
  exist yet. Keep the classifier decoupled from any specific host or provider.
- **`concept_map.py`** — decides the top-level **class** (the organizer's job, not
  the AI's), called from `enrich.py`. `STATIC_SEED` is built once at import time
  from `pcs.PCS_CATEGORIES`' sub-category names plus a small curated
  keyword/synonym table, mapping normalized subject/object/scene tokens to one of
  the 9 fixed classes. `class_for(primary_subject, objects, scene, catalog)` checks,
  in order: the catalog's `learned_concepts` store (exact normalized-subject match),
  then the static seed against the subject, then against each object/scene token —
  returning `None` on a genuine miss. On a miss `enrich.py` falls back to
  `classifier.pick_class()` and calls `remember(catalog, primary_subject,
  class_code)` to memoize the decision in `learned_concepts`, so the next photo with
  the same normalized subject is a deterministic, network-free hit.
- **`circuit_breaker.py`** — a pure three-state (`CLOSED`/`OPEN`/`HALF_OPEN`)
  circuit breaker with no I/O, now scoped to the **enrichment pass only** — the
  facts pass has no AI backend to fail, so it never consults or feeds a breaker.
  `enrich`/`watch` drive it: after `--breaker-threshold` (default 5) consecutive
  AI-perception failures it trips, the enrichment pass aborts, and (in `watch`) the
  watcher backs off (`--breaker-backoff` 60s → ×2 → `--breaker-backoff-cap` 900s)
  before a half-open probe (one real image) re-tests the backend; the facts phase
  keeps running at full speed regardless. `trip_threshold=0` disables it. Only a
  failure raised by `classifier.describe()` feeds the breaker — a failure in the
  local work after perception (taxonomy resolution, catalog update, rename) must
  never masquerade as a backend outage.
- **`hashing.py`** + **`filename.py`** — content addressing. SHA-256 is encoded as
  **unpadded Base64url, always exactly 43 chars** (`SHA256_B64URL_LEN`). Filename
  format is `[<YYYY-MM-DD>][-<descriptor>]_<sha256>.<ext>` — both prefix
  components are optional; with neither, the stem is the bare digest (e.g. an
  `Undated/` file with no human-authored name). `hashing.extract_digest_from_stem`
  locates the digest by counting `SHA256_B64URL_LEN` characters back from the end
  of the stem and validates it against the Base64url character class — it does
  **not** validate a PCS prefix (there is none anymore); legacy
  `<pcs>-<descriptor>_<digest>` stems from before this redesign still parse
  unchanged, since the prefix is otherwise unconstrained.
- **`catalog.py`** — SQLite (WAL mode). The `photos` table (keyed by the unique
  `sha256_b64url`) is the source of truth for **resumability and duplicate
  detection** (`is_known`); `upsert` is idempotent (`ON CONFLICT … DO UPDATE`) and
  list/dict fields are stored as JSON text. Additive columns
  (`date_value`/`date_tier`/`date_source`, `descriptor_value`/`descriptor_tier`/
  `descriptor_source`, `enriched_at`, `scene`) back the tier system and the
  enrichment work queue — `iter_unenriched()` selects `enriched_at IS NULL` rows
  (excluding quarantined content) in id order; `--reclassify` uses `iter_all()`
  instead. `set_placement()` updates the organized path plus date/descriptor tiers
  after a tier-gated rename; `mark_enriched()` stamps the AI perception fields and
  `enriched_at`. A `sources` table (new) is the many-to-one back-pointer set
  keyed `(sha256_b64url, source_path)` that replaces a single `original_path` for
  dedup: three copies of one photo across three exports yield one organized file
  and three `sources` rows (`record_source`, `sources_for`); `photos.original_path`
  is retained as the *first* source seen, for backward compatibility. A `taxonomy`
  table persists the self-extending PCS registry (`code`, `parent_code`, `label`,
  `folder_name`, `aliases`, `alias_of`, `active`), backing `taxonomy.py`. A
  `learned_concepts` table (`subject`, `class_code`, `hits`, timestamps) is the
  self-learning store behind `concept_map.py`'s
  `learned_concept_get`/`learned_concept_remember`. A `failed_files` table
  (`source_path`, `size`, `mtime_ns`, `fail_count`, `last_error`, `quarantined`,
  timestamps) backs **enrichment-pass-only** poison-file quarantine: a file that
  fails `--poison-max-fails` (default 5) *healthy* enrichment passes is
  quarantined — meaning "stop asking the model about this one," not "set the file
  aside": it stays fully organized, verified, and cataloged by the facts pass, and
  is excluded from `iter_unenriched()` until its bytes change. Failures during a
  breaker-tripped outage never count, so a backend outage cannot mis-quarantine
  good files. Quarantine requires a *healthy* pass to observe the failure —
  see "Known limitations" below for the accepted boundary this creates when
  poison files constitute the entire remaining queue.
- **`discovery.py`** — yields supported image files (see `SUPPORTED_EXTENSIONS`);
  supports single-file or recursive directory mode and never mutates the source.
- **`exif_reader.py`** — best-effort EXIF/GPS extraction via Pillow; returns `{}`
  rather than raising on any failure.
- **`sidecar.py`** — optional, per-image `.json` metadata file (via `--sidecar`),
  now **cumulative** rather than write-once: `merge_sidecar(organized_path,
  updates)` reads the existing sidecar (if any), deep-merges nested dicts key by
  key (`_deep_merge` — lists and scalars replace wholesale, so a caller that owns
  a list, e.g. `sources`/`history`, must pass the complete value), and writes back
  atomically (temp file + `os.replace`). Unknown keys — including hand edits — are
  preserved across runs. The facts pass merges `identity`/`sources`/`date`/
  `descriptor`/`exif`; the enrichment pass later merges `classification`. There is
  no standalone `write_sidecar` anymore — `merge_sidecar` is the only entry point.
- **`cli.py`** — Click entry point (`process`, `enrich`, `watch`, `verify`,
  `catalog list/get`).

## Critical invariants — do not break these

- **PCS codes are strings, not ints, and are owned by the taxonomy, not the
  classifier.** A valid code matches `^\d+(~\d+)*$` — plain integers for the common
  case (`"330"`), or a `~N` suffix appended when a parent's decimal slots are used up
  or a leaf needs a child (**never a dot**, e.g. `"330~1"` not `"330.1"`). Codes are
  minted append-only by `taxonomy.Taxonomy` and persisted in the catalog `taxonomy`
  table, seeded once from `pcs.PCS_CATEGORIES` (now seed data only). The **9
  top-level classes are fixed**; all growth happens beneath them, and existing codes
  are never renumbered or reused.
- **The classifier perceives; it never classifies.** `AIClassifier.describe` returns
  a `ContentDescription` (subject/scene/objects/caption/tags/ocr_text) and nothing
  taxonomy-shaped. Deciding the top-level **class** is `concept_map.class_for`'s job
  (static seed + learned store); the classifier is only consulted as a **text-only
  fallback** (`pick_class`) on a concept-map miss, and that answer is immediately
  memoized via `concept_map.remember` so the same subject never asks the AI twice.
  Deciding the **code** for a (class, label) pair is `Taxonomy.resolve_or_create`'s
  job — by reusing an exact/alias match among the target parent's existing children,
  or, failing that, consulting `AIClassifier.adjudicate(label, candidates)` for a
  semantic synonym match before minting a new one. Do not add
  string-similarity/fuzzy-ratio gating in front of the adjudicator — it was tried and
  dropped because true synonyms (e.g. "festivities" vs. "holidays") are
  string-dissimilar, which made the adjudicator unreachable.
- **Placement comes from `date_resolver.ResolvedDate.folder`**, not from the PCS
  taxonomy. A file lands at `YYYY/YYYY-MM/` when a trustworthy capture date is
  known (EXIF, then a date pattern in the original filename), or `Undated/`
  otherwise — never a guessed year. PCS lives in the catalog and in the sidecar's
  `classification` block, not in the path or the filename. `taxonomy.folder_path
  (code)` still exists and is still called, but only to record a human-readable
  classification path *inside the sidecar* — it does not decide where any byte on
  disk lives. Do not reintroduce PCS-driven placement.
- **A file is renamed or moved only when `tiers.is_upgrade(old, new)` returns
  True** — the proposed `(date_tier, descriptor_tier)` must be strictly better in
  at least one dimension and worse in neither. Equal tiers are a no-op, which is
  what makes a repeated run idempotent. An AI subject (`DESC_AI_SUBJECT`, tier 20)
  can therefore never displace a human-authored filename (`DESC_HUMAN_FILENAME`,
  tier 30) — this is enforced structurally by the predicate, not by care at any
  call site. Never add a code path that renames or relocates a file
  unconditionally. **`--reclassify` does NOT bypass this predicate** — it only
  bypasses the *work queue* (`iter_all()` instead of `iter_unenriched()`, so
  already-enriched rows are revisited); `is_upgrade` still gates the rename for
  every row it walks, so re-running `--reclassify` on an already-enriched photo
  re-records classification (catalog + sidecar) but is a guaranteed rename
  no-op, because the existing row already ties the new answer at
  `DESC_AI_SUBJECT`. See `tests/test_monotonicity.py` for the pinned behavior.
- **File mtime must never enter the date ladder.** `date_resolver.py` deliberately
  never reads `stat().st_mtime`: mtime records when a file was copied, not when a
  photo was taken, and asserting a date we can't support is exactly the quiet
  corruption the project's SHA-256 discipline exists to prevent.
- **The 43-char digest is located by counting back from the end of the stem, NOT by
  splitting on the last `_`.** Base64url legitimately contains `_`, so a naive rsplit
  corrupts parsing. This logic is duplicated in `hashing.extract_digest_from_stem`
  and `filename.parse_filename` — keep them in sync. `extract_digest_from_stem` no
  longer validates a PCS prefix (there is none in the current filename grammar); it
  validates the extracted 43 characters against the Base64url character class
  instead, and otherwise leaves the prefix unconstrained — which is also why
  legacy PCS-prefixed filenames from before this redesign still parse.
- **Non-AI failures must never feed the circuit breaker.** Only a failure raised
  by `AIClassifier.describe()` (AI-perception evidence) counts toward
  `--breaker-threshold`; a filesystem fault (a missing organized file, a failed
  rename, a permissions error) must be handled and counted separately, never
  reported to the breaker as if it were backend-outage evidence. The facts pass
  makes no AI calls at all, so it never touches the breaker.
- **Content addressing must stay stable.** The digest is over raw file *bytes*
  (streamed in 64 KiB chunks). Changing the hash algorithm, the Base64url encoding,
  or the 43-char length assumption breaks every existing filename and catalog key.
- **Originals are read-only; the organized copy is verified before it is cataloged.**
  Preserve this copy → verify → catalog ordering when editing the pipeline.
- **Runtime output directories are git-ignored, not source** (`Photos-Organized/`,
  `Review/`, `Duplicates/`, `Logs/`, `catalog.db`, etc. in `.gitignore`).

## Known limitations

- **Poison-quarantine cannot fire when the poison IS the entire remaining
  queue.** `watcher._reconcile_poison` only counts a failure toward
  `--poison-max-fails` during a *healthy* enrichment pass (>=1 success,
  breaker not tripped this pass) — see `catalog.py`'s `failed_files`
  description above. `watch()`'s rotating probe offset (implemented in
  `watcher.py`, using `catalog.iter_unenriched`'s `offset` parameter — the
  breaker itself is unaware of it) lets a half-open probe skip a stuck head
  cluster and find a working file elsewhere in the queue, so quarantine can
  fire for poison files that have *some* describable file anywhere else in
  `iter_unenriched`. But if the files tripping the breaker are (or become)
  the *whole* unenriched queue — no describable file left to probe into —
  no pass can ever be simultaneously non-tripped and contain them, so they
  stay permanently un-quarantined in that state. This is accepted, not a
  bug: an all-poison remaining queue is information-theoretically
  indistinguishable from a real backend outage, and quarantining anyway
  would risk condemning an entire library during a genuine outage — exactly
  what the tripped/no-success discard rules exist to prevent. The cost is
  bounded (one half-open probe per backoff interval, capped at
  `--breaker-backoff-cap`, 900s by default) and self-resolving (any new
  describable file — a fresh photo, a poison file whose bytes change —
  lets normal quarantine accounting resume). `watch()` logs one diagnostic
  warning per occurrence via `CONSECUTIVE_ABORT_WARNING_THRESHOLD` if this
  persists, so it is visible rather than silent. See the "Known, deliberate
  limitation" note on `watcher._reconcile_poison` for the full reasoning,
  and `tests/test_poison.py::test_poison_at_the_head_does_not_halt_enrichment`
  for the test that exercises the boundary (it asserts quarantine only for
  poison files that DO have a good neighbour elsewhere in the queue).

## Origin & design reference

Scaffolded with help from ChatGPT (which wrote the genesis design document) and
GitHub Copilot (first-draft implementation). Expect code you did not write; read the
current tree before changing it.

The genesis roadmap — **"Jetson Photo Workflow Roadmap (Rev. 2)"**, committed at
[`docs/genesis-roadmap.md`](docs/genesis-roadmap.md) — is now a **historical
document** (see the note at its top). It predates the *ImageHarbor* name and
describes the original PCS-driven folder tree and PCS-prefixed filenames, which
were superseded on 2026-08-11 by the date-derived tree and the
`[<date>][-<descriptor>]_<digest>` filename grammar documented above. Its
integrity, immutability, and resumability requirements (immutable originals,
resumable processing, full-SHA-256 duplicate detection, filename-alone integrity
verification) are unchanged and still authoritative. The current authoritative
design document for the two-pass architecture is
[`docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`](docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md).
