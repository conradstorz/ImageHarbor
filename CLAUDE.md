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
| Add face recognition | `uv sync --extra dev --extra faces` |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_pipeline.py` |
| Run one test | `uv run pytest tests/test_pcs.py::test_resolve_code_known` |
| Coverage | `uv run pytest --cov=imageharbor` |
| Run the CLI | `uv run imageharbor --help` |
| Organize a library (facts pass, no AI) | `uv run imageharbor process --source SRC --dest DEST` |
| Describe/classify the organized copies | `uv run imageharbor enrich --dest DEST --ai openai` |
| Ingest Google Takeout archives | `uv run imageharbor takeout ingest --archives DIR --dest DEST` |
| Report Takeout ingestion progress | `uv run imageharbor takeout status --catalog DEST/catalog.db` |
| Survey an archive set before ingesting (read-only, standalone) | `uv run imageharbor takeout survey --archives DIR --json report.json` |
| Re-verify integrity | `uv run imageharbor verify DEST` |
| Watch a library continuously (both passes) | `uv run imageharbor watch --source SRC --dest DEST` |
| Build the Docker image | `docker build -t imageharbor:latest .` |
| Run the watcher (compose) | `docker compose up -d` (see `docs/deploy-docker.md`) |
| Query the catalog | `uv run imageharbor catalog list --catalog DEST/catalog.db` |
| Rebuild sidecars from the catalog (cannot recover Google Takeout metadata — see `sidecar backfill` below) | `uv run imageharbor sidecar backfill --dest DEST` |

`process` takes no `--ai`/`--ai-*`/`--breaker-*`/`--poison-*` flags and makes no
network call — those flags live on `enrich` (and on `watch`, which drives both
passes). There is no linter/formatter configured. `pyproject.toml` is the single
source of truth for deps, extras, pytest config, and the `imageharbor` entry point.
`--sidecar/--no-sidecar` defaults to **on** (`default=True`) on `process`,
`enrich`, `watch`, and `takeout ingest` — `--no-sidecar` is the opt-out, not the
default; nothing else needs to be remembered.

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
  enters the catalog unverified. `process_file`'s `source_label` overrides what
  gets recorded as the logical source in `sources` (Takeout ingestion passes
  `"<archive>!<member path>"`, since the real source is a zip member, not the
  staged file's own path). `ExternalEvidence` (`date`, `original_name`) is the
  parameter object for facts a caller obtained elsewhere — in practice Google
  Takeout's per-media JSON — that `Pipeline` unpacks into the two resolvers
  rather than passing down, so neither resolver learns anything about Takeout;
  `date` feeds `date_resolver.resolve_date`'s `external_date` and
  `original_name` feeds `descriptor.resolve_descriptor`'s `original_name`.
  Google's `creationTime` must never be placed in `ExternalEvidence.date` — it
  is upload time, not capture time. `consume_source=True` (used only by
  Takeout ingestion, on a staging file the caller owns and considers
  disposable) changes the copy → verify → catalog ordering described under
  "Critical invariants" below to rename → verify → catalog — verification still
  reads the file at its destination, so nothing enters the catalog unverified
  either way.
- **`takeout/`** — Google Takeout archive ingestion, a third entry point into the
  facts pass (`imageharbor takeout ingest`). Two phases: a **survey** that reads
  only zip central directories (no decompression) and builds ONE pairing index
  across every archive in the batch, then a resumable **ingest** that extracts
  one member at a time into `<dest>/.takeout-staging/` and hands it to
  `Pipeline.process_file`. The survey itself is two passes: pass one records
  every `complete` archive's member paths from `takeout_members` (no zip
  reopened) so a batch that already finished still contributes to the pairing
  index; pass two then reopens a `complete` archive — and only that archive —
  when the freshly-built index newly resolves a sidecar for one of its members
  (an image/video with no `sidecar_path` on record), returning that member to
  `pending` so it re-ingests as a duplicate through
  `_maybe_upgrade_from_duplicate` and upgrades in place, handling the
  photos-first-then-sidecars-later ordering with no new placement code. Archives
  are opened `'r'` only and are never modified. Makes **no AI calls**, so —
  exactly like the facts pass — it never consults or feeds the circuit breaker.
  Videos are enumerated and recorded as `deferred` with their capture date but no
  bytes are copied; video ingestion is a deliberate later project. The global
  (not per-archive) pairing index is load-bearing: Google's multi-part zips split
  by size across the file list, so a photo and its `.json` routinely land in
  different parts.
  `ingest.py`'s `_index_albums` (reads `Albums.json` into an
  `(archive_id, folder) -> AlbumMetadata` map, activating
  `metadata.parse_album_metadata`) and `_preserve_provenance` (calls
  `provenance.preserve`) both run from **`_ingest_archive`, not `_survey`** —
  `_survey` never reopens a `complete` archive's zip except for the narrow
  late-sidecar-discovery case described above, and orphan detection needs the
  whole-batch pairing index, which isn't finished building until `_survey`
  returns; `_ingest_archive` is the point where a zip handle for that specific
  archive is already open for the work it's about to do anyway. Seven
  modules: `metadata.py` (pure Google-JSON parser, never raises; timestamp
  parsing uses epoch + timedelta arithmetic instead of `datetime.fromtimestamp`
  to remain platform-independent for pre-1970 dates — the latter raises `OSError`
  on Windows while succeeding on Linux, which meant the same archive produced
  different capture dates depending on host OS), `pairing.py`
  (pure media→sidecar matcher that returns `None` rather than guess),
  `archive.py` (identity, enumeration, classification, extraction), `ingest.py`
  (orchestration — the primary module with side effects), `provenance.py`
  (preserves non-media members verbatim, described below), `survey.py` (read-only
  measurement of an archive set -- two passes: central directories to build the
  whole-batch pairing index, then a reopen to sniff members whose extension is
  unrecognized and read per-media sidecars), and `report.py` (pure: turns a
  collected inventory into the report document, split from `survey.py` for the
  same reason `projections.py` is split from `stats.py`).
- **`takeout/provenance.py`** — preserves every archive member that is **not**
  an image or video, verbatim, under
  `<organized_dir>/.takeout-provenance/<archive_id>/` (keyed by the SHA-256 of
  the zip's own bytes, so a renamed/moved archive resolves to the same room).
  The rule is deliberately **uncurated: preserve everything that is not
  media** — deciding which unknown file is "worth" keeping is precisely where
  "never lose" degrades into "lose the thing nobody thought about", so
  `archive_browser.html` (Google's offline viewer, ~169 KB per archive) is kept
  for the same reason the Picasa face-tags file is. `manifest.json` records a
  digest per preserved document; a member whose digest already matches what's
  on record is skipped, which is what makes re-preserving the same archive a
  no-op. `preserve()` takes an `orphaned` set of member paths — media-JSON
  sidecars with no media member anywhere in the batch — computed by the caller
  from the whole-batch pairing index, and files those under `orphaned/`
  instead of their normal member path.
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
  record. `resolve_date`'s `external_date` keyword populates the
  `DATE_EXTERNAL_SIDECAR` (30) rung, which sits below EXIF `DateTimeOriginal` and
  above `DateTimeDigitized`/`DateTime` — in practice Google Takeout's
  `photoTakenTime`. Google's `creationTime` is deliberately excluded for the
  same reason mtime is: it records upload time, not capture time.
- **`descriptor.py`** — resolves a `ResolvedDescriptor` from the original
  filename's stem: a stem that doesn't match a `CAMERA_PATTERNS` entry (IMG_1234,
  DSC0042, Screenshot_…, WhatsApp Image …, etc.) is human-authored and gets
  `DESC_HUMAN_FILENAME` (tier 30); a camera-generated stem gets `DESC_NONE` (tier
  0) and waits for the AI enrichment pass to fill it at `DESC_AI_SUBJECT` (tier
  20). `resolve_descriptor` takes two optional keyword parameters for callers
  with better evidence than the path itself: `original_name` (e.g. Google
  Takeout's `title`, the pre-truncation name of a member whose stem the export
  truncated) supplies a **better spelling** of the name, not a vote that a
  human authored it — a camera verdict from **either** name wins, and the
  title's spelling is preferred only once both names have passed the camera
  check. This matters because Google's `title` keeps characters the zip
  member name had to sanitize for the filesystem, so the two can differ in
  exactly the characters a pattern anchors on; `date_str` (the `YYYY-MM-DD`
  the date ladder actually resolved) is compared against the normalized
  descriptor, and a descriptor that merely restates the date is discarded as
  `DESC_NONE` — the folder and the filename's date prefix already say it, so
  keeping it would state the same fact twice. `CAMERA_PATTERNS` gained two
  entries for the Takeout branch: a bare `YYYY-MM-DD(N)?` date (a date is not
  a description) and a Hangouts/AlbumArchive row id of the form
  `\d{10,}[\W_]?account[\W_]?id[\W_]?\d+` (a Google Takeout filename shape,
  not human intent).
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
  A `takeout_archives` table (`archive_id` = SHA-256 of the zip's own bytes,
  `status` partial|complete|corrupt) and a `takeout_members` table
  (`(archive_id, member_path)`, with terminal statuses `ingested`/`duplicate`/
  `deferred`/`parsed`/`ignored`/`skipped_trash` and non-terminal `pending`/
  `failed`) back Takeout ingestion's four idempotency layers. Both are purely
  additive, so `SCHEMA_VERSION` stays `"2"` and an existing catalog upgrades in
  place.
  Two more additive tables back the operational dashboard (`dashboard/`,
  below), also without bumping `SCHEMA_VERSION`: a `runs` table (`id`, `kind`
  'facts'|'enrich', `started_at`, `ended_at` NULL while a pass is in flight or
  was interrupted by a crash, `scanned`/`copied`/`duplicates`/`errors`/
  `enriched`/`enrich_failed`, `breaker_state`, `paused`) is one row per pass,
  inserted at start and updated at end, and is the sole evidence
  `dashboard/projections.py` reasons from; and a `settings` table (`key`,
  `value`, `updated_at`) holding at most three rows — `paused` ('0'/'1', no
  env counterpart), `interval` (seconds), `enrich` ('0'/'1') — where a
  present row overrides the corresponding env var and an absent one means
  "follow config". **`Catalog.lock`** (a public, reentrant `threading.RLock`)
  guards every public `Catalog` method — added 2026-08-19 (final
  whole-branch-review finding, pre-merge) after `cli.py`'s single shared
  `Catalog` was found to be reached concurrently by the dashboard's
  `daemon_threads=True` HTTP server and the watcher loop, which
  `check_same_thread=False` *permits* but does not make *safe*: measured
  under load, 55 raw `sqlite3`/`SystemError` exceptions out of the writer in
  25 seconds. `RLock`, not `Lock`, because several methods call other
  guarded methods on the same object from the same thread (e.g. `upsert`
  calls `get_by_sha256`). A second connection (one for the dashboard, one
  for the watcher) was considered and rejected: `ControlPlane` is read from
  *both* threads, so splitting the connection would still leave that seam
  unguarded — the lock is the smaller change that actually closes the gap.
  `dashboard/stats.py`'s three sections that run aggregate SQL directly
  against `catalog._conn` (no `Catalog` wrapper method covers them) acquire
  this same lock around their query blocks.

  **Corrected 2026-08-19** (this section previously described a
  two-connection architecture — "the dashboard writes settings rows from its
  own connection while the watcher writes photo rows from this one" — that
  was never actually implemented; the spec's Concurrency section made the
  same claim and has been corrected too): there has only ever been **one**
  connection, now guarded end-to-end by `Catalog.lock` as described above.
  `__init__` still sets `PRAGMA busy_timeout=5000` on that one connection,
  but it governs contention *between separate SQLite connections* — with
  only one connection in the process, sharing one `sqlite3.Connection`
  object under one Python-level lock, this pragma is **inert for the
  current topology**: two threads never reach SQLite concurrently in the
  first place, so there is nothing for SQLite's own busy-wait to arbitrate.
  It is kept anyway as an explicit pin at the point of use, for the day a
  second connection is added (e.g. as a pure read-side optimization) and
  this contention becomes real again.
- **`discovery.py`** — yields supported image files (see `SUPPORTED_EXTENSIONS`);
  supports single-file or recursive directory mode and never mutates the source.
  Also defines `VIDEO_EXTENSIONS`, for **classification only** — `discover_images`
  still yields images and nothing else; video ingestion is a separate, later
  project. Takeout ingestion (`takeout/archive.py`) uses `VIDEO_EXTENSIONS` to
  enumerate videos and record them as `deferred` with a capture date, so that
  later project starts from a complete work queue rather than from zero, but no
  video bytes are ever copied by any current code path.
- **`content_type.py`** — pure, I/O-free identification of media by magic
  bytes: `sniff(head) -> "image" | "video" | None` and
  `canonical_extension(head)`. **The extension stays the first rung**; only a
  file whose extension is unrecognized is read and sniffed, so no existing
  classification changes. It exists because a real Google Takeout export
  delivers thousands of genuine photographs under extensions like `.screen`,
  `.tile`, and `.tmp`. Currently called only by `takeout/survey.py` — the
  pipeline does not consult it yet.
- **`exif_reader.py`** — best-effort EXIF/GPS extraction via Pillow; returns `{}`
  rather than raising on any failure.
- **`sidecar_schema.py`** — the sidecar merge policy, pure and I/O-free (no
  filesystem, no import from the rest of the package — the same split that made
  `takeout/metadata.py` and `takeout/pairing.py` exhaustively testable). One
  rule governs it: **a sidecar may gain information and may never lose any.**
  `merge(base, updates, *, observed_at)` is total (never raises) and returns a
  document containing every value present in either argument — a superseded
  tiered/versioned block value is relocated into that block's `history[]`
  rather than overwritten, a changed flat-map key moves its old value to its
  own history list (`exif` → `exif_history[]`, `identity` →
  `identity_history[]` — table-driven via `FLAT_MAP_HISTORY_KEYS`, not
  special-cased per key), and a keyed list (`sources`, `albums`,
  `people`, `provenance`) only ever gains entries, keyed on `path` /
  `(archive_id, folder)` / `name` / `digest` respectively. **The idempotence
  property that makes the never-lose rule usable rather than a slow leak:**
  every history append dedupes on the *value* (`_core()`, which strips
  annotation fields), never on a timestamp — a timestamp inside the dedup key
  would make every history list grow on every re-run, forever. `merge(merge(B,
  U), U)` is required to be byte-identical to `merge(B, U)`; see
  `tests/test_sidecar_schema.py::test_never_loses_a_value_over_a_random_merge_sequence`
  for the property test this exists to satisfy. `_ANNOTATION_FIELDS`
  (`observed_at`, `superseded_at`, `first_seen`, `last_seen`, `rejected`,
  `history`) is the registry that makes dedup possible: **any new annotation
  key added to a history entry must be added here**, or that entry can never
  match itself on a later merge and the list grows unboundedly. This is not a
  hypothetical failure mode — a `rejected` flag left out of this set once
  shipped as a Critical bug that grew a history entry per `watch` cycle,
  forever. The registry governs all three merge paths the same way:
  `_merge_tiered`/`_merge_versioned` use it (via `_core()`) to decide whether
  a superseded *block* matches itself for dedup, and `_merge_keyed_list` uses
  it directly, per field, to decide whether a changed field on an existing
  entry (e.g. `people[].confirmed_at`) advances in place like `last_seen`
  rather than relocating the old value to that entry's `history[]` — the
  mechanism differs, but skipping either one reintroduces the same unbounded
  growth. (`_merge_keyed_list` did not consult the registry at all until this
  was found and fixed during Task 8 — see the "Fix round 1" note in
  `.superpowers/sdd/task-8-report.md`.) `migrate()` upgrades a v1 sidecar to v2 (the old `takeout` block
  becomes a `provenance[]` entry of `kind:
  "imageharbor_v1_takeout_block"`), itself losslessly and idempotently.
- **`sidecar.py`** — optional, per-image `.json` metadata file (on by default —
  see `--sidecar/--no-sidecar` below). Policy lives entirely in
  `sidecar_schema.py`; this module owns only I/O: `read_sidecar` (returns `{}`
  if absent), `merge_sidecar(organized_path, updates)` (reads the existing
  sidecar, calls `sidecar_schema.merge`, writes back atomically via a temp file
  + `os.replace`), and quarantining a sidecar it cannot parse. **A corrupt
  sidecar is renamed aside (`<name>.json.corrupt-<timestamp>`) rather than
  treated as empty** — the previous behavior returned `{}` for an unreadable
  file, which meant the next merge silently overwrote bytes nobody had actually
  read, exactly the data loss the never-lose rule exists to prevent; quarantine
  preserves the original bytes and lets a fresh sidecar be built beside them.
  Unknown keys — including hand edits — are preserved across every merge. The
  facts pass merges `identity`/`sources`/`date`/`descriptor`/`exif`; the
  enrichment pass later merges `classification`. There is no standalone
  `write_sidecar` — `merge_sidecar` is the only entry point.
- **`backfill.py`** — `backfill_sidecars(organized_dir, catalog, *, dry_run=False)`
  rebuilds/merges a sidecar for every cataloged photo from what the catalog and
  the organized copy's own EXIF still hold, for a library organized before
  sidecars were the default. What it can write is bounded by what the catalog
  holds: Google Takeout `provenance[]` is not recoverable this way (only
  re-ingesting the archives restores it). Before building its update dict it
  reads the file's *existing* sidecar and omits the `date`/`descriptor` block
  entirely when that block is already recorded at a tier >= the catalog's —
  a first-hand observation a real pass already wrote must never be
  re-asserted from the catalog's lossier columns. When it does write a date,
  it writes `date.date_str` (the bare `YYYY-MM-DD` the catalog actually
  holds), never a fabricated `T00:00:00` — the catalog stores a date
  *string*, so reconstructing a `datetime` from it invents a time-of-day
  nothing ever measured, and because sidecar history is never pruned, that
  fabrication would be permanent. `dry_run=True` performs no filesystem
  writes at all, including no quarantine of an unparseable existing sidecar —
  it passes `quarantine=False` to `sidecar.read_sidecar` for exactly that
  reason.
- **`cli.py`** — Click entry point (`process`, `enrich`, `watch`, `verify`,
  `catalog list/get`, `takeout ingest/status`, `sidecar backfill`, `faces
  scan/cluster/calibrate/status/models download`). `watch` gains two
  dashboard flags alongside its existing `--sidecar`-style options:
  `--dashboard-port` (`IMAGEHARBOR_DASHBOARD_PORT`, default `8080`) and
  `--no-dashboard` (a bare flag; the dashboard is on by default), plus four
  faces flags: `--faces/--no-faces` (`IMAGEHARBOR_FACES`, off by default —
  a new, heavier, opt-in extra must not start running face detection just
  because `watch` was invoked), `--face-model-dir`
  (`IMAGEHARBOR_FACE_MODEL_DIR`), `--face-threshold`
  (`IMAGEHARBOR_FACE_THRESHOLD` — parsed by hand, not `type=float`, because
  `docker-compose.yml` ships it as `""` on purpose until `faces calibrate`
  has measured a real value, and click's float type raises on an empty
  envvar string), and `--face-recluster-threshold`
  (`IMAGEHARBOR_FACE_RECLUSTER_THRESHOLD`, default `500`). `--faces`
  requested but the extra not importable is not fatal — it degrades to one
  warning, exactly like an already-bound dashboard port does, and `watch`
  builds one `dashboard.control.ControlPlane` per run (now also carrying
  `env_faces`) and passes the *object* itself into
  `watcher.watch(..., control=control)` — see `dashboard/` below for why
  that matters.
- **`dashboard/`** — the operational dashboard and control gateway that
  `watch` serves in-process on a daemon thread: library stats, evidence
  quality, work queues, pass history, a projection of remaining work, and (if
  a `FaceStore` was wired in) a People review queue, plus pause/resume, a
  poll-interval override, and AI-enrichment/faces toggles. See
  `docs/superpowers/specs/2026-08-19-dashboard-design.md` for the full design.
  Five modules, split the same way `sidecar_schema.py` is split from
  `sidecar.py`: `projections.py` (pure, no I/O — the logic most likely to be
  wrong), `stats.py` (reads the catalog — and, when given one, a `FaceStore`
  — into the `/api/stats` document), `control.py` (the pause flag and the
  `settings`-table override precedence, now including `faces` alongside
  `interval`/`enrich`), `server.py` (`http.server`, routing, the page),
  `people.py` (the People review API — `confirm`/`reject`/`merge`/`split`,
  thin validating wrappers around the matching `FaceStore` method; see
  `imageharbor/faces/` below for why `confirm` never writes a sidecar
  synchronously).
  - **Never-stop-the-watcher rule.** A dashboard failure — the port already
    bound, the server thread raising, a stats query failing — logs a warning
    and lets organizing continue; it never aborts a pass or the process. This
    is the same reasoning that keeps a sidecar failure from failing an image
    that is already copied, verified, and cataloged: observability is
    subordinate to the work. Concretely: `server.serve()` catches the bind
    `OSError` and returns `None` instead of raising; every request handler in
    `server.py` catches its own failures and returns a JSON error rather than
    ever raising into `http.server`'s default 500-with-traceback; and every
    section function in `stats.collect()` is wrapped by `_safe()`, so one
    failing query (e.g. `queues`) reports itself as `None` in the document
    instead of taking the whole page down.
  - **Pause is between photos, never mid-photo, in both passes.** Copy →
    verify → catalog is atomic per photo (facts pass) and the equivalent
    describe → resolve → catalog-write → tier-gated-rename is atomic per row
    (enrichment pass); `ControlPlane.pause_check()` is consulted only at
    those boundaries, never inside either atomic unit. `watcher.watch()`
    forwards `control.pause_check` into `run_once`/`run_pass`/
    `enrich_library` for exactly this reason — a pause landing mid-pass still
    only takes effect at the next file/row boundary, and that pass's `runs`
    row is recorded with `paused=1`. Pause also survives a process restart:
    `ControlPlane.__init__` seeds its in-memory flag from the `settings`
    table's `paused` row, so a container that comes back after being
    deliberately paused stays paused rather than silently resuming.
  - **`watch()` takes the `ControlPlane` object, not `interval`/
    `enrich_enabled` values.** The loop runs once for the life of the
    container, so `control.pause_check()`, `control.interval`, and
    `control.enrich_enabled` are re-read fresh on every iteration rather than
    captured once at call time — a plain float/bool argument would freeze at
    startup, and a dashboard edit would update the UI and persist to the
    database while never actually changing runtime behavior until a restart.
    The `interval`/`enrich_enabled` parameters still exist on `watch()` for
    callers that pass `control=None` (tests, and any future non-dashboard
    caller); they behave exactly as before the dashboard existed.
  - **Projections refuse to guess.** `dashboard/projections.py` returns a
    `stalled` or `unknown` status — never a fabricated ETA — whenever the
    breaker is OPEN, the system is paused, there has been no recent progress,
    the run history is stale (older than the caller's derived staleness
    window) or unparseable, or the computed rate is implausible (e.g. a pass
    measured under `MIN_PASS_SECONDS`). A confident wrong ETA sends an
    operator away when they should have looked; this is the same instinct
    that puts a photo in `Undated/` rather than guessing a year.
  - **`sqlite3.Row` is not a `collections.abc.Mapping`.** `projections.project()`
    filters incoming run rows with `isinstance(r, Mapping)` before trusting
    them, and a bare `sqlite3.Row` — which supports index/name lookup but not
    the full `Mapping` protocol — fails that check. A real production bug
    during implementation passed `Catalog.recent_runs()` rows straight into
    `project()` without converting them first: every test used plain dicts
    and stayed green, while every real row was silently discarded and the
    projection reported `unknown` forever. `dashboard/stats.py` now converts
    with `dict(row)` at the catalog boundary before rows reach `projections`
    or the JSON document — do the same at any new call site that crosses
    from a `sqlite3.Row` into code that expects a `Mapping`.
  - **The projections module conflates readability with meaning — a known,
    not-yet-fixed gap.** Across three review rounds, eight defects were found
    in `projections.py`, every one an unreadable or implausible input (an
    unparseable timestamp, a negative backlog, a sub-second pass duration, a
    timezone-naive/aware mismatch) treated as if it were a valid one, because
    `None`/`0`/an empty collection each did double duty for "absent",
    "unreadable", *and* "genuinely zero" at different call sites. An explicit
    `Unreadable` sentinel at each parse site, distinct from a real `None`
    and a real `0`, would turn the next such defect into a type error
    instead of a silent misread — this is a deliberate follow-up, not done
    here.
- **`faces/`** — a third pass, independent of facts and enrichment, that
  detects faces, embeds and clusters them, and proposes person names from
  photos Google Photos already tagged. It makes **no AI-backend call and no
  ongoing network call** — everything runs in-process against local ONNX
  models (one-time weights fetched by `faces models download` /
  `download.py`, checksum-verified before use) — so it never touches
  `AIClassifier`, never consults or feeds the circuit breaker, and needs no
  account or API key. Split into a **pure core**, testable with zero model
  weights, and a thin **I/O shell** around it, the same "logic vs. plumbing"
  split as `sidecar_schema.py`/`sidecar.py`:
  - Pure: `decode.py` (raw YuNet ONNX output → boxes/landmarks), `align.py`
    (landmark-based face warp), `cluster.py` (`cluster_faces` — the
    `MixedModelError` guard lives here), `attribute.py` (name proposals from
    cluster + Google-tag overlap), `calibrate.py` (measures the clustering
    threshold from the library's own anchor photos), `names.py`
    (`normalize`/`case_variants`), and `preprocess.py` (`build_blob`, the
    single preprocessing path `detect.py` and `embed.py` both call — a wrong
    channel order here doesn't raise, it quietly returns worse embeddings,
    which is why `tests/faces/test_preprocess.py` pins channel order,
    mean/std, and NCHW layout against what `models.py` declares rather than
    trusting a similarity threshold that could just drift).
  - I/O: `detect.py`/`embed.py` (the only two modules that import
    `onnxruntime`, wrapping a loaded ONNX session — construct one per
    long-lived worker, never per photo), `store.py` (`FaceStore` — owns the
    `faces`/`face_scan`/`clusters`/`people`/`proposals` tables in the same
    SQLite file `Catalog` uses, via its own connection; only `confirm`/`merge`
    ever *assign a new* person to a cluster — `replace_clusters` also writes
    `clusters.person_id`, but only to restore one a human already confirmed),
    and `runner.py` (`scan` — per-photo, resumable; `build_clusters` —
    whole-library; `propagate_sidecars` — writes confirmed names into
    sidecars, guarded so a repeat run is byte-identical; `google_names` —
    reads Google-tagged names straight from sidecars for `cluster`/
    `calibrate`, which only take `--dest`).
  - `imageharbor.faces.HAS_ONNX` (set at import time, catching every failure
    — missing package, ABI mismatch, anything — as the same "can't run a
    model" answer) is the one importability signal the whole feature gates
    on. `cli.py`'s `_require_onnx()` turns a raw `ModuleNotFoundError` into
    an actionable message for the `faces` subcommands; `watcher.
    faces_available()` is the identical check for `watch`, read fresh on
    every cycle (never cached at import time) so a test's
    `monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)` is actually visible.
  - **`watch --faces` wiring.** `cli.py` builds one `Detector`/`Embedder`/
    `FaceStore` per run (loading an ONNX session is too expensive to repeat
    every poll) into a `watcher.FacesConfig`, then `watch()`'s third pass —
    after facts and enrichment — runs `runner.scan` (per-photo, `should_stop`
    wired to the same pause check the other two passes use) and
    `runner.propagate_sidecars` every cycle, but calls `runner.build_clusters`
    — a whole-library operation — only when `FaceStore.unclustered_face_
    count` exceeds `FacesConfig.recluster_threshold` (default `500`) or no
    cluster exists yet. **`watch` never passes `--recluster`**, for the same
    reason it never passes `--reclassify` (see the enrichment monotonicity
    invariant above): a continuous loop must never treat "run again" as
    "start over". If `faces_available()` is false, or clustering is due but
    no threshold is configured yet, `watch()` logs exactly one warning for
    the life of the run and skips — never per cycle, so a permanently-absent
    extra or an un-calibrated threshold doesn't flood the log for days —
    while organizing and enrichment continue unaffected. Crop cache:
    `<catalog_dir>/face-crops/<ab>/<cd>/<digest>-<i>.jpg`, sharded two
    levels, on the catalog volume rather than the (possibly NAS-mounted)
    `--dest` tree — regenerable from the stored bbox at any time, so losing
    it costs a re-crop, never a re-detect.

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
- **A sidecar may gain information; it may never lose any.** Every merge into a
  sidecar goes through `sidecar_schema.merge`, which is total (never raises)
  and relocates a superseded value into a `history[]` list rather than
  overwriting it — never drop a value on the floor to keep a sidecar smaller.
  This is only usable because history dedupes on the *value* with annotation
  fields stripped (`_core()`/`_ANNOTATION_FIELDS`), not on `observed_at` or any
  other timestamp: a repeated `process`/`enrich`/`watch`/`takeout ingest` run
  must leave every sidecar byte-identical
  (`merge(merge(B, U), U) == merge(B, U)`). Do not add a new annotation key to a
  history entry without adding it to `sidecar_schema._ANNOTATION_FIELDS` — an
  omission there means the entry can never match itself on a later merge, and
  the history list grows without bound on every run. This shipped once as a
  Critical bug and is why the registry comment on `_ANNOTATION_FIELDS` is as
  blunt as it is. One unbounded-growth path remains and is accepted, not a
  bug: `classification` is a `VERSIONED_BLOCKS` entry (no tier gate), so
  looping `enrich --reclassify` against a nondeterministic AI backend — one
  that answers a different caption/subject for the same image on every call
  — records a new `classification` history entry per distinct answer,
  forever. It is user-driven (an operator has to choose to loop
  `--reclassify`) and not exposed on `watch`, which never passes
  `--reclassify`.
- **Takeout pairing never guesses, and Takeout archives are never written to.**
  If no pairing rung yields exactly one sidecar match, `pairing.sidecar_for`
  returns `None` and the member is ingested from EXIF and its filename alone —
  a fully correct outcome. A *wrong* pairing writes another photo's capture date
  into this photo's name and folder, which is exactly the quiet corruption the
  SHA-256 discipline exists to prevent. Separately: archives are opened `'r'`
  only, and Google's `creationTime` must never reach `resolve_date`.
- **A pairing's confidence decides what it may contribute — never which engine
  produced it.** `own` may supply capture date, title, and face tags. `related`
  supplies the capture date only: its sidecar names a *different* file (usually
  this file's unedited original before an `-edited` copy), so the title and
  `people` inside it describe that other photo, not this one, and
  `_ingest_image`/`_merge_takeout_sidecar` must never apply them here. This
  holds identically whether the pairing came from a `Takeout_Inventory` index
  read by `index_reader.py` (`--takeout-index`, optional and auto-detected —
  without it, ingestion behaves exactly as it always has) or from the built-in
  six-rung ladder in `pairing.py` — `confidence` is the only thing either path
  is allowed to branch on, so a photo's title and face tags can never depend on
  whether a second tool was run. A `related` pairing's raw Google JSON is still
  preserved as provenance (never dropped — see the previous invariant) but
  labelled with its `confidence` and `pair_rule`, so the GPS coordinates and
  other fields inside it read as self-describing rather than silently
  authoritative.
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
  Preserve this copy → verify → catalog ordering when editing the pipeline. The
  one exception is `consume_source=True` (used only by Takeout ingestion, on a
  staging file the caller owns and created as disposable, never a real
  original): the ordering becomes rename → verify → catalog, with verification
  still reading the destination — nothing enters the catalog unverified either
  way.
- **Runtime output directories are git-ignored, not source** (`Photos-Organized/`,
  `Review/`, `Duplicates/`, `Logs/`, `catalog.db`, etc. in `.gitignore`).
- **Faces never rename or move a file.** No code path in `imageharbor/faces/`
  calls `tiers.is_upgrade` or `relocate` — a detected/clustered/confirmed face
  changes catalog rows and, once confirmed, a photo's sidecar `people` list,
  never its path or filename. This is the same "identity lives in the
  sidecar, not the name" posture PCS classification already has, applied to a
  fact that is far more personal to get wrong.
- **No identity is written without human confirmation.** `FaceStore.confirm`
  and `FaceStore.merge` are the *only* two methods that ever *assign a new*
  person to a cluster; `record_proposals` (what `faces cluster` calls) only
  ever writes to the `proposals` table and is guaranteed never to touch
  `person_id` — see `store.py`'s own mutation-tested guarantee.
  `replace_clusters` (a recluster) also writes `clusters.person_id`, but only
  to restore a person a human already confirmed onto its old cluster; it
  never invents one, so no identity is ever written without a human behind
  it. A rejected proposal is recorded as rejected, not deleted, so the same
  wrong guess isn't re-proposed every pass. This mirrors `pairing.sidecar_for`
  returning `None` rather than guessing a Takeout pairing, and
  `date_resolver` refusing mtime rather than assert a date it can't support —
  applied here to a person's name, which is both easier to get wrong
  unnoticed and more personal than a wrong date.
- **Embeddings are never compared across `embed_model` values.** A vector from
  one model shares a coordinate space with another only by coincidence, so a
  cross-model comparison is not merely wrong, it's a *plausible-looking*
  number that means nothing — worse than an error. `cluster.cluster_faces`
  raises `MixedModelError` rather than silently averaging or comparing across
  models, and `FaceStore.iter_face_vectors` filters to one `embed_model`
  before anything reaches clustering. Do not add a call path that pools
  vectors from two `embed_model`s to "get more data" — reprocessing under the
  new model is the only correct way to compare across a model swap.
- **Face failures never feed the circuit breaker.** That circuit is reserved
  for `AIClassifier.describe()` failures (AI-perception evidence about the
  backend); a face-scan failure (a corrupt image, a decode error) is a local
  filesystem/image fault with no bearing on the AI backend's health.
  `faces.runner.scan` catches per-photo exceptions itself and records them
  into `failed_files` via `catalog.record_file_failure` — the same table the
  enrichment pass's poison-quarantine bookkeeping uses — and never calls
  `breaker.record_failure`. `watch()`'s faces-pass block (`watcher.py`)
  mirrors this: it wraps the whole pass in its own `try/except` so a crash
  there can't take the loop down, and that block never references *breaker*
  at all.
- **Name identity is exact.** No fuzzy, similarity-based, or case-insensitive
  name merging, ever — `names.normalize` only strips and collapses
  whitespace; it deliberately leaves case alone. This library's own tagged
  vocabulary is why: `Conrad Storz` (3,309 photos) and `Conrad Storz III`
  (980 photos) are a father and son distinguished only by a suffix, and
  `pete storz`/`claire Storz`'s lowercase drift looks identical in *shape* to
  that real distinction. A scheme that can't tell those two cases apart must
  not auto-merge either one — case variants are only ever *reported*
  (`names.case_variants`) for a human to confirm.
- **Embeddings are L2-normalized where they are produced.** `Embedder.embed_
  batch` normalizes every vector before returning it, so cosine similarity
  and Euclidean distance stay equivalent for every downstream consumer —
  `cluster.py`'s centroid averaging, `calibrate.py`'s pairwise similarities —
  without each of them re-normalizing (or worse, one of them forgetting to).
  A stored embedding that somehow isn't unit-length is a bug upstream of
  storage, not something a consumer should silently correct for.

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
