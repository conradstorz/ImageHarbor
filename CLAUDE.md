# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ImageHarbor is

A deterministic, resumable CLI that organizes a photo library. Its three verbs —
**Classify. Verify. Preserve.** — map directly to the design:

- **Classify** — each image is assigned a PCS (Photo Classification Standard) code
  that decides the destination folder tree. A pluggable AI backend only *perceives*
  the image (subject/scene/objects/caption/tags); the organizer (`concept_map.py` +
  `taxonomy.py`) is what actually decides the class and folder.
- **Verify** — every file is content-addressed by SHA-256; the digest is embedded
  in the filename so any file can later be re-verified against its own name.
- **Preserve** — originals are treated as read-only. Files are *copied* (never
  moved/modified) into the organized tree, verified after copy, and recorded in a
  SQLite catalog.

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
| Organize a library | `uv run imageharbor process --source SRC --dest DEST` |
| Re-verify integrity | `uv run imageharbor verify DEST` |
| Watch a library continuously | `uv run imageharbor watch --source SRC --dest DEST` |
| Build the Docker image | `docker build -t imageharbor:latest .` |
| Run the watcher (compose) | `docker compose up -d` (see `docs/deploy-docker.md`) |
| Query the catalog | `uv run imageharbor catalog list --catalog DEST/catalog.db` |

There is no linter/formatter configured. `pyproject.toml` is the single source of
truth for deps, extras, pytest config, and the `imageharbor` entry point.

## Architecture

Single package `imageharbor/`, orchestrated by a linear pipeline. The flow for one
image (`pipeline.Pipeline._do_process`) is the spine of the whole system, in order:

`hash → duplicate check → EXIF → describe (perception) → concept_map.class_for
(AI pick_class + remember on miss) → taxonomy.resolve_or_create(class,
primary_subject) → build filename → compute dest path → copy → verify copy →
upsert catalog (content) → optional sidecar`

Module responsibilities:

- **`pipeline.py`** — the orchestrator above. After hashing/dedup/EXIF, it calls
  `classifier.describe()` for pure perception, then `concept_map.class_for()` to
  pick one of the 9 fixed top-level classes; only on a concept-map miss does it
  fall back to `classifier.pick_class()` and memoize the result via
  `concept_map.remember()`. It then calls `taxonomy.resolve_or_create(cls,
  content.primary_subject, adjudicator=...)` with a top-level class and **no
  `sub_parent`** — `primary_subject` is the level-2 label. Owns the
  copy-then-verify-then-catalog ordering and the `PipelineStats`/`ProcessResult`
  result types. If a post-copy integrity check fails, the copy is deleted and an
  error is raised — nothing enters the catalog unverified.
- **`pcs.py`** — **seed data + helpers only**: `PCS_CATEGORIES` defines the 9 fixed
  top-level classes (100–900) and their original sub-codes, used once to seed the
  catalog `taxonomy` table on first run (`Taxonomy.ensure_seeded`). `resolve_code`
  (int → int, unknown → 900) is retained for legacy/tooling use, but the pipeline no
  longer calls it — code assignment and folder-path resolution now live in
  `taxonomy.py`. `pcs.parent_folder_name`/`sub_folder_name` were **removed**.
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
  aliases one code onto another after the fact. `folder_path(code)` walks
  parent_code links to build the slash-joined destination folder tree (this is what
  the pipeline now uses instead of `pcs.parent_folder_name`/`sub_folder_name`).
  `snapshot_text()` renders the current taxonomy for the classifier prompt.
  `taxonomy.py` itself was not touched by the perception/organization reframe — the
  pipeline now calls `resolve_or_create(class, primary_subject)` with a fixed
  top-level class and **no `sub_parent`**, so in practice the taxonomy is
  effectively **two levels** (fixed class → `primary_subject` sub-category); the
  `sub_parent`/`~N`-under-a-leaf machinery still exists but its call sites are
  currently unused.
- **`ai_classifier.py`** — perception only. `AIClassifier` ABC with two
  implementations chosen by the `--ai` flag: `StubClassifier` (default;
  deterministic, no network — derives a subject/tags from filename keywords, used
  by all tests) and `OpenAIClassifier` (optional, gated behind the `openai` extra
  and imported lazily). `describe(image_path, exif_data) -> ContentDescription`
  (`primary_subject`, `scene`, `objects`, `caption`, `tags`, `ocr_text`,
  `model_version`) is the only required method — **the classifier never picks a
  class or a PCS code; it only reports what it sees.** `PhotoClassification` and
  `classify()` are gone. Two more ABC methods support the organizer: `pick_class
  (content, classes) -> str` is a **text-only fallback** the pipeline calls only
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
  the AI's). `STATIC_SEED` is built once at import time from
  `pcs.PCS_CATEGORIES`' sub-category names plus a small curated
  keyword/synonym table, mapping normalized subject/object/scene tokens to one of
  the 9 fixed classes. `class_for(primary_subject, objects, scene, catalog)` checks,
  in order: the catalog's `learned_concepts` store (exact normalized-subject match),
  then the static seed against the subject, then against each object/scene token —
  returning `None` on a genuine miss. On a miss the pipeline falls back to
  `classifier.pick_class()` and calls `remember(catalog, primary_subject,
  class_code)` to memoize the decision in `learned_concepts`, so the next photo with
  the same normalized subject is a deterministic, network-free hit.
- **`hashing.py`** + **`filename.py`** — content addressing. SHA-256 is encoded as
  **unpadded Base64url, always exactly 43 chars** (`SHA256_B64URL_LEN`). Filename
  format is `<pcs>-<descriptor>_<sha256>.<ext>`.
- **`catalog.py`** — SQLite (WAL mode). The `photos` table (keyed by the unique
  `sha256_b64url`) is the source of truth for **resumability and duplicate
  detection** (`is_known`); `upsert` is idempotent (`ON CONFLICT … DO UPDATE`) and
  list/dict fields are stored as JSON text. A second `taxonomy` table persists the
  self-extending PCS registry (`code`, `parent_code`, `label`, `folder_name`,
  `aliases`, `alias_of`, `active`), backing `taxonomy.py`. A third `learned_concepts`
  table (`subject`, `class_code`, `hits`, timestamps) is the self-learning store
  behind `concept_map.py`'s `learned_concept_get`/`learned_concept_remember`.
- **`discovery.py`** — yields supported image files (see `SUPPORTED_EXTENSIONS`);
  supports single-file or recursive directory mode and never mutates the source.
- **`exif_reader.py`** — best-effort EXIF/GPS extraction via Pillow; returns `{}`
  rather than raising on any failure.
- **`sidecar.py`** — optional per-image `.json` metadata file (via `--sidecar`).
- **`cli.py`** — Click entry point (`process`, `verify`, `catalog list/get`).

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
- **Folder paths come from `taxonomy.folder_path(code)`**, which walks `parent_code`
  links from the top-level ancestor down to `code`. `pcs.parent_folder_name` and
  `pcs.sub_folder_name` were **removed** — do not reintroduce a two-level-only
  folder scheme; the taxonomy can now be deeper than two levels via `~N` codes.
- **The 43-char digest is located by counting back from the end of the stem, NOT by
  splitting on the last `_`.** Base64url legitimately contains `_`, so a naive rsplit
  corrupts parsing. This logic is duplicated in `hashing.extract_digest_from_stem`
  and `filename.parse_filename` — keep them in sync.
- **Content addressing must stay stable.** The digest is over raw file *bytes*
  (streamed in 64 KiB chunks). Changing the hash algorithm, the Base64url encoding,
  or the 43-char length assumption breaks every existing filename and catalog key.
- **Originals are read-only; the organized copy is verified before it is cataloged.**
  Preserve this copy → verify → catalog ordering when editing the pipeline.
- **Runtime output directories are git-ignored, not source** (`Photos-Organized/`,
  `Review/`, `Duplicates/`, `Logs/`, `catalog.db`, etc. in `.gitignore`).

## Origin & design reference

Scaffolded with help from ChatGPT (which wrote the genesis design document) and
GitHub Copilot (first-draft implementation). Expect code you did not write; read the
current tree before changing it.

The genesis roadmap — **"Jetson Photo Workflow Roadmap (Rev. 2)"**, committed at
[`docs/genesis-roadmap.md`](docs/genesis-roadmap.md) — is the authoritative spec for
intent and acceptance criteria (immutable originals, resumable processing,
full-SHA-256 duplicate detection, filename-alone integrity verification). It predates
the *ImageHarbor* name, so ignore naming there. The current code already satisfies its
core pipeline, PCS, filename, and verification requirements.
