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
| Same, as one JSON document on stdout (exit 0/1/2) | `uv run imageharbor process … --json` (also `enrich`, `verify`) |
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

**Enrichment pass** (`enrichment.enrich_library`, driven by `imageharbor enrich`) —
reads the organized copy, resumable, AI-dependent:

`unenriched rows → describe → concept_map/pick_class → taxonomy resolve →
catalog → tier-gated rename → sidecar`

The facts pass is what makes a full library organize possible without an AI
backend at all: hashing, dedup, EXIF, and copying are pure local computation, and
placement/naming are decided from facts (EXIF + original filename) rather than
from AI perception. The enrichment pass adds description/classification later,
and can only ever *improve* a file — see the monotonicity invariant under
"Critical invariants" below for the rule that guarantees this.

## Module map

One line per module. **Full per-module detail, rationale, and bug history live in
[`docs/architecture.md`](docs/architecture.md)** — read the relevant section there
before changing a module; the invariants below are the rules that must hold.

**Facts pass**
- `pipeline.py` — facts-pass orchestrator, no AI calls. Copy → fsync → verify → catalog; `ExternalEvidence`, `consume_source` (Takeout only).
- `date_resolver.py` — `ResolvedDate`: EXIF > filename pattern > `Undated/`; Takeout `photoTakenTime` via `external_date`. Never mtime, never `creationTime`.
- `descriptor.py` — `ResolvedDescriptor` from original filename stem (human = tier 30, camera = tier 0).
- `tiers.py` — date/descriptor ladders and `is_upgrade(old, new)`, the sole gate on every rename.
- `relocate.py` — `target_path`, `apply_relocation`, digest-based self-healing.
- `hashing.py` + `filename.py` — SHA-256 as 43-char unpadded Base64url; `[<date>][-<descriptor>]_<digest>.<ext>`.
- `discovery.py`, `content_type.py`, `exif_reader.py`, `util.py` — leaf helpers (file discovery, magic-byte sniffing, EXIF, `now_iso`/`fsync_file`).

**Enrichment pass**
- `enrichment.py` — `enrich_library`; describe → class → taxonomy → catalog → tier-gated rename → sidecar; drives the breaker.
- `ai_classifier.py` — perception only (`describe`), plus `pick_class` fallback and `adjudicate`; `StubClassifier` / `OpenAIClassifier`. Backend must stay swappable.
- `concept_map.py` — decides the top-level class (static seed + learned store).
- `taxonomy.py` + `pcs.py` — self-extending string-coded PCS taxonomy (`TaxonomyStore`); `pcs.py` is seed data only.
- `circuit_breaker.py` — pure 3-state breaker, enrichment-only.

**Storage / metadata**
- `catalog.py` — SQLite WAL, one connection guarded by `Catalog.lock` (RLock); `photos`, `sources`, `taxonomy`, `learned_concepts`, `failed_files`, `takeout_*`, `runs`, `settings`; `TakeoutStore`/`TaxonomyStore` composed on it; `run_select` is the guarded read door.
- `sidecar_schema.py` — pure merge policy (gain, never lose; history dedupes on value; `_ANNOTATION_FIELDS` registry).
- `sidecar.py` — sidecar I/O, atomic write, corrupt sidecar quarantined aside.
- `backfill.py` — rebuild sidecars from the catalog (cannot recover Takeout provenance).

**Entry points**
- `api.py` — public facade and only promised import surface; `--json` and 0/1/2 exit codes are a stable contract.
- `cli.py` — Click CLI: `process`, `enrich`, `watch`, `verify`, `catalog`, `takeout`, `sidecar backfill`, `faces`.
- `watcher.py` — `watch` loop (facts → enrich → faces), breaker backoff, poison reconcile.
- `takeout/` — Google Takeout ingestion (survey → resumable ingest → `provenance.py`). Global pairing index; pairing never guesses; archives opened `'r'` only; no AI calls.
- `dashboard/` — in-process ops dashboard on loopback (`projections`, `stats`, `control`, `server`, `people`). Never stops the watcher; pause only between photos; projections refuse to guess.
- `faces/` — independent local-ONNX pass: detect, embed, cluster, propose names. Never renames files; no identity without human confirmation; no cross-model comparison.

## Critical invariants — do not break these

Terse rules only. The full rationale, failure history, and edge cases for each are
in [`docs/invariants.md`](docs/invariants.md) (same order).

**Classification & placement**
- **PCS codes are strings** matching `^\d+(~\d+)*$` (`"330"`, `"330~1"` — never a dot). Minted append-only by `taxonomy.Taxonomy`, never renumbered or reused. The 9 top-level classes are fixed.
- **The classifier perceives; it never classifies.** `describe()` returns a `ContentDescription` only. Class = `concept_map.class_for` (`pick_class` is a text-only fallback, memoized via `remember`); code = `Taxonomy.resolve_or_create` (exact/alias match, else `adjudicate`, else mint). No string-similarity gating before the adjudicator.
- **Placement comes from `ResolvedDate.folder`** (`YYYY/YYYY-MM/` or `Undated/`), never from PCS. PCS lives in the catalog and the sidecar `classification` block only.
- **Rename/move only when `tiers.is_upgrade(old, new)` is True.** Never add an unconditional rename path. `--reclassify` bypasses only the work queue, not this predicate.
- **File mtime never enters the date ladder**; Takeout `creationTime` never reaches `resolve_date`.

**Integrity**
- **Originals are read-only; copy → fsync → verify → catalog.** Use `copyfile` → `fsync_file` → `copystat`, never `copy2`. `consume_source=True` (Takeout staging only) uses rename → verify → catalog.
- **Content addressing is stable.** SHA-256 over raw bytes, 64 KiB chunks, 43-char unpadded Base64url. Locate the digest by counting back 43 chars from the stem end, never `rsplit("_")`; keep `hashing.extract_digest_from_stem` and `filename.parse_filename` in sync.
- **A sidecar may gain information, never lose any.** All merges go through `sidecar_schema.merge`; history dedupes on value, never timestamp; `merge(merge(B,U),U) == merge(B,U)`. Any new history annotation key must be added to `_ANNOTATION_FIELDS`. (Accepted exception: looping `enrich --reclassify` on a nondeterministic backend grows `classification` history; `watch` never does this.)
- **Runtime output dirs** (`Photos-Organized/`, `Review/`, `Duplicates/`, `Logs/`, `catalog.db`) are git-ignored, not source.

**Takeout**
- **Pairing never guesses; archives are opened `'r'` only.** No single match → `None` → ingest from EXIF + filename.
- **Pairing confidence decides contribution, not engine.** `own` → date, title, people. `related` → capture date only (its title/people describe a different file). `related` raw JSON is still preserved, labelled with `confidence`/`pair_rule`.

**Circuit breaker**
- **Only `describe()` and `pick_class` failures feed the breaker.** Filesystem faults, taxonomy/catalog errors, and `adjudicate` (caught, mints a new code) never do. The facts pass and faces pass never touch it.

**Faces**
- **Faces never rename or move a file** (no `tiers.is_upgrade`/`relocate` under `faces/`).
- **No identity without human confirmation.** Only `FaceStore.confirm`/`merge` assign a new person to a cluster; `record_proposals` writes `proposals` only; `replace_clusters` only restores an already-confirmed person. Rejected proposals are kept, not deleted.
- **Never compare embeddings across `embed_model`** (`MixedModelError`; `iter_face_vectors` filters to one model).
- **Embeddings are L2-normalized where produced** (`Embedder.embed_batch`); consumers never re-normalize.
- **Name identity is exact.** No fuzzy or case-insensitive merging; `names.normalize` strips/collapses whitespace only; case variants are reported for a human.
- **Face scan failures** go to `failed_files` via `record_file_failure`, never the breaker; the watcher's faces block is wrapped in its own `try/except`.

**Public contract**
- **`--json` documents and exit codes (0/1/2) are stable.** `to_dict()` keys may be added, never removed or retyped; outcome constants and everything in `imageharbor.__all__` do not change. `organize-my-life` depends on this.

## Known limitations

- **Poison-quarantine cannot fire when the poison is the entire remaining queue.** Quarantine counts only during a *healthy* enrichment pass, and an all-poison queue is indistinguishable from a real outage. Accepted: cost is one half-open probe per backoff interval (cap 900s), it self-resolves when any describable file appears, and `watch` logs a warning. Detail: `docs/architecture.md` (`catalog.py` `failed_files`), the note on `watcher._reconcile_poison`, and `tests/test_poison.py::test_poison_at_the_head_does_not_halt_enrichment`.

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
