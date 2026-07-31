# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ImageHarbor is

A deterministic, resumable CLI that organizes a photo library. Its three verbs —
**Classify. Verify. Preserve.** — map directly to the design:

- **Classify** — each image is assigned a PCS (Photo Classification Standard) code
  by a pluggable AI classifier, which decides the destination folder tree.
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

`hash → duplicate check → EXIF → classify → resolve PCS code → build filename →
compute dest path → copy → verify copy → upsert catalog → optional sidecar`

Module responsibilities:

- **`pipeline.py`** — the orchestrator above. Owns the copy-then-verify-then-catalog
  ordering and the `PipelineStats`/`ProcessResult` result types. If a post-copy
  integrity check fails, the copy is deleted and an error is raised — nothing enters
  the catalog unverified.
- **`pcs.py`** — the PCS taxonomy (`PCS_CATEGORIES`, codes 100–900 with sub-codes).
  Maps a code to its two-level folder path (`parent_folder_name` → `sub_folder_name`,
  e.g. `300-places/330-beach/`). `resolve_code` falls back **unknown → 900
  (miscellaneous)**; this fallback is relied on throughout, so classifiers may return
  any int safely.
- **`ai_classifier.py`** — `AIClassifier` ABC with two implementations chosen by the
  `--ai` flag: `StubClassifier` (default; deterministic, no network — infers a code
  from filename keywords, used by all tests) and `OpenAIClassifier` (optional, gated
  behind the `openai` extra and imported lazily). Both return a `PhotoClassification`.
  Add new backends by subclassing `AIClassifier` and wiring them in `cli.py`.
  **Design intent (important):** this abstraction exists so the *AI server doing the
  work is swappable*, not just the vendor. The project was inspired by a self-hosted
  AI server (a Jetson Orin Nano on the local network), but nothing is hard-wired to
  it — a local/Jetson HTTP backend is an expected future implementation that does not
  exist yet. Keep the classifier decoupled from any specific host or provider.
- **`hashing.py`** + **`filename.py`** — content addressing. SHA-256 is encoded as
  **unpadded Base64url, always exactly 43 chars** (`SHA256_B64URL_LEN`). Filename
  format is `<pcs>-<descriptor>_<sha256>.<ext>`.
- **`catalog.py`** — SQLite (WAL mode), one `photos` table keyed by the unique
  `sha256_b64url`. `upsert` is idempotent (`ON CONFLICT … DO UPDATE`); list/dict
  fields are stored as JSON text. This table is the source of truth for
  **resumability and duplicate detection** (`is_known`).
- **`discovery.py`** — yields supported image files (see `SUPPORTED_EXTENSIONS`);
  supports single-file or recursive directory mode and never mutates the source.
- **`exif_reader.py`** — best-effort EXIF/GPS extraction via Pillow; returns `{}`
  rather than raising on any failure.
- **`sidecar.py`** — optional per-image `.json` metadata file (via `--sidecar`).
- **`cli.py`** — Click entry point (`process`, `verify`, `catalog list/get`).

## Critical invariants — do not break these

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
