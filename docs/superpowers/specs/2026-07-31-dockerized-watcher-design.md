# ImageHarbor — Dockerized Watcher + OpenAI-Compatible AI Backend

**Status:** Approved design (pending spec review)
**Date:** 2026-07-31
**Author:** Conrad Storz (with Claude Code)

## 1. Purpose

Make ImageHarbor deployable as a long-running service that continuously
organizes a photo library living on a Synology NAS, using a self-hosted,
OpenAI-compatible AI server (a Jetson Orin Nano on the LAN) for classification.

Two components are delivered together:

1. **A generalized AI backend** — the existing `OpenAIClassifier` extended to
   talk to *any* OpenAI-compatible endpoint (base URL + model configurable), so
   the AI *server* is swappable, not just the vendor.
2. **A containerized polling watcher** — a new `imageharbor watch` command,
   packaged as a Docker image + compose file, that runs on an always-on Linux
   host, reads photos from a read-only NAS mount, and writes the organized
   library back to a read-write NAS mount.

## 2. Deployment context (decisions)

| Decision | Choice |
|----------|--------|
| Where the container runs | A separate always-on **amd64 Linux host** (not the NAS) |
| NAS access | NAS shares **mounted on the host**, then **bind-mounted** into the container (container holds no NAS credentials) |
| Source mount | **Read-only** (`:ro`) |
| Dest mount | Read-write |
| Run model | **Continuous polling** (rescan on an interval; not inotify) |
| Catalog storage | **Local Docker volume** on the host — *not* on the NAS |
| AI backend | **OpenAI-compatible HTTP** (Jetson), configurable base URL + model; `stub` remains the offline default |
| Watch loop location | **Inside the app** (`imageharbor watch`), not external cron |

### Rationale for the non-obvious ones

- **Polling, not inotify:** `inotify`/filesystem events do not propagate
  reliably over SMB/CIFS for changes made by other machines (phones and other
  PCs uploading to the NAS). Polling + the resumable catalog is the correct
  model for a networked source.
- **Catalog on a local volume:** SQLite over SMB/CIFS is unreliable — network
  file locking can corrupt a WAL database. Only the organized *image copies* go
  to the NAS. If the host is lost, the catalog can be rebuilt by re-scanning the
  organized tree, because every filename embeds the full SHA-256 (self-verifying).
- **Host bind-mounts:** keeps NAS credentials in the host's mount config
  (`fstab`/`autofs`), never in the container or compose file.

## 3. Component 1 — Generalized OpenAI-compatible classifier

### 3.1 Changes to `imageharbor/ai_classifier.py`

Extend `OpenAIClassifier.__init__` to accept:

- `base_url: str | None = None` — passed through to `openai.OpenAI(base_url=...)`.
  `None` keeps the official OpenAI endpoint. For the Jetson, set e.g.
  `http://<jetson-host>:11434/v1` (Ollama) or whatever the server exposes.
- `model: str = "gpt-4o-mini"` — already a parameter; keep, but it becomes the
  primary knob for local models (e.g. `llava`, `qwen2-vl`).
- `api_key: str | None = None` — already present; local servers usually ignore
  it, but the `openai` SDK requires *some* value, so default to a placeholder
  (e.g. `"not-needed"`) when none is supplied, so construction does not fail.
- `timeout: float = 60.0` — request timeout in seconds, passed to the OpenAI
  client (`openai.OpenAI(timeout=...)` / per-request). Prevents a slow or
  unreachable Jetson from hanging the watcher. On timeout the call raises, the
  pipeline records that image as an `error`, and the watcher continues.

Behavior otherwise unchanged: same system prompt, same PCS-constrained JSON
contract, same robust parsing (already hardened against wrong-typed fields and
invalid JSON, falling back to code 900).

`MODEL_VERSION` continues to reflect the model string so the catalog records
which model classified each image.

### 3.2 CLI wiring (`imageharbor/cli.py`)

`process` and the new `watch` command gain shared AI options:

- `--ai [stub|openai]` (existing; `openai` now means "any OpenAI-compatible
  endpoint").
- `--ai-base-url TEXT` (env `IMAGEHARBOR_AI_BASE_URL`)
- `--ai-model TEXT` (env `IMAGEHARBOR_AI_MODEL`, default `gpt-4o-mini`)
- `--ai-timeout FLOAT` (env `IMAGEHARBOR_AI_TIMEOUT`, default `60`)
- `--openai-key` / env `OPENAI_API_KEY` and/or `IMAGEHARBOR_AI_API_KEY`
  (existing key handling, plus the placeholder default above).

Classifier construction is centralized in one helper so `process` and `watch`
build it identically. Missing-`openai`-package still surfaces as a clean
`click.ClickException` (already implemented).

## 4. Component 2 — `imageharbor watch` command

### 4.1 Behavior

A long-running poll loop:

```
setup: open catalog, build classifier, install SIGTERM/SIGINT handler
loop:
    pass_stats = process_new_files(source, dest, ...)
    log summary: "watch pass: copied=.. duplicates=.. errors=.. skipped=.."
    if shutdown requested: break
    sleep(interval)  # interruptible by the signal
teardown: close catalog, exit 0
```

- Reuses the existing `Pipeline`. The pipeline's `is_known()` check already makes
  re-processing idempotent (a file whose content is catalogued is a duplicate).
- Runs a single pass immediately on start (no initial wait).
- Interval configurable via `--interval` / `IMAGEHARBOR_INTERVAL` (default 300s).
- `--source`, `--dest`, `--catalog`, `--duplicates`, `--sidecar`, `--no-recursive`
  mirror `process`, all env-var backed for the container.

### 4.2 Seen-source-files cache (network-I/O optimization)

**Problem:** a naive pass re-hashes *every* source file every interval to check
`is_known()` — prohibitively expensive over a network mount for a large library.

**Solution:** a local cache of source files already seen, keyed by absolute
source path, storing `(size, mtime_ns)`. Before hashing a discovered file, the
watcher checks the cache:

- **Hit and unchanged** (same size + mtime): skip — do not hash, do not re-copy.
- **Miss or changed:** hash and run the normal pipeline; on success, record/update
  the cache entry.

Design points:

- The cache lives in a new table `source_seen(source_path TEXT PRIMARY KEY,
  size INTEGER, mtime_ns INTEGER, sha256_b64url TEXT, seen_at TEXT)` in the
  **local** catalog database (same file as `photos`, on the local volume).
- The content SHA-256 remains the sole source of truth for identity and
  duplicate detection. This cache is a pure optimization; deleting it only
  causes a one-time full re-hash, never incorrect output.
- Encapsulated in `catalog.py` (`Catalog.source_is_unchanged(path, size, mtime_ns)`
  and `Catalog.record_source_seen(...)`), so the pipeline/watcher do not embed
  SQL. `process` may optionally use it too, but the watcher is the primary user.
- Rationale for path+size+mtime: cheap `os.stat` over the network vs. a full file
  read; standard, well-understood change-detection heuristic.

### 4.3 Signal handling

Install handlers for SIGTERM (Docker stop) and SIGINT (Ctrl-C) that set a
shutdown flag and interrupt the sleep, so the current pass finishes cleanly (or
the sleep aborts immediately) and the process exits 0. No image is left in a
half-copied state because the pipeline already deletes an unverified copy before
raising.

## 5. Component 3 — Docker packaging

### 5.1 `Dockerfile`

- Base: `python:3.12-slim` (amd64).
- Install the project **with the `openai` extra** (e.g. `pip install .[openai]`
  or `uv pip install`), into a system location.
- Create and run as a **non-root** user; the process only needs read on the
  source mount and write on the dest + local catalog volume.
- `ENTRYPOINT ["imageharbor"]`, default `CMD ["watch"]` (configured via env).
- Logs to stdout/stderr (captured by Docker).

### 5.2 `docker-compose.yml`

One service, e.g. `imageharbor`:

- `image` built from the Dockerfile.
- `command: watch` (source/dest/interval from env).
- Volumes:
  - `- /mnt/nas/photos:/data/source:ro` (host-mounted NAS source, read-only)
  - `- /mnt/nas/photos-organized:/data/dest` (host-mounted NAS dest, read-write)
  - `- imageharbor-catalog:/data/catalog` (local named volume for `catalog.db`)
- Environment: the `IMAGEHARBOR_*` variables (section 6).
- `restart: unless-stopped`.
- Named volume `imageharbor-catalog` declared at the bottom.

The compose file documents (in comments) that the two NAS paths must already be
mounted on the host (CIFS/NFS via `fstab`/`autofs`) and that credentials live
there, not in compose.

## 6. Config surface (environment variables)

| Env var | CLI equiv | Default | Meaning |
|---------|-----------|---------|---------|
| `IMAGEHARBOR_SOURCE` | `--source` | (required) | Source dir (mounted read-only) |
| `IMAGEHARBOR_DEST` | `--dest` | (required) | Organized library root |
| `IMAGEHARBOR_CATALOG` | `--catalog` | `/data/catalog/catalog.db` | Catalog path (local volume) |
| `IMAGEHARBOR_INTERVAL` | `--interval` | `300` | Seconds between watch passes |
| `IMAGEHARBOR_AI` | `--ai` | `stub` | `stub` or `openai` |
| `IMAGEHARBOR_AI_BASE_URL` | `--ai-base-url` | (OpenAI default) | OpenAI-compatible endpoint (Jetson) |
| `IMAGEHARBOR_AI_MODEL` | `--ai-model` | `gpt-4o-mini` | Model name |
| `IMAGEHARBOR_AI_API_KEY` | `--openai-key` | `not-needed` | API key (local servers ignore) |
| `IMAGEHARBOR_AI_TIMEOUT` | `--ai-timeout` | `60` | Per-request timeout (s) |
| `IMAGEHARBOR_DUPLICATES` | `--duplicates` | (unset) | Duplicates dir (optional) |
| `IMAGEHARBOR_SIDECAR` | `--sidecar` | `false` | Write JSON sidecars |

CLI flags take precedence over env vars where both are given (standard Click
`envvar=` behavior).

## 7. Testing

All tests offline and deterministic (no network, no real Docker build in CI):

- **Generalized classifier:** `base_url`, `model`, `api_key` placeholder, and
  `timeout` are threaded into the OpenAI client correctly (mock the `openai`
  module / client, assert constructor + call args). Timeout raising →
  classify surfaces the error (pipeline turns it into an image `error`).
- **`watch` command:**
  - one pass processes new files then respects the shutdown flag (inject a fake
    clock/sleep and a classifier stub; assert a single pass then exit);
  - the seen-cache causes an unchanged file to be skipped on the second pass
    (assert the file is not re-hashed / `compute_sha256_b64url` not called for it);
  - a changed file (different mtime/size) is reprocessed;
  - SIGTERM/SIGINT sets shutdown and the loop exits cleanly.
- **Seen-cache (`catalog.py`):** `source_is_unchanged` / `record_source_seen`
  round-trip; changed size or mtime → not unchanged; missing entry → not
  unchanged.
- Existing suite stays green.

Docker: provide the `Dockerfile` and `docker-compose.yml`; a short manual
smoke-test procedure is documented in the deploy doc rather than automated.

## 8. Documentation

- A `docs/deploy-docker.md` covering: host NAS mount setup (CIFS/NFS example),
  building the image, the compose file, env var reference, pointing at the
  Jetson, and the manual smoke test.
- Update `CLAUDE.md` commands table with `watch` and the Docker workflow.

## 9. Out of scope (deferred)

- True inotify/event-driven watching (unreliable over SMB — polling is the
  deliberate choice).
- Running the container on the NAS itself.
- A human "Review/" triage workflow for low-confidence classifications.
- Multi-instance/concurrent watchers (the catalog is single-writer; see the
  deferred concurrency-locking note in project memory).

## 10. Acceptance criteria

1. `OpenAIClassifier` can be pointed at a local OpenAI-compatible server via a
   configurable base URL + model and classify an image (verified with a mocked
   client) without contacting the official OpenAI API.
2. `imageharbor watch` runs continuous passes, processes new/changed files,
   skips unchanged files without re-hashing them, and shuts down cleanly on
   SIGTERM/SIGINT.
3. A built container, given a read-only source bind-mount, a read-write dest
   bind-mount, and a local catalog volume, organizes photos into the dest and
   keeps `catalog.db` on the local volume.
4. Originals are never modified; every organized file remains self-verifying;
   re-runs are resumable — all existing invariants preserved.
5. The full existing test suite plus the new tests pass.
