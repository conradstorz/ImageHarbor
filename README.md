# ImageHarbor

ImageHarbor: Classify. Verify. Preserve.

ImageHarbor is a deterministic, resumable CLI that organizes a photo library. It
reads from a read-only source, never modifies originals, content-addresses every
file by SHA-256, copies it into an organized tree, verifies the copy, and records
everything in a SQLite catalog with an optional JSON sidecar per image.

## Two passes

Organizing is split into two independent passes:

- **Facts pass** (`imageharbor process`) — hashes, deduplicates, reads EXIF,
  resolves a capture date and a descriptor from facts alone (EXIF and the
  original filename — never AI), copies, verifies, and catalogs. It makes **no
  AI calls and requires no AI backend**. A run with the AI backend permanently
  offline is a *finished* run, not a degraded one.
- **Enrichment pass** (`imageharbor enrich`) — reads the organized copies (the
  source volume need not even be mounted), describes each image with a pluggable
  AI backend, and classifies it against a self-extending PCS taxonomy. It only
  ever *improves* a file: a file is renamed or moved only when the result is
  strictly better than what's already on record (a machine-generated filename can
  be upgraded by an AI-derived subject, but an AI subject can never displace a
  human-authored name). Safe to interrupt and resume at any time.

`imageharbor watch` runs both passes continuously against a source directory.

## Filename grammar

```
[<YYYY-MM-DD>][-<descriptor>]_<43-char-digest>.<ext>
```

Both prefix components are optional. Placement in the tree comes from the
resolved capture date (`YYYY/YYYY-MM/`, or `Undated/` when no trustworthy date is
available) — never from PCS classification, and never a guessed year.

Example:

```
2019/2019-07/2019-07-04-emmas-graduation_I1cOwYO0nb9H_KiZ7xu4vTpYGRrp-u_wDq2Y5ChXiqA.jpg
```

PCS classification (once `enrich` has run) is recorded in the catalog and in the
image's JSON sidecar, not in the path or filename.

## Main commands

| Command | Purpose |
|---|---|
| `imageharbor process --source SRC --dest DEST` | Organize a library (facts pass, no AI). |
| `imageharbor enrich --dest DEST --ai openai` | Describe/classify already-organized images. |
| `imageharbor watch --source SRC --dest DEST` | Continuously run both passes on an interval. |
| `imageharbor verify DEST` | Re-verify every organized file's digest against its filename. |

Run `imageharbor --help` (or `<command> --help`) for the full flag list.

## Learn more

- [`CLAUDE.md`](CLAUDE.md) — architecture, module responsibilities, and the
  invariants that must not be broken.
- [`docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`](docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md) —
  the design behind the facts/enrichment split and the tier system.
- [`docs/deploy-docker.md`](docs/deploy-docker.md) — running ImageHarbor as a
  continuous Docker watcher against a NAS and a self-hosted AI server.
