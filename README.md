# ImageHarbor

ImageHarbor: Classify. Verify. Preserve.

ImageHarbor is a deterministic, resumable CLI that organizes a photo library. It
reads from a read-only source, never modifies originals, content-addresses every
file by SHA-256, copies it into an organized tree, verifies the copy, and records
everything in a SQLite catalog with a JSON sidecar per image. A sidecar is
written by default (`--no-sidecar` opts out) and is append-only — it accumulates
everything ever learned about a photo, from any source, and never loses a
previously recorded value.

## Two passes

Organizing is split into two independent passes:

- **Facts pass** (`imageharbor process`) — hashes, deduplicates, reads EXIF,
  resolves a capture date and a descriptor from facts alone (EXIF and the
  original filename — never AI), copies, verifies, and catalogs. It makes **no
  AI calls and requires no AI backend**. A run with the AI backend permanently
  offline is a *finished* run, not a degraded one. Google Takeout `.zip` exports
  are ingested with `imageharbor takeout ingest`, which walks the archives
  read-only and feeds each member through the same facts pass — Google's
  `photoTakenTime` supplies a capture date when EXIF has none. Videos are
  inventoried for a later project but not copied.
- **Enrichment pass** (`imageharbor enrich`) — reads the organized copies (the
  source volume need not even be mounted), describes each image with a pluggable
  AI backend, and classifies it against a self-extending PCS taxonomy. It only
  ever *improves* a file: a file is renamed or moved only when the result is
  strictly better than what's already on record (a machine-generated filename can
  be upgraded by an AI-derived subject, but an AI subject can never displace a
  human-authored name). Safe to interrupt and resume at any time.

`imageharbor watch` runs both passes continuously against a source directory.

## Operational dashboard

`imageharbor watch` serves a small operational dashboard on
`http://<host>:8080/` by default (`--dashboard-port` to change the port,
`--no-dashboard` to disable it). It reports library stats, evidence quality
(the date/descriptor tier tables above, as live counts), work queues, pass
history, and a projection of when the remaining backlog will clear — or an
honest `stalled`/`unknown` when the evidence doesn't support a number (AI
backend unreachable, paused, no recent progress). It also exposes three
controls:

- **Pause / Resume** — stops the watcher between photos (never mid-photo) in
  both passes, and survives a container restart.
- **Poll interval** — overrides `IMAGEHARBOR_INTERVAL` at runtime.
- **AI enrichment on/off** — lets the facts pass keep organizing at full speed
  while the AI backend is down or intentionally disabled.

Any override is shown with a warning line naming the config value it is
currently overriding (e.g. `⚠ overriding IMAGEHARBOR_INTERVAL=300`), with a
one-click revert back to that value. A dashboard failure (port already bound,
a query error) never stops organizing — it logs a warning and the watcher
carries on. See
[`docs/superpowers/specs/2026-08-19-dashboard-design.md`](docs/superpowers/specs/2026-08-19-dashboard-design.md)
for the full design and
[`docs/deploy-docker.md`](docs/deploy-docker.md) for reaching it in a
container.

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
| `imageharbor takeout ingest --archives DIR --dest DEST` | Ingest Google Takeout archives. |
| `imageharbor takeout status --catalog DEST/catalog.db` | Report Takeout ingestion progress. |
| `imageharbor watch --source SRC --dest DEST` | Continuously run both passes on an interval. |
| `imageharbor verify DEST` | Re-verify every organized file's digest against its filename. |
| `imageharbor sidecar backfill --dest DEST` | Rebuild/merge sidecars for a library organized before sidecars were the default. Cannot recover Google Takeout metadata for already-organized files — that requires re-ingesting the original archives. |

Run `imageharbor --help` (or `<command> --help`) for the full flag list.

## Learn more

- [`CLAUDE.md`](CLAUDE.md) — architecture, module responsibilities, and the
  invariants that must not be broken.
- [`docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`](docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md) —
  the design behind the facts/enrichment split and the tier system.
- [`docs/deploy-docker.md`](docs/deploy-docker.md) — running ImageHarbor as a
  continuous Docker watcher against a NAS and a self-hosted AI server.
