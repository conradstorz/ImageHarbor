# Roadmap

## Where this fits

Four projects, two jobs.

**ImageHarbor is the daily tool.** It keeps a personal photo collection
organized and identifies what is in it — where a photo was taken and who is in
it. It runs continuously, and it is the only one of the four that does.

**The Takeout trio are occasional tools.** They exist to acquire files from
Google's cloud and work out what those files are. You run them when an export
lands, not every day.

| Project | Job | Cadence |
|---|---|---|
| [Google-Takeout-Downloader](https://github.com/conradstorz/Google-Takeout-Downloader) | Fetch the archives without babysitting them. | Once per export |
| [Takeout Scout](https://github.com/conradstorz/Takeout-Scout) | Explore an export in a browser. Human-facing. | Once per export |
| [Takeout_Inventory](https://github.com/conradstorz/Takeout_Inventory) | Catalog an export and publish a machine-readable pairing index. | Once per export |
| **ImageHarbor** | Organize, verify and identify the photo library itself. | Continuous |

The trio hands off to ImageHarbor: `imageharbor takeout ingest` consumes the
archives, and `--takeout-index` consumes the SQLite index `Takeout_Inventory`
publishes.

## Where it is now

In daily use as a Docker watcher. Shipped and working:

- **Facts pass** — hash, deduplicate, read EXIF, resolve a date and descriptor
  from facts alone, copy, verify, catalog. No AI required.
- **Enrichment pass** — pluggable AI backend for description and PCS
  classification. Only ever improves a file.
- **Faces pass** — local ONNX detection, embedding and clustering, with a human
  confirmation gate before any name is written.
- **Takeout ingestion** — six-rung sidecar pairing ladder, optionally
  reinforced by `Takeout_Inventory`'s index.
- **Operational dashboard** — live stats, evidence-quality tiers, work queues,
  pause/resume, and the People review queue.

## What is next

Roughly in the order it matters.

1. **Settle the Jetson overlap.** A separate image processor under development
   on Jetson hardware will do scene description *and* face finding. Scene
   description slots into the existing enrichment path cleanly; face finding
   overlaps this project's own faces pass. A second detector rescans the whole
   library, doubles face storage, and forces a choice about which embeddings the
   named clusters belong to. Decide before investing hours naming clusters.

2. **Find a calibration route for the faces threshold.** `faces calibrate`
   measures a clustering threshold from photos Google already tagged with
   `people[]`. The live catalog has none — every sidecar parsed, zero tags — so
   the threshold in use was chosen by eye. A threshold nobody measured is a
   number nobody can defend.

3. **Fix the provenance basename collisions.** Preserved documents in the
   `orphaned/` and `albums/` buckets are stored under their bare basename, so
   two same-named documents in one archive overwrite each other silently.

4. **Bring the AI backend back.** The Ollama endpoint is offline on purpose
   while its replacement is built. The enrichment pass logging connection errors
   and an open circuit breaker every cycle is expected, not a fault.

## Settled, and not up for revisiting

These are invariants, not preferences. Changing one is a redesign.

- **Originals are immutable.** Nothing in this project writes to a source file.
- **AI is never load-bearing.** A run with the AI backend permanently offline is
  a *finished* run, not a degraded one.
- **Faces never appear in a filename.** Identity lives in the catalog and the
  sidecar.
- **No name reaches a photo without a human confirming the cluster.**
- **Sidecars are append-only.** A previously recorded value is never lost.

## Licence

AGPL-3.0-or-later. This matters to the roadmap: anything that consumes this
project by *importing* it inherits the network clause. `Takeout-Scout`
(GPL-3.0-or-later) deliberately runs `Takeout_Inventory` as a subprocess for
exactly this reason, and the same boundary applies here.
