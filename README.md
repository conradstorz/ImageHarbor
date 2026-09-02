# ImageHarbor

ImageHarbor: Classify. Verify. Preserve.

> **Where this fits.** ImageHarbor is the *daily* tool — it keeps a photo
> library organized and identifies what is in it, continuously. Three companion
> tools handle the *occasional* job of getting files out of Google's cloud and
> working out what they are:
> [Google-Takeout-Downloader](https://github.com/conradstorz/Google-Takeout-Downloader)
> fetches an export, [Takeout Scout](https://github.com/conradstorz/Takeout-Scout)
> explores it in a browser, and
> [Takeout_Inventory](https://github.com/conradstorz/Takeout_Inventory) catalogs
> it for machines. See [ROADMAP.md](ROADMAP.md).

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

`imageharbor watch` runs both passes continuously against a source directory
(plus the optional third pass below, with `--faces`).

## Faces (optional)

`imageharbor faces` detects faces in already-organized photos, embeds and
clusters them, and proposes names from the people Google Photos already
tagged in those same photos — entirely in-process, against local ONNX
models. It needs no account and no API key, and no network beyond a one-time
~261 MB model download (`imageharbor faces models download`). Faces never
rename or move a file and **never appear in a filename** — identity lives in
the catalog and, once confirmed, in the photo's JSON sidecar. **No name is
written to a photo until a human confirms that cluster** in the operational
dashboard's People review queue; until then, a cluster carries only a
machine proposal.

The clustering threshold has no sensible default — measure it from your own
library before clustering, never guess it:

```
imageharbor faces scan      --dest DEST
imageharbor faces calibrate --dest DEST
imageharbor faces cluster   --dest DEST --threshold <measured value>
```

`imageharbor watch --faces` runs scanning and confirmed-name propagation as a
third continuous pass every cycle; whole-library reclustering only runs when
there's enough new, unclustered work to justify it (or no clustering has
happened yet), never on every poll.

If a Takeout export preserved a Picasa contact roster
(`.takeout-provenance/<archive_id>/contacts.xml`), `imageharbor faces roster
--dest DEST` imports its names into the review UI's autocomplete list. The
roster carries no photo reference at all, so those names seed vocabulary
only — they are never attached to a cluster or a photo. Running it against a
library with no such file is the normal case and reports `0`, not an error.

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
| `imageharbor takeout survey --archives DIR` | Measure an archive set and report what ingestion would do with it. Read-only and standalone: no catalog, no destination, no AI backend, no network. |
| `imageharbor watch --source SRC --dest DEST` | Continuously run the facts and enrichment passes (and the faces pass, with `--faces`) on an interval. |
| `imageharbor verify DEST` | Re-verify every organized file's digest against its filename. |
| `imageharbor sidecar backfill --dest DEST` | Rebuild/merge sidecars for a library organized before sidecars were the default. Cannot recover Google Takeout metadata for already-organized files — that requires re-ingesting the original archives. |
| `imageharbor faces scan --dest DEST` | Detect and embed faces in organized photos; `faces calibrate`/`faces cluster`/`faces status` group and review them (needs the optional `faces` extra). |
| `imageharbor faces roster --dest DEST` | Import a preserved Picasa contact roster's names as autocomplete vocabulary, if the export had one — never attached to a cluster or photo. |

Run `imageharbor --help` (or `<command> --help`) for the full flag list.

## Google Takeout ingestion

`imageharbor takeout ingest --archives DIR --dest DEST` pairs each photo and
video in a set of Google Takeout `.zip` exports with its Google JSON sidecar
using a built-in six-rung matching ladder (exact name, the newer
`supplemental-metadata` spelling, copy-suffix, `-edited` derivatives,
case-insensitive retry, and truncation recovery) — no other tooling is
required.

`--takeout-index PATH` is **optional**. It points at a `takeout-index.sqlite`
published by the separate `Takeout_Inventory` tool, which can resolve some
pairings the built-in ladder deliberately never attempts (for example, a
sidecar in a different folder from its photo). A
`takeout-index.sqlite` sitting beside `--archives` is auto-detected and used
automatically; pass `--takeout-index` only to point at one that's named or
located differently. Without an index — the default — ingestion behaves
exactly as it always has.

Every pairing carries one of three confidence values, whether it came from
an index or the built-in ladder, and that value — not which engine produced
it — decides what the pairing may contribute:

| Confidence | Sidecar names... | Contributes |
|---|---|---|
| `own` | this exact file | capture date, title, face tags |
| `related` | a *different* file — usually this file's unedited original | capture date only |
| `none` | nothing (no match found) | nothing — the file organizes from EXIF and its filename alone |

A `related` pairing never contributes a title or face tags, because the
sidecar it names describes a different photograph: applying its title would
rename this file after its sibling, and its face tags belong to that other
image. Its capture date is still trusted (one tier below a sidecar that
names this file directly), because the related file is usually the same
photograph before an edit.

The raw Google JSON behind a `related` pairing is not discarded — it's
preserved as provenance the same as any other sidecar, but labelled with its
`confidence` and `pair_rule` rather than treated as authoritative. That
labelling matters because the document's own fields (GPS coordinates
included) describe the *related* file, not this one; reading them back later
without the label would silently misattribute them.

## Learn more

- [`ROADMAP.md`](ROADMAP.md) — where this sits among its companion projects,
  what is next, and what has been settled.
- [`CLAUDE.md`](CLAUDE.md) — architecture, module responsibilities, and the
  invariants that must not be broken.
- [`docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`](docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md) —
  the design behind the facts/enrichment split and the tier system.
- [`docs/deploy-docker.md`](docs/deploy-docker.md) — running ImageHarbor as a
  continuous Docker watcher against a NAS and a self-hosted AI server.

## Licence

Copyright (C) 2026 Conrad Storz. Released under the
[GNU Affero General Public License v3.0 or later](LICENSE).

Free to use, study, modify and share. Two conditions come with that:

- **Credit stays with the work.** Copyright and licence notices must be
  preserved in any copy or derivative.
- **Derivatives stay free.** Anyone who distributes a modified version — or
  runs one as a network service that other people use, which the operational
  dashboard makes easy to do — must publish its source under the same
  licence. Nobody can take this closed and sell it as their own product.

Using it on your own photo library, personally or at work, needs no
permission.
