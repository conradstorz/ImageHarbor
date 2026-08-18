# Lossless sidecars — design

**Date:** 2026-08-18
**Status:** approved, not yet implemented
**Extends:** [`2026-08-11-facts-first-pipeline-design.md`](2026-08-11-facts-first-pipeline-design.md),
[`2026-08-12-google-takeout-ingestion-design.md`](2026-08-12-google-takeout-ingestion-design.md)

## Goal

Make the per-image JSON sidecar the complete, permanent record of everything
ImageHarbor has ever learned about a photo — from any source, at any time —
under one rule:

> **A sidecar may gain information. It may never lose any.**

Three changes deliver that:

1. **Sidecars become the default.** `--no-sidecar` opts out; nothing else has to
   be remembered.
2. **The merge becomes append-only.** A superseded value moves into a history
   list rather than being overwritten, and lists accumulate rather than replace.
3. **Source documents are preserved verbatim.** Google's own JSON is stored
   byte-for-byte alongside the parsed view, so fields nobody has thought about
   yet survive with no code change.

## Non-goals

- **Making the catalog the source of truth for sidecar content.** The catalog
  stays a rebuildable index; the sidecar stays the portable projection that
  survives catalog loss and tolerates hand edits. Reversing that was considered
  and rejected — it would break hand-editing, which the never-lose rule exists
  in part to protect.
- **Recovering Google metadata for already-organized files.** `sidecar backfill`
  writes what the catalog holds. Google's data is only recoverable by
  re-ingesting the original archives.
- **Attaching Picasa face tags to individual photos.** They carry no photo
  reference; see "Ground truth" below. They are preserved, not attached.
- **A sidecar query/search verb.** The catalog already answers queries.

## Ground truth

Measured against `takeout-20230618T004316Z-001.zip` (79 MB, 196 members,
AlbumArchive schema) and against the organized output of ingesting it at
commit `a434a72`.

**What the current sidecar drops.** Of Google's nine per-photo fields:

| Field | Present | Fate today |
|---|---|---|
| `photoTakenTime` | 86/86 | kept, load-bearing |
| `title` | 86/86 | kept |
| `creationTime` | 86/86 | kept, provenance only |
| `exif` | 52/86 | kept |
| `geoData` | 49/86 | kept |
| `sizeBytes` | 86/86 | parsed, then discarded |
| `imageViews` | 86/86 | never read |
| `height` / `width` | 86/86 | never read |

**What whole files hold.** Four member kinds are classified and never opened:

- `Albums.json` (×21) — the album's real title (e.g. `"Hangout: Conrad Storz ●
  Herbie (Tony) Hughes"`), access level, and date. A parser for this file
  exists in `takeout/metadata.py` and **is never called** by anything.
- `picasa_web_album_face_tags.json` (385 KB) — **1,496 entries, 1,469 of them
  named, 73 distinct people**, each with a bounding box.
- `picasa_web_album_tags.json` — 49 media keys carrying 12 distinct free-text
  tags (e.g. `"annie's camera"`).
- `archive_browser.html` — Google's offline viewer.

**The face tags cannot be attached to photos, and this is load-bearing for the
design.** Each entry carries only `tag`, `boundingBox`, and `creationTime` —
no filename and no `mediaKey`. Independently, **0 of the 86 per-photo sidecars
carry a `mediaKey`**, so the album tags cannot be joined either. The bounding
boxes' `width`/`height` are the source image's dimensions (285 distinct pairs,
735 entries sharing `1600×1200`), which makes a dimension-based join ambiguous
for the largest cluster. Guessing was rejected: this project's pairing rule is
already "never guess", and a face attached to the wrong photo is exactly the
silent corruption that rule exists to prevent.

**Eight sidecars have no media member in this part** — including
`P1010089.JPG(1).json` and `P1010089.JPG(2).json`, whose photos live in an
archive part not present. Today they are classified `parsed` and forgotten.

**Sizes.** Google's per-photo JSON averages 503 B (max 618 B). Current
ImageHarbor sidecars average 2,540 B. Embedding the raw document costs roughly
20% per sidecar.

## The bug this fixes

`sidecar._deep_merge` merges nested dicts key by key but **replaces lists
wholesale**:

```python
merged[key] = value   # lists and scalars replace
```

So merging `{"people": []}` over `{"people": ["Judy", "Pete"]}` discards both
names. The current docstring acknowledges this ("callers that own a list must
pass the complete value") and pushes the burden onto every call site. That is
the opposite of the guarantee this design commits to, and it is a live data-loss
path the moment any caller passes a partial list.

## Architecture

The merge policy is the part most likely to be wrong, so it gets no I/O — the
same split that made `takeout/metadata.py` and `takeout/pairing.py`
exhaustively testable.

| Module | Purpose | I/O |
|---|---|---|
| `sidecar_schema.py` | **New.** The merge policy: field shapes, history, list keying, migration from v1. Pure functions over dicts. | **no** |
| `sidecar.py` | Read, atomic write, and the `merge_sidecar` entry point. Delegates all policy to `sidecar_schema`. | yes |

`sidecar.py` keeps its current public surface (`sidecar_path_for`,
`read_sidecar`, `merge_sidecar`) so no caller changes.

### `sidecar_schema.py`

```python
SCHEMA_VERSION = 2

def merge(base: dict, updates: dict, *, observed_at: str) -> dict: ...
def migrate(doc: dict) -> dict: ...          # v1 -> v2, itself lossless
def is_noop(base: dict, merged: dict) -> bool: ...
```

`merge` is pure and total: it never raises, and for any input returns a document
containing every value present in either argument.

## Schema v2

Current values sit at the top level for readability; superseded ones move
beneath them.

```json
{
  "schema_version": 2,
  "identity": { "sha256_b64url": "…", "size": 408242, "ext": "jpg" },

  "date": {
    "value": "2008-12-12T12:32:27", "tier": 30, "source": "external_sidecar",
    "history": [
      { "value": "2005-08-11T18:57:39", "tier": 20, "source": "exif_other",
        "observed_at": "2026-08-17T18:51:18Z",
        "superseded_at": "2026-08-19T09:02:44Z" }
    ]
  },
  "descriptor":     { "value": "…", "tier": 30, "source": "…", "history": [] },
  "classification": { "code": "330", "label": "…", "folder_path": "…",
                      "model_version": "…", "history": [] },

  "sources": [
    { "path": "/nas/t-001.zip!Takeout/AlbumArchive/Blogger/2008-12-12-Blogger Pictures/P1010089.JPG",
      "folder": "2008-12-12-Blogger Pictures",
      "first_seen": "…", "last_seen": "…" }
  ],
  "albums": [
    { "title": "Hangout: Conrad Storz ● Herbie (Tony) Hughes",
      "folder": "2008-12-12-Blogger Pictures",
      "archive_id": "rZRl…", "access": "protected", "date": "2018-04-25T16:43:27" }
  ],
  "people": [
    { "name": "Emma", "source": "google_photos_people" }
  ],
  "exif": { "Make": "OLYMPUS OPTICAL CO.,LTD", "…": "…" },
  "exif_history": [
    { "key": "Orientation", "value": 6.0, "observed_at": "…", "superseded_at": "…" }
  ],

  "provenance": [
    { "kind": "takeout_media_json",
      "archive_id": "rZRl…", "archive": "takeout-20230618T004316Z-001.zip",
      "member": "Takeout/AlbumArchive/Blogger/2008-12-12-Blogger Pictures/P1010089.JPG.json",
      "observed_at": "2026-08-17T18:51:18Z",
      "digest": "<sha256_b64url of the raw bytes>",
      "raw": { "title": "P1010089.JPG", "imageViews": "12", "height": "1600", "…": "…" } }
  ]
}
```

`provenance[].raw` is Google's document verbatim. `imageViews`, `height`,
`width`, and `sizeBytes` return through it without being modelled, and so will
any field Google adds later.

## Merge policy

Five rules. Each list is keyed so that re-observing the same fact is a no-op.

`people[]` is populated from a Google Photos export's inline `people` field
(names only). It is **not** populated from the Picasa face tags, which carry no
photo reference — `bounding_box` therefore appears only if some future source
supplies both a name and a box for an identified photo.

| Shape | Fields | Rule | Identity key |
|---|---|---|---|
| Tiered scalar | `date`, `descriptor` | Higher `tier` wins. Equal tier keeps the incumbent. The loser appends to `history[]` with `observed_at` and `superseded_at`. | — |
| Untiered scalar block | `classification` | A changed `code` appends the previous block to `history[]`. Equal is a no-op. | — |
| Keyed list | `sources`, `albums`, `people` | Append-only union. Existing entries update `last_seen` only. | `path` · `(archive_id, folder)` · `(name, bounding_box)` |
| Raw list | `provenance` | Append-only. A document whose digest is already present is skipped entirely. | `digest` |
| Flat map | `exif`, `identity` | Key-by-key. A changed value appends `{key, value, observed_at}` to `exif_history[]`. | key |
| Anything else | unknown / hand-written keys | Preserved untouched. | — |

**The guarantee, stated so it can be tested:** for any base `B` and updates `U`,
`merge(B, U)` contains every value present in `B` and every value present in
`U`. Nothing is removed; a superseded value is relocated, never dropped.

**Idempotence follows from the keying:** `merge(merge(B, U), U)` is byte-identical
to `merge(B, U)`. A repeated `process`, `enrich`, or `takeout ingest` run must
leave every sidecar unchanged, matching the idempotency discipline the rest of
the system already holds.

## Capture

Seven inputs write into a sidecar. Each owns its own keys and none coordinates
with another.

| Input | Writes | Notes |
|---|---|---|
| Facts pass (`pipeline.py`) | `identity`, `sources`, `date`, `descriptor`, `exif` | `sources[].folder` is the directory the file was found in — for a plain `process` run as well as for Takeout |
| Enrichment pass (`enrich.py`) | `classification` | Gains history; `model_version` recorded per observation |
| Takeout per-media JSON | `provenance[]`, `albums[]`, `people[]`, and the evidence feeding `date`/`descriptor` | Raw document preserved verbatim |
| `Albums.json` | `albums[].title`, `access`, `date` | **Activates `takeout/metadata.parse_album_metadata`, which currently exists and is never called** |
| Archive-level documents | the provenance room, below | Not per-photo |
| Backfill verb | whatever the catalog holds, plus a fresh EXIF read | See "Migration and backfill" |
| Hand edits | anything | Never touched by any pass |

### Album membership

A photo's album is the immediate parent directory of its member path. The
album's *title* comes from the `Albums.json` sitting in that directory. When no
`Albums.json` exists, the entry records `folder` with `title: null` — the
directory name is a fact even when the human-readable title is absent.

A photo appearing in several archives accumulates the union of its albums, since
`albums[]` is keyed by `(archive_id, folder)`.

## The provenance room

Every archive member that is **not** an image or a video is preserved verbatim
under the organized root:

```
Photos-Organized/
  .takeout-provenance/
    <archive_id>/
      manifest.json                       # archive name, size, ingested_at,
                                          # full member inventory with digests
      picasa_web_album_face_tags.json     # verbatim
      picasa_web_album_tags.json          # verbatim
      archive_browser.html                # verbatim
      albums/2008-12-12-Blogger Pictures/Albums.json
      orphaned/P1010089.JPG(1).json       # sidecar whose media is in another part
```

The rule is deliberately uncurated: **preserve everything that is not media.**
Deciding which unknown file is worth keeping is precisely where "never lose"
degrades into "lose the thing nobody thought about". `archive_browser.html`
(169 KB per archive) is kept for that reason alone.

`manifest.json` records a digest per preserved document, so re-ingesting an
archive rewrites nothing.

`orphaned/` holds media JSON whose photo is absent from the batch. If the
missing archive part later arrives, the existing second-pass reopen mechanism
pairs it normally; the orphaned copy is then redundant but harmless.

The directory is git-ignored alongside `.takeout-staging/`.

## CLI

`--sidecar/--no-sidecar` changes to `default=True` on `process`, `enrich`,
`watch`, and `takeout ingest`. No flag is renamed; the opt-out already exists.

One new verb, following the `catalog list`/`catalog get` group precedent:

```
imageharbor sidecar backfill --dest DEST [--catalog PATH] [--dry-run]
```

```
$ imageharbor sidecar backfill --dest /nas/Photos-Organized --dry-run
  4156 cataloged / 4156 missing a sidecar / 0 present
  [DRY-RUN] nothing written

$ imageharbor sidecar backfill --dest /nas/Photos-Organized
  written 4156 / unchanged 0 / failed 0
```

`--dry-run` doubles as the audit, so there is no separate `verify` verb.

## Migration and backfill

**v1 → v2 is a merge, not a conversion**, and therefore needs no separate
correctness argument: it obeys the same never-lose rule as every other write.

- scalar blocks gain an empty `history[]`
- the v1 `takeout` block moves into `provenance[]` (as `kind:
  "imageharbor_v1_takeout_block"`, since the original raw document is not
  available) and its album folder into `albums[]`
- `schema_version` becomes `2`
- every unrecognized v1 key is carried across untouched

**Backfill writes what the catalog holds** — identity, sources with folder,
date and descriptor with tiers and sources — plus a fresh EXIF read from the
organized file, which is already on disk and cheap. `provenance[]` is empty for
backfilled files. Recovering Google's metadata for an already-organized library
requires re-ingesting the original archives.

Backfill **merges** rather than skips. A file that already has a sidecar gets the
catalog's view merged into it, which the never-lose rule makes safe and which a
skip would not — a sidecar written by an older version would otherwise stay thin
forever. Merging an already-complete sidecar is a no-op by construction, so the
run reports it as unchanged rather than rewriting it.

Backfill is resumable, and a sidecar write failure is recorded and does not stop
the run.

## Changes to existing modules

| File | Change |
|---|---|
| `sidecar_schema.py` | **new** — pure merge policy, history, keying, v1 migration |
| `sidecar.py` | delegate merging to `sidecar_schema`; keep the public surface; remove `_deep_merge` |
| `pipeline.py` | write `sources[].folder`; no behavioral change otherwise |
| `enrich.py` | **no change needed** — it has written `model_version` into the classification block since `d14f06b`. Verified 2026-08-18; the original change table asserted this without checking. What the block gains is *significance*: history now makes it possible to tell which model produced which answer. |
| `takeout/ingest.py` | write `provenance[].raw`; populate `albums[]` from `Albums.json`; write the provenance room; record orphaned sidecars |
| `takeout/metadata.py` | `parse_album_metadata` gains `access` and `date`; becomes reachable |
| `cli.py` | flip four `--sidecar` defaults; add the `sidecar` group with `backfill` |
| `.gitignore` | add `.takeout-provenance/` |

## Error handling

Unchanged in principle: **a sidecar failure must never fail an image that is
already copied, verified, and catalogued.** Every sidecar write stays inside a
try/except that logs and continues.

Two additions:

- A malformed existing sidecar is currently treated as empty, which under the
  new rule would *destroy* data. Instead: the unreadable file is renamed to
  `<name>.json.corrupt-<timestamp>` and a fresh sidecar is written. Nothing is
  overwritten in place.
- A provenance-room write failure is logged and does not fail the member; the
  archive remains the original.

## Testing

The merge policy is pure, so its tests are exhaustive rather than illustrative.

**`tests/test_sidecar_schema.py`** — the core.

- **The never-lose property, over randomized merge sequences.** For a generated
  sequence of merges, assert every value ever written is present in the final
  document — at top level or in a history list. This is the formal statement of
  the guarantee, not a sample of it.
- **Idempotence.** `merge(merge(B, U), U) == merge(B, U)`, byte-identical.
- **Monotonic size.** The merged document is never smaller than the base.
- **The list-truncation bug, pinned.** `{"people": []}` merged over
  `{"people": ["Judy", "Pete"]}` keeps both names. This fails against today's
  `_deep_merge`.
- **Tier rules.** Higher tier wins and demotes the incumbent to history; equal
  tier is a no-op and does not append history.
- **Keying.** Re-observing the same source path, album, person, or provenance
  digest adds nothing.
- **Hand-written keys** survive every merge shape.
- **v1 migration** preserves every v1 key, including unrecognized ones.

**`tests/test_sidecar.py`** — I/O.

- Atomic write survives an interrupted run.
- A corrupt existing sidecar is renamed, not overwritten, and the new file is
  complete.

**`tests/test_takeout_ingest.py`** — capture.

- `provenance[].raw` matches the archive member's bytes exactly.
- `albums[]` carries the title from `Albums.json`, and `folder` with a null
  title when no `Albums.json` exists.
- The provenance room contains every non-media member; re-ingesting writes
  nothing new.
- An orphaned media JSON lands in `orphaned/`.
- The same photo in two archives accumulates two `provenance` entries and the
  union of albums.

**`tests/test_cli.py`** — surface.

- `process` writes a sidecar with no flag; `--no-sidecar` suppresses it.
- `sidecar backfill --dry-run` reports counts and writes nothing.
- `sidecar backfill` writes one sidecar per cataloged file and is idempotent.

**Real-export round trip** — ingest the calibrating archive, re-ingest, assert
every sidecar is byte-identical; add a second archive and assert only additions.

## Costs, stated

- Sidecars grow from ~2.5 KB to roughly 3.5 KB with raw provenance — about
  350 MB per 100k photos.
- The provenance room adds ~400 KB per archive for this export, dominated by
  the face-tags file and the HTML viewer.
- A `--no-sidecar` run now diverges from the default. Anyone scripting against
  current behavior will see sidecar files appear where none did before.

## Open items

- **The Picasa face tags remain unattached.** 1,469 named entries covering 73
  people are preserved verbatim and joined to nothing. If a future export
  supplies a `mediaKey` in the per-photo JSON, the join becomes possible and the
  preserved documents make it retroactive. Until then, attaching them would
  require guessing.
- **A newer Google Photos export supplies `people` inline** (names, no bounding
  boxes) in each photo's JSON. `takeout/metadata.py` already parses that field;
  it simply never appears in the AlbumArchive schema. Re-exporting from Google
  Photos proper would populate `people[]` without any code change.
- **`exif_history[]` has no consumer.** It is written because the never-lose
  rule requires it, but nothing reads it yet. If it proves noisy in practice —
  EXIF re-reads should be stable for identical bytes — it is a candidate for
  removal, not a candidate for silently overwriting.
