# Design: facts-first pipeline with monotonic naming

**Date:** 2026-08-11
**Status:** Approved (brainstorming)
**Scope:** `imageharbor/date_resolver.py` (new), `imageharbor/descriptor.py` (new),
`imageharbor/tiers.py` (new), `imageharbor/enrich.py` (new), `imageharbor/pipeline.py`,
`imageharbor/filename.py`, `imageharbor/catalog.py`, `imageharbor/sidecar.py`,
`imageharbor/watcher.py`, `imageharbor/cli.py`, plus tests.

## Motivation

The pipeline currently derives a photo's *destination folder* from its PCS class,
and PCS class comes from the AI backend. Every organizing decision is therefore
downstream of a model that is slow, remote, and sometimes down. A run with no AI
produces nothing.

Two consequences drove this redesign:

1. **The library cannot be organized at disk speed.** Hashing, dedup, EXIF
   extraction, and copying are pure local computation, but they sit behind an AI
   call that takes ~30 s per image against the Jetson.
2. **Re-runs can degrade a name.** With the classifier as the only descriptor
   source, a later run with a weaker or differently-behaving model can replace a
   good name with a worse one. Nothing in the system prevents this today.

This design splits the work into a **facts pass** (no AI, disk-speed, complete on
its own) and an **enrichment pass** (AI, resumable, optional), and introduces a
**tier system** that makes "a re-run never degrades a file" a checkable predicate
rather than an aspiration.

## Goals

1. A full library organize that never blocks on, or imports, an AI backend.
2. Re-runnability: repeated runs converge to a fixed point and never lose
   information.
3. Placement derived from facts only, so it can be permanent.
4. Duplicate sources recorded as many-to-one back-pointers, not a single path.
5. Cumulative sidecars that accrete metadata across runs and preserve hand edits.
6. Preserve every existing content-addressing and read-only-originals invariant.

## Non-goals

Named explicitly so implementation does not drift into them:

- **Google Takeout ingest.** Date tier 30 is reserved for it; nothing is built.
- **Video support.** `SUPPORTED_EXTENSIONS` is unchanged.
- **Reverse geocoding / place in the filename.** Requires a geocoding dependency.
- **People tags or face recognition.**
- **Suggestion / report mode.** This design writes; it does not propose.
- **Migration of the existing organized tree.** Superseded by rebuild (below).

## Decisions (locked during brainstorming)

| # | Decision | Rationale |
|---|---|---|
| 1 | Filenames must be human-comprehensible and self-describing | A name should survive being copied out of the tree |
| 2 | Date is the canonical folder tree; PCS moves to the sidecar | Placement must be fact-derived so it can be permanent |
| 3 | PCS code is removed from the filename | Class is now revisable; keeping it in the name would force renames |
| 4 | Undated files go to `Undated/`, never a guessed year | The project's character is refusing to assert the unverifiable |
| 5 | File mtime does **not** establish a date | mtime is frequently an artifact of copying, not capture |
| 6 | A human-authored original filename outranks any AI subject | The person who typed it knew something the model does not |
| 7 | Existing library is rebuilt from source, not migrated | Content-addressed + resumable makes rebuild nearly free |

## Architecture

### Two passes

**Facts pass** — `imageharbor process` (same verb, narrowed job):

```
discover → hash → dedup → EXIF → resolve date → resolve descriptor
        → copy → verify → catalog upsert → sidecar merge
```

Imports no classifier and makes no network call. Its output is a complete,
verified, deduplicated, organized library. A run with the backend permanently
offline is a *finished* run, not a degraded one.

**Enrichment pass** — `imageharbor enrich` (new verb):

```
select rows where enriched_at IS NULL → describe → concept_map.class_for
       → (pick_class fallback + remember) → taxonomy.resolve_or_create
       → catalog update → sidecar merge → conditional rename
```

Enrichment reads the **organized copy**, not the source. The copy's bytes are
verified identical to the original, so enrichment runs correctly when the source
NAS is unmounted or offline.

The circuit breaker and poison-file quarantine move to the enrichment pass
unchanged. They were always about AI-backend health; confining them to the pass
that has an AI backend is a simplification, not a rewrite.

### CLI surface

`process` loses `--ai`, `--ai-*`, and all `--breaker-*` / `--poison-*` flags;
they move to `enrich`, which takes the same source-agnostic options plus
`--reclassify`. `watch` keeps every flag it has, since it drives both passes.
`--sidecar/--no-sidecar` is unchanged and sidecars remain **optional** — the
catalog stays the source of truth, and a sidecar is a portable projection of it.

### Watcher

`imageharbor watch` runs the facts pass on its existing interval, and the
enrichment pass as a second, lower-priority phase after each facts sweep. A
tripped breaker aborts the enrichment phase only; the facts phase continues at
full speed.

## The tier system

Two independent integer ladders, persisted in the catalog. Gaps are intentional,
so future sources slot in without renumbering — the same append-only discipline
the PCS taxonomy uses.

### Date tier — decides placement

| Rank | Source id | Origin |
|---|---|---|
| 40 | `exif_original` | EXIF `DateTimeOriginal` |
| 30 | `external_sidecar` | *(reserved)* Takeout `photoTakenTime` |
| 20 | `exif_other` | `DateTimeDigitized`, `DateTime` |
| 10 | `filename_pattern` | Date parsed from the original filename |
| 0 | `none` | No trustworthy date → `Undated/` |

File mtime is deliberately absent. It is not evidence of capture time.

### Descriptor tier — decides the name

| Rank | Source id | Origin |
|---|---|---|
| 30 | `human_filename` | Original stem not matching a camera pattern |
| 20 | `ai_subject` | Classifier `primary_subject` |
| 0 | `none` | No descriptor available |

### The monotonicity rule

> A file is renamed or moved **only** when the proposed
> `(date_tier, descriptor_tier)` is strictly greater in at least one dimension
> and lower in neither.

Everything else follows from this single predicate:

- **A re-run at equal tier is a no-op.** The second AI answer does not displace
  the first. Repeated enrichment is stable; this is the "idempotent within
  reason" property.
- **An AI subject (20) cannot displace a human filename (30).** Information is
  protected structurally, not by care at the call site.
- **The only automatic move is `Undated/ → YYYY/YYYY-MM/`** — date tier `0 →
  anything`. A move between two *known* dates happens only when the new date
  comes from a strictly higher-ranked source — a tier-30 Takeout timestamp may
  correct a tier-10 filename guess, but two dates of equal rank never swap.
  This is what lets a weak filename-derived date self-correct once real
  evidence arrives; without it a wrong guess would be permanent.
- `--reclassify` bypasses the predicate for a deliberate re-do.

### What actually triggers a date upgrade

With Takeout out of scope, the live trigger is **a duplicate arriving from a
better-named path**. Identical bytes mean identical EXIF, but not identical
filenames: a photo first seen as `IMG_1234.jpg` (no EXIF, tier 0, `Undated/`) and
later found at `2019-07-04 party.jpg` yields a tier-10 date, graduating the file
to `2019/2019-07/`. The same holds for descriptors — a duplicate found at
`Emma's graduation.jpg` upgrades a tier-0 or tier-20 name to tier 30. Dedup
therefore does real organizing work rather than merely skipping copies, and the
`Undated/ → dated` path is exercised in production, not only by tests.

## Layout and filename

```
2019/2019-07/2019-07-04-emmas-graduation_<43-char-digest>.jpg
2019/2019-07/2019-07-04_<43-char-digest>.jpg           # no descriptor yet
Undated/beach-trip-scan_<43-char-digest>.jpg
Undated/<43-char-digest>.jpg                            # neither
```

Folder granularity is `YYYY/YYYY-MM/`. Year-only produces unmanageably large
directories on a decade-scale library; day-level produces thousands of sparse
ones.

Filename grammar — both prefix components optional:

```
[<YYYY-MM-DD>][-<descriptor>]_<43-char-digest>.<ext>
```

Descriptor rules are unchanged from `filename.normalize_descriptor`: lowercase,
ASCII alphanumeric, hyphen-joined, 1–3 words, ≤30 chars. The 100-char filename
cap holds; removing the PCS prefix grows the human-readable budget to ~52 chars.

**The critical parsing invariant is preserved.** The digest is still located by
counting `SHA256_B64URL_LEN` characters back from the end of the stem, never by
splitting on `_`. `hashing.extract_digest_from_stem` is unchanged. Only
`filename.parse_filename`'s interpretation of the *prefix* changes: everything
before the digest separator is split into an optional leading `YYYY-MM-DD` and an
optional descriptor.

### Camera-generated filename patterns

Matched case-insensitively against the original stem; a match means "no human
information here," so the descriptor falls to tier 0 and awaits the AI:

```
IMG_1234, IMG-20190704-WA0001, DSC0042, DSCN0042, DSCF0042, _DSC0042,
PXL_20190704_123456789, MVIMG_20190704_123456, P1000042, PICT0042,
100_0042, CIMG0042, SAM_0042, GOPR0042, DJI_0042,
Screenshot_2019-07-04-12-33-11, Screen Shot 2019-07-04 at 12.33.11,
WhatsApp Image 2019-07-04 at 12.33.11, Signal-2019-07-04-123311,
FB_IMG_1562243591, received_101234567890,
20190704_123456, 2019-07-04 12.33.11, 1562243591 (bare epoch)
```

The list lives in `descriptor.py` as an ordered tuple of compiled patterns, kept
adjacent to its fixture table so additions arrive with tests.

Note the overlap with date extraction: many camera patterns *contain* a usable
date. `date_resolver` and `descriptor` both inspect the original stem, but answer
different questions — "is there a date here?" and "is there meaning here?" — and
a single filename commonly answers yes to the first and no to the second.

## Data model

### `sources` table (new)

```sql
CREATE TABLE IF NOT EXISTS sources (
    sha256_b64url TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    size          INTEGER,
    mtime_ns      INTEGER,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    PRIMARY KEY (sha256_b64url, source_path)
);
CREATE INDEX IF NOT EXISTS idx_sources_digest ON sources(sha256_b64url);
```

One row per distinct source path. `photos` retains the single canonical
`organized_path`. Dedup becomes: digest already in `photos` → upsert a `sources`
row, skip the copy. Three copies of one photo across three exports yield one
organized file and three back-pointers.

`photos.original_path` is retained as the *first* source seen, for backward
compatibility with existing queries and the `catalog get` command.

### New `photos` columns

`date_value`, `date_tier`, `date_source`, `descriptor_value`, `descriptor_tier`,
`descriptor_source`, `enriched_at`. The tiers back the monotonicity predicate;
`enriched_at IS NULL` is the enrichment pass's work queue.

Added via additive `ALTER TABLE` guarded by a column-existence check, matching
how `catalog.py` already handles schema setup.

### Sidecar — cumulative

Read → merge → atomic replace (temp file + `os.replace`). Unknown keys are
preserved so hand edits and future fields survive a re-run.

```json
{
  "schema_version": 1,
  "identity":   { "sha256_b64url": "...", "size": 3847221, "ext": "jpg" },
  "sources":    [ { "path": "...", "first_seen": "...", "last_seen": "..." } ],
  "date":       { "value": "2019-07-04T12:33:11", "tier": 40,
                  "source": "exif_original" },
  "descriptor": { "value": "emmas-graduation", "tier": 30,
                  "source": "human_filename" },
  "exif":       { },
  "classification": {
      "pcs_code": "330", "folder_path": "300-places/330-beach",
      "primary_subject": "beach", "scene": "...", "caption": "...",
      "objects": [], "tags": [], "ocr_text": "", "model_version": "..."
  },
  "history":    [ { "pass": "facts", "at": "...", "action": "copied" } ]
}
```

The PCS taxonomy survives intact inside `classification` — fully expressive, and
now revisable without moving a byte on disk.

## Error handling and crash safety

**Renames** are the one new mutation, and get an explicit ordering: **rename the
file, then update the catalog.** A crash between the two leaves the catalog
pointing at a path that no longer exists — which is self-healing, because the
file is content-addressed. Recovery globs the expected folder for
`*_<digest>.<ext>`, confirms the digest, and repairs the row. The recovery path
uses the same property the rest of the system is built on.

**Enrichment failure** leaves the file at its current tier, which is valid by
construction: the facts pass already gave it a real name and a real home. There
is no partial state to reconcile and nothing to roll back.

**Copy failure** retains the existing behavior — the copy is deleted, an error is
raised, and nothing enters the catalog unverified.

**Poison quarantine** continues to count only failures observed while the breaker
is closed, and now applies only within the enrichment pass. A file that cannot be
described is still fully organized by the facts pass; quarantine means "stop
asking the model about this one," not "set this file aside."

## Testing

Per-module unit tests, plus four property-shaped suites carrying the guarantees:

1. **Monotonicity table** — every `(old_tier, new_tier)` pair asserts
   rename-iff-strictly-better, across both dimensions including the mixed cases
   (better date, worse descriptor → no change). This is the spec's central claim
   and gets exhaustive rather than exemplary coverage.
2. **Re-run stability** — `facts → facts` yields zero changes on the second run.
   `facts → enrich → facts → enrich` reaches a fixed point and stays there.
3. **AI-down non-degradation** — enrichment with a classifier raising on every
   call asserts no name, path, or catalog field changes.
4. **Fixture tables** — real-world filenames → camera-vs-human verdict; EXIF and
   filename combinations → expected date tier and destination folder.

Plus: identical bytes at three source paths → one organized file and three
`sources` rows; a sidecar with a hand-added key survives a re-run; a rename
interrupted before the catalog update is recovered by digest lookup.

## Rebuild, not migration

The existing hpz440 library is rebuilt by pointing the facts pass at the same
source with a fresh destination. No migration tool is written. The
`learned_concepts` and `taxonomy` tables are copied forward from the old catalog
so previously-paid AI calls are not repeated; this is a table copy, not a
transformation. The old tree is deleted after the new one is reviewed.

## Consequences for existing docs

`CLAUDE.md` and `docs/genesis-roadmap.md` both describe the PCS folder tree as
the destination and the PCS code as a filename component. Both statements become
false. `CLAUDE.md` is updated as part of implementation; the roadmap is a
historical document and gets a short note pointing here rather than an edit.
