# Rebuilding the organized library

ImageHarbor's PCS/date redesign (2026-08-11, see
[`docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`](superpowers/specs/2026-08-11-facts-first-pipeline-design.md))
changed where files are placed (date-derived tree instead of PCS folders) and how
they are named (`[<date>][-<descriptor>]_<digest>` instead of a PCS-prefixed
name). There is **no migration tool** — the existing hpz440 library is rebuilt
from the original source, not transformed in place. This is deliberate: the
pipeline is content-addressed and resumable, which makes a from-scratch rebuild
nearly free, and far safer than trying to reinterpret an already-organized tree.

This runbook covers rebuilding the live hpz440 deployment. See
[`docs/deploy-docker.md`](deploy-docker.md) for how the container/NAS mounts are
set up, and [`CLAUDE.md`](../CLAUDE.md) for the two-pass architecture this
produces.

**Why this runbook must be followed rather than pointing `--dest` at the old
catalog.** A pre-redesign catalog has photo rows but no `sources` rows (that
table didn't exist yet) and no `date_tier`/`descriptor_tier`/`enriched_at`
values — `Catalog._ensure_photo_columns` would make it *open* cleanly by
adding the missing columns, but every existing row would then read as
`date_tier=0` (Undated) and `enriched_at IS NULL` (never enriched). The next
`enrich` (or `watch`) pass would treat the entire existing library as
unenriched and undated, and relocate the whole already-organized tree into
`Undated/`. `Catalog.__init__` now detects this shape (photo rows present, no
`schema_version` stamp, `sources` empty) and refuses to open the catalog at
all, raising `LegacyCatalogError` and pointing here, rather than silently
corrupting placement — so skipping this runbook and pointing at the old
catalog directly fails fast instead of quietly wrecking the library.

## The core guarantee

**The source is never modified by any step below.** `process` only ever reads
from `--source` and writes to `--dest`; nothing in ImageHarbor writes to, moves,
renames, or deletes an original. That means an aborted or botched rebuild costs
nothing but disk space and time — the source tree used for the OLD organized
library is exactly as good a starting point for a retry as it was before you
began. There is no point in this procedure where a mistake can lose a photo.

## Overview

1. Stop the container running the old watcher.
2. Copy the `learned_concepts` and `taxonomy` tables from the old catalog into a
   fresh one, so previously-paid AI calls are not repeated.
3. Run `process` against the same source, into a **new** destination.
4. Spot-check the new tree.
5. Run `enrich` against the new destination.
6. Verify the new tree with `imageharbor verify`.
7. Only after all of the above look right, delete the old organized tree.

## 1. Stop the old container

```bash
ssh claude@hpz440
docker compose -f /path/to/imageharbor/docker-compose.yml down
```

Confirm nothing is still writing to the old catalog or organized tree:

```bash
docker ps | grep imageharbor   # should show nothing
```

The NAS source mount can stay mounted — it is read-only and nothing below writes
to it.

## 2. Carry the taxonomy and learned concepts forward

The old catalog's `taxonomy` table holds every PCS code that has been minted
(including any `~N` codes and adjudicated aliases), and `learned_concepts` holds
every subject → class mapping the AI has already resolved. Both are pure
classification memory — nothing in them refers to a file path — so they can be
copied wholesale into a fresh catalog before the new `process` run even starts.
This means enrichment on the rebuilt library reuses prior AI decisions instead of
re-asking the model for photos it has already classified before.

**Where the catalog actually lives on hpz440.** The commands below write
`catalog.db` inside the destination directory, which is the right shape for a
local one-off rebuild. The live hpz440 deployment does **not** do this: its
catalog is a Docker named volume (`imageharbor-catalog*` → `/data/catalog`)
precisely because SQLite over a CIFS mount corrupts, and `--dest` is a CIFS
share. On that host, substitute the volume for every `<dest>/catalog.db` path
below and pass `--catalog` explicitly (`process` has no `IMAGEHARBOR_CATALOG`
envvar — only `enrich` and `watch` do). Build the *new* catalog in a **new**
volume so the pre-redesign one survives as the rollback point.

Two ownership traps when working through containers, both of which surface as a
bare `unable to open database file` or `Permission denied`:

- A freshly created Docker volume is owned by `root`, but the image runs as
  `harbor` (uid 1000). `chown -R 1000:1000` the volume before first use.
- The host account and the container user are different uids (on hpz440,
  `claude` is 1005 vs `harbor` 1000), and the dest share is mounted
  `uid=1000,forceuid`. Files the container writes therefore **cannot be deleted
  from an ssh shell** — run `rm` inside a container that mounts the share
  instead. This applies to step 7's cleanup as much as to any scratch directory.

**Copy the tables before the first `enrich` run.** A fresh catalog's `taxonomy`
table is *empty*: seeding happens in `Taxonomy.ensure_seeded`, which `enrich.py`
calls at the start of an enrichment pass, not when the catalog is opened. That is
what makes the plain `INSERT` below safe — it would raise a UNIQUE constraint
error against a pre-seeded table. It also means the copied rows win permanently:
`ensure_seeded` early-returns once the table is non-empty, so the old (grown)
taxonomy is preserved rather than overwritten by the 9-class seed.

Create the new destination directory first (an empty `catalog.db` there is fine —
`Catalog.__init__` creates the schema on open):

```bash
mkdir -p /mnt/nas/photos-organized-v2
uv run python -c "
from pathlib import Path
from imageharbor.catalog import Catalog
Catalog(Path('/mnt/nas/photos-organized-v2/catalog.db')).close()
"
```

Then copy the two tables with `sqlite3` `ATTACH` + `INSERT INTO … SELECT` (this
is a straight table copy, not a transformation — same columns, same types):

```bash
sqlite3 /mnt/nas/photos-organized-v2/catalog.db <<'SQL'
ATTACH DATABASE '/mnt/nas/photos-organized/catalog.db' AS old;

INSERT INTO taxonomy (code, parent_code, label, folder_name, aliases, alias_of, active, created_at)
SELECT code, parent_code, label, folder_name, aliases, alias_of, active, created_at
FROM old.taxonomy;

INSERT INTO learned_concepts (subject, class_code, hits, created_at, updated_at)
SELECT subject, class_code, hits, created_at, updated_at
FROM old.learned_concepts;

DETACH DATABASE old;
SQL
```

Sanity-check the row counts match the old catalog:

```bash
sqlite3 /mnt/nas/photos-organized/catalog.db    "SELECT count(*) FROM taxonomy;"
sqlite3 /mnt/nas/photos-organized-v2/catalog.db "SELECT count(*) FROM taxonomy;"

sqlite3 /mnt/nas/photos-organized/catalog.db    "SELECT count(*) FROM learned_concepts;"
sqlite3 /mnt/nas/photos-organized-v2/catalog.db "SELECT count(*) FROM learned_concepts;"
```

Do **not** copy `photos`, `sources`, `source_seen`, or `failed_files` — those are
all keyed to organized paths and per-file processing history from the old tree,
and the point of a rebuild is to regenerate them correctly under the new layout.

## 3. Run the facts pass against the same source, into the new destination

Point `process` at the same NAS source used by the old deployment, but the
**new** destination directory (never overwrite the old one — that is what keeps
this reversible):

**Choosing the new destination when the dest is a share.** hpz440's dest is the
whole `Photos-Organized` share, so there is no sibling directory to hold a
`-v2`. Reusing the share root is safe and is what the 2026-08-12 rebuild did:
the new date folders (`2001/`…`Undated/`) cannot collide with the old PCS
folders (`100-people`…`900-miscellaneous`), so both trees coexist as siblings
and cutover needs no path change. The trade-off is that step 6's `verify` walks
the old tree too (harmless — legacy names still parse and still verify), and
step 7 deletes the nine PCS directories rather than one parent. The alternative
— a `v2/` subdirectory — keeps verify cleanly scoped but bakes `v2` into every
recorded path permanently, since moving the tree later would strand every
`organized_path` in the catalog.

```bash
uv run imageharbor process \
  --source /mnt/nas/photos \
  --dest /mnt/nas/photos-organized-v2 \
  --sidecar
```

This makes no AI calls and requires no AI backend running. Expect it to run at
disk/network speed — hashing, EXIF reads, and copies only. Re-running this
command is always safe: it is fully idempotent (content-addressed dedup means a
second run against the same source is a fast no-op pass over already-known
digests).

## 4. Spot-check the new tree

Before spending AI budget on enrichment, sanity-check placement:

```bash
find /mnt/nas/photos-organized-v2 -maxdepth 2 -type d | sort
# expect YYYY/YYYY-MM directories plus Undated/

uv run imageharbor catalog list --catalog /mnt/nas/photos-organized-v2/catalog.db --limit 20
```

Spot-check a handful of files with known dates land in the right `YYYY/YYYY-MM/`
folder, and that a few deliberately-undated or camera-named files (e.g.
`IMG_1234.jpg` with no EXIF) land in `Undated/` rather than being guessed into a
year.

**Expect `Undated/` to be larger than intuition suggests.** A source tree
organized into year folders does *not* transfer those years: the parent
directory name is not date evidence, only EXIF and then a date pattern in the
original filename are (see the date ladder in `date_resolver.py`). A
`2001/P1010009.JPG` with no EXIF date and no date in its stem lands in
`Undated/`, by design — asserting a date the file cannot support is the quiet
corruption the ladder exists to prevent.

Do not extrapolate the rate from one folder. In the 2026-08-12 rebuild the first
sampled folder ran 27-undated-of-66 (41%), but the full 3,832-file result was
**420 undated, 11.0%** — EXIF `DateTimeOriginal` covered 2,574 and other EXIF
date fields another 837. If the real proportion ever is uncomfortable, the fix is
a new, explicitly-ranked rung in the ladder (below EXIF and filename), not a
change to placement — and it needs its own spec, because a folder name is weaker
evidence than either existing rung.

## 5. Run the enrichment pass

```bash
uv run imageharbor enrich \
  --dest /mnt/nas/photos-organized-v2 \
  --sidecar \
  --ai openai \
  --ai-base-url http://jetson.local:11434/v1 \
  --ai-model qwen2.5vl:3b
```

Because `taxonomy` and `learned_concepts` were carried forward in step 2, most
subjects the model has seen before resolve instantly from `learned_concepts`
without a network call; only genuinely new subjects hit the AI backend. This is
safe to interrupt and re-run — enrichment is resumable (`enriched_at IS NULL`
drives the work queue) and a re-run only ever improves a file's name, never
degrades it (see the monotonicity invariant in `CLAUDE.md`).

## 6. Verify the new tree

```bash
uv run imageharbor verify /mnt/nas/photos-organized-v2
```

Every organized file's embedded digest must match its content. A non-zero exit
or any `FAIL` lines means something is wrong and must be investigated before the
old tree is touched.

## 7. Cut over

Once steps 3–6 look right:

```bash
# Update docker-compose.yml / environment to point --dest (and IMAGEHARBOR_DEST)
# at the new organized directory, then:
docker compose -f /path/to/imageharbor/docker-compose.yml up -d
```

Watch the first few `watch` passes in the logs to confirm the new catalog path
is being used and both passes are running.

**Only after** the new deployment has been running cleanly for a while — and you
are confident you no longer need to fall back — delete the old organized tree
and its catalog:

```bash
rm -rf /mnt/nas/photos-organized   # the OLD tree, not photos-organized-v2
```

On hpz440 this must run **inside a container**, because the organized files are
owned by the container's uid and an ssh shell cannot unlink them (see the
ownership traps in step 2). With the rebuild sharing the share root, the targets
are the nine PCS directories, not the share itself:

Dry-run the glob first and read the list before deleting anything:

```bash
docker run --rm -v /mnt/nas/organized:/data/dest --entrypoint sh imageharbor:latest -c 'for d in /data/dest/[1-9]00-*; do echo "$d"; done'
```

That must list exactly the nine PCS directories. `[1-9]00-*` cannot match a year
directory (`2001` has no `00-`), `Undated`, or `#recycle` -- verified against the
live tree on 2026-08-12. Once the list looks right:

```bash
docker run --rm -v /mnt/nas/organized:/data/dest --entrypoint sh imageharbor:latest -c 'rm -rf /data/dest/[1-9]00-*'
```

Leave `#recycle` (the NAS's own trash directory) alone. Verify the date-derived
tree is still intact afterwards before considering the old catalog volume
disposable — and keep that volume even then, since it is the only pre-redesign
record of what was classified.

There is no rush on this step. Disk space is the only cost of leaving the old
tree in place, and every step above reads the untouched source, so keeping the
old tree around costs nothing but storage while you gain confidence in the new
one.
