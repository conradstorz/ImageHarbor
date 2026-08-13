# Google Takeout ingestion — design

**Date:** 2026-08-12
**Status:** approved, not yet implemented
**Extends:** [`2026-08-11-facts-first-pipeline-design.md`](2026-08-11-facts-first-pipeline-design.md)

## Goal

Ingest Google Takeout `.zip` archives into the ImageHarbor library so that every
photo they contain is organized, verified, and cataloged — **without ever
modifying an archive**, and such that running the ingest twice, or resuming it
after a `kill -9`, costs nothing and changes nothing.

Takeout adds exactly three things the existing facts pass does not have:

1. **A container to walk** instead of a directory tree.
2. **An external metadata sidecar** (Google's per-media `.json`) that outranks
   most of the evidence in the file itself.
3. **Enough volume** that "just re-scan everything" has a real cost, so the skip
   decision has to be cheap *and* provable.

Content addressing already makes ingestion idempotent at the *photo* level. This
design adds the layers above it that make ingestion idempotent at the *archive*
and *member* levels, so a re-run never decompresses a byte it has already seen.

## Non-goals

Each of these is a clean later project, deliberately excluded:

- `.tgz` archives. Google's `.tgz` parts are a true split stream with no
  per-part central directory; the manifest strategy below does not apply to them.
- **Video ingestion.** Videos are enumerated and recorded as `deferred` with
  their `photoTakenTime`, so the inventory is complete and a later video project
  starts from a work queue rather than from zero. No video bytes are copied.
- Materializing albums in the tree (folders, symlinks). Album membership is
  recorded in the catalog and sidecar; placement stays date-derived.
- `watch` integration. Ingestion is a hand-run verb.
- A separate destination for archived or trashed content.

## Ground truth

This design was calibrated against a real export on the user's machine:
`takeout-20230618T004316Z-001.zip`, 79 MB, 196 members. The findings that
changed the design are recorded here because they are the difference between a
spec that works on this data and one that does not.

**The export is `AlbumArchive`, not `Google Photos`.** Its tree is
`Takeout/AlbumArchive/Hangouts/<album>/`. There is no `Google Photos/` directory
anywhere in it. **The member walk must be service-agnostic** — classify by
extension and sidecar adjacency, never by a hard-coded service path.

**The per-media JSON schema, verbatim:**

```json
{
  "title": "2015-03-09.jpg",
  "imageViews": "12",
  "creationTime":   { "timestampSeconds": "1425920628", "formatted": "Mar 9, 2015, 5:03:48 PM UTC" },
  "photoTakenTime": { "timestampSeconds": "1425905792", "formatted": "Mar 9, 2015, 12:56:32 PM UTC" },
  "geoData": { "latitude": 38.2768361, "longitude": -85.73573890000002 },
  "height": "2432", "width": "4320",
  "exif": { "apertureFNumber": 2.4, "cameraModel": "XT1056", "exposureTime": 0.01666,
            "focalLength": 4.499, "isoEquivalent": 640 },
  "sizeBytes": "3698139"
}
```

Load-bearing notes:

- The timestamp key is **`timestampSeconds`**. Newer Google Photos exports use
  **`timestamp`**. The parser accepts both.
- `creationTime` (17:03 UTC) differs from `photoTakenTime` (12:56 UTC) on the
  same file. This is direct evidence for the rule below that `creationTime` never
  enters the date ladder.
- This schema has **no `description` and no `people`** fields. Google Photos
  exports do. The parser treats every field as optional.
- Album metadata is **`Albums.json`** (20 of them, one per album folder), not the
  `metadata.json` that Google Photos exports use. Both are accepted.

**Pairing is solved by three candidate forms, verified at 86/86 = 100%:**

```
NAME.EXT      ->  NAME.EXT.json
NAME.EXT      ->  NAME.EXT.supplemental-metadata.json   (newer exports)
NAME(N).EXT   ->  NAME.EXT(N).json                      (suffix moves AFTER .json)
```

The `(N)` displacement is present verbatim in the real export:

```
2015-03-09.jpg       2015-03-09.jpg.json
2015-03-09(1).jpg    2015-03-09.jpg(1).json
2015-03-09(2).jpg    2015-03-09.jpg(2).json
```

**`title` equals the export filename in this schema** (`"2015-03-09.jpg"` for
member `2015-03-09.jpg`). The "recover the true original filename from `title`"
rung is therefore a no-op for this export. It is still implemented, because it is
the correct source and it *does* matter for Google Photos exports, which truncate
member stems at roughly 47 characters.

**Member names carry Unicode and shell-hostile characters**: `●` (U+25CF), `+`,
`=`, spaces, parentheses. Nothing may assume ASCII or shell-safe member paths.

## A pre-existing bug this export exposes

Running the current `descriptor.resolve_descriptor` against real member names:

```
865948477697870747_account_id=1.jpg  ->  tier 30  '865948477697870747-account-id'
2015-03-09.jpg                       ->  tier 30  '2015-03-09'
2015-03-09(1).jpg                    ->  tier 30  '2015-03-09'
```

No `CAMERA_PATTERNS` entry matches these, so they resolve to
`DESC_HUMAN_FILENAME` (tier 30). Because 30 outranks `DESC_AI_SUBJECT` (20),
`tiers.is_upgrade` would **permanently prevent the enrichment pass from ever
naming these files** — 62 photos in this one archive locked to a Hangouts row ID
forever. The second case also produces `2015-03-09-2015-03-09_<digest>.jpg`, with
the date stated twice.

This is in scope for this project because Takeout ingestion is what surfaces it
at volume. Two fixes in `descriptor.py`:

1. Add `CAMERA_PATTERNS` entries for the machine-generated shapes this export
   contains — at minimum `^\d{10,}_account_id=\d+$` (Hangouts/AlbumArchive row
   IDs) and a bare `^\d{4}-\d{2}-\d{2}(\(\d+\))?$` (a date is not a description;
   the date ladder already captured it).
2. Discard a normalized descriptor that equals the resolved date string, as
   `DESC_NONE`. A descriptor that merely restates the folder and the filename
   prefix carries no information.

Both are additions to the existing pattern table and land with fixture rows in
`tests/test_descriptor.py`, matching that module's stated convention.

## Architecture

A new package `imageharbor/takeout/`, split so the module most likely to be
*wrong* has no I/O and can be tested exhaustively as a pure function.

| Module | Purpose | I/O |
|---|---|---|
| `takeout/archive.py` | Archive identity, central-directory enumeration, member classification, member → tempfile extraction | yes |
| `takeout/metadata.py` | Parse Google JSON → `TakeoutMetadata`. Never raises | **no** |
| `takeout/pairing.py` | Match a media member to its sidecar across Google's naming mutations | **no** |
| `takeout/ingest.py` | Orchestrator `ingest_archives()`; mirrors `enrich.enrich_library`'s shape | yes |

### `takeout/archive.py`

```python
@dataclass(frozen=True)
class MemberInfo:
    path: str          # member path inside the zip, verbatim
    size: int          # uncompressed size, from the central directory
    crc32: int
    kind: str          # "image" | "video" | "metadata" | "album" | "other"

@dataclass(frozen=True)
class ArchiveIdentity:
    archive_id: str    # SHA-256 b64url of the .zip itself
    path: Path
    size: int
    mtime_ns: int

def identify(path: Path, catalog: Catalog) -> ArchiveIdentity: ...
def iter_members(zf: zipfile.ZipFile) -> Iterator[MemberInfo]: ...
def classify(member_path: str) -> str: ...
def extract_to(zf, member: MemberInfo, staging_dir: Path) -> Path: ...
```

- `identify` tries a `(path, size, mtime_ns)` fast path against
  `takeout_archives` first; only on a miss does it hash. Hashing reuses
  `hashing.compute_sha256_b64url` unchanged — no new hashing code, so the
  content-addressing invariant surface is untouched.
- `iter_members` reads only the central directory. **No decompression.**
- `classify` uses `discovery.SUPPORTED_EXTENSIONS` as the single source of truth
  for `image`, and a new `VIDEO_EXTENSIONS` frozenset in `discovery.py` for
  `video` (`.mp4 .mov .m4v .3gp .avi .mkv .webm`). `.json` is `metadata`, or
  `album` when its basename is `Albums.json` or `metadata.json`. Everything else
  is `other`.
- `extract_to` streams the member to a temp file. `zipfile` verifies CRC on a
  full read, so a corrupted member raises rather than yielding bad bytes.

### `takeout/metadata.py`

Pure. Handed `bytes`, returns a dataclass. Never raises — malformed, truncated,
or absent input returns an empty `TakeoutMetadata`, exactly the discipline
`exif_reader.read_exif` uses.

```python
@dataclass(frozen=True)
class TakeoutMetadata:
    title: str | None            # original filename per Google
    description: str | None      # absent in AlbumArchive schema
    photo_taken_at: datetime | None
    creation_at: datetime | None # RECORDED ONLY, never used for placement
    latitude: float | None
    longitude: float | None
    people: tuple[str, ...]
    favorited: bool
    size_bytes: int | None

def parse_photo_metadata(raw: bytes) -> TakeoutMetadata: ...
def parse_album_metadata(raw: bytes) -> AlbumMetadata: ...
```

Timestamp extraction accepts `timestampSeconds` (AlbumArchive) **and**
`timestamp` (Google Photos), both as strings, both epoch seconds UTC.

### `takeout/pairing.py`

A pure function over a list of member *names*. This is the risk concentrate, so
it gets a fixture table like `descriptor.CAMERA_PATTERNS` has.

```python
def build_index(members: Iterable[str]) -> PairingIndex: ...
def sidecar_for(media_path: str, index: PairingIndex) -> str | None: ...
```

Candidate generation, in order, for media member `NAME.EXT`:

1. `NAME(N).EXT` → `NAME.EXT(N).json` — checked first; the `(N)` form is
   unambiguous and must not be shadowed by the generic rule.
2. `NAME.EXT.json`
3. `NAME.EXT.supplemental-metadata.json`
4. `-edited` variant: strip a trailing `-edited` from the stem and retry 1–3.
   Google does not emit a sidecar for edited derivatives.
5. Truncation recovery: unique prefix match among sidecars in the *same
   directory* whose stem is a prefix of `NAME.EXT` — accepted only when exactly
   one candidate matches.
6. Case-insensitive extension retry.

**Ambiguity policy: never guess.** If no rule produces exactly one match, return
`None`. The member is ingested from EXIF and filename alone, and the miss is
recorded by leaving `takeout_members.sidecar_path` NULL on an otherwise
`ingested` row — it is not a distinct status, because a photo without Google
metadata is fully and correctly organized, not a failure. `takeout status`
counts those rows. A wrong pairing writes another photo's date into this photo's
name — precisely the quiet corruption this project exists to prevent, and far
worse than an absent date.

### `takeout/ingest.py`

The only module with side effects. Its signature mirrors `enrich_library`:

```python
@dataclass
class IngestStats:
    archives_seen: int = 0
    archives_skipped: int = 0     # already complete
    archives_corrupt: int = 0
    ingested: int = 0
    duplicates: int = 0
    deferred: int = 0             # videos
    skipped_trash: int = 0
    failed: int = 0
    missing_metadata: int = 0     # ingested, but no sidecar could be paired

def ingest_archives(
    archives_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    *,
    include_trash: bool = False,
    write_sidecars: bool = False,
    dry_run: bool = False,
) -> IngestStats: ...
```

### `ExternalEvidence`

The one new type crossing into the facts pass. Defined in `pipeline.py`, beside
`ProcessResult`, because it is that module's parameter object:

```python
@dataclass(frozen=True)
class ExternalEvidence:
    """Facts about an image that are not in its bytes or its current path."""
    date: datetime | None = None        # Google photoTakenTime
    original_name: str | None = None    # Google `title`, pre-truncation
```

`Pipeline` unpacks it into the two resolvers rather than passing the object
down, so neither resolver learns anything about Takeout:

```python
date       = resolve_date(path, exif, external_date=evidence.date if evidence else None)
descriptor = resolve_descriptor(path, original_name=evidence.original_name if evidence else None)
```

`resolve_date`'s new parameter is a bare `datetime | None`. When present and
plausible it resolves at `DATE_EXTERNAL_SIDECAR`, ranked below the
`DateTimeOriginal` rung and above the rest.

## Data flow

Ingestion is **two-phase**, and the reason is not cosmetic.

Google's multi-part zips split by size across the file list, so `IMG_1234.jpg`
can land in part 1 while `IMG_1234.jpg.json` lands in part 2. A per-archive
pairing index would silently lose metadata at every part boundary. Therefore the
pairing index is built across **every archive in the batch** before any member is
ingested.

```
PHASE 1 — survey (all archives; central-directory reads only, no decompression)
  for each *.zip under archives_dir:
      identity: (path,size,mtime_ns) fast path -> known archive_id
                else SHA-256 the archive
      if takeout_archives.status == 'complete': skip entirely
      iter_members -> upsert takeout_archives(status='partial') + takeout_members
          kind='image'                    -> status='pending'  (work item)
          kind='video'                    -> status='pending'  (work item)
          kind='metadata' | 'album'       -> status='parsed'   (terminal; read on demand)
          kind='other'                    -> status='ignored'  (terminal)
          any member under a Trash/ tree, unless --include-trash
                                          -> status='skipped_trash' (terminal)
  build ONE global pairing index across all surveyed members

PHASE 2 — ingest (per member; resumable at any point)
  for each member with kind='image' and status='pending' or 'failed':
      sidecar  = pairing_index.get(member)         # may live in a different zip
      meta     = parse_photo_metadata(read that sidecar)    # lazily, on demand
      tmp      = extract_to(zf, member, staging)   # zipfile verifies CRC here
      evidence = ExternalEvidence(date=meta.photo_taken_at, original_name=meta.title)
      result   = pipeline.process_file(tmp, source_label=..., evidence=evidence)
      record member status from result + sidecar_path; rm tmp
  for each member with kind='video' and status='pending' or 'failed':
      meta = parse_photo_metadata(pairing_index.get(member))   # date only, no copy
      status='deferred', taken_at=meta.photo_taken_at
  archive status -> 'complete'   (no member left at 'pending' or 'failed')
```

Metadata is read **lazily** through the index rather than parsed up front: a
60 GB export can hold 100k+ members, and holding only name→name strings keeps the
index near 10 MB instead of hundreds.

`source_label` is the logical source recorded in `sources` and
`photos.original_path`:

```
/nas/takeout/takeout-001.zip!Takeout/AlbumArchive/Hangouts/<album>/2015-03-09.jpg
```

It is stable across runs and across machines that mount the archive at the same
path, so the back-pointer set stays meaningful.

### Late-arriving metadata resolves itself

If part 2 has not been downloaded yet, its sidecars are missing and the photos in
part 1 ingest with no Google date — landing in `Undated/`. When part 2 arrives
and the ingest is re-run, those photos hash as duplicates, and
`pipeline._maybe_upgrade_from_duplicate` re-evaluates tiers against the new
evidence: `is_upgrade((0, d), (30, d))` is True, and the file relocates from
`Undated/` into `2015/2015-03/`.

**No new code path is required for this.** The monotonic upgrade machinery
already handles late-arriving evidence, provided `_maybe_upgrade_from_duplicate`
is given the `evidence` argument. This is the single most important integration
point in the design.

### Trash

Takeout exports may contain a `Trash/` tree. Those members are **enumerated** (so
the inventory is honest) but **not ingested**: status `skipped_trash`.
`--include-trash` overrides. Deleted photos should not silently re-enter the
library, but they are also the photos most likely to be regretted deletions, so
the flag exists and the rows are recorded either way.

## Facts integration

Only the date ladder gains a rung, and it already exists:

```
DATE_EXIF_ORIGINAL    = 40   EXIF DateTimeOriginal
DATE_EXTERNAL_SIDECAR = 30   Google photoTakenTime          <- now populated
DATE_EXIF_OTHER       = 20   DateTimeDigitized, DateTime
DATE_FILENAME_PATTERN = 10
DATE_NONE             =  0
```

`tiers.py` is **unchanged** — `DATE_EXTERNAL_SIDECAR` was reserved for exactly
this on 2026-08-11.

Rules that are not negotiable:

- **`creationTime` never enters the date ladder.** It records when the file was
  uploaded to Google Photos, not when the photo was taken — the same category of
  claim as file mtime, which `date_resolver.py` deliberately refuses. The real
  export shows the two differing by four hours on the same file. It is recorded
  in the sidecar as provenance and nothing more.
- **`geoData`, `people[]`, `favorited`, album membership, and Google's `exif`
  block are recorded, never load-bearing.** They go to the catalog and the JSON
  sidecar. They cannot move or rename a file, because placement is date-derived.
- **No new descriptor tier.** Google's `title` feeds the existing
  `DESC_HUMAN_FILENAME` rung with better evidence; it does not outrank it.

## Data model

Two additive tables. `SCHEMA_VERSION` **stays `"2"`**: no existing row is
reinterpreted and no existing column changes meaning, so
`Catalog._guard_legacy_catalog` correctly does not fire and existing catalogs
upgrade in place.

```sql
CREATE TABLE IF NOT EXISTS takeout_archives (
    archive_id    TEXT PRIMARY KEY,          -- SHA-256 b64url of the .zip
    last_path     TEXT    NOT NULL,
    size          INTEGER NOT NULL,
    mtime_ns      INTEGER NOT NULL,
    member_count  INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'partial',  -- partial|complete|corrupt
    last_error    TEXT    NOT NULL DEFAULT '',
    first_seen_at TEXT    NOT NULL,
    last_seen_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS takeout_members (
    archive_id    TEXT    NOT NULL,
    member_path   TEXT    NOT NULL,
    kind          TEXT    NOT NULL,   -- image|video|metadata|album|other
    size          INTEGER NOT NULL,
    crc32         INTEGER NOT NULL,
    status        TEXT    NOT NULL,   -- pending|ingested|duplicate|deferred
                                      -- |parsed|ignored|skipped_trash|failed
    sha256_b64url TEXT,               -- set when ingested/duplicate
    taken_at      TEXT,               -- photoTakenTime, ISO; for deferred videos
    sidecar_path  TEXT,               -- resolved sidecar member, or NULL
    last_error    TEXT    NOT NULL DEFAULT '',
    updated_at    TEXT    NOT NULL,
    PRIMARY KEY (archive_id, member_path)
);
CREATE INDEX IF NOT EXISTS idx_takeout_members_status  ON takeout_members(status);
CREATE INDEX IF NOT EXISTS idx_takeout_members_archive ON takeout_members(archive_id);
```

## Idempotency

Four layers, each cheaper than the one below it:

| Layer | Key | Skips |
|---|---|---|
| Archive fast path | `(last_path, size, mtime_ns)` + `status='complete'` | the entire zip, without hashing it |
| Archive identity | `archive_id` = SHA-256 of the zip | a renamed or moved zip, without re-ingesting |
| Member | `(archive_id, member_path)` at a terminal status | extraction and decompression |
| Content | `sha256_b64url` present in `photos` | the copy; records a `sources` back-pointer instead |

The member layer is what makes the skip **provable** rather than probable: the
same archive digest implies the same central directory, which implies the same
bytes at that member path. There is no CRC32 gamble — CRC32 is stored for
diagnostics and for detecting an archive that changed underneath a stale row,
never as the sole basis for skipping.

Terminal statuses, never revisited: `ingested`, `duplicate`, `deferred`,
`parsed`, `ignored`, `skipped_trash`. Non-terminal: `pending`, `failed`.
`failed` members are retried on the next run — an ingest failure here is a local
filesystem or archive fault, not a backend outage, so there is no quarantine
ladder and no backoff.

## Failure model

**Non-destructive, concretely.** Archives are opened `'r'` only, never `'a'` or
`'w'`. Nothing is written into, alongside, or in place of an archive. Extraction
targets `<dest>/.takeout-staging/`, git-ignored, with temp files removed in a
`finally`. A leftover staging file after a kill is inert debris, not state: phase
2 resumes from `takeout_members`, never from what is on the staging floor.

**Isolation at three scopes.**

- *Member*: any exception ingesting one member is caught, recorded as
  `status='failed'` with its error, and the loop continues — mirroring
  `pipeline._process_one`'s existing try/except.
- *Archive*: `BadZipFile`, truncation, or a permission error marks that archive
  `corrupt` with the error and moves to the next one.
- *Batch*: neither of the above ever aborts the run.

**No circuit breaker.** This pass makes no AI calls, exactly like the facts pass,
so there is nothing for a breaker to observe and it must not touch one. This is
the same reasoning that keeps `pipeline.py` breaker-free.

**Crash safety.** Member status commits after each member, so `kill -9`
mid-archive costs one member's work. The archive stays `partial` and the next run
resumes at the first non-terminal member. A member is marked `ingested` only
*after* `process_file` returns a non-error result, so the existing
copy → verify → catalog ordering remains the sole arbiter of truth and
`takeout_members` can only lag it, never lead it.

**Staging handoff.** `Pipeline` gains an opt-in `consume_source: bool = False`.
When True, the copy step uses `os.replace` instead of `shutil.copy2`, because the
source is a disposable staging file ImageHarbor owns. This halves write I/O on
ingest — material at 60 GB per export over a NAS mount. The guarded invariant
survives intact: **the original is the zip, which is never touched**; the staging
file is not an original. Ordering becomes rename → verify → catalog, and verify
still reads the file at its destination. `process` and `watch` never set the
flag, so their behavior is byte-for-byte unchanged.

## CLI

A Click **group** with two subcommands, following the existing `catalog
list`/`catalog get` precedent in `cli.py`. There is no default subcommand —
Click has no first-class support for one, and `catalog` already establishes the
explicit-verb pattern.

```
imageharbor takeout ingest --archives DIR --dest DEST [--catalog PATH]
                           [--sidecar] [--include-trash] [--dry-run]
imageharbor takeout status [--catalog PATH]
```

`process` is untouched. It remains what `CLAUDE.md` documents it as: a filesystem
walk with no container logic and no network call.

```
$ imageharbor takeout ingest --archives /nas/takeout --dest /nas/Photos-Organized
  takeout-001.zip   [new]        4,102 img   318 vid
  takeout-002.zip   [new]        3,880 img   291 vid
  takeout-003.zip   [complete]   skipped
  ingested 7,982 / duplicates 1,204 / deferred 609 / failed 0

$ imageharbor takeout status
  3 archives: 2 complete, 1 partial (4,102 / 8,000 members)
  609 videos deferred · 12 members missing metadata
```

## Changes to existing modules

Every signature change is a keyword argument defaulting to `None`/`False`, so
current behavior is byte-for-byte unchanged when the argument is omitted.

| File | Change |
|---|---|
| `tiers.py` | **none** — `DATE_EXTERNAL_SIDECAR` already exists |
| `date_resolver.py` | `resolve_date(path, exif, external_date: datetime \| None = None)`; new rung between `DATE_EXIF_ORIGINAL` and `DATE_EXIF_OTHER` |
| `descriptor.py` | `resolve_descriptor(path, original_name=None)`; new `CAMERA_PATTERNS` entries; discard a descriptor equal to the date string |
| `discovery.py` | add `VIDEO_EXTENSIONS` frozenset (classification only; `discover_images` still yields images only) |
| `pipeline.py` | `process_file(path, *, source_label=None, evidence=None)`; thread `evidence` into `_maybe_upgrade_from_duplicate`; `Pipeline(consume_source=False)` |
| `catalog.py` | two additive tables + accessors; `SCHEMA_VERSION` unchanged |
| `cli.py` | new `takeout` group with `ingest` and `status` subcommands |
| `.gitignore` | add `.takeout-staging/` and `imageharbor/*.zip` |

## Testing

Pure modules get exhaustive table tests; the orchestrator gets behavioral tests
on synthetic zips built in `tmp_path` with `zipfile`. **No 79 MB fixture is
committed** — the synthetic zips replicate the real export's name shapes exactly,
which is what actually matters.

- **`test_takeout_pairing.py`** — fixture table of the real shapes: `(N)`
  displacement (`2015-03-09(1).jpg` → `2015-03-09.jpg(1).json`),
  `.supplemental-metadata.json`, `-edited`, case-flipped extensions, stem
  truncation, Unicode and `+`/`=`/`●` in paths, and the **no-confident-match**
  cases that must return `None` rather than guess.
- **`test_takeout_metadata.py`** — the verbatim AlbumArchive payload above; a
  Google Photos payload using `timestamp` instead of `timestampSeconds`;
  malformed, truncated, empty, and non-dict input. Must never raise. Asserts
  `creationTime` is parsed into `creation_at` but never returned as the placement
  date.
- **`test_takeout_archive.py`** — identity fast path vs. digest path; a renamed
  archive resolving to the same `archive_id`; enumeration without decompression;
  a deliberately CRC-corrupted member that fails the member without failing the
  archive.
- **`test_takeout_ingest.py`** — the behavioral core:
  - a second run extracts **zero** members, asserted by counting calls to
    `extract_to`, not by timing
  - resume after a simulated mid-archive crash processes only the remainder
  - a corrupt archive does not stop its neighbours in the same batch
  - videos land `deferred` with `taken_at` populated and **no bytes copied**
  - `Trash/` members land `skipped_trash`; `--include-trash` ingests them
  - **the late-sidecar case**: ingest part 1 alone, assert `Undated/`; add part 2
    carrying the sidecars, re-run, assert the file relocated to `2015/2015-03/`
- **`test_date_resolver.py`** — ladder ordering: EXIF `DateTimeOriginal` beats
  `photoTakenTime`; `photoTakenTime` beats `DateTimeDigitized` and beats a
  filename pattern; `creationTime` is never consulted.
- **`test_descriptor.py`** — new fixture rows for `_account_id=` shapes and bare
  dates, asserting tier 0; `original_name` overriding a truncated member stem.
- **`test_monotonicity.py`** — re-ingesting an already-organized photo from a
  Takeout archive is a rename **no-op** when tiers tie.
- **`test_pipeline.py`** — `consume_source=True` removes the staging file and
  still verifies before cataloging; `consume_source=False` remains a copy.
- **`test_cli.py`** — `takeout` and `takeout status`.

## Open items

- The `.tgz` variant is out of scope and structurally incompatible with the
  manifest strategy. If it is ever needed it is a separate design, not a flag.
- `pairing.py`'s truncation-recovery rule (candidate 5) is the one rule **not**
  exercised by the available export, which has no truncated stems. It is
  implemented from documented Google Photos behavior and should be re-calibrated
  the first time a real Google Photos export is ingested.
- Hangouts chat images inside `AlbumArchive` are ingested like any other photo.
  If they prove undesirable in the library, the fix is a member-kind filter, not
  a change to this design.
