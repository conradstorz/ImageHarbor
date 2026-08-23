# Takeout Survey and Media Coverage

**Date:** 2026-08-23
**Status:** Design approved, implementation not started

## Why this exists

A 361.68 GiB Google Takeout export covering 1968–2026 was measured against the
current code before any of it was processed. The measurement is recorded in
"Evidence" below. It found that a `takeout ingest` run today would organize
roughly 77,000 photos (~100 GiB) and leave **234 GiB of video plus 4,008 real
photographs** sitting in `.takeout-provenance/` — preserved, verified, and
invisible to the catalog. It also found that **60% of media files would receive a
permanent tier-30 descriptor, roughly half of those a meaningless
machine-generated string** that enrichment could never replace.

None of that is data loss; the uncurated preserve-everything rule catches all of
it. It is a coverage failure, and coverage failures are quiet. This design closes
them and adds the read-only survey that would have surfaced them without anyone
having to go looking.

## Evidence

Measured directly from the archive set on 2026-08-23. All figures come from zip
central directories plus 32-byte content sniffs; no archive was modified.

| Quantity | Value |
| --- | --- |
| Parts | 181 (175 `.zip` + 6 raw `.mp4`), zero sequence gaps |
| Total | 361.68 GiB compressed / 345.4 GiB uncompressed |
| Members | 150,024 |
| Recognized images | 73,168 (99.5 GiB) |
| Recognized video | 1,966 (233.7 GiB) |
| Google JSON sidecars | 70,783 |
| Unrecognized by current code | 4,107 (12.1 GiB) |
| Google Photos subfolders | 239 (28 `Photos from YYYY`, 211 named albums) |
| Year range | 1968–2026 |

The 4,107 unrecognized members are not documents. Content sniffing resolved them
into three groups:

- **~4,008 genuine photographs** behind wrong extensions: `.screen` (3,963, bare
  19-digit names, sniffed 60 → 59 JPEG + 1 GIF), `.tile` (26, all JPEG), `.tmp`
  (19, 18 JPEG), and one member named literally `.jpg`.
- **~71 legacy videos** (11.5 GiB) outside `VIDEO_EXTENSIONS`, so they receive
  not even a `deferred` row: `.mts` (20, 3.7 GiB), `.vob` (7, 6.2 GiB), `.mpeg`
  (31), `.flv` (6), `.wmv` (4), `.mod` (1), plus two extensionless `MVIMG_*`
  Motion Photos (ISO-BMFF).
- **27 real documents** — `.kmz`, `.html`, `.csv`, `.txt`, `.ico`. Provenance is
  already the right answer for these.

Two further findings drive the design:

**Descriptor coverage.** Running the real `descriptor.is_camera_generated` over
all 79,211 media names judged **47,460 (60%) human-authored**, tier
`DESC_HUMAN_FILENAME` (30). Because `tiers.is_upgrade` forbids
`DESC_AI_SUBJECT` (20) from displacing tier 30, those descriptors are
**permanent** — enrichment can never improve them. Clustering the names by shape
separated them cleanly:

- Machine-generated, wrongly protected (~24,000): `2019_01_24_951` (9,055),
  `5889904872927499858` (4,928), `VID_20190726_231516` (4,590),
  `IMG_20150206_124238978_HDR` (1,520), `00034XTR_00034_BURST20190727182200`
  (1,232), `2014-10-12_12-44-56_HDR` (872), `2011-10-27 20.46.29 +0.0Ev` (402).
- Genuinely human, correctly protected (~8,000): `Scouts and Halloween 002`,
  `Kratz Cropped Scans_116`, `Bettie_Joe_0001`, `annies camera 2005 052`,
  `Pete_s camera 2006 134`, `AnnieM0112`, `frm SD card 116`.

The predicate defends the second group correctly. The defect is a bounded,
enumerable set of machine shapes, not a flaw in the idea.

**Loose parts pair across archives.** The six raw `.mp4` files are not strays —
their name suffixes (`-062`, `-067`, `-072`, `-089`, `-122`, `-162`) are exactly
the six part numbers absent from the zip sequence. Google delivers a media file
larger than the part size as its own raw part. Every one of them has its JSON
sidecar inside a **different** zip (089↔081, 062↔061, 162↔160, 122↔120). The
whole-batch pairing index already handles precisely this.

**A distrusted date cluster.** All 210 members of `Photos from 1968` carry the
identical `photoTakenTime` of `1968-01-12 10:35:03 UTC` — to the second — while
`creationTime` shows uploads in Dec 2013/Jan 2014. That is a camera with a dead
clock, not a capture date.

## Design

### 1. Content-based media detection

New module `imageharbor/content_type.py`, pure and I/O-free — it takes bytes and
returns a verdict, with no filesystem access, the same split as
`sidecar_schema.py` and `takeout/pairing.py`.

```
sniff(head: bytes) -> "image" | "video" | None
canonical_extension(head: bytes) -> str | None
HEAD_BYTES = 32
```

Signatures cover what this corpus actually contains plus the common remainder:
JPEG, PNG, GIF, BMP, TIFF, WebP, ICO; ISO-BMFF by `ftyp` brand (MP4/MOV/3GP/M4V/
HEIC/AVIF), Matroska/WebM, RIFF/AVI, MPEG-PS (`00 00 01 BA`, covering `.vob` and
`.mod`), MPEG-TS, ASF/WMV, FLV.

**The extension remains the first rung.** Only a member whose extension is
unrecognized is read and sniffed. A recognized extension is never second-guessed,
so no existing behavior changes and the fast path stays free. Cost on this set is
a 32-byte read for ~3% of members.

A sniffed member takes the **canonical** extension in the organized tree —
`5889904872927499858.screen` lands as `…_<digest>.jpg`. The organized library
must contain openable files, and the original member name survives verbatim in
`sources` and in the sidecar, so nothing is lost.

### 2. Video as first-class facts-pass media

The facts pass is already format-agnostic: hash → dedup → EXIF → date →
descriptor → target path → copy → verify → catalog → sidecar. `exif_reader`
returns `{}` for video, which both resolvers already tolerate, and dates come
from `photoTakenTime` and from the `VID_`/`PXL_` filename patterns. Video needs
no new placement or naming logic and keeps the existing filename grammar.

**New invariant — video must never enter the enrichment queue.** If
`iter_unenriched()` yields a `.vob`, `classifier.describe()` fails, and that
failure feeds the circuit breaker as if the AI backend were down; five of them
stall enrichment for the entire library. This would convert a coverage
improvement into an outage, so it is enforced structurally rather than by care at
the call site.

A `media_kind` column (`'image'` | `'video'`) is added to `photos` and filtered
out of `iter_unenriched()`. The column is additive, so `SCHEMA_VERSION` stays
`"2"` and existing catalogs upgrade in place — the same approach taken for `runs`
and `settings`.

### 3. Descriptor patterns

Two changes to `descriptor.py`:

1. **Decoration stripping before matching.** Google's decorations — `(1)`,
   `-edited`, `-ANIMATION`, `-MOTION`, `-COLLAGE`, `-EFFECTS`, `-PANO`, `~2`,
   `_BURST…` — are removed before the stem is tested against `CAMERA_PATTERNS`.
   This alone reclassifies ~9,800 members.
2. **New machine shapes** added to `CAMERA_PATTERNS` for the clusters listed
   under Evidence.

Both directions must be pinned by tests. A pattern that is too greedy is worse
than the bug it fixes: a wrong tier-30→tier-0 verdict discards a real
human-authored name permanently. The ~8,000 genuinely human names above are
regression fixtures.

### 4. Loose archive parts

A non-zip media file in the archive directory is treated as a single-member part
and registered in the same whole-batch pairing index. No new pairing logic: the
index is already global, which is why a loose part's sidecar in a different zip
resolves correctly.

### 5. Distrusted date clusters

A timestamp shared to the exact second by a large number of files is evidence of
a broken clock, not of a capture moment. Such members are placed in `Undated/`
rather than under the year they assert.

**This is the one place the design overrides stated evidence, so its boundaries
are drawn tightly.**

- **Detection is a batch property; resolution stays per-file and pure.** The
  orchestrator computes the distrusted set during the survey phase that already
  builds the pairing index, and passes it *into* `resolve_date` as a
  `distrusted_timestamps` parameter. `date_resolver` remains pure and per-file
  and gains no knowledge of batches.
- **Suppression applies to every date source, not just external evidence.** These
  files carry the bad value in EXIF as well; suppressing only the Google JSON
  would leave them filed under 1968 anyway.
- **The threshold is a flag** (`--distrust-cluster-threshold`, default 25). A
  burst of shots can legitimately share a second; 210 files cannot.
- **The claimed timestamp is never lost.** The sidecar records the original
  value together with the reason it was distrusted, per the never-lose rule. The
  folder says `Undated/`; the sidecar still says what the file claimed.
- **Scope is Takeout ingestion only.** `process` has no survey phase in which to
  compute a batch property. Extending it there is deliberately out of scope.

### 6. `takeout survey`

```
imageharbor takeout survey --archives DIR [--json PATH] [--catalog PATH]
```

Read-only: archives are opened `'r'`, nothing is written to them or to the
organized tree. Cost is a central-directory read per archive plus a 32-byte sniff
for unrecognized members — measured at ~21s for this 175-archive set.

Two modules, split the way the rest of the project splits:

- `takeout/survey.py` — I/O: enumerate archives, read central directories, sniff.
- `takeout/report.py` — **pure**: takes a collected inventory, returns the report
  document and anomaly list. The anomaly rules and projection math are the logic
  most likely to be wrong, so they live where they can be tested without a
  filesystem. This is the reasoning that put `projections.py` behind `stats.py`.

`report.py` also owns `find_distrusted_timestamps(counts, threshold)`, shared by
both the survey and ingest so the two can never disagree.

Report sections:

1. **Archive-set integrity** — parts, sequence gaps, loose media parts,
   unreadable archives, total bytes.
2. **Inventory** — members by class and extension, counts and uncompressed bytes.
3. **Anomalies** — misnamed media (extension says document, bytes say image);
   unrecognized formats; orphan sidecars; media with no sidecar; distrusted date
   clusters; dates before photography or in the future.
4. **Projection** — estimated organized count, destination bytes required,
   per-year distribution.

**The survey inherits the refuse-to-guess rule.** It cannot know the true
duplicate count without hashing 345 GiB, so it reports a **name-collision upper
bound**, labeled as such, and never a "duplicates" figure. A confident wrong
dedup number would send an operator into an ingest with the wrong space budget —
the same failure mode `dashboard/projections.py` exists to avoid.

Output is a JSON document plus a human-readable terminal summary. The owner-facing
report is published as a private web page from the JSON.

## Testing

- `content_type.sniff` — table-driven over real magic-byte prefixes, including
  the ones observed here (`.screen`→JPEG, `.tile`→JPEG, `.tmp`→JPEG,
  `MVIMG_*`→ISO-BMFF, `.vob`→MPEG-PS).
- Descriptor patterns — both directions pinned: every machine shape resolves to
  tier 0, and every human fixture stays at tier 30.
- **Video never appears in `iter_unenriched()`** — the invariant test for §2.
- Distrusted clusters — a cluster at/over threshold lands in `Undated/`; one
  under it does not; the claimed timestamp survives in the sidecar either way.
- Loose parts — a loose part pairs with a sidecar located in a different archive.
- Survey purity — `report.py` tested with synthetic inventories, including the
  refuse-to-guess cases.
- Survey is read-only — archive digests are unchanged after a survey run.

## Order of work

Steps 1–4 write nothing at all. The owner sees the report before any photo is
touched.

| # | Step | Writes |
| --- | --- | --- |
| 1 | `content_type.py` + tests | none |
| 2 | Descriptor patterns + tests | none |
| 3 | `takeout survey` + report | none |
| 4 | **Run it, publish the owner report, collect decisions** | **gate** |
| 5 | Video first-class, `media_kind`, enrichment exclusion | code only |
| 6 | Loose parts in ingest | code only |
| 7 | Ingest run | first write to the library |

## Open input

**Destination path.** `D:` has 139 GiB free against a requirement of roughly
334 GiB before dedup (100 GiB stills + 234 GiB video), on top of the 361 GiB the
archives already occupy. The organized tree must therefore land on the NAS. The
path is required before step 7 and not before steps 1–6.

## Out of scope

- Re-typing members whose extension is already recognized. A `.jpg` that is
  secretly a PNG keeps current behavior; only unrecognized extensions are sniffed.
- Distrusted-cluster detection in `process`. It has no batch survey phase.
- AI enrichment of video. Video is organized by facts and never enriched.
- Curating `.takeout-provenance/`. The uncurated preserve-everything rule stands.
