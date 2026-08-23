# Takeout Survey

**Date:** 2026-08-23
**Status:** Implemented and merged as `imageharbor takeout survey` (PR #16). The
items under "Deferred" below remain unbuilt.

## Why this exists

A 361.68 GiB Google Takeout export covering 1968–2026 was measured against the
current code before any of it was processed. The measurement, recorded under
"Evidence" below, found that a `takeout ingest` run today would organize roughly
77,000 photos (~100 GiB) and leave **234 GiB of video plus 4,008 real
photographs** in `.takeout-provenance/` — preserved, verified, and invisible to
the catalog. It also found that **60% of media files would receive a permanent
tier-30 descriptor, roughly half of those a meaningless machine-generated
string** that enrichment could never replace.

None of that is data loss; the uncurated preserve-everything rule catches all of
it. It is a coverage failure, and coverage failures are quiet. Every one of them
was found by looking, and none of them would have announced itself during a run.

**This spec builds the looking, and only the looking.** A standalone, read-only
`takeout survey` command that measures an archive set and reports what would
happen to it. The pipeline changes the evidence argues for are designed here too,
under "Deferred", so the reasoning is not lost — but they are explicitly not
built by this work.

## Scope

**In scope.** A `takeout survey` CLI verb that:

- runs **standalone** — no catalog, no destination, no watcher, no Docker, no AI
  backend, no network. It needs only a directory of archives.
- is **read-only** — archives are opened `'r'`; nothing is written to them, to a
  library, or to a catalog.
- **changes no pipeline behavior.** It reports what the current code *would* do.

**Out of scope,** and deliberately so — each is designed under "Deferred":

- Making video first-class in the facts pass.
- Widening the camera-name patterns.
- Ingesting loose (non-zip) archive parts.
- Changing where distrusted-date files are placed.
- Any web/dashboard surface. The survey is a command-line tool; folding it into
  the dashboard and the ingest system comes later.

## Evidence

Measured directly from the archive set on 2026-08-23, from zip central
directories plus 32-byte content sniffs. No archive was modified.

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

The 4,107 unrecognized members are mostly not documents. Content sniffing
resolved them into three groups:

- **~4,008 genuine photographs** behind wrong extensions: `.screen` (3,963, bare
  19-digit names; sniffed 60 → 59 JPEG + 1 GIF), `.tile` (26, all JPEG), `.tmp`
  (19, 18 JPEG), and one member named literally `.jpg`.
- **~71 legacy videos** (11.5 GiB) outside `VIDEO_EXTENSIONS`, receiving not even
  a `deferred` row: `.mts` (20, 3.7 GiB), `.vob` (7, 6.2 GiB), `.mpeg` (31),
  `.flv` (6), `.wmv` (4), `.mod` (1), plus two extensionless `MVIMG_*` Motion
  Photos (ISO-BMFF).
- **27 real documents** — `.kmz`, `.html`, `.csv`, `.txt`, `.ico`. Provenance is
  already the right answer for these.

**Descriptor coverage.** Running the real `descriptor.is_camera_generated` over
all 79,211 media names judged **47,460 (60%) human-authored**, tier
`DESC_HUMAN_FILENAME` (30). Because `tiers.is_upgrade` forbids `DESC_AI_SUBJECT`
(20) from displacing tier 30, those descriptors are **permanent**. Clustering by
name shape separated them cleanly:

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
their suffixes (`-062`, `-067`, `-072`, `-089`, `-122`, `-162`) are exactly the
six part numbers absent from the zip sequence. Google delivers a media file
larger than the part size as its own raw part. Each has its JSON sidecar inside a
**different** zip (089↔081, 062↔061, 162↔160, 122↔120).

**A distrusted date cluster.** All 210 members of `Photos from 1968` carry the
identical `photoTakenTime` of `1968-01-12 10:35:03 UTC` — to the second — while
`creationTime` shows uploads in Dec 2013/Jan 2014. A dead camera clock, not a
capture date.

## Design

### `imageharbor/content_type.py` — pure, I/O-free

```
sniff(head: bytes) -> "image" | "video" | None
canonical_extension(head: bytes) -> str | None
HEAD_BYTES = 32
```

Takes bytes, returns a verdict, no filesystem access — the same split as
`sidecar_schema.py` and `takeout/pairing.py`, and for the same reason: it is
exhaustively testable without fixtures on disk.

Signatures cover what this corpus contains plus the common remainder: JPEG, PNG,
GIF, BMP, TIFF, WebP, ICO; ISO-BMFF by `ftyp` brand (MP4/MOV/3GP/M4V/HEIC/AVIF);
Matroska/WebM; RIFF/AVI; MPEG-PS (`00 00 01 BA`, covering `.vob` and `.mod`);
MPEG-TS; ASF/WMV; FLV.

**The extension remains the first rung.** Only a member whose extension is
unrecognized is read and sniffed, so the fast path stays free and a recognized
extension is never second-guessed. Cost on this set is a 32-byte read for ~3% of
members.

This module is written now because the survey cannot report the misnamed-media
finding without it. Nothing in the pipeline calls it yet.

### `imageharbor/takeout/survey.py` — I/O

Enumerates archives in a directory, reads central directories, sniffs
unrecognized members, and collects an inventory. Opens archives `'r'` only.

Also detects **loose parts**: a non-zip media file in the archive directory whose
name carries a part-number suffix matching a gap in the zip sequence. The survey
reports these; teaching ingest to consume them is deferred.

### `imageharbor/takeout/report.py` — pure

Takes a collected inventory and returns the report document and anomaly list. The
anomaly rules and projection math are the logic most likely to be wrong, so they
live where they can be tested without a filesystem. This is the reasoning that
put `projections.py` behind `stats.py`.

Owns `find_distrusted_timestamps(counts, threshold)`, so that when ingest later
acts on clusters it shares one implementation with the survey and the two can
never disagree.

### CLI

```
imageharbor takeout survey --archives DIR [--json PATH] [--distrust-threshold N]
```

`--archives` is the only required option. `--json` writes the machine-readable
document; without it, only the human-readable summary is printed.
`--distrust-threshold` defaults to 25 — a burst of shots can legitimately share
a second; 210 files cannot.

### Report contents

1. **Archive-set integrity** — parts, sequence gaps, loose media parts,
   unreadable archives, total bytes.
2. **Inventory** — members by class and extension, counts and uncompressed bytes.
3. **Anomalies** — misnamed media (extension says document, bytes say image);
   unrecognized formats; orphan sidecars; media with no sidecar; distrusted date
   clusters; dates before photography or in the future; and the descriptor-tier
   distribution, which is how the 60% finding surfaces without anyone going
   looking for it.
4. **Projection** — estimated organized count, destination bytes required,
   per-year distribution.

**The survey inherits the refuse-to-guess rule.** It cannot know the true
duplicate count without hashing 345 GiB, so it reports a **name-collision upper
bound**, labeled as such, and never a "duplicates" figure. A confident wrong
dedup number would send an operator into an ingest with the wrong space budget —
the failure mode `dashboard/projections.py` exists to avoid.

## Testing

- `content_type.sniff` — table-driven over real magic-byte prefixes, including
  those observed here (`.screen`→JPEG, `.tile`→JPEG, `.tmp`→JPEG,
  `MVIMG_*`→ISO-BMFF, `.vob`→MPEG-PS).
- `report.py` — pure-function tests over synthetic inventories, including every
  refuse-to-guess case.
- Distrusted clusters — a cluster at/over threshold is reported; one under it is
  not.
- Loose parts — a media file filling a sequence gap is identified as a part.
- **Survey is read-only** — archive digests unchanged after a run. This is the
  property that makes it safe to run against the live 361 GiB set.
- Standalone operation — the command runs with no catalog, no destination, and no
  network.

## Deferred

Designed from the evidence above, not built by this work. The survey reports each
condition; none of them changes behavior yet.

- **Video first-class in the facts pass.** The facts pass is already
  format-agnostic, and dates come from `photoTakenTime` and `VID_`/`PXL_`
  filename patterns. When this is built it carries a load-bearing invariant:
  **video must never enter the enrichment queue.** A `.vob` reaching
  `iter_unenriched()` fails `classifier.describe()`, the circuit breaker reads
  that as a backend outage, and five of them stall enrichment for the whole
  library — turning a coverage fix into an outage. Enforce it structurally with a
  `media_kind` column filtered out of `iter_unenriched()`, additive so
  `SCHEMA_VERSION` stays `"2"`.
- **Descriptor patterns.** Strip Google's decorations (`(1)`, `-edited`,
  `-ANIMATION`, `-MOTION`, `~2`, `_BURST…`) before matching — worth ~9,800 on its
  own — then add the machine shapes listed under Evidence. Both directions need
  pinning: a too-greedy pattern is worse than the bug, because a wrong
  tier-30→tier-0 verdict discards a real human-authored name permanently.
- **Loose parts in ingest.** Register a loose part as a single-member part in the
  whole-batch pairing index. No new pairing logic is needed — the index is
  already global, which is exactly why a loose part's sidecar in a different zip
  resolves.
- **Distrusted clusters affecting placement.** Route them to `Undated/` rather
  than the year they assert. Detection is a batch property but resolution must
  stay per-file and pure: the orchestrator computes the set and passes it *into*
  `resolve_date`. Suppression must cover EXIF too — these files carry the bad
  value there as well, so suppressing only the Google JSON would leave them under
  1968 regardless. The claimed timestamp is still written to the sidecar; the
  never-lose rule does not bend for a value we distrust.
- **Web surface.** `takeout ingest` serves no dashboard today; only `watch` does.
  Folding survey and ingest into the dashboard comes later.

## Open input

**Destination path**, needed only when ingest is eventually run — not by this
work. `D:` has 139 GiB free against a requirement of roughly 334 GiB before dedup
(100 GiB stills + 234 GiB video), on top of the 361 GiB the archives occupy, so
the organized tree must land on the NAS.

## Related, and not part of this work

The fix for the dashboard's `GET /` returning 500 in the container lives on
`fix/dashboard-page-packaging` and is **not merged to `main`**. A container built
from `main` serves `/api/stats` but renders no page.
