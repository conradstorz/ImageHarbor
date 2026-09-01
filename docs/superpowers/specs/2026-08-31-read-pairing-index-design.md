# Read the Takeout pairing index

**Date:** 2026-08-31
**Status:** implemented on `feat/read-pairing-index`.
Consumes the index published by `Takeout_Inventory` (AGPL-3.0, same owner),
designed in that repo's
`docs/superpowers/specs/2026-08-29-inventory-scout-merge-design.md`.

## Why this exists

`imageharbor/takeout/pairing.py` answers "which Google JSON sidecar describes
this media file?" and answers it well: six rungs, verified 86/86 against a
real export, and a refusal to guess when no rung yields exactly one match.

It cannot answer the second question. `sidecar_for()` returns a path or
`None`, so an exact match and an `-edited` copy inheriting its **original's**
sidecar are indistinguishable to every caller. On the 79,221-media export
measured on 2026-08-31 that gap covers **13,078 files, 16.5%** — and for each
of them the sidecar's `latitude`/`longitude` describe a different photograph.

That is not hypothetical. It is the bug GooglePhotosTakeoutHelper filed as
issue #139: photos silently acquiring another photo's GPS. ImageHarbor's whole
premise is that a wrong value is worse than no value, and this is the one
place the pipeline currently cannot tell the difference.

`Takeout_Inventory` now publishes that distinction. This spec consumes it.

**What this is not.** It is not a fix for cross-archive pairing. `ingest.py`
already builds one index across every archive in a batch, with a comment
naming the reason. That was right before this work and is unchanged by it.

## Scope

**In.** An optional index reader; per-archive verification with per-archive
fallback; a confidence value on every pairing from **both** paths; a new date
tier; a drop-list for evidence a related sidecar cannot support; reporting.
One upstream addition to `Takeout_Inventory` so verification is possible.

**Out.** Removing `pairing.py`. Requiring the index. The operational
dashboard. Any change to how archives are opened, hashed, copied or verified.

## The measured facts this rests on

From the real 388 GB export, 175 archives, 2026-08-31:

| | |
| --- | --- |
| media files | 79,221 |
| `own` pairings | 66,137 (83.5%) |
| `related` pairings | 13,078 (16.5%) |
| unpaired | 6 |
| of the `related`, `-edited` copies | 11,895 (91%) |
| of the `related`, cross-extension | 82 |

The 91% matters: for an `-edited` copy the original's capture instant is
**this photograph's** capture instant. The date is good evidence. The
location is not, because location is a property of the file the sidecar
names.

## Upstream change to `Takeout_Inventory`

The index publishes `media` and `sidecar` tables and no archive metadata, so
per-archive verification is impossible against it today. `Inventory.archives`
already carries `name`, `size`, `mtime`, `members`, `error` per archive; it is
simply not written out.

Add to the published index:

```sql
CREATE TABLE archive (
  name     TEXT PRIMARY KEY,
  size     INTEGER NOT NULL,
  mtime    INTEGER NOT NULL,
  members  INTEGER NOT NULL,
  error    TEXT
);

CREATE TABLE index_meta (
  key      TEXT PRIMARY KEY,
  value    TEXT
);   -- seeded with schema_version and the tool version
```

`schema_version` starts at **1**. A reader that finds a higher value, or no
`index_meta` table at all, must treat the index as unusable rather than guess
at an older layout.

## Design

### `imageharbor/takeout/index_reader.py` — new, I/O at construction only

```python
@dataclass(frozen=True)
class IndexedPairing:
    sidecar: str | None      # member path of the sidecar, or None
    confidence: str          # "own" | "related" | "none"
    rule: str                # the rule that produced it, recorded not acted on


class IndexPairings:
    """Pairings read from a Takeout_Inventory index.

    Loaded once, then pure. Only archives whose name, size and mtime match
    the index are covered; everything else falls back to pairing.py.
    """

    @classmethod
    def open(cls, path: Path, archives: dict[str, os.stat_result]) -> "IndexPairings": ...

    def covers(self, archive_name: str) -> bool: ...
    def sidecar_for(self, member_path: str) -> IndexedPairing | None: ...
```

`open()` raises on a missing, unreadable, or schema-incompatible file. The
caller decides whether that is fatal — it is when the path was given
explicitly, and is not when the file was auto-detected.

The reader asserts its expected column set on open. A schema drift in
`Takeout_Inventory` must surface as a clear error, never as a wrong answer.

### `imageharbor/takeout/pairing.py` — one signature change

`sidecar_for()` returns the confidence its own rungs imply, so the policy
below is unconditional rather than something that only happens when someone
remembered to run a scan first:

- rungs 1–3 (copy-suffix, legacy `.json`, `supplemental-metadata`) → `own`
- rung 4 (`-edited` derivative, inheriting the original's sidecar) → `related`
- rung 5 is a case-insensitive **retry of rungs 1–4**, so it inherits the
  confidence of whichever rung actually matched. It is not a confidence of its
  own, and treating it as `own` would silently mislabel every case-differing
  `-edited` file. This is the one place the mapping is not one-to-one with the
  rung number, and the implementation must thread the underlying rung out of
  rung 5 rather than reporting the retry.
- rung 6 (truncation recovery, a unique prefix match among unclaimed sidecars)
  → `own`. It resolves a truncated spelling of **this file's own** name, not a
  different file's.
- no match → `none`

This is the single most important decision in the spec. If confidence existed
only on the indexed path, then dropping GPS for a related pairing would be a
behaviour that appears and disappears depending on whether a second tool had
been run, which is exactly the kind of conditional correctness this project
does not accept.

### `imageharbor/takeout/ingest.py` — one decision point

Where it now calls `pairing.build_index(all_members)`, it additionally
resolves an index if one is available, and routes each member by whether its
owning archive is covered. Both routes yield `(sidecar_path, confidence)`;
nothing downstream knows or cares which ran.

### `imageharbor/tiers.py` — one new tier

```python
DATE_EXIF_ORIGINAL     = 40
DATE_EXTERNAL_SIDECAR  = 30   # this file's own Google sidecar
DATE_RELATED_SIDECAR   = 25   # a related file's sidecar - usually its unedited original
DATE_EXIF_OTHER        = 20
DATE_FILENAME_PATTERN  = 10
DATE_NONE              = 0
```

25 sits above `EXIF_OTHER` deliberately: for an `-edited` copy this is the
same photograph's capture instant, which is better evidence than a
`DateTimeDigitized` recording when the file was written. `tiers.better()`
already exists, so an `own` match found later upgrades a `related` one and
never the reverse.

### The drop-list

Enforced where the sidecar is parsed, not scattered through the pipeline.

| From a `related` sidecar | Kept | Why |
| --- | --- | --- |
| `photo_taken_at` | yes, tier 25 | usually the same photograph |
| `latitude` / `longitude` | **no** | a property of the file the sidecar names |
| `people` | **no** | face tags belong to that file |
| `title` | **no** | the *original's* filename; would rename an edit after its parent |
| `trashed` / `archived` / `from_partner` | yes | describe the underlying photo; both copies share its status |

**Amended 2026-08-31 during planning, after reading how these values actually
travel.** `latitude`/`longitude` are parsed into `TakeoutMetadata` but never
reach `ExternalEvidence`, which carries only `date` and `original_name`. The
coordinates exist in ImageHarbor only inside the raw Google JSON document that
`_write_takeout_sidecar` stores verbatim under `provenance`. So the drop-list
above applies to the values ImageHarbor **acts on**:

- `ExternalEvidence.date` → kept, at tier 25
- `ExternalEvidence.original_name` (Google `title`) → dropped
- the `people` block → dropped

The raw document itself is **kept, and labelled**. Deleting it would violate
the project's preserve-everything discipline and destroy the audit trail; the
provenance entry instead gains `"confidence"` and `"pair_rule"` keys, so the
`geoData` inside `raw` is self-describing rather than silently authoritative.
A consumer that reads coordinates out of a `related` provenance entry is then
making a visible choice, not an accident.

The rule and confidence are written into the append-only JSON sidecar
alongside the date. A year from now the catalog must be able to answer "why
does this photo have a date and no location?" without re-deriving anything,
and if a rule turns out to be wrong the affected files must be queryable
rather than indistinguishable.

`related` pairings still count as claiming their sidecar for the
provenance/orphan accounting. Reporting such a sidecar as orphaned would
overstate the residue, and the residue is the number that has to stay honest.

## Locating the index

`--takeout-index PATH`, plus auto-detection of `takeout-index.sqlite` beside
the archives directory. Auto-detection is silent when absent.

## Error handling

Every case falls back, counts, and reports. A stale or broken index must
never stop an ingest, because the built-in pairing is always a correct answer.

| Situation | Behaviour |
| --- | --- |
| `--takeout-index` given, missing or unreadable | error, non-zero exit |
| auto-detected index corrupt or unopenable | warn, fall back everywhere |
| `schema_version` newer than known, or `index_meta` absent | warn, fall back everywhere |
| archive absent from `archive`, or size/mtime disagree | that archive's members fall back; counted |
| archive verified but a member is absent from the index | that member falls back; counted **separately** |
| index names a sidecar not present on disk | drop the pairing, fall back; counted |

## Reporting

Four lines in the ingest summary, written for someone deciding whether to
trust the run:

```
takeout index : takeout-index.sqlite (schema 1, 175 archives)
  archives    : 175 indexed, 0 fell back
  pairings    : 66,137 own · 13,078 related · 6 unpaired
  fallbacks   : 0 members, 0 missing sidecars
```

A run with no index says so on one line. "Did it use the index?" must never
be a question answered by reading logs.

## Testing

1. **Equivalence.** Over an archive set both paths cover, the index and
   `pairing.py` must never name *different* sidecars for the same member. The
   index pairing where `pairing.py` returns `None` is expected — it has rules
   ImageHarbor lacks. Naming a different sidecar is a genuine divergence
   between two implementations of the same domain, and this is the only test
   that can catch it.
2. **Optionality.** An ingest with a deliberately mismatched index must
   produce catalog output identical to the same ingest with no index at all.
   This is what makes "optional" safe rather than merely intended.
3. **The drop-list, mutation-proven.** A `related` pairing yields a date at
   tier 25 and no `latitude`, `longitude`, `people` or `title`. Re-enable each
   dropped field one at a time and confirm the test fails for each. A writer
   blanking every `lat`/`lon` passed the entire `Takeout_Inventory` suite
   green until its final review; this one gets proven.
4. **Verification and fallback units.** One test per row of the error table,
   each asserting both the fallback and its counter. A silent fallback is the
   failure mode.
5. **Real corpus, opt-in and marked**, against the real export and its real
   index. Invariants only, never counts — the corpus is not a fixture.

**A coupling named rather than hidden.** These tests build synthetic index
databases from a schema literal. If `Takeout_Inventory` changes that schema,
these tests keep passing while production breaks. The `schema_version` check
and the reader's explicit expected-column assertion turn that into a clear
error instead of a wrong answer, but only layer 5 catches it fully. This is a
real seam between two repositories and it is accepted knowingly.

## Deferred

- A dashboard panel for pairing provenance. Separate design.
- Backfilling confidence onto photos ingested before this change. They carry
  a tier-30 date from a sidecar whose confidence was never recorded; deciding
  whether to re-derive is its own piece of work.
