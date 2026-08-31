# Read the Takeout Pairing Index — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ImageHarbor read a `Takeout_Inventory` pairing index when one is present, so every Google sidecar pairing carries a confidence saying whether its location and title may be trusted.

**Architecture:** Confidence becomes a property of *every* pairing, from both the indexed path and ImageHarbor's own six-rung `pairing.py`, so the policy that drops a related sidecar's title and people is unconditional. The index is optional and verified per archive; anything it does not cover falls back to the built-in pairing, counted and reported.

**Tech Stack:** Python ≥3.11, stdlib `sqlite3`, `click` for the CLI, pytest. `uv` for everything.

## Global Constraints

- **Two repositories.** Task 1 is in `D:\Users\Conrad\Documents\programming\Takeout_Inventory` (branch `master`). Tasks 2–8 are in `D:\Users\Conrad\Documents\programming\ImageHarbor` (branch `feat/read-pairing-index`). Never commit one repo's work in the other.
- **`Takeout_Inventory` is a single file.** All Task 1 implementation goes in `takeout_inventory.py`; no new modules. Its PEP 723 header stays `dependencies = ["rich>=13.7"]`.
- **A wrong pairing is worse than no pairing.** Where a rule cannot produce exactly one answer, produce none. Never break a tie.
- **A stale or broken index must never fail an ingest.** The built-in pairing is always a correct answer; every index problem falls back, counts, and reports.
- **Never weaken an existing assertion.** Changes to `tests/` are additions, except where a task explicitly says a test must change.
- Run tests with `uv run pytest`. Never `pip`, never `venv`.
- Do not chain shell commands with `&&`. Do not delete, move or tidy files you were not asked to change, including untracked files.

**Spec:** [`docs/superpowers/specs/2026-08-31-read-pairing-index-design.md`](../specs/2026-08-31-read-pairing-index-design.md)

---

### Task 1: Publish archive identity in the index (Takeout_Inventory)

**Repo:** `Takeout_Inventory`, branch `master`.

**Files:**
- Modify: `takeout_inventory.py` — `INDEX_SCHEMA`, `write_index_sqlite`, `write_index_json`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `Inventory.archives`, a `list[dict]` whose entries are `{"name": str, "size": int, "mtime": int, "members": int, "error": str | None}`.
- Produces: two new tables in the published index, `archive` and `index_meta`, and an `"archives"` list plus `"schema_version"` in the JSON output.

**Why:** ImageHarbor must verify that the index describes the archives actually on disk. Without per-archive identity it cannot, and would have to trust a possibly-stale index.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_index.py`:

```python
def test_sqlite_index_publishes_archive_identity(tmp_path):
    inv = build_inventory()
    inv.archives = [{"name": "part-1.zip", "size": 1024, "mtime": 1700000000,
                     "members": 7, "error": None}]
    db = tmp_path / "i.sqlite"
    tf.write_index_sqlite(inv, db)
    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT name, size, mtime, members, error FROM archive").fetchone()
    assert row == ("part-1.zip", 1024, 1700000000, 7, None)


def test_sqlite_index_publishes_a_schema_version(tmp_path):
    db = tmp_path / "i.sqlite"
    tf.write_index_sqlite(build_inventory(), db)
    con = sqlite3.connect(db)
    value = con.execute(
        "SELECT value FROM index_meta WHERE key = 'schema_version'").fetchone()[0]
    # A reader that finds a HIGHER value must refuse the file rather than
    # guess at a layout it does not know.
    assert value == str(tf.INDEX_SCHEMA_VERSION)


def test_json_index_publishes_archive_identity(tmp_path):
    inv = build_inventory()
    inv.archives = [{"name": "part-1.zip", "size": 1024, "mtime": 1700000000,
                     "members": 7, "error": None}]
    out = tmp_path / "i.json"
    tf.write_index_json(inv, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == tf.INDEX_SCHEMA_VERSION
    assert doc["archives"][0]["name"] == "part-1.zip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL — `sqlite3.OperationalError: no such table: archive`

- [ ] **Step 3: Write the implementation**

Add beside `INDEX_SQLITE` / `INDEX_JSON`:

```python
# Bump when the published index's shape changes. A consumer that finds a
# HIGHER value must refuse the file rather than guess at a layout it does not
# know; one that finds no index_meta table at all is reading a pre-versioned
# index and must refuse it too.
INDEX_SCHEMA_VERSION = 1
```

Append to `INDEX_SCHEMA`, after the existing `CREATE INDEX` statements:

```sql
DROP TABLE IF EXISTS archive;
DROP TABLE IF EXISTS index_meta;

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
);
```

In `_build_index_sqlite` (the shared builder both the staged and direct writers
use), after the media inserts and before `con.commit()`:

```python
        for a in inv.archives:
            con.execute(
                "INSERT OR REPLACE INTO archive (name, size, mtime, members, error)"
                " VALUES (?,?,?,?,?)",
                (a.get("name"), a.get("size", 0), a.get("mtime", 0),
                 a.get("members", 0), a.get("error")),
            )
        con.execute("INSERT INTO index_meta (key, value) VALUES (?,?)",
                    ("schema_version", str(INDEX_SCHEMA_VERSION)))
        con.execute("INSERT INTO index_meta (key, value) VALUES (?,?)",
                    ("tool_version", VERSION))
```

In `_build_index_json`'s payload, add two keys beside `version`:

```python
    payload = {"version": VERSION, "schema_version": INDEX_SCHEMA_VERSION,
               "scanned_at": inv.scanned_at, "takeout_dir": inv.takeout_dir,
               "archives": list(inv.archives), "media": rows}
```

Read the two builder functions before editing — the direct and staged writers
share them, and both outputs must gain the tables.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest`
Expected: PASS — 247 tests

- [ ] **Step 5: Commit**

```bash
git add takeout_inventory.py tests/test_index.py
git commit -m "feat: publish archive identity and a schema version in the index"
```

---

### Task 2: `pairing.py` reports confidence

**Repo:** `ImageHarbor`, branch `feat/read-pairing-index`. All later tasks are in this repo.

**Files:**
- Modify: `imageharbor/takeout/pairing.py`
- Modify: `imageharbor/takeout/ingest.py` (call sites only)
- Test: `tests/test_takeout_pairing.py`

**Interfaces:**
- Produces:
  - `OWN = "own"`, `RELATED = "related"`, `NO_MATCH = "none"`
  - `@dataclass(frozen=True) class Pairing: sidecar: str | None; confidence: str`
  - `sidecar_for(media_path, index) -> Pairing` — **return type changes**

**The key insight:** confidence follows the *name variant* that produced the
candidate, not the rung number. `_name_variants` yields the member's own name
first and its pre-`-edited` original second. The first is `own`; the second is
`related`, because that sidecar names a different file. Threading it through
`_candidates` makes rung 5 (the case-insensitive retry) inherit the right
answer automatically, which a rung-number mapping would get wrong.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_takeout_pairing.py`:

```python
def test_exact_match_is_own():
    index = tf_pairing.build_index([
        "T/GP/2019/IMG_1.jpg", "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"])
    p = tf_pairing.sidecar_for("T/GP/2019/IMG_1.jpg", index)
    assert p.sidecar == "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"
    assert p.confidence == tf_pairing.OWN


def test_edited_derivative_is_related():
    # The sidecar names IMG_1.jpg, not IMG_1-edited.jpg. Its location and
    # title belong to a different file.
    index = tf_pairing.build_index([
        "T/GP/2019/IMG_1-edited.jpg", "T/GP/2019/IMG_1.jpg",
        "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"])
    p = tf_pairing.sidecar_for("T/GP/2019/IMG_1-edited.jpg", index)
    assert p.sidecar == "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"
    assert p.confidence == tf_pairing.RELATED


def test_case_insensitive_retry_keeps_the_underlying_confidence():
    # Rung 5 retries rungs 1-4 case-insensitively. It is NOT a confidence of
    # its own: a case-differing -edited file is still related.
    index = tf_pairing.build_index([
        "T/GP/2019/IMG_1-EDITED.JPG", "T/GP/2019/img_1.jpg.json"])
    p = tf_pairing.sidecar_for("T/GP/2019/IMG_1-EDITED.JPG", index)
    assert p.sidecar == "T/GP/2019/img_1.jpg.json"
    assert p.confidence == tf_pairing.RELATED


def test_truncation_recovery_is_own():
    # Rung 6 resolves a truncated spelling of THIS file's own name. Google
    # truncates the WHOLE name, not the stem with a full extension re-appended
    # - build the fixture the way the module's existing rung-6 tests do, or
    # the sidecar's media part is never a prefix of the media's basename and
    # the rung never fires.
    long_stem = "A_very_long_original_filename_that_google_truncated"
    index = tf_pairing.build_index([
        f"T/GP/2019/{long_stem}.jpg",
        f"T/GP/2019/{(long_stem + '.jpg')[:40]}.supplemental-metadata.json"])
    p = tf_pairing.sidecar_for(f"T/GP/2019/{long_stem}.jpg", index)
    assert p.confidence == tf_pairing.OWN


def test_no_match_is_none():
    index = tf_pairing.build_index(["T/GP/2019/lonely.jpg"])
    p = tf_pairing.sidecar_for("T/GP/2019/lonely.jpg", index)
    assert p.sidecar is None
    assert p.confidence == tf_pairing.NO_MATCH
```

Check the existing test module's import alias for `pairing` and match it; the
name `tf_pairing` above is a placeholder for whatever that module already uses.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_pairing.py -v`
Expected: FAIL — `AttributeError: module 'imageharbor.takeout.pairing' has no attribute 'OWN'`

- [ ] **Step 3: Write the implementation**

In `pairing.py`, near the other module constants:

```python
OWN = "own"           # the sidecar names this file: title and location are its own
RELATED = "related"   # it names a different file (this one's unedited original)
NO_MATCH = "none"


@dataclass(frozen=True)
class Pairing:
    """A sidecar match and how far it may be trusted.

    `confidence` follows the NAME VARIANT that produced the candidate, not the
    rung number: rung 5 retries rungs 1-4 case-insensitively, so a
    case-differing `-edited` file must still come back `related`.
    """
    sidecar: str | None
    confidence: str
```

Change `_name_variants` to carry the confidence with each variant:

```python
def _name_variants(name: str) -> list[tuple[str, str]]:
    """(name, confidence) pairs: the member's own name, then its pre-`-edited`
    original (rung 4), whose sidecar describes a different file."""
    variants = [(name, OWN)]
    base, dot, ext = name.rpartition(".")
    if dot and base.lower().endswith(_EDITED):
        variants.append((f"{base[: -len(_EDITED)]}.{ext}", RELATED))
    return variants
```

Change `_candidates` to return `list[tuple[str, str]]`, pairing each generated
candidate path with the confidence of the variant it came from. Every `out.append(...)`
inside the loop becomes `out.append((<path>, confidence))` where `confidence`
is the variant's. Read the existing function and thread it through without
altering the candidate ORDER — rung 1 must still precede the generic rule.

Change `_exact_match` to return `Pairing | None`:

```python
def _exact_match(media_path: str, index: PairingIndex) -> Pairing | None:
    """Rungs 1-4, then rung 5 (the same candidates, case-insensitively)."""
    candidates = _candidates(media_path)
    for candidate, confidence in candidates:
        if candidate in index.sidecars:
            return Pairing(candidate, confidence)
    for candidate, confidence in candidates:
        hit = index.sidecars_ci.get(candidate.lower())
        if hit is not None:
            return Pairing(hit, confidence)
    return None
```

Change `sidecar_for`:

```python
def sidecar_for(media_path: str, index: PairingIndex) -> Pairing:
    """Return *media_path*'s pairing. `sidecar` is None if none is certain."""
    if _is_sidecar(media_path) or media_path in index.ambiguous_media:
        return Pairing(None, NO_MATCH)
    exact = _exact_match(media_path, index)
    if exact is not None:
        return exact
    # Rung 6 resolves a truncated spelling of this file's OWN name.
    truncated = _truncation_match(media_path, index)
    return Pairing(truncated, OWN if truncated is not None else NO_MATCH)
```

`build_index` also calls `_candidates` when populating `claimed`; update that
call site to unpack the tuples. Read it before editing.

Then update every `sidecar_for` call site. There are FOUR, not three: three
in `ingest.py` (around lines 336, 760, and the image path) and one in
`imageharbor/takeout/survey.py` (around line 262). Grep for `sidecar_for(`
across the package rather than trusting this list. Each currently binds a
`str | None`; each now binds a `Pairing` and reads `.sidecar`. Missing the
survey.py site turns its `is not None` / `is None` checks into silently
always-true tests against a `Pairing` object. Do not change behaviour in
this task beyond the type: confidence is threaded in Task 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest`
Expected: PASS — full suite, no regressions

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/pairing.py imageharbor/takeout/ingest.py tests/test_takeout_pairing.py
git commit -m "feat: every pairing carries a confidence"
```

---

### Task 3: The date tier

**Files:**
- Modify: `imageharbor/tiers.py`
- Test: `tests/test_tiers.py`

**Interfaces:**
- Produces: `DATE_RELATED_SIDECAR = 25` and its entry in `DATE_SOURCE_NAMES`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tiers.py`:

```python
def test_related_sidecar_ranks_below_own_and_above_exif_other():
    # An -edited copy's original carries the same photograph's capture
    # instant, which beats a DateTimeDigitized recording when the file was
    # written -- but it is weaker than this file's own sidecar.
    assert tiers.DATE_EXTERNAL_SIDECAR > tiers.DATE_RELATED_SIDECAR
    assert tiers.DATE_RELATED_SIDECAR > tiers.DATE_EXIF_OTHER


def test_related_sidecar_has_a_source_name():
    assert tiers.DATE_SOURCE_NAMES[tiers.DATE_RELATED_SIDECAR] == "related_sidecar"


def test_an_own_sidecar_upgrades_a_related_one():
    # The upgrade helper is `is_upgrade()`, NOT `better()`, and it takes
    # TUPLES of (date_tier, descriptor_tier) rather than two bare ints. Read
    # its real signature before writing this - the form below is illustrative.
    assert tiers.is_upgrade(
        (tiers.DATE_RELATED_SIDECAR, tiers.DESC_NONE),
        (tiers.DATE_EXTERNAL_SIDECAR, tiers.DESC_NONE))
    assert not tiers.is_upgrade(
        (tiers.DATE_EXTERNAL_SIDECAR, tiers.DESC_NONE),
        (tiers.DATE_RELATED_SIDECAR, tiers.DESC_NONE))
```

Check `tests/test_tiers.py`'s existing import style and `is_upgrade()`'s real
signature and argument ORDER before writing; match them.

**Expect collateral failures.** Adding a tier constant breaks any test that
enumerates the ladder exhaustively — `tests/test_dashboard_stats.py` is known
to. Those tests must be EXTENDED to include the new tier, never relaxed to
stop checking exhaustively.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tiers.py -v`
Expected: FAIL — `AttributeError: module 'imageharbor.tiers' has no attribute 'DATE_RELATED_SIDECAR'`

- [ ] **Step 3: Write the implementation**

```python
DATE_EXIF_ORIGINAL = 40      # EXIF DateTimeOriginal
DATE_EXTERNAL_SIDECAR = 30   # Google Takeout photoTakenTime, via ExternalEvidence.date
DATE_RELATED_SIDECAR = 25    # photoTakenTime from a RELATED file's sidecar -
                             # usually this file's unedited original, so the
                             # same photograph's capture instant. Above
                             # EXIF_OTHER, which records when a file was
                             # written rather than when a photo was taken.
DATE_EXIF_OTHER = 20         # DateTimeDigitized, DateTime
DATE_FILENAME_PATTERN = 10   # date parsed out of the original filename
DATE_NONE = 0                # no trustworthy date -> Undated/
```

And in `DATE_SOURCE_NAMES`, between the existing entries:

```python
    DATE_RELATED_SIDECAR: "related_sidecar",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add imageharbor/tiers.py tests/test_tiers.py
git commit -m "feat: a date tier for a related file's sidecar"
```

---

### Task 4: The index reader

**Files:**
- Create: `imageharbor/takeout/index_reader.py`
- Test: `tests/test_takeout_index_reader.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class IndexUnusable(Exception)`
  - `@dataclass(frozen=True) class IndexedPairing: sidecar: str | None; confidence: str; rule: str`
  - `class IndexPairings` with `open(path, archive_stats) -> IndexPairings`, `covers(archive_name) -> bool`, `sidecar_for(member_path) -> IndexedPairing | None`, and a `.stats` mapping.

**Contract:** `open()` raises `IndexUnusable` for a missing file, an unreadable
file, a missing or newer `schema_version`, or a missing expected column. The
caller decides whether that is fatal.

- [ ] **Step 1: Write the failing test**

Create `tests/test_takeout_index_reader.py`:

```python
"""Reading a Takeout_Inventory pairing index.

Databases here are built from a schema literal. That is a real seam between
two repositories: if Takeout_Inventory changes its schema these tests keep
passing while production breaks, which is why the reader asserts its expected
columns and the version on open.
"""
import sqlite3

import pytest

from imageharbor.takeout import index_reader


SCHEMA = """
CREATE TABLE sidecar (
  id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, name TEXT NOT NULL,
  title TEXT, taken_at TEXT, lat REAL, lon REAL, device TEXT,
  trashed INTEGER, archived INTEGER, from_partner INTEGER,
  parse_error TEXT, role TEXT);
CREATE TABLE media (
  id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, area TEXT NOT NULL,
  folder TEXT NOT NULL, name TEXT NOT NULL, ext TEXT, size INTEGER,
  actual_type TEXT, sidecar_id INTEGER REFERENCES sidecar(id),
  rule TEXT NOT NULL, confidence TEXT NOT NULL);
CREATE TABLE archive (
  name TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
  members INTEGER NOT NULL, error TEXT);
CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def make_index(path, *, version="1", archives=(("part-1.zip", 100, 5, 2, None),),
               media=(), sidecars=()):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for row in archives:
        con.execute("INSERT INTO archive VALUES (?,?,?,?,?)", row)
    for row in sidecars:
        con.execute("INSERT INTO sidecar (id, archive, path, name)"
                    " VALUES (?,?,?,?)", row)
    for row in media:
        con.execute("INSERT INTO media (archive, path, area, folder, name,"
                    " sidecar_id, rule, confidence) VALUES (?,?,?,?,?,?,?,?)", row)
    if version is not None:
        con.execute("INSERT INTO index_meta VALUES ('schema_version', ?)", (version,))
    con.commit()
    con.close()
    return path


def stats_for(size=100, mtime=5):
    class S:
        st_size = size
        st_mtime = mtime
    return S()


def test_open_reads_pairings(tmp_path):
    db = make_index(
        tmp_path / "i.sqlite",
        sidecars=[(1, "part-1.zip", "T/GP/a.jpg.json", "a.jpg.json")],
        media=[("part-1.zip", "T/GP/a.jpg", "GP", "GP", "a.jpg", 1, "exact", "own")])
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    p = idx.sidecar_for("T/GP/a.jpg")
    assert p.sidecar == "T/GP/a.jpg.json"
    assert p.confidence == "own"
    assert p.rule == "exact"


def test_orphan_media_reads_back_as_no_sidecar(tmp_path):
    db = make_index(
        tmp_path / "i.sqlite",
        media=[("part-1.zip", "T/GP/x.jpg", "GP", "GP", "x.jpg", None,
                "orphan", "none")])
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    p = idx.sidecar_for("T/GP/x.jpg")
    assert p.sidecar is None
    assert p.confidence == "none"


def test_unknown_member_returns_none(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    assert idx.sidecar_for("T/GP/never-seen.jpg") is None


def test_archive_with_a_different_size_is_not_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for(size=999)})
    assert not idx.covers("part-1.zip")


def test_archive_with_a_different_mtime_is_not_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for(mtime=999)})
    assert not idx.covers("part-1.zip")


def test_archive_absent_from_the_index_is_not_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-9.zip": stats_for()})
    assert not idx.covers("part-9.zip")


def test_matching_archive_is_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    assert idx.covers("part-1.zip")


def test_a_missing_file_is_unusable(tmp_path):
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(tmp_path / "nope.sqlite", {})


def test_a_newer_schema_version_is_unusable(tmp_path):
    db = make_index(tmp_path / "i.sqlite", version="2")
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})


def test_an_unversioned_index_is_unusable(tmp_path):
    db = make_index(tmp_path / "i.sqlite", version=None)
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})


def test_a_non_database_file_is_unusable(tmp_path):
    bad = tmp_path / "i.sqlite"
    bad.write_bytes(b"this is not a database")
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(bad, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_index_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.takeout.index_reader'`

- [ ] **Step 3: Write the implementation**

Create `imageharbor/takeout/index_reader.py`:

```python
"""Read a pairing index published by Takeout_Inventory.

The index answers what ImageHarbor's own `pairing.py` cannot: not merely which
sidecar describes a media file, but how far to trust it. An `-edited` copy
inherits its ORIGINAL's sidecar, whose title and location belong to a
different file.

Optional by design. Every failure here falls back to `pairing.py`, which is
always a correct answer -- so this module raises rather than repairs, and the
caller decides whether a given failure is fatal.

Loaded eagerly into memory: an index for a 388 GB export holds ~79k media rows
of short strings, which is a few tens of MB and removes any per-member query
from the ingest loop.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_MEDIA_COLUMNS = frozenset({
    "archive", "path", "sidecar_id", "rule", "confidence"})
_SIDECAR_COLUMNS = frozenset({"id", "path"})
_ARCHIVE_COLUMNS = frozenset({"name", "size", "mtime"})


class IndexUnusable(Exception):
    """The index is absent, unreadable, or of a shape this code does not know."""


@dataclass(frozen=True)
class IndexedPairing:
    sidecar: str | None
    confidence: str
    rule: str


@dataclass(frozen=True)
class IndexPairings:
    """Pairings read from a Takeout_Inventory index.

    Only archives whose name, size and mtime match the index are covered;
    every other member falls back to `pairing.py`.
    """

    path: Path
    covered: frozenset[str] = frozenset()
    uncovered: frozenset[str] = frozenset()
    pairings: Mapping[str, IndexedPairing] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path, archive_stats: Mapping[str, Any]) -> "IndexPairings":
        """Load *path*, verifying it against the archives actually on disk.

        `archive_stats` maps an archive's file name to its `os.stat_result`
        (anything with `st_size` and `st_mtime` works).
        """
        path = Path(path)
        if not path.is_file():
            raise IndexUnusable(f"no index at {path}")
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise IndexUnusable(f"cannot open {path}: {exc}") from exc
        try:
            cls._verify(con, path)
            covered, uncovered = cls._verify_archives(con, archive_stats)
            pairings = cls._read_pairings(con, covered)
        except sqlite3.Error as exc:
            raise IndexUnusable(f"{path} is not a readable index: {exc}") from exc
        finally:
            con.close()
        return cls(path=path, covered=frozenset(covered),
                   uncovered=frozenset(uncovered), pairings=pairings)

    @staticmethod
    def _verify(con: sqlite3.Connection, path: Path) -> None:
        """Version and column checks.

        The column check exists because this file and its producer live in
        different repositories: a schema change there must surface here as a
        clear error, never as a wrong answer.
        """
        try:
            row = con.execute(
                "SELECT value FROM index_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise IndexUnusable(
                f"{path} has no index_meta table; it predates schema versioning"
            ) from exc
        if row is None:
            raise IndexUnusable(f"{path} records no schema_version")
        try:
            found = int(row[0])
        except (TypeError, ValueError) as exc:
            raise IndexUnusable(f"{path} has a non-numeric schema_version") from exc
        if found > SCHEMA_VERSION:
            raise IndexUnusable(
                f"{path} is schema {found}; this build knows {SCHEMA_VERSION}")
        for table, expected in (("media", _MEDIA_COLUMNS),
                                ("sidecar", _SIDECAR_COLUMNS),
                                ("archive", _ARCHIVE_COLUMNS)):
            names = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            missing = expected - names
            if missing:
                raise IndexUnusable(
                    f"{path} table {table} is missing {sorted(missing)}")

    @staticmethod
    def _verify_archives(
        con: sqlite3.Connection, archive_stats: Mapping[str, Any],
    ) -> tuple[set[str], set[str]]:
        indexed = {
            name: (size, mtime)
            for name, size, mtime in con.execute(
                "SELECT name, size, mtime FROM archive")
        }
        covered: set[str] = set()
        uncovered: set[str] = set()
        for name, st in archive_stats.items():
            entry = indexed.get(name)
            if entry is not None and entry == (st.st_size, int(st.st_mtime)):
                covered.add(name)
            else:
                uncovered.add(name)
        return covered, uncovered

    @staticmethod
    def _read_pairings(
        con: sqlite3.Connection, covered: set[str],
    ) -> dict[str, IndexedPairing]:
        out: dict[str, IndexedPairing] = {}
        for m_path, m_archive, s_path, rule, confidence in con.execute(
            "SELECT m.path, m.archive, s.path, m.rule, m.confidence"
            " FROM media m LEFT JOIN sidecar s ON s.id = m.sidecar_id"
        ):
            if m_archive not in covered:
                continue
            out[m_path] = IndexedPairing(
                sidecar=s_path, confidence=confidence, rule=rule)
        return out

    def covers(self, archive_name: str) -> bool:
        return archive_name in self.covered

    def sidecar_for(self, member_path: str) -> IndexedPairing | None:
        """The pairing for *member_path*, or None if the index does not have it."""
        return self.pairings.get(member_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_takeout_index_reader.py -v`
Expected: PASS, 12 tests

Run: `uv run pytest`
Expected: PASS, full suite

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/index_reader.py tests/test_takeout_index_reader.py
git commit -m "feat: read a Takeout_Inventory pairing index"
```

---

### Task 5: Route ingest through the index, and apply the policy

**Files:**
- Modify: `imageharbor/takeout/ingest.py`
- Test: `tests/test_takeout_ingest.py`

**Interfaces:**
- Consumes: `pairing.Pairing`, `pairing.OWN/RELATED/NO_MATCH`, `index_reader.IndexPairings`, `tiers.DATE_RELATED_SIDECAR`.
- Produces: `TakeoutIngestor.__init__` gains `index: IndexPairings | None = None`; `IngestStats` gains five counters; a private `self._pairing_for(member_path, archive_name) -> pairing.Pairing`.

**The policy, applied where the sidecar is used:**

| From a `related` pairing | Action |
| --- | --- |
| `ExternalEvidence.date` | kept, resolved at `DATE_RELATED_SIDECAR` |
| `ExternalEvidence.original_name` (Google `title`) | **passed as `None`** |
| the `people` block | **not written** |
| the raw provenance document | **written, and labelled** with `confidence` and `pair_rule` |

The raw document is kept because deleting it would violate the
preserve-everything discipline and destroy the audit trail. Labelling it makes
the `geoData` inside `raw` self-describing rather than silently authoritative.

- [ ] **Step 1: Write the failing test**

The module is `tests/test_takeout_ingest.py` (not `tests/takeout/`). It already
has everything needed: `_jpeg(n)`, `_sidecar(title, seconds)`, `_zip(path,
entries)`, the `dirs` and `catalog` fixtures, and `D` as the member directory
prefix. `_sidecar` already emits a `geoData` block. Extend `_sidecar` with an
optional `people` argument rather than writing a second builder:

```python
def _sidecar(title: str, seconds: int, people: tuple[str, ...] = ()) -> bytes:
    doc = {
        "title": title,
        "creationTime": {"timestampSeconds": str(seconds + 14836)},
        "photoTakenTime": {"timestampSeconds": str(seconds)},
        "geoData": {"latitude": 38.2768361, "longitude": -85.7357389},
    }
    if people:
        doc["people"] = [{"name": n} for n in people]
    return json.dumps(doc).encode()


def _read_sidecar(dest: Path, stem_contains: str) -> dict:
    """The JSON sidecar ImageHarbor wrote beside the one organized file whose
    name contains *stem_contains*."""
    hits = [p for p in dest.rglob("*.json") if stem_contains in p.name]
    assert len(hits) == 1, [p.name for p in hits]
    return json.loads(hits[0].read_text(encoding="utf-8"))
```

Then append:

```python
def test_related_pairing_keeps_the_date_and_drops_title_and_people(
    dirs, catalog: Catalog
) -> None:
    """An -edited copy inherits its ORIGINAL's sidecar. The capture instant is
    this photograph's; the title and the people are the original's."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1-edited.jpg": _jpeg(2),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792,
                                        people=("Alice",)),
    })
    ingest_archives(archives, dest, catalog)

    edited = _read_sidecar(dest, "IMG_1-edited")
    prov = edited["provenance"][0]
    assert prov["confidence"] == "related"
    assert prov["pair_rule"]                       # recorded, never blank
    # The document is kept verbatim - deleting it would destroy the audit
    # trail - but it is labelled, so the coordinates in it are not silently
    # this photo's.
    assert prov["raw"]["geoData"]["latitude"] == 38.2768361
    assert "people" not in edited


def test_own_pairing_keeps_title_and_people(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792,
                                        people=("Alice",)),
    })
    ingest_archives(archives, dest, catalog)

    own = _read_sidecar(dest, "IMG_1")
    assert own["provenance"][0]["confidence"] == "own"
    assert own["people"] == [{"name": "Alice", "source": "google_photos_people"}]


def test_an_uncovered_archive_falls_back_and_is_counted(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """A stale index must never fail an ingest, and never be silent about it."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792),
    })
    # An index describing an archive with the right name and the wrong size.
    stale = _make_stale_index(tmp_path / "takeout-index.sqlite",
                              name="takeout-001.zip", size=1, mtime=1)
    stats = ingest_archives(archives, dest, catalog, index_path=stale)

    assert stats.index_archives_covered == 0
    assert stats.index_archives_fell_back == 1
    assert stats.ingested == 1        # identical to a no-index run
    assert stats.missing_metadata == 0
```

`_make_stale_index` builds a minimal index using the same schema literal as
`tests/test_takeout_index_reader.py`; import it from there rather than
duplicating the SQL.

Read two neighbouring ingest tests before writing, and confirm `_read_sidecar`
matches where sidecars are actually written — if `--no-sidecar` semantics or
the sidecar's location differ from the assumption above, follow the code.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_takeout_ingest.py -v`
Expected: FAIL — `KeyError: 'confidence'`

- [ ] **Step 3: Write the implementation**

Add to `IngestStats`:

```python
    # Takeout index (optional). A run without one leaves all five at zero and
    # reports "no index"; they are never a failure signal on their own.
    # None when no index was used at all, which the summary reports on one
    # line so "did it use the index?" is never answered by reading logs.
    index_path: Path | None = None
    index_archives_covered: int = 0
    index_archives_fell_back: int = 0
    index_members_fell_back: int = 0    # covered archive, member absent from index
    index_sidecars_missing: int = 0     # index named a sidecar not on disk
    pairings_related: int = 0
```

Add the routing helper:

```python
    def _pairing_for(self, member_path: str, archive_name: str) -> pairing.Pairing:
        """The pairing for one member, from the index when it covers this
        archive and knows this member, otherwise from the built-in rungs.

        Every fallback is counted. A silent fallback would make a stale index
        indistinguishable from a working one.
        """
        if self.index is not None and self.index.covers(archive_name):
            found = self.index.sidecar_for(member_path)
            if found is None:
                self.stats.index_members_fell_back += 1
            elif found.sidecar is not None and found.sidecar not in self._all_members:
                self.stats.index_sidecars_missing += 1
            else:
                return pairing.Pairing(found.sidecar, found.confidence)
        return pairing.sidecar_for(member_path, self.pairing_index)
```

`self._all_members` is a `frozenset` of every member path in the batch; build
it beside `self.pairing_index`, from the same `all_members` list.

Replace the three direct `pairing.sidecar_for(...)` calls with
`self._pairing_for(member_path, identity.path.name)`, and count
`stats.pairings_related` where the confidence is `RELATED`.

Where `ExternalEvidence` is constructed (around line 611), gate the title:

```python
            result = self.pipeline.process_file(
                staged,
                source_label=self._label(identity.path, member_path),
                evidence=ExternalEvidence(
                    date=meta.photo_taken_at,
                    # A related sidecar's `title` is the ORIGINAL's filename.
                    # Feeding it to the descriptor ladder would rename an edit
                    # after its parent.
                    original_name=(
                        meta.title if confidence == pairing.OWN else None),
                ),
            )
```

Thread the resolved `confidence` into `_write_takeout_sidecar`. There, gate the
people block and label the provenance entry:

```python
                entry: dict[str, Any] = {
                    "kind": "takeout_media_json",
                    "archive_id": identity.archive_id,
                    "archive": identity.path.name,
                    "member": sidecar_member,
                    "digest": _digest_bytes(sidecar_raw),
                    # How this document came to be attached to this photo. A
                    # `related` document describes a DIFFERENT file, so the
                    # geoData inside `raw` is not this photo's location.
                    # Recorded rather than deleted: dropping it would destroy
                    # the audit trail.
                    "confidence": confidence,
                    "pair_rule": pair_rule,
                }
```

```python
            # Face tags belong to the file the sidecar names. A related
            # sidecar names a different file.
            if meta.people and confidence == pairing.OWN:
                updates["people"] = [
                    {"name": n, "source": "google_photos_people"} for n in meta.people
                ]
```

For the date tier: find where the Takeout date is handed to the date resolver
and pass `DATE_RELATED_SIDECAR` instead of `DATE_EXTERNAL_SIDECAR` when the
confidence is `RELATED`. Read `pipeline.py` and `date_resolver.py` to find how
`ExternalEvidence.date` is currently ranked, and thread the tier through the
same path rather than inventing a second one.

`pair_rule` comes from the index when it supplied the pairing, and is the
string `"builtin"` when the built-in rungs did — the built-in ladder does not
name its rungs, and inventing names for them here would be fiction.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest`
Expected: PASS, full suite

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/ingest.py tests/test_takeout_ingest.py
git commit -m "feat: route pairings through the index and apply the related policy"
```

---

### Task 6: CLI flag, auto-detection, and reporting

**Files:**
- Modify: `imageharbor/cli.py`, `imageharbor/takeout/ingest.py` (`ingest_archives` signature and summary)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `--takeout-index PATH` on `imageharbor takeout ingest`; `ingest_archives(..., index_path: Path | None = None)`; four summary lines.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_takeout_ingest_rejects_an_explicit_missing_index(tmp_path):
    """You named something specific and did not get it."""
    result = runner.invoke(cli, [
        "takeout", "ingest", "--archives", str(archives), "--dest", str(dest),
        "--takeout-index", str(tmp_path / "absent.sqlite")])
    assert result.exit_code != 0
    assert "index" in result.output.lower()


def test_takeout_ingest_without_an_index_says_so(tmp_path):
    result = runner.invoke(cli, [
        "takeout", "ingest", "--archives", str(archives), "--dest", str(dest)])
    assert result.exit_code == 0
    # "Did it use the index?" must never be answered by reading logs.
    assert "no index" in result.output.lower()
```

Match the module's existing runner/fixture names.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `no such option: --takeout-index`

- [ ] **Step 3: Write the implementation**

Add the option to `takeout_ingest`, beside `--include-trash`:

```python
@click.option(
    "--takeout-index",
    "index_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "A takeout-index.sqlite published by Takeout_Inventory. Supplies the "
        "confidence of each sidecar pairing. Auto-detected beside --archives "
        "when present; pass this to require a specific one."
    ),
)
```

Thread `index_path` into `ingest_archives`. In the ingestor, resolve it:

- explicit path → `IndexPairings.open(...)`; on `IndexUnusable`, re-raise as a
  CLI error with the message
- no explicit path → look for `archives_dir / "takeout-index.sqlite"`; if
  absent, no index, silently; if present but `IndexUnusable`, log a warning
  and continue with no index

Archive stats for verification come from the same `Path.stat()` the ingestor
already performs per archive; collect them before opening the index.

Add the summary lines:

```python
    if stats.index_path is None:
        click.echo("takeout index : none - pairings from built-in rules")
    else:
        click.echo(f"takeout index : {stats.index_path.name} "
                   f"(schema {index_reader.SCHEMA_VERSION}, "
                   f"{stats.index_archives_covered + stats.index_archives_fell_back} archives)")
        click.echo(f"  archives    : {stats.index_archives_covered} indexed, "
                   f"{stats.index_archives_fell_back} fell back")
        click.echo(f"  pairings    : {stats.ingested - stats.pairings_related} own · "
                   f"{stats.pairings_related} related · "
                   f"{stats.missing_metadata} unpaired")
        click.echo(f"  fallbacks   : {stats.index_members_fell_back} members, "
                   f"{stats.index_sidecars_missing} missing sidecars")
```

Read the existing summary block and match its formatting and alignment.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest`
Expected: PASS

Run: `uv run imageharbor takeout ingest --help`
Expected: `--takeout-index` listed

- [ ] **Step 5: Commit**

```bash
git add imageharbor/cli.py imageharbor/takeout/ingest.py tests/test_cli.py
git commit -m "feat: --takeout-index, auto-detection, and index reporting"
```

---

### Task 7: The two invariant tests

**Files:**
- Test: `tests/test_takeout_index_equivalence.py` (create)

**Interfaces:** none — these are property tests over existing behaviour.

**Why these two:** every other test in this plan checks a mechanism. These
check the two properties the design rests on, and they are the only tests that
would catch a decoding divergence between the two repositories, or a fallback
that half-applies.

- [ ] **Step 1: Write the tests**

Create `tests/test_takeout_index_equivalence.py`:

```python
"""The two properties the optional-index design rests on."""


def test_the_two_pairing_paths_never_name_different_sidecars(tmp_path):
    """The index may pair where the built-in rungs return None -- it has rules
    ImageHarbor lacks. It must never name a DIFFERENT sidecar for the same
    member: that would be two implementations of one domain disagreeing, and
    whichever ran would decide a photo's date.
    """
    # Build one synthetic export covering every shape both paths handle:
    # exact, supplemental, copy-suffix (N), -edited, case-differing, truncated.
    ...
    for member in media_members:
        builtin = pairing.sidecar_for(member, builtin_index)
        indexed = index.sidecar_for(member)
        if builtin.sidecar is None or indexed is None or indexed.sidecar is None:
            continue
        assert builtin.sidecar == indexed.sidecar, (
            f"{member}: built-in says {builtin.sidecar}, "
            f"index says {indexed.sidecar}")


def test_a_mismatched_index_changes_nothing(tmp_path):
    """What makes 'optional' safe rather than merely intended."""
    without = _ingest(archives, dest_a, index_path=None)
    with_stale = _ingest(archives, dest_b, index_path=stale_index)

    assert _catalog_rows(dest_b) == _catalog_rows(dest_a)
    assert with_stale.index_archives_covered == 0
    assert with_stale.index_archives_fell_back == len(archive_names)
```

Build the synthetic index by calling `Takeout_Inventory`'s own writer if that
repo is importable from the test environment; otherwise construct the SQLite
from the same schema literal `test_index_reader.py` uses. Say which you did in
a comment — if it is the literal, the equivalence test is checking ImageHarbor
against ImageHarbor's *idea* of the schema, which is weaker and must be
labelled as such.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_takeout_index_equivalence.py -v`
Expected: PASS

- [ ] **Step 3: Prove the equivalence test can fail**

Temporarily make the index's pairing for one member name a different existing
sidecar, confirm the test FAILS naming both paths, then restore. Report the
assertion text. A test that cannot fail is this project's documented weak spot:
a previous plan shipped seven of them, and a later one shipped six changes
whose complete reverts passed the suite green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_takeout_index_equivalence.py
git commit -m "test: pin the equivalence and optionality invariants"
```

---

### Task 8: Corpus test and documentation

**Files:**
- Test: `tests/test_takeout_corpus_index.py` (create)
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Write the corpus test**

Follow this repo's existing opt-in corpus convention — read how the current
corpus-marked tests skip when no export is present and match it exactly.

```python
def test_indexed_and_builtin_agree_on_the_real_export():
    """Invariants only. The corpus is not a fixture: never assert a count."""
    ...
    assert covered == len(archive_names), "the index should cover a matching export"
    for member in sample:
        builtin = pairing.sidecar_for(member, builtin_index)
        indexed = index.sidecar_for(member)
        if builtin.sidecar and indexed and indexed.sidecar:
            assert builtin.sidecar == indexed.sidecar
    assert all(p.confidence in ("own", "related", "none")
               for p in index.pairings.values())
```

- [ ] **Step 2: Run it**

Run: `uv run pytest -m corpus -v`
Expected: PASS if an export and index are present; SKIPPED otherwise. Both are
acceptable outcomes.

- [ ] **Step 3: Document it**

In `README.md`, in the Takeout ingestion section: what `--takeout-index` is,
that it is optional, and what the three confidence values mean — specifically
that a `related` pairing contributes a date but never a title or people, and
that its provenance document is labelled rather than trusted.

In `CLAUDE.md`, add an invariant next to the existing ones:

> **A pairing's confidence decides what it may contribute.** `own` may supply
> date, title and people. `related` supplies the date only — its sidecar names
> a different file, and its title and location belong to that file. This holds
> whether the pairing came from a Takeout_Inventory index or the built-in
> rungs; the policy must never depend on whether a second tool was run.

- [ ] **Step 4: Commit**

```bash
git add tests/test_takeout_corpus_index.py README.md CLAUDE.md
git commit -m "test: corpus invariants for the pairing index; document it"
```

---

## Notes against the spec

- The spec's drop-list named `latitude`/`longitude` as dropped. They are not
  fields ImageHarbor carries: they exist only inside the raw Google JSON
  stored under `provenance`. The spec was amended during planning — the raw
  document is kept and **labelled** with `confidence` and `pair_rule` rather
  than deleted, because deleting it would violate the preserve-everything
  discipline. Tasks 5 and 8 implement the amended version.
- The spec described confidence as following the rung number, then corrected
  itself for rung 5. Task 2 implements the cleaner form the code allows:
  confidence follows the **name variant**, which makes rung 5 correct for free.
