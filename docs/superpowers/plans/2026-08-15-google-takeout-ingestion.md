# Google Takeout Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest Google Takeout `.zip` archives into the ImageHarbor library so every photo they contain is organized, verified, and cataloged — without ever modifying an archive, and such that re-running or resuming costs nothing and changes nothing.

**Architecture:** A new `imageharbor/takeout/` package with three pure, exhaustively-testable modules (`metadata.py`, `pairing.py`, and the classification half of `archive.py`) and one orchestrator with side effects (`ingest.py`). Ingestion is two-phase: a survey that reads only zip central directories and builds a *global* pairing index across every archive in the batch, then a resumable per-member ingest that hands each extracted member to the existing facts pass via a new `ExternalEvidence` parameter object. The date ladder gains one already-reserved rung; nothing else about placement or naming changes.

**Tech Stack:** Python 3, stdlib `zipfile`/`json`/`sqlite3`, Click for the CLI, pytest for tests, `uv` for everything.

## Global Constraints

Every task's requirements implicitly include this section. These are copied from
`docs/superpowers/specs/2026-08-12-google-takeout-ingestion-design.md` and
`CLAUDE.md`'s "Critical invariants".

- **Archives are opened `'r'` only.** Never `'a'`, never `'w'`. Nothing is written into, alongside, or in place of an archive.
- **`creationTime` never enters the date ladder.** It records upload time, not capture time. Parse it, record it as provenance, never place a file with it.
- **`geoData`, `people[]`, `favorited`, album membership, and Google's `exif` block are recorded, never load-bearing.** They cannot move or rename a file.
- **No new descriptor tier.** Google's `title` feeds the existing `DESC_HUMAN_FILENAME` (30) rung with better evidence; it does not outrank it.
- **`tiers.py` is not modified.** `DATE_EXTERNAL_SIDECAR = 30` already exists and is reserved for exactly this.
- **`SCHEMA_VERSION` stays `"2"`.** Both new tables are purely additive; no existing row is reinterpreted and no existing column changes meaning.
- **No circuit breaker.** This pass makes no AI calls, exactly like the facts pass. It must not consult or feed a breaker.
- **Never guess a pairing.** If no rule produces exactly one sidecar match, return `None`. A wrong pairing writes another photo's date into this photo's name — worse than an absent date.
- **Every new parameter on an existing function defaults to today's behavior** (`None`/`False`), so `process`, `enrich`, `watch`, and `watcher.py` behave byte-for-byte identically when the argument is omitted. New parameters on existing *methods* are keyword-only; new parameters on `__init__` follow the surrounding constructor's existing positional-or-keyword style.
- **Copy → verify → catalog ordering is preserved**, including on the `consume_source` path (rename → verify → catalog).
- **Terminal member statuses are never revisited:** `ingested`, `duplicate`, `deferred`, `parsed`, `ignored`, `skipped_trash`. Non-terminal: `pending`, `failed`.
- Commands: `uv sync --extra dev` to install, `uv run pytest` to test. Never `pip`, never `venv`.
- Do not chain shell commands with `&&`; run them as separate calls.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `imageharbor/descriptor.py` | modify | New `CAMERA_PATTERNS` rows; `original_name` and `date_str` keyword params |
| `imageharbor/date_resolver.py` | modify | `external_date` keyword param; new `DATE_EXTERNAL_SIDECAR` rung |
| `imageharbor/discovery.py` | modify | `VIDEO_EXTENSIONS` frozenset (classification only) |
| `imageharbor/catalog.py` | modify | `takeout_archives` + `takeout_members` tables and accessors |
| `imageharbor/pipeline.py` | modify | `ExternalEvidence`; `process_file(source_label=, evidence=)`; `consume_source` |
| `imageharbor/takeout/__init__.py` | create | Package marker; re-export `ingest_archives`, `IngestStats` |
| `imageharbor/takeout/metadata.py` | create | Pure Google-JSON parser. Never raises |
| `imageharbor/takeout/pairing.py` | create | Pure media→sidecar matcher across Google's naming mutations |
| `imageharbor/takeout/archive.py` | create | Archive identity, central-directory enumeration, member classification, extraction |
| `imageharbor/takeout/ingest.py` | create | Two-phase orchestrator; the only module here with side effects |
| `imageharbor/cli.py` | modify | `takeout ingest` / `takeout status` command group |
| `.gitignore` | modify | `.takeout-staging/`, `imageharbor/*.zip` |
| `tests/test_takeout_metadata.py` | create | Parser table tests |
| `tests/test_takeout_pairing.py` | create | Pairing fixture table, including must-return-None cases |
| `tests/test_takeout_archive.py` | create | Identity fast path, enumeration, CRC failure |
| `tests/test_takeout_ingest.py` | create | Behavioral core: idempotency, resume, late sidecar, trash, videos |
| `tests/test_descriptor.py` | modify | New camera-pattern rows; `original_name`; date-equal discard |
| `tests/test_date_resolver.py` | modify | Ladder ordering with `external_date` |
| `tests/test_pipeline.py` | modify | `consume_source`; `source_label`; `evidence` |
| `tests/test_monotonicity.py` | modify | Re-ingest is a rename no-op when tiers tie |
| `tests/test_cli.py` | modify | `takeout ingest` / `takeout status` |
| `CLAUDE.md`, `README.md` | modify | Document the new package, tables, and verbs |

---

### Task 1: Descriptor fixes — machine-generated Takeout names and date-equal descriptors

The real export contains `865948477697870747_account_id=1.jpg` and
`2015-03-09(1).jpg`. Neither matches any current `CAMERA_PATTERNS` entry, so both
resolve to `DESC_HUMAN_FILENAME` (tier 30), which outranks `DESC_AI_SUBJECT` (20)
and would **permanently** prevent enrichment from ever naming those files. The
second also produces `2015-03-09-2015-03-09_<digest>.jpg`, stating the date twice.

**Files:**
- Modify: `imageharbor/descriptor.py`
- Modify: `imageharbor/pipeline.py:200-201` and `imageharbor/pipeline.py:285-286`
- Test: `tests/test_descriptor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resolve_descriptor(source_path: Path, *, original_name: str | None = None, date_str: str | None = None) -> ResolvedDescriptor`. Both new parameters are keyword-only and default to `None`, so the existing one-argument call in `enrich.py` and every existing test keep working.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_descriptor.py`. The first two lists extend the existing
module-level `CAMERA_STEMS` / `HUMAN_STEMS` tables — append the new rows to
`CAMERA_STEMS` in place rather than creating a second list, so the existing
`test_camera_stems_are_detected` parametrization picks them up:

```python
# Append these entries to the existing CAMERA_STEMS list:
    "865948477697870747_account_id=1",
    "112233445566778899_account_id=0",
    "2015-03-09",
    "2015-03-09(1)",
    "2015-03-09(12)",
```

Then add these tests at the end of the file:

```python
def test_account_id_stem_is_not_a_human_name(tmp_path: Path) -> None:
    """A Hangouts row id is machine-generated; it must not lock out enrichment."""
    path = tmp_path / "865948477697870747_account_id=1.jpg"
    assert resolve_descriptor(path).tier == tiers.DESC_NONE


def test_bare_date_stem_is_not_a_descriptor(tmp_path: Path) -> None:
    """A date is not a description -- the date ladder already captured it."""
    path = tmp_path / "2015-03-09.jpg"
    assert resolve_descriptor(path).tier == tiers.DESC_NONE


def test_bare_date_with_copy_suffix_is_not_a_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "2015-03-09(1).jpg"
    assert resolve_descriptor(path).tier == tiers.DESC_NONE


def test_date_shaped_stem_with_words_survives(tmp_path: Path) -> None:
    """Only a BARE date is discarded; a date plus human words is information."""
    path = tmp_path / "2015-03-09 emma birthday.jpg"
    resolved = resolve_descriptor(path)
    assert resolved.tier == tiers.DESC_HUMAN_FILENAME
    assert resolved.value == "2015-03-09-emma"


def test_descriptor_equal_to_resolved_date_is_discarded(tmp_path: Path) -> None:
    """A descriptor that merely restates the date carries no information."""
    path = tmp_path / "2015.03.09.jpg"
    # Without date_str this normalizes to "2015-03-09" at tier 30.
    assert resolve_descriptor(path).tier == tiers.DESC_HUMAN_FILENAME
    # With the date the ladder actually resolved, it is redundant.
    assert resolve_descriptor(path, date_str="2015-03-09").tier == tiers.DESC_NONE


def test_original_name_overrides_a_truncated_member_stem(tmp_path: Path) -> None:
    """Google's `title` is the pre-truncation filename: strictly better evidence."""
    path = tmp_path / "emma-graduation-ceremony-at-the-high-scho.jpg"
    resolved = resolve_descriptor(
        path, original_name="emma graduation ceremony at the high school.jpg"
    )
    assert resolved.tier == tiers.DESC_HUMAN_FILENAME
    assert resolved.value == "emma-graduation-ceremony"


def test_original_name_that_is_camera_generated_still_yields_none(tmp_path: Path) -> None:
    path = tmp_path / "truncated-thing.jpg"
    assert resolve_descriptor(path, original_name="IMG_1234.jpg").tier == tiers.DESC_NONE


def test_blank_original_name_falls_back_to_the_member_stem(tmp_path: Path) -> None:
    path = tmp_path / "beach trip.jpg"
    assert resolve_descriptor(path, original_name="  ").value == "beach-trip"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_descriptor.py -q`
Expected: FAIL — `test_account_id_stem_is_not_a_human_name` and the other new
tests fail on tier mismatch; `resolve_descriptor()` raises `TypeError` for the
unexpected keyword arguments `date_str` / `original_name`.

- [ ] **Step 3: Add the camera patterns**

In `imageharbor/descriptor.py`, append these two entries to the `CAMERA_PATTERNS`
tuple, immediately after the `re.compile(r"^\d{9,13}$")` bare-epoch entry:

```python
    # Hangouts / AlbumArchive row ids, present at volume in Google Takeout
    # exports: 865948477697870747_account_id=1.jpg
    re.compile(r"^\d{10,}_account_id=\d+$", re.I),
    # A BARE date, with or without Google's (N) copy suffix. A date is not a
    # description -- the date ladder already captured it, and keeping it here
    # would state the same fact twice in one filename. A date followed by
    # human words ("2015-03-09 emma birthday") does NOT match and survives.
    re.compile(r"^\d{4}-\d{2}-\d{2}(\(\d+\))?$"),
```

- [ ] **Step 4: Add the two keyword parameters**

Replace `resolve_descriptor` in `imageharbor/descriptor.py` with:

```python
def resolve_descriptor(
    source_path: Path,
    *,
    original_name: str | None = None,
    date_str: str | None = None,
) -> ResolvedDescriptor:
    """Derive a descriptor from *source_path*'s original filename.

    Returns tier ``DESC_HUMAN_FILENAME`` when the stem carries human intent,
    and ``DESC_NONE`` when it does not -- leaving the slot open for the AI
    enrichment pass to fill at the lower ``DESC_AI_SUBJECT`` tier.

    Parameters
    ----------
    original_name:
        A filename known to be closer to the original than *source_path*'s own
        -- Google Takeout's ``title``, which is the pre-truncation name of a
        member whose stem the export truncated. When supplied and non-blank it
        REPLACES the path's stem as the evidence, because it is strictly the
        better source. It does not create a new tier: a human-authored
        ``title`` lands at ``DESC_HUMAN_FILENAME`` like any other, and a
        camera-generated one is discarded like any other.
    date_str:
        The ``YYYY-MM-DD`` the date ladder actually resolved for this file, when
        the caller already knows it. A descriptor that merely restates the date
        carries no information beyond what the folder and the filename's date
        prefix already say, so it is discarded as ``DESC_NONE``.
    """
    stem = source_path.stem
    if original_name:
        candidate = Path(original_name.strip()).stem
        if candidate:
            stem = candidate

    if not stem or is_camera_generated(stem):
        return _NONE

    normalized = normalize_descriptor(stem)
    # normalize_descriptor falls back to "photo" for input with no usable
    # characters; that is not information, so treat it as absent.
    if not normalized or normalized == "photo":
        return _NONE

    if date_str and normalized == date_str:
        return _NONE

    return ResolvedDescriptor(
        value=normalized,
        tier=tiers.DESC_HUMAN_FILENAME,
        source=tiers.DESC_SOURCE_NAMES[tiers.DESC_HUMAN_FILENAME],
    )
```

- [ ] **Step 5: Wire `date_str` through the two pipeline call sites**

In `imageharbor/pipeline.py`, at line 201 (inside `_do_process`), change:

```python
        descriptor = resolve_descriptor(source_path)
```

to:

```python
        descriptor = resolve_descriptor(source_path, date_str=date.date_str)
```

And at line 286 (inside `_maybe_upgrade_from_duplicate`), change:

```python
        descriptor = resolve_descriptor(source_path)
```

to:

```python
        descriptor = resolve_descriptor(source_path, date_str=date.date_str)
```

Both sites already resolve `date` on the preceding line, so no reordering is needed.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all tests green (395 existing + the new descriptor tests).

- [ ] **Step 7: Commit**

```bash
git add imageharbor/descriptor.py imageharbor/pipeline.py tests/test_descriptor.py
git commit -m "fix: machine-generated Takeout stems must not lock out enrichment"
```

---

### Task 2: Date ladder — the external-sidecar rung

`tiers.DATE_EXTERNAL_SIDECAR = 30` was reserved on 2026-08-11 for exactly this.
This task populates it: Google's `photoTakenTime` ranks below EXIF
`DateTimeOriginal` and above `DateTimeDigitized`/`DateTime` and the filename rung.

**Files:**
- Modify: `imageharbor/date_resolver.py`
- Test: `tests/test_date_resolver.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `resolve_date(source_path: Path, exif_data: dict[str, Any], *, external_date: datetime | None = None) -> ResolvedDate`. Keyword-only with a default, so both existing call sites and all existing tests keep working. `external_date` is a **naive UTC** `datetime` (see Task 4 for why naive).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_date_resolver.py` (the file already has a `_p(tmp_path, name)`
helper that builds a `Path`; reuse it):

```python
from datetime import datetime


def test_exif_original_outranks_external_sidecar(tmp_path) -> None:
    exif = {"DateTimeOriginal": "2019:07:04 12:33:11"}
    resolved = resolve_date(
        _p(tmp_path, "IMG_1234.jpg"), exif, external_date=datetime(2015, 3, 9, 12, 56, 32)
    )
    assert resolved.tier == tiers.DATE_EXIF_ORIGINAL
    assert resolved.value == datetime(2019, 7, 4, 12, 33, 11)


def test_external_sidecar_outranks_exif_digitized(tmp_path) -> None:
    exif = {"DateTimeDigitized": "2019:07:04 12:33:11"}
    resolved = resolve_date(
        _p(tmp_path, "IMG_1234.jpg"), exif, external_date=datetime(2015, 3, 9, 12, 56, 32)
    )
    assert resolved.tier == tiers.DATE_EXTERNAL_SIDECAR
    assert resolved.source == "external_sidecar"
    assert resolved.value == datetime(2015, 3, 9, 12, 56, 32)
    assert resolved.folder == "2015/2015-03"


def test_external_sidecar_outranks_a_filename_pattern(tmp_path) -> None:
    resolved = resolve_date(
        _p(tmp_path, "IMG_20190704_123456.jpg"), {}, external_date=datetime(2015, 3, 9)
    )
    assert resolved.tier == tiers.DATE_EXTERNAL_SIDECAR
    assert resolved.value == datetime(2015, 3, 9)


def test_external_sidecar_is_used_when_there_is_nothing_else(tmp_path) -> None:
    resolved = resolve_date(_p(tmp_path, "photo.jpg"), {}, external_date=datetime(2015, 3, 9))
    assert resolved.tier == tiers.DATE_EXTERNAL_SIDECAR


def test_implausible_external_date_is_ignored(tmp_path) -> None:
    """An out-of-range external date must fall through, not be asserted."""
    resolved = resolve_date(_p(tmp_path, "photo.jpg"), {}, external_date=datetime(1600, 1, 1))
    assert resolved.tier == tiers.DATE_NONE
    assert resolved.folder == UNDATED_FOLDER


def test_external_date_none_leaves_the_ladder_unchanged(tmp_path) -> None:
    exif = {"DateTimeDigitized": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), exif, external_date=None)
    assert resolved.tier == tiers.DATE_EXIF_OTHER
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_date_resolver.py -q`
Expected: FAIL — `resolve_date() got an unexpected keyword argument 'external_date'`.

- [ ] **Step 3: Split the EXIF field table and insert the rung**

In `imageharbor/date_resolver.py`, replace the `_EXIF_FIELDS` constant:

```python
# EXIF fields in ladder order: (field name, tier).
_EXIF_FIELDS: tuple[tuple[str, int], ...] = (
    ("DateTimeOriginal", tiers.DATE_EXIF_ORIGINAL),
    ("DateTimeDigitized", tiers.DATE_EXIF_OTHER),
    ("DateTime", tiers.DATE_EXIF_OTHER),
)
```

with:

```python
# The top EXIF rung, kept separate because the external-sidecar rung sits
# between it and the rest of the ladder.
_EXIF_PRIMARY_FIELD = "DateTimeOriginal"

# The remaining EXIF fields, all at the same lower tier.
_EXIF_OTHER_FIELDS: tuple[str, ...] = ("DateTimeDigitized", "DateTime")
```

- [ ] **Step 4: Rewrite `resolve_date`**

Replace `resolve_date` with:

```python
def resolve_date(
    source_path: Path,
    exif_data: dict[str, Any],
    *,
    external_date: datetime | None = None,
) -> ResolvedDate:
    """Resolve *source_path*'s capture date from EXIF, an external sidecar, then
    its filename.

    Rungs are tried highest-first and the first plausible hit wins.  File mtime
    is never consulted.

    *external_date* is a capture date asserted by a trustworthy source outside
    the file's own bytes and path -- in practice Google Takeout's
    ``photoTakenTime``.  It sits below EXIF ``DateTimeOriginal``, which is the
    camera's own record, and above ``DateTimeDigitized``/``DateTime``, which
    frequently record a scan or an edit rather than the capture.  An
    implausible value is ignored rather than asserted, exactly like an
    implausible EXIF value.

    Google's ``creationTime`` must NEVER be passed here: it records when a file
    was uploaded, which is the same category of claim as file mtime.
    """
    dt = _parse_exif_datetime(exif_data.get(_EXIF_PRIMARY_FIELD))
    if dt is not None:
        return ResolvedDate(
            value=dt,
            tier=tiers.DATE_EXIF_ORIGINAL,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_EXIF_ORIGINAL],
        )

    if external_date is not None and _plausible(external_date):
        return ResolvedDate(
            value=external_date,
            tier=tiers.DATE_EXTERNAL_SIDECAR,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_EXTERNAL_SIDECAR],
        )

    for field in _EXIF_OTHER_FIELDS:
        dt = _parse_exif_datetime(exif_data.get(field))
        if dt is not None:
            return ResolvedDate(
                value=dt,
                tier=tiers.DATE_EXIF_OTHER,
                source=tiers.DATE_SOURCE_NAMES[tiers.DATE_EXIF_OTHER],
            )

    dt = date_from_filename(source_path.stem)
    if dt is not None:
        return ResolvedDate(
            value=dt,
            tier=tiers.DATE_FILENAME_PATTERN,
            source=tiers.DATE_SOURCE_NAMES[tiers.DATE_FILENAME_PATTERN],
        )

    logger.debug("No trustworthy date for %s -> %s", source_path.name, UNDATED_FOLDER)
    return _UNDATED
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/date_resolver.py tests/test_date_resolver.py
git commit -m "feat: populate the external-sidecar rung of the date ladder"
```

---

### Task 3: Catalog — `takeout_archives` and `takeout_members`

Two additive tables plus accessors. `SCHEMA_VERSION` stays `"2"`: no existing row
is reinterpreted and no existing column changes meaning, so
`Catalog._guard_legacy_catalog` correctly does not fire and existing catalogs
upgrade in place on open.

**Files:**
- Modify: `imageharbor/catalog.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all on `Catalog`:
  - `takeout_archive_get(archive_id: str) -> sqlite3.Row | None`
  - `takeout_archive_get_by_stat(last_path: str, size: int, mtime_ns: int) -> sqlite3.Row | None`
  - `takeout_archive_upsert(*, archive_id: str, last_path: str, size: int, mtime_ns: int, member_count: int = 0, status: str = "partial", last_error: str = "") -> None`
  - `takeout_archive_set_status(archive_id: str, status: str, last_error: str = "") -> None`
  - `takeout_archives_all() -> list[sqlite3.Row]`
  - `takeout_member_add(*, archive_id: str, member_path: str, kind: str, size: int, crc32: int, status: str) -> None`
  - `takeout_member_set(archive_id: str, member_path: str, *, status: str, sha256_b64url: str | None = None, taken_at: str | None = None, sidecar_path: str | None = None, last_error: str = "") -> None`
  - `takeout_members_pending(archive_id: str) -> list[sqlite3.Row]`
  - `takeout_members_all(archive_id: str) -> list[sqlite3.Row]`
  - `takeout_members_unskip_trash(archive_id: str) -> int`
  - `takeout_status_counts() -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_catalog.py`:

```python
def test_takeout_archive_roundtrip(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/nas/t1.zip", size=79, mtime_ns=1, member_count=196
        )
        row = cat.takeout_archive_get("A" * 43)
        assert row["status"] == "partial"
        assert row["member_count"] == 196
        assert row["last_path"] == "/nas/t1.zip"


def test_takeout_archive_stat_fast_path(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/nas/t1.zip", size=79, mtime_ns=1
        )
        assert cat.takeout_archive_get_by_stat("/nas/t1.zip", 79, 1)["archive_id"] == "A" * 43
        assert cat.takeout_archive_get_by_stat("/nas/t1.zip", 79, 2) is None
        assert cat.takeout_archive_get_by_stat("/other.zip", 79, 1) is None


def test_takeout_archive_upsert_updates_location_not_identity(tmp_path) -> None:
    """A moved archive keeps its id; only where it lives changes."""
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/old/t1.zip", size=79, mtime_ns=1
        )
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/new/t1.zip", size=79, mtime_ns=9
        )
        assert len(cat.takeout_archives_all()) == 1
        assert cat.takeout_archive_get("A" * 43)["last_path"] == "/new/t1.zip"


def test_takeout_member_add_never_resets_a_terminal_status(tmp_path) -> None:
    """Re-surveying an archive must not drag ingested members back to pending."""
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="a/b.jpg", kind="image",
            size=10, crc32=1, status="pending",
        )
        cat.takeout_member_set("A" * 43, "a/b.jpg", status="ingested", sha256_b64url="D" * 43)
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="a/b.jpg", kind="image",
            size=10, crc32=1, status="pending",
        )
        rows = cat.takeout_members_all("A" * 43)
        assert len(rows) == 1
        assert rows[0]["status"] == "ingested"
        assert rows[0]["sha256_b64url"] == "D" * 43


def test_takeout_members_pending_returns_pending_and_failed_only(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        for name, status in (
            ("p.jpg", "pending"), ("f.jpg", "failed"), ("i.jpg", "ingested"),
            ("d.jpg", "duplicate"), ("v.mp4", "deferred"), ("m.json", "parsed"),
            ("o.txt", "ignored"), ("t.jpg", "skipped_trash"),
        ):
            cat.takeout_member_add(
                archive_id="A" * 43, member_path=name, kind="image",
                size=1, crc32=1, status=status,
            )
        assert {r["member_path"] for r in cat.takeout_members_pending("A" * 43)} == {
            "p.jpg", "f.jpg",
        }


def test_takeout_members_unskip_trash(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="Trash/x.jpg", kind="image",
            size=1, crc32=1, status="skipped_trash",
        )
        assert cat.takeout_members_unskip_trash("A" * 43) == 1
        assert cat.takeout_members_pending("A" * 43)[0]["member_path"] == "Trash/x.jpg"


def test_takeout_status_counts(tmp_path) -> None:
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/t1.zip", size=1, mtime_ns=1, status="complete"
        )
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="a.jpg", kind="image",
            size=1, crc32=1, status="ingested",
        )
        cat.takeout_member_add(
            archive_id="A" * 43, member_path="b.mp4", kind="video",
            size=1, crc32=1, status="deferred",
        )
        counts = cat.takeout_status_counts()
        assert counts["archives"]["complete"] == 1
        assert counts["members"]["ingested"] == 1
        assert counts["members"]["deferred"] == 1
        # a.jpg was ingested with no sidecar recorded
        assert counts["missing_metadata"] == 1


def test_takeout_tables_do_not_bump_the_schema_version(tmp_path) -> None:
    from imageharbor.catalog import SCHEMA_VERSION

    assert SCHEMA_VERSION == "2"
    with Catalog(tmp_path / "c.db") as cat:
        cat.takeout_archive_upsert(
            archive_id="A" * 43, last_path="/t.zip", size=1, mtime_ns=1
        )
    # Reopening must not raise LegacyCatalogError or lose the row.
    with Catalog(tmp_path / "c.db") as cat:
        assert cat.takeout_archive_get("A" * 43) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: FAIL — `AttributeError: 'Catalog' object has no attribute 'takeout_archive_upsert'`.

- [ ] **Step 3: Add the tables to `_SCHEMA`**

In `imageharbor/catalog.py`, append to the `_SCHEMA` string, immediately before
the closing `"""` (after the `meta` table):

```sql
CREATE TABLE IF NOT EXISTS takeout_archives (
    archive_id    TEXT PRIMARY KEY,          -- SHA-256 b64url of the .zip itself
    last_path     TEXT    NOT NULL,
    size          INTEGER NOT NULL,
    mtime_ns      INTEGER NOT NULL,
    member_count  INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'partial',  -- partial|complete|corrupt
    last_error    TEXT    NOT NULL DEFAULT '',
    first_seen_at TEXT    NOT NULL,
    last_seen_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_takeout_archives_stat
    ON takeout_archives(last_path, size, mtime_ns);

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

- [ ] **Step 4: Add the accessors**

In `imageharbor/catalog.py`, insert this block immediately before the
`# ------` comment that introduces the `Taxonomy` section (just before
`def taxonomy_is_empty`):

```python
    # ------------------------------------------------------------------
    # Google Takeout ingestion
    #
    # Both tables are purely additive: no existing row is reinterpreted and no
    # existing column changes meaning, so SCHEMA_VERSION stays "2" and
    # `_guard_legacy_catalog` correctly does not fire for a catalog that
    # predates them.
    # ------------------------------------------------------------------

    def takeout_archive_get(self, archive_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM takeout_archives WHERE archive_id = ?", (archive_id,)
        ).fetchone()

    def takeout_archive_get_by_stat(
        self, last_path: str, size: int, mtime_ns: int
    ) -> sqlite3.Row | None:
        """The identity fast path: recognise an archive without hashing it.

        A match on (path, size, mtime_ns) is not proof of identical content, but
        it is never used as one: it only avoids re-hashing an archive we have
        already hashed at that exact path/size/mtime. Any change to any of the
        three falls through to the digest.
        """
        return self._conn.execute(
            """
            SELECT * FROM takeout_archives
            WHERE last_path = ? AND size = ? AND mtime_ns = ?
            """,
            (last_path, size, mtime_ns),
        ).fetchone()

    def takeout_archives_all(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute("SELECT * FROM takeout_archives ORDER BY last_path")
        )

    def takeout_archive_upsert(
        self,
        *,
        archive_id: str,
        last_path: str,
        size: int,
        mtime_ns: int,
        member_count: int = 0,
        status: str = "partial",
        last_error: str = "",
    ) -> None:
        """Record an archive, keyed by the digest of its own bytes.

        `last_path` and `mtime_ns` move on conflict -- the same archive may be
        copied or re-downloaded elsewhere -- but `archive_id` never does, so a
        renamed archive is recognised rather than re-ingested. `first_seen_at`
        is written once.
        """
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO takeout_archives (
                archive_id, last_path, size, mtime_ns, member_count, status,
                last_error, first_seen_at, last_seen_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(archive_id) DO UPDATE SET
                last_path    = excluded.last_path,
                mtime_ns     = excluded.mtime_ns,
                member_count = excluded.member_count,
                status       = excluded.status,
                last_error   = excluded.last_error,
                last_seen_at = excluded.last_seen_at
            """,
            (
                archive_id, last_path, size, mtime_ns, member_count, status,
                last_error, now, now,
            ),
        )
        self._conn.commit()

    def takeout_archive_set_status(
        self, archive_id: str, status: str, last_error: str = ""
    ) -> None:
        self._conn.execute(
            """
            UPDATE takeout_archives
            SET status = ?, last_error = ?, last_seen_at = ?
            WHERE archive_id = ?
            """,
            (status, last_error, _now_iso(), archive_id),
        )
        self._conn.commit()

    def takeout_member_add(
        self,
        *,
        archive_id: str,
        member_path: str,
        kind: str,
        size: int,
        crc32: int,
        status: str,
    ) -> None:
        """Record a member seen in an archive's central directory.

        DO NOTHING on conflict, deliberately. `archive_id` is the digest of the
        archive's own bytes, so the same id implies the same central directory,
        which implies the same kind/size/crc at the same member path -- there is
        nothing to refresh. What there IS to protect is `status`: re-surveying
        an archive must never drag an already-ingested member back to 'pending'
        and re-extract it.
        """
        self._conn.execute(
            """
            INSERT INTO takeout_members (
                archive_id, member_path, kind, size, crc32, status, updated_at
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(archive_id, member_path) DO NOTHING
            """,
            (archive_id, member_path, kind, size, crc32, status, _now_iso()),
        )
        self._conn.commit()

    def takeout_member_set(
        self,
        archive_id: str,
        member_path: str,
        *,
        status: str,
        sha256_b64url: str | None = None,
        taken_at: str | None = None,
        sidecar_path: str | None = None,
        last_error: str = "",
    ) -> None:
        """Record the outcome of ingesting one member."""
        self._conn.execute(
            """
            UPDATE takeout_members SET
                status = ?, sha256_b64url = ?, taken_at = ?, sidecar_path = ?,
                last_error = ?, updated_at = ?
            WHERE archive_id = ? AND member_path = ?
            """,
            (
                status, sha256_b64url, taken_at, sidecar_path, last_error,
                _now_iso(), archive_id, member_path,
            ),
        )
        self._conn.commit()

    def takeout_members_pending(self, archive_id: str) -> list[sqlite3.Row]:
        """Members still owed work: 'pending' (never tried) or 'failed' (retry).

        An ingest failure is a local filesystem or archive fault, not a backend
        outage, so a failed member is simply retried next run -- there is no
        quarantine ladder and no backoff here.
        """
        return list(
            self._conn.execute(
                """
                SELECT * FROM takeout_members
                WHERE archive_id = ? AND status IN ('pending', 'failed')
                ORDER BY member_path
                """,
                (archive_id,),
            )
        )

    def takeout_members_all(self, archive_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM takeout_members WHERE archive_id = ? ORDER BY member_path",
                (archive_id,),
            )
        )

    def takeout_members_unskip_trash(self, archive_id: str) -> int:
        """Return trash members to the work queue; returns how many moved.

        Called only when --include-trash is passed, so a user who changes their
        mind is not blocked by the terminal status recorded on an earlier run.
        """
        cur = self._conn.execute(
            """
            UPDATE takeout_members SET status = 'pending', updated_at = ?
            WHERE archive_id = ? AND status = 'skipped_trash'
            """,
            (_now_iso(), archive_id),
        )
        self._conn.commit()
        return cur.rowcount

    def takeout_status_counts(self) -> dict[str, Any]:
        """Aggregates for `imageharbor takeout status`."""
        archives = {
            row["status"]: row["n"]
            for row in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM takeout_archives GROUP BY status"
            )
        }
        members = {
            row["status"]: row["n"]
            for row in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM takeout_members GROUP BY status"
            )
        }
        missing = self._conn.execute(
            """
            SELECT COUNT(*) AS n FROM takeout_members
            WHERE kind = 'image' AND status IN ('ingested', 'duplicate')
              AND sidecar_path IS NULL
            """
        ).fetchone()["n"]
        return {"archives": archives, "members": members, "missing_metadata": missing}
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: takeout_archives and takeout_members catalog tables"
```

---

### Task 4: `takeout/metadata.py` — the pure Google-JSON parser

Handed `bytes`, returns a dataclass, **never raises** — the same discipline
`exif_reader.read_exif` uses. Malformed, truncated, empty, or absent input
returns an empty `TakeoutMetadata`.

**Files:**
- Create: `imageharbor/takeout/__init__.py`
- Create: `imageharbor/takeout/metadata.py`
- Test: `tests/test_takeout_metadata.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `TakeoutMetadata` frozen dataclass: `title: str | None`, `description: str | None`, `photo_taken_at: datetime | None`, `creation_at: datetime | None`, `latitude: float | None`, `longitude: float | None`, `people: tuple[str, ...]`, `favorited: bool`, `size_bytes: int | None`
  - `EMPTY: TakeoutMetadata` — the all-defaults singleton
  - `AlbumMetadata` frozen dataclass: `title: str | None`, `description: str | None`
  - `parse_photo_metadata(raw: bytes) -> TakeoutMetadata`
  - `parse_album_metadata(raw: bytes) -> AlbumMetadata`
- All datetimes are **naive UTC**. The rest of the date ladder is naive (EXIF has no timezone), `date_from_row` reconstructs naive, and mixing aware and naive values in one catalog column would make the stored ISO strings inconsistent. Convert with `datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_metadata.py`:

```python
"""Tests for the Google Takeout per-media JSON parser.

The first payload below is verbatim from the real export this design was
calibrated against (takeout-20230618T004316Z-001.zip, AlbumArchive schema).
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from imageharbor.takeout.metadata import (
    EMPTY,
    parse_album_metadata,
    parse_photo_metadata,
)

ALBUM_ARCHIVE_PAYLOAD = b"""{
  "title": "2015-03-09.jpg",
  "imageViews": "12",
  "creationTime":   { "timestampSeconds": "1425920628", "formatted": "Mar 9, 2015, 5:03:48 PM UTC" },
  "photoTakenTime": { "timestampSeconds": "1425905792", "formatted": "Mar 9, 2015, 12:56:32 PM UTC" },
  "geoData": { "latitude": 38.2768361, "longitude": -85.73573890000002 },
  "height": "2432", "width": "4320",
  "exif": { "apertureFNumber": 2.4, "cameraModel": "XT1056", "exposureTime": 0.01666,
            "focalLength": 4.499, "isoEquivalent": 640 },
  "sizeBytes": "3698139"
}"""


def test_parses_the_real_album_archive_payload() -> None:
    meta = parse_photo_metadata(ALBUM_ARCHIVE_PAYLOAD)
    assert meta.title == "2015-03-09.jpg"
    assert meta.photo_taken_at == datetime(2015, 3, 9, 12, 56, 32)
    assert meta.latitude == pytest.approx(38.2768361)
    assert meta.longitude == pytest.approx(-85.73573890000002)
    assert meta.size_bytes == 3698139
    # Absent in this schema; every field is optional.
    assert meta.description is None
    assert meta.people == ()
    assert meta.favorited is False


def test_creation_time_is_parsed_but_is_not_the_placement_date() -> None:
    """creationTime is upload time. It is recorded and never placed with.

    In the real export the two differ by four hours on the same file -- direct
    evidence for the rule.
    """
    meta = parse_photo_metadata(ALBUM_ARCHIVE_PAYLOAD)
    assert meta.creation_at == datetime(2015, 3, 9, 17, 3, 48)
    assert meta.creation_at != meta.photo_taken_at


def test_accepts_the_google_photos_timestamp_key() -> None:
    """Newer Google Photos exports use `timestamp`, not `timestampSeconds`."""
    raw = json.dumps(
        {
            "title": "IMG_1234.jpg",
            "description": "at the lake",
            "photoTakenTime": {"timestamp": "1425905792", "formatted": "..."},
            "people": [{"name": "Emma"}, {"name": "Sam"}],
            "favorited": True,
        }
    ).encode()
    meta = parse_photo_metadata(raw)
    assert meta.photo_taken_at == datetime(2015, 3, 9, 12, 56, 32)
    assert meta.description == "at the lake"
    assert meta.people == ("Emma", "Sam")
    assert meta.favorited is True


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json at all",
        b'{"photoTakenTime": {"timestampSeconds": "1425905792"',  # truncated
        b"[]",                       # top-level list
        b'"a string"',               # top-level scalar
        b"null",
        b'{"photoTakenTime": "not a dict"}',
        b'{"photoTakenTime": {"timestampSeconds": "not a number"}}',
        b'{"geoData": "not a dict"}',
        b'{"people": "not a list"}',
        b'{"sizeBytes": "not a number"}',
        b"\xff\xfe\x00garbage",      # not valid UTF-8
    ],
)
def test_malformed_input_never_raises(raw: bytes) -> None:
    meta = parse_photo_metadata(raw)
    assert meta.photo_taken_at is None
    assert meta.title is None


def test_implausible_timestamp_is_dropped() -> None:
    raw = json.dumps({"photoTakenTime": {"timestampSeconds": "-99999999999"}}).encode()
    assert parse_photo_metadata(raw).photo_taken_at is None


def test_null_island_geodata_is_treated_as_absent() -> None:
    """Google writes 0.0/0.0 for 'no location', which is not a location."""
    raw = json.dumps({"geoData": {"latitude": 0.0, "longitude": 0.0}}).encode()
    meta = parse_photo_metadata(raw)
    assert meta.latitude is None
    assert meta.longitude is None


def test_empty_strings_become_none() -> None:
    raw = json.dumps({"title": "", "description": "   "}).encode()
    meta = parse_photo_metadata(raw)
    assert meta.title is None
    assert meta.description is None


def test_empty_singleton_is_all_defaults() -> None:
    assert EMPTY.title is None
    assert EMPTY.photo_taken_at is None
    assert EMPTY.people == ()
    assert EMPTY.favorited is False


def test_parse_album_metadata() -> None:
    raw = json.dumps({"title": "Hangout: Emma", "description": "chat images"}).encode()
    album = parse_album_metadata(raw)
    assert album.title == "Hangout: Emma"
    assert album.description == "chat images"


def test_parse_album_metadata_never_raises() -> None:
    assert parse_album_metadata(b"{{{").title is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_takeout_metadata.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.takeout'`.

- [ ] **Step 3: Create the package marker**

Create `imageharbor/takeout/__init__.py`:

```python
"""Google Takeout archive ingestion.

A container walk rather than a filesystem walk. Three of the four modules here
are pure (`metadata`, `pairing`, and `archive.classify`) so the logic most
likely to be wrong can be tested exhaustively without a zip on disk;
`ingest` is the only module with side effects.

Archives are opened read-only and are never modified, moved, or written
alongside. Ingestion makes no AI calls, so -- exactly like the facts pass --
it never consults or feeds the circuit breaker.
"""

from __future__ import annotations

from .ingest import IngestStats, ingest_archives

__all__ = ["IngestStats", "ingest_archives"]
```

Note: this import will fail until Task 8 creates `ingest.py`. To keep the tree
importable in the meantime, create the file with only the docstring for now and
add the imports in Task 8.

- [ ] **Step 4: Write `metadata.py`**

Create `imageharbor/takeout/metadata.py`:

```python
"""Parse Google Takeout's per-media and per-album JSON sidecars.

Pure: handed ``bytes``, returns a dataclass, touches no filesystem. It never
raises -- malformed, truncated, empty, or absent input returns an empty
result, the same discipline ``exif_reader.read_exif`` uses. A sidecar is
supplementary evidence; a corrupt one must degrade a photo to "no external
date", never fail it.

Two export generations are in circulation and both are accepted: AlbumArchive
uses ``timestampSeconds``, newer Google Photos exports use ``timestamp``. Every
field is optional in both -- the AlbumArchive schema has no ``description`` and
no ``people`` at all.

All datetimes returned here are **naive UTC**. The rest of the date ladder is
naive (EXIF carries no timezone) and ``date_resolver.date_from_row`` rebuilds
naive values from the catalog, so returning aware datetimes would put two
incompatible kinds of value in one column.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Matches date_resolver's plausibility window: photography began in 1826, and
# anything past 2100 is a dead clock or a bad parse.
_MIN_YEAR = 1826
_MAX_YEAR = 2100

# Google writes 0.0/0.0 when it has no location. Null Island is not a location.
_NULL_ISLAND = (0.0, 0.0)


@dataclass(frozen=True)
class TakeoutMetadata:
    """What Google recorded about one media file.

    Only ``photo_taken_at`` and ``title`` are load-bearing (they feed the date
    and descriptor ladders). Everything else is recorded as provenance and can
    never move or rename a file.
    """

    title: str | None = None
    description: str | None = None
    photo_taken_at: datetime | None = None
    # RECORDED ONLY. creationTime is when the file was uploaded to Google
    # Photos, not when the photo was taken -- the same category of claim as
    # file mtime, which date_resolver.py deliberately refuses. In the real
    # export the two differ by four hours on the same file. It must never be
    # passed to resolve_date().
    creation_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    people: tuple[str, ...] = ()
    favorited: bool = False
    size_bytes: int | None = None


@dataclass(frozen=True)
class AlbumMetadata:
    """What Google recorded about one album (Albums.json / metadata.json)."""

    title: str | None = None
    description: str | None = None


EMPTY = TakeoutMetadata()
EMPTY_ALBUM = AlbumMetadata()


def _load(raw: bytes) -> dict[str, Any] | None:
    """Decode *raw* into a JSON object, or None if it is not one."""
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        logger.debug("Unparseable Takeout sidecar (%s); treating as absent", exc)
        return None
    return data if isinstance(data, dict) else None


def _text(value: Any) -> str | None:
    """A non-blank string, or None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _timestamp(block: Any) -> datetime | None:
    """Epoch seconds out of a Google timestamp block, as naive UTC.

    Accepts ``timestampSeconds`` (AlbumArchive) and ``timestamp`` (Google
    Photos); both are strings holding epoch seconds UTC.
    """
    if not isinstance(block, dict):
        return None
    for key in ("timestampSeconds", "timestamp"):
        raw = block.get(key)
        if raw is None:
            continue
        try:
            seconds = int(str(raw).strip())
        except (TypeError, ValueError):
            continue
        try:
            dt = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            continue
        if _MIN_YEAR <= dt.year <= _MAX_YEAR:
            return dt
    return None


def _geo(block: Any) -> tuple[float | None, float | None]:
    if not isinstance(block, dict):
        return (None, None)
    try:
        lat = float(block["latitude"])
        lon = float(block["longitude"])
    except (KeyError, TypeError, ValueError):
        return (None, None)
    if (lat, lon) == _NULL_ISLAND:
        return (None, None)
    return (lat, lon)


def _people(block: Any) -> tuple[str, ...]:
    if not isinstance(block, list):
        return ()
    names = []
    for entry in block:
        if isinstance(entry, dict):
            name = _text(entry.get("name"))
        else:
            name = _text(entry)
        if name:
            names.append(name)
    return tuple(names)


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_photo_metadata(raw: bytes) -> TakeoutMetadata:
    """Parse one per-media JSON sidecar. Never raises."""
    data = _load(raw)
    if data is None:
        return EMPTY

    latitude, longitude = _geo(data.get("geoData"))
    # Some exports carry an emptied `geoData` alongside a populated
    # `geoDataExif`. Fall back to it, still as provenance only.
    if latitude is None:
        latitude, longitude = _geo(data.get("geoDataExif"))

    return TakeoutMetadata(
        title=_text(data.get("title")),
        description=_text(data.get("description")),
        photo_taken_at=_timestamp(data.get("photoTakenTime")),
        creation_at=_timestamp(data.get("creationTime")),
        latitude=latitude,
        longitude=longitude,
        people=_people(data.get("people")),
        favorited=data.get("favorited") is True,
        size_bytes=_int(data.get("sizeBytes")),
    )


def parse_album_metadata(raw: bytes) -> AlbumMetadata:
    """Parse an Albums.json / metadata.json album descriptor. Never raises."""
    data = _load(raw)
    if data is None:
        return EMPTY_ALBUM
    return AlbumMetadata(
        title=_text(data.get("title")),
        description=_text(data.get("description")),
    )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_takeout_metadata.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/takeout/__init__.py imageharbor/takeout/metadata.py tests/test_takeout_metadata.py
git commit -m "feat: pure parser for Google Takeout per-media JSON"
```

---

### Task 5: `takeout/pairing.py` — media → sidecar matching

The risk concentrate. A wrong pairing writes another photo's date into this
photo's name — precisely the quiet corruption the project exists to prevent.
**Never guess:** if no rule produces exactly one match, return `None`.

**Files:**
- Create: `imageharbor/takeout/pairing.py`
- Test: `tests/test_takeout_pairing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PairingIndex` dataclass (not frozen — it holds dicts)
  - `build_index(members: Iterable[str]) -> PairingIndex` — takes **all** member paths in the batch; it decides internally which are sidecars (`.json`) and which are media.
  - `sidecar_for(media_path: str, index: PairingIndex) -> str | None`

**Deliberate deviation from the spec, documented in the module:** the spec lists
truncation recovery as rung 5 and the case-insensitive extension retry as rung 6.
This implementation runs the case-insensitive retry **before** truncation
recovery. An exact match modulo extension case is strictly stronger evidence
than a unique-prefix match, so running the fuzzier rung first could shadow it.
The ordering of rungs 1–4 is unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_pairing.py`:

```python
"""Tests for media -> sidecar pairing across Google's naming mutations.

The (N)-displacement rows are verbatim from the real export:
    2015-03-09.jpg       2015-03-09.jpg.json
    2015-03-09(1).jpg    2015-03-09.jpg(1).json
    2015-03-09(2).jpg    2015-03-09.jpg(2).json
"""

from __future__ import annotations

import pytest

from imageharbor.takeout.pairing import build_index, sidecar_for

D = "Takeout/AlbumArchive/Hangouts/album"

MEMBERS = [
    f"{D}/2015-03-09.jpg",
    f"{D}/2015-03-09.jpg.json",
    f"{D}/2015-03-09(1).jpg",
    f"{D}/2015-03-09.jpg(1).json",
    f"{D}/2015-03-09(2).jpg",
    f"{D}/2015-03-09.jpg(2).json",
    f"{D}/IMG_1234.jpg",
    f"{D}/IMG_1234.jpg.supplemental-metadata.json",
    f"{D}/IMG_9999.jpg",                       # no sidecar anywhere
    f"{D}/edited-thing.jpg",
    f"{D}/edited-thing.jpg.json",
    f"{D}/edited-thing-edited.jpg",            # derivative: no sidecar of its own
    f"{D}/UPPER.JPG",
    f"{D}/UPPER.jpg.json",                     # extension case differs
    f"{D}/party ●●● 2015 + friends = fun.jpg",
    f"{D}/party ●●● 2015 + friends = fun.jpg.json",
    f"{D}/Albums.json",
]


@pytest.fixture()
def index():
    return build_index(MEMBERS)


@pytest.mark.parametrize(
    "media, expected",
    [
        (f"{D}/2015-03-09.jpg", f"{D}/2015-03-09.jpg.json"),
        (f"{D}/2015-03-09(1).jpg", f"{D}/2015-03-09.jpg(1).json"),
        (f"{D}/2015-03-09(2).jpg", f"{D}/2015-03-09.jpg(2).json"),
        (f"{D}/IMG_1234.jpg", f"{D}/IMG_1234.jpg.supplemental-metadata.json"),
        (f"{D}/edited-thing.jpg", f"{D}/edited-thing.jpg.json"),
        # Google emits no sidecar for an edited derivative; it inherits the
        # original's.
        (f"{D}/edited-thing-edited.jpg", f"{D}/edited-thing.jpg.json"),
        (f"{D}/UPPER.JPG", f"{D}/UPPER.jpg.json"),
        (
            f"{D}/party ●●● 2015 + friends = fun.jpg",
            f"{D}/party ●●● 2015 + friends = fun.jpg.json",
        ),
    ],
)
def test_pairing_table(index, media, expected) -> None:
    assert sidecar_for(media, index) == expected


def test_no_confident_match_returns_none(index) -> None:
    """Never guess: a photo without Google metadata is still fully organized."""
    assert sidecar_for(f"{D}/IMG_9999.jpg", index) is None


def test_a_sidecar_is_not_paired_with_itself(index) -> None:
    assert sidecar_for(f"{D}/2015-03-09.jpg.json", index) is None


def test_a_sidecar_in_another_directory_is_not_matched() -> None:
    """Pairing never crosses a directory boundary."""
    index = build_index(["a/x.jpg", "b/x.jpg.json"])
    assert sidecar_for("a/x.jpg", index) is None


def test_paren_form_is_not_shadowed_by_the_generic_rule() -> None:
    """`NAME(N).EXT.json` also exists in some exports; the displaced form wins."""
    index = build_index(
        ["d/p.jpg", "d/p.jpg.json", "d/p(1).jpg", "d/p.jpg(1).json", "d/p(1).jpg.json"]
    )
    assert sidecar_for("d/p(1).jpg", index) == "d/p.jpg(1).json"


def test_truncation_recovery_accepts_a_unique_prefix() -> None:
    long_media = "d/emma-graduation-ceremony-at-the-high-school-2.jpg"
    long_sidecar = "d/emma-graduation-ceremony-at-the-high-schoo.json"
    index = build_index([long_media, long_sidecar])
    assert sidecar_for(long_media, index) == long_sidecar


def test_truncation_recovery_refuses_an_ambiguous_prefix() -> None:
    media = "d/emma-graduation-ceremony-at-the-high-school-2.jpg"
    index = build_index(
        [
            media,
            "d/emma-graduation-ceremony-at-the-high-schoo.json",
            "d/emma-graduation-ceremony-at-the-high-scho.json",
        ]
    )
    assert sidecar_for(media, index) is None


def test_truncation_recovery_never_steals_a_claimed_sidecar() -> None:
    """A sidecar that exactly pairs with another member is off limits."""
    index = build_index(["d/photo.jpg", "d/photo.jpg.json", "d/photo.jpg-extra.jpg"])
    assert sidecar_for("d/photo.jpg-extra.jpg", index) is None


def test_truncation_recovery_ignores_a_too_short_prefix() -> None:
    """A short sidecar name prefixes half the directory; that is a guess."""
    index = build_index(["d/abcdefgh.jpg", "d/abc.json"])
    assert sidecar_for("d/abcdefgh.jpg", index) is None


def test_root_level_members_pair() -> None:
    index = build_index(["x.jpg", "x.jpg.json"])
    assert sidecar_for("x.jpg", index) == "x.jpg.json"


def test_media_with_no_extension_does_not_crash() -> None:
    index = build_index(["d/noext", "d/noext.json"])
    assert sidecar_for("d/noext", index) == "d/noext.json"


def test_index_is_global_across_archives() -> None:
    """Google splits by size, so a photo and its sidecar land in different parts.

    The index is built from every member in the batch precisely so that the
    part boundary is invisible here.
    """
    index = build_index(["d/a.jpg", "d/b.jpg", "d/a.jpg.json", "d/b.jpg.json"])
    assert sidecar_for("d/a.jpg", index) == "d/a.jpg.json"
    assert sidecar_for("d/b.jpg", index) == "d/b.jpg.json"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_takeout_pairing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.takeout.pairing'`.

- [ ] **Step 3: Write `pairing.py`**

Create `imageharbor/takeout/pairing.py`:

```python
"""Match a Takeout media member to its Google JSON sidecar.

Pure: string work over member paths, no filesystem, no zip. Zip member paths
always use ``/`` regardless of the host OS, so this module splits on ``/``
directly and never touches ``pathlib``.

Google mutates sidecar names in several ways, and the rules below were verified
at 86/86 = 100% against the real export. The single most important rule is the
one that produces no answer: **if no rung yields exactly one match, return
None.** The member is then ingested from EXIF and its filename alone, which is
a fully correct outcome -- a photo without Google metadata is organized,
verified, and cataloged like any other. A WRONG pairing, by contrast, writes
another photo's capture date into this photo's name and folder, which is
exactly the quiet corruption the project's SHA-256 discipline exists to
prevent.

Rung order:

1. ``NAME(N).EXT`` -> ``NAME.EXT(N).json``  (the suffix moves AFTER .json)
2. ``NAME.EXT`` -> ``NAME.EXT.json``
3. ``NAME.EXT`` -> ``NAME.EXT.supplemental-metadata.json``  (newer exports)
4. ``-edited`` derivatives: strip the suffix and retry 1-3. Google emits no
   sidecar for an edited copy; it inherits the original's.
5. Case-insensitive retry of 1-4.
6. Truncation recovery: a unique prefix match among UNCLAIMED sidecars in the
   same directory. Google Photos truncates member stems at roughly 47
   characters, and truncates the media name and its sidecar name to different
   lengths.

Rungs 5 and 6 are swapped relative to the design document, deliberately: an
exact match that differs only in extension case is strictly stronger evidence
than a unique-prefix match, so running the fuzzier rung first could shadow it.
Rungs 1-4 keep the specified order.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

_JSON_SUFFIX = ".json"
_SUPPLEMENTAL = ".supplemental-metadata.json"
_EDITED = "-edited"

# Google's copy suffix, e.g. "2015-03-09(1)".
_PAREN_RE = re.compile(r"^(?P<base>.*)\((?P<n>\d+)\)$")

# A truncation-recovery prefix shorter than this is not evidence, it is a
# coincidence: "abc.json" is a prefix of every "abc*" in the directory.
_MIN_TRUNCATION_PREFIX = 12


@dataclass
class PairingIndex:
    """Precomputed lookups over every member path in the batch.

    Holds only name->name strings: a 60 GB export can carry 100k+ members, and
    keeping the index at strings rather than parsed metadata keeps it near
    10 MB instead of hundreds. Sidecar CONTENT is read lazily, on demand, by
    the caller.
    """

    sidecars: frozenset[str] = frozenset()
    # Lowercased path -> the one real path, or None when two members differ
    # only in case (in which case a case-insensitive match would be a guess).
    sidecars_ci: Mapping[str, str | None] = field(default_factory=dict)
    # Directory -> sidecar paths inside it, for truncation recovery.
    by_dir: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Sidecars already matched exactly (rungs 1-5) by some media member.
    # Truncation recovery must never steal one of these.
    claimed: frozenset[str] = frozenset()


def _split(member_path: str) -> tuple[str, str]:
    """Return ``(directory, name)``; directory is ``""`` at the archive root."""
    directory, _, name = member_path.rpartition("/")
    return directory, name


def _is_sidecar(member_path: str) -> bool:
    return member_path.lower().endswith(_JSON_SUFFIX)


def _name_variants(name: str) -> list[str]:
    """The member name, then its pre-``-edited`` original (rung 4)."""
    variants = [name]
    base, dot, ext = name.rpartition(".")
    if dot and base.lower().endswith(_EDITED):
        variants.append(f"{base[: -len(_EDITED)]}.{ext}")
    return variants


def _candidates(media_path: str) -> list[str]:
    """Every exact sidecar path *media_path* could legitimately pair with."""
    directory, name = _split(media_path)
    prefix = f"{directory}/" if directory else ""
    out: list[str] = []
    for variant in _name_variants(name):
        base, dot, ext = variant.rpartition(".")
        if not dot:  # no extension at all
            base, ext = variant, ""
        # Rung 1 first: the (N) form is unambiguous and must not be shadowed
        # by the generic rule, which some exports ALSO satisfy.
        match = _PAREN_RE.match(base)
        if match and ext:
            out.append(f"{prefix}{match.group('base')}.{ext}({match.group('n')}){_JSON_SUFFIX}")
        out.append(f"{prefix}{variant}{_JSON_SUFFIX}")
        out.append(f"{prefix}{variant}{_SUPPLEMENTAL}")
    return out


def _media_part(sidecar_name: str) -> str:
    """The media-name portion of a sidecar's basename."""
    lower = sidecar_name.lower()
    if lower.endswith(_SUPPLEMENTAL):
        return sidecar_name[: -len(_SUPPLEMENTAL)]
    return sidecar_name[: -len(_JSON_SUFFIX)]


def _exact_match(media_path: str, index: PairingIndex) -> str | None:
    """Rungs 1-4, then rung 5 (the same candidates, case-insensitively)."""
    candidates = _candidates(media_path)
    for candidate in candidates:
        if candidate in index.sidecars:
            return candidate
    for candidate in candidates:
        hit = index.sidecars_ci.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _truncation_match(media_path: str, index: PairingIndex) -> str | None:
    """Rung 6: a unique prefix match among unclaimed sidecars in this directory."""
    directory, name = _split(media_path)
    matches = [
        sidecar
        for sidecar in index.by_dir.get(directory, ())
        if sidecar not in index.claimed
        and len(_media_part(_split(sidecar)[1])) >= _MIN_TRUNCATION_PREFIX
        and name.startswith(_media_part(_split(sidecar)[1]))
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        logger.debug(
            "Ambiguous truncation recovery for %s (%d candidates); no pairing",
            media_path, len(matches),
        )
    return None


def build_index(members: Iterable[str]) -> PairingIndex:
    """Build a pairing index over *members*.

    *members* is every member path in the batch -- media and sidecars, across
    every archive. Google's multi-part zips split by size across the file list,
    so ``IMG_1234.jpg`` can land in part 1 while ``IMG_1234.jpg.json`` lands in
    part 2; a per-archive index would silently lose metadata at every part
    boundary.
    """
    all_members = list(members)
    sidecars = [m for m in all_members if _is_sidecar(m)]
    media = [m for m in all_members if not _is_sidecar(m)]

    sidecar_set = frozenset(sidecars)

    ci: dict[str, str | None] = {}
    for sidecar in sidecars:
        key = sidecar.lower()
        # A second member differing only in case makes a case-insensitive hit
        # a guess rather than a match, so poison the key instead.
        ci[key] = None if key in ci else sidecar

    by_dir: dict[str, list[str]] = {}
    for sidecar in sidecars:
        by_dir.setdefault(_split(sidecar)[0], []).append(sidecar)

    partial = PairingIndex(
        sidecars=sidecar_set,
        sidecars_ci=ci,
        by_dir={k: tuple(sorted(v)) for k, v in by_dir.items()},
    )

    claimed = {
        match
        for media_path in media
        if (match := _exact_match(media_path, partial)) is not None
    }
    partial.claimed = frozenset(claimed)
    return partial


def sidecar_for(media_path: str, index: PairingIndex) -> str | None:
    """Return *media_path*'s sidecar member path, or None if none is certain."""
    if _is_sidecar(media_path):
        return None
    exact = _exact_match(media_path, index)
    if exact is not None:
        return exact
    return _truncation_match(media_path, index)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_takeout_pairing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/pairing.py tests/test_takeout_pairing.py
git commit -m "feat: Takeout media-to-sidecar pairing that refuses to guess"
```

---

### Task 6: `takeout/archive.py` — identity, enumeration, classification, extraction

**Files:**
- Create: `imageharbor/takeout/archive.py`
- Modify: `imageharbor/discovery.py`
- Test: `tests/test_takeout_archive.py`

**Interfaces:**
- Consumes: `Catalog.takeout_archive_get_by_stat` (Task 3).
- Produces:
  - `discovery.VIDEO_EXTENSIONS: frozenset[str]`
  - `MemberInfo(path: str, size: int, crc32: int, kind: str)` frozen dataclass
  - `ArchiveIdentity(archive_id: str, path: Path, size: int, mtime_ns: int)` frozen dataclass
  - `KIND_IMAGE`/`KIND_VIDEO`/`KIND_METADATA`/`KIND_ALBUM`/`KIND_OTHER` string constants
  - `classify(member_path: str) -> str`
  - `is_trash(member_path: str) -> bool`
  - `iter_members(zf: zipfile.ZipFile) -> Iterator[MemberInfo]`
  - `identify(path: Path, catalog: Catalog) -> ArchiveIdentity`
  - `read_member(zf: zipfile.ZipFile, member_path: str) -> bytes`
  - `extract_to(zf: zipfile.ZipFile, member: MemberInfo, staging_dir: Path) -> Path`
  - `discard_staged(staged: Path) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_archive.py`:

```python
"""Tests for Takeout archive identity, enumeration, and extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.takeout import archive


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


# --- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "member, expected",
    [
        ("Takeout/AlbumArchive/Hangouts/a/2015-03-09.jpg", archive.KIND_IMAGE),
        ("Takeout/x/PHOTO.JPG", archive.KIND_IMAGE),
        ("Takeout/x/clip.mp4", archive.KIND_VIDEO),
        ("Takeout/x/clip.MOV", archive.KIND_VIDEO),
        ("Takeout/x/2015-03-09.jpg.json", archive.KIND_METADATA),
        ("Takeout/x/a.jpg.supplemental-metadata.json", archive.KIND_METADATA),
        ("Takeout/x/Albums.json", archive.KIND_ALBUM),
        ("Takeout/x/metadata.json", archive.KIND_ALBUM),
        ("Takeout/x/archive_browser.html", archive.KIND_OTHER),
        ("Takeout/x/notes.txt", archive.KIND_OTHER),
        ("Takeout/x/noextension", archive.KIND_OTHER),
    ],
)
def test_classify(member, expected) -> None:
    assert archive.classify(member) == expected


def test_classification_is_service_agnostic() -> None:
    """The real export is AlbumArchive, not 'Google Photos'. Never key on a path."""
    assert archive.classify("Takeout/AlbumArchive/Hangouts/x/a.jpg") == archive.KIND_IMAGE
    assert archive.classify("Takeout/Google Photos/2015/a.jpg") == archive.KIND_IMAGE
    assert archive.classify("some/unheard/of/service/a.jpg") == archive.KIND_IMAGE


@pytest.mark.parametrize(
    "member, expected",
    [
        ("Takeout/Google Photos/Trash/a.jpg", True),
        ("Takeout/Google Photos/trash/a.jpg", True),
        ("Trash/a.jpg", True),
        ("Takeout/Google Photos/Trashy Album/a.jpg", False),
        ("Takeout/Google Photos/2015/a.jpg", False),
    ],
)
def test_is_trash(member, expected) -> None:
    assert archive.is_trash(member) is expected


# --- enumeration -----------------------------------------------------------


def test_iter_members_reads_only_the_central_directory(tmp_path: Path) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa", "d/a.jpg.json": b"{}", "d/": b""})
    with zipfile.ZipFile(z, "r") as zf:
        members = list(archive.iter_members(zf))
    paths = {m.path for m in members}
    assert paths == {"d/a.jpg", "d/a.jpg.json"}   # the directory entry is skipped
    by_path = {m.path: m for m in members}
    assert by_path["d/a.jpg"].size == 3
    assert by_path["d/a.jpg"].kind == archive.KIND_IMAGE
    assert by_path["d/a.jpg"].crc32 != 0


def test_iter_members_does_not_decompress(tmp_path: Path, monkeypatch) -> None:
    """Enumeration must be central-directory only, even on a huge archive."""
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    with zipfile.ZipFile(z, "r") as zf:
        def _boom(*args, **kwargs):
            raise AssertionError("iter_members must not open a member")

        monkeypatch.setattr(zf, "open", _boom)
        assert len(list(archive.iter_members(zf))) == 1


# --- identity --------------------------------------------------------------


def test_identify_hashes_on_a_miss(tmp_path: Path, catalog: Catalog) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    identity = archive.identify(z, catalog)
    assert len(identity.archive_id) == 43
    assert identity.size == z.stat().st_size


def test_identify_uses_the_stat_fast_path(tmp_path: Path, catalog: Catalog, monkeypatch) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    identity = archive.identify(z, catalog)
    catalog.takeout_archive_upsert(
        archive_id=identity.archive_id,
        last_path=str(z),
        size=identity.size,
        mtime_ns=identity.mtime_ns,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("the fast path must not re-hash the archive")

    monkeypatch.setattr(archive, "compute_sha256_b64url", _boom)
    again = archive.identify(z, catalog)
    assert again.archive_id == identity.archive_id


def test_a_renamed_archive_resolves_to_the_same_id(tmp_path: Path, catalog: Catalog) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    first = archive.identify(z, catalog)
    catalog.takeout_archive_upsert(
        archive_id=first.archive_id, last_path=str(z), size=first.size,
        mtime_ns=first.mtime_ns,
    )
    renamed = tmp_path / "renamed.zip"
    z.rename(renamed)
    assert archive.identify(renamed, catalog).archive_id == first.archive_id


# --- extraction ------------------------------------------------------------


def test_extract_to_preserves_the_member_basename(tmp_path: Path) -> None:
    """Downstream date/descriptor resolution reads the staged file's NAME."""
    z = _zip(tmp_path / "t.zip", {"d/2015-03-09.jpg": b"bytes"})
    staging = tmp_path / "staging"
    with zipfile.ZipFile(z, "r") as zf:
        member = next(archive.iter_members(zf))
        staged = archive.extract_to(zf, member, staging)
        assert staged.name == "2015-03-09.jpg"
        assert staged.read_bytes() == b"bytes"
    archive.discard_staged(staged)
    assert not staged.exists()


def test_extract_to_isolates_colliding_basenames(tmp_path: Path) -> None:
    z = _zip(tmp_path / "t.zip", {"a/x.jpg": b"one", "b/x.jpg": b"two"})
    staging = tmp_path / "staging"
    with zipfile.ZipFile(z, "r") as zf:
        members = list(archive.iter_members(zf))
        first = archive.extract_to(zf, members[0], staging)
        second = archive.extract_to(zf, members[1], staging)
        assert first != second
        assert {first.read_bytes(), second.read_bytes()} == {b"one", b"two"}


def test_a_corrupted_member_raises_rather_than_yielding_bad_bytes(tmp_path: Path) -> None:
    """zipfile verifies CRC on a full read; a bad member must fail loudly."""
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"a" * 200})
    raw = bytearray(z.read_bytes())
    # Corrupt the compressed payload without touching the central directory.
    raw[60:70] = b"\x00" * 10
    z.write_bytes(bytes(raw))

    with zipfile.ZipFile(z, "r") as zf:
        member = next(archive.iter_members(zf))
        with pytest.raises(Exception):
            archive.extract_to(zf, member, tmp_path / "staging")


def test_read_member_returns_bytes(tmp_path: Path) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.json": b'{"title": "x"}'})
    with zipfile.ZipFile(z, "r") as zf:
        assert archive.read_member(zf, "d/a.json") == b'{"title": "x"}'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_takeout_archive.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.takeout.archive'`.

- [ ] **Step 3: Add `VIDEO_EXTENSIONS` to `discovery.py`**

In `imageharbor/discovery.py`, immediately after the `SUPPORTED_EXTENSIONS`
definition, add:

```python
# Video extensions, for CLASSIFICATION ONLY. `discover_images` still yields
# images and nothing else -- video ingestion is a separate, later project.
# Takeout ingestion enumerates videos and records them as `deferred` with
# their capture date, so that project starts from a complete work queue rather
# than from zero, but no video bytes are ever copied.
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".m4v",
        ".3gp",
        ".avi",
        ".mkv",
        ".webm",
    }
)
```

- [ ] **Step 4: Write `archive.py`**

Create `imageharbor/takeout/archive.py`:

```python
"""Archive identity, enumeration, classification, and member extraction.

Archives are opened ``'r'`` only. Nothing here writes into, alongside, or in
place of an archive -- the zip IS the original, and originals are read-only.

Enumeration reads only the central directory, so surveying a 60 GB export
costs a seek, not a decompression pass. Extraction is per member, on demand,
into a staging directory the caller owns.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ..discovery import SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS
from ..hashing import compute_sha256_b64url

if TYPE_CHECKING:
    from ..catalog import Catalog

logger = logging.getLogger(__name__)

KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_METADATA = "metadata"
KIND_ALBUM = "album"
KIND_OTHER = "other"

# Album descriptors. AlbumArchive exports use Albums.json; Google Photos
# exports use metadata.json. Both are accepted.
_ALBUM_BASENAMES = frozenset({"albums.json", "metadata.json"})

# A path component named exactly "trash" (any case). "Trashy Album" is not a
# trash tree, so an endswith/contains test would be wrong here.
_TRASH_COMPONENT = "trash"

# Characters a Windows filesystem refuses. Member names in the real export
# carry Unicode and shell-hostile characters (● U+25CF, +, =, spaces,
# parentheses) -- all of which are legal on every supported filesystem and
# must survive untouched, because the staged file's NAME is evidence the date
# and descriptor resolvers read. Only genuinely illegal characters are
# replaced.
_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


@dataclass(frozen=True)
class MemberInfo:
    """One entry from an archive's central directory."""

    path: str    # member path inside the zip, verbatim
    size: int    # uncompressed size
    crc32: int   # stored for diagnostics; NEVER the sole basis for a skip
    kind: str


@dataclass(frozen=True)
class ArchiveIdentity:
    """Which archive this is, and where it was found."""

    archive_id: str   # SHA-256 b64url of the .zip's own bytes
    path: Path
    size: int
    mtime_ns: int


def classify(member_path: str) -> str:
    """Classify a member by extension and basename alone.

    Deliberately service-agnostic: the real export is ``Takeout/AlbumArchive/
    Hangouts/<album>/`` with no ``Google Photos/`` directory anywhere in it, so
    keying on a service path would classify nothing. ``discovery`` is the
    single source of truth for what counts as an image.
    """
    name = member_path.rpartition("/")[2]
    lower = name.lower()
    if lower.endswith(".json"):
        return KIND_ALBUM if lower in _ALBUM_BASENAMES else KIND_METADATA
    _, dot, ext = lower.rpartition(".")
    suffix = f".{ext}" if dot else ""
    if suffix in SUPPORTED_EXTENSIONS:
        return KIND_IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return KIND_VIDEO
    return KIND_OTHER


def is_trash(member_path: str) -> bool:
    """True if *member_path* lives under a Trash tree."""
    return any(
        part.lower() == _TRASH_COMPONENT for part in member_path.split("/")[:-1]
    )


def iter_members(zf: zipfile.ZipFile) -> Iterator[MemberInfo]:
    """Yield every file member of *zf*. Reads the central directory only."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        yield MemberInfo(
            path=info.filename,
            size=info.file_size,
            crc32=info.CRC,
            kind=classify(info.filename),
        )


def identify(path: Path, catalog: "Catalog") -> ArchiveIdentity:
    """Identify the archive at *path*, hashing it only when necessary.

    The ``(path, size, mtime_ns)`` fast path avoids re-hashing an archive we
    have already hashed at exactly that location and stat. It is never treated
    as proof of content: any change to any of the three falls through to the
    digest, which is what actually keys the archive.
    """
    stat = path.stat()
    row = catalog.takeout_archive_get_by_stat(str(path), stat.st_size, stat.st_mtime_ns)
    if row is not None:
        return ArchiveIdentity(
            archive_id=row["archive_id"],
            path=path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    logger.debug("Hashing archive %s (%d bytes)", path.name, stat.st_size)
    return ArchiveIdentity(
        archive_id=compute_sha256_b64url(path),
        path=path,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def read_member(zf: zipfile.ZipFile, member_path: str) -> bytes:
    """Return one member's bytes. Used for small JSON sidecars only."""
    with zf.open(member_path, "r") as fh:
        return fh.read()


def _safe_name(name: str) -> str:
    cleaned = _ILLEGAL_NAME_CHARS.sub("_", name).rstrip(" .")
    return cleaned or "member"


def extract_to(zf: zipfile.ZipFile, member: MemberInfo, staging_dir: Path) -> Path:
    """Stream *member* to a staging file and return its path.

    The staged file keeps the member's BASENAME, because that name is evidence:
    ``date_resolver``'s filename rung and ``descriptor``'s camera-pattern table
    both read it. Each member gets its own directory under *staging_dir*, so two
    members with the same basename in different archive directories cannot
    collide.

    ``zipfile`` verifies CRC32 on a full read, so a corrupted member raises here
    rather than yielding bad bytes into the library.

    The caller owns cleanup: pass the returned path to :func:`discard_staged`
    in a ``finally``.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    holder = Path(tempfile.mkdtemp(dir=str(staging_dir)))
    dest = holder / _safe_name(member.path.rpartition("/")[2])
    with zf.open(member.path, "r") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 65536)
    return dest


def discard_staged(staged: Path) -> None:
    """Remove a staged file and the private directory holding it.

    A leftover staging file after a kill is inert debris, not state: phase 2
    resumes from `takeout_members`, never from what is on the staging floor.
    """
    shutil.rmtree(staged.parent, ignore_errors=True)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_takeout_archive.py -q`
Expected: PASS.

If `test_a_corrupted_member_raises_rather_than_yielding_bad_bytes` does not
raise, the byte range being corrupted landed outside the compressed payload —
adjust the slice so it falls inside the local file data (print
`zf.getinfo("d/a.jpg").header_offset` to find the payload start) and re-run.
Do not weaken the assertion.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/discovery.py imageharbor/takeout/archive.py tests/test_takeout_archive.py
git commit -m "feat: Takeout archive identity, enumeration and extraction"
```

---

### Task 7: Pipeline — `ExternalEvidence`, `source_label`, `consume_source`

Three additions to the facts pass, all keyword-only and all defaulting to
today's behavior, so `process`, `watch`, and `watcher.py` are byte-for-byte
unchanged when the arguments are omitted.

**Files:**
- Modify: `imageharbor/pipeline.py`
- Test: `tests/test_pipeline.py`, `tests/test_monotonicity.py`

**Interfaces:**
- Consumes: `resolve_descriptor(..., original_name=, date_str=)` (Task 1), `resolve_date(..., external_date=)` (Task 2).
- Produces:
  - `ExternalEvidence(date: datetime | None = None, original_name: str | None = None)` frozen dataclass, defined in `pipeline.py` beside `ProcessResult`
  - `Pipeline(..., consume_source: bool = False)`
  - `Pipeline.process_file(image_path: Path, *, source_label: str | None = None, evidence: ExternalEvidence | None = None) -> ProcessResult`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`:

```python
from datetime import datetime

from imageharbor.pipeline import ExternalEvidence


def test_consume_source_moves_instead_of_copying(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = _make_jpeg(staging / "beach.jpg")

    pipeline = Pipeline(staging, organized_dir, catalog, consume_source=True)
    result = pipeline.process_file(staged)

    assert result.status == "copied"
    assert not staged.exists()                      # consumed
    assert result.organized_path.exists()
    assert verify_pcs_file(result.organized_path)   # verified AFTER the move


def test_consume_source_defaults_to_copying(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    original = source_dir / "beach_photo.jpg"
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    result = pipeline.process_file(original)

    assert original.exists()                        # untouched
    assert result.organized_path.exists()


def test_source_label_is_what_gets_recorded(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The staging path is disposable; the logical source is the archive member."""
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = _make_jpeg(staging / "beach.jpg")
    label = "/nas/takeout/t1.zip!Takeout/AlbumArchive/a/beach.jpg"

    pipeline = Pipeline(staging, organized_dir, catalog)
    result = pipeline.process_file(staged, source_label=label)

    row = catalog.get_by_sha256(result.sha256_b64url)
    assert row["original_path"] == label
    assert [r["source_path"] for r in catalog.sources_for(result.sha256_b64url)] == [label]


def test_evidence_date_places_the_file(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    staged = _make_jpeg(tmp_path / "IMG_1234.jpg")
    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    result = pipeline.process_file(
        staged, evidence=ExternalEvidence(date=datetime(2015, 3, 9, 12, 56, 32))
    )

    assert result.organized_path.parent == organized_dir / "2015" / "2015-03"
    row = catalog.get_by_sha256(result.sha256_b64url)
    assert row["date_tier"] == tiers.DATE_EXTERNAL_SIDECAR
    assert row["date_source"] == "external_sidecar"


def test_evidence_original_name_names_the_file(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    staged = _make_jpeg(tmp_path / "truncated-stem.jpg")
    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    result = pipeline.process_file(
        staged, evidence=ExternalEvidence(original_name="emma birthday party.jpg")
    )

    assert "emma-birthday-party" in result.organized_path.name
    row = catalog.get_by_sha256(result.sha256_b64url)
    assert row["descriptor_tier"] == tiers.DESC_HUMAN_FILENAME


def test_evidence_none_leaves_behavior_unchanged(
    source_dir: Path, organized_dir: Path, catalog: Catalog
) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    with_none = pipeline.process_file(source_dir / "beach_photo.jpg", evidence=None)
    assert with_none.organized_path.parent == organized_dir / "Undated"
```

Add to `tests/test_monotonicity.py`:

```python
def test_late_evidence_upgrades_a_duplicate_out_of_undated(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The late-arriving-sidecar case, at the pipeline level.

    Part 1 ingests with no Google date -> Undated/. Part 2 arrives carrying the
    sidecar, the bytes hash as a duplicate, and the EXISTING monotonic upgrade
    machinery relocates the file. No new code path is needed for this --
    only that `_maybe_upgrade_from_duplicate` is given the evidence.
    """
    from datetime import datetime

    from imageharbor.pipeline import ExternalEvidence, Pipeline

    staged = tmp_path / "IMG_1234.jpg"
    staged.write_bytes(b"\xff\xd8\xff\xe0" + b"\x07" * 16 + b"\xff\xd9")

    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    first = pipeline.process_file(staged)
    assert first.organized_path.parent == organized_dir / "Undated"

    second = pipeline.process_file(
        staged, evidence=ExternalEvidence(date=datetime(2015, 3, 9, 12, 56, 32))
    )
    assert second.status == "duplicate"

    row = catalog.get_by_sha256(first.sha256_b64url)
    assert Path(row["organized_path"]).parent == organized_dir / "2015" / "2015-03"
    assert Path(row["organized_path"]).exists()
    assert not first.organized_path.exists()


def test_re_ingesting_the_same_evidence_is_a_rename_no_op(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    from datetime import datetime

    from imageharbor.pipeline import ExternalEvidence, Pipeline

    staged = tmp_path / "IMG_1234.jpg"
    staged.write_bytes(b"\xff\xd8\xff\xe0" + b"\x08" * 16 + b"\xff\xd9")
    evidence = ExternalEvidence(date=datetime(2015, 3, 9))

    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    first = pipeline.process_file(staged, evidence=evidence)
    before = catalog.get_by_sha256(first.sha256_b64url)["organized_path"]

    pipeline.process_file(staged, evidence=evidence)
    after = catalog.get_by_sha256(first.sha256_b64url)["organized_path"]

    assert before == after
    assert Path(after).exists()
```

`tests/test_monotonicity.py` may not already define `organized_dir`/`catalog`
fixtures — if it does not, copy them from `tests/test_pipeline.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py tests/test_monotonicity.py -q`
Expected: FAIL — `ImportError: cannot import name 'ExternalEvidence'`.

- [ ] **Step 3: Add `ExternalEvidence` and the `os` import**

In `imageharbor/pipeline.py`, add `import os` next to `import shutil`, add
`from datetime import datetime` to the imports, and add this dataclass
immediately after `ProcessResult`:

```python
@dataclass(frozen=True)
class ExternalEvidence:
    """Facts about an image that are not in its bytes or its current path.

    The parameter object for evidence a caller obtained elsewhere -- in
    practice Google Takeout's per-media JSON. ``Pipeline`` unpacks it into the
    two resolvers rather than passing it down, so neither resolver learns
    anything about Takeout.

    Google's ``creationTime`` must NEVER be placed in ``date``: it records when
    a file was uploaded, not when the photo was taken.
    """

    date: datetime | None = None           # e.g. Google photoTakenTime
    original_name: str | None = None       # e.g. Google `title`, pre-truncation
```

- [ ] **Step 4: Add `consume_source` to `__init__`**

In `Pipeline.__init__`, add the parameter after `dry_run` and store it:

```python
    def __init__(
        self,
        source_dir: Path,
        organized_dir: Path,
        catalog: Catalog,
        duplicates_dir: Path | None = None,
        write_sidecars: bool = False,
        dry_run: bool = False,
        consume_source: bool = False,
    ) -> None:
        ...
        self.dry_run = dry_run
        # When True the "source" is a disposable staging file this process
        # created and owns, so the copy step MOVES instead of copying -- half
        # the write I/O, which is material at 60 GB per export over a NAS
        # mount. The guarded invariant is untouched: the original is the zip,
        # which is never opened for writing; a staging file is not an original.
        # `process` and `watch` never set this.
        self.consume_source = consume_source
```

Add to the class docstring's parameter list:

```
    consume_source:
        When True the source file is MOVED into the organized tree rather than
        copied, because the caller created it as disposable staging. Ordering
        becomes rename -> verify -> catalog; verification still reads the file
        at its destination, so nothing enters the catalog unverified.
```

- [ ] **Step 5: Thread `source_label` and `evidence` through**

Replace `process_file`, `_process_one`, and the head of `_do_process`:

```python
    def process_file(
        self,
        image_path: Path,
        *,
        source_label: str | None = None,
        evidence: ExternalEvidence | None = None,
    ) -> ProcessResult:
        """Process a single image file and return its result.

        *source_label* is the LOGICAL source recorded in `sources` and
        `photos.original_path`, when that differs from where the bytes
        currently sit -- e.g. ``/nas/t1.zip!Takeout/.../2015-03-09.jpg`` for a
        member staged out of an archive. It is stable across runs and across
        machines that mount the archive at the same path, so the back-pointer
        set stays meaningful after the staging file is gone.

        *evidence* supplies facts from outside the file's bytes and path; see
        :class:`ExternalEvidence`.
        """
        result = self._process_one(image_path, source_label=source_label, evidence=evidence)
        _log_result(result)
        return result

    def _process_one(
        self,
        source_path: Path,
        *,
        source_label: str | None = None,
        evidence: ExternalEvidence | None = None,
    ) -> ProcessResult:
        try:
            return self._do_process(source_path, source_label=source_label, evidence=evidence)
        except Exception as exc:
            logger.exception("Unexpected error processing %s", source_path)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url="",
                status="error",
                error=str(exc),
            )

    def _do_process(
        self,
        source_path: Path,
        *,
        source_label: str | None = None,
        evidence: ExternalEvidence | None = None,
    ) -> ProcessResult:
        # The logical identity of these bytes, which may outlive the path they
        # currently sit at.
        label = source_label or str(source_path)

        # Step 1: hash original
        sha256_b64url = compute_sha256_b64url(source_path)
        stat = source_path.stat()
```

In the duplicate branch of `_do_process`, replace the three `str(source_path)`
identity uses with `label` and pass the evidence through:

```python
            if not self.dry_run:
                self.catalog.mark_duplicate(sha256_b64url, label)
                self.catalog.record_source(
                    sha256_b64url, label, stat.st_size, stat.st_mtime_ns
                )
                self._maybe_upgrade_from_duplicate(
                    source_path, sha256_b64url, evidence=evidence
                )
```

Replace the facts step:

```python
        # Step 4: facts -- date decides the folder, descriptor decides the name.
        date = resolve_date(
            source_path, exif_data, external_date=evidence.date if evidence else None
        )
        descriptor = resolve_descriptor(
            source_path,
            original_name=evidence.original_name if evidence else None,
            date_str=date.date_str,
        )
```

Replace the copy step (Step 6/7):

```python
        # Step 6: copy (or MOVE, when the source is disposable staging)
        organized_path.parent.mkdir(parents=True, exist_ok=True)
        if organized_path.exists() and verify_file(organized_path, sha256_b64url):
            logger.debug(
                "Destination already present and verified, skipping copy: %s",
                organized_path,
            )
            if self.consume_source:
                source_path.unlink(missing_ok=True)
        else:
            if self.consume_source:
                os.replace(str(source_path), str(organized_path))
            else:
                shutil.copy2(str(source_path), str(organized_path))

            # Step 7: verify before anything is recorded. This reads the file
            # at its DESTINATION either way, so the move path is verified
            # exactly as strictly as the copy path.
            if not verify_file(organized_path, sha256_b64url):
                organized_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Integrity check failed after copying {source_path} -> {organized_path}"
                )
```

Replace the catalog step's identity fields:

```python
        self.catalog.upsert(
            sha256_b64url=sha256_b64url,
            original_path=label,
            organized_path=str(organized_path),
            ...
            processing_history=[
                {
                    "event": "facts",
                    "source": label,
                    "destination": str(organized_path),
                }
            ],
        )
        self.catalog.record_source(
            sha256_b64url, label, stat.st_size, stat.st_mtime_ns
        )
```

- [ ] **Step 6: Thread evidence into the duplicate-upgrade path**

This is the single most important integration point in the design: it is what
makes a late-arriving sidecar relocate a photo out of `Undated/` with no new
code path. Change the signature and the two resolver calls in
`_maybe_upgrade_from_duplicate`:

```python
    def _maybe_upgrade_from_duplicate(
        self,
        source_path: Path,
        sha256_b64url: str,
        *,
        evidence: ExternalEvidence | None = None,
    ) -> None:
        """Re-evaluate a known file's tiers against a newly-seen source path.

        Identical bytes mean identical EXIF, but not identical filenames: the
        same photo found at a better-named path can supply a date or a
        descriptor the first copy lacked.  *evidence* is the same channel for
        facts that live outside the file entirely -- a Takeout sidecar that
        only arrived in a later archive part.
        """
        row = self.catalog.get_by_sha256(sha256_b64url)
        if row is None or not row["organized_path"]:
            return

        date = resolve_date(
            source_path, {}, external_date=evidence.date if evidence else None
        )
        descriptor = resolve_descriptor(
            source_path,
            original_name=evidence.original_name if evidence else None,
            date_str=date.date_str,
        )
```

The rest of the method is unchanged — `tiers.is_upgrade` still gates the rename,
so this can only ever improve a file.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_pipeline.py tests/test_monotonicity.py -q`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS, all green — including `tests/test_watcher.py`, which calls
`process_file(path)` positionally and must be unaffected.

- [ ] **Step 8: Commit**

```bash
git add imageharbor/pipeline.py tests/test_pipeline.py tests/test_monotonicity.py
git commit -m "feat: external evidence, source labels and consumable sources in the facts pass"
```

---

### Task 8: `takeout/ingest.py` — the two-phase orchestrator

**Files:**
- Create: `imageharbor/takeout/ingest.py`
- Modify: `imageharbor/takeout/__init__.py` (add the re-exports)
- Test: `tests/test_takeout_ingest.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces:
  - `IngestStats` dataclass with int fields `archives_seen`, `archives_skipped`, `archives_corrupt`, `ingested`, `duplicates`, `deferred`, `skipped_trash`, `failed`, `missing_metadata`, and a `per_archive: list[dict]` for CLI reporting
  - `ingest_archives(archives_dir: Path, organized_dir: Path, catalog: Catalog, *, include_trash: bool = False, write_sidecars: bool = False, dry_run: bool = False) -> IngestStats`
  - `STAGING_DIR_NAME = ".takeout-staging"`

**Two design decisions to implement, both departures from a naive reading of the spec:**

1. **Complete archives still contribute to the pairing index.** The spec's phase-1 pseudocode skips a complete archive "entirely". Doing that literally opens a hole: if part 2 (all sidecars → all members `parsed` → archive `complete`) is ingested before part 1 arrives, part 1's photos could never see part 2's sidecars. Instead, a complete archive's member paths are loaded from `takeout_members` — no zip is opened and nothing is decompressed, so it stays as cheap as skipping.
2. **`--dry-run` writes nothing to the catalog.** The survey runs against the real catalog for reads (so "already complete" is reported honestly), but every write and every extraction is suppressed. This is more useful than the `process --dry-run` in-memory-catalog trick and keeps the report accurate.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_ingest.py`:

```python
"""Behavioral tests for Takeout ingestion.

Synthetic zips built in tmp_path replicate the real export's name shapes. No
79 MB fixture is committed -- the shapes are what actually matter.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.takeout import archive as archive_mod
from imageharbor.takeout.ingest import ingest_archives

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"
D = "Takeout/AlbumArchive/Hangouts/album"


def _jpeg(n: int) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes([n]) * 16 + b"\xff\xd9"


def _sidecar(title: str, seconds: int) -> bytes:
    return json.dumps(
        {
            "title": title,
            "creationTime": {"timestampSeconds": str(seconds + 14836)},
            "photoTakenTime": {"timestampSeconds": str(seconds)},
            "geoData": {"latitude": 38.2768361, "longitude": -85.7357389},
        }
    ).encode()


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


@pytest.fixture()
def dirs(tmp_path: Path):
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    dest.mkdir()
    return archives, dest


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


# --- the happy path --------------------------------------------------------


def test_ingests_a_photo_with_its_sidecar(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(
        archives / "takeout-001.zip",
        {
            f"{D}/2015-03-09.jpg": _jpeg(1),
            f"{D}/2015-03-09.jpg.json": _sidecar("2015-03-09.jpg", 1425905792),
        },
    )
    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.failed == 0
    assert stats.missing_metadata == 0
    organized = list((dest / "2015" / "2015-03").glob("*.jpg"))
    assert len(organized) == 1


def test_a_photo_without_a_sidecar_is_still_fully_organized(dirs, catalog: Catalog) -> None:
    """No Google metadata is not a failure. It is a photo dated from itself."""
    archives, dest = dirs
    _zip(archives / "t.zip", {f"{D}/IMG_9999.jpg": _jpeg(2)})
    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.missing_metadata == 1
    assert len(list((dest / "Undated").glob("*.jpg"))) == 1


# --- idempotency and resume ------------------------------------------------


def test_a_second_run_extracts_zero_members(dirs, catalog: Catalog, monkeypatch) -> None:
    """Asserted by counting extractions, not by timing."""
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            f"{D}/a.jpg": _jpeg(3),
            f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
            f"{D}/b.jpg": _jpeg(4),
        },
    )
    ingest_archives(archives, dest, catalog)

    calls = []
    real = archive_mod.extract_to
    monkeypatch.setattr(
        archive_mod, "extract_to",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    stats = ingest_archives(archives, dest, catalog)

    assert calls == []
    assert stats.archives_skipped == 1
    assert stats.ingested == 0


def test_resume_after_a_mid_archive_crash_processes_only_the_remainder(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {f"{D}/a.jpg": _jpeg(5), f"{D}/b.jpg": _jpeg(6), f"{D}/c.jpg": _jpeg(7)},
    )

    real = archive_mod.extract_to
    seen: list[str] = []

    def _crash_on_third(zf, member, staging):
        seen.append(member.path)
        if len(seen) == 3:
            raise KeyboardInterrupt("simulated kill -9")
        return real(zf, member, staging)

    monkeypatch.setattr(archive_mod, "extract_to", _crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        ingest_archives(archives, dest, catalog)

    monkeypatch.setattr(archive_mod, "extract_to", real)
    extracted: list[str] = []
    monkeypatch.setattr(
        archive_mod, "extract_to",
        lambda zf, m, s: (extracted.append(m.path), real(zf, m, s))[1],
    )
    stats = ingest_archives(archives, dest, catalog)

    assert extracted == [f"{D}/c.jpg"]
    assert stats.ingested == 1


def test_a_corrupt_archive_does_not_stop_its_neighbours(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    (archives / "broken.zip").write_bytes(b"this is not a zip file")
    _zip(archives / "good.zip", {f"{D}/a.jpg": _jpeg(8)})

    stats = ingest_archives(archives, dest, catalog)

    assert stats.archives_corrupt == 1
    assert stats.ingested == 1


# --- videos, trash, duplicates ---------------------------------------------


def test_videos_are_deferred_with_a_date_and_no_bytes_copied(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            f"{D}/clip.mp4": b"not really an mp4",
            f"{D}/clip.mp4.json": _sidecar("clip.mp4", 1425905792),
        },
    )
    stats = ingest_archives(archives, dest, catalog)

    assert stats.deferred == 1
    assert stats.ingested == 0
    assert list(dest.rglob("*.mp4")) == []

    identity = catalog.takeout_archives_all()[0]["archive_id"]
    row = [m for m in catalog.takeout_members_all(identity) if m["kind"] == "video"][0]
    assert row["status"] == "deferred"
    assert row["taken_at"].startswith("2015-03-09")


def test_trash_is_enumerated_but_not_ingested(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {"Takeout/Google Photos/Trash/deleted.jpg": _jpeg(9), f"{D}/kept.jpg": _jpeg(10)},
    )
    stats = ingest_archives(archives, dest, catalog)

    assert stats.skipped_trash == 1
    assert stats.ingested == 1
    identity = catalog.takeout_archives_all()[0]["archive_id"]
    statuses = {m["member_path"]: m["status"] for m in catalog.takeout_members_all(identity)}
    assert statuses["Takeout/Google Photos/Trash/deleted.jpg"] == "skipped_trash"


def test_include_trash_ingests_previously_skipped_trash(dirs, catalog: Catalog) -> None:
    """A user who changes their mind must not be blocked by a terminal status."""
    archives, dest = dirs
    _zip(archives / "t.zip", {"Takeout/Google Photos/Trash/deleted.jpg": _jpeg(11)})

    ingest_archives(archives, dest, catalog)
    assert list(dest.rglob("*.jpg")) == []

    stats = ingest_archives(archives, dest, catalog, include_trash=True)
    assert stats.ingested == 1
    assert len(list(dest.rglob("*.jpg"))) == 1


def test_the_same_photo_in_two_archives_yields_one_file_and_two_sources(
    dirs, catalog: Catalog
) -> None:
    archives, dest = dirs
    _zip(archives / "t1.zip", {f"{D}/a.jpg": _jpeg(12)})
    _zip(archives / "t2.zip", {f"{D}/copy-of-a.jpg": _jpeg(12)})

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.duplicates == 1
    assert len(list(dest.rglob("*.jpg"))) == 1
    # catalog.iter_all() returns an iterator, not a list.
    row = next(iter(catalog.iter_all()))
    assert len(catalog.sources_for(row["sha256_b64url"])) == 2
    assert all("!" in r["source_path"] for r in catalog.sources_for(row["sha256_b64url"]))


# --- the late-sidecar case: the heart of the design ------------------------


def test_a_sidecar_arriving_in_a_later_part_relocates_the_photo(
    dirs, catalog: Catalog
) -> None:
    """Google splits by size, so a photo and its sidecar land in different parts.

    Ingest part 1 alone -> Undated/. Add part 2 carrying the sidecar, re-run,
    and the existing monotonic-upgrade machinery relocates the file.
    """
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {f"{D}/IMG_1234.jpg": _jpeg(13)})

    ingest_archives(archives, dest, catalog)
    assert len(list((dest / "Undated").glob("*.jpg"))) == 1

    _zip(
        archives / "takeout-002.zip",
        {f"{D}/IMG_1234.jpg.json": _sidecar("IMG_1234.jpg", 1425905792)},
    )
    ingest_archives(archives, dest, catalog)

    assert list((dest / "Undated").glob("*.jpg")) == []
    assert len(list((dest / "2015" / "2015-03").glob("*.jpg"))) == 1
```

Note on `test_the_same_photo_in_two_archives_yields_one_file_and_two_sources`:
drop the dead `digest = ...` line when implementing; it is left here only to
flag that `catalog.iter_all()` returns an iterator, so use `next(iter(...))`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_takeout_ingest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.takeout.ingest'`.

- [ ] **Step 3: Write `ingest.py`**

Create `imageharbor/takeout/ingest.py`:

```python
"""Two-phase Google Takeout ingestion.

The only module in this package with side effects. Its shape mirrors
``enrich.enrich_library``: iterate a work queue held in the catalog, do one
unit of work, commit the outcome, continue.

Phase 1 surveys every archive by reading central directories only, and builds
ONE pairing index across the whole batch. Phase 2 ingests member by member,
committing after each, so a ``kill -9`` costs one member's work.

This pass makes no AI calls, exactly like the facts pass, so there is nothing
for a circuit breaker to observe and it must not touch one.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..catalog import Catalog
from ..pipeline import ExternalEvidence, Pipeline
from ..sidecar import merge_sidecar
from . import archive, metadata, pairing

logger = logging.getLogger(__name__)

STAGING_DIR_NAME = ".takeout-staging"

# Member statuses.
_PENDING = "pending"
_INGESTED = "ingested"
_DUPLICATE = "duplicate"
_DEFERRED = "deferred"
_PARSED = "parsed"
_IGNORED = "ignored"
_SKIPPED_TRASH = "skipped_trash"
_FAILED = "failed"


@dataclass
class IngestStats:
    """Aggregated outcome of one ingest run."""

    archives_seen: int = 0
    archives_skipped: int = 0     # already complete
    archives_corrupt: int = 0
    ingested: int = 0
    duplicates: int = 0
    deferred: int = 0             # videos: enumerated and dated, never copied
    skipped_trash: int = 0
    failed: int = 0
    missing_metadata: int = 0     # ingested, but no sidecar could be paired
    per_archive: list[dict] = field(default_factory=list)


def _initial_status(member: archive.MemberInfo, include_trash: bool) -> str:
    if archive.is_trash(member.path) and not include_trash:
        return _SKIPPED_TRASH
    if member.kind in (archive.KIND_IMAGE, archive.KIND_VIDEO):
        return _PENDING
    if member.kind in (archive.KIND_METADATA, archive.KIND_ALBUM):
        return _PARSED    # terminal: read on demand, never a work item
    return _IGNORED


def ingest_archives(
    archives_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    *,
    include_trash: bool = False,
    write_sidecars: bool = False,
    dry_run: bool = False,
) -> IngestStats:
    """Ingest every ``*.zip`` under *archives_dir* into *organized_dir*."""
    return _Ingestor(
        archives_dir=archives_dir,
        organized_dir=organized_dir,
        catalog=catalog,
        include_trash=include_trash,
        write_sidecars=write_sidecars,
        dry_run=dry_run,
    ).run()


class _Ingestor:
    """Holds the state the two phases share."""

    def __init__(
        self,
        *,
        archives_dir: Path,
        organized_dir: Path,
        catalog: Catalog,
        include_trash: bool,
        write_sidecars: bool,
        dry_run: bool,
    ) -> None:
        self.archives_dir = archives_dir
        self.organized_dir = organized_dir
        self.catalog = catalog
        self.include_trash = include_trash
        self.write_sidecars = write_sidecars
        self.dry_run = dry_run
        self.staging_dir = organized_dir / STAGING_DIR_NAME
        self.stats = IngestStats()
        # Which archive holds each member, so a sidecar in another part can be
        # read without re-surveying.
        self.owner: dict[str, Path] = {}
        self.pairing_index = pairing.PairingIndex()
        self.pipeline = Pipeline(
            source_dir=self.staging_dir,
            organized_dir=organized_dir,
            catalog=catalog,
            write_sidecars=write_sidecars,
            consume_source=True,
        )

    # -- phase 1 ------------------------------------------------------------

    def _survey(self) -> list[tuple[archive.ArchiveIdentity, list[archive.MemberInfo]]]:
        """Enumerate every archive; return the ones with work left to do.

        The member lists come back with the identities so `--dry-run` can report
        from them directly: a dry run writes no `takeout_members` rows, so the
        catalog work queue would read empty and under-report.
        """
        todo: list[tuple[archive.ArchiveIdentity, list[archive.MemberInfo]]] = []
        all_members: list[str] = []

        for path in sorted(p for p in self.archives_dir.glob("*.zip") if p.is_file()):
            self.stats.archives_seen += 1
            identity = archive.identify(path, self.catalog)
            row = self.catalog.takeout_archive_get(identity.archive_id)

            if row is not None and row["status"] == "complete":
                reopened = False
                if self.include_trash and not self.dry_run:
                    # A user who changes their mind must not be blocked by the
                    # terminal status an earlier run recorded.
                    moved = self.catalog.takeout_members_unskip_trash(identity.archive_id)
                    if moved:
                        self.catalog.takeout_archive_set_status(
                            identity.archive_id, "partial"
                        )
                        reopened = True
                if not reopened:
                    # A complete archive still contributes its member paths to
                    # the pairing index. Skipping it outright would hide the
                    # sidecars of an archive ingested BEFORE the part holding
                    # the photos they describe -- the late-sidecar case, in
                    # reverse. Member paths come from the catalog, so no zip is
                    # opened and nothing is decompressed.
                    self.stats.archives_skipped += 1
                    for member in self.catalog.takeout_members_all(identity.archive_id):
                        all_members.append(member["member_path"])
                        self.owner[member["member_path"]] = path
                    continue

            try:
                with zipfile.ZipFile(path, "r") as zf:
                    members = list(archive.iter_members(zf))
            except (zipfile.BadZipFile, OSError) as exc:
                logger.error("Archive %s is unreadable: %s", path.name, exc)
                self.stats.archives_corrupt += 1
                if not self.dry_run:
                    self.catalog.takeout_archive_upsert(
                        archive_id=identity.archive_id,
                        last_path=str(path),
                        size=identity.size,
                        mtime_ns=identity.mtime_ns,
                        status="corrupt",
                        last_error=str(exc),
                    )
                continue

            if not self.dry_run:
                self.catalog.takeout_archive_upsert(
                    archive_id=identity.archive_id,
                    last_path=str(path),
                    size=identity.size,
                    mtime_ns=identity.mtime_ns,
                    member_count=len(members),
                    status="partial",
                )
                for member in members:
                    self.catalog.takeout_member_add(
                        archive_id=identity.archive_id,
                        member_path=member.path,
                        kind=member.kind,
                        size=member.size,
                        crc32=member.crc32,
                        status=_initial_status(member, self.include_trash),
                    )

            for member in members:
                all_members.append(member.path)
                self.owner[member.path] = path
                if _initial_status(member, self.include_trash) == _SKIPPED_TRASH:
                    self.stats.skipped_trash += 1

            todo.append((identity, members))

        # ONE index across every archive in the batch. Google's multi-part
        # zips split by size across the file list, so a photo and its sidecar
        # routinely land in different parts; a per-archive index would silently
        # lose metadata at every part boundary.
        self.pairing_index = pairing.build_index(all_members)
        return todo

    # -- phase 2 ------------------------------------------------------------

    def _read_sidecar(self, sidecar_member: str) -> metadata.TakeoutMetadata:
        owner = self.owner.get(sidecar_member)
        if owner is None:
            return metadata.EMPTY
        try:
            with zipfile.ZipFile(owner, "r") as zf:
                return metadata.parse_photo_metadata(archive.read_member(zf, sidecar_member))
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            logger.warning("Unreadable sidecar %s (%s); ingesting without it",
                           sidecar_member, exc)
            return metadata.EMPTY

    def _label(self, archive_path: Path, member_path: str) -> str:
        return f"{archive_path}!{member_path}"

    def _ingest_image(
        self,
        zf: zipfile.ZipFile,
        identity: archive.ArchiveIdentity,
        row,
    ) -> None:
        member_path = row["member_path"]
        member = archive.MemberInfo(
            path=member_path, size=row["size"], crc32=row["crc32"], kind=row["kind"]
        )
        sidecar_member = pairing.sidecar_for(member_path, self.pairing_index)
        meta = self._read_sidecar(sidecar_member) if sidecar_member else metadata.EMPTY

        staged = None
        try:
            staged = archive.extract_to(zf, member, self.staging_dir)
            result = self.pipeline.process_file(
                staged,
                source_label=self._label(identity.path, member_path),
                evidence=ExternalEvidence(
                    date=meta.photo_taken_at, original_name=meta.title
                ),
            )
        except Exception as exc:  # a member failure never fails the archive
            logger.warning("Failed to ingest %s: %s", member_path, exc, exc_info=True)
            self.stats.failed += 1
            self.catalog.takeout_member_set(
                identity.archive_id, member_path, status=_FAILED, last_error=str(exc)
            )
            return
        finally:
            if staged is not None:
                archive.discard_staged(staged)

        if result.status == "error":
            self.stats.failed += 1
            self.catalog.takeout_member_set(
                identity.archive_id, member_path, status=_FAILED, last_error=result.error
            )
            return

        status = _DUPLICATE if result.status == "duplicate" else _INGESTED
        if status == _DUPLICATE:
            self.stats.duplicates += 1
        else:
            self.stats.ingested += 1
        if sidecar_member is None:
            self.stats.missing_metadata += 1

        # The member is marked terminal only AFTER process_file returned a
        # non-error result, so the copy -> verify -> catalog ordering remains
        # the sole arbiter of truth and takeout_members can only lag it.
        self.catalog.takeout_member_set(
            identity.archive_id,
            member_path,
            status=status,
            sha256_b64url=result.sha256_b64url,
            taken_at=meta.photo_taken_at.isoformat() if meta.photo_taken_at else None,
            sidecar_path=sidecar_member,
        )

        if self.write_sidecars and result.organized_path is not None:
            self._merge_takeout_sidecar(result.organized_path, identity, member_path,
                                        sidecar_member, meta)

    def _merge_takeout_sidecar(
        self, organized_path: Path, identity, member_path: str,
        sidecar_member: str | None, meta: metadata.TakeoutMetadata,
    ) -> None:
        """Record Google's metadata as provenance. None of it is load-bearing.

        A sidecar failure must never fail an image that is already copied,
        verified, and catalogued.
        """
        try:
            merge_sidecar(
                organized_path,
                {
                    "takeout": {
                        "archive": identity.path.name,
                        "archive_id": identity.archive_id,
                        "member": member_path,
                        "sidecar": sidecar_member,
                        # Album membership, recorded not materialized: the
                        # containing directory IS the album in every Takeout
                        # layout. Placement stays date-derived, so this is a
                        # record and never a path.
                        "album": member_path.rpartition("/")[0].rpartition("/")[2] or None,
                        "title": meta.title,
                        "description": meta.description,
                        "photo_taken_time": (
                            meta.photo_taken_at.isoformat() if meta.photo_taken_at else None
                        ),
                        # Provenance only: creationTime is upload time, and is
                        # never allowed to place a file.
                        "creation_time": (
                            meta.creation_at.isoformat() if meta.creation_at else None
                        ),
                        "latitude": meta.latitude,
                        "longitude": meta.longitude,
                        "people": list(meta.people),
                        "favorited": meta.favorited,
                    }
                },
            )
        except Exception:
            logger.warning(
                "Failed to write Takeout sidecar block for %s; image is organized "
                "and catalogued", organized_path.name, exc_info=True,
            )

    def _defer_video(self, identity: archive.ArchiveIdentity, row) -> None:
        """Record a video with its capture date. No bytes are copied."""
        member_path = row["member_path"]
        sidecar_member = pairing.sidecar_for(member_path, self.pairing_index)
        meta = self._read_sidecar(sidecar_member) if sidecar_member else metadata.EMPTY
        self.catalog.takeout_member_set(
            identity.archive_id,
            member_path,
            status=_DEFERRED,
            taken_at=meta.photo_taken_at.isoformat() if meta.photo_taken_at else None,
            sidecar_path=sidecar_member,
        )
        self.stats.deferred += 1

    def _ingest_archive(self, identity: archive.ArchiveIdentity) -> None:
        pending = self.catalog.takeout_members_pending(identity.archive_id)
        if not pending:
            self.catalog.takeout_archive_set_status(identity.archive_id, "complete")
            return

        images = [r for r in pending if r["kind"] == archive.KIND_IMAGE]
        videos = [r for r in pending if r["kind"] == archive.KIND_VIDEO]

        try:
            with zipfile.ZipFile(identity.path, "r") as zf:
                for row in images:
                    self._ingest_image(zf, identity, row)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.error("Archive %s failed mid-ingest: %s", identity.path.name, exc)
            self.stats.archives_corrupt += 1
            self.catalog.takeout_archive_set_status(
                identity.archive_id, "corrupt", str(exc)
            )
            return

        for row in videos:
            self._defer_video(identity, row)

        if not self.catalog.takeout_members_pending(identity.archive_id):
            self.catalog.takeout_archive_set_status(identity.archive_id, "complete")

        self.stats.per_archive.append(
            {"archive": identity.path.name, "members": len(images) + len(videos)}
        )

    # -- driver -------------------------------------------------------------

    def run(self) -> IngestStats:
        todo = self._survey()

        if self.dry_run:
            # Report what phase 2 WOULD do, without extracting a byte or
            # writing a row. Counted from the surveyed member lists, NOT from
            # the catalog work queue -- a dry run wrote no rows, so the queue
            # would read empty.
            for _identity, members in todo:
                for member in members:
                    if _initial_status(member, self.include_trash) != _PENDING:
                        continue
                    if member.kind == archive.KIND_IMAGE:
                        self.stats.ingested += 1
                    elif member.kind == archive.KIND_VIDEO:
                        self.stats.deferred += 1
            return self.stats

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        for identity, _members in todo:
            self._ingest_archive(identity)
        return self.stats
```

- [ ] **Step 4: Complete the package `__init__.py`**

Replace the contents of `imageharbor/takeout/__init__.py` with the version
shown in Task 4 Step 3, now including the re-export lines.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_takeout_ingest.py -q`
Expected: PASS.

Note: `test_resume_after_a_mid_archive_crash_processes_only_the_remainder`
depends on `_ingest_archive` iterating `takeout_members_pending` in
`member_path` order (the catalog method already sorts) — if the assertion on
which member remains fails, check that ordering rather than loosening the test.

Run: `uv run pytest -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/takeout/ingest.py imageharbor/takeout/__init__.py tests/test_takeout_ingest.py
git commit -m "feat: two-phase Google Takeout ingestion orchestrator"
```

---

### Task 9: CLI — `takeout ingest` and `takeout status`

A Click **group** with two subcommands, following the existing `catalog
list`/`catalog get` precedent. There is no default subcommand — Click has no
first-class support for one, and `catalog` already establishes the
explicit-verb pattern. `process` is untouched.

**Files:**
- Modify: `imageharbor/cli.py`
- Modify: `.gitignore`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ingest_archives`, `IngestStats` (Task 8); `Catalog.takeout_status_counts` (Task 3).
- Produces the CLI surface:

```
imageharbor takeout ingest --archives DIR --dest DEST [--catalog PATH]
                           [--sidecar] [--include-trash] [--dry-run]
imageharbor takeout status [--catalog PATH]
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (the file already imports `CliRunner` and `main`;
reuse them):

```python
import json
import zipfile


def _takeout_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_takeout_ingest(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {
            "Takeout/AlbumArchive/a/2015-03-09.jpg": b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9",
            "Takeout/AlbumArchive/a/2015-03-09.jpg.json": json.dumps(
                {"title": "2015-03-09.jpg",
                 "photoTakenTime": {"timestampSeconds": "1425905792"}}
            ).encode(),
        },
    )

    result = CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert "ingested 1" in result.output
    assert (dest / "2015" / "2015-03").exists()


def test_takeout_ingest_dry_run_writes_nothing(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {"Takeout/a/x.jpg": b"\xff\xd8\xff\xe0" + b"\x01" * 16 + b"\xff\xd9"},
    )

    result = CliRunner().invoke(
        main,
        ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "[DRY-RUN]" in result.output
    assert not (dest / "catalog.db").exists()


def test_takeout_status(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(
        archives / "t.zip",
        {"Takeout/a/x.jpg": b"\xff\xd8\xff\xe0" + b"\x02" * 16 + b"\xff\xd9"},
    )
    CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )

    result = CliRunner().invoke(
        main, ["takeout", "status", "--catalog", str(dest / "catalog.db")]
    )
    assert result.exit_code == 0, result.output
    assert "1 archive" in result.output


def test_takeout_group_has_no_default_subcommand() -> None:
    result = CliRunner().invoke(main, ["takeout"])
    assert "ingest" in result.output
    assert "status" in result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL — `Error: No such command 'takeout'.`

- [ ] **Step 3: Add the command group**

In `imageharbor/cli.py`, add the import near the top:

```python
from .takeout.ingest import ingest_archives
```

and insert this block immediately before the `# --- catalog query ---` comment:

```python
# ---------------------------------------------------------------------------
# takeout
# ---------------------------------------------------------------------------


@click.group()
def takeout_cmd() -> None:
    """Ingest Google Takeout archives.

    Archives are opened read-only and are never modified. Ingestion is a
    hand-run verb: `watch` does not drive it.
    """


@takeout_cmd.command(name="ingest")
@click.option(
    "--archives",
    "archives_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory holding Google Takeout .zip archives (read-only).",
)
@click.option(
    "--dest",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Root directory for the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.  Defaults to <dest>/catalog.db.",
)
@click.option(
    "--sidecar/--no-sidecar",
    default=False,
    show_default=True,
    help="Write a JSON sidecar alongside each organized image.",
)
@click.option(
    "--include-trash",
    is_flag=True,
    default=False,
    help="Also ingest members under a Trash/ tree (skipped by default).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Survey the archives and report, without extracting or writing anything.",
)
def takeout_ingest(
    archives_dir: Path,
    dest: Path,
    catalog_path: Path | None,
    sidecar: bool,
    include_trash: bool,
    dry_run: bool,
) -> None:
    """Ingest Google Takeout archives from ARCHIVES into DEST.

    This is a facts pass: it makes no AI calls and requires no AI backend. Run
    `enrich` afterwards to describe and classify the organized copies.
    """
    _guard_dest_not_inside_source(archives_dir, dest)

    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    if dry_run:
        # Nothing may touch the disk: no dest tree, no catalog file. The
        # in-memory catalog reads empty, so every archive reports as new --
        # which is the honest answer for a run that will not record anything.
        catalog_target = Path(":memory:")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        catalog_target = catalog_path

    with Catalog(catalog_target) as catalog:
        stats = ingest_archives(
            archives_dir,
            dest,
            catalog,
            include_trash=include_trash,
            write_sidecars=sidecar,
            dry_run=dry_run,
        )

    if dry_run:
        click.echo("[DRY-RUN] No files were extracted and nothing was recorded.")
    click.echo(
        f"archives {stats.archives_seen} "
        f"(skipped {stats.archives_skipped}, corrupt {stats.archives_corrupt})"
    )
    click.echo(
        f"ingested {stats.ingested} / duplicates {stats.duplicates} / "
        f"deferred {stats.deferred} / trash {stats.skipped_trash} / "
        f"failed {stats.failed}"
    )
    if stats.missing_metadata:
        click.echo(f"{stats.missing_metadata} ingested without Google metadata")

    if stats.failed or stats.archives_corrupt:
        sys.exit(1)


@takeout_cmd.command(name="status")
@click.option(
    "--catalog",
    "catalog_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.",
)
def takeout_status(catalog_path: Path) -> None:
    """Report Takeout ingestion progress."""
    with Catalog(catalog_path) as cat:
        counts = cat.takeout_status_counts()

    archives = counts["archives"]
    total = sum(archives.values())
    detail = ", ".join(f"{n} {status}" for status, n in sorted(archives.items()))
    click.echo(f"{total} archive{'s' if total != 1 else ''}: {detail or 'none'}")

    members = counts["members"]
    member_detail = ", ".join(f"{n} {status}" for status, n in sorted(members.items()))
    click.echo(f"members: {member_detail or 'none'}")

    if counts["missing_metadata"]:
        click.echo(f"{counts['missing_metadata']} members missing Google metadata")


# Alias so `imageharbor takeout ingest` works
main.add_command(takeout_cmd, name="takeout")
```

- [ ] **Step 4: Update `.gitignore`**

Append to `.gitignore`:

```
# Takeout ingestion staging (inert debris after a kill; never state)
.takeout-staging/
# Real Takeout exports used for calibration are never committed
imageharbor/*.zip
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_cli.py -q`
Expected: PASS.

Run: `uv run imageharbor takeout --help`
Expected: shows `ingest` and `status`.

Run: `uv run pytest -q`
Expected: PASS, all green.

- [ ] **Step 6: Commit**

```bash
git add imageharbor/cli.py .gitignore tests/test_cli.py
git commit -m "feat: imageharbor takeout ingest and takeout status"
```

---

### Task 10: Documentation and final verification

`CLAUDE.md` is the architecture document every future session reads first. A new
package, two new catalog tables, a newly-populated date rung, and two new CLI
verbs all belong in it.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-12-google-takeout-ingestion-design.md`

- [ ] **Step 1: Update the `CLAUDE.md` command table**

In the "Commands" table, add after the `enrich` row:

```markdown
| Ingest Google Takeout archives | `uv run imageharbor takeout ingest --archives DIR --dest DEST` |
| Report Takeout ingestion progress | `uv run imageharbor takeout status --catalog DEST/catalog.db` |
```

- [ ] **Step 2: Document the package in `CLAUDE.md`'s Architecture section**

Add a module entry after the `pipeline.py` bullet:

```markdown
- **`takeout/`** — Google Takeout archive ingestion, a third entry point into the
  facts pass (`imageharbor takeout ingest`). Two phases: a **survey** that reads
  only zip central directories (no decompression) and builds ONE pairing index
  across every archive in the batch, then a resumable **ingest** that extracts
  one member at a time into `<dest>/.takeout-staging/` and hands it to
  `Pipeline.process_file`. Four modules: `metadata.py` (pure Google-JSON parser,
  never raises), `pairing.py` (pure media→sidecar matcher that returns `None`
  rather than guess), `archive.py` (identity, enumeration, classification,
  extraction), `ingest.py` (the only module with side effects). Archives are
  opened `'r'` only and are never modified. Makes **no AI calls**, so — exactly
  like the facts pass — it never consults or feeds the circuit breaker. Videos
  are enumerated and recorded as `deferred` with their capture date but no bytes
  are copied; video ingestion is a deliberate later project. The global (not
  per-archive) pairing index is load-bearing: Google's multi-part zips split by
  size across the file list, so a photo and its `.json` routinely land in
  different parts.
```

- [ ] **Step 3: Update the `catalog.py` and `date_resolver.py` entries in `CLAUDE.md`**

Append to the `catalog.py` bullet:

```markdown
  A `takeout_archives` table (`archive_id` = SHA-256 of the zip's own bytes,
  `status` partial|complete|corrupt) and a `takeout_members` table
  (`(archive_id, member_path)`, with terminal statuses `ingested`/`duplicate`/
  `deferred`/`parsed`/`ignored`/`skipped_trash` and non-terminal `pending`/
  `failed`) back Takeout ingestion's four idempotency layers. Both are purely
  additive, so `SCHEMA_VERSION` stays `"2"` and an existing catalog upgrades in
  place.
```

Append to the `date_resolver.py` bullet:

```markdown
  `resolve_date`'s `external_date` keyword populates the `DATE_EXTERNAL_SIDECAR`
  (30) rung, which sits below EXIF `DateTimeOriginal` and above
  `DateTimeDigitized`/`DateTime` — in practice Google Takeout's
  `photoTakenTime`. Google's `creationTime` is deliberately excluded for the
  same reason mtime is: it records upload time, not capture time.
```

- [ ] **Step 4: Add the invariant to `CLAUDE.md`'s "Critical invariants"**

```markdown
- **Takeout pairing never guesses, and Takeout archives are never written to.**
  If no pairing rung yields exactly one sidecar match, `pairing.sidecar_for`
  returns `None` and the member is ingested from EXIF and its filename alone —
  a fully correct outcome. A *wrong* pairing writes another photo's capture date
  into this photo's name and folder, which is exactly the quiet corruption the
  SHA-256 discipline exists to prevent. Separately: archives are opened `'r'`
  only, and Google's `creationTime` must never reach `resolve_date`.
```

- [ ] **Step 5: Update `README.md`**

Find the command list or usage section in `README.md` and add the two verbs to
it in that section's existing format. If it is a table, the rows are:

```markdown
| Ingest Google Takeout archives | `uv run imageharbor takeout ingest --archives DIR --dest DEST` |
| Report Takeout ingestion progress | `uv run imageharbor takeout status --catalog DEST/catalog.db` |
```

Also add one sentence to whatever overview paragraph describes `process`:

> Google Takeout `.zip` exports are ingested with `imageharbor takeout ingest`,
> which walks the archives read-only and feeds each member through the same
> facts pass — Google's `photoTakenTime` supplies a capture date when EXIF has
> none. Videos are inventoried for a later project but not copied.

- [ ] **Step 6: Mark the spec implemented**

In `docs/superpowers/specs/2026-08-12-google-takeout-ingestion-design.md`,
change the status line and record the two implementation-time decisions:

```markdown
**Status:** implemented 2026-08-15 (see `docs/superpowers/plans/2026-08-15-google-takeout-ingestion.md`)

**Implementation notes — two deliberate departures from the text above:**

1. `pairing.py` runs the case-insensitive retry (rung 6 in this document)
   *before* truncation recovery (rung 5). An exact match modulo extension case
   is strictly stronger evidence than a unique-prefix match, so running the
   fuzzier rung first could shadow it. Rungs 1–4 keep the specified order.
2. Phase 1 does **not** skip a `complete` archive "entirely": it loads that
   archive's member paths from `takeout_members` (no zip opened, nothing
   decompressed) so they still contribute to the global pairing index. Skipping
   them outright would hide the sidecars of an archive ingested *before* the
   part holding the photos they describe — the late-sidecar case in reverse.
```

- [ ] **Step 7: Final verification**

Run: `uv run pytest -q`
Expected: PASS, all green, zero failures.

Run: `uv run pytest --cov=imageharbor -q`
Expected: coverage report; every new module in `imageharbor/takeout/` should be
covered. Investigate any new module below ~85%.

Run: `uv run imageharbor --help`
Expected: `takeout` appears alongside `process`, `enrich`, `watch`, `verify`,
`catalog`.

Run against the real export, which is already on disk at
`imageharbor/takeout-20230618T004316Z-001.zip` (79 MB, 196 members, gitignored).
Copy it to a scratch directory first so the archives directory holds only it:

```bash
mkdir -p /tmp/ih-takeout/archives
cp imageharbor/takeout-20230618T004316Z-001.zip /tmp/ih-takeout/archives/
uv run imageharbor takeout ingest --archives /tmp/ih-takeout/archives --dest /tmp/ih-takeout/organized --sidecar
```

Expected: exits 0, reports non-zero `ingested`, `failed 0`. Then verify the
integrity of every organized copy and confirm re-running changes nothing:

```bash
uv run imageharbor verify /tmp/ih-takeout/organized
uv run imageharbor takeout ingest --archives /tmp/ih-takeout/archives --dest /tmp/ih-takeout/organized --sidecar
```

Expected: `verify` reports zero failures; the second ingest reports
`archives 1 (skipped 1, corrupt 0)` and `ingested 0`.

Confirm the archive was not modified:

```bash
sha256sum imageharbor/takeout-20230618T004316Z-001.zip /tmp/ih-takeout/archives/takeout-20230618T004316Z-001.zip
```

Expected: identical digests.

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md README.md docs/superpowers/specs/2026-08-12-google-takeout-ingestion-design.md
git commit -m "docs: document Takeout ingestion in the architecture reference"
```

---

## Notes for the implementer

**The one integration point that matters most.** Late-arriving metadata requires
no new code path. When part 2 arrives with the sidecars for part 1's photos,
those photos hash as duplicates, and `pipeline._maybe_upgrade_from_duplicate`
re-evaluates tiers against the new evidence: `is_upgrade((0, d), (30, d))` is
True, so the file relocates from `Undated/` into `2015/2015-03/`. This works
*only* because Task 7 Step 6 threads `evidence` into that method. If you find
yourself writing a special case for late sidecars, stop — you have missed that
step.

**What `--reclassify` teaches about `is_upgrade`.** Every rename in this system,
including every one Takeout ingestion causes, goes through `tiers.is_upgrade`.
There is no "force" path and there must not be one. Re-ingesting an
already-organized photo with the same evidence ties on both dimensions and is a
guaranteed rename no-op — that is what makes the whole thing idempotent.

**The four idempotency layers, cheapest first.** Archive stat fast path → archive
digest → member terminal status → content digest. Each one skips strictly more
work than the one below it. CRC32 is stored for diagnostics and for spotting an
archive that changed under a stale row; it is never the sole basis for a skip.
