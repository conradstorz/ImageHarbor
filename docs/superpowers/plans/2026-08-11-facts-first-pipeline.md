# Facts-First Pipeline + Monotonic Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split processing into a facts pass (no AI, disk-speed, complete on its own) and a resumable AI enrichment pass, with a date-derived folder tree and a tier system that guarantees a re-run never degrades a file.

**Architecture:** Two independently runnable passes over the library. The facts pass hashes, dedups, reads EXIF, resolves a date and descriptor from facts alone, copies, verifies, and catalogs. The enrichment pass later reads the *organized copy*, asks the AI for a description, updates the catalog and sidecar, and renames the file only when a two-ladder tier comparison says the result is strictly better. Placement comes from the date (`YYYY/YYYY-MM/`, or `Undated/`); PCS classification moves into the sidecar and catalog where it can be revised without touching disk.

**Tech Stack:** Python ≥3.10, `uv` for all dependency management and execution, Click for CLI, Pillow for EXIF, SQLite (WAL) for the catalog, pytest for tests. No linter/formatter is configured.

**Spec:** `docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python ≥3.10.** All modules start with `from __future__ import annotations`.
- **Use `uv` for everything.** `uv sync --extra dev` to install, `uv run pytest` to test, `uv run imageharbor` to run. Never `pip install`, `python -m venv`, or `source .venv/bin/activate`.
- **Do not chain shell commands with `&&`.** Issue separate commands.
- **The digest is 43 unpadded Base64url characters** (`hashing.SHA256_B64URL_LEN`) and is located by counting back from the **end of the stem**, never by splitting on `_`. Base64url legitimately contains `_`.
- **Content addressing must stay stable.** SHA-256 over raw file bytes, streamed in 64 KiB chunks. Do not change the algorithm, encoding, or length assumption.
- **Originals are read-only.** Files are copied, never moved or modified at the source. The copy is verified before it is cataloged; if verification fails the copy is deleted and nothing is cataloged.
- **Filenames stay ≤100 characters.**
- **PCS codes are strings** matching `^\d+(~\d+)*$` — `~N` suffixes, **never a dot**. `taxonomy.py`, `pcs.py`, and `concept_map.py` are **not modified by this plan**.
- **Every task ends with a commit.** Tests must pass before committing.

## Spec correction discovered during planning

The spec states `hashing.extract_digest_from_stem` is unchanged. **That is wrong** and Task 2 fixes it. The current implementation validates the prefix — it requires a `-` and a valid PCS code before the digest separator (`hashing.py:87-94`). Under the new grammar, `Undated/<digest>.jpg` has no prefix at all and would be rejected. The counting-back-43 *invariant* is preserved; the prefix *validation* is what must be relaxed, and a Base64url character-class check replaces it so the relaxation stays safe.

## File Structure

**New modules:**

| File | Responsibility |
|---|---|
| `imageharbor/tiers.py` | Tier constants and the `is_upgrade` predicate. Pure, no I/O, no imports from the package. |
| `imageharbor/date_resolver.py` | Resolve a capture date from EXIF and the original filename; report tier and folder. |
| `imageharbor/descriptor.py` | Decide whether an original filename is camera-generated or human-authored; produce a descriptor. |
| `imageharbor/relocate.py` | Compute a target path, apply a rename safely, and self-heal a catalog row after an interrupted rename. |
| `imageharbor/enrich.py` | The AI enrichment pass. |

**Modified:**

| File | Change |
|---|---|
| `imageharbor/hashing.py` | Relax prefix validation in `extract_digest_from_stem`. |
| `imageharbor/filename.py` | New grammar `[<date>][-<descriptor>]_<digest>.<ext>`. |
| `imageharbor/catalog.py` | `sources` table, tier columns on `photos`, new accessors. |
| `imageharbor/sidecar.py` | Cumulative read-merge-atomic-write. |
| `imageharbor/pipeline.py` | Facts pass only — no classifier, no taxonomy. |
| `imageharbor/watcher.py` | Two-phase sweep. |
| `imageharbor/cli.py` | AI/breaker/poison flags move to a new `enrich` verb. |
| `CLAUDE.md` | Documentation of the new architecture. |

**Not modified:** `taxonomy.py`, `pcs.py`, `concept_map.py`, `ai_classifier.py`, `circuit_breaker.py`, `discovery.py`, `exif_reader.py`.

---

### Task 1: Tier constants and the monotonicity predicate

**Files:**
- Create: `imageharbor/tiers.py`
- Test: `tests/test_tiers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DATE_EXIF_ORIGINAL=40`, `DATE_EXTERNAL_SIDECAR=30`, `DATE_EXIF_OTHER=20`, `DATE_FILENAME_PATTERN=10`, `DATE_NONE=0`, `DESC_HUMAN_FILENAME=30`, `DESC_AI_SUBJECT=20`, `DESC_NONE=0`, `DATE_SOURCE_NAMES: dict[int, str]`, `DESC_SOURCE_NAMES: dict[int, str]`, `is_upgrade(old: tuple[int, int], new: tuple[int, int]) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tiers.py`:

```python
"""Tests for the tier ladders and the monotonicity predicate."""

import pytest

from imageharbor import tiers


def test_date_ladder_is_ordered():
    assert (
        tiers.DATE_EXIF_ORIGINAL
        > tiers.DATE_EXTERNAL_SIDECAR
        > tiers.DATE_EXIF_OTHER
        > tiers.DATE_FILENAME_PATTERN
        > tiers.DATE_NONE
    )


def test_descriptor_ladder_puts_humans_above_ai():
    assert tiers.DESC_HUMAN_FILENAME > tiers.DESC_AI_SUBJECT > tiers.DESC_NONE


def test_source_names_cover_every_rank():
    assert set(tiers.DATE_SOURCE_NAMES) == {40, 30, 20, 10, 0}
    assert set(tiers.DESC_SOURCE_NAMES) == {30, 20, 0}


# (old_date, old_desc), (new_date, new_desc), expected
UPGRADE_CASES = [
    # Equal in both dimensions is never an upgrade -- this is what makes a
    # re-run a no-op.
    ((40, 30), (40, 30), False),
    ((0, 0), (0, 0), False),
    # Strictly better in one dimension, equal in the other.
    ((40, 0), (40, 20), True),
    ((0, 20), (40, 20), True),
    # Strictly better in both.
    ((0, 0), (40, 30), True),
    # Worse in either dimension is never an upgrade, even if the other improves.
    ((40, 30), (40, 20), False),
    ((40, 30), (0, 30), False),
    ((0, 30), (40, 20), False),
    ((40, 20), (20, 30), False),
]


@pytest.mark.parametrize("old,new,expected", UPGRADE_CASES)
def test_is_upgrade(old, new, expected):
    assert tiers.is_upgrade(old, new) is expected


def test_ai_can_never_displace_a_human_filename():
    """The central information-preservation guarantee."""
    human = (tiers.DATE_EXIF_ORIGINAL, tiers.DESC_HUMAN_FILENAME)
    ai = (tiers.DATE_EXIF_ORIGINAL, tiers.DESC_AI_SUBJECT)
    assert tiers.is_upgrade(human, ai) is False
    assert tiers.is_upgrade((tiers.DATE_EXIF_ORIGINAL, tiers.DESC_NONE), ai) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tiers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.tiers'`

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/tiers.py`:

```python
"""Quality tiers for date and descriptor provenance.

Two independent integer ladders decide what a re-run is allowed to change.
Ranks are spaced by 10 so a future source slots in without renumbering -- the
same append-only discipline the PCS taxonomy uses.

This module is pure: no I/O, and no imports from the rest of the package.
"""

from __future__ import annotations

# --- Date tier: decides placement -----------------------------------------
DATE_EXIF_ORIGINAL = 40      # EXIF DateTimeOriginal
DATE_EXTERNAL_SIDECAR = 30   # reserved: Google Takeout photoTakenTime
DATE_EXIF_OTHER = 20         # DateTimeDigitized, DateTime
DATE_FILENAME_PATTERN = 10   # date parsed out of the original filename
DATE_NONE = 0                # no trustworthy date -> Undated/

# File mtime is deliberately absent from this ladder. It is evidence of when a
# file was copied, not of when a photo was taken.

DATE_SOURCE_NAMES: dict[int, str] = {
    DATE_EXIF_ORIGINAL: "exif_original",
    DATE_EXTERNAL_SIDECAR: "external_sidecar",
    DATE_EXIF_OTHER: "exif_other",
    DATE_FILENAME_PATTERN: "filename_pattern",
    DATE_NONE: "none",
}

# --- Descriptor tier: decides the name ------------------------------------
DESC_HUMAN_FILENAME = 30     # original stem that no camera pattern matched
DESC_AI_SUBJECT = 20         # classifier primary_subject
DESC_NONE = 0                # nothing available

DESC_SOURCE_NAMES: dict[int, str] = {
    DESC_HUMAN_FILENAME: "human_filename",
    DESC_AI_SUBJECT: "ai_subject",
    DESC_NONE: "none",
}


def is_upgrade(old: tuple[int, int], new: tuple[int, int]) -> bool:
    """Return True if *new* is strictly better than *old*.

    Both arguments are ``(date_tier, descriptor_tier)``.  An upgrade requires
    a strict improvement in at least one dimension and no regression in either.
    Equality in both dimensions is NOT an upgrade, which is what makes a
    repeated run a no-op.
    """
    old_date, old_desc = old
    new_date, new_desc = new
    if new_date < old_date or new_desc < old_desc:
        return False
    return new_date > old_date or new_desc > old_desc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tiers.py -v`
Expected: PASS — 13 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/tiers.py tests/test_tiers.py
git commit -m "feat: tier ladders and monotonicity predicate"
```

---

### Task 2: Relax digest extraction to the new filename grammar

**Files:**
- Modify: `imageharbor/hashing.py:70-107`
- Test: `tests/test_hashing.py`

**Interfaces:**
- Consumes: `SHA256_B64URL_LEN` from Task 0 (pre-existing).
- Produces: `extract_digest_from_stem(stem: str) -> str | None` accepting a bare digest stem, a `<prefix>_<digest>` stem, and legacy PCS stems. `verify_pcs_file(path: Path) -> bool` keeps its name and signature.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hashing.py`:

```python
from imageharbor.hashing import extract_digest_from_stem

_D = "qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI"  # 43 chars, valid base64url


def test_extract_accepts_bare_digest_stem():
    """Undated/<digest>.jpg has no prefix at all."""
    assert extract_digest_from_stem(_D) == _D


def test_extract_accepts_date_and_descriptor():
    assert extract_digest_from_stem(f"2019-07-04-emmas-graduation_{_D}") == _D


def test_extract_accepts_date_only():
    assert extract_digest_from_stem(f"2019-07-04_{_D}") == _D


def test_extract_accepts_descriptor_only():
    """Undated file that still has a human name."""
    assert extract_digest_from_stem(f"beach-trip-scan_{_D}") == _D


def test_extract_still_accepts_legacy_pcs_names():
    """Files organized by the old scheme must remain verifiable."""
    assert extract_digest_from_stem(f"330-beach_{_D}") == _D


def test_extract_rejects_non_base64url_tail():
    bad = "!" * 43
    assert extract_digest_from_stem(f"2019-07-04_{bad}") is None


def test_extract_rejects_empty_prefix():
    assert extract_digest_from_stem(f"_{_D}") is None


def test_extract_rejects_short_stem():
    assert extract_digest_from_stem("abc") is None


def test_extract_rejects_missing_separator():
    assert extract_digest_from_stem(f"2019-07-04x{_D}") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hashing.py -v -k "extract_accepts or extract_rejects"`
Expected: FAIL — `test_extract_accepts_bare_digest_stem` and `test_extract_accepts_descriptor_only` return `None` (the current code requires a `-` and a valid PCS code before the separator).

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/hashing.py`, replace the `_PCS_CODE_RE` definition (line 16) with:

```python
# The digest is unpadded Base64url: exactly 43 characters from the RFC 4648 §5
# alphabet. Validating the character class is what lets the *prefix* be
# unconstrained -- the filename grammar allows a bare digest, a date, a
# descriptor, both, or (for legacy files) a PCS code.
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
```

Then replace `extract_digest_from_stem` (lines 70-95) with:

```python
def extract_digest_from_stem(stem: str) -> str | None:
    """Extract the Base64url digest from an organized filename stem.

    The grammar is ``[<date>][-<descriptor>]_<digest>``, where both prefix
    components are optional -- so a stem may be nothing but the digest itself.
    Because base64url may contain ``_``, the separator is located by counting
    back exactly :data:`SHA256_B64URL_LEN` characters from the end of the stem
    rather than splitting on the last underscore.

    Legacy ``<pcs>-<descriptor>_<digest>`` stems parse unchanged, so files
    organized by the previous scheme remain verifiable.

    Returns the 43-character digest, or None if the stem does not match.
    """
    # A stem that is nothing but the digest: Undated/<digest>.jpg
    if len(stem) == SHA256_B64URL_LEN:
        return stem if _B64URL_RE.match(stem) else None

    # Otherwise we need at least a one-character prefix plus the separator.
    if len(stem) < SHA256_B64URL_LEN + 2:
        return None
    sep_idx = len(stem) - SHA256_B64URL_LEN - 1
    if stem[sep_idx] != "_":
        return None
    if not stem[:sep_idx]:
        return None
    digest = stem[sep_idx + 1 :]
    return digest if _B64URL_RE.match(digest) else None
```

Update the `verify_pcs_file` docstring to say "an organized file" rather than "a PCS-named file"; leave its name and signature alone so `cli.verify` is untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: PASS — all tests including the pre-existing ones.

Then check the rest of the suite: `uv run pytest -q`
Expected: **5 failures in `tests/test_filename.py`, and nothing else.**

This is a known, intentional red window. `filename.parse_filename` currently
leans on `extract_digest_from_stem` to pre-validate the PCS prefix, so relaxing
the extractor drops that guard until Task 3 rewrites `parse_filename` to own it.
Confirm the failures are confined to `test_filename.py`; **do not** edit
`imageharbor/filename.py` to make them pass — that file belongs to Task 3, which
replaces `parse_filename` and `ParsedFilename` wholesale.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/hashing.py tests/test_hashing.py
git commit -m "fix: accept prefix-free and date-prefixed stems in digest extraction"
```

---

### Task 3: New filename grammar

**Files:**
- Modify: `imageharbor/filename.py`
- Test: `tests/test_filename.py`

**Interfaces:**
- Consumes: `hashing.SHA256_B64URL_LEN`, `hashing.extract_digest_from_stem` (Task 2).
- Produces: `build_filename(date_str: str | None, descriptor: str | None, sha256_b64url: str, extension: str) -> str`; `parse_filename(filename: str) -> ParsedFilename | None` where `ParsedFilename` is a TypedDict with keys `date: str | None`, `descriptor: str`, `sha256_b64url: str`, `extension: str`. `normalize_descriptor` is unchanged. The old `generate_filename` is **removed**.

- [ ] **Step 1: Write the failing test**

Replace the contents of `tests/test_filename.py` with:

```python
"""Tests for the organized filename grammar."""

import pytest

from imageharbor.filename import build_filename, normalize_descriptor, parse_filename

_D = "qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI"


# --- normalize_descriptor (behavior preserved from the previous scheme) ----

def test_normalize_lowercases_and_hyphenates():
    assert normalize_descriptor("Indiana Dunes") == "indiana-dunes"


def test_normalize_keeps_at_most_three_words():
    assert normalize_descriptor("a b c d e") == "a-b-c"


def test_normalize_falls_back_to_photo():
    assert normalize_descriptor("!!!") == "photo"


# --- build_filename -------------------------------------------------------

def test_build_with_date_and_descriptor():
    assert (
        build_filename("2019-07-04", "emmas-graduation", _D, "jpg")
        == f"2019-07-04-emmas-graduation_{_D}.jpg"
    )


def test_build_with_date_only():
    assert build_filename("2019-07-04", None, _D, "jpg") == f"2019-07-04_{_D}.jpg"


def test_build_with_descriptor_only():
    assert build_filename(None, "beach-scan", _D, "jpg") == f"beach-scan_{_D}.jpg"


def test_build_with_neither_is_a_bare_digest():
    assert build_filename(None, None, _D, "jpg") == f"{_D}.jpg"


def test_build_treats_empty_descriptor_as_absent():
    assert build_filename("2019-07-04", "", _D, "jpg") == f"2019-07-04_{_D}.jpg"


def test_build_normalizes_the_extension():
    assert build_filename(None, None, _D, ".JPEG") == f"{_D}.jpeg"


def test_build_stays_within_100_chars_and_truncates_descriptor_not_date():
    name = build_filename("2019-07-04", "a" * 80, _D, "jpg")
    assert len(name) <= 100
    assert name.startswith("2019-07-04-")
    assert name.endswith(f"_{_D}.jpg")


def test_build_disambiguates_a_date_shaped_descriptor_with_no_date():
    """A scan named "2019.07.04.jpg" normalizes to a date-shaped descriptor.

    Emitting it verbatim would produce a name identical to a genuinely dated
    file's, asserting a date the system never established and contradicting the
    Undated/ folder it lives in.
    """
    name = build_filename(None, "2019-07-04", _D, "jpg")
    assert name == f"20190704_{_D}.jpg"
    parsed = parse_filename(name)
    assert parsed["date"] is None
    assert parsed["descriptor"] == "20190704"


def test_a_real_date_is_still_emitted_verbatim():
    """The guard must only fire when no date was supplied."""
    assert build_filename("2019-07-04", None, _D, "jpg") == f"2019-07-04_{_D}.jpg"


def test_build_output_round_trips():
    name = build_filename("2019-07-04", "emmas-graduation", _D, "jpg")
    parsed = parse_filename(name)
    assert parsed == {
        "date": "2019-07-04",
        "descriptor": "emmas-graduation",
        "sha256_b64url": _D,
        "extension": "jpg",
    }


# --- parse_filename -------------------------------------------------------

@pytest.mark.parametrize(
    "name,date,descriptor",
    [
        (f"2019-07-04-emmas-graduation_{_D}.jpg", "2019-07-04", "emmas-graduation"),
        (f"2019-07-04_{_D}.jpg", "2019-07-04", ""),
        (f"beach-scan_{_D}.jpg", None, "beach-scan"),
        (f"{_D}.jpg", None, ""),
        # A legacy PCS name has no date, so the whole prefix is the descriptor.
        (f"330-beach_{_D}.jpg", None, "330-beach"),
    ],
)
def test_parse_variants(name, date, descriptor):
    parsed = parse_filename(name)
    assert parsed is not None
    assert parsed["date"] == date
    assert parsed["descriptor"] == descriptor
    assert parsed["sha256_b64url"] == _D


def test_parse_accepts_a_full_path():
    parsed = parse_filename(f"/lib/2019/2019-07/2019-07-04_{_D}.jpg")
    assert parsed is not None
    assert parsed["date"] == "2019-07-04"


def test_parse_rejects_a_non_organized_name():
    assert parse_filename("IMG_1234.jpg") is None


def test_parse_does_not_mistake_a_numeric_descriptor_for_a_date():
    parsed = parse_filename(f"2019-summer_{_D}.jpg")
    assert parsed is not None
    assert parsed["date"] is None
    assert parsed["descriptor"] == "2019-summer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_filename.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_filename'`

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/filename.py`, keep `normalize_descriptor` and its constants exactly as they are. Replace `generate_filename`, `ParsedFilename`, and `parse_filename` with:

```python
# A date prefix is exactly YYYY-MM-DD. Anchored so a descriptor that merely
# starts with digits (e.g. "2019-summer") is not misread as a date.
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:-(.+))?$")


def build_filename(
    date_str: str | None,
    descriptor: str | None,
    sha256_b64url: str,
    extension: str,
) -> str:
    """Return an organized filename.

    Format: ``[<date>][-<descriptor>]_<digest>.<ext>``.  Both prefix components
    are optional; with neither, the stem is the bare digest.

    The total length is guaranteed <= 100 characters.  The descriptor is
    truncated first (the date is never sacrificed, since it must agree with the
    folder the file lives in); a pathologically long extension is truncated
    last.
    """
    ext = re.sub(r"[^a-z0-9]", "", extension.lower().rsplit(".", 1)[-1])
    suffix = f".{ext}" if ext else ""
    desc = normalize_descriptor(descriptor) if descriptor else ""

    # A date-shaped descriptor with no date supplied would re-parse as a date
    # the system never established, and would contradict the Undated/ folder the
    # file lives in. Strip the hyphens so the grammar stays unambiguous.
    # Reachable in practice: "2019.07.04.jpg" normalizes to "2019-07-04".
    if date_str is None and _DATE_PREFIX_RE.match(desc):
        desc = desc.replace("-", "")

    def assemble(d: str) -> str:
        prefix = "-".join(part for part in (date_str or "", d) if part)
        return f"{prefix}_{sha256_b64url}{suffix}" if prefix else f"{sha256_b64url}{suffix}"

    name = assemble(desc)
    if len(name) > _MAX_FILENAME_LEN and desc:
        overflow = len(name) - _MAX_FILENAME_LEN
        desc = desc[: max(0, len(desc) - overflow)].rstrip("-")
        name = assemble(desc)

    if len(name) > _MAX_FILENAME_LEN and ext:
        overflow = len(name) - _MAX_FILENAME_LEN
        ext = ext[: max(0, len(ext) - overflow)]
        suffix = f".{ext}" if ext else ""
        name = assemble(desc)

    return name


class ParsedFilename(TypedDict):
    date: str | None
    descriptor: str
    sha256_b64url: str
    extension: str


def parse_filename(filename: str) -> ParsedFilename | None:
    """Parse an organized filename, or return None if it is not one.

    Accepts bare filenames and full paths.  The digest is located by counting
    back from the end of the stem (see
    :func:`~imageharbor.hashing.extract_digest_from_stem`); everything before
    the separator is split into an optional ``YYYY-MM-DD`` date and an optional
    descriptor.
    """
    p = Path(filename)
    stem = p.stem
    ext = p.suffix.lstrip(".").lower()

    digest = extract_digest_from_stem(stem)
    if digest is None:
        return None

    # Recover the prefix: everything before "_<digest>", or "" for a bare digest.
    prefix = "" if len(stem) == SHA256_B64URL_LEN else stem[: len(stem) - SHA256_B64URL_LEN - 1]

    date: str | None = None
    descriptor = prefix
    match = _DATE_PREFIX_RE.match(prefix)
    if match:
        date = match.group(1)
        descriptor = match.group(2) or ""

    return ParsedFilename(
        date=date,
        descriptor=descriptor,
        sha256_b64url=digest,
        extension=ext,
    )
```

Update the import at the top of the file to bring in `SHA256_B64URL_LEN` alongside `extract_digest_from_stem` (it is already imported).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_filename.py -v`
Expected: PASS

Note: `uv run pytest -q` will now FAIL in `tests/test_pipeline.py` because `pipeline.py` still imports `generate_filename`. That is expected and is fixed in Task 8. Confirm the failure is confined to that import:

Run: `uv run pytest -q 2>&1 | tail -20`
Expected: failures only in `test_pipeline.py` / `test_cli.py` / `test_watcher.py`, all from `ImportError: cannot import name 'generate_filename'`.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/filename.py tests/test_filename.py
git commit -m "feat: date-prefixed filename grammar, replacing the PCS prefix"
```

---

### Task 4: Descriptor resolution from the original filename

**Files:**
- Create: `imageharbor/descriptor.py`
- Modify: `imageharbor/filename.py` — `normalize_descriptor` only (see Step 0)
- Test: `tests/test_descriptor.py`, `tests/test_filename.py`

**Interfaces:**
- Consumes: `tiers.DESC_HUMAN_FILENAME`, `tiers.DESC_NONE`, `tiers.DESC_SOURCE_NAMES` (Task 1); `filename.normalize_descriptor` (Task 3).

- [ ] **Step 0: Teach `normalize_descriptor` to drop apostrophes**

`normalize_descriptor` replaces every non-alphanumeric run with a space, so an
apostrophe becomes a word break: `"Emma's graduation"` → `emma-s-graduation`.
With a 3-word cap that burns a slot on a stray `s`, and `"Dad's birthday party"`
silently loses `party`. Apostrophes are common in exactly the human-authored
filenames this tier exists to preserve, so delete them instead of splitting on
them.

In `imageharbor/filename.py`, inside `normalize_descriptor`, replace:

```python
    lowered = text.lower()
```

with:

```python
    # Drop apostrophes rather than splitting on them: the generic non-alphanumeric
    # rule below would turn "Emma's graduation" into "emma-s-graduation", burning
    # one of only three word slots on a stray "s". U+2019 is the curly apostrophe
    # macOS and Windows insert automatically.
    lowered = text.lower().replace("'", "").replace("’", "")
```

Add to `tests/test_filename.py`, after `test_normalize_keeps_at_most_three_words`:

```python
def test_normalize_drops_apostrophes_instead_of_splitting_on_them():
    assert normalize_descriptor("Emma's graduation") == "emmas-graduation"
    assert normalize_descriptor("Dad's birthday party") == "dads-birthday-party"


def test_normalize_drops_the_curly_apostrophe_too():
    assert normalize_descriptor("Emma’s graduation") == "emmas-graduation"
```

Run: `uv run pytest tests/test_filename.py -v`
Expected: PASS — 23 tests.
- Produces: `is_camera_generated(stem: str) -> bool`; `resolve_descriptor(source_path: Path) -> ResolvedDescriptor` where `ResolvedDescriptor` is a frozen dataclass with fields `value: str`, `tier: int`, `source: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_descriptor.py`:

```python
"""Tests for camera-pattern detection and descriptor resolution."""

from pathlib import Path

import pytest

from imageharbor import tiers
from imageharbor.descriptor import is_camera_generated, resolve_descriptor

CAMERA_STEMS = [
    "IMG_1234",
    "IMG-20190704-WA0001",
    "img_20190704_123456",
    "DSC0042",
    "DSCN0042",
    "DSCF0042",
    "_DSC0042",
    "PXL_20190704_123456789",
    "MVIMG_20190704_123456",
    "P1000042",
    "PICT0042",
    "100_0042",
    "CIMG0042",
    "SAM_0042",
    "GOPR0042",
    "DJI_0042",
    "Screenshot_2019-07-04-12-33-11",
    "Screen Shot 2019-07-04 at 12.33.11",
    "WhatsApp Image 2019-07-04 at 12.33.11",
    "Signal-2019-07-04-123311",
    "FB_IMG_1562243591",
    "received_101234567890",
    "20190704_123456",
    "2019-07-04 12.33.11",
    "1562243591",
]

HUMAN_STEMS = [
    "Emma's graduation",
    "beach trip 2019",
    "grandpa and the tractor",
    "kitchen remodel before",
    "scan0001 aunt martha",
    "Christmas",
    # Regression cases: each of these was destroyed by an earlier, looser
    # pattern. A false positive here discards a human's name permanently in
    # favour of an AI guess, so they are pinned deliberately.
    "Screenshot - grandpas last text message",
    "WhatsApp Image of the new puppy",
    "Sam_1",
    "Sam_2",
]


@pytest.mark.parametrize("stem", CAMERA_STEMS)
def test_camera_stems_are_detected(stem):
    assert is_camera_generated(stem) is True


@pytest.mark.parametrize("stem", HUMAN_STEMS)
def test_human_stems_are_not_camera_generated(stem):
    assert is_camera_generated(stem) is False


def test_camera_detection_is_case_insensitive():
    assert is_camera_generated("img_1234") is True
    assert is_camera_generated("dsc0042") is True


def test_resolve_keeps_a_human_name_at_the_top_tier(tmp_path):
    path = tmp_path / "Emma's graduation.jpg"
    path.write_bytes(b"x")
    resolved = resolve_descriptor(path)
    assert resolved.value == "emmas-graduation"
    assert resolved.tier == tiers.DESC_HUMAN_FILENAME
    assert resolved.source == "human_filename"


def test_resolve_discards_a_camera_name(tmp_path):
    path = tmp_path / "IMG_1234.jpg"
    path.write_bytes(b"x")
    resolved = resolve_descriptor(path)
    assert resolved.value == ""
    assert resolved.tier == tiers.DESC_NONE
    assert resolved.source == "none"


def test_resolve_discards_a_stem_that_normalizes_to_nothing(tmp_path):
    """A stem of pure punctuation carries no information."""
    path = tmp_path / "___.jpg"
    path.write_bytes(b"x")
    resolved = resolve_descriptor(path)
    assert resolved.tier == tiers.DESC_NONE


def test_resolve_truncates_to_three_words(tmp_path):
    path = tmp_path / "the big family reunion picnic.jpg"
    path.write_bytes(b"x")
    assert resolve_descriptor(path).value == "the-big-family"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_descriptor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.descriptor'`

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/descriptor.py`:

```python
"""Descriptor resolution from the original filename.

An original filename is itself a fact, and often the best one available: a
person who typed "Emma's graduation" knew something no model will recover from
the pixels.  Camera-generated names carry no such information, so they are
discarded and the descriptor waits for the AI enrichment pass.

The pattern list below is the one empirical claim in this module.  Keep it
adjacent to its fixture table in tests/test_descriptor.py so additions arrive
with tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import tiers
from .filename import normalize_descriptor

# Matched case-insensitively against the full original stem. A match means
# "no human information here".
CAMERA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^_?img[-_]?\d+$", re.I),                      # IMG_1234, _IMG0042
    re.compile(r"^img[-_]\d{8}[-_]wa\d+$", re.I),              # IMG-20190704-WA0001
    re.compile(r"^img[-_]\d{8}[-_]\d{6}$", re.I),              # IMG_20190704_123456
    re.compile(r"^_?dsc[nf]?[-_]?\d+$", re.I),                 # DSC0042, DSCN, DSCF, _DSC
    re.compile(r"^p?xl[-_]\d{8}[-_]\d+$", re.I),               # PXL_20190704_123456789
    re.compile(r"^mvimg[-_]\d{8}[-_]\d{6}$", re.I),            # MVIMG_20190704_123456
    re.compile(r"^p\d{7}$", re.I),                             # P1000042 (Panasonic)
    re.compile(r"^pict\d+$", re.I),                            # PICT0042
    re.compile(r"^\d{3}[-_]\d{4}$", re.I),                     # 100_0042
    re.compile(r"^cimg\d+$", re.I),                            # CIMG0042 (Casio)
    # Samsung's format is exactly 4 digits. Do NOT relax this to \d+ -- "sam" is
    # also a person's name, and "Sam_1.jpg" is an ordinary way to label photos
    # of someone. A tier-0 verdict discards that name permanently.
    re.compile(r"^sam[-_]\d{4}$", re.I),                       # SAM_0042 (Samsung)
    re.compile(r"^gopr\d+$", re.I),                            # GOPR0042
    re.compile(r"^dji[-_]\d+$", re.I),                         # DJI_0042
    # These three require a DIGIT after the auto-generated prefix, so the
    # timestamp forms match but an appended human suffix survives:
    # "Screenshot - grandpas last text message" stays tier 30. An open .*$ here
    # would swallow it. One pattern covers "Screenshot" and "Screen Shot" both,
    # since [-_ ]? already matches zero separator characters.
    re.compile(r"^screen[-_ ]?shot[-_ ]?\d.*$", re.I),         # Screenshot_2019-...
    re.compile(r"^whatsapp[ -](image|video)[ -]?\d.*$", re.I), # WhatsApp Image 2019-...
    re.compile(r"^signal[-_]\d{4}-\d{2}-\d{2}.*$", re.I),      # Signal-2019-07-04-...
    re.compile(r"^fb[-_]img[-_]\d+$", re.I),                   # FB_IMG_1562243591
    re.compile(r"^received[-_]\d+$", re.I),                    # received_101234567890
    re.compile(r"^\d{8}[-_]\d{6}$", re.I),                     # 20190704_123456
    re.compile(r"^\d{4}-\d{2}-\d{2}[ _]\d{2}\.\d{2}\.\d{2}$"), # 2019-07-04 12.33.11
    re.compile(r"^\d{9,13}$"),                                 # bare epoch seconds/ms
)


@dataclass(frozen=True)
class ResolvedDescriptor:
    """A descriptor together with the provenance that justifies its tier."""

    value: str
    tier: int
    source: str


_NONE = ResolvedDescriptor(value="", tier=tiers.DESC_NONE, source=tiers.DESC_SOURCE_NAMES[tiers.DESC_NONE])


def is_camera_generated(stem: str) -> bool:
    """Return True if *stem* looks machine-generated rather than human-authored."""
    candidate = stem.strip()
    return any(pattern.match(candidate) for pattern in CAMERA_PATTERNS)


def resolve_descriptor(source_path: Path) -> ResolvedDescriptor:
    """Derive a descriptor from *source_path*'s original filename.

    Returns tier ``DESC_HUMAN_FILENAME`` when the stem carries human intent,
    and ``DESC_NONE`` when it does not -- leaving the slot open for the AI
    enrichment pass to fill at the lower ``DESC_AI_SUBJECT`` tier.
    """
    stem = source_path.stem
    if not stem or is_camera_generated(stem):
        return _NONE

    normalized = normalize_descriptor(stem)
    # normalize_descriptor falls back to "photo" for input with no usable
    # characters; that is not information, so treat it as absent.
    if not normalized or normalized == "photo":
        return _NONE

    return ResolvedDescriptor(
        value=normalized,
        tier=tiers.DESC_HUMAN_FILENAME,
        source=tiers.DESC_SOURCE_NAMES[tiers.DESC_HUMAN_FILENAME],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_descriptor.py -v`
Expected: PASS — 37 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/descriptor.py tests/test_descriptor.py
git commit -m "feat: descriptor resolution with camera-pattern detection"
```

---

### Task 5: Date resolution and folder derivation

**Files:**
- Create: `imageharbor/date_resolver.py`
- Test: `tests/test_date_resolver.py`

**Interfaces:**
- Consumes: `tiers.DATE_*` constants and `tiers.DATE_SOURCE_NAMES` (Task 1).
- Produces: `resolve_date(source_path: Path, exif_data: dict[str, Any]) -> ResolvedDate`; `date_from_filename(stem: str) -> datetime | None`; `ResolvedDate` frozen dataclass with fields `value: datetime | None`, `tier: int`, `source: str` and properties `folder: str` and `date_str: str | None`; `UNDATED_FOLDER = "Undated"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_date_resolver.py`:

```python
"""Tests for the date ladder and folder derivation."""

from datetime import datetime
from pathlib import Path

import pytest

from imageharbor import tiers
from imageharbor.date_resolver import UNDATED_FOLDER, date_from_filename, resolve_date


def _p(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"x")
    return path


# --- the ladder -----------------------------------------------------------

def test_exif_original_is_the_top_rung(tmp_path):
    exif = {
        "DateTimeOriginal": "2019:07:04 12:33:11",
        "DateTimeDigitized": "2020:01:01 00:00:00",
        "DateTime": "2021:01:01 00:00:00",
    }
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), exif)
    assert resolved.value == datetime(2019, 7, 4, 12, 33, 11)
    assert resolved.tier == tiers.DATE_EXIF_ORIGINAL
    assert resolved.source == "exif_original"


def test_falls_back_to_other_exif_fields(tmp_path):
    exif = {"DateTime": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), exif)
    assert resolved.tier == tiers.DATE_EXIF_OTHER
    assert resolved.source == "exif_other"


def test_falls_back_to_the_filename(tmp_path):
    resolved = resolve_date(_p(tmp_path, "IMG_20190704_123456.jpg"), {})
    assert resolved.value == datetime(2019, 7, 4, 12, 34, 56)
    assert resolved.tier == tiers.DATE_FILENAME_PATTERN
    assert resolved.source == "filename_pattern"


def test_no_evidence_means_undated(tmp_path):
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), {})
    assert resolved.value is None
    assert resolved.tier == tiers.DATE_NONE
    assert resolved.source == "none"


def test_mtime_is_never_used(tmp_path):
    """mtime is evidence of copying, not of capture."""
    path = _p(tmp_path, "IMG_1234.jpg")
    import os
    os.utime(path, (1562243591, 1562243591))
    assert resolve_date(path, {}).tier == tiers.DATE_NONE


# --- filename patterns ----------------------------------------------------

@pytest.mark.parametrize(
    "stem,expected",
    [
        ("IMG_20190704_123456", datetime(2019, 7, 4, 12, 34, 56)),
        ("PXL_20190704_123456789", datetime(2019, 7, 4, 12, 34, 56)),
        ("20190704_123456", datetime(2019, 7, 4, 12, 34, 56)),
        ("Screenshot_2019-07-04-12-33-11", datetime(2019, 7, 4, 12, 33, 11)),
        ("2019-07-04 12.33.11", datetime(2019, 7, 4, 12, 33, 11)),
        ("WhatsApp Image 2019-07-04 at 12.33.11", datetime(2019, 7, 4)),
        ("beach trip 2019-07-04", datetime(2019, 7, 4)),
        ("IMG-20190704-WA0001", datetime(2019, 7, 4)),
        ("2019.07.04", datetime(2019, 7, 4)),
        ("2019 07 04", datetime(2019, 7, 4)),
    ],
)
def test_date_from_filename_hits(stem, expected):
    assert date_from_filename(stem) == expected


@pytest.mark.parametrize(
    "stem",
    ["IMG_1234", "DSC0042", "Emma's graduation", "1562243591", "20190732_123456"],
)
def test_date_from_filename_misses(stem):
    """A bare epoch is deliberately not decoded, and 07-32 is not a date."""
    assert date_from_filename(stem) is None


def test_implausible_years_are_rejected(tmp_path):
    exif = {"DateTimeOriginal": "1601:01:01 00:00:00"}
    assert resolve_date(_p(tmp_path, "x.jpg"), exif).tier == tiers.DATE_NONE


def test_malformed_exif_date_is_ignored(tmp_path):
    exif = {"DateTimeOriginal": "not a date", "DateTime": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "x.jpg"), exif)
    assert resolved.tier == tiers.DATE_EXIF_OTHER


def test_exif_zero_date_is_ignored(tmp_path):
    """Cameras with a dead clock emit all-zero timestamps."""
    exif = {"DateTimeOriginal": "0000:00:00 00:00:00"}
    assert resolve_date(_p(tmp_path, "x.jpg"), exif).tier == tiers.DATE_NONE


# --- folder derivation ----------------------------------------------------

def test_folder_is_year_and_month(tmp_path):
    exif = {"DateTimeOriginal": "2019:07:04 12:33:11"}
    resolved = resolve_date(_p(tmp_path, "x.jpg"), exif)
    assert resolved.folder == "2019/2019-07"
    assert resolved.date_str == "2019-07-04"


def test_undated_folder(tmp_path):
    resolved = resolve_date(_p(tmp_path, "IMG_1234.jpg"), {})
    assert resolved.folder == UNDATED_FOLDER
    assert resolved.date_str is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_date_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.date_resolver'`

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/date_resolver.py`:

```python
"""Capture-date resolution.

The date is the load-bearing fact of the organized library: it decides the
folder a file lands in, and placement is meant to be permanent.  So every rung
of this ladder is evidence about when the photo was *taken*, and file mtime --
which records when a file was last copied -- is deliberately absent.

A file with no trustworthy date goes to ``Undated/`` and waits.  Asserting a
year we cannot support would be exactly the quiet corruption the project's
SHA-256 discipline exists to prevent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import tiers

logger = logging.getLogger(__name__)

UNDATED_FOLDER = "Undated"

# EXIF stores timestamps as "YYYY:MM:DD HH:MM:SS".
_EXIF_FORMAT = "%Y:%m:%d %H:%M:%S"

# Photography began in 1826; anything earlier is a dead clock or a bad parse.
_MIN_YEAR = 1826
_MAX_YEAR = 2100

# EXIF fields in ladder order: (field name, tier).
_EXIF_FIELDS: tuple[tuple[str, int], ...] = (
    ("DateTimeOriginal", tiers.DATE_EXIF_ORIGINAL),
    ("DateTimeDigitized", tiers.DATE_EXIF_OTHER),
    ("DateTime", tiers.DATE_EXIF_OTHER),
)

# Filename date patterns, most specific first. Each yields named groups.
# A bare epoch is deliberately NOT decoded: it is indistinguishable from an
# ordinary counter, so treating it as a timestamp would invent evidence.
_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 20190704_123456 / IMG_20190704_123456 / PXL_20190704_123456789
    re.compile(
        r"(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})[-_](?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"
    ),
    # 2019-07-04-12-33-11 / 2019-07-04_12-33-11
    re.compile(
        r"(?P<Y>\d{4})-(?P<M>\d{2})-(?P<D>\d{2})[-_](?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})"
    ),
    # 2019-07-04 12.33.11
    re.compile(
        r"(?P<Y>\d{4})-(?P<M>\d{2})-(?P<D>\d{2})[ _](?P<h>\d{2})\.(?P<m>\d{2})\.(?P<s>\d{2})"
    ),
    # Date only: 2019-07-04
    re.compile(r"(?P<Y>\d{4})-(?P<M>\d{2})-(?P<D>\d{2})"),
    # Date only, dotted or space-separated: 2019.07.04 / 2019 07 04.
    # Without this rung such a file is Undated, and its descriptor normalizes
    # to a date-shaped token that build_filename then has to disambiguate.
    # Reading the date properly is the better outcome.
    re.compile(r"(?P<Y>\d{4})[.\s](?P<M>\d{2})[.\s](?P<D>\d{2})"),
    # Date only, compact and delimited: IMG-20190704-WA0001
    #
    # KNOWN FALSE POSITIVE, accepted deliberately: this matches ANY 8-digit run
    # bounded by - or _ that is calendar-valid and in range, so
    # "Order_20230615_001" reads as 2023-06-15. Accepted because in a photo
    # library a bounded _YYYYMMDD_ token is overwhelmingly a real date, and
    # anchoring this to camera prefixes would stop dating legitimate files like
    # "vacation_20190704_beach.jpg". It lands at DATE_FILENAME_PATTERN (10) --
    # the weakest non-zero rung, below every EXIF source -- and the source is
    # recorded, so a higher-ranked date can correct it later.
    re.compile(r"[-_](?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})[-_]"),
)


@dataclass(frozen=True)
class ResolvedDate:
    """A capture date together with the provenance that justifies its tier."""

    value: datetime | None
    tier: int
    source: str

    @property
    def date_str(self) -> str | None:
        """``YYYY-MM-DD`` for the filename, or None when undated."""
        return self.value.strftime("%Y-%m-%d") if self.value else None

    @property
    def folder(self) -> str:
        """Destination folder relative to the organized root."""
        if self.value is None:
            return UNDATED_FOLDER
        return f"{self.value.year:04d}/{self.value.year:04d}-{self.value.month:02d}"


_UNDATED = ResolvedDate(
    value=None, tier=tiers.DATE_NONE, source=tiers.DATE_SOURCE_NAMES[tiers.DATE_NONE]
)


def _plausible(dt: datetime) -> bool:
    return _MIN_YEAR <= dt.year <= _MAX_YEAR


def _parse_exif_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.strptime(raw.strip(), _EXIF_FORMAT)
    except ValueError:
        return None
    return dt if _plausible(dt) else None


def date_from_filename(stem: str) -> datetime | None:
    """Extract a capture date from a filename stem, or None."""
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        parts = match.groupdict()
        try:
            dt = datetime(
                int(parts["Y"]),
                int(parts["M"]),
                int(parts["D"]),
                int(parts.get("h") or 0),
                int(parts.get("m") or 0),
                int(parts.get("s") or 0),
            )
        except ValueError:
            continue  # e.g. month 13, or 07-32
        if _plausible(dt):
            return dt
    return None


def resolve_date(source_path: Path, exif_data: dict[str, Any]) -> ResolvedDate:
    """Resolve *source_path*'s capture date from EXIF, then from its filename.

    Rungs are tried highest-first and the first plausible hit wins.  File mtime
    is never consulted.
    """
    for field, tier in _EXIF_FIELDS:
        dt = _parse_exif_datetime(exif_data.get(field))
        if dt is not None:
            return ResolvedDate(value=dt, tier=tier, source=tiers.DATE_SOURCE_NAMES[tier])

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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_date_resolver.py -v`
Expected: PASS — 27 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/date_resolver.py tests/test_date_resolver.py
git commit -m "feat: date ladder with EXIF and filename rungs, mtime excluded"
```

---

### Task 6: Catalog — sources table, tier columns, enrichment queue

**Files:**
- Modify: `imageharbor/catalog.py:14-76` (schema), `imageharbor/catalog.py:111-201` (init and upsert)
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: nothing new.
- Produces on `Catalog`:
  - `record_source(sha256_b64url: str, source_path: str, size: int, mtime_ns: int) -> None`
  - `sources_for(sha256_b64url: str) -> list[sqlite3.Row]`
  - `upsert(...)` gains keyword args `date_value: str | None = None`, `date_tier: int = 0`, `date_source: str = "none"`, `descriptor_value: str = ""`, `descriptor_tier: int = 0`, `descriptor_source: str = "none"`
  - `set_placement(sha256_b64url: str, *, organized_path: str, date_value: str | None, date_tier: int, date_source: str, descriptor_value: str, descriptor_tier: int, descriptor_source: str) -> None`
  - `iter_unenriched(limit: int | None = None) -> list[sqlite3.Row]`
  - `mark_enriched(sha256_b64url: str, *, pcs_primary: str, pcs_name: str, secondary_tags: list[str], ai_caption: str, objects: list[str], ocr_text: str, model_version: str, scene: str) -> None`
  - `tiers_for(sha256_b64url: str) -> tuple[int, int]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog.py`:

```python
def test_record_source_accumulates_back_pointers(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a/one.jpg")
        cat.record_source("D1", "/a/one.jpg", 100, 111)
        cat.record_source("D1", "/b/two.jpg", 100, 222)
        cat.record_source("D1", "/c/three.jpg", 100, 333)
        rows = cat.sources_for("D1")
        assert {r["source_path"] for r in rows} == {"/a/one.jpg", "/b/two.jpg", "/c/three.jpg"}


def test_record_source_is_idempotent_and_updates_last_seen(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a/one.jpg")
        cat.record_source("D1", "/a/one.jpg", 100, 111)
        first = cat.sources_for("D1")[0]["first_seen_at"]
        cat.record_source("D1", "/a/one.jpg", 100, 111)
        rows = cat.sources_for("D1")
        assert len(rows) == 1
        assert rows[0]["first_seen_at"] == first


def test_upsert_stores_tiers(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(
            sha256_b64url="D1",
            original_path="/a/one.jpg",
            date_value="2019-07-04",
            date_tier=40,
            date_source="exif_original",
            descriptor_value="emmas-graduation",
            descriptor_tier=30,
            descriptor_source="human_filename",
        )
        assert cat.tiers_for("D1") == (40, 30)
        row = cat.get_by_sha256("D1")
        assert row["date_value"] == "2019-07-04"
        assert row["descriptor_source"] == "human_filename"


def test_tiers_for_unknown_digest_is_zero(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        assert cat.tiers_for("nope") == (0, 0)


def test_iter_unenriched_is_the_work_queue(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.upsert(sha256_b64url="D2", original_path="/b.jpg", organized_path="/lib/b.jpg")
        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D1", "D2"}

        cat.mark_enriched(
            "D1",
            pcs_primary="330",
            pcs_name="beach",
            secondary_tags=["sand"],
            ai_caption="a beach",
            objects=["sand"],
            ocr_text="",
            model_version="stub-1",
            scene="outdoor",
        )
        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D2"}


def test_iter_unenriched_respects_limit(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        for i in range(5):
            cat.upsert(
                sha256_b64url=f"D{i}",
                original_path=f"/{i}.jpg",
                organized_path=f"/lib/{i}.jpg",
            )
        assert len(cat.iter_unenriched(limit=2)) == 2


def test_iter_unenriched_excludes_quarantined_content(tmp_path):
    """Quarantine means "stop asking the model", so the row leaves the queue.

    failed_files is keyed by source path and photos by digest, so this only
    works if the exclusion joins through the sources table.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D1"}

        cat.record_file_failure("/a.jpg", 10, 111, "boom")
        cat.quarantine_file("/a.jpg")

        assert cat.iter_unenriched() == []


def test_iter_unenriched_excludes_content_quarantined_via_any_source(tmp_path):
    """Identical bytes fail identically, so one quarantined path condemns them."""
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_source("D1", "/b.jpg", 10, 222)

        cat.record_file_failure("/b.jpg", 10, 222, "boom")
        cat.quarantine_file("/b.jpg")

        assert cat.iter_unenriched() == []


def test_quarantine_survives_a_metadata_only_mtime_change(tmp_path):
    """A touch must not lift a quarantine whose bytes never changed.

    iter_unenriched correlates the exclusion on (path, size, mtime_ns). If
    record_source overwrote those stats on re-observation, a backup tool or a
    CIFS remount touching the file would silently re-admit known-poison content
    to the AI queue.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_file_failure("/a.jpg", 10, 111, "boom")
        cat.quarantine_file("/a.jpg")
        assert cat.iter_unenriched() == []

        # Same bytes, same digest, only the mtime moved.
        cat.record_source("D1", "/a.jpg", 10, 999)

        assert cat.iter_unenriched() == []


def test_record_source_freezes_stats_for_unchanged_content(tmp_path):
    """The row is keyed by digest, so its stats describe fixed content."""
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        first = cat.sources_for("D1")[0]["last_seen_at"]
        cat.record_source("D1", "/a.jpg", 10, 999)

        row = cat.sources_for("D1")[0]
        assert row["mtime_ns"] == 111
        assert row["size"] == 10
        assert row["last_seen_at"] >= first


def test_new_content_at_a_quarantined_path_re_enters_the_queue(tmp_path):
    """Quarantine is scoped to the exact bytes that failed.

    CLAUDE.md: a quarantined file is skipped thereafter "until its bytes
    change". Replacing a poison photo with a fixed one under the same filename
    must lift the exclusion for the new content -- and nothing else can, since
    record_file_failure's stale-stat reset only runs if the file is attempted.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_file_failure("/a.jpg", 10, 111, "boom")
        cat.quarantine_file("/a.jpg")
        assert cat.iter_unenriched() == []

        # Same path, new bytes: new digest, new size and mtime.
        cat.upsert(sha256_b64url="D2", original_path="/a.jpg", organized_path="/lib/b.jpg")
        cat.record_source("D2", "/a.jpg", 20, 222)

        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D2"}


def test_a_merely_failing_file_stays_in_the_queue(tmp_path):
    """Only QUARANTINED content leaves; a file still accruing failures retries."""
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/a.jpg")
        cat.record_source("D1", "/a.jpg", 10, 111)
        cat.record_file_failure("/a.jpg", 10, 111, "boom")  # not yet quarantined

        assert {r["sha256_b64url"] for r in cat.iter_unenriched()} == {"D1"}


def test_iter_unenriched_skips_rows_with_no_organized_copy(tmp_path):
    """The enrichment pass reads the ORGANIZED copy, not the source.

    A row with no organized_path has no file for it to open, so it must never
    reach the queue — `Path(None)` raises TypeError.
    """
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg")
        assert cat.iter_unenriched() == []


def test_set_placement_updates_path_and_tiers(tmp_path):
    from imageharbor.catalog import Catalog

    with Catalog(tmp_path / "c.db") as cat:
        cat.upsert(sha256_b64url="D1", original_path="/a.jpg", organized_path="/lib/Undated/x.jpg")
        cat.set_placement(
            "D1",
            organized_path="/lib/2019/2019-07/2019-07-04-beach_x.jpg",
            date_value="2019-07-04",
            date_tier=40,
            date_source="exif_original",
            descriptor_value="beach",
            descriptor_tier=20,
            descriptor_source="ai_subject",
        )
        row = cat.get_by_sha256("D1")
        assert row["organized_path"] == "/lib/2019/2019-07/2019-07-04-beach_x.jpg"
        assert cat.tiers_for("D1") == (40, 20)


def test_existing_catalog_gains_new_columns(tmp_path):
    """An older DB must open and upgrade without losing rows."""
    import sqlite3
    from imageharbor.catalog import Catalog

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "sha256_b64url TEXT NOT NULL UNIQUE, original_path TEXT NOT NULL, "
        "organized_path TEXT, pcs_version TEXT NOT NULL DEFAULT '1', "
        "pcs_primary TEXT NOT NULL DEFAULT '900', pcs_name TEXT NOT NULL DEFAULT 'miscellaneous', "
        "secondary_tags TEXT NOT NULL DEFAULT '[]', ai_caption TEXT NOT NULL DEFAULT '', "
        "objects TEXT NOT NULL DEFAULT '[]', ocr_text TEXT NOT NULL DEFAULT '', "
        "exif TEXT NOT NULL DEFAULT '{}', model_version TEXT NOT NULL DEFAULT 'unknown', "
        "processing_history TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, processed_at TEXT)"
    )
    conn.execute(
        "INSERT INTO photos (sha256_b64url, original_path, created_at) VALUES ('OLD', '/x.jpg', 'now')"
    )
    conn.commit()
    conn.close()

    with Catalog(db) as cat:
        assert cat.get_by_sha256("OLD") is not None
        assert cat.tiers_for("OLD") == (0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_catalog.py -v -k "source or tier or unenriched or placement or new_columns"`
Expected: FAIL — `AttributeError: 'Catalog' object has no attribute 'record_source'`

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/catalog.py`, append to the `_SCHEMA` string (before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS sources (
    sha256_b64url TEXT    NOT NULL,
    source_path   TEXT    NOT NULL,
    size          INTEGER,
    mtime_ns      INTEGER,
    first_seen_at TEXT    NOT NULL,
    last_seen_at  TEXT    NOT NULL,
    PRIMARY KEY (sha256_b64url, source_path)
);
CREATE INDEX IF NOT EXISTS idx_sources_digest ON sources(sha256_b64url);
```

Add module-level constant after `_SCHEMA`:

```python
# Columns added to `photos` after the original schema shipped. Applied
# additively on open so an existing catalog upgrades in place.
_ADDED_PHOTO_COLUMNS: tuple[tuple[str, str], ...] = (
    ("date_value", "TEXT"),
    ("date_tier", "INTEGER NOT NULL DEFAULT 0"),
    ("date_source", "TEXT NOT NULL DEFAULT 'none'"),
    ("descriptor_value", "TEXT NOT NULL DEFAULT ''"),
    ("descriptor_tier", "INTEGER NOT NULL DEFAULT 0"),
    ("descriptor_source", "TEXT NOT NULL DEFAULT 'none'"),
    ("scene", "TEXT NOT NULL DEFAULT ''"),
    ("enriched_at", "TEXT"),
)
```

In `Catalog.__init__`, after `self._conn.executescript(_SCHEMA)` and before `self._conn.commit()`:

```python
        self._ensure_photo_columns()
```

Add the method immediately after `__init__`:

```python
    def _ensure_photo_columns(self) -> None:
        """Add post-1.0 columns to `photos` if this DB predates them."""
        existing = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(photos)")
        }
        for name, ddl in _ADDED_PHOTO_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE photos ADD COLUMN {name} {ddl}")
                logger.debug("Catalog upgraded: added photos.%s", name)
```

Extend `upsert`'s signature with the six new keyword arguments (after `processing_history`):

```python
        date_value: str | None = None,
        date_tier: int = 0,
        date_source: str = "none",
        descriptor_value: str = "",
        descriptor_tier: int = 0,
        descriptor_source: str = "none",
```

Add the six values to the `params` tuple (append before `now,  # created_at`) and add the six column names to both the `INSERT INTO photos (...)` column list and the `ON CONFLICT ... DO UPDATE SET` list, following the existing style. Add six `?` placeholders to the `VALUES` clause.

Add these methods after `record_source_seen`:

```python
    def record_source(
        self, sha256_b64url: str, source_path: str, size: int, mtime_ns: int
    ) -> None:
        """Record that *source_path* holds the bytes identified by the digest.

        One row per distinct source path: this is the many-to-one back-pointer
        set that replaces a single `original_path`. `first_seen_at` is written
        once and never updated.

        `size` and `mtime_ns` are likewise written once. The row is keyed by
        digest, so its content is fixed by definition -- size is a function of
        that content and cannot change, and an mtime that moves without the
        bytes moving is metadata noise (a touch, a backup tool, a CIFS
        remount). Overwriting the stats on re-observation would let that noise
        silently lift a quarantine, because `iter_unenriched` correlates the
        exclusion on exactly this `(path, size, mtime_ns)` triple. Only
        `last_seen_at` moves.
        """
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO sources (
                sha256_b64url, source_path, size, mtime_ns, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256_b64url, source_path) DO UPDATE SET
                last_seen_at = excluded.last_seen_at
            """,
            (sha256_b64url, source_path, size, mtime_ns, now, now),
        )
        self._conn.commit()

    def sources_for(self, sha256_b64url: str) -> list[sqlite3.Row]:
        """All known source paths for a digest, oldest first."""
        return list(
            self._conn.execute(
                "SELECT * FROM sources WHERE sha256_b64url = ? ORDER BY first_seen_at",
                (sha256_b64url,),
            )
        )

    def tiers_for(self, sha256_b64url: str) -> tuple[int, int]:
        """Return ``(date_tier, descriptor_tier)``; ``(0, 0)`` if unknown."""
        row = self._conn.execute(
            "SELECT date_tier, descriptor_tier FROM photos WHERE sha256_b64url = ?",
            (sha256_b64url,),
        ).fetchone()
        if row is None:
            return (0, 0)
        return (row["date_tier"] or 0, row["descriptor_tier"] or 0)

    def set_placement(
        self,
        sha256_b64url: str,
        *,
        organized_path: str,
        date_value: str | None,
        date_tier: int,
        date_source: str,
        descriptor_value: str,
        descriptor_tier: int,
        descriptor_source: str,
    ) -> None:
        """Record a new organized path and the tiers that justified it."""
        self._conn.execute(
            """
            UPDATE photos SET
                organized_path = ?, date_value = ?, date_tier = ?, date_source = ?,
                descriptor_value = ?, descriptor_tier = ?, descriptor_source = ?,
                processed_at = ?
            WHERE sha256_b64url = ?
            """,
            (
                organized_path, date_value, date_tier, date_source,
                descriptor_value, descriptor_tier, descriptor_source,
                _now_iso(), sha256_b64url,
            ),
        )
        self._conn.commit()

    def iter_unenriched(self, limit: int | None = None) -> list[sqlite3.Row]:
        """Rows the AI enrichment pass has not yet processed.

        Excludes quarantined content. Quarantine means "stop asking the model
        about this one", so a quarantined file must leave the queue entirely --
        otherwise the poison file is re-described on every pass forever, which
        is the exact cost quarantine exists to eliminate.

        `failed_files` is keyed by source path while `photos` is keyed by
        digest, so the exclusion joins through `sources`. Content reachable from
        several paths is quarantined if ANY of them is: identical bytes fail
        identically, so one quarantined path condemns the content.

        The join also correlates on `size` and `mtime_ns`, which is what scopes
        the exclusion to the exact bytes that failed. Without it, a path reused
        by new content -- replacing a poison photo with a fixed one under the
        same filename -- would inherit the old quarantine forever, since the new
        digest's `sources` row shares that path. Nothing could lift it either:
        `record_file_failure`'s stale-stat reset only runs if the file is
        attempted, and it never would be. That would break the documented
        contract that a quarantined file is skipped only "until its bytes
        change".

        `--reclassify` deliberately bypasses this (it walks `iter_all`): asking
        again is the whole point of an explicit re-do.
        """
        sql = (
            "SELECT * FROM photos p WHERE p.enriched_at IS NULL "
            "AND p.organized_path IS NOT NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM sources s"
            "  JOIN failed_files f ON f.source_path = s.source_path"
            "   AND f.size = s.size AND f.mtime_ns = s.mtime_ns"
            "  WHERE s.sha256_b64url = p.sha256_b64url AND f.quarantined = 1"
            ") ORDER BY p.id"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return list(self._conn.execute(sql, params))

    def mark_enriched(
        self,
        sha256_b64url: str,
        *,
        pcs_primary: str,
        pcs_name: str,
        secondary_tags: list[str],
        ai_caption: str,
        objects: list[str],
        ocr_text: str,
        model_version: str,
        scene: str = "",
    ) -> None:
        """Store the AI's perception and stamp the row as enriched."""
        self._conn.execute(
            """
            UPDATE photos SET
                pcs_primary = ?, pcs_name = ?, secondary_tags = ?, ai_caption = ?,
                objects = ?, ocr_text = ?, model_version = ?, scene = ?,
                enriched_at = ?
            WHERE sha256_b64url = ?
            """,
            (
                pcs_primary, pcs_name, _json(secondary_tags), ai_caption,
                _json(objects), ocr_text, model_version, scene,
                _now_iso(), sha256_b64url,
            ),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_catalog.py -v`
Expected: PASS — all pre-existing catalog tests plus the 8 new ones.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: sources back-pointers, tier columns, and an enrichment queue"
```

---

### Task 7: Cumulative sidecars

**Files:**
- Modify: `imageharbor/sidecar.py` (full rewrite, 21 lines → ~90)
- Test: `tests/test_sidecar.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `SIDECAR_SCHEMA_VERSION = 1`; `sidecar_path_for(organized_path: Path) -> Path`; `read_sidecar(organized_path: Path) -> dict[str, Any]`; `merge_sidecar(organized_path: Path, updates: dict[str, Any]) -> Path`. `write_sidecar(organized_path, metadata)` is **removed**.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sidecar.py`:

```python
"""Tests for cumulative sidecar merging."""

import json

from imageharbor.sidecar import (
    SIDECAR_SCHEMA_VERSION,
    merge_sidecar,
    read_sidecar,
    sidecar_path_for,
)


def test_sidecar_path_appends_json_rather_than_replacing_a_suffix(tmp_path):
    """A stem containing dots must not lose part of its name."""
    img = tmp_path / "2019-07-04-v1.2_abc.jpg"
    assert sidecar_path_for(img).name == "2019-07-04-v1.2_abc.json"


def test_merge_creates_and_stamps_schema_version(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    data = read_sidecar(img)
    assert data["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert data["identity"]["sha256_b64url"] == "D1"


def test_merge_is_cumulative_across_runs(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}, "date": {"tier": 40}})
    merge_sidecar(img, {"classification": {"pcs_code": "330"}})
    data = read_sidecar(img)
    assert data["identity"]["sha256_b64url"] == "D1"
    assert data["date"]["tier"] == 40
    assert data["classification"]["pcs_code"] == "330"


def test_merge_deep_merges_nested_dicts(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1", "size": 10}})
    merge_sidecar(img, {"identity": {"size": 20}})
    data = read_sidecar(img)
    assert data["identity"] == {"sha256_b64url": "D1", "size": 20}


def test_merge_preserves_hand_added_keys(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    path = sidecar_path_for(img)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["my_note"] = "grandma's kitchen, not a beach"
    data["identity"]["my_field"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")

    # The second merge must REWRITE the same nested dict the hand edit lives in
    # -- an update that only touched an unrelated key would not prove anything.
    merge_sidecar(img, {"identity": {"sha256_b64url": "D2"}, "classification": {"pcs_code": "330"}})
    after = read_sidecar(img)
    assert after["my_note"] == "grandma's kitchen, not a beach"
    assert after["identity"]["my_field"] == 1
    assert after["identity"]["sha256_b64url"] == "D2"


def test_bytes_valued_exif_serializes_as_text_not_python_repr(tmp_path):
    """Real EXIF carries raw bytes (ExifVersion, SceneType, MakerNote).

    A bare default=str would not raise, but would write Python repr syntax
    into the file -- "b'0230'" instead of "0230" -- and a sidecar is meant to
    be portable and human-readable.
    """
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"exif": {"ExifVersion": b"0230", "MakerNote": b"\x00\xff"}})
    raw = sidecar_path_for(img).read_text(encoding="utf-8")
    assert "b'" not in raw
    assert read_sidecar(img)["exif"]["ExifVersion"] == "0230"


def test_a_scalar_replaced_by_a_dict_does_not_corrupt(tmp_path):
    """Type mismatches between runs replace cleanly rather than raising."""
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"date": "2019-07-04"})
    merge_sidecar(img, {"date": {"value": "2019-07-04", "tier": 40}})
    assert read_sidecar(img)["date"] == {"value": "2019-07-04", "tier": 40}


def test_lists_are_replaced_not_appended(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"sources": [{"path": "/a.jpg"}]})
    merge_sidecar(img, {"sources": [{"path": "/a.jpg"}, {"path": "/b.jpg"}]})
    assert len(read_sidecar(img)["sources"]) == 2


def test_read_of_a_missing_sidecar_is_empty(tmp_path):
    assert read_sidecar(tmp_path / "nope.jpg") == {}


def test_corrupt_sidecar_does_not_raise_and_is_rebuilt(tmp_path):
    img = tmp_path / "a.jpg"
    sidecar_path_for(img).write_text("{not json", encoding="utf-8")
    assert read_sidecar(img) == {}
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    assert read_sidecar(img)["identity"]["sha256_b64url"] == "D1"


def test_no_temp_file_is_left_behind(tmp_path):
    img = tmp_path / "a.jpg"
    merge_sidecar(img, {"identity": {"sha256_b64url": "D1"}})
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sidecar.py -v`
Expected: FAIL — `ImportError: cannot import name 'SIDECAR_SCHEMA_VERSION'`

- [ ] **Step 3: Write minimal implementation**

Replace the whole of `imageharbor/sidecar.py` with:

```python
"""Cumulative JSON sidecar files.

A sidecar accretes across runs rather than being rewritten: the facts pass
writes identity, sources, date, descriptor, and EXIF; the enrichment pass later
adds classification.  Unknown keys are preserved, so a hand-written correction
survives every subsequent run.

The catalog remains the source of truth; a sidecar is a portable projection of
it that travels with the image.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION = 1


def _json_default(o: Any) -> Any:
    """Fallback for values ``json.dumps`` cannot serialize natively.

    Real EXIF carries raw ``bytes`` (ExifVersion, SceneType, MakerNote) and
    other exotic types. A bare ``default=str`` would not raise, but it writes
    Python repr syntax into the file — ``"b'0230'"`` rather than ``"0230"`` —
    and a sidecar is meant to be a portable, human-readable projection.

    This deliberately mirrors ``catalog._json_default`` rather than importing
    it, so this module stays dependency-free apart from the standard library.
    Keep the two in sync.
    """
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).decode("utf-8", "replace")
    return str(o)


def sidecar_path_for(organized_path: Path) -> Path:
    """Return the sidecar path for *organized_path*.

    Composed explicitly from the stem rather than via ``with_suffix``, so the
    intent is stated in the code: the sidecar's name is the image's stem plus
    ``.json``, whatever dots the stem contains.
    """
    return organized_path.with_name(f"{organized_path.stem}.json")


def read_sidecar(organized_path: Path) -> dict[str, Any]:
    """Return the existing sidecar contents, or ``{}`` if absent or unreadable.

    A corrupt sidecar is reported and treated as empty rather than raising: it
    must never block an image that is already copied, verified, and cataloged.
    """
    path = sidecar_path_for(organized_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable sidecar %s (%s); treating as empty", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base*, returning a new dict.

    Nested dicts merge key-by-key so a partial update never drops a sibling
    field.  Lists and scalars replace wholesale -- callers that own a list
    (``sources``, ``history``) pass the complete value.
    """
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def merge_sidecar(organized_path: Path, updates: dict[str, Any]) -> Path:
    """Merge *updates* into the sidecar for *organized_path* and write it back.

    The write is atomic (temp file in the same directory, then ``os.replace``)
    so an interrupted run cannot leave a half-written sidecar.
    """
    path = sidecar_path_for(organized_path)
    merged = _deep_merge(read_sidecar(organized_path), updates)
    merged["schema_version"] = SIDECAR_SCHEMA_VERSION

    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sidecar.py -v`
Expected: PASS — 9 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/sidecar.py tests/test_sidecar.py
git commit -m "feat: cumulative sidecars with atomic writes and preserved keys"
```

---

### Task 8: Relocation — target paths, safe renames, self-healing

**Files:**
- Create: `imageharbor/relocate.py`
- Test: `tests/test_relocate.py`

**Interfaces:**
- Consumes: `date_resolver.ResolvedDate` (Task 5), `filename.build_filename` (Task 3), `hashing.verify_file` and `hashing.extract_digest_from_stem` (Task 2).
- Produces: `target_path(organized_dir: Path, date: ResolvedDate, descriptor: str, sha256_b64url: str, extension: str) -> Path`; `apply_relocation(old_path: Path, new_path: Path) -> None`; `find_by_digest(organized_dir: Path, sha256_b64url: str) -> Path | None`; `resolve_organized_path(organized_dir: Path, recorded: Path, sha256_b64url: str) -> Path | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_relocate.py`:

```python
"""Tests for target-path computation, safe renames, and self-healing."""

import hashlib
from datetime import datetime

import pytest

from imageharbor.date_resolver import ResolvedDate
from imageharbor.hashing import encode_base64url
from imageharbor.relocate import (
    apply_relocation,
    find_by_digest,
    resolve_organized_path,
    target_path,
)

_D = "qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI"

# A REAL content/digest pair. apply_relocation's "already done" branch verifies
# the destination against the digest embedded in its own filename, which is the
# project's core invariant. A fixture pairing arbitrary bytes with a placeholder
# digest would not exercise that branch honestly -- in a real organized tree a
# file's embedded digest always IS its actual hash.
_CONTENT = b"content"
_CONTENT_DIGEST = encode_base64url(hashlib.sha256(_CONTENT).digest())


def _dated():
    return ResolvedDate(value=datetime(2019, 7, 4), tier=40, source="exif_original")


def _undated():
    return ResolvedDate(value=None, tier=0, source="none")


def test_target_path_for_a_dated_file(tmp_path):
    path = target_path(tmp_path, _dated(), "beach", _D, "jpg")
    assert path == tmp_path / "2019" / "2019-07" / f"2019-07-04-beach_{_D}.jpg"


def test_target_path_for_an_undated_file(tmp_path):
    path = target_path(tmp_path, _undated(), "beach", _D, "jpg")
    assert path == tmp_path / "Undated" / f"beach_{_D}.jpg"


def test_target_path_with_no_descriptor(tmp_path):
    path = target_path(tmp_path, _undated(), "", _D, "jpg")
    assert path == tmp_path / "Undated" / f"{_D}.jpg"


def test_apply_relocation_moves_the_file(tmp_path):
    old = tmp_path / "Undated" / f"{_D}.jpg"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"content")
    new = tmp_path / "2019" / "2019-07" / f"2019-07-04_{_D}.jpg"

    apply_relocation(old, new)

    assert not old.exists()
    assert new.read_bytes() == b"content"


def test_apply_relocation_is_a_no_op_when_paths_match(tmp_path):
    path = tmp_path / f"{_D}.jpg"
    path.write_bytes(b"content")
    apply_relocation(path, path)
    assert path.read_bytes() == b"content"


def test_apply_relocation_refuses_to_clobber_different_content(tmp_path):
    """The destination's bytes do not match the digest in its own name."""
    old = tmp_path / f"a_{_CONTENT_DIGEST}.jpg"
    old.write_bytes(_CONTENT)
    new = tmp_path / f"b_{_CONTENT_DIGEST}.jpg"
    new.write_bytes(b"different")

    with pytest.raises(FileExistsError):
        apply_relocation(old, new)

    assert old.exists()
    assert new.read_bytes() == b"different"


def test_apply_relocation_tolerates_an_identical_destination(tmp_path):
    """A crash after rename but before the catalog update leaves this state.

    Both files carry the same digest in their names because they hold the same
    bytes -- content addressing guarantees it -- so the destination verifies
    against its own name and the relocation is treated as already done.
    """
    old = tmp_path / f"a_{_CONTENT_DIGEST}.jpg"
    old.write_bytes(_CONTENT)
    new = tmp_path / f"b_{_CONTENT_DIGEST}.jpg"
    new.write_bytes(_CONTENT)

    apply_relocation(old, new)

    assert not old.exists()
    assert new.read_bytes() == _CONTENT


def test_apply_relocation_rejects_a_destination_that_fails_its_own_digest(tmp_path):
    """A corrupted destination must never be accepted as "already done".

    This is what byte-comparing old against new would miss: two files can be
    identical to each other while neither matches the identity its name claims.
    """
    old = tmp_path / f"a_{_CONTENT_DIGEST}.jpg"
    old.write_bytes(b"corrupt")
    new = tmp_path / f"b_{_CONTENT_DIGEST}.jpg"
    new.write_bytes(b"corrupt")

    with pytest.raises(FileExistsError):
        apply_relocation(old, new)

    assert old.exists()


def test_find_by_digest_locates_a_moved_file(tmp_path):
    target = tmp_path / "2019" / "2019-07" / f"2019-07-04-beach_{_D}.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"content")
    assert find_by_digest(tmp_path, _D) == target


def test_find_by_digest_ignores_a_sidecar_carrying_the_same_digest(tmp_path):
    """A sidecar's stem is its image's stem, so it carries the same digest.

    If self-healing returned the sidecar, the catalog would record a .json as
    the image's organized_path.

    The stems differ here deliberately, because that is the real scenario: the
    image was renamed and its sidecar left behind under the OLD name. With
    IDENTICAL stems this test would be vacuous -- "jpg" sorts before "json", so
    sorted() alone would pick the image and the test would pass even with the
    extension filter removed. Here the orphan sorts FIRST, so only the filter
    can save it.
    """
    root = tmp_path / "2019" / "2019-07"
    root.mkdir(parents=True)
    orphan = root / f"aaa-old-name_{_D}.json"
    orphan.write_text("{}", encoding="utf-8")
    img = root / f"zzz-new-name_{_D}.jpg"
    img.write_bytes(b"content")

    # Guards this test against silently becoming vacuous again.
    assert sorted([orphan, img])[0] == orphan

    assert find_by_digest(tmp_path, _D) == img


def test_find_by_digest_returns_none_when_absent(tmp_path):
    assert find_by_digest(tmp_path, _D) is None


def test_resolve_organized_path_prefers_the_recorded_path(tmp_path):
    recorded = tmp_path / "Undated" / f"{_D}.jpg"
    recorded.parent.mkdir(parents=True)
    recorded.write_bytes(b"content")
    assert resolve_organized_path(tmp_path, recorded, _D) == recorded


def test_resolve_organized_path_self_heals_after_an_interrupted_rename(tmp_path):
    """The catalog points at the old path; the file is already at the new one."""
    recorded = tmp_path / "Undated" / f"{_D}.jpg"
    actual = tmp_path / "2019" / "2019-07" / f"2019-07-04_{_D}.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"content")

    assert resolve_organized_path(tmp_path, recorded, _D) == actual


def test_resolve_organized_path_returns_none_when_truly_gone(tmp_path):
    assert resolve_organized_path(tmp_path, tmp_path / "gone.jpg", _D) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_relocate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.relocate'`

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/relocate.py`:

```python
"""Target-path computation and safe relocation within the organized tree.

Organized copies belong to ImageHarbor, so moving and renaming them is legal --
unlike originals, which are never touched.  A relocation is applied to the
filesystem *before* the catalog is updated; a crash in between is recoverable
because the file is content-addressed, so `resolve_organized_path` can find it
by digest and repair the row.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .date_resolver import ResolvedDate
from .discovery import SUPPORTED_EXTENSIONS
from .filename import build_filename
from .hashing import extract_digest_from_stem, verify_file

logger = logging.getLogger(__name__)


def target_path(
    organized_dir: Path,
    date: ResolvedDate,
    descriptor: str,
    sha256_b64url: str,
    extension: str,
) -> Path:
    """Where a file with these facts belongs, under *organized_dir*."""
    name = build_filename(date.date_str, descriptor, sha256_b64url, extension)
    return organized_dir / date.folder / name


def apply_relocation(old_path: Path, new_path: Path) -> None:
    """Move *old_path* to *new_path*.

    A destination that already holds byte-identical content is accepted (this
    is the state left by a crash between the rename and the catalog update);
    a destination holding *different* content raises rather than clobbering.
    """
    if old_path == new_path:
        return

    new_path.parent.mkdir(parents=True, exist_ok=True)

    if new_path.exists():
        digest = extract_digest_from_stem(new_path.stem)
        if digest and verify_file(new_path, digest):
            logger.debug("Destination already present and verified: %s", new_path)
            old_path.unlink(missing_ok=True)
            return
        raise FileExistsError(
            f"Refusing to overwrite {new_path}: it holds different content"
        )

    os.replace(old_path, new_path)
    logger.info("Relocated %s -> %s", old_path.name, new_path)


def find_by_digest(organized_dir: Path, sha256_b64url: str) -> Path | None:
    """Locate an IMAGE anywhere under *organized_dir* by its embedded digest.

    This is the self-healing path: content addressing means a file that moved
    is never actually lost.

    Restricted to supported image extensions deliberately. A JSON sidecar's
    name is its image's stem with ``.json`` substituted, so the sidecar's stem
    carries the SAME digest. Without this filter, an image that moved while its
    sidecar stayed behind yields two glob matches, and a stale sidecar can be
    returned as though it were the organized copy -- after which the catalog
    records a ``.json`` as the image's path. Results are sorted so the choice is
    deterministic rather than filesystem-order dependent.
    """
    for candidate in sorted(organized_dir.rglob(f"*{sha256_b64url}.*")):
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if candidate.is_file() and extract_digest_from_stem(candidate.stem) == sha256_b64url:
            return candidate
    return None


def resolve_organized_path(
    organized_dir: Path, recorded: Path, sha256_b64url: str
) -> Path | None:
    """Return where the file actually is, repairing a stale recorded path.

    Returns None if the file is genuinely missing from the organized tree.
    """
    if recorded.exists():
        return recorded
    found = find_by_digest(organized_dir, sha256_b64url)
    if found is not None:
        logger.info(
            "Catalog path was stale (%s); found by digest at %s", recorded, found
        )
    return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_relocate.py -v`
Expected: PASS — 12 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/relocate.py tests/test_relocate.py
git commit -m "feat: relocation with content-addressed self-healing"
```

---

### Task 9: Rewrite the pipeline as the facts pass

**Files:**
- Modify: `imageharbor/pipeline.py` (remove classifier/taxonomy/concept_map; rewrite `_do_process`, `_update_catalog`, `_write_sidecar`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `date_resolver.resolve_date`, `descriptor.resolve_descriptor`, `relocate.target_path`, `sidecar.merge_sidecar`, `catalog.record_source`/`upsert`, `tiers.*`.
- Produces: `Pipeline(source_dir, organized_dir, catalog, duplicates_dir=None, write_sidecars=False, dry_run=False)` — **the `classifier` parameter is removed**. `Pipeline.run(recursive: bool = True) -> PipelineStats` — **the `breaker` parameter is removed** (the facts pass has no AI to fail). `ProcessResult` and `PipelineStats` are unchanged.

- [ ] **Step 1: Write the failing test**

Replace the AI-dependent tests in `tests/test_pipeline.py`. Keep any test that only asserts hashing/dedup/copy/verify behavior, updating constructor calls to drop `classifier=`. Add:

```python
from imageharbor import tiers
from imageharbor.catalog import Catalog
from imageharbor.pipeline import Pipeline


def _make_image(path, content=b"fake-image-bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_facts_pass_makes_no_ai_call(tmp_path, monkeypatch):
    """The facts pass must not import or invoke a classifier."""
    import imageharbor.ai_classifier as ai

    def boom(*args, **kwargs):
        raise AssertionError("the facts pass called the AI")

    monkeypatch.setattr(ai.StubClassifier, "describe", boom)

    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "Emma's graduation.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
    assert stats.copied == 1


def test_human_named_undated_file_lands_in_undated(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "Emma's graduation.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        result = stats.results[0]
        assert result.organized_path.parent == dest / "Undated"
        assert result.organized_path.name.startswith("emmas-graduation_")
        assert cat.tiers_for(result.sha256_b64url) == (
            tiers.DATE_NONE,
            tiers.DESC_HUMAN_FILENAME,
        )


def test_camera_named_file_gets_no_descriptor(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "IMG_1234.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        result = stats.results[0]
        assert result.organized_path.stem == result.sha256_b64url
        assert cat.tiers_for(result.sha256_b64url) == (tiers.DATE_NONE, tiers.DESC_NONE)


def test_filename_date_places_the_file(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "IMG_20190704_123456.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        result = stats.results[0]
        assert result.organized_path.parent == dest / "2019" / "2019-07"
        assert result.organized_path.name.startswith("2019-07-04_")


def test_duplicates_record_back_pointers_and_copy_once(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "a" / "IMG_1234.jpg", b"same")
    _make_image(src / "b" / "IMG_5678.jpg", b"same")
    _make_image(src / "c" / "Emma's graduation.jpg", b"same")

    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat).run()
        assert stats.copied == 1
        assert stats.duplicates == 2
        digest = stats.results[0].sha256_b64url
        assert len(cat.sources_for(digest)) == 3


def test_rerunning_the_facts_pass_changes_nothing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "IMG_20190704_123456.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        first = Pipeline(src, dest, cat).run()
        paths_after_first = sorted(p.name for p in dest.rglob("*.jpg"))
        second = Pipeline(src, dest, cat).run()
        paths_after_second = sorted(p.name for p in dest.rglob("*.jpg"))

    assert first.copied == 1
    assert second.copied == 0
    assert second.duplicates == 1
    assert paths_after_first == paths_after_second


def test_sidecar_records_facts_and_sources(tmp_path):
    from imageharbor.sidecar import read_sidecar

    src, dest = tmp_path / "src", tmp_path / "dest"
    _make_image(src / "Emma's graduation.jpg")
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat, write_sidecars=True).run()
    data = read_sidecar(stats.results[0].organized_path)
    assert data["descriptor"]["tier"] == tiers.DESC_HUMAN_FILENAME
    assert data["date"]["source"] == "none"
    assert len(data["sources"]) == 1
    assert "classification" not in data
```

These tests deliberately use byte content that is not a real image. `read_exif` returns `{}` for unreadable files rather than raising, so dates come from the filename rung — which is exactly the path under test. Do not add an EXIF-writing dependency.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_filename' from 'imageharbor.filename'`

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/pipeline.py`:

Replace the import block (lines 11-22) with:

```python
from .catalog import Catalog
from .date_resolver import resolve_date
from .descriptor import resolve_descriptor
from .discovery import discover_images
from .exif_reader import read_exif
from .hashing import compute_sha256_b64url, verify_file
from .relocate import target_path
from .sidecar import merge_sidecar
```

Replace `Pipeline.__init__` (lines 94-114) with:

```python
    def __init__(
        self,
        source_dir: Path,
        organized_dir: Path,
        catalog: Catalog,
        duplicates_dir: Path | None = None,
        write_sidecars: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_dir = source_dir
        self.organized_dir = organized_dir
        self.catalog = catalog
        self.duplicates_dir = duplicates_dir
        self.write_sidecars = write_sidecars
        self.dry_run = dry_run
        self._dry_run_seen: set[str] = set()
```

Update the class docstring to drop the `classifier` parameter and state that this pass makes no AI calls.

Replace `run` (lines 120-152) with:

```python
    def run(self, recursive: bool = True) -> PipelineStats:
        """Process all images under :attr:`source_dir`.

        This pass never calls an AI backend, so there is no breaker to feed and
        no systemic-outage abort: it runs at disk speed and completes.
        """
        stats = PipelineStats()
        self._dry_run_seen.clear()
        for image_path in discover_images(self.source_dir, recursive=recursive):
            result = self._process_one(image_path)
            stats.record(result)
            _log_result(result)
        return stats
```

Replace `process_file` (lines 154-160) with:

```python
    def process_file(self, image_path: Path) -> ProcessResult:
        """Process a single image file and return its result."""
        result = self._process_one(image_path)
        _log_result(result)
        return result
```

Replace `_do_process` (lines 178-299) with:

```python
    def _do_process(self, source_path: Path) -> ProcessResult:
        # Step 1: hash original
        sha256_b64url = compute_sha256_b64url(source_path)
        stat = source_path.stat()

        # Step 2: duplicate detection. A duplicate still records a back-pointer
        # -- the same bytes reachable from another path is information, not
        # noise, and a better-named path can upgrade the file on a later pass.
        if self.catalog.is_known(sha256_b64url) or (
            self.dry_run and sha256_b64url in self._dry_run_seen
        ):
            if not self.dry_run:
                self.catalog.mark_duplicate(sha256_b64url, str(source_path))
                self.catalog.record_source(
                    sha256_b64url, str(source_path), stat.st_size, stat.st_mtime_ns
                )
                if self.duplicates_dir:
                    self._copy_to_duplicates(source_path, sha256_b64url)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="duplicate",
            )

        if self.dry_run:
            self._dry_run_seen.add(sha256_b64url)
            return ProcessResult(
                source_path=source_path,
                sha256_b64url=sha256_b64url,
                status="copied",
                organized_path=None,
            )

        # Step 3: EXIF (best effort; returns {} rather than raising)
        exif_data = read_exif(source_path)

        # Step 4: facts -- date decides the folder, descriptor decides the name.
        date = resolve_date(source_path, exif_data)
        descriptor = resolve_descriptor(source_path)

        # Step 5: destination
        extension = source_path.suffix.lstrip(".").lower()
        organized_path = target_path(
            self.organized_dir, date, descriptor.value, sha256_b64url, extension
        )

        # Step 6: copy
        organized_path.parent.mkdir(parents=True, exist_ok=True)
        if organized_path.exists() and verify_file(organized_path, sha256_b64url):
            logger.debug(
                "Destination already present and verified, skipping copy: %s",
                organized_path,
            )
        else:
            shutil.copy2(str(source_path), str(organized_path))

            # Step 7: verify before anything is recorded
            if not verify_file(organized_path, sha256_b64url):
                organized_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Integrity check failed after copying {source_path} -> {organized_path}"
                )

        # Step 8: catalog
        self.catalog.upsert(
            sha256_b64url=sha256_b64url,
            original_path=str(source_path),
            organized_path=str(organized_path),
            exif=exif_data,
            date_value=date.date_str,
            date_tier=date.tier,
            date_source=date.source,
            descriptor_value=descriptor.value,
            descriptor_tier=descriptor.tier,
            descriptor_source=descriptor.source,
            processing_history=[
                {
                    "event": "facts",
                    "source": str(source_path),
                    "destination": str(organized_path),
                }
            ],
        )
        self.catalog.record_source(
            sha256_b64url, str(source_path), stat.st_size, stat.st_mtime_ns
        )

        # Step 9: optional sidecar. A sidecar failure must never fail an image
        # that is already copied, verified, and catalogued.
        if self.write_sidecars:
            try:
                self._write_sidecar(
                    organized_path, sha256_b64url, stat.st_size, extension,
                    date, descriptor, exif_data,
                )
            except Exception:
                logger.warning(
                    "Failed to write sidecar for %s; image is organized and catalogued",
                    organized_path,
                    exc_info=True,
                )

        return ProcessResult(
            source_path=source_path,
            sha256_b64url=sha256_b64url,
            status="copied",
            organized_path=organized_path,
        )
```

Delete `_classes` and `_update_catalog`. Replace `_write_sidecar` (lines 346-373) with:

```python
    def _write_sidecar(
        self,
        organized_path: Path,
        sha256_b64url: str,
        size: int,
        extension: str,
        date: "ResolvedDate",
        descriptor: "ResolvedDescriptor",
        exif_data: dict[str, Any],
    ) -> None:
        sources = [
            {
                "path": row["source_path"],
                "first_seen": row["first_seen_at"],
                "last_seen": row["last_seen_at"],
            }
            for row in self.catalog.sources_for(sha256_b64url)
        ]
        merge_sidecar(
            organized_path,
            {
                "identity": {
                    "sha256_b64url": sha256_b64url,
                    "size": size,
                    "ext": extension,
                },
                "sources": sources,
                "date": {
                    "value": date.value.isoformat() if date.value else None,
                    "tier": date.tier,
                    "source": date.source,
                },
                "descriptor": {
                    "value": descriptor.value,
                    "tier": descriptor.tier,
                    "source": descriptor.source,
                },
                "exif": exif_data,
            },
        )
```

Add to the `TYPE_CHECKING` block at the top:

```python
if TYPE_CHECKING:
    from .date_resolver import ResolvedDate
    from .descriptor import ResolvedDescriptor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

`tests/test_cli.py` and `tests/test_watcher.py` will still fail (they construct `Pipeline(classifier=...)`); Tasks 11 and 12 fix them.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline becomes the AI-free facts pass"
```

---

### Task 10: The enrichment pass

**Files:**
- Create: `imageharbor/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: `catalog.iter_unenriched`/`mark_enriched`/`set_placement`/`tiers_for`/`sources_for`, `tiers.is_upgrade`, `relocate.*`, `date_resolver.ResolvedDate`, `taxonomy.Taxonomy`, `concept_map`, `ai_classifier.AIClassifier`, `circuit_breaker.CircuitBreaker`.
- Produces: `EnrichStats` dataclass with `total`, `enriched`, `renamed`, `errors`, `aborted: bool`; `enrich_library(catalog, organized_dir, classifier, *, write_sidecars=False, breaker=None, limit=None, reclassify=False) -> EnrichStats`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_enrich.py`:

```python
"""Tests for the AI enrichment pass and its non-degradation guarantee."""

import pytest

from imageharbor import tiers
from imageharbor.ai_classifier import AIClassifier, ContentDescription, StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrich import enrich_library
from imageharbor.pipeline import Pipeline


class FixedClassifier(StubClassifier):
    """A classifier that always reports the same subject."""

    def __init__(self, subject="beach"):
        self._subject = subject

    def describe(self, image_path, exif_data=None):
        return ContentDescription(
            primary_subject=self._subject,
            scene="outdoor",
            objects=["sand"],
            caption="a beach",
            tags=["sand"],
            ocr_text="",
            model_version="fixed-1",
        )


class BrokenClassifier(AIClassifier):
    """Every call fails, as during a backend outage."""

    def describe(self, image_path, exif_data=None):
        raise RuntimeError("backend down")


def _make(tmp_path, name, content=b"fake-image-bytes"):
    src = tmp_path / "src"
    path = src / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return src


def _facts(tmp_path, name, content=b"fake-image-bytes"):
    src = _make(tmp_path, name, content)
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    stats = Pipeline(src, dest, cat, write_sidecars=True).run()
    return cat, dest, stats.results[0]


def test_enrichment_names_a_camera_named_file(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    assert result.organized_path.name.startswith("2019-07-04_")

    stats = enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    assert stats.enriched == 1
    assert stats.renamed == 1
    row = cat.get_by_sha256(result.sha256_b64url)
    assert row["organized_path"].endswith(f"2019-07-04-beach_{result.sha256_b64url}.jpg")
    assert cat.tiers_for(result.sha256_b64url) == (
        tiers.DATE_FILENAME_PATTERN,
        tiers.DESC_AI_SUBJECT,
    )
    cat.close()


def test_enrichment_never_displaces_a_human_filename(tmp_path):
    cat, dest, result = _facts(tmp_path, "Emma's graduation.jpg")
    before = result.organized_path.name

    stats = enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    assert stats.enriched == 1
    assert stats.renamed == 0
    row = cat.get_by_sha256(result.sha256_b64url)
    assert row["organized_path"].endswith(before)
    # The classification is still recorded -- only the *name* is protected.
    assert row["pcs_primary"]
    cat.close()


def test_a_second_enrichment_run_is_a_no_op(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier())
    after_first = cat.get_by_sha256(result.sha256_b64url)["organized_path"]

    second = enrich_library(cat, dest, FixedClassifier(subject="mountain"))

    assert second.total == 0
    assert cat.get_by_sha256(result.sha256_b64url)["organized_path"] == after_first
    cat.close()


def test_a_backend_outage_degrades_nothing(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    before_path = result.organized_path
    before_tiers = cat.tiers_for(result.sha256_b64url)

    stats = enrich_library(cat, dest, BrokenClassifier())

    assert stats.errors == 1
    assert stats.enriched == 0
    assert before_path.exists()
    assert cat.get_by_sha256(result.sha256_b64url)["organized_path"] == str(before_path)
    assert cat.tiers_for(result.sha256_b64url) == before_tiers
    cat.close()


def test_a_tripped_breaker_aborts_the_pass(tmp_path):
    from imageharbor.circuit_breaker import CircuitBreaker

    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    (src / "IMG_3.jpg").write_bytes(b"three")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    breaker = CircuitBreaker(trip_threshold=2, backoff_base=1.0, backoff_cap=1.0)
    stats = enrich_library(cat, dest, BrokenClassifier(), breaker=breaker)

    assert stats.aborted is True
    assert stats.errors == 2
    cat.close()


def test_enrichment_adds_classification_to_the_sidecar(tmp_path):
    from imageharbor.sidecar import read_sidecar

    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    new_path = cat.get_by_sha256(result.sha256_b64url)["organized_path"]
    from pathlib import Path

    data = read_sidecar(Path(new_path))
    assert data["classification"]["primary_subject"] == "beach"
    # Facts written by the earlier pass survive the merge.
    assert data["identity"]["sha256_b64url"] == result.sha256_b64url
    assert data["date"]["tier"] == tiers.DATE_FILENAME_PATTERN
    cat.close()


def test_enrichment_self_heals_a_stale_catalog_path(tmp_path):
    """Simulates a crash between the rename and the catalog update."""
    import shutil

    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    old = result.organized_path
    moved = old.parent / f"2019-07-04-moved_{result.sha256_b64url}.jpg"
    shutil.move(str(old), str(moved))

    stats = enrich_library(cat, dest, FixedClassifier())

    assert stats.errors == 0
    assert stats.enriched == 1
    cat.close()


def test_self_heal_without_upgrade_carries_the_sidecar(tmp_path):
    """A stale path repaired at a tier that blocks renaming must keep its sidecar.

    A human-named file cannot be renamed by the AI pass, so it takes the
    self-heal branch rather than the rename branch. If that branch left the
    sidecar behind, the merge would rebuild it from an empty base and lose
    every fact the first pass recorded.
    """
    import shutil
    from pathlib import Path

    from imageharbor.sidecar import read_sidecar, sidecar_path_for

    cat, dest, result = _facts(tmp_path, "Emma's graduation.jpg")
    old = result.organized_path
    assert read_sidecar(old)["identity"]["sha256_b64url"] == result.sha256_b64url

    # Relocate the file and its sidecar is left behind by an external actor.
    moved = old.parent / f"moved-{old.name}"
    shutil.move(str(old), str(moved))

    enrich_library(cat, dest, FixedClassifier(), write_sidecars=True)

    healed = Path(cat.get_by_sha256(result.sha256_b64url)["organized_path"])
    data = read_sidecar(healed)
    assert data["identity"]["sha256_b64url"] == result.sha256_b64url
    assert data["descriptor"]["tier"] == tiers.DESC_HUMAN_FILENAME
    assert data["classification"]["primary_subject"] == "beach"
    assert not sidecar_path_for(old).exists()
    cat.close()


def test_a_local_failure_does_not_wedge_the_pass(tmp_path):
    """An exception after perception must not escape or block later rows.

    The queue is ordered by id and a row that raises is never marked enriched
    or failed, so an escaping exception would crash on the same row forever.
    """
    src = _make(tmp_path, "IMG_1.jpg", b"one")
    (src / "IMG_2.jpg").write_bytes(b"two")
    dest = tmp_path / "dest"
    cat = Catalog(tmp_path / "c.db")
    Pipeline(src, dest, cat).run()

    calls = {"n": 0}
    real_mark = cat.mark_enriched

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("catalog is busy")
        return real_mark(*args, **kwargs)

    cat.mark_enriched = flaky

    stats = enrich_library(cat, dest, FixedClassifier())

    assert stats.total == 2
    assert stats.errors == 1
    assert stats.enriched == 1
    assert len(stats.failed) == 1  # the failure is visible to quarantine
    cat.close()


def test_reclassify_skips_rows_with_no_organized_copy(tmp_path):
    """--reclassify walks the whole catalog, including rows iter_unenriched hides."""
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    cat.upsert(sha256_b64url="ORPHAN", original_path="/gone.jpg")

    stats = enrich_library(cat, dest, FixedClassifier(), reclassify=True)

    assert stats.total == 1  # the orphan is skipped, not crashed on
    cat.close()


def test_reclassify_forces_a_second_pass(tmp_path):
    cat, dest, result = _facts(tmp_path, "IMG_20190704_123456.jpg")
    enrich_library(cat, dest, FixedClassifier(subject="beach"))

    stats = enrich_library(cat, dest, FixedClassifier(subject="mountain"), reclassify=True)

    assert stats.total == 1
    assert cat.get_by_sha256(result.sha256_b64url)["pcs_name"]
    cat.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_enrich.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.enrich'`

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/enrich.py`:

```python
"""The AI enrichment pass.

Runs after the facts pass, independently and resumably.  It reads the
*organized copy* rather than the source: the bytes are verified identical, so
enrichment works when the source volume is unmounted.

Enrichment can only ever improve a file.  It writes classification to the
catalog and sidecar unconditionally, but renames the file only when
:func:`~imageharbor.tiers.is_upgrade` says the result is strictly better --
so an AI subject can never displace a human-authored filename, and a repeated
run is a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from . import concept_map, tiers
from .ai_classifier import AIClassifier
from .catalog import Catalog
from .date_resolver import ResolvedDate
from .relocate import apply_relocation, resolve_organized_path, target_path
from .sidecar import merge_sidecar
from .taxonomy import Taxonomy

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class EnrichStats:
    """Aggregated statistics for an enrichment pass."""

    total: int = 0
    enriched: int = 0
    renamed: int = 0
    errors: int = 0
    aborted: bool = False
    # Digests that failed this pass. The watcher feeds these to poison-file
    # reconciliation: with no AI in the facts pass, this is now the ONLY source
    # of the per-file failure signal quarantine depends on.
    failed: list[str] = field(default_factory=list)


def _date_from_row(row) -> ResolvedDate:
    """Rebuild a ResolvedDate from stored catalog columns."""
    raw = row["date_value"]
    value = None
    if raw:
        try:
            value = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            logger.warning("Unparseable stored date %r; treating as undated", raw)
    tier = row["date_tier"] or tiers.DATE_NONE
    return ResolvedDate(
        value=value,
        tier=tier,
        source=row["date_source"] or tiers.DATE_SOURCE_NAMES[tiers.DATE_NONE],
    )


def enrich_library(
    catalog: Catalog,
    organized_dir: Path,
    classifier: AIClassifier,
    *,
    write_sidecars: bool = False,
    breaker: CircuitBreaker | None = None,
    limit: int | None = None,
    reclassify: bool = False,
) -> EnrichStats:
    """Describe and classify organized images that have not been enriched yet.

    When a *breaker* is supplied, a systemic run of failures trips it and
    aborts the pass -- continuing would only churn a dead backend.
    """
    stats = EnrichStats()
    taxonomy = Taxonomy(catalog)
    taxonomy.ensure_seeded()

    if reclassify:
        # iter_all has no organized_path filter, unlike iter_unenriched -- whose
        # guard exists precisely because Path(None) raises TypeError. --reclassify
        # walks the WHOLE catalog, so it must re-apply that guard itself rather
        # than rely on today's single insert path always populating the column.
        rows = [r for r in catalog.iter_all() if r["organized_path"]]
        if limit is not None:
            rows = rows[:limit]
    else:
        rows = catalog.iter_unenriched(limit)

    classes = [(n.code, n.label) for n in taxonomy.children(None)]

    for row in rows:
        stats.total += 1
        digest = row["sha256_b64url"]
        recorded = Path(row["organized_path"])

        actual = resolve_organized_path(organized_dir, recorded, digest)
        if actual is None:
            logger.error("Organized file missing for %s (%s)", digest, recorded)
            stats.errors += 1
            stats.failed.append(digest)
            continue

        try:
            content = classifier.describe(actual, {})
        except Exception as exc:
            logger.warning("Enrichment failed for %s: %s", actual.name, exc)
            stats.errors += 1
            stats.failed.append(digest)
            if breaker is not None:
                breaker.record_failure()
                if breaker.is_open():
                    logger.error(
                        "AI backend appears down — aborting enrichment after "
                        "%d consecutive failures",
                        breaker.trip_threshold,
                    )
                    stats.aborted = True
                    break
            continue

        if breaker is not None:
            breaker.record_success()

        # INDENT EVERYTHING BELOW, from "Organization:" through the sidecar
        # merge, into this try block. Everything from here on is LOCAL work --
        # taxonomy, catalog, filesystem -- and must be isolated per row for two
        # reasons. First, a failure here is not a backend outage, so it must
        # never feed the breaker. Second, the queue is ordered by id and a row
        # that raises is never marked enriched OR failed, so an escaping
        # exception would crash on the same row every subsequent pass and
        # permanently block every row behind it. This mirrors
        # Pipeline._process_one, which wraps its whole per-file body for the
        # same reason. An escape would also bypass stats.failed entirely,
        # silently disabling the poison-file quarantine that consumes it.
        #
        #     try:
        #         <organization / mark_enriched / rename / sidecar>
        #     except Exception as exc:
        #         logger.exception(
        #             "Post-perception enrichment failed for %s: %s",
        #             actual.name, exc,
        #         )
        #         stats.errors += 1
        #         stats.failed.append(digest)
        #         continue

        # Organization: our code picks the class; the AI is only a fallback.
        cls = concept_map.class_for(
            content.primary_subject, content.objects, content.scene, catalog
        )
        if cls is None:
            cls = classifier.pick_class(content, classes)
            concept_map.remember(catalog, content.primary_subject, cls)

        pcs_code = taxonomy.resolve_or_create(
            cls, content.primary_subject, adjudicator=classifier.adjudicate
        )
        node = taxonomy.get(pcs_code)
        pcs_name = node.label if node else content.primary_subject

        catalog.mark_enriched(
            digest,
            pcs_primary=pcs_code,
            pcs_name=pcs_name,
            secondary_tags=content.tags,
            ai_caption=content.caption,
            objects=content.objects,
            ocr_text=content.ocr_text,
            model_version=content.model_version,
            scene=content.scene,
        )
        stats.enriched += 1

        # Naming: only if strictly better.
        date = _date_from_row(row)
        old = (date.tier, row["descriptor_tier"] or tiers.DESC_NONE)
        new = (date.tier, tiers.DESC_AI_SUBJECT)
        final_path = actual

        if tiers.is_upgrade(old, new):
            from .filename import normalize_descriptor

            descriptor = normalize_descriptor(content.primary_subject)
            proposed = target_path(
                organized_dir, date, descriptor, digest, actual.suffix.lstrip(".").lower()
            )
            try:
                # Filesystem first, catalog second: a crash in between is
                # recovered by digest lookup on the next pass.
                apply_relocation(actual, proposed)
                catalog.set_placement(
                    digest,
                    organized_path=str(proposed),
                    date_value=date.date_str,
                    date_tier=date.tier,
                    date_source=date.source,
                    descriptor_value=descriptor,
                    descriptor_tier=tiers.DESC_AI_SUBJECT,
                    descriptor_source=tiers.DESC_SOURCE_NAMES[tiers.DESC_AI_SUBJECT],
                )
                final_path = proposed
                stats.renamed += 1
            except OSError as exc:
                logger.warning("Rename failed for %s: %s", actual.name, exc)
        elif str(actual) != row["organized_path"]:
            # Self-healed a stale path without otherwise changing anything.
            # The sidecar must follow the file here too. Without this, a file
            # whose descriptor tier already blocks an AI rename (a human
            # filename) but which was relocated externally would leave its
            # sidecar orphaned at the old path -- and the merge below would
            # then build a fresh one at the new location from an empty base,
            # silently dropping the facts pass's identity/sources/date/
            # descriptor data.
            old_sidecar = sidecar_path_for(recorded)
            new_sidecar = sidecar_path_for(actual)
            if old_sidecar.exists() and old_sidecar != new_sidecar:
                old_sidecar.replace(new_sidecar)
            catalog.set_placement(
                digest,
                organized_path=str(actual),
                date_value=date.date_str,
                date_tier=date.tier,
                date_source=date.source,
                descriptor_value=row["descriptor_value"] or "",
                descriptor_tier=row["descriptor_tier"] or tiers.DESC_NONE,
                descriptor_source=row["descriptor_source"] or "none",
            )

        if write_sidecars:
            try:
                merge_sidecar(
                    final_path,
                    {
                        "classification": {
                            "pcs_code": pcs_code,
                            "folder_path": taxonomy.folder_path(pcs_code),
                            "primary_subject": content.primary_subject,
                            "scene": content.scene,
                            "caption": content.caption,
                            "objects": content.objects,
                            "tags": content.tags,
                            "ocr_text": content.ocr_text,
                            "model_version": content.model_version,
                        }
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to update sidecar for %s", final_path, exc_info=True
                )

    return stats
```

Note: when a rename happens, the old sidecar keeps the old name. Add immediately after `stats.renamed += 1`:

```python
                # Carry the sidecar along with the file it describes.
                from .sidecar import sidecar_path_for

                old_sidecar = sidecar_path_for(actual)
                if old_sidecar.exists():
                    old_sidecar.replace(sidecar_path_for(proposed))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_enrich.py -v`
Expected: PASS — 8 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/enrich.py tests/test_enrich.py
git commit -m "feat: AI enrichment pass with tier-gated renaming"
```

---

### Task 11: Two-phase watcher

**Files:**
- Modify: `imageharbor/watcher.py`
- Test: `tests/test_watcher.py`

**Interfaces:**
- Consumes: `pipeline.Pipeline` (Task 9), `enrich.enrich_library` (Task 10).
- Produces: `watch(...)` gains an `enrich_enabled: bool = True` parameter and calls `enrich_library` after each facts sweep. The breaker governs only the enrichment phase.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_watcher.py`:

```python
def test_facts_phase_runs_even_when_the_breaker_is_open(tmp_path, monkeypatch):
    """A dead AI backend must not stop the library being organized."""
    from imageharbor.catalog import Catalog
    from imageharbor.circuit_breaker import CircuitBreaker
    from imageharbor import watcher

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    breaker = CircuitBreaker(trip_threshold=1, backoff_base=1.0, backoff_cap=1.0)
    breaker.record_failure()
    assert breaker.is_open()

    calls = {"enrich": 0}

    def fake_enrich(*args, **kwargs):
        calls["enrich"] += 1
        from imageharbor.enrich import EnrichStats

        return EnrichStats()

    monkeypatch.setattr(watcher, "enrich_library", fake_enrich)

    with Catalog(tmp_path / "c.db") as cat:
        watcher.run_once(src, dest, cat, classifier=None, breaker=breaker)

    assert (dest / "2019" / "2019-07").exists()
    assert calls["enrich"] == 0  # breaker open -> enrichment skipped


def test_enrich_phase_runs_after_the_facts_phase(tmp_path, monkeypatch):
    from imageharbor.catalog import Catalog
    from imageharbor.ai_classifier import StubClassifier
    from imageharbor import watcher

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    order = []
    real_enrich = watcher.enrich_library

    def tracking_enrich(*args, **kwargs):
        order.append("enrich")
        return real_enrich(*args, **kwargs)

    monkeypatch.setattr(watcher, "enrich_library", tracking_enrich)

    with Catalog(tmp_path / "c.db") as cat:
        watcher.run_once(src, dest, cat, classifier=StubClassifier(), breaker=None)

    assert order == ["enrich"]
    assert (dest / "2019" / "2019-07").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_watcher.py -v -k "facts_phase or enrich_phase"`
Expected: FAIL — `AttributeError: module 'imageharbor.watcher' has no attribute 'run_once'`

- [ ] **Step 3: Write minimal implementation**

Read the current `imageharbor/watcher.py` in full before editing. Then:

Add the import `from .enrich import enrich_library` at the top.

Extract the existing per-pass body into a new function, keeping the existing poison/quarantine reconciliation attached to the enrichment phase:

```python
def run_once(
    source: Path,
    dest: Path,
    catalog: Catalog,
    *,
    classifier: AIClassifier | None,
    breaker: CircuitBreaker | None,
    duplicates_dir: Path | None = None,
    write_sidecars: bool = False,
    recursive: bool = True,
    enrich_enabled: bool = True,
) -> tuple[PipelineStats, EnrichStats | None]:
    """One full sweep: the facts phase, then the enrichment phase.

    The facts phase never consults the breaker -- it makes no AI calls, so a
    dead backend has no bearing on whether the library can be organized.
    """
    # The facts leg MUST go through run_pass, not Pipeline.run(). run_pass
    # consults catalog.source_is_unchanged (a cheap os.stat) and only calls
    # process_file for new or changed files. Pipeline.run() re-hashes every file
    # it walks -- a full read of the whole library on every pass. Over the CIFS
    # NAS mount this watcher exists to serve, that is the exact cost the module
    # was written to avoid.
    facts = run_pass(pipeline, catalog, source, recursive=recursive)

    enrich_stats = None
    if enrich_enabled and classifier is not None:
        if breaker is not None and breaker.is_open():
            logger.info("Breaker open — skipping the enrichment phase this pass")
        else:
            enrich_stats = enrich_library(
                catalog, dest, classifier,
                write_sidecars=write_sidecars,
                breaker=breaker,
            )
    return facts, enrich_stats
```

Rewire the existing `watch` loop to call `run_once` per pass, keep the existing backoff/half-open logic driving the breaker between passes, and update the stop-summary to report facts and enrichment counts separately. Update the `Pipeline(...)` construction anywhere else in the file to drop the `classifier=` argument.

**`run_pass` must stop feeding the circuit breaker.** It currently calls
`breaker.record_success()` / `record_failure()` from per-file results. That
wiring existed because the only failures were AI failures. The facts pass has no
AI, so every error it can now return is an I/O error — a permissions problem, an
unreadable file, a full disk. Feeding those to the AI breaker would let a
filesystem fault masquerade as a backend outage and back the watcher off for
fifteen minutes waiting for a backend that was never sick. Remove the breaker
calls from `run_pass` entirely; the breaker is now driven solely by
`enrich_library`.

**Poison quarantine moves to the enrichment phase.** The existing reconciliation
counted a file toward quarantine when `pipeline.process_file()` returned
`status == "error"` — and the only thing that ever produced that error was the
AI classifier raising inside `describe()`. The facts pass makes no AI calls, so
that signal is now permanently absent and quarantine would never fire.

This is a re-wiring, not a redesign. `watcher.py` keeps `_reconcile_poison`,
`_copy_to_quarantine`, and the failed-file buffer exactly as they are, together
with the safety property that failures observed while the breaker is OPEN never
count — a backend outage must not mis-quarantine good files. What changes is
the source of failures: they now come from the enrichment phase's per-file
errors rather than from facts-pass results.

`enrich_library` must therefore report which digests failed. Extend `EnrichStats`
with `failed: list[str]` (digests, appended in the same place `stats.errors` is
incremented) and have `run_once` feed those to the reconciliation.

Quarantine now means **"stop asking the model about this one"**, not "set this
file aside". A file that cannot be described is still fully organized, verified,
and catalogued by the facts pass — only its enrichment is abandoned.

- [ ] **Step 3a: Rewrite `tests/test_poison.py` for the new signal path**

Two tests fail after Task 9 because they simulate poison via a raising
classifier reaching the facts pass: `test_poison_file_quarantined_after_k_healthy_passes`
and `test_quarantine_copies_to_dir_when_set`. Rewrite both to drive failures
through `enrich_library` instead — the classifier raises during enrichment, the
facts pass still succeeds, and the file is quarantined after
`--poison-max-fails` healthy enrichment attempts.

Keep the existing safety test asserting that failures during a breaker-tripped
outage never count toward quarantine; that property is unchanged and is the
reason the reconciliation exists. Add one test asserting a quarantined file is
still present and verifiable in the organized tree — quarantine must not remove
or relocate it.

Add one more, asserting quarantine actually **stops the asking**: after a file
is quarantined, a further pass must make zero `describe()` calls for it. Count
the calls rather than asserting a status — the bookkeeping can look right while
the model is still being hit every pass, which is the whole cost quarantine
exists to eliminate.

```python
def test_a_quarantined_file_is_never_described_again(tmp_path):
    """Quarantine means "stop asking the model about this one"."""
    calls: list[str] = []

    class Counting(_FailsForContent):
        def describe(self, image_path, exif_data=None):
            calls.append(image_path.name)
            return super().describe(image_path, exif_data)

    # ... drive passes until the file is quarantined, then:
    before = len(calls)
    watcher.run_once(src, dest, cat, classifier=Counting(...), breaker=breaker)
    assert len(calls) == before  # not one further AI call
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_watcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add imageharbor/watcher.py tests/test_watcher.py
git commit -m "feat: two-phase watcher — facts always, enrichment when healthy"
```

---

### Task 12: CLI — move AI flags to a new `enrich` verb

**Files:**
- Modify: `imageharbor/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.Pipeline`, `enrich.enrich_library`, `watcher.watch`.
- Produces: `imageharbor process` with no AI/breaker/poison flags; `imageharbor enrich --dest --catalog --ai [--ai-*] [--breaker-*] [--limit] [--reclassify] [--sidecar/--no-sidecar]`; `imageharbor watch` unchanged in flags.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_process_no_longer_accepts_ai_flags(tmp_path):
    from click.testing import CliRunner
    from imageharbor.cli import main

    src = tmp_path / "src"
    src.mkdir()
    result = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(tmp_path / "d"), "--ai", "stub"]
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_process_organizes_without_any_ai(tmp_path):
    from click.testing import CliRunner
    from imageharbor.cli import main

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    result = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert (dest / "2019" / "2019-07").exists()


def test_enrich_command_exists_and_reports(tmp_path):
    from click.testing import CliRunner
    from imageharbor.cli import main

    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"bytes")
    dest = tmp_path / "dest"

    runner = CliRunner()
    runner.invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    result = runner.invoke(
        main, ["enrich", "--dest", str(dest), "--ai", "stub"]
    )
    assert result.exit_code == 0, result.output
    assert "enriched" in result.output.lower()


def test_enrich_accepts_limit_and_reclassify(tmp_path):
    from click.testing import CliRunner
    from imageharbor.cli import main

    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(
        main,
        ["enrich", "--dest", str(dest), "--ai", "stub", "--limit", "1", "--reclassify"],
    )
    assert result.exit_code == 0, result.output
```

Update any existing CLI test that passes `--ai` to `process` so it targets `enrich` instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v -k "process_no_longer or enrich_command"`
Expected: FAIL — `enrich` is not a registered command; `process` still accepts `--ai`.

- [ ] **Step 3: Write minimal implementation**

Read `imageharbor/cli.py` in full before editing. Then:

1. From the `process` command, delete the `--ai`, `--ai-base-url`, `--ai-model`, `--ai-timeout`, `--openai-key`, and `--breaker-threshold` options and their corresponding function parameters. Remove the `_build_classifier` and `_build_breaker` calls from `process` and the `breaker=` argument to `pipeline.run()`. Keep `--source`, `--dest`, `--catalog`, `--duplicates`, `--sidecar/--no-sidecar`, `--dry-run`, `--no-recursive`.

2. Add a new `enrich` command after `process`, reusing the option style already in the file (including `envvar=` where `watch` uses one):

```python
@main.command()
@click.option("--dest", required=True, type=click.Path(path_type=Path),
              envvar="IMAGEHARBOR_DEST", help="Root of the organized library.")
@click.option("--catalog", "catalog_path", type=click.Path(path_type=Path),
              envvar="IMAGEHARBOR_CATALOG",
              help="Catalog path. Defaults to <dest>/catalog.db.")
@click.option("--sidecar/--no-sidecar", default=False,
              envvar="IMAGEHARBOR_SIDECAR", help="Update JSON sidecars.")
@click.option("--ai", type=click.Choice(["stub", "openai"]), default="stub",
              show_default=True, envvar="IMAGEHARBOR_AI", help="Classifier backend.")
@click.option("--ai-base-url", envvar="IMAGEHARBOR_AI_BASE_URL", default=None,
              help="OpenAI-compatible base URL.")
@click.option("--ai-model", envvar="IMAGEHARBOR_AI_MODEL", default=None,
              help="Vision model name.")
@click.option("--ai-timeout", envvar="IMAGEHARBOR_AI_TIMEOUT", default=60.0,
              show_default=True, type=float, help="Per-call timeout in seconds.")
@click.option("--openai-key", envvar=["IMAGEHARBOR_AI_API_KEY", "OPENAI_API_KEY"],
              default=None, help="API key for the AI backend.")
@click.option("--breaker-threshold", envvar="IMAGEHARBOR_BREAKER_THRESHOLD",
              default=5, show_default=True, type=int,
              help="Consecutive AI failures before aborting. 0 disables.")
@click.option("--limit", default=None, type=int,
              help="Process at most this many images.")
@click.option("--reclassify", is_flag=True, default=False,
              help="Re-run classification on already-enriched images.")
def enrich(dest, catalog_path, sidecar, ai, ai_base_url, ai_model, ai_timeout,
           openai_key, breaker_threshold, limit, reclassify):
    """Describe and classify already-organized images.

    Reads the organized copies, so the original source volume need not be
    mounted. Safe to interrupt and re-run: a file is only ever renamed when
    the result is strictly better.
    """
    catalog_file = catalog_path or (dest / "catalog.db")
    classifier = _build_classifier(ai, ai_base_url, ai_model, ai_timeout, openai_key)
    breaker = _build_breaker(breaker_threshold, 60.0, 900.0)

    with Catalog(catalog_file) as cat:
        stats = enrich_library(
            cat, dest, classifier,
            write_sidecars=sidecar,
            breaker=breaker,
            limit=limit,
            reclassify=reclassify,
        )

    click.echo(
        f"Enriched: {stats.enriched}  Renamed: {stats.renamed}  "
        f"Errors: {stats.errors}  Total: {stats.total}"
    )
    if stats.aborted:
        click.echo("Aborted early: the AI backend appears to be down.")
        sys.exit(1)
```

3. Add `from .enrich import enrich_library` to the imports.

4. In `watch`, update the `watcher.watch(...)` call to match Task 11's signature.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

Then the whole suite: `uv run pytest -q`
Expected: PASS — every test in the project.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/cli.py tests/test_cli.py
git commit -m "feat: split AI flags out of process into a new enrich command"
```

---

### Task 13: Cross-cutting guarantee tests

**Files:**
- Test: `tests/test_monotonicity.py` (create)

**Interfaces:**
- Consumes: everything. This task adds no production code.

- [ ] **Step 1: Write the failing test**

Create `tests/test_monotonicity.py`:

```python
"""End-to-end guarantees: re-runs converge and never degrade a file."""

from pathlib import Path

from imageharbor.ai_classifier import AIClassifier, ContentDescription, StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrich import enrich_library
from imageharbor.pipeline import Pipeline


class Fixed(StubClassifier):
    def __init__(self, subject):
        self._subject = subject

    def describe(self, image_path, exif_data=None):
        return ContentDescription(
            primary_subject=self._subject, scene="s", objects=[], caption="c",
            tags=[], ocr_text="", model_version="fixed-1",
        )


class Broken(AIClassifier):
    def describe(self, image_path, exif_data=None):
        raise RuntimeError("down")


def _snapshot(dest: Path) -> set[str]:
    return {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}


def _library(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"one")
    (src / "Emma's graduation.jpg").write_bytes(b"two")
    (src / "IMG_1234.jpg").write_bytes(b"three")
    return src, tmp_path / "dest"


def test_facts_then_enrich_reaches_a_fixed_point(tmp_path):
    src, dest = _library(tmp_path)
    with Catalog(tmp_path / "c.db") as cat:
        Pipeline(src, dest, cat, write_sidecars=True).run()
        enrich_library(cat, dest, Fixed("beach"), write_sidecars=True)
        after_first_cycle = _snapshot(dest)

        for _ in range(3):
            Pipeline(src, dest, cat, write_sidecars=True).run()
            enrich_library(cat, dest, Fixed("mountain"), write_sidecars=True)

        assert _snapshot(dest) == after_first_cycle


def test_an_outage_between_good_runs_loses_nothing(tmp_path):
    src, dest = _library(tmp_path)
    with Catalog(tmp_path / "c.db") as cat:
        Pipeline(src, dest, cat, write_sidecars=True).run()
        enrich_library(cat, dest, Fixed("beach"), write_sidecars=True)
        healthy = _snapshot(dest)

        # reclassify=True is REQUIRED, not incidental. After the healthy pass
        # every row has enriched_at set, so the default query finds nothing and
        # Broken() would never be called -- the test would pass without the
        # outage ever happening. Assert it genuinely failed, too.
        broken = enrich_library(cat, dest, Broken(), write_sidecars=True, reclassify=True)
        assert broken.errors > 0
        assert broken.enriched == 0

        Pipeline(src, dest, cat, write_sidecars=True).run()
        enrich_library(cat, dest, Broken(), write_sidecars=True, reclassify=True)

        assert _snapshot(dest) == healthy


def test_a_library_organized_with_no_ai_at_all_is_complete(tmp_path):
    """The facts pass alone must produce a fully organized, verified library."""
    from imageharbor.hashing import verify_pcs_file

    src, dest = _library(tmp_path)
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat, write_sidecars=True).run()

    assert stats.copied == 3
    images = [p for p in dest.rglob("*.jpg")]
    assert len(images) == 3
    assert all(verify_pcs_file(p) for p in images)


def test_a_better_named_duplicate_upgrades_the_descriptor(tmp_path):
    """Dedup does real organizing work, not just copy-skipping."""
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "IMG_1234.jpg").write_bytes(b"same")
    dest = tmp_path / "dest"

    with Catalog(tmp_path / "c.db") as cat:
        first = Pipeline(src, dest, cat).run()
        digest = first.results[0].sha256_b64url
        assert first.results[0].organized_path.stem == digest

        (src / "b" / "Emma's graduation.jpg").write_bytes(b"same")
        Pipeline(src, dest, cat).run()

        assert len(cat.sources_for(digest)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monotonicity.py -v`
Expected: The first three tests PASS (the machinery exists). `test_a_better_named_duplicate_upgrades_the_descriptor` FAILS — the facts pass records the back-pointer but does not yet re-evaluate tiers for a known digest.

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/pipeline.py`, inside `_do_process`'s duplicate branch, after `self.catalog.record_source(...)`, add the upgrade check:

```python
                self._maybe_upgrade_from_duplicate(source_path, sha256_b64url)
```

And add the method:

```python
    def _maybe_upgrade_from_duplicate(
        self, source_path: Path, sha256_b64url: str
    ) -> None:
        """Re-evaluate a known file's tiers against a newly-seen source path.

        Identical bytes mean identical EXIF, but not identical filenames: the
        same photo found at a better-named path can supply a date or a
        descriptor the first copy lacked.
        """
        row = self.catalog.get_by_sha256(sha256_b64url)
        if row is None or not row["organized_path"]:
            return

        date = resolve_date(source_path, {})
        descriptor = resolve_descriptor(source_path)
        old = (row["date_tier"] or 0, row["descriptor_tier"] or 0)
        new = (max(old[0], date.tier), max(old[1], descriptor.tier))
        if not tiers.is_upgrade(old, new):
            return

        recorded = Path(row["organized_path"])
        actual = resolve_organized_path(self.organized_dir, recorded, sha256_b64url)
        if actual is None:
            logger.warning("Cannot upgrade %s: organized file missing", sha256_b64url)
            return

        best_date = date if date.tier >= old[0] else _date_from_row(row)
        best_descriptor = (
            descriptor.value if descriptor.tier >= old[1] else (row["descriptor_value"] or "")
        )
        proposed = target_path(
            self.organized_dir, best_date, best_descriptor, sha256_b64url,
            actual.suffix.lstrip(".").lower(),
        )
        try:
            apply_relocation(actual, proposed)
        except OSError as exc:
            logger.warning("Upgrade rename failed for %s: %s", actual.name, exc)
            return

        old_sidecar = sidecar_path_for(actual)
        if old_sidecar.exists():
            old_sidecar.replace(sidecar_path_for(proposed))

        self.catalog.set_placement(
            sha256_b64url,
            organized_path=str(proposed),
            date_value=best_date.date_str,
            date_tier=best_date.tier,
            date_source=best_date.source,
            descriptor_value=best_descriptor,
            descriptor_tier=new[1],
            descriptor_source=(
                descriptor.source if descriptor.tier >= old[1]
                else (row["descriptor_source"] or "none")
            ),
        )
        logger.info("Upgraded %s from a better-named duplicate", proposed.name)
```

Add the needed imports to `pipeline.py`:

```python
from . import tiers
from .enrich import _date_from_row
from .relocate import apply_relocation, resolve_organized_path, target_path
from .sidecar import merge_sidecar, sidecar_path_for
```

If importing `_date_from_row` from `enrich` creates a circular import (`enrich` imports `pipeline`? it does not — verify), move `_date_from_row` into `date_resolver.py` as a public `date_from_row(row)` and import it from there in both modules.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monotonicity.py -v`
Expected: PASS — 4 tests

Then the full suite: `uv run pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_monotonicity.py imageharbor/pipeline.py
git commit -m "feat: upgrade tiers from a better-named duplicate; add guarantee tests"
```

---

### Task 14: Documentation and rebuild runbook

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/rebuild.md`
- Modify: `docs/genesis-roadmap.md` (add a pointer note only)
- Modify: `README.md`

**Interfaces:**
- Consumes: the finished implementation.
- Produces: no code.

- [ ] **Step 1: Update CLAUDE.md**

Rewrite these sections to match the implementation:

- **Commands table** — add `uv run imageharbor enrich --dest DEST --ai openai`; note that `process` makes no AI calls.
- **Architecture** — replace the single-pipeline flow with the two-pass flow. New spine for the facts pass: `hash → dedup (+ back-pointer, + duplicate upgrade) → EXIF → resolve date → resolve descriptor → target path → copy → verify → catalog → sidecar`. New spine for enrichment: `unenriched rows → describe → concept_map/pick_class → taxonomy resolve → catalog → tier-gated rename → sidecar`.
- **Module responsibilities** — add `tiers.py`, `date_resolver.py`, `descriptor.py`, `relocate.py`, `enrich.py`; rewrite the `pipeline.py`, `sidecar.py`, and `catalog.py` entries.
- **Critical invariants** — replace the "Folder paths come from `taxonomy.folder_path(code)`" invariant with:
  - Placement comes from `date_resolver.ResolvedDate.folder`; PCS lives in the catalog and sidecar, not in the path or filename.
  - A file is renamed or moved only when `tiers.is_upgrade` returns True. Never add a code path that renames unconditionally.
  - `taxonomy.folder_path(code)` is still used, but only to record a classification path *inside the sidecar*.
  - The digest is still located by counting 43 characters back from the end of the stem. Note that `extract_digest_from_stem` no longer validates a PCS prefix and instead validates the Base64url character class.
  - File mtime must never be added to the date ladder.

- [ ] **Step 2: Write the rebuild runbook**

Create `docs/rebuild.md` covering: stopping the hpz440 container; copying `learned_concepts` and `taxonomy` from the old catalog into a fresh one (`sqlite3` `ATTACH` + `INSERT INTO … SELECT`); running `process` against the same source with a new dest; spot-checking the tree; running `enrich`; verifying with `imageharbor verify`; and only then deleting the old tree. State explicitly that the source is never modified, so an aborted rebuild costs nothing but disk.

- [ ] **Step 3: Add the roadmap pointer**

At the top of `docs/genesis-roadmap.md`, under the title, add:

```markdown
> **Historical document.** The PCS folder tree and PCS-prefixed filenames
> described below were superseded on 2026-08-11 by the date-derived tree and
> the `[<date>][-<descriptor>]_<digest>` filename grammar. See
> `docs/superpowers/specs/2026-08-11-facts-first-pipeline-design.md`. The
> integrity, immutability, and resumability requirements are unchanged and
> still authoritative.
```

- [ ] **Step 4: Expand README.md**

Replace the 3-line README with a short orientation: what ImageHarbor is, the two passes, the filename grammar with one example, the four main commands, and links to `CLAUDE.md`, the design spec, and `docs/deploy-docker.md`.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -q`
Expected: PASS — the full suite.

Run: `uv run imageharbor --help`
Expected: `process`, `enrich`, `watch`, `verify`, and `catalog` all listed.

```bash
git add CLAUDE.md README.md docs/rebuild.md docs/genesis-roadmap.md
git commit -m "docs: document the facts-first pipeline and the rebuild procedure"
```

---

## Self-Review

**Spec coverage:** Every spec section maps to a task — two passes (9, 10, 11), tier system (1), monotonicity rule (1, 10, 13), layout and grammar (2, 3), camera patterns (4), date ladder (5), `sources` table and new columns (6), cumulative sidecars (7), crash safety (8), the four property suites (13, plus 4/5's fixture tables), CLI surface (12), rebuild-not-migration (14), docs consequences (14). The reserved `DATE_EXTERNAL_SIDECAR` rung is defined in Task 1 and deliberately unused, matching the spec's non-goal.

**Known risks flagged for the implementer:**

1. **Task 13's `_date_from_row` import** may create a cycle if `enrich.py` ever imports `pipeline.py`. It currently does not; the fallback (move the helper to `date_resolver.py`) is written into the step.
2. **Task 11 requires reading `watcher.py` in full first** — the existing backoff/half-open loop and poison reconciliation must be preserved, and this plan does not reproduce that code.
3. **Task 12 requires reading `cli.py` in full first** for the same reason.
4. **Task 3 leaves the suite red** until Task 9. This is expected and stated in the step; do not "fix" it by keeping `generate_filename`.
5. **`test_poison.py` and `test_concept_map.py`** may need constructor updates if they build a `Pipeline` with `classifier=`. Fix them in whichever task first turns them red.
