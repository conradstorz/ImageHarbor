# Self-Extending PCS Taxonomy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded PCS taxonomy with a guard-railed, self-extending registry: the AI returns a label + parent, the system owns all numbering (3 bounded levels under a fixed 9-class spine, `~`-extended overflow), with normalize+adjudicate dedup and a merge escape hatch.

**Architecture:** The taxonomy becomes data in a new `taxonomy` table in the catalog DB, wrapped by a new `imageharbor/taxonomy.py` (`Taxonomy` class: seed, folder-path, numbering, resolve-or-create, merge). The classifier stops emitting codes and instead returns `(top_parent, label, sub_parent?)`; the pipeline passes it a taxonomy snapshot and resolves the label to a code. Filenames carry a string code matching `^\d+(~\d+)*$`.

**Tech Stack:** Python 3.10+, SQLite, Click, Pillow, optional `openai`. `uv` for dev/test. Standard library `difflib` for fuzzy matching (no new deps).

## Global Constraints

- Python floor `>=3.10`; `from __future__ import annotations` in new modules; no newer-only syntax.
- Runtime deps limited to `Pillow` + `click`; `openai` only via the extra, imported lazily. No new third-party deps (`difflib`/`re`/`json` are stdlib).
- **PCS codes are strings matching `^\d+(~\d+)*$`** — plain integers for the common 3-level case; a `~N` suffix only for overflow/depth. **Never a dot.** `~` is filesystem-safe and absent from the base64url alphabet.
- The **9 top-level classes are fixed** (`100`…`900`), never renumbered; growth happens beneath. Codes are **append-only** — never renumbered or deleted (merge aliases instead).
- Preserve every existing invariant: originals read-only; copy→verify→catalog ordering; the 43-char digest is located by counting back from the end of the stem (unchanged).
- The taxonomy lives in the catalog DB (local volume), seeded from `pcs.PCS_CATEGORIES`, reconstructable from folder names.
- Tests offline/deterministic — no network, no real `openai`; the AI adjudicator is mocked or `None`.
- `uv run pytest`; do NOT chain shell commands with `&&`.
- Match existing code/test style (plain pytest functions/`Test*` classes, `tmp_path`, section-comment headers).

---

## File Structure

**Create:**
- `imageharbor/taxonomy.py` — `TaxonomyNode`, `Taxonomy` (seed, folder_path, numbering/mint, resolve_or_create, merge, snapshot, normalization).
- `tests/test_taxonomy.py`

**Modify:**
- `imageharbor/hashing.py` — `extract_digest_from_stem` PCS validation → `^\d+(~\d+)*$`.
- `imageharbor/filename.py` — `generate_filename`/`parse_filename` handle string codes; `ParsedFilename.pcs_code: str`.
- `imageharbor/catalog.py` — `taxonomy` table in `_SCHEMA`; taxonomy CRUD methods; `pcs_primary` column → TEXT.
- `imageharbor/ai_classifier.py` — `PhotoClassification` carries `(top_parent, label, sub_parent)`; `classify(..., taxonomy_snapshot)`; `adjudicate`; Stub + OpenAI updated.
- `imageharbor/pipeline.py` — hold a `Taxonomy`, pass snapshot to classify, resolve code, use `folder_path`, store string code.
- `imageharbor/cli.py` — `catalog list` string formatting; nothing else structural.
- `imageharbor/pcs.py` — keep seed data (`PCS_CATEGORIES`, `PCS_VERSION`); drop `parent_folder_name`/`sub_folder_name` (moved to `taxonomy.folder_path`).
- Tests: `tests/test_hashing.py`, `tests/test_filename.py`, `tests/test_catalog.py`, `tests/test_ai_classifier.py`, `tests/test_pcs.py`, `tests/test_pipeline.py`, `tests/test_cli.py`.

---

## Task 1: Filename & hashing accept string `~` codes

**Files:**
- Modify: `imageharbor/hashing.py` (`extract_digest_from_stem`)
- Modify: `imageharbor/filename.py` (`generate_filename`, `parse_filename`, `ParsedFilename`)
- Test: `tests/test_hashing.py`, `tests/test_filename.py`

**Interfaces:**
- Produces: `generate_filename(pcs_code: str, descriptor: str, sha256_b64url: str, extension: str) -> str`; `parse_filename(...) -> {"pcs_code": str, ...}`; codes validate `^\d+(~\d+)*$`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_hashing.py`:
```python
def test_extract_digest_accepts_tilde_code() -> None:
    digest = "A" * 43
    assert extract_digest_from_stem(f"540~1-holiday_{digest}") == digest

def test_extract_digest_rejects_dotted_code() -> None:
    digest = "A" * 43
    assert extract_digest_from_stem(f"540.1-holiday_{digest}") is None
```
Add to `tests/test_filename.py`:
```python
def test_generate_and_parse_tilde_code() -> None:
    name = generate_filename("540~1", "christmas eve", "A" * 43, "jpg")
    assert name.startswith("540~1-")
    parsed = parse_filename(name)
    assert parsed is not None
    assert parsed["pcs_code"] == "540~1"
    assert parsed["descriptor"] == "christmas-eve"

def test_parse_plain_code_is_string() -> None:
    parsed = parse_filename(f"330-beach_{'A' * 43}.jpg")
    assert parsed is not None
    assert parsed["pcs_code"] == "330"  # str, not int

def test_parse_rejects_dotted_code() -> None:
    assert parse_filename(f"540.1-x_{'A' * 43}.jpg") is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_hashing.py tests/test_filename.py -k "tilde or dotted or plain_code_is_string" -v`
Expected: FAIL (dotted currently accepted / `pcs_code` currently `int`).

- [ ] **Step 3: Implement**

In `imageharbor/hashing.py`, add near the top:
```python
import re

_PCS_CODE_RE = re.compile(r"^\d+(~\d+)*$")
```
In `extract_digest_from_stem`, replace the PCS validation line:
```python
    pcs_part = prefix[:dash_idx]
    if not (pcs_part.isascii() and pcs_part.isdigit()) or not prefix[dash_idx + 1:]:
        return None
```
with:
```python
    pcs_part = prefix[:dash_idx]
    if not (pcs_part.isascii() and _PCS_CODE_RE.match(pcs_part)) or not prefix[dash_idx + 1:]:
        return None
```

In `imageharbor/filename.py`:
- Change the signature/annotation of `generate_filename` first arg to `pcs_code: str` and use it directly in the f-strings (it is already interpolated as `{pcs_code}`; no numeric assumption exists, so only the annotation changes).
- In `ParsedFilename`, change `pcs_code: int` to `pcs_code: str`.
- In `parse_filename`, replace:
```python
    pcs_str, descriptor = prefix.split("-", 1)
    pcs_code = int(pcs_str)
```
with:
```python
    pcs_str, descriptor = prefix.split("-", 1)
    pcs_code = pcs_str  # keep as string; codes may contain '~'
```
(Reusing `extract_digest_from_stem` already guarantees `pcs_str` matches `^\d+(~\d+)*$`.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_hashing.py tests/test_filename.py -q`
Expected: PASS. (Update any existing test that asserted `pcs_code == 330` as an int to `== "330"`.)

- [ ] **Step 5: Commit**
```bash
git add imageharbor/hashing.py imageharbor/filename.py tests/test_hashing.py tests/test_filename.py
git commit -m "feat: filenames carry string PCS codes with ~ extension"
```

---

## Task 2: `taxonomy` table + Catalog data access

**Files:**
- Modify: `imageharbor/catalog.py` (`_SCHEMA`, `pcs_primary` type, new methods)
- Test: `tests/test_catalog.py`

**Interfaces:**
- Produces on `Catalog`: `taxonomy_is_empty() -> bool`; `taxonomy_insert(code, parent_code, label, folder_name, aliases=None, alias_of=None) -> None`; `taxonomy_get(code) -> sqlite3.Row | None`; `taxonomy_children(parent_code) -> list[sqlite3.Row]`; `taxonomy_all() -> list[sqlite3.Row]`; `taxonomy_set_alias(from_code, to_code) -> None`; `taxonomy_set_aliases(code, aliases: list) -> None`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_catalog.py`:
```python
def test_taxonomy_seed_insert_get_children(catalog: Catalog) -> None:
    assert catalog.taxonomy_is_empty() is True
    catalog.taxonomy_insert("500", None, "events", "500-events")
    catalog.taxonomy_insert("540", "500", "holidays", "540-holidays")
    assert catalog.taxonomy_is_empty() is False
    row = catalog.taxonomy_get("540")
    assert row["label"] == "holidays"
    assert row["parent_code"] == "500"
    kids = catalog.taxonomy_children("500")
    assert [k["code"] for k in kids] == ["540"]
    tops = catalog.taxonomy_children(None)
    assert [t["code"] for t in tops] == ["500"]

def test_taxonomy_set_alias(catalog: Catalog) -> None:
    catalog.taxonomy_insert("540", "500", "holidays", "540-holidays")
    catalog.taxonomy_insert("550", "500", "festivities", "550-festivities")
    catalog.taxonomy_set_alias("550", "540")
    row = catalog.taxonomy_get("550")
    assert row["alias_of"] == "540"
    assert row["active"] == 0

def test_taxonomy_set_aliases(catalog: Catalog) -> None:
    import json
    catalog.taxonomy_insert("540", "500", "holidays", "540-holidays")
    catalog.taxonomy_set_aliases("540", ["festivities", "xmas"])
    assert json.loads(catalog.taxonomy_get("540")["aliases"]) == ["festivities", "xmas"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_catalog.py -k taxonomy -v`
Expected: FAIL — no such methods/table.

- [ ] **Step 3: Implement**

In `imageharbor/catalog.py`, append to the `_SCHEMA` string (before its closing `"""`):
```sql

CREATE TABLE IF NOT EXISTS taxonomy (
    code         TEXT    PRIMARY KEY,
    parent_code  TEXT,
    label        TEXT    NOT NULL,
    folder_name  TEXT    NOT NULL,
    aliases      TEXT    NOT NULL DEFAULT '[]',
    alias_of     TEXT,
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_taxonomy_parent ON taxonomy(parent_code);
```
Change the `photos` table column `pcs_primary INTEGER NOT NULL DEFAULT 900` to `pcs_primary TEXT NOT NULL DEFAULT '900'`, and change the `upsert` parameter default `pcs_primary: int = 900` to `pcs_primary: str = "900"`.

Add these methods to the `Catalog` class:
```python
    # ------------------------------------------------------------------
    # Taxonomy
    # ------------------------------------------------------------------

    def taxonomy_is_empty(self) -> bool:
        cur = self._conn.execute("SELECT 1 FROM taxonomy LIMIT 1")
        return cur.fetchone() is None

    def taxonomy_insert(
        self,
        code: str,
        parent_code: str | None,
        label: str,
        folder_name: str,
        aliases: list[str] | None = None,
        alias_of: str | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO taxonomy (code, parent_code, label, folder_name,
                                  aliases, alias_of, active, created_at)
            VALUES (?,?,?,?,?,?,1,?)
            ON CONFLICT(code) DO NOTHING
            """,
            (code, parent_code, label, folder_name, _json(aliases or []), alias_of, _now_iso()),
        )
        self._conn.commit()

    def taxonomy_get(self, code: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM taxonomy WHERE code=?", (code,))
        return cur.fetchone()

    def taxonomy_children(self, parent_code: str | None) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM taxonomy WHERE parent_code IS ? ORDER BY code", (parent_code,)
        )
        return cur.fetchall()

    def taxonomy_all(self) -> list[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM taxonomy WHERE active=1 ORDER BY code")
        return cur.fetchall()

    def taxonomy_set_alias(self, from_code: str, to_code: str) -> None:
        self._conn.execute(
            "UPDATE taxonomy SET alias_of=?, active=0 WHERE code=?", (to_code, from_code)
        )
        self._conn.commit()

    def taxonomy_set_aliases(self, code: str, aliases: list[str]) -> None:
        self._conn.execute(
            "UPDATE taxonomy SET aliases=? WHERE code=?", (_json(aliases), code)
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: add taxonomy table + data access to catalog"
```

---

## Task 3: Taxonomy module — seeding, folder paths, numbering

**Files:**
- Create: `imageharbor/taxonomy.py`
- Modify: `imageharbor/pcs.py` (remove `parent_folder_name`/`sub_folder_name`), `tests/test_pcs.py`
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Consumes: Task-2 `Catalog.taxonomy_*` methods; `pcs.PCS_CATEGORIES`.
- Produces: `Taxonomy(catalog)` with `ensure_seeded()`, `get(code)->TaxonomyNode|None`, `children(parent)->list`, `folder_path(code)->str`, `mint_child(parent_code, label)->str`, `resolve_alias(code)->str`. Module helper `slug(label)->str`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_taxonomy.py`:
```python
"""Tests for the extensible taxonomy."""
from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.taxonomy import Taxonomy, slug


@pytest.fixture()
def tax(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    t = Taxonomy(cat)
    t.ensure_seeded()
    yield t
    cat.close()


def test_slug() -> None:
    assert slug("Christmas Eve!") == "christmas-eve"


def test_seed_has_fixed_spine(tax: Taxonomy) -> None:
    tops = [n.code for n in tax.children(None)]
    for c in ("100", "200", "300", "400", "500", "600", "700", "800", "900"):
        assert c in tops
    assert tax.get("330").label == "beach"


def test_folder_path(tax: Taxonomy) -> None:
    assert tax.folder_path("300") == "300-places"
    assert tax.folder_path("330") == "300-places/330-beach"


def test_mint_next_sub_and_leaf(tax: Taxonomy) -> None:
    # 500-events currently has 510/520/530 seeded; next sub is 540
    code = tax.mint_child("500", "holidays")
    assert code == "540"
    assert tax.folder_path("540") == "500-events/540-holidays"
    # a leaf under 540
    leaf = tax.mint_child("540", "christmas")
    assert leaf == "541"
    assert tax.folder_path("541") == "500-events/540-holidays/541-christmas"


def test_mint_overflow_uses_tilde(tax: Taxonomy) -> None:
    # Fill all 9 integer leaves under 540, then overflow -> 540~1
    tax.mint_child("500", "holidays")  # 540
    for i in range(9):
        tax.mint_child("540", f"leaf{i}")   # 541..549
    overflow = tax.mint_child("540", "tenth")
    assert overflow == "540~1"
    assert tax.folder_path("540~1") == "500-events/540-holidays/540~1-tenth"


def test_mint_is_append_only(tax: Taxonomy) -> None:
    a = tax.mint_child("500", "alpha")   # 540
    b = tax.mint_child("500", "beta")    # 550
    assert a == "540" and b == "550"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `imageharbor/taxonomy.py`**
```python
"""Extensible PCS taxonomy backed by the catalog `taxonomy` table.

Codes are strings matching ``^\\d+(~\\d+)*$``: plain integers for the common
three-level case (parent 500 -> sub 540 -> leaf 541), and a ``~N`` suffix for
overflow / depth. The 9 top-level classes are fixed; growth happens beneath and
is append-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import Catalog
from .pcs import PCS_CATEGORIES

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOP_RE = re.compile(r"^[1-9]00$")
_SUB_RE = re.compile(r"^[1-9][1-9]0$")


def slug(label: str) -> str:
    """Filesystem-safe folder slug: lowercase, ascii-alnum, hyphen-joined."""
    return _SLUG_RE.sub("-", label.lower()).strip("-") or "unnamed"


@dataclass
class TaxonomyNode:
    code: str
    parent_code: str | None
    label: str
    folder_name: str
    aliases: list[str] = field(default_factory=list)
    alias_of: str | None = None
    active: bool = True


def _node(row) -> TaxonomyNode:
    import json
    return TaxonomyNode(
        code=row["code"],
        parent_code=row["parent_code"],
        label=row["label"],
        folder_name=row["folder_name"],
        aliases=json.loads(row["aliases"]),
        alias_of=row["alias_of"],
        active=bool(row["active"]),
    )


class Taxonomy:
    """Catalog-backed taxonomy registry."""

    def __init__(self, catalog: Catalog) -> None:
        self._cat = catalog

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def ensure_seeded(self) -> None:
        """Seed from the legacy PCS_CATEGORIES on first use."""
        if not self._cat.taxonomy_is_empty():
            return
        # Insert parents first (parent_code None), then children.
        for code, cat in sorted(PCS_CATEGORIES.items()):
            parent = None if cat.parent is None else str(cat.parent)
            self._cat.taxonomy_insert(
                str(code), parent, cat.name, f"{code}-{slug(cat.name)}"
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, code: str) -> TaxonomyNode | None:
        row = self._cat.taxonomy_get(code)
        return _node(row) if row else None

    def children(self, parent_code: str | None) -> list[TaxonomyNode]:
        return [_node(r) for r in self._cat.taxonomy_children(parent_code)]

    def resolve_alias(self, code: str) -> str:
        """Follow the alias_of chain to the canonical active code."""
        seen: set[str] = set()
        cur = code
        while cur and cur not in seen:
            seen.add(cur)
            node = self.get(cur)
            if node is None or node.alias_of is None:
                return cur
            cur = node.alias_of
        return cur

    def folder_path(self, code: str) -> str:
        """Slash-joined folder path from the top-level ancestor to `code`."""
        parts: list[str] = []
        cur: str | None = code
        while cur:
            node = self.get(cur)
            if node is None:
                break
            parts.append(node.folder_name)
            cur = node.parent_code
        return "/".join(reversed(parts))

    # ------------------------------------------------------------------
    # Numbering (append-only)
    # ------------------------------------------------------------------

    def _integer_child_slots(self, parent_code: str) -> list[str]:
        if _TOP_RE.match(parent_code):
            base = int(parent_code)
            return [str(base + 10 * k) for k in range(1, 10)]
        if _SUB_RE.match(parent_code):
            base = int(parent_code)
            return [str(base + k) for k in range(1, 10)]
        return []  # leaf or ~-extended node: decimal children only

    def mint_child(self, parent_code: str, label: str) -> str:
        existing = {n.code for n in self.children(parent_code)}
        for candidate in self._integer_child_slots(parent_code):
            if candidate not in existing:
                return self._create(candidate, parent_code, label)
        # Overflow / depth: next ~N under parent_code
        n = 0
        prefix = parent_code + "~"
        for c in existing:
            if c.startswith(prefix) and c[len(prefix):].isdigit():
                n = max(n, int(c[len(prefix):]))
        return self._create(f"{parent_code}~{n + 1}", parent_code, label)

    def _create(self, code: str, parent_code: str, label: str) -> str:
        self._cat.taxonomy_insert(code, parent_code, label, f"{code}-{slug(label)}")
        return code
```

In `imageharbor/pcs.py`, delete `parent_folder_name` and `sub_folder_name` (they are replaced by `Taxonomy.folder_path`). Keep `PCS_CATEGORIES`, `PCS_VERSION`, `VALID_CODES`, `get_category`, `resolve_code`.

In `tests/test_pcs.py`, delete the tests that reference `parent_folder_name`/`sub_folder_name` (`test_parent_folder_name_*`, `test_sub_folder_name_*`, `test_parent_and_sub_folder_consistent`) and their import. Keep the rest.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_taxonomy.py tests/test_pcs.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add imageharbor/taxonomy.py imageharbor/pcs.py tests/test_taxonomy.py tests/test_pcs.py
git commit -m "feat: taxonomy module with seeding, folder paths, append-only numbering"
```

---

## Task 4: Resolution (normalize + adjudicate), merge, snapshot

**Files:**
- Modify: `imageharbor/taxonomy.py` (`resolve_or_create`, `merge`, `snapshot_text`, `_normalize`)
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Produces: `Taxonomy.resolve_or_create(top_parent: str, label: str, sub_parent: str | None = None, adjudicator=None) -> str`; `Taxonomy.merge(from_code, to_code) -> None`; `Taxonomy.snapshot_text() -> str`. `adjudicator` is `Callable[[str, list[str]], str | None]`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_taxonomy.py`:
```python
def test_resolve_reuses_existing_child(tax: Taxonomy) -> None:
    # "beach" already exists as 330 under 300
    assert tax.resolve_or_create("300", "Beaches") == "330"  # normalized reuse

def test_resolve_mints_when_new(tax: Taxonomy) -> None:
    code = tax.resolve_or_create("500", "holidays")
    assert code == "540"
    # second time reuses
    assert tax.resolve_or_create("500", "holidays") == "540"

def test_resolve_adjudicator_merges_synonym(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")  # 540
    calls = []
    def adj(label, candidates):
        calls.append((label, tuple(candidates)))
        return "holidays"  # the model says festivities == holidays
    code = tax.resolve_or_create("500", "festivities", adjudicator=adj)
    assert code == "540"          # reused, not minted
    assert calls                  # adjudicator consulted
    assert "festivities" in tax.get("540").aliases  # alias recorded

def test_resolve_no_adjudicator_mints_new(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")           # 540
    code = tax.resolve_or_create("500", "festivities")  # no adjudicator
    assert code == "550"  # minted as new sibling

def test_resolve_sub_parent_places_leaf(tax: Taxonomy) -> None:
    tax.resolve_or_create("500", "holidays")  # 540
    code = tax.resolve_or_create("500", "christmas", sub_parent="540")
    assert code == "541"

def test_merge_redirects_future_resolution(tax: Taxonomy) -> None:
    a = tax.resolve_or_create("500", "holidays")     # 540
    b = tax.resolve_or_create("500", "festivities")  # 550
    tax.merge(b, a)
    assert tax.resolve_alias(b) == a
    # a future exact-hit on the merged label redirects
    assert tax.resolve_or_create("500", "festivities") == a

def test_snapshot_text_lists_categories(tax: Taxonomy) -> None:
    s = tax.snapshot_text()
    assert "100" in s and "people" in s
    assert "330" in s and "beach" in s
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_taxonomy.py -k "resolve or merge or snapshot" -v`
Expected: FAIL — methods missing.

- [ ] **Step 3: Implement**

Add to `imageharbor/taxonomy.py` (imports at top: `import json`, `from difflib import SequenceMatcher`, `from typing import Callable`):
```python
    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(label: str) -> str:
        text = _SLUG_RE.sub(" ", label.lower()).strip()
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in text.split()]
        return " ".join(words)

    def resolve_or_create(
        self,
        top_parent: str,
        label: str,
        sub_parent: str | None = None,
        adjudicator: Callable[[str, list[str]], str | None] | None = None,
    ) -> str:
        target = sub_parent if sub_parent and self.get(sub_parent) else top_parent
        norm = self._normalize(label)
        kids = self.children(target)

        # Exact / alias hit
        for k in kids:
            names = [self._normalize(k.label)] + [self._normalize(a) for a in k.aliases]
            if norm in names:
                return self.resolve_alias(k.code)

        # Fuzzy near-miss -> adjudicate
        near = [k for k in kids if SequenceMatcher(None, norm, self._normalize(k.label)).ratio() >= 0.8]
        if near and adjudicator is not None:
            matched = adjudicator(label, [k.label for k in near])
            if matched:
                for k in near:
                    if k.label == matched:
                        aliases = k.aliases + [label]
                        self._cat.taxonomy_set_aliases(k.code, aliases)
                        return self.resolve_alias(k.code)

        return self.mint_child(target, label)

    def merge(self, from_code: str, to_code: str) -> None:
        target = self.get(to_code)
        src = self.get(from_code)
        if target is None or src is None:
            return
        self._cat.taxonomy_set_aliases(to_code, target.aliases + [src.label])
        self._cat.taxonomy_set_alias(from_code, to_code)

    def snapshot_text(self) -> str:
        """Compact `code label` view grouped by hierarchy for the AI prompt."""
        rows = self._cat.taxonomy_all()
        by_parent: dict[str | None, list] = {}
        for r in rows:
            by_parent.setdefault(r["parent_code"], []).append(r)

        lines: list[str] = []

        def emit(code: str, label: str, depth: int) -> None:
            lines.append(f"{'  ' * depth}{code} {label}")
            for child in by_parent.get(code, []):
                emit(child["code"], child["label"], depth + 1)

        for top in by_parent.get(None, []):
            emit(top["code"], top["label"], 0)
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_taxonomy.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add imageharbor/taxonomy.py tests/test_taxonomy.py
git commit -m "feat: taxonomy resolve_or_create (dedup+adjudicate), merge, snapshot"
```

---

## Task 5: Classifier contract — label + parent, adjudicate

**Files:**
- Modify: `imageharbor/ai_classifier.py`
- Test: `tests/test_ai_classifier.py`

**Interfaces:**
- Produces: `PhotoClassification(top_parent: str, label: str, sub_parent: str | None, descriptor, caption, objects, secondary_tags, ocr_text, model_version, pcs_version)`; `AIClassifier.classify(image_path, exif_data, taxonomy_snapshot: str) -> PhotoClassification`; `AIClassifier.adjudicate(label: str, candidates: list[str]) -> str | None` (default `None`).

- [ ] **Step 1: Write failing tests**

Adjust `tests/test_ai_classifier.py` (replace code-based assertions). Representative new tests:
```python
def test_stub_returns_parent_and_label() -> None:
    c = StubClassifier().classify(Path("beach_sunset.jpg"), {}, taxonomy_snapshot="")
    assert c.top_parent == "300"
    assert c.label == "beach"
    assert c.sub_parent is None

def test_stub_unknown_is_misc() -> None:
    c = StubClassifier().classify(Path("random_xyz.jpg"), {}, taxonomy_snapshot="")
    assert c.top_parent == "900"
    assert c.label == "miscellaneous"

def test_stub_adjudicate_returns_none() -> None:
    assert StubClassifier().adjudicate("festivities", ["holidays"]) is None

def test_stub_classify_is_deterministic() -> None:
    a = StubClassifier().classify(Path("my_dog.jpg"), {}, "")
    b = StubClassifier().classify(Path("my_dog.jpg"), {}, "")
    assert (a.top_parent, a.label, a.descriptor) == (b.top_parent, b.label, b.descriptor)
```
(Keep OpenAI tests, updated: the fake client returns JSON with `top_parent`/`label`; `classify` takes a snapshot arg; add a test that the snapshot text appears in the system prompt sent to the mocked client, and that `adjudicate` parses the model's reply.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ai_classifier.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `imageharbor/ai_classifier.py`:

Replace the `PhotoClassification` dataclass:
```python
@dataclass
class PhotoClassification:
    """All AI-derived metadata for a single image."""

    top_parent: str              # one of the 9 fixed classes, e.g. "500"
    label: str                   # category label, e.g. "holidays" (reuse or new)
    sub_parent: str | None = None  # existing sub code to place a new leaf under
    descriptor: str = "photo"    # 1-3 words for the filename
    caption: str = ""
    objects: list[str] = field(default_factory=list)
    secondary_tags: list[str] = field(default_factory=list)
    ocr_text: str = ""
    model_version: str = "stub-1.0"
    pcs_version: str = PCS_VERSION
```

Change the ABC:
```python
class AIClassifier(ABC):
    @abstractmethod
    def classify(
        self, image_path: Path, exif_data: dict[str, Any], taxonomy_snapshot: str
    ) -> PhotoClassification: ...

    def adjudicate(self, label: str, candidates: list[str]) -> str | None:
        """Return the candidate the label is a synonym of, or None. Default: no match."""
        return None
```

`StubClassifier.classify`: keep the keyword→code scan, then translate the matched code into `(top_parent, label)` via the seed data:
```python
    def classify(self, image_path, exif_data, taxonomy_snapshot):
        from .pcs import PCS_CATEGORIES
        stem = image_path.stem.lower()
        pcs_code = 900
        for pattern, code in self._keyword_map():
            if any(kw in set(re.sub(r"[^a-z0-9]+", " ", stem).split()) for kw in pattern.split("|")):
                pcs_code = code
                break
        cat = PCS_CATEGORIES.get(pcs_code) or PCS_CATEGORIES[900]
        top_parent = str((pcs_code // 100) * 100)
        words = [w for w in re.sub(r"[^a-z0-9]+", " ", stem).split() if len(w) > 1][:2]
        descriptor = " ".join(words) if words else "photo"
        return PhotoClassification(
            top_parent=top_parent,
            label=cat.name,
            descriptor=descriptor,
            caption=f"Stub classification for {image_path.name}",
            model_version=self.MODEL_VERSION,
        )
```
(Move the existing `keyword_map` list into a `_keyword_map()` method or module constant so it is reused; keep the same keyword sets. `adjudicate` inherits the default `None`.)

`OpenAIClassifier`:
- `classify(self, image_path, exif_data, taxonomy_snapshot)`: build the system prompt from a new template that embeds `taxonomy_snapshot` and instructs the model to return JSON `{"top_parent","label","sub_parent"(optional),"descriptor","caption","objects","secondary_tags","ocr_text"}`, reusing an existing category if one fits. Parse defensively (as today): `top_parent = str(data.get("top_parent","900"))`, `label = str(data.get("label","miscellaneous"))`, `sub_parent = data.get("sub_parent") or None`, list fields coerced as in the current hardened code.
- Add:
```python
    def adjudicate(self, label: str, candidates: list[str]) -> str | None:
        prompt = (
            f"Is '{label}' the same category as any of these: {candidates}? "
            f"Reply with the exact matching item, or NONE."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
        )
        answer = (resp.choices[0].message.content or "").strip()
        for c in candidates:
            if c.lower() == answer.lower():
                return c
        return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ai_classifier.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add imageharbor/ai_classifier.py tests/test_ai_classifier.py
git commit -m "feat: classifier returns label+parent; add adjudicate"
```

---

## Task 6: Pipeline + CLI integration

**Files:**
- Modify: `imageharbor/pipeline.py`, `imageharbor/cli.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Taxonomy` (T3/T4), the new classifier contract (T5), string-code filenames (T1).
- Produces: a pipeline that resolves labels to codes and organizes into `taxonomy.folder_path(code)`; catalog stores string `pcs_primary`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pipeline.py` (fixtures already exist):
```python
def test_pipeline_uses_taxonomy_codes(source_dir, organized_dir, catalog) -> None:
    pipeline = Pipeline(source_dir, organized_dir, catalog)
    stats = pipeline.run()
    assert stats.copied == 2 and stats.errors == 0
    # beach_photo.jpg -> 300-places/330-beach ; mountain_view.jpg -> 340-mountains
    organized = [p.as_posix() for p in organized_dir.rglob("*.jpg")]
    assert any("300-places/330-beach/330-" in p for p in organized)


def test_pipeline_mints_new_category(organized_dir, catalog, tmp_path) -> None:
    # A custom classifier that proposes a brand-new label under events
    from imageharbor.ai_classifier import AIClassifier, PhotoClassification

    class NewCatClassifier(AIClassifier):
        def classify(self, image_path, exif_data, taxonomy_snapshot):
            return PhotoClassification(top_parent="500", label="holidays", descriptor="xmas")

    src = tmp_path / "src2"; src.mkdir()
    _make_jpeg(src / "a.jpg")
    pipeline = Pipeline(src, organized_dir, catalog, classifier=NewCatClassifier())
    pipeline.run()
    assert any("500-events/540-holidays/540-" in p.as_posix() for p in organized_dir.rglob("*.jpg"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k "taxonomy or mints_new" -v`
Expected: FAIL — pipeline still calls `resolve_code`/`parent_folder_name`.

- [ ] **Step 3: Implement**

In `imageharbor/pipeline.py`:
- Imports: replace `from .pcs import parent_folder_name, resolve_code, sub_folder_name` with `from .taxonomy import Taxonomy`.
- In `__init__`, add `self.taxonomy = Taxonomy(catalog)`.
- In `run()` (and `process_file`), call `self.taxonomy.ensure_seeded()` once at the start of `run()` (and in `process_file` before `_process_one`).
- In `_do_process`, replace steps 4–7:
```python
        # Step 4: EXIF
        exif_data = read_exif(source_path)

        # Step 4b: classify (AI gets the current taxonomy snapshot)
        snapshot = self.taxonomy.snapshot_text()
        classification = self.classifier.classify(source_path, exif_data, snapshot)

        # Step 5: resolve label -> code (dedup + optional AI adjudication)
        pcs_code = self.taxonomy.resolve_or_create(
            classification.top_parent,
            classification.label,
            classification.sub_parent,
            adjudicator=self.classifier.adjudicate,
        )
        node = self.taxonomy.get(pcs_code)
        pcs_name = node.label if node else classification.label

        # Step 6: filename
        descriptor = normalize_descriptor(classification.descriptor)
        extension = source_path.suffix.lstrip(".").lower()
        filename = generate_filename(pcs_code, descriptor, sha256_b64url, extension)

        # Step 7: output path
        organized_path = self.organized_dir / self.taxonomy.folder_path(pcs_code) / filename
```
- In `_update_catalog`, drop the `PCS_CATEGORIES` lookup; accept `pcs_code: str` and `pcs_name: str` and pass them straight to `catalog.upsert(pcs_primary=pcs_code, pcs_name=pcs_name, ...)`. Update the call site and remove `pcs_version` reliance if needed (keep `classification.pcs_version`). Update `_write_sidecar` similarly (it reads `parse_filename(...)["pcs_code"]`, now a str — fine).

In `imageharbor/cli.py`, in `catalog_list`, change the numeric format:
```python
        click.echo(f"{row['sha256_b64url'][:12]}…  {str(row['pcs_primary']):>5}  {row['organized_path']}")
```
(Replace the `:3d` int format.) No other CLI change is required — `process`/`watch` already construct the classifier and Pipeline, and the Pipeline now owns the taxonomy.

- [ ] **Step 4: Run the focused tests, then the full suite**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -q`
Expected: PASS (update any remaining test that asserted an int `pcs_primary` or old folder helpers).

Run: `uv run pytest -q`
Expected: PASS — entire suite green.

- [ ] **Step 5: Commit**
```bash
git add imageharbor/pipeline.py imageharbor/cli.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: pipeline resolves labels via taxonomy; CLI prints string codes"
```

---

## Task 7: Docs — CLAUDE.md invariants + genesis note

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the invariants**

In `CLAUDE.md`, under "Critical invariants", update the PCS/filename bullets to state:
- PCS codes are now **strings** matching `^\d+(~\d+)*$` (a `~N` suffix for overflow/depth; **never a dot**), owned by the self-extending taxonomy in the catalog `taxonomy` table (seeded from `pcs.PCS_CATEGORIES`).
- Folder paths come from `taxonomy.folder_path(code)`, not `pcs.parent_folder_name`/`sub_folder_name` (removed).
- The classifier returns `(top_parent, label, sub_parent?)`; the pipeline owns code assignment. The 43-char digest counting-back rule is unchanged.

Update the Architecture section's `pcs.py` bullet to note it now holds only seed data + helpers, and add a `taxonomy.py` bullet.

- [ ] **Step 2: Commit**
```bash
git add CLAUDE.md
git commit -m "docs: update invariants for the self-extending taxonomy"
```

---

## Self-Review

**Spec coverage:**
- §3 registry (table + data) → Tasks 2, 3 (seeding). ✓
- §4 code/folder scheme + filename impact → Tasks 1, 3 (`folder_path`, numbering). ✓
- §5 classifier contract + snapshot passing → Task 5, wired in Task 6. ✓
- §6 resolve (normalize + adjudicate) → Task 4. ✓
- §7 merge → Task 4. ✓
- §8 single-call snapshot → Task 5 (prompt) + Task 6 (pipeline passes snapshot). ✓
- §9 determinism → covered by append-only numbering (Task 3) + reuse (Task 4). ✓
- §10 migration/seeding → Task 3 `ensure_seeded`; `resolve_code` fallback retained; test-data wipe is an ops step (task #16), not code. ✓
- §11 testing → tests in every task. ✓
- §13 acceptance → Tasks 1–6 + full-suite run in Task 6 Step 4.

**Placeholder scan:** no TBD/TODO; every code step shows real code. Two steps say "keep the existing keyword sets / OpenAI hardening" — those refer to code already present in the repo, not new content to invent.

**Type consistency:** `pcs_code` is a `str` end-to-end (Tasks 1→3→6); `PhotoClassification` fields (`top_parent`, `label`, `sub_parent`, `descriptor`) are defined in Task 5 and consumed identically in Task 6; `Taxonomy.resolve_or_create(top_parent, label, sub_parent, adjudicator)` matches its Task-6 call site; `Catalog.taxonomy_*` signatures match between Tasks 2 and 3/4.

**Deferred (spec §12):** web dashboard, embedding dedup, two-stage prompting, physically relocating merged files — intentionally not planned.
