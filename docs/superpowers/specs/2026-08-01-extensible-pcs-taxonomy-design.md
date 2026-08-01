# ImageHarbor — Self-Extending PCS Taxonomy

**Status:** Approved design (pending spec review)
**Date:** 2026-08-01
**Author:** Conrad Storz (with Claude Code)

## 1. Purpose

Today the PCS taxonomy is a fixed, hardcoded vocabulary (`PCS_CATEGORIES` in
`pcs.py`). Against a real library the AI wants categories the taxonomy doesn't
have — it invented a nonexistent code `540` for "holidays/parties", which fell
back to `900-miscellaneous`. This spec makes the taxonomy **grow to fit the
collection**, Dewey-Decimal style, under guard rails that preserve ImageHarbor's
determinism and immutable, self-verifying filenames.

Delivered together: an extensible taxonomy registry, a changed classifier
contract (AI describes, system numbers), a label→code resolver with dedup, and a
merge escape hatch. The **web dashboard is a separate session** and out of scope.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Governance | **Guard-railed auto-growth** — AI can trigger new codes, but only through system rules |
| Classifier contract | **AI returns a label + parent; the system owns all numbering** (AI never picks numbers) |
| Hierarchy shape | **Three bounded levels** under a **fixed 9-class spine**; a `~`-suffix for overflow / rare 4th level (the dot is banned from codes) |
| Top-level classes | **Fixed** (`100-people … 900-miscellaneous`) — never grow, never renumber |
| Dedup guardrail | **Normalize, then AI-adjudicate near-misses** (option B); embeddings (C) out of scope |
| Merge | A `merge(X → Y)` alias operation is included |
| Prompt strategy | **Single call, whole taxonomy in the prompt**; two-stage is a documented fallback only |

## 3. Component 1 — Taxonomy registry (data, not code)

The taxonomy moves out of `pcs.py` into a **`taxonomy` table in the catalog
SQLite DB** (local volume, the existing source of truth). Because folder names
encode codes (`540-holidays`), the registry is also reconstructable from the
organized tree if ever lost.

```sql
CREATE TABLE IF NOT EXISTS taxonomy (
    code         TEXT PRIMARY KEY,              -- "100","540","541","540.1"
    parent_code  TEXT,                          -- NULL for top-level; else parent's code
    label        TEXT    NOT NULL,              -- canonical label, e.g. "holidays"
    folder_name  TEXT    NOT NULL,              -- "<code>-<slug(label)>"
    aliases      TEXT    NOT NULL DEFAULT '[]', -- JSON list of alternate labels folded in
    alias_of     TEXT,                          -- if set, this code is merged into another
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    NOT NULL
);
```

**Seeding / migration:** on catalog open, if `taxonomy` is empty, seed it from
the current `PCS_CATEGORIES` (codes 100–930 with their names and parents), so
existing behavior and any already-organized integer-coded files remain valid.
`pcs.py` keeps the seed data + pure helpers; the *live* taxonomy is the table.

**Registry API** (new `imageharbor/taxonomy.py`, backed by the catalog):
- `get(code) -> TaxonomyNode | None`
- `children(parent_code) -> list[TaxonomyNode]`
- `resolve_alias(code) -> code` (follow `alias_of` to the canonical code)
- `folder_path(code) -> str` (join ancestor `folder_name`s)
- `resolve_or_create(parent_code, label, sub_parent=None, adjudicator=None) -> code`
- `merge(from_code, to_code) -> None`
- `mint_child(parent_code, label) -> code`

## 4. Component 2 — Code & folder scheme (⚠️ changes a filename invariant)

Codes are hierarchical strings under the fixed spine:

- **Level 1 (parent):** `100,200,…,900` — fixed.
- **Level 2 (sub):** children of `P00` are `P10,P20,…,P90` (9 integer slots).
- **Level 3 (leaf):** children of `PY0` are `PY1,…,PY9` (9 integer slots).
- **Overflow / depth:** when a node's 9 integer child slots are exhausted, or the
  node is already a leaf, the next child appends a **filename-safe `~`
  separator**: `C~1, C~2, …` (`parent_code` of `540~1` is `540`). This unifies
  "10th sibling" overflow and "4th level" depth into one rule: `C~N` is the N-th
  extended child of `C`.

**The dot is banned from codes.** A `.` in the code would collide with the
extension separator (the exact multi-dot hazard already fixed in `generate_filename`).
`~` is legal on all target filesystems (Windows/SMB/Synology/macOS/Linux) and is
**not** in the base64url alphabet, so it can never collide with the 43-char digest.

**Folders** mirror ancestry: code `541` → `500-events/540-holidays/541-christmas/`;
code `540~1` → `500-events/540-holidays/540~1-<label>/`. The **leaf code** is what
goes in the filename: `541-christmas_<sha>.jpg`.

**Filename-invariant impact:** the `<pcs>` field becomes a **string token matching
`^\d+(~\d+)*$`** (digits, optionally followed by `~`-separated groups — **never a
dot**), so:
- `hashing.extract_digest_from_stem` — its PCS validation changes from
  `isdigit()` to matching the code pattern `^\d+(~\d+)*$`. The 43-char digest
  counting-back logic is **unchanged** (base64url never contains `~`).
- `filename.parse_filename` — `pcs_code` becomes a **`str`**, not `int`.
- `pcs.parent_folder_name`/`sub_folder_name` are replaced by
  `taxonomy.folder_path(code)`.
- CLAUDE.md "critical invariants" is updated to describe dotted codes.

## 5. Component 3 — Classifier contract (AI describes, system numbers)

`PhotoClassification` stops carrying an AI-chosen `pcs_code`. The AI instead
returns a **proposal**:

- `top_parent`: one of the 9 fixed classes (e.g. `500` / "events")
- `label`: a short category label (reuse an existing one if it fits, else new)
- `sub_parent` (optional): an existing sub-label to place a new leaf under
- plus the existing `caption`, `objects`, `secondary_tags`, `ocr_text`, `descriptor`

The resolved numeric `code` is computed by the **pipeline via the resolver**
(§6), not by the classifier.

The classifier stays decoupled from the catalog: the **pipeline passes the
current taxonomy snapshot** (the compact `code: label` view) into `classify()`
per image, so the classifier never queries the registry itself. `classify`'s
signature gains the snapshot argument.

- **`StubClassifier`** (offline/deterministic): maps filename keywords to a
  `(top_parent, label)` deterministically — no numbers, no network. Used by tests.
- **`OpenAIClassifier`**: the prompt now includes the **current taxonomy**
  (compact `code: label` lines grouped by parent) and instructs: "Pick the
  top-level class. **Reuse** an existing category if it fits; otherwise propose a
  short new `label` under a parent." Returns the proposal JSON. It also exposes an
  **`adjudicate(label, candidates) -> matched|None`** method used as the Line-2
  backstop; `StubClassifier.adjudicate` returns `None` (skip → deterministic).

## 6. Component 4 — Label → code resolution (the guard rail)

`resolve_or_create(parent_code, label, sub_parent, adjudicator)`:

1. **Normalize** `label`: lowercase, trim, collapse punctuation/whitespace,
   naive singularize, slug. (Deterministic.)
2. **Line-1 already happened** (the AI was shown existing categories and asked to
   reuse). Now match the normalized label against existing `label`/`aliases`
   within the chosen parent's subtree: **exact/alias hit → return that code.**
3. **Fuzzy near-miss** (token/edit-distance above a threshold but not exact) →
   call `adjudicator(label, [candidate labels])` (one AI call). If it returns a
   match, **reuse that code** and record the incoming label as an alias. If the
   adjudicator is `None` (stub/tests) or returns none, treat as new.
4. **No match → mint**: `mint_child(parent_or_sub, label)` assigns the lowest
   free integer child at the next level, or the next `.N` decimal on overflow;
   insert the row; return the new code.

Resolution only ever **adds** rows; it never renumbers or deletes.

## 7. Component 5 — Merge escape hatch

`merge(from_code, to_code)` sets `from_code.alias_of = to_code` (and folds
`from_code`'s label into `to_code.aliases`). Effects:
- Future classifications that resolve to `from_code` **redirect to `to_code`**
  (via `resolve_alias`).
- **Existing files are not moved or renamed** — that would break the immutability
  invariant. Their on-disk folder for `from_code` remains; the registry alias
  keeps `from_code` resolvable so nothing is orphaned. (Physically relocating
  old files is explicitly a future/dashboard concern, not this spec.)

## 8. Prompt strategy & run-time

Start **single-call**: the whole taxonomy is sent as compact text in each
classification prompt (a few hundred `code: label` lines is small), keeping it at
~1 AI call (~30 s/image on the Jetson 3b) and letting Line-1 reuse work. If the
taxonomy ever grows large enough to bloat the prompt, the documented fallback is
**two-stage** (call 1 picks the top-level class, call 2 works within that
subtree) — which roughly doubles calls/run-time and is therefore not the default.

## 9. Determinism (honest consequence)

New-code assignment is **order-dependent**: the first photo that needs "holidays"
mints it. Once minted, resolution is **stable and deterministic** for a given
registry state. Append-only + dedup keep this from causing churn. Given a fixed
registry, the same photo always resolves to the same code.

## 10. Migration & backward compatibility

- Seed the registry from `PCS_CATEGORIES`; existing integer codes stay valid.
- The `resolve_code`-style ultimate fallback to `900-miscellaneous` is preserved
  for any classifier that fails to produce a usable parent/label.
- The smoke-test artifacts are being wiped before the first real run, so no
  mixed-scheme data persists into production.

## 11. Testing (offline, deterministic — no network)

- **Registry:** seeding from `PCS_CATEGORIES`; `children`; `folder_path` for
  1/2/3-level and `~`-extended codes; append-only mint; overflow → `~N`.
- **Resolution:** exact/alias reuse; fuzzy near-miss → adjudicator reuse (mocked
  adjudicator); no-match → mint next code; `adjudicator=None` → deterministic new.
- **Merge:** `resolve_alias` follows `alias_of`; a merged code redirects future
  resolutions; existing files untouched.
- **Classifier:** `StubClassifier` returns `(top_parent,label)` deterministically;
  `OpenAIClassifier` prompt includes the taxonomy and parses the proposal (mocked
  client); `adjudicate` mocked.
- **Filename/hashing:** `extract_digest_from_stem` and `parse_filename` accept
  `~`-extended codes (`540~1`) and reject codes containing a dot; digest
  counting-back intact; `pcs_code` is a `str`.
- **Determinism:** same photo + same registry → same code, twice.
- The full existing suite stays green.

## 12. Out of scope

- The **web dashboard / control plane** (separate session) — including the *UI*
  for merge/curation and physically relocating merged files.
- **Embedding-based** dedup (option C).
- **Two-stage** prompting (fallback only, not built now).
- Growing or renumbering the **9 top-level classes**.

## 13. Acceptance criteria

1. The taxonomy is persisted in the catalog, seeded from the current PCS, and
   reconstructable from the organized tree.
2. The classifier returns a label + parent and **never** picks a number; the
   system assigns every code.
3. A genuinely new label mints the next code under its parent (3 integer levels,
   decimal overflow), append-only; the old `540`-style hallucination now yields a
   real, reused-or-minted code instead of dumping into `900`.
4. Dedup reuses existing codes via normalize + alias + AI-adjudicated near-miss;
   `merge(X→Y)` aliases codes without moving existing files.
5. Filenames/folders support `~`-extended codes (no dots in codes); the 43-char
   digest invariant is intact; `parse_filename` handles string codes.
6. Determinism holds: fixed registry → same photo → same code.
7. The full existing suite plus the new tests pass.
