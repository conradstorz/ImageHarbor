# ImageHarbor — Perception / Organization Reframe

**Status:** Approved design (pending spec review)
**Date:** 2026-08-02
**Author:** Conrad Storz (with Claude Code)

## 1. Purpose

The first live Jetson run exposed the root problem: the classifier asks a small
(3B) vision model to do three things at once — understand the image, *navigate
our hierarchical taxonomy* (pick a `top_parent`, decide reuse-vs-new), and
propose a `sub_parent`. The middle job is where it falls apart (echoing prompt
lines like `"610 plants"`, inventing `"events-6"`, misfiling bands into
`aircraft`).

This reframe **separates perception from organization**:

- **The AI does perception only** — "what's in this photo?" → a content JSON.
  This is the task vision models are reliable at.
- **Our code does organization** — turns that content into a taxonomy code +
  filename, deterministically wherever possible.

It reuses the taxonomy backend already built (registry, `resolve_or_create`,
mint/dedup/`merge`, folder paths, catalog, filename layer) and replaces the
classifier contract + adds a class-decision layer in front of it.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Split | **AI describes; our code organizes** |
| Class-decision engine | **Hybrid** — deterministic concept-map first, text-AI step for the tail |
| Content schema | `{primary_subject, scene, objects[], caption, tags[], ocr_text}` |
| Label source | **`primary_subject` becomes the category label** |
| Hierarchy | **Two levels** — 9 fixed classes → flat, deduplicated subject sub-categories. **Level-3 / `sub_parent` dropped.** |
| Concept-map | **Self-learning** — memoize each AI-step `subject → class` decision so repeat subjects become deterministic hits |

## 3. Component 1 — Perception contract (the AI's whole job)

`PhotoClassification` (which carried `top_parent`/`label`/`sub_parent`) is
replaced by a content record:

```python
@dataclass
class ContentDescription:
    primary_subject: str          # single main subject, 1-3 words ("marching band")
    scene: str = ""               # setting/context ("outdoor parade")
    objects: list[str] = []       # notable objects/entities
    caption: str = ""             # one-sentence description
    tags: list[str] = []          # descriptive keywords
    ocr_text: str = ""            # visible text, or empty
    model_version: str = "stub-1.0"
```

`AIClassifier` methods:

- `describe(image_path, exif_data) -> ContentDescription` — **vision**. The
  model's only image-facing job: look and describe. Prompt: return the JSON
  above, no codes/classes/labels.
- `pick_class(content: ContentDescription, classes: list[tuple[str, str]]) -> str`
  — **text-only**. Given the content and the 9 `(code, label)` classes, return
  the single best class **code**. A far smaller ask than today. Guarded: an
  invalid/unknown return coerces to `"900"`.
- `adjudicate(label, candidates) -> str | None` — **retained**, used by
  `resolve_or_create` for subject-label dedup at level 2.

Implementations:
- **`StubClassifier`** (offline/deterministic): `describe` derives
  `primary_subject`/tags from filename keywords; `pick_class` maps
  deterministically (or `900`); `adjudicate` returns `None`.
- **`OpenAIClassifier`**: `describe` = vision prompt → content JSON (parsed
  defensively); `pick_class` = text prompt → class code; `adjudicate` = text
  synonym check. All lazy-import `openai`.

## 4. Component 2 — Self-learning concept-map

New `imageharbor/concept_map.py` (catalog-backed for the learned half).

**Static seed** (`concept → class_code`), bootstrapped from the existing
`pcs.PCS_CATEGORIES`: every legacy sub-category *name* maps to its parent class
(`portrait → 100`, `pet → 210→200`, `beach → 300`, `car → 400`, `sports → 500`,
`landscape → 600`, `receipt → 700`, `building → 800`, …), plus a curated set of
common synonyms (`band/parade/wedding/concert → 500`, `dog/cat/bird → 200`, …).

**Learned store** — a new catalog table:
```sql
CREATE TABLE IF NOT EXISTS learned_concepts (
    subject     TEXT    PRIMARY KEY,   -- normalized primary_subject
    class_code  TEXT    NOT NULL,      -- assigned top-level class
    hits        INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT
);
```

**Lookup** `class_for(content) -> str | None` (reusing `taxonomy._normalize_label`
so learned-store keys, static-seed keys, and taxonomy labels all normalize
identically):
1. `norm = _normalize_label(primary_subject)`.
2. **Learned hit** — `learned_concepts` exact match on `norm` → return its class.
3. **Static hit** — `norm`, or any `objects`/`scene` keyword, matches the static
   table → return that class.
4. Otherwise `None` (a miss → the pipeline calls `pick_class`).

**Learn** `remember(subject, class_code)` — upsert into `learned_concepts`
(increment `hits`). Called after every `pick_class` result, so the *next* photo
with that subject is a deterministic concept-map hit — fewer AI calls and more
determinism over time. (A wrong memoized mapping is correctable with the same
`merge`/edit tooling; a future dashboard action.)

## 5. Component 3 — Taxonomy simplified to two levels

- `resolve_or_create(class_code, label, adjudicator=None)` — **`sub_parent`
  parameter removed**. `class_code` is always one of the 9 fixed classes; `label`
  is the `primary_subject`. It reuses or mints a **level-2** sub under the class.
- `mint_child` is only ever called with a top-level class, so it allocates
  `X10..X90` then `~N` overflow. The level-3 integer-slot branch is no longer
  exercised (kept harmless or removed).
- **Retained unchanged:** append-only numbering, `resolve_alias`, the self-label
  reuse guard, the degenerate-label guard, `merge`, `snapshot_text`,
  `folder_path` (now walks at most 2 levels).
- Net effect: the tree is `class / subject`, e.g. `500-events/510-marching-band/`.
  The `celebrations/celebrations` redundancy is impossible by construction.

## 6. Component 4 — Pipeline orchestration

Per image (`_do_process`), replacing the classify/resolve steps:

```
hash -> duplicate-check -> EXIF
content   = classifier.describe(path, exif)
cls       = concept_map.class_for(content)
if cls is None:
    cls   = classifier.pick_class(content, CLASSES)   # 9 (code,label) pairs
    concept_map.remember(content.primary_subject, cls)
label     = content.primary_subject
code      = taxonomy.resolve_or_create(cls, label, adjudicator=classifier.adjudicate)
descriptor= normalize_descriptor(content.primary_subject)
filename  = generate_filename(code, descriptor, sha, ext)
dest      = organized_dir / taxonomy.folder_path(code) / filename
-> copy -> verify -> catalog(content: caption/objects/tags/ocr/scene) -> sidecar
```

`ensure_seeded()` seeds both the taxonomy and the static concept-map on first
use; dry-run still performs zero AI calls / zero writes (short-circuits before
`describe`).

`CLASSES` = the 9 fixed top-level `(code, label)` pairs from `PCS_CATEGORIES`.

## 7. Determinism (honest)

- **Class decision** is deterministic once a subject is learned (concept-map hit);
  only a subject's *first* encounter uses the AI `pick_class`.
- **`primary_subject` itself still varies** across images of similar things
  ("marching band" vs "band" vs "musicians") because the vision step is
  non-deterministic — so labels still vary and rely on the dedup/`merge` backend.
  Net: materially more stable and cheaper than the current design, not perfectly
  reproducible (that would need a rules-only engine, which was not chosen).

## 8. Migration & backward compatibility

- This **replaces** the current classifier contract (`top_parent/label/sub_parent`
  → `describe`/`pick_class`) and drops level-3.
- The taxonomy registry, catalog, filename/hashing layers are reused as-is (minus
  the `sub_parent` path).
- The paused live deployment is wiped and restarted fresh on the new pipeline;
  no mixed-scheme data persists.

## 9. Testing (offline, deterministic)

- **Perception:** `StubClassifier.describe` deterministic content from filename;
  `pick_class` deterministic; `OpenAIClassifier.describe`/`pick_class` parse a
  mocked client response and embed the right prompt (no network).
- **Concept-map:** static seed from `PCS_CATEGORIES` (a legacy sub-name resolves
  to its parent class); learned hit beats static; `remember` round-trip;
  `class_for` returns `None` on a genuine miss.
- **Taxonomy 2-level:** `resolve_or_create(class, subject)` mints a level-2 sub;
  reuse/dedup/merge still hold; no `sub_parent` path.
- **Pipeline:** stub end-to-end organizes into `class/subject`; a miss triggers
  `pick_class` + `remember`, and the second identical subject is a concept-map
  hit (assert `pick_class` NOT called the second time).
- Full existing suite updated for the new contract stays green.

## 10. Out of scope

- The **web dashboard / control plane** (its own session) — including a UI to
  review/edit learned mappings and merge categories.
- **Embedding-based** matching.
- Physically relocating already-organized files after a `merge`/remap.

## 11. Acceptance criteria

1. The AI returns only a content description; it never picks a code, class, or
   navigates the taxonomy.
2. The class is decided by a deterministic concept-map first, falling back to a
   text-only `pick_class`; the concept-map **learns** each fallback decision so
   repeat subjects stop calling the AI.
3. The taxonomy is two levels (fixed class → `primary_subject` sub); level-3 and
   `sub_parent` are gone; dedup/`merge`/append-only numbering still hold.
4. Invariants preserved: originals read-only; copy→verify→catalog; self-verifying
   filenames; catalog on the local volume; dry-run does zero AI/taxonomy writes.
5. The full existing suite plus new perception/concept-map tests pass.
