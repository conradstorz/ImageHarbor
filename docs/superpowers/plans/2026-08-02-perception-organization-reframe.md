# Perception / Organization Reframe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AI do perception only (describe the image → content JSON) and let our code do organization: a self-learning concept-map decides the class, the `primary_subject` becomes a level-2 taxonomy label, and level-3/`sub_parent` are retired.

**Architecture:** New `describe`/`pick_class` classifier contract replaces `classify`. A new `concept_map.py` (static seed from the legacy PCS sub-names + a `learned_concepts` catalog table) decides the class deterministically where it can, falling back to a text-only `pick_class`. The taxonomy backend (registry/`resolve_or_create`/mint/dedup/`merge`/folders) is reused as-is; the 2-level shape emerges from always calling `resolve_or_create(class, primary_subject)` with no `sub_parent`.

**Tech Stack:** Python 3.10+, SQLite, Click, Pillow, optional `openai`. `uv` for dev/test. Stdlib only for new logic.

## Global Constraints

- Python floor `>=3.10`; `from __future__ import annotations` in new modules; no newer-only syntax.
- Runtime deps limited to `Pillow` + `click`; `openai` only via the extra, imported lazily. No new third-party deps.
- The AI **never** picks a code/class navigating the tree in `describe`; classification of content→class is our code (`concept_map`) or a separate text-only `pick_class`.
- `primary_subject` is the level-2 label; the 9 top-level classes stay fixed; codes stay strings `^\d+(~\d+)*$`, append-only. Reuse `taxonomy.resolve_or_create` (call it with a top-level class and no `sub_parent`).
- Preserve every invariant: originals read-only; copy→verify→catalog ordering; the 43-char digest counting-back; catalog on the local volume; dry-run does zero AI calls / zero taxonomy writes.
- Normalization is shared: the concept-map and learned store key on `taxonomy._normalize_label`.
- Tests offline/deterministic — no network, no real `openai`; the AI is stubbed or the client mocked.
- `uv run pytest`; do NOT chain shell commands with `&&`. Match existing code/test style.

---

## File Structure

**Modify:**
- `imageharbor/catalog.py` — `learned_concepts` table + `learned_concept_get` / `learned_concept_remember`.
- `imageharbor/ai_classifier.py` — add `ContentDescription`, `describe`, `pick_class` (Task 3); remove `PhotoClassification`/`classify` (Task 4).
- `imageharbor/pipeline.py` — orchestrate describe → concept-map/pick_class → `resolve_or_create(class, subject)` (Task 4).
- Tests: `tests/test_catalog.py`, `tests/test_ai_classifier.py`, `tests/test_pipeline.py`, `tests/test_cli.py`.
- `CLAUDE.md` (Task 5).

**Create:**
- `imageharbor/concept_map.py` + `tests/test_concept_map.py`.

**Untouched:** `imageharbor/taxonomy.py` (the 2-level behavior emerges from how the pipeline calls it; level-3 code paths simply go unused), `pcs.py`, `filename.py`, `hashing.py`, `discovery.py`.

---

## Task 1: `learned_concepts` table + catalog access

**Files:** Modify `imageharbor/catalog.py`; Test `tests/test_catalog.py`.

**Interfaces:**
- Produces: `Catalog.learned_concept_get(subject: str) -> str | None`; `Catalog.learned_concept_remember(subject: str, class_code: str) -> None`.

- [ ] **Step 1: Write failing tests**
```python
def test_learned_concept_roundtrip(catalog: Catalog) -> None:
    assert catalog.learned_concept_get("marching band") is None
    catalog.learned_concept_remember("marching band", "500")
    assert catalog.learned_concept_get("marching band") == "500"

def test_learned_concept_remember_updates_and_counts(catalog: Catalog) -> None:
    catalog.learned_concept_remember("widget", "200")
    catalog.learned_concept_remember("widget", "300")  # correction / re-seen
    assert catalog.learned_concept_get("widget") == "300"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_catalog.py -k learned_concept -v`
Expected: FAIL — no such methods.

- [ ] **Step 3: Implement**

Append to `_SCHEMA`:
```sql

CREATE TABLE IF NOT EXISTS learned_concepts (
    subject     TEXT    PRIMARY KEY,
    class_code  TEXT    NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT
);
```
Add methods to `Catalog`:
```python
    def learned_concept_get(self, subject: str) -> str | None:
        cur = self._conn.execute(
            "SELECT class_code FROM learned_concepts WHERE subject=?", (subject,)
        )
        row = cur.fetchone()
        return row["class_code"] if row else None

    def learned_concept_remember(self, subject: str, class_code: str) -> None:
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO learned_concepts (subject, class_code, hits, created_at, updated_at)
            VALUES (?,?,1,?,?)
            ON CONFLICT(subject) DO UPDATE SET
                class_code = excluded.class_code,
                hits       = hits + 1,
                updated_at = excluded.updated_at
            """,
            (subject, class_code, now, now),
        )
        self._conn.commit()
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_catalog.py -q` → PASS.
- [ ] **Step 5: Commit**
```bash
git add imageharbor/catalog.py tests/test_catalog.py
git commit -m "feat: add learned_concepts store to catalog"
```

---

## Task 2: `concept_map.py` (static seed + learned lookup)

**Files:** Create `imageharbor/concept_map.py`; Test `tests/test_concept_map.py`.

**Interfaces:**
- Consumes: Task-1 catalog methods; `pcs.PCS_CATEGORIES`; `taxonomy._normalize_label`.
- Produces: `class_for(primary_subject: str, objects: list[str], scene: str, catalog: Catalog) -> str | None`; `remember(catalog: Catalog, primary_subject: str, class_code: str) -> None`; `STATIC_SEED: dict[str, str]`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_concept_map.py`:
```python
from __future__ import annotations
from pathlib import Path
import pytest
from imageharbor.catalog import Catalog
from imageharbor import concept_map


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "c.db")
    yield cat
    cat.close()


def test_static_seed_from_pcs_subnames() -> None:
    # legacy sub-category "beach" (330) -> its class 300; "portraits" -> 100
    assert concept_map.STATIC_SEED["beach"] == "300"
    assert concept_map.STATIC_SEED["portrait"] == "100"  # normalized (no trailing s)


def test_class_for_static_hit(catalog: Catalog) -> None:
    assert concept_map.class_for("beach", [], "", catalog) == "300"
    assert concept_map.class_for("marching band", [], "", catalog) == "500"  # synonym


def test_class_for_object_or_scene_keyword(catalog: Catalog) -> None:
    assert concept_map.class_for("xyzzy", ["dog"], "", catalog) == "200"
    assert concept_map.class_for("xyzzy", [], "at the beach", catalog) == "300"


def test_class_for_miss_returns_none(catalog: Catalog) -> None:
    assert concept_map.class_for("qwertyunknownthing", [], "", catalog) is None


def test_learned_beats_static_and_roundtrips(catalog: Catalog) -> None:
    concept_map.remember(catalog, "Beach", "600")  # user/AI override
    assert concept_map.class_for("beach", [], "", catalog) == "600"  # learned wins
    concept_map.remember(catalog, "novel gizmo", "200")
    assert concept_map.class_for("novel gizmo", [], "", catalog) == "200"
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_concept_map.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `imageharbor/concept_map.py`**
```python
"""Deterministic-first concept -> class mapping (self-learning).

A static seed (bootstrapped from the legacy PCS sub-categories) maps common
subject/object keywords to one of the 9 fixed top-level classes. A learned store
in the catalog memoizes every AI-decided subject->class, so repeats become
deterministic hits and stop calling the AI. Genuine misses return None and the
pipeline falls through to the text-only pick_class step.
"""
from __future__ import annotations

from .catalog import Catalog
from .pcs import PCS_CATEGORIES
from .taxonomy import _normalize_label


def _build_static_seed() -> dict[str, str]:
    seed: dict[str, str] = {}
    # Every legacy sub-category name -> its top-level class code.
    for code, cat in PCS_CATEGORIES.items():
        if cat.parent is not None:
            seed[_normalize_label(cat.name)] = str((code // 100) * 100)
    # Curated common subjects / synonyms.
    extra = {
        "band": "500", "marching band": "500", "parade": "500", "wedding": "500",
        "concert": "500", "party": "500", "festival": "500", "graduation": "500",
        "game": "500", "match": "500",
        "dog": "200", "cat": "200", "puppy": "200", "kitten": "200",
        "horse": "200", "fish": "200",
        "sunset": "600", "sunrise": "600", "flower": "600", "tree": "600",
        "forest": "600", "river": "600", "lake": "600", "waterfall": "600",
        "car": "400", "truck": "400", "boat": "400", "plane": "400",
        "train": "400", "bicycle": "400", "motorcycle": "400",
        "baby": "100", "child": "100", "family": "100", "selfie": "100",
        "receipt": "700", "document": "700", "sign": "700", "menu": "700",
        "house": "800", "building": "800", "church": "800", "bridge": "800",
        "food": "900", "meal": "900",
    }
    for k, v in extra.items():
        seed.setdefault(_normalize_label(k), v)
    return seed


STATIC_SEED: dict[str, str] = _build_static_seed()


def class_for(
    primary_subject: str, objects: list[str], scene: str, catalog: Catalog
) -> str | None:
    """Return a top-level class code for this content, or None (a miss)."""
    subj = _normalize_label(primary_subject)
    # 1. Learned store wins (exact normalized subject).
    learned = catalog.learned_concept_get(subj)
    if learned:
        return learned
    # 2. Static seed: the subject, then object/scene keywords.
    if subj in STATIC_SEED:
        return STATIC_SEED[subj]
    for token in list(objects) + scene.split():
        norm = _normalize_label(token)
        if norm in STATIC_SEED:
            return STATIC_SEED[norm]
    return None


def remember(catalog: Catalog, primary_subject: str, class_code: str) -> None:
    """Memoize an AI-decided subject -> class so the next repeat is deterministic."""
    catalog.learned_concept_remember(_normalize_label(primary_subject), class_code)
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_concept_map.py -q` then `uv run pytest -q` → PASS.
- [ ] **Step 5: Commit**
```bash
git add imageharbor/concept_map.py tests/test_concept_map.py
git commit -m "feat: self-learning concept-map (static PCS seed + learned store)"
```

---

## Task 3: Add `describe` / `pick_class` to the classifier (additive)

**Files:** Modify `imageharbor/ai_classifier.py`; Test `tests/test_ai_classifier.py`.

Keep `PhotoClassification`/`classify` for now (removed in Task 4) so the branch stays green.

**Interfaces:**
- Produces: `ContentDescription` dataclass; `StubClassifier.describe(image_path, exif) -> ContentDescription` + `pick_class(content, classes) -> str`; `OpenAIClassifier.describe`/`pick_class`; ABC gains a default `pick_class` (returns `"900"`).

- [ ] **Step 1: Write failing tests**
```python
def test_content_description_and_stub_describe() -> None:
    from imageharbor.ai_classifier import ContentDescription, StubClassifier
    c = StubClassifier().describe(Path("marching_band_2007.jpg"), {})
    assert isinstance(c, ContentDescription)
    assert c.primary_subject == "marching"   # first >1-char word of the stem
    # deterministic
    assert StubClassifier().describe(Path("marching_band_2007.jpg"), {}).primary_subject == "marching"

def test_stub_pick_class_default_900() -> None:
    from imageharbor.ai_classifier import StubClassifier, ContentDescription
    c = ContentDescription(primary_subject="mystery")
    assert StubClassifier().pick_class(c, [("100", "people"), ("900", "miscellaneous")]) == "900"

def test_openai_describe_parses_content(monkeypatch, tiny_image) -> None:
    # fake client returns content JSON; assert ContentDescription fields
    ...  # follow the existing _install_fake_openai / Mock client pattern
def test_openai_pick_class_returns_valid_code(monkeypatch, tiny_image) -> None:
    # fake client returns "500"; assert pick_class returns "500", and an invalid reply -> "900"
    ...
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_ai_classifier.py -k "describe or pick_class or content_description" -v` → FAIL.

- [ ] **Step 3: Implement**

Add the dataclass (near `PhotoClassification`):
```python
@dataclass
class ContentDescription:
    """Pure perception output — what is in the photo, no taxonomy knowledge."""
    primary_subject: str = "photo"     # 1-3 words, the main subject
    scene: str = ""                    # setting/context
    objects: list[str] = field(default_factory=list)
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    ocr_text: str = ""
    model_version: str = "stub-1.0"
```

On the `AIClassifier` ABC, add a non-abstract default:
```python
    def pick_class(self, content: "ContentDescription", classes: list[tuple[str, str]]) -> str:
        """Pick the best class CODE from `classes`. Default: miscellaneous."""
        return "900"
```
(Do NOT add `describe` to the ABC yet — that becomes abstract in Task 4 when `classify` is removed. Implement `describe` on the concrete classes only for now.)

`StubClassifier`:
```python
    def describe(self, image_path: Path, exif_data: dict[str, Any]) -> ContentDescription:
        stem = image_path.stem.lower()
        words = [w for w in re.sub(r"[^a-z0-9]+", " ", stem).split() if len(w) > 1]
        primary = words[0] if words else "photo"
        return ContentDescription(
            primary_subject=primary,
            caption=f"Stub description for {image_path.name}",
            tags=words[:3],
            model_version=self.MODEL_VERSION,
        )
    # pick_class inherits the ABC default (900) — deterministic, no network.
```

`OpenAIClassifier`:
```python
    def describe(self, image_path: Path, exif_data: dict[str, Any]) -> ContentDescription:
        import base64
        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        suffix = image_path.suffix.lower().lstrip(".")
        media_type = _MEDIA_TYPES.get(suffix, "image/jpeg")
        system = (
            "You are a photo describer. Look at the image and respond ONLY with a "
            "JSON object with keys: primary_subject (1-3 words), scene (short), "
            "objects (array), caption (one sentence), tags (array), ocr_text (string). "
            "Do NOT categorize or classify — only describe what you see."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}", "detail": "low"}},
                    {"type": "text", "text": "Describe this image."},
                ]},
            ],
            max_tokens=400,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OpenAI describe returned invalid JSON: %s", raw)
            data = {}
        def _list(v):
            return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
        return ContentDescription(
            primary_subject=str(data.get("primary_subject") or "photo"),
            scene=str(data.get("scene", "")),
            objects=_list(data.get("objects", [])),
            caption=str(data.get("caption", "")),
            tags=_list(data.get("tags", [])),
            ocr_text=str(data.get("ocr_text", "")),
            model_version=self.MODEL_VERSION,
        )

    def pick_class(self, content: ContentDescription, classes: list[tuple[str, str]]) -> str:
        options = "\n".join(f"{code}: {label}" for code, label in classes)
        prompt = (
            "Pick the single best top-level class for this photo description.\n"
            f"subject: {content.primary_subject}\nscene: {content.scene}\n"
            f"objects: {content.objects}\ncaption: {content.caption}\n"
            f"classes:\n{options}\n"
            "Reply with ONLY the class code (e.g. 500)."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
        )
        ans = (resp.choices[0].message.content or "").strip()
        valid = {code for code, _ in classes}
        for tok in re.findall(r"\d+", ans):
            if tok in valid:
                return tok
        return "900"
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_ai_classifier.py -q` then `uv run pytest -q` → PASS (classify still present).
- [ ] **Step 5: Commit**
```bash
git add imageharbor/ai_classifier.py tests/test_ai_classifier.py
git commit -m "feat: add describe/pick_class perception contract (additive)"
```

---

## Task 4: Rewire the pipeline; retire `classify`/`PhotoClassification`

**Files:** Modify `imageharbor/pipeline.py`, `imageharbor/ai_classifier.py`; Test `tests/test_pipeline.py`, `tests/test_ai_classifier.py`, `tests/test_cli.py`.

**Interfaces:**
- Consumes: `describe`/`pick_class` (Task 3), `concept_map.class_for`/`remember` (Task 2), `taxonomy.resolve_or_create(class, label, adjudicator=)`.
- Produces: pipeline organizes into `class/primary_subject`; the old `classify`/`PhotoClassification` are gone.

- [ ] **Step 1: Write/adjust failing tests**

In `tests/test_pipeline.py` (fixtures exist):
```python
def test_pipeline_organizes_by_subject(source_dir, organized_dir, catalog) -> None:
    # beach_photo.jpg -> subject "beach" -> concept-map class 300 -> 300-places/<code>-beach
    Pipeline(source_dir, organized_dir, catalog).run()
    paths = [p.as_posix() for p in organized_dir.rglob("*.jpg")]
    assert any("/300-places/" in p and "-beach_" in p for p in paths)
    # exactly 2 levels deep under the class (class/subject/file)
    for p in organized_dir.rglob("*.jpg"):
        rel = p.relative_to(organized_dir)
        assert len(rel.parts) == 3  # class / subject / filename

def test_pipeline_learns_class_on_miss(organized_dir, catalog, tmp_path) -> None:
    from imageharbor.ai_classifier import AIClassifier, ContentDescription
    calls = []
    class MissClassifier(AIClassifier):
        def describe(self, image_path, exif_data):
            return ContentDescription(primary_subject="zonkle")  # not in concept-map
        def pick_class(self, content, classes):
            calls.append(content.primary_subject)
            return "200"
    src = tmp_path / "s"; src.mkdir()
    _make_jpeg(src / "a.jpg"); _make_jpeg(src / "b.jpg", b"\xff\xd8\xff\xe0" + b"\x02"*20 + b"\xff\xd9")
    Pipeline(src, organized_dir, catalog, classifier=MissClassifier()).run()
    assert calls == ["zonkle"]  # pick_class called ONCE; 2nd file was a learned hit
    assert any("/200-animals/" in p.as_posix() for p in organized_dir.rglob("*.jpg"))
```
Remove/replace the old `test_pipeline_mints_new_category`/`test_pipeline_uses_taxonomy_codes` that used the `top_parent/label` contract. In `tests/test_ai_classifier.py`, delete tests referencing `PhotoClassification`/`classify`.

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_pipeline.py -k "subject or learns" -v` → FAIL.

- [ ] **Step 3: Implement**

In `imageharbor/ai_classifier.py`: delete `PhotoClassification` and the abstract `classify` method; promote `describe` to an `@abstractmethod` on the ABC:
```python
    @abstractmethod
    def describe(self, image_path: Path, exif_data: dict[str, Any]) -> ContentDescription: ...
```
Remove the old `_SYSTEM_PROMPT`/`_USER_PROMPT`/`_build_pcs_list` only if now unused (grep first).

In `imageharbor/pipeline.py`:
- Import: `from . import concept_map`.
- Replace the classify/resolve block in `_do_process` (after EXIF) with:
```python
        # Perception: the AI only describes the image.
        content = self.classifier.describe(source_path, exif_data)

        # Organization: our code decides the class (concept-map first, AI fallback).
        cls = concept_map.class_for(
            content.primary_subject, content.objects, content.scene, self.catalog
        )
        if cls is None:
            cls = self.classifier.pick_class(content, self._classes())
            concept_map.remember(self.catalog, content.primary_subject, cls)

        # primary_subject is the level-2 label.
        pcs_code = self.taxonomy.resolve_or_create(
            cls, content.primary_subject, adjudicator=self.classifier.adjudicate
        )
        node = self.taxonomy.get(pcs_code)
        pcs_name = node.label if node else content.primary_subject

        descriptor = normalize_descriptor(content.primary_subject)
        extension = source_path.suffix.lstrip(".").lower()
        filename = generate_filename(pcs_code, descriptor, sha256_b64url, extension)
        organized_path = self.organized_dir / self.taxonomy.folder_path(pcs_code) / filename
```
- Add a helper on `Pipeline` for the 9 classes (from the seeded taxonomy):
```python
    def _classes(self) -> list[tuple[str, str]]:
        return [(n.code, n.label) for n in self.taxonomy.children(None)]
```
- Update `_update_catalog` to store the content fields: `ai_caption=content.caption`, `objects=content.objects`, `secondary_tags=content.tags`, `ocr_text=content.ocr_text` (and keep `exif`, `model_version=content.model_version`). Update its parameters/call accordingly. `_write_sidecar` similarly reads from `content`.
- `resolve_or_create` is called with no `sub_parent` (2-level); no signature change needed.

Update `tests/test_cli.py` only if it asserted old classifier fields (grep `PhotoClassification`, `top_parent`, `classify(`).

- [ ] **Step 4: Run focused, then full**

Run: `uv run pytest tests/test_pipeline.py tests/test_ai_classifier.py tests/test_cli.py -q`
Then: `uv run pytest -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add imageharbor/pipeline.py imageharbor/ai_classifier.py tests/
git commit -m "feat: pipeline describes then organizes via concept-map; retire classify"
```

---

## Task 5: Docs — CLAUDE.md

**Files:** Modify `CLAUDE.md`.

- [ ] **Step 1: Update** the `ai_classifier.py` and `pipeline.py` bullets and the invariants:
  - The classifier does **perception only** (`describe` → `ContentDescription`); it never picks a class/code. `pick_class` is a text-only fallback.
  - **`concept_map.py`** decides the class (static PCS-seed + self-learning `learned_concepts`); the AI `pick_class` only fires on a miss and the decision is memoized.
  - The taxonomy is effectively **two levels** now (fixed class → `primary_subject`); `resolve_or_create` is called with a top-level class and no `sub_parent`.
  - Add a `concept_map.py` bullet to the Architecture module list; update the pipeline flow line.

- [ ] **Step 2: Commit**
```bash
git add CLAUDE.md
git commit -m "docs: update for the perception/organization reframe"
```

---

## Self-Review

**Spec coverage:** §3 perception contract → Task 3; §4 concept-map (static+learned) → Tasks 1-2; §5 taxonomy 2-level → emerges in Task 4 (no taxonomy change, `resolve_or_create(class, subject)` with no `sub_parent`); §6 pipeline orchestration → Task 4; §9 tests → each task; §11 acceptance → Tasks 1-4 + full-suite run in Task 4.

**Placeholder scan:** the OpenAI `describe`/`pick_class` tests in Task 3 Step 1 are sketched (`...`) — the implementer follows the existing `_install_fake_openai`/`Mock` client pattern already in `tests/test_ai_classifier.py`; every other step has complete code. No TODO/TBD in shipped code.

**Type consistency:** `ContentDescription` fields are defined in Task 3 and consumed identically in Task 4 (`primary_subject`, `objects`, `scene`, `caption`, `tags`, `ocr_text`); `concept_map.class_for(primary_subject, objects, scene, catalog)` and `remember(catalog, subject, class)` match their Task-4 call sites; `pick_class(content, classes)` and `_classes()` returning `list[tuple[code,label]]` match; `resolve_or_create(class, label, adjudicator=)` matches the existing signature (sub_parent defaulted/unused).

**Deferred (spec §10):** dashboard, embeddings, physically relocating merged files — not planned.
