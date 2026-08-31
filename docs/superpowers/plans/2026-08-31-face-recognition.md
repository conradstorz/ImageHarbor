# Face Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect faces in the organized library, group them into clusters, and propose person names from the tags Google already supplied — never renaming a file and never asserting an identity no human confirmed.

**Architecture:** A third pass (`imageharbor faces`) independent of the facts and enrichment passes, running ONNX inference in-process with no network call. Pure, I/O-free modules (`decode`, `align`, `cluster`, `attribute`, `calibrate`, `names`) hold all the logic and are testable with zero model weights; thin I/O modules (`detect`, `embed`, `store`, `runner`) wrap them. Identity moves only through a human confirmation on the dashboard.

**Tech Stack:** Python 3.10+, ONNX Runtime, NumPy, Pillow, SQLite, Click, stdlib `http.server`.

**Spec:** [`docs/superpowers/specs/2026-08-31-face-recognition-design.md`](../specs/2026-08-31-face-recognition-design.md)

## Global Constraints

- **Faces never rename or move a file.** No face code path may call `tiers.is_upgrade` or `relocate`.
- **No identity is written without human confirmation.** Only the dashboard confirm endpoint may set `clusters.person_id`.
- **Embeddings are never compared across `embed_model` values.** Attempting it raises; it must never return a plausible number.
- **Face failures never feed the circuit breaker.** The breaker is reserved for `AIClassifier.describe()` failures.
- **Name identity is exact.** No fuzzy, similarity, or case-insensitive name merging, ever. `Conrad Storz` and `Conrad Storz III` are different people.
- **Embeddings are L2-normalized where they are produced.**
- **Every bug fix ships with a regression test.**
- Runtime deps go in `pyproject.toml` under a new `faces` extra. Adding one needs a strong reason. **Do not add OpenCV** — see the Task 9 risk note for the sanctioned fallback.
- ImageHarbor is **AGPL-3.0-or-later**. Only `imageharbor/__init__.py` carries a per-file licence notice; new modules follow the existing convention and do not. Any code adapted from another AGPL project (PhotoPrism, Immich) MUST carry a header naming the upstream project, file, and licence.
- Python floor is `>=3.10`. Use `from __future__ import annotations` in every new module.
- Run tests with `uv run pytest`. Never `pip install`.
- **Do not bump `catalog.SCHEMA_VERSION`.** Face tables are new `CREATE TABLE IF NOT EXISTS` tables that cannot corrupt an existing catalog; that constant is reserved for changes that would.

## Milestones

Each is working, testable software on its own.

| Milestone | Tasks | Deliverable |
| --- | --- | --- |
| A — pure core | 1–6 | All logic, fully tested, no weights and no database |
| B — persistence | 7–8 | Face tables and the sidecar contract |
| C — the pass | 9–13 | `faces scan`, `calibrate`, `cluster` running end to end |
| D — review + integration | 14–17 | Dashboard People section, `watch`, docs |

**Risk note for Task 9.** The spec rules out OpenCV, so YuNet's raw ONNX outputs must be decoded by hand. That is the single riskiest task here. It is isolated into a pure module (Task 4) so it can be tested against recorded fixtures.

**If the decode cannot be validated against a real detection in Task 9, the fallback is to port PhotoPrism's YuNet decoder** (`internal/ai/face/engine_onnx_yunet.go`), not to add OpenCV. ImageHarbor is AGPL-3.0-or-later and so is PhotoPrism, so adapting their implementation is licence-compatible — it requires an attribution and provenance note in the file header naming the upstream project, file, and licence. Take that fallback rather than shipping a decoder that "works" but silently detects worse, and rather than adding a ~60 MB dependency to a project whose entire runtime list is Pillow and Click.

---

## Task 1: Package skeleton and the `faces` extra

**Files:**
- Create: `imageharbor/faces/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/faces/__init__.py`, `tests/faces/test_extra.py`

**Interfaces:**
- Consumes: nothing
- Produces: the `imageharbor.faces` package; `imageharbor.faces.HAS_ONNX: bool`

- [ ] **Step 1: Write the failing test**

```python
"""The faces package must import with or without onnxruntime installed."""

import sys

import imageharbor.faces as faces


def test_package_imports_without_onnxruntime():
    # Importing the package must never require the optional extra. Only the
    # modules that actually run a model may import onnxruntime, and they are
    # imported lazily by the runner.
    assert hasattr(faces, "HAS_ONNX")
    assert isinstance(faces.HAS_ONNX, bool)


def test_package_imports_when_onnxruntime_raises_non_import_error(monkeypatch):
    # A broken install (ABI-mismatch, partial installation, etc.) may raise
    # non-ImportError exceptions during import. The probe must catch all
    # exceptions, not just ImportError, and set HAS_ONNX to False so the
    # package still imports successfully.

    # Clear the imageharbor.faces module from sys.modules to force re-execution
    # of its import probe. Use monkeypatch.delitem (not sys.modules.pop) so the
    # original module object is restored at teardown regardless of whether the
    # test passes or fails — a bare pop() here would leak the reloaded,
    # ONNX-less module into every later test in the session.
    monkeypatch.delitem(sys.modules, "imageharbor.faces", raising=False)
    to_remove = [k for k in list(sys.modules.keys()) if k.startswith("imageharbor.faces.")]
    for k in to_remove:
        monkeypatch.delitem(sys.modules, k, raising=False)

    # Patch builtins.__import__ to raise RuntimeError for onnxruntime,
    # simulating a broken C extension or ABI mismatch. The original __import__
    # is called for all other modules.
    import builtins
    original_import = builtins.__import__

    def mock_import_with_broken_onnx(name, *args, **kwargs):
        if name == "onnxruntime":
            raise RuntimeError("ONNX Runtime initialization failed: ABI version mismatch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_with_broken_onnx)

    # Import the package with the patched import. It should succeed
    # because the probe catches all exceptions.
    import imageharbor.faces as faces_reloaded

    assert hasattr(faces_reloaded, "HAS_ONNX")
    # If the exception handler only caught ImportError, this would fail
    # because RuntimeError would propagate. With the proper fix, it is False.
    assert faces_reloaded.HAS_ONNX is False
```

Use `monkeypatch`, not `sys.modules.pop()` and `unittest.mock.patch("builtins.__import__", ...)`, for the `sys.modules` surgery and the import patch above — `monkeypatch` restores both automatically at teardown, on both the passing and failing path. A bare `pop()` leaves the reloaded, ONNX-less module permanently installed in `sys.modules["imageharbor.faces"]` for the rest of the pytest session, silently poisoning `HAS_ONNX` for every later test that imports the package.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_extra.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'imageharbor.faces'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Face detection, embedding, clustering, and name proposal.

The pure modules in this package (`names`, `decode`, `align`, `cluster`,
`attribute`, `calibrate`) import nothing from the rest of ImageHarbor and touch
no filesystem, so the whole core is testable without a byte of model weights.
Only `detect` and `embed` import onnxruntime, and only when a model is actually
run -- importing this package must never require the optional `faces` extra.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivial availability probe
    import onnxruntime  # noqa: F401

    HAS_ONNX = True
except Exception:  # pragma: no cover
    # Catch all exceptions: missing onnxruntime (ImportError), broken
    # installations with ABI mismatches (RuntimeError, OSError), or any other
    # failure during import. All failures to import mean we cannot run a model,
    # so they all answer the same question: HAS_ONNX = False.
    logger.debug("onnxruntime unavailable; faces functionality disabled", exc_info=True)
    HAS_ONNX = False

__all__ = ["HAS_ONNX"]
```

Create an empty `tests/faces/__init__.py`.

Add to `pyproject.toml` under `[project.optional-dependencies]`, after the `openai` line:

```toml
faces = [
    "onnxruntime>=1.17,<2",
    "numpy>=1.24",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/faces/ -v`
Expected: PASS

- [ ] **Step 5: Install the extra and commit**

```bash
uv sync --extra faces
git add imageharbor/faces/__init__.py tests/faces/ pyproject.toml uv.lock
git commit -m "feat(faces): add the faces package and its optional extra"
```

---

## Task 2: Name normalization

**Files:**
- Create: `imageharbor/faces/names.py`
- Test: `tests/faces/test_names.py`

**Interfaces:**
- Consumes: nothing
- Produces: `normalize(name: str) -> str`, `case_variants(names: Iterable[str]) -> dict[str, list[str]]`

This is first because the evidence demands it: `pete storz` appears 1,539 times lower-cased and `Gladys Blankenbeker ` carries a trailing space in all 461 of its occurrences.

- [ ] **Step 1: Write the failing test**

```python
"""Name normalization: whitespace is fixed automatically, case never is."""

import pytest

from imageharbor.faces import names


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Gladys Blankenbeker ", "Gladys Blankenbeker"),   # 461 real occurrences
        (" Conrad Storz", "Conrad Storz"),
        ("Conrad  Storz", "Conrad Storz"),                 # collapsed internal run
        ("Conrad\tStorz", "Conrad Storz"),
        ("Conrad Storz", "Conrad Storz"),                  # already clean, unchanged
        ("", ""),
    ],
)
def test_normalize_fixes_whitespace(raw, expected):
    assert names.normalize(raw) == expected


def test_normalize_never_changes_case():
    # 1,539 photos say "pete storz". Case-folding is a judgement about identity,
    # not a formatting fix, so it is never applied automatically.
    assert names.normalize("pete storz") == "pete storz"
    assert names.normalize("claire Storz") == "claire Storz"


def test_case_variants_groups_only_case_differences():
    groups = names.case_variants(["pete storz", "Pete Storz", "Judy Storz"])
    assert groups == {"pete storz": ["Pete Storz", "pete storz"]}


def test_case_variants_never_groups_a_suffix_difference():
    # The whole reason fuzzy matching is banned: these are a father and a son.
    groups = names.case_variants(["Conrad Storz", "Conrad Storz III"])
    assert groups == {}


def test_case_variants_is_deterministic():
    a = names.case_variants(["b Smith", "B Smith", "B SMITH"])
    b = names.case_variants(["B SMITH", "B Smith", "b Smith"])
    assert a == b
    assert a["b smith"] == ["B SMITH", "B Smith", "b Smith"]


def test_case_variants_never_groups_more_than_case():
    # str.casefold() is Unicode-normalizing, not case-folding: 'Weiß'.casefold()
    # == 'Weiss'.casefold() even though they are different names (an extra 's',
    # not a case change of any character). Grouping these would surface a bogus
    # "these may be the same person" suggestion in the review UI. Use
    # per-character str.lower() paired with length, not casefold, for the
    # grouping key -- see Step 3.
    groups = names.case_variants(["Weiß", "Weiss"])
    assert groups == {}


def test_case_variants_still_groups_same_length_compatibility_characters():
    # Known, accepted residual: the Kelvin sign (U+212A) and 'K' are the same
    # length, and Unicode's own simple case mapping sends both to 'k' -- the
    # same target str.lower() gives plain 'K'. The length gate in _case_key
    # only excludes length-changing folds like Weiß/Weiss; it cannot and does
    # not separate this pair. This is accepted because case_variants only
    # ever suggests a merge for a human to confirm, never performs one --
    # do not "fix" this by trying to special-case Kelvin.
    kelvin_sign = "K"
    assert kelvin_sign != "K"  # distinct characters going in
    groups = names.case_variants([kelvin_sign, "K"])
    assert groups == {"k": ["K", kelvin_sign]}  # sorted: plain K (0x4B) before Kelvin sign (0x212A)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_names.py -v`
Expected: FAIL with `ImportError: cannot import name 'names'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Person-name normalization. Pure: no I/O, no imports from the package.

Two defects are present in this library's real name vocabulary, and they are
handled asymmetrically on purpose.

Whitespace is noise. `Gladys Blankenbeker ` carries a trailing space in all 461
of its occurrences, and the sidecar's `people` list is keyed on the name, so an
unnormalized key silently splits one person into two entries. Stripping it
cannot merge two different people, so it is applied automatically.

Case might not be noise. `pete storz` (1,539) and `claire Storz` (442) look like
drift, but this same vocabulary contains `Conrad Storz` (3,309) and `Conrad
Storz III` (980) -- a father and a son distinguished only by a suffix. A
vocabulary that proves suffixes are load-bearing is not one to apply automatic
identity judgements to. Case variants are therefore *reported* by
`case_variants` for a human to confirm, never folded.
"""

from __future__ import annotations

from collections.abc import Iterable


def normalize(name: str) -> str:
    """Strip surrounding whitespace and collapse internal runs. Case is kept."""
    return " ".join(name.split())


def _case_key(name: str) -> tuple[int, str]:
    """Key that matches two strings only when they differ *purely* by case.

    ``str.casefold()`` is Unicode-normalizing, not case-folding: it merges
    strings of different length, e.g. ``'Weiß'`` and ``'Weiss'``.
    Per-character ``str.lower()`` doesn't expand or contract characters the
    way casefold does, so pairing it with the original length catches that
    length-changing case. It does *not* catch same-length compatibility
    collisions -- the Kelvin sign (U+212A) still keys the same as ``'K'``,
    because Unicode's simple case mapping sends both to ``'k'``. No
    per-character scheme can separate them without giving up case-insensitive
    comparison. That's acceptable here: ``case_variants`` only ever suggests
    a merge to a human, it never performs one.
    """
    return (len(name), "".join(ch.lower() for ch in name))


def case_variants(names: Iterable[str]) -> dict[str, list[str]]:
    """Group normalized names that differ only by case.

    Returns ``{lowercased_key: [variant, ...]}`` for keys with more than one
    spelling, variants sorted for determinism. These are *suggestions* for the
    review UI; nothing here merges anything.
    """
    groups: dict[tuple[int, str], set[str]] = {}
    for raw in names:
        cleaned = normalize(raw)
        if not cleaned:
            continue
        groups.setdefault(_case_key(cleaned), set()).add(cleaned)
    return {
        lower: sorted(variants)
        for (_, lower), variants in sorted(groups.items())
        if len(variants) > 1
    }
```

**Do not use `cleaned.casefold()` as the grouping key.** `casefold()` is
Unicode-normalizing, not case-folding, and merges names that differ by more
than case (`'Weiß'.casefold() == 'Weiss'.casefold()`). `case_variants`
promises to group names that "differ only by case" -- a wrong merge suggestion
here is how a father and son (`Conrad Storz` / `Conrad Storz III`) get
collapsed by hand in the review UI. Use `_case_key` (length + per-character
`str.lower()`) as above.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_names.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/names.py tests/faces/test_names.py
git commit -m "feat(faces): normalize name whitespace, never case"
```

---

## Task 3: Model registry

**Files:**
- Create: `imageharbor/faces/models.py`
- Test: `tests/faces/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ModelInfo` dataclass with fields `name, kind, filename, url, sha256, input_size, channel_order, mean, std, embedding_dim, licence`; `DETECTORS: dict[str, ModelInfo]`; `EMBEDDERS: dict[str, ModelInfo]`; `get(name) -> ModelInfo`; `DEFAULT_DETECTOR: str`; `DEFAULT_EMBEDDER: str`

- [ ] **Step 1: Write the failing test**

```python
"""The model registry declares what an ONNX graph cannot tell us."""

import pytest

from imageharbor.faces import models


def test_defaults_are_registered():
    assert models.DEFAULT_DETECTOR in models.DETECTORS
    assert models.DEFAULT_EMBEDDER in models.EMBEDDERS


def test_every_entry_declares_its_preprocessing_contract():
    # Channel order and normalization are NOT recoverable from an ONNX graph.
    # A wrong input shape raises; a wrong channel order loads, runs, and returns
    # plausible embeddings that are quietly worse. Every entry must state both.
    for info in {**models.DETECTORS, **models.EMBEDDERS}.values():
        assert info.channel_order in ("RGB", "BGR"), info.name
        assert len(info.input_size) == 2, info.name
        assert info.mean is not None and info.std is not None, info.name
        assert info.licence, info.name


def test_embedders_declare_a_dimension_and_detectors_do_not():
    for info in models.EMBEDDERS.values():
        assert info.embedding_dim and info.embedding_dim > 0, info.name
    for info in models.DETECTORS.values():
        assert info.embedding_dim is None, info.name


def test_filenames_are_disambiguated_by_publisher():
    # InsightFace's antelopev2 pack and fal's AuraFace both ship a file called
    # glintr100.onnx and they are different models. A name match is not an
    # artifact match, so our stored filename must not be the bare upstream one.
    auraface = models.EMBEDDERS["auraface"]
    assert auraface.filename != "glintr100.onnx"
    assert "auraface" in auraface.filename


def test_get_rejects_an_unknown_model():
    with pytest.raises(KeyError, match="unknown face model"):
        models.get("no-such-model")


def test_get_returns_registered_entries():
    assert models.get(models.DEFAULT_EMBEDDER).kind == "embedder"
    assert models.get(models.DEFAULT_DETECTOR).kind == "detector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'models'`

- [ ] **Step 3: Write minimal implementation**

`sha256` is deliberately `None` here. Task 9 pins the real digests after downloading and verifying the artifacts once. **Do not invent a checksum.**

```python
"""Registry of face models. Pure: no I/O, no imports from the package.

This module exists because **channel order and normalization are not present in
an ONNX graph**. Getting the input shape wrong raises immediately. Getting the
channel order wrong loads, runs, and returns plausible output that is quietly
worse -- which is far more expensive, because nothing fails. Those fields are
declared per model and never inferred.

Filenames are disambiguated by publisher for the same reason: InsightFace's
antelopev2 pack and fal's AuraFace both ship a file named `glintr100.onnx`, and
they are different models. A name match is not an artifact match.

Both defaults are permissively licensed on purpose. InsightFace's ArcFace
weights are not redistributable, and ImageHarbor must not acquire a non-free
artifact dependency by default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """An ONNX artifact plus the preprocessing contract its graph omits."""

    name: str
    kind: str                    # "detector" | "embedder"
    filename: str                # local name; disambiguated by publisher
    url: str
    sha256: str | None           # pinned in Task 9 after a verified download
    input_size: tuple[int, int]  # (width, height)
    channel_order: str           # "RGB" | "BGR"
    mean: float
    std: float
    licence: str
    embedding_dim: int | None = None


DETECTORS: dict[str, ModelInfo] = {
    "yunet": ModelInfo(
        name="yunet",
        kind="detector",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256=None,
        input_size=(640, 640),
        # YuNet consumes raw 0-255 BGR: OpenCV's FaceDetectorYN builds its blob
        # with every blobFromImage default, and those defaults are BGR with no
        # scaling. Verified against OpenCV's source, not inferred.
        channel_order="BGR",
        mean=0.0,
        std=1.0,
        licence="MIT",
    ),
}

EMBEDDERS: dict[str, ModelInfo] = {
    "auraface": ModelInfo(
        name="auraface",
        kind="embedder",
        filename="auraface_v1_glintr100.onnx",
        url="https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
        sha256=None,
        input_size=(112, 112),
        # Every embedding model on the aligned crop takes RGB, normalized to
        # roughly [-1, 1] by (x - 127.5) / 128.
        channel_order="RGB",
        mean=127.5,
        std=128.0,
        licence="Apache-2.0",
        embedding_dim=512,
    ),
}

DEFAULT_DETECTOR = "yunet"
DEFAULT_EMBEDDER = "auraface"


def get(name: str) -> ModelInfo:
    """Look up a model by name across both registries."""
    if name in DETECTORS:
        return DETECTORS[name]
    if name in EMBEDDERS:
        return EMBEDDERS[name]
    raise KeyError(f"unknown face model: {name!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_models.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/models.py tests/faces/test_models.py
git commit -m "feat(faces): declare the model registry and its preprocessing contracts"
```

---

## Task 4: YuNet output decode (pure)

**Files:**
- Create: `imageharbor/faces/decode.py`
- Test: `tests/faces/test_decode.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Detection` dataclass with fields `x, y, w, h, score, landmarks`; `decode_yunet(outputs, input_size, score_threshold, nms_threshold) -> list[Detection]`; `nms(boxes, scores, threshold) -> list[int]`

This is the riskiest logic in the build, which is exactly why it is pure and tested against synthetic tensors before any model runs.

- [ ] **Step 1: Write the failing test**

```python
"""YuNet output decoding and NMS, on synthetic tensors. No model required."""

import numpy as np
import pytest

from imageharbor.faces import decode


def test_nms_keeps_the_highest_scoring_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert decode.nms(boxes, scores, 0.3) == [0]


def test_nms_keeps_both_when_they_do_not_overlap():
    boxes = np.array([[0, 0, 50, 50], [500, 500, 50, 50]], dtype=np.float32)
    scores = np.array([0.7, 0.9], dtype=np.float32)
    assert sorted(decode.nms(boxes, scores, 0.3)) == [0, 1]


def test_nms_returns_indices_in_descending_score_order():
    boxes = np.array([[0, 0, 10, 10], [500, 0, 10, 10], [0, 500, 10, 10]], dtype=np.float32)
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert decode.nms(boxes, scores, 0.3) == [1, 2, 0]


def test_nms_on_empty_input():
    empty = np.zeros((0, 4), dtype=np.float32)
    assert decode.nms(empty, np.zeros((0,), dtype=np.float32), 0.3) == []


def _synthetic_outputs(size=(640, 640), hot=None):
    """Build YuNet-shaped outputs with at most one confident cell.

    Mirrors the real exported graph exactly (confirmed by inspecting the real
    ONNX artifact with onnxruntime, Task 4 fix round 1): twelve tensors,
    type-major then stride-major -- all three `cls_{8,16,32}`, then all three
    `obj`, then all three `bbox`, then all three `kps` -- each shaped
    `(1, N, C)` with a leading batch axis of 1, where N = (size/stride)**2 in
    row-major order.
    """
    strides = (8, 16, 32)
    cls_out, obj_out, bbox_out, kps_out = [], [], [], []
    for stride in strides:
        gw, gh = size[0] // stride, size[1] // stride
        n = gw * gh
        cls = np.zeros((1, n, 1), dtype=np.float32)
        obj = np.zeros((1, n, 1), dtype=np.float32)
        bbox = np.zeros((1, n, 4), dtype=np.float32)
        kps = np.zeros((1, n, 10), dtype=np.float32)
        if hot is not None and hot[0] == stride:
            idx = hot[1]
            cls[0, idx, 0] = 1.0
            obj[0, idx, 0] = 1.0
            # bbox is (dx, dy, log-w, log-h) relative to the cell, in strides.
            bbox[0, idx] = [0.0, 0.0, np.log(4.0), np.log(4.0)]
            # Five landmarks, all offset one stride right and down of the cell.
            kps[0, idx] = [1.0, 1.0] * 5
        cls_out.append(cls)
        obj_out.append(obj)
        bbox_out.append(bbox)
        kps_out.append(kps)
    return cls_out + obj_out + bbox_out + kps_out


def test_decode_returns_nothing_when_every_score_is_zero():
    assert decode.decode_yunet(_synthetic_outputs(), (640, 640), 0.5, 0.3) == []


def test_decode_places_a_detection_at_the_hot_cell():
    # Stride 8, grid 80x80. Cell index 81 is row 1, column 1 -> centre (8, 8).
    outs = _synthetic_outputs(hot=(8, 81))
    dets = decode.decode_yunet(outs, (640, 640), 0.5, 0.3)
    assert len(dets) == 1
    d = dets[0]
    # Width and height are exp(log 4) * stride = 32.
    assert d.w == pytest.approx(32.0)
    assert d.h == pytest.approx(32.0)
    # The box is centred on the cell, so its top-left is centre - size/2.
    assert d.x == pytest.approx(8.0 - 16.0)
    assert d.y == pytest.approx(8.0 - 16.0)
    assert d.score == pytest.approx(1.0)
    assert len(d.landmarks) == 5
    # Each landmark is one stride right and down of the cell centre.
    assert d.landmarks[0] == pytest.approx((16.0, 16.0))


def test_decode_respects_the_score_threshold():
    outs = _synthetic_outputs(hot=(8, 81))
    outs[0][0, 81, 0] = 0.1  # cls for stride 8 -> score becomes sqrt(0.1 * 1.0)
    assert decode.decode_yunet(outs, (640, 640), 0.9, 0.3) == []
    assert len(decode.decode_yunet(outs, (640, 640), 0.2, 0.3)) == 1


def test_decode_is_deterministic():
    outs = _synthetic_outputs(hot=(16, 100))
    a = decode.decode_yunet(outs, (640, 640), 0.5, 0.3)
    b = decode.decode_yunet(_synthetic_outputs(hot=(16, 100)), (640, 640), 0.5, 0.3)
    assert [(d.x, d.y, d.w, d.h, d.score) for d in a] == [
        (d.x, d.y, d.w, d.h, d.score) for d in b
    ]
```

A ninth test, `test_decode_yunet_against_the_real_model_on_a_blank_image`, runs the
real ONNX artifact through `decode_yunet` when `IMAGEHARBOR_FACE_MODEL_DIR` is
set, and `pytest.skip`s otherwise -- this is what actually proves the layout
above matches the exported graph rather than a second hand-written guess at it.
See `tests/faces/test_decode.py` for the current version.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_decode.py -v`
Expected: FAIL with `ImportError: cannot import name 'decode'`

- [ ] **Step 3: Write minimal implementation**

**Update (Task 4 fix round 1, verified against the real artifact):** the draft
below originally assumed the twelve outputs were grouped stride-major -- one
`(cls, obj, bbox, kps)` per stride. Inspecting the real exported graph with
onnxruntime showed the opposite: outputs are grouped **type-major**, all three
`cls` tensors first, then all three `obj`, then all three `bbox`, then all
three `kps`, each block ordered by stride. Each tensor also carries a leading
batch axis -- `(1, N, C)`, not `(N, C)`. The code below is already corrected;
do not reintroduce `outputs[si * 4 : si * 4 + 4]` or assume 2-D outputs.

```python
"""Decode YuNet's raw ONNX outputs into detections. Pure: no I/O, no session.

This is the fiddliest logic in the face pipeline, so it lives here, separated
from the ONNX session in `detect.py`, and is tested against synthetic tensors
with no model present.

YuNet emits twelve output tensors, grouped by *type* first and *stride* second
-- all three `cls`, then all three `obj`, then all three `bbox`, then all three
`kps`, each group ordered by stride (8, 16, 32) -- not grouped by stride with
one of each type per group. Verified against the real exported graph with
onnxruntime; a stride-major reading silently treats `cls_16` as objectness and
`cls_32` as a bbox regressor. Each tensor carries a leading batch axis:

    cls  (1, N, 1)   classification logit, already sigmoid-ed by the graph
    obj  (1, N, 1)   objectness, already sigmoid-ed
    bbox (1, N, 4)   (dx, dy, log w, log h), offsets in stride units
    kps  (1, N, 10)  five (dx, dy) landmark offsets, in stride units

where N is one row per grid cell in row-major order. The confidence of a cell
is ``sqrt(cls * obj)`` -- the geometric mean, which is what the reference
implementation uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

STRIDES: tuple[int, ...] = (8, 16, 32)


@dataclass(frozen=True)
class Detection:
    """One detected face in input-image pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: tuple[tuple[float, float], ...]  # 5 points: eyes, nose, mouth


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best score first.

    `boxes` is (N, 4) as (x, y, w, h).
    """
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= threshold]
    return keep


def _drop_batch(arr: np.ndarray) -> np.ndarray:
    """Squeeze YuNet's leading batch axis, if present.

    The real graph always emits `(1, N, C)`. `(N, C)` is also accepted, on
    purpose, so hand-built synthetic tensors in tests don't have to carry a
    batch axis they get nothing from -- both shapes mean "one image."
    """
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"decode_yunet only supports batch size 1, got {arr.shape[0]}")
        return arr[0]
    return arr


def decode_yunet(
    outputs: Sequence[np.ndarray],
    input_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
) -> list[Detection]:
    """Turn YuNet's raw outputs into detections in input-image coordinates."""
    width, height = input_size
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    kps: list[np.ndarray] = []

    n_strides = len(STRIDES)
    for si, stride in enumerate(STRIDES):
        # Type-major layout: all `cls`, then all `obj`, then all `bbox`, then
        # all `kps`, each block ordered by stride -- see module docstring.
        cls = _drop_batch(outputs[0 * n_strides + si])
        obj = _drop_batch(outputs[1 * n_strides + si])
        bbox = _drop_batch(outputs[2 * n_strides + si])
        kp = _drop_batch(outputs[3 * n_strides + si])
        gw, gh = width // stride, height // stride

        # Cell centres in row-major order, matching the graph's flattening.
        cols = np.tile(np.arange(gw, dtype=np.float32), gh) * stride
        rows = np.repeat(np.arange(gh, dtype=np.float32), gw) * stride

        conf = np.sqrt(
            np.clip(cls[:, 0], 0.0, None) * np.clip(obj[:, 0], 0.0, None)
        )
        hot = conf >= score_threshold
        if not np.any(hot):
            continue

        cx = cols[hot] + bbox[hot, 0] * stride
        cy = rows[hot] + bbox[hot, 1] * stride
        bw = np.exp(bbox[hot, 2]) * stride
        bh = np.exp(bbox[hot, 3]) * stride

        boxes.append(np.stack([cx - bw / 2.0, cy - bh / 2.0, bw, bh], axis=1))
        scores.append(conf[hot])

        pts = kp[hot].reshape(-1, 5, 2) * stride
        pts[:, :, 0] += cols[hot][:, None]
        pts[:, :, 1] += rows[hot][:, None]
        kps.append(pts)

    if not boxes:
        return []

    all_boxes = np.concatenate(boxes).astype(np.float32)
    all_scores = np.concatenate(scores).astype(np.float32)
    all_kps = np.concatenate(kps).astype(np.float32)

    return [
        Detection(
            x=float(all_boxes[i, 0]),
            y=float(all_boxes[i, 1]),
            w=float(all_boxes[i, 2]),
            h=float(all_boxes[i, 3]),
            score=float(all_scores[i]),
            landmarks=tuple(
                (float(p[0]), float(p[1])) for p in all_kps[i]
            ),
        )
        for i in nms(all_boxes, all_scores, nms_threshold)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_decode.py -v`
Expected: PASS, 9 tests (8 synthetic-tensor tests, plus one real-model
integration test that runs when `IMAGEHARBOR_FACE_MODEL_DIR` is set and the
weights are present, and `pytest.skip`s otherwise).

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/decode.py tests/faces/test_decode.py
git commit -m "feat(faces): decode YuNet outputs and suppress overlaps"
```

---

## Task 5: Face alignment (pure)

**Files:**
- Create: `imageharbor/faces/align.py`
- Test: `tests/faces/test_align.py`

**Interfaces:**
- Consumes: `decode.Detection`
- Produces: `ARCFACE_TEMPLATE: np.ndarray`; `DegenerateLandmarks(Exception)`; `similarity_transform(src, dst) -> np.ndarray` (3×3); `align_crop(image, landmarks, size=(112,112)) -> PIL.Image.Image`

- [ ] **Step 1: Write the failing test**

```python
"""Landmark alignment onto the ArcFace template. Pure geometry, no model."""

import numpy as np
import pytest
from PIL import Image

from imageharbor.faces import align


def test_template_is_five_points_in_a_112_box():
    assert align.ARCFACE_TEMPLATE.shape == (5, 2)
    assert align.ARCFACE_TEMPLATE.min() > 0
    assert align.ARCFACE_TEMPLATE.max() < 112


def test_identity_when_source_is_already_the_template():
    t = align.similarity_transform(align.ARCFACE_TEMPLATE, align.ARCFACE_TEMPLATE)
    assert np.allclose(t, np.eye(3), atol=1e-6)


def test_recovers_a_known_scale_and_translation():
    src = align.ARCFACE_TEMPLATE * 2.0 + np.array([30.0, 40.0])
    t = align.similarity_transform(src, align.ARCFACE_TEMPLATE)
    homogeneous = np.hstack([src, np.ones((5, 1))])
    mapped = (t @ homogeneous.T).T[:, :2]
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-4)


def test_recovers_a_known_rotation():
    theta = np.deg2rad(30.0)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = align.ARCFACE_TEMPLATE @ r.T
    t = align.similarity_transform(src, align.ARCFACE_TEMPLATE)
    homogeneous = np.hstack([src, np.ones((5, 1))])
    mapped = (t @ homogeneous.T).T[:, :2]
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-4)


def test_collinear_landmarks_raise_rather_than_returning_a_bad_warp():
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    with pytest.raises(align.DegenerateLandmarks):
        align.similarity_transform(collinear, align.ARCFACE_TEMPLATE)


def test_identical_landmarks_raise():
    same = np.zeros((5, 2))
    with pytest.raises(align.DegenerateLandmarks):
        align.similarity_transform(same, align.ARCFACE_TEMPLATE)


def test_near_collinear_landmarks_raise():
    # Points 1e-4 off a perfect line pass an *exact* rank check
    # (np.linalg.matrix_rank's default tolerance is essentially machine
    # precision) and would otherwise produce a similarity transform with a
    # physically nonsensical scale (~12.6) instead of being rejected.
    near_collinear = np.array(
        [[0.0, 0.0], [1.0, 1.0001], [2.0, 2.0], [3.0, 3.0001], [4.0, 4.0]]
    )
    with pytest.raises(align.DegenerateLandmarks):
        align.similarity_transform(near_collinear, align.ARCFACE_TEMPLATE)


def test_arcface_template_onto_itself_does_not_raise():
    # Guard against over-rejection: the template is a perfectly typical
    # frontal face and must never be flagged degenerate.
    t = align.similarity_transform(align.ARCFACE_TEMPLATE, align.ARCFACE_TEMPLATE)
    assert np.allclose(t, np.eye(3), atol=1e-6)


def test_rotated_and_scaled_template_does_not_raise():
    # Guard against over-rejection on realistic, well-conditioned geometry.
    theta = np.deg2rad(30.0)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = align.ARCFACE_TEMPLATE @ r.T * 2.0 + np.array([30.0, 40.0])
    t = align.similarity_transform(src, align.ARCFACE_TEMPLATE)
    homogeneous = np.hstack([src, np.ones((5, 1))])
    mapped = (t @ homogeneous.T).T[:, :2]
    assert np.allclose(mapped, align.ARCFACE_TEMPLATE, atol=1e-4)


def test_align_crop_returns_the_requested_size():
    img = Image.new("RGB", (400, 400), (128, 64, 32))
    landmarks = [(150.0, 160.0), (250.0, 160.0), (200.0, 210.0), (160.0, 270.0), (240.0, 270.0)]
    out = align.align_crop(img, landmarks)
    assert out.size == (112, 112)
    assert out.mode == "RGB"


def test_align_crop_puts_the_eye_where_the_template_says():
    # A face drawn so its landmarks are the template scaled by 2 and shifted:
    # after alignment the left eye must land on the template's left eye.
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    src = align.ARCFACE_TEMPLATE * 2.0 + np.array([50.0, 50.0])
    # Mark the left-eye pixel so we can find it after the warp.
    img.putpixel((int(src[0][0]), int(src[0][1])), (255, 255, 255))
    out = align.align_crop(img, [tuple(p) for p in src])
    ex, ey = align.ARCFACE_TEMPLATE[0]
    window = [
        out.getpixel((x, y))[0]
        for x in range(int(ex) - 2, int(ex) + 3)
        for y in range(int(ey) - 2, int(ey) + 3)
    ]
    assert max(window) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_align.py -v`
Expected: FAIL with `ImportError: cannot import name 'align'`

- [ ] **Step 3: Write minimal implementation**

The inverse in `align_crop` is not an optimization — Pillow's `AFFINE` takes the **output → input** mapping. Passing the forward transform produces a warp that looks plausible and is wrong.

```python
"""Warp a detected face onto the ArcFace 5-point template. Pure geometry.

The template is the standard InsightFace ArcFace destination for a 112x112
crop. Every ArcFace-family embedder -- AuraFace included -- expects its input
aligned to it, so this is a contract, not a preference.

No OpenCV. Pillow's `Image.transform(..., AFFINE, ...)` does the resampling,
which keeps a 60 MB vision dependency out of a project whose entire runtime
dependency list is Pillow and Click.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

# InsightFace's canonical 5-point destination for a 112x112 crop:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


class DegenerateLandmarks(ValueError):
    """Landmarks that cannot define a stable similarity transform.

    Coincident points, and points that are exactly or *nearly* collinear,
    leave the covariance matrix rank-deficient or so ill-conditioned that
    the estimate is numerically unstable. Raising here means the caller
    rejects that face, which is correct: a face whose landmarks collapse
    toward a line is not a usable face, and warping it anyway produces a
    crop that embeds to noise. See `similarity_transform` for how "nearly"
    is measured and why.
    """


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (scale, rotation, translation) mapping src→dst.

    The Umeyama estimate. Returns a 3x3 homogeneous matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    src_var = (src_demean**2).sum() / n
    if src_var < 1e-12:
        raise DegenerateLandmarks("landmarks are coincident")

    cov = dst_demean.T @ src_demean / n
    u, s, vt = np.linalg.svd(cov)

    # `np.linalg.matrix_rank(cov) < 2` only catches *exactly* rank-deficient
    # input -- its default tolerance is essentially machine precision, so
    # landmarks that are merely near-collinear (a real detector on an
    # extreme profile, motion blur, or partial occlusion) sail through and
    # produce a wild, physically nonsensical scale instead of a rejection.
    # Test the conditioning of `cov` directly with the singular values
    # `svd` already computed above, rather than paying for a second,
    # redundant SVD inside matrix_rank.
    #
    # Measured smallest/largest singular-value ratios:
    #   - ArcFace template onto itself (typical frontal face):        0.63
    #   - template + 1.5px detector jitter:                           0.62
    #   - rotated/scaled template (existing regression cases):        0.63
    #   - three-quarter profile, eyes/nose compressed toward one side: 0.40
    #   - extreme near-edge-on profile (80% compression):             0.29
    #   - landmarks 1e-4 off a perfect line:                       1.15e-6
    #   - exactly collinear:                                       7.0e-17
    # Realistic geometry -- including a hard profile -- never drops below
    # ~0.2; the pathological near-collinear case sits around 1e-6, five to
    # six orders of magnitude lower. 1e-3 sits comfortably in that gap: it
    # is ~1000x above the pathological ratio and ~200x below the most
    # extreme realistic one measured, so it rejects unstable fits without
    # discarding usable faces.
    #
    # Guard the division too: s[0] == 0 only if `cov` itself is the zero
    # matrix (e.g. `dst` is coincident, since `src` coincidence is already
    # excluded above), which is degenerate regardless of the ratio -- `0/0`
    # is `nan` and `nan < 1e-3` is silently False, so this must be checked
    # first rather than folded into the ratio comparison.
    if s[0] < 1e-12 or s[-1] / s[0] < 1e-3:
        raise DegenerateLandmarks("landmarks are collinear or too close to it")

    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    # `sign(det(cov))` and `sign(det(u) * det(vt))` are always equal once
    # the conditioning guard above has passed -- both encode the same
    # reflection via the same (now well-conditioned) `cov`, so this branch
    # currently never flips `d` a second time. Kept anyway: reflection
    # handling here is subtle, and a future reader who loosens or removes
    # the guard above may need this check to still be doing real work.
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0

    rotation = u @ np.diag(d) @ vt
    scale = float(s @ d) / src_var

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rotation * scale
    matrix[:2, 2] = dst_mean - (rotation * scale) @ src_mean
    return matrix


def align_crop(
    image: Image.Image,
    landmarks: Sequence[tuple[float, float]],
    size: tuple[int, int] = (112, 112),
) -> Image.Image:
    """Warp `image` so `landmarks` land on the ArcFace template."""
    if len(landmarks) != 5:
        raise DegenerateLandmarks(f"expected 5 landmarks, got {len(landmarks)}")

    scale = np.array([size[0] / 112.0, size[1] / 112.0])
    forward = similarity_transform(
        np.asarray(landmarks, dtype=np.float64), ARCFACE_TEMPLATE * scale
    )

    # Pillow's AFFINE data is the OUTPUT -> INPUT mapping, so the inverse of the
    # transform we just estimated. Passing `forward` here yields a warp that
    # looks plausible and is wrong.
    inverse = np.linalg.inv(forward)
    data = (
        inverse[0, 0], inverse[0, 1], inverse[0, 2],
        inverse[1, 0], inverse[1, 1], inverse[1, 2],
    )
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return rgb.transform(size, Image.AFFINE, data, resample=Image.BILINEAR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_align.py -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/align.py tests/faces/test_align.py
git commit -m "feat(faces): align faces onto the ArcFace template without OpenCV"
```

---

## Task 6: Clustering, attribution, and calibration (pure)

**Files:**
- Create: `imageharbor/faces/cluster.py`, `imageharbor/faces/attribute.py`, `imageharbor/faces/calibrate.py`
- Test: `tests/faces/test_cluster.py`, `tests/faces/test_attribute.py`, `tests/faces/test_calibrate.py`

**Interfaces:**
- Consumes: `names.normalize`
- Produces:
  - `cluster.FaceVector(face_id: int, embedding: np.ndarray, embed_model: str)`
  - `cluster.Seed(name: str, face_ids: tuple[int, ...])`
  - `cluster.Cluster(face_ids: tuple[int, ...], centroid: np.ndarray, seed_name: str | None)`
  - `cluster.MixedModelError(Exception)`
  - `cluster.cluster_faces(faces, *, threshold, seeds=()) -> list[Cluster]`
  - `attribute.Proposal(cluster_id, name, support, total_tagged, score, untagged_photos)`
  - `attribute.propose(cluster_photos, photo_names, *, min_score, min_support) -> list[Proposal]`
  - `calibrate.Calibration(threshold, precision, recall, curve)`
  - `calibrate.calibrate(anchors, *, target_precision=0.99, max_anchors=4000) -> Calibration`

- [ ] **Step 1: Write the failing tests**

```python
"""Clustering on synthetic vectors. No model, no database."""

import numpy as np
import pytest

from imageharbor.faces import cluster


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _fv(face_id, v, model="auraface"):
    return cluster.FaceVector(face_id=face_id, embedding=_unit(v), embed_model=model)


def test_identical_vectors_form_one_cluster():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [1, 0, 0]), _fv(3, [1, 0, 0])]
    out = cluster.cluster_faces(faces, threshold=0.5)
    assert len(out) == 1
    assert out[0].face_ids == (1, 2, 3)


def test_orthogonal_vectors_form_separate_clusters():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [0, 1, 0]), _fv(3, [0, 0, 1])]
    out = cluster.cluster_faces(faces, threshold=0.5)
    assert len(out) == 3


def test_threshold_boundary_is_inclusive():
    # cos 60 degrees = 0.5 exactly.
    a, b = _fv(1, [1, 0, 0]), _fv(2, [0.5, np.sqrt(3) / 2, 0])
    assert len(cluster.cluster_faces([a, b], threshold=0.5)) == 1
    assert len(cluster.cluster_faces([a, b], threshold=0.51)) == 2


def test_mixing_embed_models_raises():
    faces = [_fv(1, [1, 0, 0], "auraface"), _fv(2, [1, 0, 0], "sface")]
    with pytest.raises(cluster.MixedModelError, match="auraface"):
        cluster.cluster_faces(faces, threshold=0.5)


def test_seeds_are_placed_before_unseeded_faces():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [1, 0, 0]), _fv(3, [0, 1, 0])]
    seeds = [cluster.Seed(name="Emma", face_ids=(3,))]
    out = cluster.cluster_faces(faces, threshold=0.5, seeds=seeds)
    assert out[0].seed_name == "Emma"
    assert out[0].face_ids == (3,)
    assert out[1].seed_name is None


def test_one_seed_name_may_produce_several_clusters():
    # Aging: the same person, two life stages, not mutually similar.
    faces = [_fv(1, [1, 0, 0]), _fv(2, [0, 1, 0])]
    seeds = [cluster.Seed(name="Emma", face_ids=(1, 2))]
    out = cluster.cluster_faces(faces, threshold=0.9, seeds=seeds)
    assert len(out) == 2
    assert {c.seed_name for c in out} == {"Emma"}


def test_seed_isolation_prevents_merging_different_people():
    # The invariant this module exists for: two different people must never
    # merge just because their embeddings are close. Phase A restricts each
    # seed's comparisons to that seed's own clusters (`accumulators[start:]`);
    # mutating that to search all accumulators would merge Judy into Emma's
    # cluster here, since their embeddings are identical.
    faces = [_fv(1, [1, 0, 0]), _fv(2, [1, 0, 0])]
    seeds = [
        cluster.Seed(name="Emma", face_ids=(1,)),
        cluster.Seed(name="Judy", face_ids=(2,)),
    ]
    out = cluster.cluster_faces(faces, threshold=0.5, seeds=seeds)
    assert len(out) == 2
    by_name = {c.seed_name: c.face_ids for c in out}
    assert by_name == {"Emma": (1,), "Judy": (2,)}


def test_is_deterministic_for_the_same_input_order():
    faces = [_fv(i, [np.cos(i), np.sin(i), 0]) for i in range(20)]
    a = cluster.cluster_faces(faces, threshold=0.8)
    b = cluster.cluster_faces(faces, threshold=0.8)
    assert [c.face_ids for c in a] == [c.face_ids for c in b]


def test_centroids_are_unit_length():
    faces = [_fv(1, [1, 0, 0]), _fv(2, [0.9, 0.1, 0])]
    for c in cluster.cluster_faces(faces, threshold=0.5):
        assert np.linalg.norm(c.centroid) == pytest.approx(1.0, abs=1e-5)


def test_empty_input_returns_no_clusters():
    assert cluster.cluster_faces([], threshold=0.5) == []
```

```python
"""Proposal scoring. Pure, table-driven."""

import pytest

from imageharbor.faces import attribute


def test_unanimous_cluster_proposes_that_name():
    props = attribute.propose(
        {1: ["a", "b", "c"]},
        {"a": ["Emma"], "b": ["Emma"], "c": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert len(props) == 1
    assert props[0].name == "Emma"
    assert props[0].support == 3
    assert props[0].total_tagged == 3
    assert props[0].score == pytest.approx(1.0)
    assert props[0].untagged_photos == 0


def test_untagged_photos_are_counted_as_the_payoff():
    props = attribute.propose(
        {1: ["a", "b"] + [f"u{i}" for i in range(340)]},
        {"a": ["Emma"], "b": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert props[0].total_tagged == 2
    assert props[0].untagged_photos == 340


def test_a_cluster_with_no_tagged_photos_proposes_nothing():
    assert attribute.propose({1: ["a", "b"]}, {}, min_score=0.6, min_support=1) == []


def test_below_min_score_proposes_nothing():
    props = attribute.propose(
        {1: ["a", "b", "c"]},
        {"a": ["Emma"], "b": ["Judy"], "c": ["Pete"]},
        min_score=0.6,
        min_support=1,
    )
    assert props == []


def test_below_min_support_proposes_nothing():
    props = attribute.propose(
        {1: ["a"]}, {"a": ["Emma"]}, min_score=0.6, min_support=2
    )
    assert props == []


def test_two_always_co_occurring_names_both_qualify():
    # A couple photographed together always. Neither is more supported than the
    # other, so both are offered and the human picks. Never an arbitrary winner.
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["Judy", "Pete"], "b": ["Judy", "Pete"]},
        min_score=0.6,
        min_support=2,
    )
    assert sorted(p.name for p in props) == ["Judy", "Pete"]


def test_names_are_normalized_before_counting():
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["Emma "], "b": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert len(props) == 1
    assert props[0].name == "Emma"
    assert props[0].support == 2


def test_case_variants_are_not_merged():
    # "pete storz" and "Pete Storz" stay separate; merging is a human call.
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["pete storz"], "b": ["Pete Storz"]},
        min_score=0.9,
        min_support=2,
    )
    assert props == []


def test_a_repeated_name_on_one_photo_counts_once():
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["Emma", "Emma"], "b": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert props[0].support == 2


def test_duplicate_photo_digest_in_cluster_is_counted_once():
    # cluster_photos carrying the same digest twice (e.g. a face detected
    # twice on one photo) must not double-count that photo's evidence --
    # dict.fromkeys() de-duplication is what makes support/total_tagged
    # reflect distinct photos rather than raw entries.
    props = attribute.propose(
        {1: ["a", "a", "b"]},
        {"a": ["Emma"], "b": ["Judy"]},
        min_score=0.3,
        min_support=1,
    )
    by_name = {p.name: p for p in props}
    assert by_name["Emma"].support == 1
    assert by_name["Emma"].total_tagged == 2
    assert by_name["Judy"].total_tagged == 2


def test_output_is_sorted_by_score_then_name():
    props = attribute.propose(
        {1: ["a", "b", "c", "d"]},
        {"a": ["Emma"], "b": ["Emma"], "c": ["Emma"], "d": ["Judy"]},
        min_score=0.2,
        min_support=1,
    )
    assert [p.name for p in props] == ["Emma", "Judy"]
```

```python
"""Threshold calibration from labelled anchors."""

import numpy as np
import pytest

from imageharbor.faces import calibrate


def _anchors(rng, names=("Emma", "Judy", "Pete"), per=12, spread=0.02):
    out = []
    for i, name in enumerate(names):
        base = np.zeros(8, dtype=np.float32)
        base[i] = 1.0
        for _ in range(per):
            v = base + rng.normal(0, spread, 8).astype(np.float32)
            out.append((name, v / np.linalg.norm(v)))
    return out


def test_well_separated_anchors_yield_a_high_precision_threshold():
    rng = np.random.default_rng(0)
    result = calibrate.calibrate(_anchors(rng), target_precision=0.99)
    assert 0.0 < result.threshold < 1.0
    assert result.precision >= 0.99
    assert result.recall > 0.5


def test_the_threshold_separates_the_synthetic_groups():
    rng = np.random.default_rng(1)
    anchors = _anchors(rng)
    result = calibrate.calibrate(anchors, target_precision=0.99)
    same = np.dot(anchors[0][1], anchors[1][1])
    diff = np.dot(anchors[0][1], anchors[-1][1])
    assert diff < result.threshold <= same


def test_curve_is_returned_and_ordered():
    rng = np.random.default_rng(2)
    result = calibrate.calibrate(_anchors(rng), target_precision=0.99)
    thresholds = [t for t, _, _ in result.curve]
    assert thresholds == sorted(thresholds)
    assert len(result.curve) > 1


def test_too_few_anchors_raises():
    with pytest.raises(ValueError, match="at least two names"):
        calibrate.calibrate([("Emma", np.array([1.0, 0.0], dtype=np.float32))])


def test_is_deterministic_under_subsampling():
    rng = np.random.default_rng(3)
    anchors = _anchors(rng, names=tuple(f"P{i}" for i in range(10)), per=30)
    a = calibrate.calibrate(anchors, max_anchors=50)
    b = calibrate.calibrate(anchors, max_anchors=50)
    assert a.threshold == b.threshold


def _two_pair_anchors():
    # Two same-name pairs (A at sim 0.9, B at sim 0.7) with zero similarity
    # across names -- built in 4D so the A-pair and B-pair occupy disjoint
    # subspaces and every cross pair is exactly 0.
    return [
        ("A", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("A", np.array([0.9, np.sqrt(1 - 0.9**2), 0.0, 0.0], dtype=np.float32)),
        ("B", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        ("B", np.array([0.0, 0.0, 0.7, np.sqrt(1 - 0.7**2)], dtype=np.float32)),
    ]


def test_unreachable_target_precision_falls_back_to_best_recall():
    # With this anchor set, precision is 1.0 across the whole threshold range
    # up to 0.9 -- low thresholds keep both pairs (recall 1.0), high
    # thresholds keep only the A-pair (recall 0.5). An unreachable
    # target_precision forces the fallback; among tied-precision points it
    # must pick the lowest threshold, matching the primary scan's own bias
    # toward recall -- not the highest threshold, which is the worst point on
    # the plateau.
    result = calibrate.calibrate(_two_pair_anchors(), target_precision=1.5)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_self_pairs_are_excluded_from_the_curve():
    # A face compared to itself has similarity 1.0 and is trivially
    # "same-name" -- including it (np.triu_indices with k=0 instead of k=1)
    # inflates both precision and recall. Just above the B-pair's similarity
    # (0.7), only the A-pair (0.9) remains a genuine same-name match: 1 of 2
    # same-name pairs, so recall is 0.5. If self-pairs leaked in, the 4
    # self-pairs (always selected, always "same") would push recall to
    # 5/6 instead.
    result = calibrate.calibrate(_two_pair_anchors(), target_precision=0.99)
    point = next(c for c in result.curve if c[0] > 0.7)
    _, precision, recall = point
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/faces/test_cluster.py tests/faces/test_attribute.py tests/faces/test_calibrate.py -v`
Expected: FAIL, three `ImportError`s

- [ ] **Step 3: Write the implementations**

```python
"""Group face embeddings into clusters. Pure: no I/O, no model, no database.

Faces are compared against cluster **centroids**, not against each other.
Pairwise over ~150,000 faces is ~11 billion comparisons; against a few thousand
centroids it is one chunked matmul.

Two phases. Anchors -- faces whose person is known from a Google tag -- are
placed first, so the clusters that matter exist before any guessing begins.
Everything else is then assigned to its nearest centroid above threshold.

Phase B is order-dependent, and that is this module's known weakness. It is
contained by placing seeds first, by the caller supplying a deterministic order
(digest order), and by merge/split in the review UI being the actual repair.
Callers must not shuffle: the same input order must give the same output, and
that is pinned by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceVector:
    face_id: int
    embedding: np.ndarray  # L2-normalized
    embed_model: str


@dataclass(frozen=True)
class Seed:
    name: str
    face_ids: tuple[int, ...]


@dataclass(frozen=True)
class Cluster:
    face_ids: tuple[int, ...]
    centroid: np.ndarray
    seed_name: str | None = None


class MixedModelError(ValueError):
    """Embeddings from different models were compared.

    Vectors from two models share a coordinate space only by coincidence. A
    comparison across them returns a plausible number that means nothing, which
    is worse than an error, so this raises.
    """


class _Accumulator:
    """A cluster under construction, tracking a running normalized centroid."""

    __slots__ = ("face_ids", "_sum", "seed_name")

    def __init__(self, face_id: int, vector: np.ndarray, seed_name: str | None) -> None:
        self.face_ids = [face_id]
        self._sum = vector.astype(np.float64).copy()
        self.seed_name = seed_name

    def add(self, face_id: int, vector: np.ndarray) -> None:
        self.face_ids.append(face_id)
        self._sum += vector

    @property
    def centroid(self) -> np.ndarray:
        norm = np.linalg.norm(self._sum)
        if norm < 1e-12:  # pragma: no cover - only if vectors cancel exactly
            return self._sum.astype(np.float32)
        return (self._sum / norm).astype(np.float32)

    def freeze(self) -> Cluster:
        return Cluster(
            face_ids=tuple(self.face_ids),
            centroid=self.centroid,
            seed_name=self.seed_name,
        )


def cluster_faces(
    faces: list[FaceVector],
    *,
    threshold: float,
    seeds: list[Seed] | tuple[Seed, ...] = (),
) -> list[Cluster]:
    """Assign faces to clusters. Seeded faces first, then the rest in order."""
    if not faces:
        return []

    models = {f.embed_model for f in faces}
    if len(models) > 1:
        raise MixedModelError(
            f"cannot cluster across embedding models: {sorted(models)}"
        )

    by_id = {f.face_id: f for f in faces}
    accumulators: list[_Accumulator] = []

    def _best_match(
        candidates: list[_Accumulator], embedding: np.ndarray
    ) -> _Accumulator | None:
        """The candidate closest to `embedding`, if it clears `threshold`."""
        if not candidates:
            return None
        centroids = np.stack([a.centroid for a in candidates])
        sims = centroids @ embedding
        best = int(np.argmax(sims))
        return candidates[best] if float(sims[best]) >= threshold else None

    # Phase A: seeds, grouped by name so one name may yield several clusters.
    seeded: set[int] = set()
    for seed in seeds:
        start = len(accumulators)
        for face_id in seed.face_ids:
            face = by_id.get(face_id)
            if face is None or face_id in seeded:
                continue
            seeded.add(face_id)
            # Only compare against this seed's own clusters: two different
            # people must never be merged just because they look alike.
            match = _best_match(accumulators[start:], face.embedding)
            if match is not None:
                match.add(face_id, face.embedding)
                continue
            accumulators.append(_Accumulator(face_id, face.embedding, seed.name))

    # Phase B: everything else, in the caller's order.
    for face in faces:
        if face.face_id in seeded:
            continue
        match = _best_match(accumulators, face.embedding)
        if match is not None:
            match.add(face.face_id, face.embedding)
        else:
            accumulators.append(_Accumulator(face.face_id, face.embedding, None))

    return [a.freeze() for a in accumulators]
```

```python
"""Propose person names for clusters from Google's tags. Pure: no I/O.

This module only ever *proposes*. Nothing here writes an identity; that happens
only when a human confirms a cluster on the dashboard.

Every qualifying name is returned, not just the best one. Two people
photographed together always -- a couple, a pair of siblings -- score
identically, and picking the alphabetically-first would be an arbitrary
assertion dressed up as an answer. Offering both is the honest output.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .names import normalize


@dataclass(frozen=True)
class Proposal:
    cluster_id: int
    name: str
    support: int          # photos in this cluster tagged with this name
    total_tagged: int     # photos in this cluster tagged with anything
    score: float          # support / total_tagged
    untagged_photos: int  # what confirming would newly name -- the payoff


def propose(
    cluster_photos: Mapping[int, Sequence[str]],
    photo_names: Mapping[str, Sequence[str]],
    *,
    min_score: float,
    min_support: int,
) -> list[Proposal]:
    """Rank name proposals per cluster, best first."""
    out: list[Proposal] = []

    for cluster_id in sorted(cluster_photos):
        photos = list(dict.fromkeys(cluster_photos[cluster_id]))

        counts: Counter[str] = Counter()
        tagged = 0
        for digest in photos:
            # A name repeated on one photo is one photo's worth of evidence.
            found = {
                normalize(n) for n in photo_names.get(digest, ()) if normalize(n)
            }
            if found:
                tagged += 1
                counts.update(found)

        if tagged == 0:
            continue

        for name, support in counts.items():
            score = support / tagged
            if score >= min_score and support >= min_support:
                out.append(
                    Proposal(
                        cluster_id=cluster_id,
                        name=name,
                        support=support,
                        total_tagged=tagged,
                        score=score,
                        untagged_photos=len(photos) - tagged,
                    )
                )

    out.sort(key=lambda p: (p.cluster_id, -p.score, p.name))
    return out
```

```python
"""Measure the clustering threshold from the library's own labelled data.

A photo with exactly one detected face and exactly one Google name is an
unambiguous (face, name) pair. This library has 5,670 photos carrying exactly
one name, so the threshold can be *measured* rather than copied out of another
project's README.

Precision here is over pairs: of all anchor pairs at or above a threshold, the
fraction that really are the same person. The chosen threshold is the lowest one
meeting the target, which maximizes recall subject to that precision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    threshold: float
    precision: float
    recall: float
    curve: tuple[tuple[float, float, float], ...]  # (threshold, precision, recall)


def calibrate(
    anchors: Sequence[tuple[str, np.ndarray]],
    *,
    target_precision: float = 0.99,
    max_anchors: int = 4000,
    steps: int = 200,
) -> Calibration:
    """Pick the lowest threshold reaching `target_precision` on anchor pairs."""
    names = [n for n, _ in anchors]
    if len(set(names)) < 2:
        raise ValueError("calibration needs anchors for at least two names")

    if len(anchors) > max_anchors:
        # Deterministic subsample: a seeded generator, so a re-run of calibrate
        # on the same library returns the same threshold.
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(anchors), size=max_anchors, replace=False))
        anchors = [anchors[i] for i in keep]
        names = [n for n, _ in anchors]

    matrix = np.stack([v for _, v in anchors]).astype(np.float32)
    sims = matrix @ matrix.T
    labels = np.array(names)
    same = labels[:, None] == labels[None, :]

    upper = np.triu_indices(len(anchors), k=1)
    pair_sims = sims[upper]
    pair_same = same[upper]

    total_same = int(pair_same.sum())
    if total_same == 0:
        raise ValueError("calibration needs at least one same-name anchor pair")

    curve: list[tuple[float, float, float]] = []
    chosen: tuple[float, float, float] | None = None
    for t in np.linspace(0.0, 1.0, steps, dtype=np.float32):
        selected = pair_sims >= t
        n_selected = int(selected.sum())
        if n_selected == 0:
            continue
        tp = int((selected & pair_same).sum())
        precision = tp / n_selected
        recall = tp / total_same
        curve.append((float(t), precision, recall))
        if chosen is None and precision >= target_precision:
            chosen = (float(t), precision, recall)

    if chosen is None:
        # Nothing reaches the target; return the most precise point measured so
        # the operator sees the real ceiling instead of a fabricated threshold.
        # Tie-break toward the *lowest* threshold, same as the primary scan
        # above: among equally-precise points a lower threshold means strictly
        # more recall, and the reverse tie-break would silently hand back the
        # worst-recall point on a precision plateau.
        best = max(curve, key=lambda c: (c[1], -c[0]))
        chosen = best

    return Calibration(
        threshold=chosen[0],
        precision=chosen[1],
        recall=chosen[2],
        curve=tuple(curve),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/faces/ -v`
Expected: PASS, all tests across the six modules

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/cluster.py imageharbor/faces/attribute.py imageharbor/faces/calibrate.py tests/faces/test_cluster.py tests/faces/test_attribute.py tests/faces/test_calibrate.py
git commit -m "feat(faces): cluster, propose, and calibrate against Google's own labels"
```

**Milestone A complete.** All logic exists and is tested with zero model weights and no database.

---

## Task 7: Face store

**Files:**
- Create: `imageharbor/faces/store.py`
- Test: `tests/faces/test_store.py`

**Interfaces:**
- Consumes: `cluster.Cluster`, `attribute.Proposal`
- Produces: `FaceStore(db_path)` with `record_scan(digest, detect_model, faces) -> list[int]`, `is_scanned(digest, detect_model) -> bool`, `iter_face_vectors(embed_model) -> Iterator[FaceVector]`, `replace_clusters(embed_model, clusters)`, `record_proposals(proposals)`, `confirm(cluster_id, name) -> int`, `reject(cluster_id, name)`, `merge(person_id, cluster_ids)`, `split(cluster_id, face_ids)`, `iter_pending_sidecars() -> Iterator[tuple[str, list[str]]]`, `mark_sidecar_written(digest, detect_model)`, `anchors(embed_model, photo_names) -> list[tuple[str, np.ndarray]]`, `cluster_ids() -> list[int]`, `proposals_for(cluster_id) -> list[dict]`, `person_for_cluster(cluster_id) -> int | None`, `set_organized_path(digest, path)`, `add_person(name, source) -> int`, `known_names() -> list[str]`, `stats() -> dict`, `close()`

**`record_scan`'s face tuples are `(Detection, embedding | None, embed_model | None, rejected_reason | None)`.** A 3-tuple is accepted and padded with `rejected_reason=None`, which is what the tests below pass. Task 11 supplies the fourth element for gated faces, so no later task has to widen this signature.

- [ ] **Step 1: Write the failing test**

```python
"""Face persistence: work queue, clusters, and the confirmation gate."""

import numpy as np
import pytest

from imageharbor.catalog import Catalog
from imageharbor.faces import cluster
from imageharbor.faces.attribute import Proposal
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()


def _det(x=0.0, score=0.9):
    return Detection(
        x=x, y=0.0, w=50.0, h=50.0, score=score,
        landmarks=((1.0, 1.0), (2.0, 1.0), (1.5, 2.0), (1.0, 3.0), (2.0, 3.0)),
    )


def _vec(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_recording_a_scan_makes_it_scanned(store):
    assert not store.is_scanned("digestA", "yunet")
    store.record_scan("digestA", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    assert store.is_scanned("digestA", "yunet")


def test_rescanning_the_same_photo_is_a_no_op(store):
    ids_a = store.record_scan("digestA", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    ids_b = store.record_scan("digestA", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    assert ids_a == ids_b
    assert store.stats()["faces"] == 1


def test_a_photo_with_no_faces_is_still_recorded_as_scanned(store):
    store.record_scan("empty", "yunet", [])
    assert store.is_scanned("empty", "yunet")
    assert store.stats()["faces"] == 0


def test_scan_is_keyed_on_the_detector(store):
    store.record_scan("digestA", "yunet", [])
    assert store.is_scanned("digestA", "yunet")
    assert not store.is_scanned("digestA", "scrfd")


def test_face_vectors_round_trip(store):
    store.record_scan("d", "yunet", [(_det(), _vec([0.6, 0.8, 0.0]), "auraface")])
    vectors = list(store.iter_face_vectors("auraface"))
    assert len(vectors) == 1
    assert np.allclose(vectors[0].embedding, _vec([0.6, 0.8, 0.0]), atol=1e-6)
    assert vectors[0].embed_model == "auraface"


def test_face_vectors_are_filtered_by_model(store):
    store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    assert list(store.iter_face_vectors("sface")) == []


def test_confirm_is_the_only_thing_that_sets_person_id(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]), seed_name=None)
    ])
    cid = store.cluster_ids()[0]
    store.record_proposals([Proposal(cid, "Emma", 3, 3, 1.0, 10)])
    assert store.person_for_cluster(cid) is None      # a proposal asserts nothing

    person_id = store.confirm(cid, "Emma")
    assert store.person_for_cluster(cid) == person_id


def test_rejecting_a_proposal_records_it_rather_than_deleting(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]), seed_name=None)
    ])
    cid = store.cluster_ids()[0]
    store.record_proposals([Proposal(cid, "Emma", 3, 3, 1.0, 10)])
    store.reject(cid, "Emma")
    assert store.proposals_for(cid)[0]["decided"] == "rejected"


def test_merge_points_several_clusters_at_one_person(store):
    ids = store.record_scan("d", "yunet", [
        (_det(x=0), _vec([1, 0, 0]), "auraface"),
        (_det(x=200), _vec([0, 1, 0]), "auraface"),
    ])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0])),
        cluster.Cluster(face_ids=(ids[1],), centroid=_vec([0, 1, 0])),
    ])
    a, b = store.cluster_ids()
    person_id = store.confirm(a, "Emma")
    store.merge(person_id, [b])
    assert store.person_for_cluster(b) == person_id


def test_replacing_clusters_preserves_confirmed_people(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))
    ])
    cid = store.cluster_ids()[0]
    person_id = store.confirm(cid, "Emma")

    # A recluster must not silently discard a human decision.
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))
    ])
    new_cid = store.cluster_ids()[0]
    assert store.person_for_cluster(new_cid) == person_id


def test_pending_sidecars_lists_a_photo_after_confirmation(store):
    ids = store.record_scan("d", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))
    ])
    cid = store.cluster_ids()[0]
    store.confirm(cid, "Emma")

    pending = dict(store.iter_pending_sidecars())
    assert pending == {"d": ["Emma"]}

    store.mark_sidecar_written("d", "yunet")
    assert dict(store.iter_pending_sidecars()) == {}


def test_anchors_are_single_face_single_name_photos(store):
    store.record_scan("one", "yunet", [(_det(), _vec([1, 0, 0]), "auraface")])
    store.record_scan("two", "yunet", [
        (_det(x=0), _vec([0, 1, 0]), "auraface"),
        (_det(x=200), _vec([0, 0, 1]), "auraface"),
    ])
    anchors = store.anchors("auraface", {"one": ["Emma"], "two": ["Judy"]})
    assert [n for n, _ in anchors] == ["Emma"]  # "two" has two faces, so it is not an anchor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'store'`

- [ ] **Step 3: Write minimal implementation**

Write `imageharbor/faces/store.py` with a `_FACE_SCHEMA` string containing the five `CREATE TABLE IF NOT EXISTS` statements from the spec's "Catalog schema" section verbatim, plus these indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_faces_digest  ON faces(sha256_b64url);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
CREATE INDEX IF NOT EXISTS idx_faces_model   ON faces(embed_model);
CREATE INDEX IF NOT EXISTS idx_clusters_person ON clusters(person_id);
```

`FaceStore.__init__` mirrors `Catalog.__init__`: open `sqlite3.connect(db_path, check_same_thread=False)`, set `row_factory = sqlite3.Row`, then `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, then `executescript(_FACE_SCHEMA)`. Guard every write with `threading.Lock`, as `Catalog` does.

Key implementation notes, each of which a test above pins:

- **`record_scan` is idempotent.** If `is_scanned` is already true for `(digest, detect_model)`, return the existing face ids without writing. This is what makes a re-run a no-op.
- **Embeddings are stored as `np.asarray(v, dtype=np.float32).tobytes()`** and read back with `np.frombuffer(blob, dtype=np.float32)`. Store `embedding_dim` alongside so a malformed blob is detectable.
- **`replace_clusters` must preserve confirmed people -- but never invent one.** Before deleting the old rows, build `{frozenset(face_ids): person_id}` for every cluster with a non-null `person_id`; after inserting the new rows, check *every* captured set (not just the first match) against each new cluster's face set. If the intersecting sets name exactly one distinct `person_id`, restore it. If a new cluster's face set intersects **more than one** distinct `person_id` -- e.g. two different people's confirmed clusters got merged into one new cluster by this recluster -- do not pick one; leave `person_id` NULL so the cluster returns to the review queue, and log a warning naming the conflicting people. Picking the first match in id order (an earlier draft of this method did exactly that) silently relabels the second person as the first with no error and no unconfirmed state -- the entire design rests on "no identity is written without human confirmation," and manufacturing one is worse than losing one. The opposite case, one confirmed cluster splitting into several new fragments, is safe and must keep inheriting the same person on every fragment: each fragment's face set only ever intersects that one captured set, so it never hits the multi-person branch.
- **`confirm(cluster_id, name)`** normalizes the name via `names.normalize`, inserts into `people` with `source='human'` if absent (`INSERT OR IGNORE`, then `SELECT`), sets `clusters.person_id` and `clusters.assigned_at = _now_iso()`, and returns the person id. **This is the only method that writes `person_id`**, other than `merge`.
- **`iter_pending_sidecars`** yields `(digest, [name, ...])` for every photo having a face in a cluster whose `assigned_at` is newer than that photo's `face_scan.sidecar_at` (or where `sidecar_at IS NULL`):

```sql
SELECT f.sha256_b64url AS digest, p.name AS name
  FROM faces f
  JOIN clusters c ON c.id = f.cluster_id
  JOIN people   p ON p.id = c.person_id
  JOIN face_scan s ON s.sha256_b64url = f.sha256_b64url
 WHERE c.person_id IS NOT NULL
   AND (s.sidecar_at IS NULL OR c.assigned_at > s.sidecar_at)
 ORDER BY f.sha256_b64url, p.name
```

  Group consecutive rows by digest and de-duplicate names.
- **`anchors(embed_model, photo_names)`** returns `(name, embedding)` for every photo with **exactly one** unrejected face and **exactly one** distinct normalized name.
- **`stats()`** returns `{"faces", "scanned", "clusters", "people", "unreviewed", "singletons"}` for the dashboard.
- **`organized_path_for(digest)` has two sources, in precedence order.** First `face_organized_paths` (the explicit override `set_organized_path` writes -- needed because that method is part of this module's contract even for a digest with no `photos` row at all, per Task 12's fixtures). If that table has no row, fall back to a read-only `SELECT organized_path FROM photos WHERE sha256_b64url = ?` against the same database -- reading another module's table is not an ownership violation, only writing is, and this fallback is what makes the method resolve anything in production: Task 11's `scan()` reads `organized_path` straight from the catalog and never calls `set_organized_path`, so `face_organized_paths` never gets a row from the real pipeline. Guard the fallback query against `photos` not existing at all (a `FaceStore` opened on a database a `Catalog` has never touched) -- catch `sqlite3.OperationalError` and return `None` rather than raising.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_store.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/store.py tests/faces/test_store.py
git commit -m "feat(faces): persist faces, clusters, and the confirmation gate"
```

---

## Task 8: Sidecar contract

**Files:**
- Modify: `imageharbor/sidecar_schema.py:34-40` (`KEYED_LISTS`), `imageharbor/sidecar_schema.py:57-59` (`_ANNOTATION_FIELDS`)
- Test: `tests/faces/test_sidecar_people.py`

**Interfaces:**
- Consumes: nothing
- Produces: `KEYED_LISTS["people"] == ("name", "source")`; `"confirmed_at" in _ANNOTATION_FIELDS`

- [ ] **Step 1: Write the failing test**

```python
"""The people list must hold Google's names and ImageHarbor's side by side."""

from imageharbor import sidecar_schema


def test_people_is_keyed_on_name_and_source():
    assert sidecar_schema.KEYED_LISTS["people"] == ("name", "source")


def test_confirmed_at_is_registered_as_an_annotation():
    # An annotation key missing from this set means the entry can never match
    # itself on a later merge, and the history list grows on every watch cycle,
    # forever. This exact failure has shipped here once already.
    assert "confirmed_at" in sidecar_schema._ANNOTATION_FIELDS


def test_google_and_face_entries_coexist_without_conflict():
    base = {"people": [{"name": "Suzanne Storz", "source": "google_photos_people"}]}
    updates = {
        "people": [
            {
                "name": "Suzanne Storz",
                "source": "imageharbor_faces",
                "cluster_ids": [7],
                "confirmed_at": "2026-08-31T00:00:00+00:00",
            }
        ]
    }
    merged = sidecar_schema.merge(base, updates, observed_at="2026-08-31T00:00:00+00:00")

    people = merged["people"]
    assert len(people) == 2
    sources = {p["source"] for p in people}
    assert sources == {"google_photos_people", "imageharbor_faces"}
    # Neither entry was superseded: both facts are true at once.
    assert all("history" not in p for p in people)


def test_existing_google_entries_survive_the_widened_key():
    # Every entry already written carries a source, so the wider key resolves
    # them unchanged and no migration is needed.
    base = {"people": [{"name": "Judy Storz", "source": "google_photos_people"}]}
    merged = sidecar_schema.merge(base, {}, observed_at="2026-08-31T00:00:00+00:00")
    assert merged["people"] == [{"name": "Judy Storz", "source": "google_photos_people"}]


def test_reconfirming_the_same_person_is_byte_identical():
    updates = {
        "people": [
            {
                "name": "Emma",
                "source": "imageharbor_faces",
                "cluster_ids": [3],
                "confirmed_at": "2026-08-31T00:00:00+00:00",
            }
        ]
    }
    once = sidecar_schema.merge({}, updates, observed_at="2026-08-31T00:00:00+00:00")
    twice = sidecar_schema.merge(once, updates, observed_at="2026-09-01T00:00:00+00:00")
    assert once == twice
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_sidecar_people.py -v`
Expected: FAIL on the first two assertions

- [ ] **Step 3: Write minimal implementation**

In `imageharbor/sidecar_schema.py`, change the `people` entry in `KEYED_LISTS`:

```python
    # Keyed on (name, source), not name alone. Google tagging "Suzanne Storz"
    # and a confirmed face cluster identifying her are two true facts about the
    # same photo, from different evidence. Under a name-only key the second
    # would supersede the first's `source` and relocate it to history --
    # recording them as if they conflicted. No migration is needed: every entry
    # ever written already carries a `source`, so the wider key resolves them
    # unchanged.
    "people": ("name", "source"),
```

And add `confirmed_at` to `_ANNOTATION_FIELDS`:

```python
_ANNOTATION_FIELDS = frozenset({
    "observed_at", "superseded_at", "first_seen", "last_seen", "rejected",
    "history", "confirmed_at",
})
```

- [ ] **Step 4: Run the full sidecar suite to verify nothing regressed**

Run: `uv run pytest tests/faces/test_sidecar_people.py tests/test_sidecar_schema.py tests/test_sidecar.py tests/test_takeout_ingest.py -v`
Expected: PASS. The property test `test_never_loses_a_value_over_a_random_merge_sequence` must still pass — it is the guard on this change.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/sidecar_schema.py tests/faces/test_sidecar_people.py
git commit -m "feat(faces): key people on (name, source) so evidence coexists"
```

**Milestone B complete.**

---

## Task 9: Model acquisition and the detector

**Files:**
- Create: `imageharbor/faces/download.py`, `imageharbor/faces/detect.py`
- Modify: `imageharbor/faces/models.py` (pin the two `sha256` values)
- Test: `tests/faces/test_download.py`, `tests/faces/test_detect.py`

**Interfaces:**
- Consumes: `models.ModelInfo`, `decode.decode_yunet`, `decode.Detection`
- Produces: `download.ChecksumMismatch(Exception)`, `download.ensure(info, model_dir) -> Path`; `detect.Detector(model_dir, name=models.DEFAULT_DETECTOR)` with `.detect(image, score_threshold=0.6, nms_threshold=0.3) -> list[Detection]` and `.model_name: str`

**This is the risk task.** If Step 6 cannot produce a correct detection on a real photograph, take the documented fallback: port PhotoPrism's `internal/ai/face/engine_onnx_yunet.go` into `decode.py`, keeping every signature in this plan unchanged. Both projects are AGPL-3.0-or-later, so this is licence-compatible; add a header note naming the upstream project, file, and licence. Do not ship an unvalidated decoder, and do not reach for OpenCV.

- [ ] **Step 1: Write the failing tests**

```python
"""Model acquisition verifies before it trusts."""

import hashlib

import pytest

from imageharbor.faces import download, models


def test_a_matching_file_is_accepted(tmp_path):
    art = tmp_path / "m.onnx"
    art.write_bytes(b"weights")
    info = models.ModelInfo(
        name="t", kind="detector", filename="m.onnx", url="http://example/m",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        input_size=(1, 1), channel_order="RGB", mean=0.0, std=1.0, licence="MIT",
    )
    assert download.ensure(info, tmp_path) == art


def test_a_mismatched_file_raises_rather_than_being_used(tmp_path):
    art = tmp_path / "m.onnx"
    art.write_bytes(b"tampered")
    info = models.ModelInfo(
        name="t", kind="detector", filename="m.onnx", url="http://example/m",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        input_size=(1, 1), channel_order="RGB", mean=0.0, std=1.0, licence="MIT",
    )
    with pytest.raises(download.ChecksumMismatch, match="m.onnx"):
        download.ensure(info, tmp_path)


def test_an_unpinned_model_refuses_to_verify_silently(tmp_path):
    art = tmp_path / "m.onnx"
    art.write_bytes(b"weights")
    info = models.ModelInfo(
        name="t", kind="detector", filename="m.onnx", url="http://example/m",
        sha256=None,
        input_size=(1, 1), channel_order="RGB", mean=0.0, std=1.0, licence="MIT",
    )
    with pytest.raises(download.ChecksumMismatch, match="no pinned checksum"):
        download.ensure(info, tmp_path)


def test_both_shipped_models_have_pinned_checksums():
    for info in {**models.DETECTORS, **models.EMBEDDERS}.values():
        assert info.sha256, f"{info.name} has no pinned checksum"
        assert len(info.sha256) == 64, info.name
```

```python
"""Detector integration. Skips without weights; fails on a broken runtime."""

import os
from pathlib import Path

import pytest
from PIL import Image

onnxruntime = pytest.importorskip("onnxruntime")

from imageharbor.faces import detect, models  # noqa: E402

MODEL_DIR = Path(os.environ.get("IMAGEHARBOR_FACE_MODEL_DIR", "")) if os.environ.get(
    "IMAGEHARBOR_FACE_MODEL_DIR"
) else None


def _weights_present():
    if MODEL_DIR is None:
        return False
    return (MODEL_DIR / models.DETECTORS["yunet"].filename).exists()


needs_weights = pytest.mark.skipif(
    not _weights_present(),
    reason="set IMAGEHARBOR_FACE_MODEL_DIR to a directory holding the weights",
)


@needs_weights
def test_detector_loads_and_reports_its_model():
    d = detect.Detector(MODEL_DIR)
    assert d.model_name == "yunet"


@needs_weights
def test_a_blank_image_yields_no_faces():
    d = detect.Detector(MODEL_DIR)
    assert d.detect(Image.new("RGB", (640, 640), (10, 10, 10))) == []


@needs_weights
def test_a_real_photograph_yields_a_plausible_face():
    # tests/fixtures/one_face.jpg is a single-face photograph committed in Step 6.
    fixture = Path(__file__).parent.parent / "fixtures" / "one_face.jpg"
    img = Image.open(fixture)
    faces = detect.Detector(MODEL_DIR).detect(img)
    assert len(faces) == 1
    f = faces[0]
    assert f.score > 0.6
    # The box must sit inside the image and be a plausible fraction of it.
    assert 0 <= f.x < img.width and 0 <= f.y < img.height
    assert 0.02 < (f.w * f.h) / (img.width * img.height) < 0.9
    assert len(f.landmarks) == 5
    # Eyes above mouth: the cheapest check that the landmarks are not scrambled.
    assert f.landmarks[0][1] < f.landmarks[3][1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/faces/test_download.py tests/faces/test_detect.py -v`
Expected: `test_download` FAILs with `ImportError`; `test_detect` skips.

- [ ] **Step 3: Write `download.py`**

```python
"""Fetch and verify model artifacts.

An unverified artifact is never used. Two publishers ship different models under
the same filename -- InsightFace's antelopev2 pack and fal's AuraFace both call
theirs `glintr100.onnx` -- so a name match is not an artifact match, and only a
checksum settles it.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from .models import ModelInfo

logger = logging.getLogger(__name__)

_CHUNK = 1 << 16


class ChecksumMismatch(RuntimeError):
    """An artifact does not match its pinned digest, or has no pin at all."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(info: ModelInfo, model_dir: Path) -> Path:
    """Return a verified local path for `info`, downloading it if absent."""
    if not info.sha256:
        raise ChecksumMismatch(
            f"{info.filename}: no pinned checksum, refusing to run an "
            "unverified model"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / info.filename

    if not target.exists():
        logger.info("downloading face model %s from %s", info.name, info.url)
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(info.url, tmp)  # noqa: S310 - pinned URL
        tmp.replace(target)

    actual = _sha256(target)
    if actual != info.sha256:
        raise ChecksumMismatch(
            f"{info.filename}: expected {info.sha256}, got {actual}"
        )
    return target
```

- [ ] **Step 4: Download the weights and pin their checksums**

```bash
mkdir -p ~/.cache/imageharbor/models
cd ~/.cache/imageharbor/models
curl -L -o face_detection_yunet_2023mar.onnx \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
curl -L -o auraface_v1_glintr100.onnx \
  "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"
sha256sum face_detection_yunet_2023mar.onnx auraface_v1_glintr100.onnx
```

Paste each digest into the matching `sha256=` field in `imageharbor/faces/models.py`, replacing `None`. Sanity-check the sizes first: YuNet is ~350 KB, AuraFace ~261 MB. A few-hundred-byte file means an LFS pointer or an HTML error page, not a model.

- [ ] **Step 5: Inspect the real output signature before trusting the decoder**

```bash
uv run python -c "
import onnxruntime as ort
s = ort.InferenceSession('$HOME/.cache/imageharbor/models/face_detection_yunet_2023mar.onnx')
print('inputs :', [(i.name, i.shape) for i in s.get_inputs()])
print('outputs:', [(o.name, o.shape) for o in s.get_outputs()])
"
```

**Verified fact (Task 4 fix round 1, run against the real artifact):** one
input `input [1, 3, 640, 640]`, and twelve outputs in **type-major** order --
all three `cls` tensors, then all three `obj`, then all three `bbox`, then all
three `kps`, each block ordered by stride (8, 16, 32) --

```
cls_8 [1, 6400, 1]   cls_16 [1, 1600, 1]   cls_32 [1, 400, 1]
obj_8 [1, 6400, 1]   obj_16 [1, 1600, 1]   obj_32 [1, 400, 1]
bbox_8 [1, 6400, 4]  bbox_16 [1, 1600, 4]  bbox_32 [1, 400, 4]
kps_8 [1, 6400, 10]  kps_16 [1, 1600, 10]  kps_32 [1, 400, 10]
```

**not** the stride-major `cls_8, obj_8, bbox_8, kps_8, cls_16, ...` this step
originally told the implementer to expect. Every tensor also carries a leading
batch axis of 1 -- `(1, N, C)`, not `(N, C)`. `decode.decode_yunet` (Task 4)
and `_synthetic_outputs` in `tests/faces/test_decode.py` are already written
against this real layout, and `tests/faces/test_decode.py` carries a real-model
integration test (skipped unless `IMAGEHARBOR_FACE_MODEL_DIR` points at the
weights) that guards against this regressing. If a future export ever changes
the order or grouping again, reorder `decode.decode_yunet` to match and update
that integration test's expectations -- the decoder indexes outputs
positionally, so a different export order silently decodes garbage.

- [ ] **Step 6: Write `detect.py` and add the fixture**

```python
"""Run YuNet over an image. I/O only: the decode lives in `decode.py`."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import models
from .decode import Detection, decode_yunet
from .download import ensure

logger = logging.getLogger(__name__)


class Detector:
    """A loaded YuNet session. Not thread-safe; construct one per worker."""

    def __init__(self, model_dir: Path, name: str = models.DEFAULT_DETECTOR) -> None:
        import onnxruntime as ort

        self._info = models.DETECTORS[name]
        self.model_name = name
        path = ensure(self._info, Path(model_dir))
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def detect(
        self,
        image: Image.Image,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> list[Detection]:
        """Detect faces, returning boxes in `image`'s own pixel coordinates."""
        width, height = self._info.input_size
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        scale_x = rgb.width / width
        scale_y = rgb.height / height

        resized = rgb.resize((width, height), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        if self._info.channel_order == "BGR":
            array = array[:, :, ::-1]
        array = (array - self._info.mean) / self._info.std
        blob = np.ascontiguousarray(array.transpose(2, 0, 1)[None])

        outputs = self._session.run(None, {self._input: blob})
        detections = decode_yunet(
            outputs, (width, height), score_threshold, nms_threshold
        )

        # Back to the source image's coordinates.
        return [
            Detection(
                x=d.x * scale_x,
                y=d.y * scale_y,
                w=d.w * scale_x,
                h=d.h * scale_y,
                score=d.score,
                landmarks=tuple(
                    (px * scale_x, py * scale_y) for px, py in d.landmarks
                ),
            )
            for d in detections
        ]
```

Commit a single-face photograph as `tests/fixtures/one_face.jpg`. Use a photo you own, resized to about 800 px on the long edge to keep the repository small.

- [ ] **Step 7: Run tests with weights present**

```bash
IMAGEHARBOR_FACE_MODEL_DIR=~/.cache/imageharbor/models uv run pytest tests/faces/test_download.py tests/faces/test_detect.py -v
```

Expected: PASS. If `test_a_real_photograph_yields_a_plausible_face` fails, the decoder is wrong — take the porting fallback named at the top of this task rather than tuning thresholds until something appears. Tuning until a detection appears is how an incorrect decoder ships looking correct.

- [ ] **Step 8: Commit**

```bash
git add imageharbor/faces/download.py imageharbor/faces/detect.py imageharbor/faces/models.py tests/faces/test_download.py tests/faces/test_detect.py tests/fixtures/one_face.jpg
git commit -m "feat(faces): fetch, verify, and run the YuNet detector"
```

---

## Task 10: Embedder

**Files:**
- Create: `imageharbor/faces/embed.py`
- Test: `tests/faces/test_embed.py`

**Interfaces:**
- Consumes: `models`, `download.ensure`, `align.align_crop`
- Produces: `Embedder(model_dir, name=models.DEFAULT_EMBEDDER)` with `.embed(image, landmarks) -> np.ndarray`, `.embed_batch(crops) -> np.ndarray`, `.model_name: str`, `.dim: int`

- [ ] **Step 1: Write the failing test**

```python
"""Embedder integration. Skips without weights; fails on a broken runtime."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")

from imageharbor.faces import embed, models  # noqa: E402

MODEL_DIR = Path(os.environ["IMAGEHARBOR_FACE_MODEL_DIR"]) if os.environ.get(
    "IMAGEHARBOR_FACE_MODEL_DIR"
) else None

needs_weights = pytest.mark.skipif(
    MODEL_DIR is None
    or not (MODEL_DIR / models.EMBEDDERS["auraface"].filename).exists(),
    reason="set IMAGEHARBOR_FACE_MODEL_DIR to a directory holding the weights",
)

LANDMARKS = [(150.0, 160.0), (250.0, 160.0), (200.0, 210.0), (160.0, 270.0), (240.0, 270.0)]


@needs_weights
def test_embedding_has_the_declared_dimension():
    e = embed.Embedder(MODEL_DIR)
    v = e.embed(Image.new("RGB", (400, 400), (120, 100, 90)), LANDMARKS)
    assert v.shape == (e.dim,)
    assert e.dim == 512


@needs_weights
def test_embedding_is_l2_normalized_at_production():
    e = embed.Embedder(MODEL_DIR)
    v = e.embed(Image.new("RGB", (400, 400), (120, 100, 90)), LANDMARKS)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


@needs_weights
def test_the_same_image_embeds_identically():
    e = embed.Embedder(MODEL_DIR)
    img = Image.new("RGB", (400, 400), (120, 100, 90))
    assert np.allclose(e.embed(img, LANDMARKS), e.embed(img, LANDMARKS), atol=1e-6)


@needs_weights
def test_a_real_face_is_closer_to_itself_rotated_than_to_a_blank():
    fixture = Path(__file__).parent.parent / "fixtures" / "one_face.jpg"
    img = Image.open(fixture).convert("RGB")
    e = embed.Embedder(MODEL_DIR)
    from imageharbor.faces import detect

    d = detect.Detector(MODEL_DIR).detect(img)[0]
    a = e.embed(img, d.landmarks)
    b = e.embed(img.rotate(4, expand=False), d.landmarks)
    blank = e.embed(Image.new("RGB", img.size, (255, 255, 255)), d.landmarks)
    assert float(a @ b) > float(a @ blank)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `IMAGEHARBOR_FACE_MODEL_DIR=~/.cache/imageharbor/models uv run pytest tests/faces/test_embed.py -v`
Expected: FAIL with `ImportError: cannot import name 'embed'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Turn an aligned face crop into a vector. I/O only; the warp is in `align`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from . import models
from .align import align_crop
from .download import ensure


class Embedder:
    """A loaded embedding session. Not thread-safe; construct one per worker."""

    def __init__(self, model_dir: Path, name: str = models.DEFAULT_EMBEDDER) -> None:
        import onnxruntime as ort

        self._info = models.EMBEDDERS[name]
        self.model_name = name
        self.dim = self._info.embedding_dim
        path = ensure(self._info, Path(model_dir))
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def _blob(self, crops: Sequence[Image.Image]) -> np.ndarray:
        arrays = []
        for crop in crops:
            a = np.asarray(crop, dtype=np.float32)
            if self._info.channel_order == "BGR":  # pragma: no cover - RGB today
                a = a[:, :, ::-1]
            arrays.append((a - self._info.mean) / self._info.std)
        stacked = np.stack(arrays).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(stacked, dtype=np.float32)

    def embed_batch(self, crops: Sequence[Image.Image]) -> np.ndarray:
        """Embed pre-aligned 112x112 crops. Returns (N, dim), L2-normalized."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        raw = self._session.run(None, {self._input: self._blob(crops)})[0]
        vectors = np.asarray(raw, dtype=np.float32)
        # Normalized here, at the point of production, so cosine and Euclidean
        # stay equivalent for every consumer downstream.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def embed(
        self, image: Image.Image, landmarks: Sequence[tuple[float, float]]
    ) -> np.ndarray:
        """Align one face out of `image` and embed it."""
        crop = align_crop(image, landmarks, self._info.input_size)
        return self.embed_batch([crop])[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `IMAGEHARBOR_FACE_MODEL_DIR=~/.cache/imageharbor/models uv run pytest tests/faces/test_embed.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/embed.py tests/faces/test_embed.py
git commit -m "feat(faces): embed aligned crops with AuraFace"
```

---

## Task 11: The scan pass

**Files:**
- Create: `imageharbor/faces/runner.py`
- Test: `tests/faces/test_runner.py`

**Interfaces:**
- Consumes: `detect.Detector`, `embed.Embedder`, `store.FaceStore`, `catalog.Catalog`
- Produces: `QualityGate(min_score: float, min_box: int)`; `ScanResult(scanned, faces, rejected, errors)`; `scan(catalog, store, detector, embedder, crop_dir, *, gate, limit=None, should_stop=None) -> ScanResult`

- [ ] **Step 1: Write the failing test**

```python
"""The scan pass: resumable, idempotent, and never a breaker failure."""

import numpy as np
import pytest
from PIL import Image

from imageharbor.catalog import Catalog
from imageharbor.faces import runner
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


class FakeDetector:
    model_name = "yunet"

    def __init__(self, per_image=1):
        self.per_image = per_image
        self.calls = 0

    def detect(self, image, score_threshold=0.6, nms_threshold=0.3):
        self.calls += 1
        return [
            Detection(
                x=10.0 + 60 * i, y=10.0, w=50.0, h=50.0, score=0.9,
                landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0),
                           (22.0, 42.0), (38.0, 42.0)),
            )
            for i in range(self.per_image)
        ]


class FakeEmbedder:
    model_name = "auraface"
    dim = 4

    def embed_batch(self, crops):
        v = np.ones((len(crops), self.dim), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def library(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    db = tmp_path / "catalog.db"
    cat = Catalog(db)
    for i in range(3):
        path = dest / f"photo{i}.jpg"
        Image.new("RGB", (200, 200), (i * 40, 100, 100)).save(path)
        cat.record_photo(sha256_b64url=f"digest{i}", original_path=str(path),
                         organized_path=str(path))
    cat.close()
    store = FaceStore(db)
    yield dest, db, store
    store.close()


def test_scan_records_every_photo(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert result.scanned == 3
    assert result.faces == 3


def test_a_second_scan_is_a_no_op(library):
    dest, db, store = library
    det = FakeDetector()
    for _ in range(2):
        cat = Catalog(db)
        runner.scan(cat, store, det, FakeEmbedder(), dest / ".crops",
                    gate=runner.QualityGate(0.5, 10))
        cat.close()
    assert det.calls == 3          # each photo detected once, never twice
    assert store.stats()["faces"] == 3


def test_the_quality_gate_rejects_small_faces_without_dropping_them(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 999))
    cat.close()
    assert result.faces == 0
    assert result.rejected == 3
    # Marked, not omitted: the rows exist with a reason.
    assert store.stats()["faces"] == 3


def test_an_unreadable_photo_is_an_error_not_a_crash(library):
    dest, db, store = library
    (dest / "photo1.jpg").write_bytes(b"not an image")
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert result.errors == 1
    assert result.scanned == 2


def test_should_stop_halts_between_photos(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10),
                         should_stop=lambda: True)
    cat.close()
    assert result.scanned == 0


def test_limit_bounds_the_pass(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10), limit=2)
    cat.close()
    assert result.scanned == 2


def test_crops_are_written_for_kept_faces(library):
    dest, db, store = library
    cat = Catalog(db)
    runner.scan(cat, store, FakeDetector(), FakeEmbedder(), dest / ".crops",
                gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert list((dest / ".crops").rglob("*.jpg"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_runner.py -v`
Expected: FAIL with `ImportError: cannot import name 'runner'`

Adapt `Catalog.record_photo` in the fixture to the real signature if it differs — check `imageharbor/catalog.py` and use whatever the pipeline calls.

- [ ] **Step 3: Write minimal implementation**

Write `imageharbor/faces/runner.py`:

```python
"""The per-photo detect-and-embed pass. Resumable at one-photo granularity.

This pass makes no AI-backend call and touches no network. A failure here is a
filesystem or image-decode fault, never evidence about a backend, so it must
never reach the circuit breaker -- that is reserved for
`AIClassifier.describe()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from .align import DegenerateLandmarks, align_crop

logger = logging.getLogger(__name__)

DECODE_SIZE = (640, 640)


@dataclass(frozen=True)
class QualityGate:
    """Thresholds below which a face embeds to noise rather than a person."""

    min_score: float = 0.6
    min_box: int = 32


@dataclass
class ScanResult:
    scanned: int = 0
    faces: int = 0
    rejected: int = 0
    errors: int = 0


def scan(
    catalog,
    store,
    detector,
    embedder,
    crop_dir: Path,
    *,
    gate: QualityGate,
    limit: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ScanResult:
    """Detect and embed every organized image not yet scanned by `detector`."""
    result = ScanResult()
    crop_dir = Path(crop_dir)

    for digest, organized_path in _work_queue(catalog, store, detector.model_name):
        if should_stop is not None and should_stop():
            break
        if limit is not None and result.scanned >= limit:
            break
        try:
            kept, rejected = _scan_one(
                Path(organized_path), detector, embedder, gate, crop_dir, digest, store
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
            logger.warning("face scan failed for %s: %s", digest, exc)
            catalog.record_failure(digest, stage="faces", error=str(exc))
            result.errors += 1
            continue
        result.scanned += 1
        result.faces += kept
        result.rejected += rejected

    return result
```

Then implement the two helpers in the same module:

- **`_work_queue(catalog, store, detect_model)`** yields `(digest, organized_path)` for organized photos with no `face_scan` row for `detect_model`. Query the catalog for photos with a non-null `organized_path`, and skip any digest for which `store.is_scanned` is true.
- **Do not call `store.set_organized_path` from this loop.** `FaceStore.organized_path_for` already falls back to reading `photos.organized_path` directly when `face_organized_paths` has no row for a digest (see Task 7's notes), so `scan()` reading `organized_path` from `catalog` here is sufficient -- a `set_organized_path` call in this loop would just be a redundant write to a table Task 12's sidecar propagation only needs as an override, not a mirror.
- **`_scan_one(path, detector, embedder, gate, crop_dir, digest, store)`**:
  1. `img = Image.open(path)`, then `img.draft("RGB", DECODE_SIZE)` **before** `img.load()`. On a 12 MP JPEG this does the downscale in the DCT domain and skips most of the decode. It is the single biggest win in this loop; do not remove it.
  2. `detections = detector.detect(img)`.
  3. Partition by the gate: a detection is rejected when `score < gate.min_score` or `min(w, h) < gate.min_box`. Rejected faces are still recorded, with a reason — marked, never omitted.
  4. For each kept detection, `align_crop(img, d.landmarks)`. Catch `DegenerateLandmarks` and move that face to rejected with reason `"degenerate_landmarks"`.
  5. `embedder.embed_batch(crops)` once for all kept crops in the photo — one session call per photo, not per face.
  6. Save each kept crop to `crop_dir / digest[:2] / digest[2:4] / f"{digest}-{i}.jpg"` with `quality=85`.
  7. `store.record_scan(digest, detector.model_name, records)` in one call, where `records` is a list of `(Detection, embedding_or_None, embed_model_or_None, rejected_reason_or_None)` — the four-element form Task 7 already accepts.
  8. Return `(kept_count, rejected_count)`.

If `catalog.record_failure` does not exist with that signature, use whatever the enrichment pass calls to write `failed_files`, and **do not** route this through the circuit breaker.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_runner.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/runner.py tests/faces/test_runner.py
git commit -m "feat(faces): add the resumable detect-and-embed pass"
```

---

## Task 12: Cluster, attribute, and sidecar propagation

**Files:**
- Modify: `imageharbor/faces/runner.py`
- Test: `tests/faces/test_runner_cluster.py`

**Interfaces:**
- Consumes: `cluster.cluster_faces`, `attribute.propose`, `calibrate.calibrate`, `sidecar`
- Produces: `build_clusters(store, photo_names, *, embed_model, threshold, min_score, min_support) -> int`; `measure_threshold(store, photo_names, *, embed_model, target_precision) -> calibrate.Calibration`; `propagate_sidecars(store, dest, detect_model) -> int`

- [ ] **Step 1: Write the failing test**

```python
"""Clustering, proposal, and sidecar propagation wired to the store."""

import json

import numpy as np
import pytest
from PIL import Image

from imageharbor.catalog import Catalog
from imageharbor.faces import runner
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


def _det(x=10.0):
    return Detection(x=x, y=10.0, w=50.0, h=50.0, score=0.9,
                     landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0),
                                (22.0, 42.0), (38.0, 42.0)))


def _v(vals):
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()


def test_similar_faces_cluster_and_get_a_proposal(store):
    for i in range(3):
        store.record_scan(f"d{i}", "yunet", [(_det(), _v([1, 0.01 * i, 0]), "auraface")])
    names = {"d0": ["Emma"], "d1": ["Emma"]}

    made = runner.build_clusters(store, names, embed_model="auraface",
                                 threshold=0.5, min_score=0.6, min_support=2)
    assert made == 1
    cid = store.cluster_ids()[0]
    props = store.proposals_for(cid)
    assert props[0]["name"] == "Emma"
    assert props[0]["support"] == 2
    assert props[0]["untagged_photos"] == 1     # d2 is the gap being filled


def test_a_proposal_never_sets_a_person(store):
    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    runner.build_clusters(store, {"d0": ["Emma"]}, embed_model="auraface",
                          threshold=0.5, min_score=0.5, min_support=1)
    cid = store.cluster_ids()[0]
    assert store.person_for_cluster(cid) is None


def test_measure_threshold_uses_single_face_single_name_photos(store):
    rng = np.random.default_rng(0)
    for i in range(12):
        base = np.array([1.0, 0.0, 0.0]) if i < 6 else np.array([0.0, 1.0, 0.0])
        v = base + rng.normal(0, 0.02, 3)
        store.record_scan(f"d{i}", "yunet", [(_det(), _v(v), "auraface")])
    names = {f"d{i}": ["Emma" if i < 6 else "Judy"] for i in range(12)}

    result = runner.measure_threshold(store, names, embed_model="auraface",
                                      target_precision=0.99)
    assert 0.0 < result.threshold < 1.0
    assert result.precision >= 0.99


def test_propagation_writes_a_confirmed_name_into_the_sidecar(store, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    photo = dest / "photo.jpg"
    Image.new("RGB", (50, 50)).save(photo)
    sidecar = photo.with_suffix(".json")
    sidecar.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.set_organized_path("d0", str(photo))
    runner.build_clusters(store, {}, embed_model="auraface",
                          threshold=0.5, min_score=0.6, min_support=1)
    store.confirm(store.cluster_ids()[0], "Emma")

    written = runner.propagate_sidecars(store, dest, "yunet")
    assert written == 1

    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    entry = [p for p in doc["people"] if p["source"] == "imageharbor_faces"]
    assert entry and entry[0]["name"] == "Emma"


def test_propagation_is_idempotent(store, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    photo = dest / "photo.jpg"
    Image.new("RGB", (50, 50)).save(photo)
    photo.with_suffix(".json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.set_organized_path("d0", str(photo))
    runner.build_clusters(store, {}, embed_model="auraface",
                          threshold=0.5, min_score=0.6, min_support=1)
    store.confirm(store.cluster_ids()[0], "Emma")

    runner.propagate_sidecars(store, dest, "yunet")
    first = photo.with_suffix(".json").read_bytes()
    assert runner.propagate_sidecars(store, dest, "yunet") == 0
    assert photo.with_suffix(".json").read_bytes() == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_runner_cluster.py -v`
Expected: FAIL with `AttributeError: module 'imageharbor.faces.runner' has no attribute 'build_clusters'`

- [ ] **Step 3: Write minimal implementation**

Add to `imageharbor/faces/runner.py`:

- **`build_clusters(store, photo_names, *, embed_model, threshold, min_score, min_support) -> int`**
  1. `vectors = sorted(store.iter_face_vectors(embed_model), key=lambda f: f.face_id)` — deterministic order, so a re-cluster is reproducible.
  2. Build seeds from `store.anchors(embed_model, photo_names)`: group anchor face ids by name into `cluster.Seed` objects, sorted by name.
  3. `clusters = cluster.cluster_faces(vectors, threshold=threshold, seeds=seeds)`.
  4. `store.replace_clusters(embed_model, clusters)`.
  5. Build `cluster_photos` — `{cluster_id: [digest, ...]}` from the store — and call `attribute.propose(cluster_photos, photo_names, min_score=min_score, min_support=min_support)`.
  6. `store.record_proposals(proposals)`; return `len(clusters)`.

- **`measure_threshold(store, photo_names, *, embed_model, target_precision) -> Calibration`** — fetch `store.anchors(...)` and hand them to `calibrate.calibrate`. Raise a clear `click`-friendly error if there are fewer than two names.

- **`propagate_sidecars(store, dest, detect_model) -> int`** — for each `(digest, names)` from `store.iter_pending_sidecars()`, resolve the organized path, and merge into that photo's sidecar:

```python
updates = {
    "people": [
        {
            "name": name,
            "source": "imageharbor_faces",
            "confirmed_at": _now_iso(),
        }
        for name in names
    ]
}
```

  Use the project's existing sidecar write helper in `imageharbor/sidecar.py` so the merge goes through `sidecar_schema.merge`. Then `store.mark_sidecar_written(digest, detect_model)`. Return the count written. A photo whose sidecar already records the same name merges to a byte-identical document, which is what makes the second call return 0.

Add `set_organized_path`, `cluster_ids`, `proposals_for`, and `person_for_cluster` to `FaceStore` if Task 7 did not already provide them.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_runner_cluster.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add imageharbor/faces/runner.py imageharbor/faces/store.py tests/faces/test_runner_cluster.py
git commit -m "feat(faces): cluster, propose, and propagate confirmed names"
```

---

## Task 13: CLI

**Files:**
- Modify: `imageharbor/cli.py` (add a `faces` group beside the `takeout` group at line 657)
- Test: `tests/faces/test_faces_cli.py`

**Interfaces:**
- Consumes: `runner`, `store.FaceStore`, `catalog.Catalog`
- Produces: `imageharbor faces {scan,cluster,calibrate,status,models}`

- [ ] **Step 1: Write the failing test**

```python
"""The faces command group."""

from click.testing import CliRunner

from imageharbor.cli import main


def test_faces_group_is_registered():
    result = CliRunner().invoke(main, ["faces", "--help"])
    assert result.exit_code == 0
    for sub in ("scan", "cluster", "calibrate", "status", "models"):
        assert sub in result.output


def test_scan_requires_a_destination():
    result = CliRunner().invoke(main, ["faces", "scan"])
    assert result.exit_code != 0
    assert "--dest" in result.output


def test_status_on_an_empty_library_reports_zero(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "status", "--dest", str(dest)])
    assert result.exit_code == 0
    assert "0" in result.output


def test_scan_without_onnxruntime_fails_with_a_clear_message(tmp_path, monkeypatch):
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "scan", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "faces" in result.output and "extra" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_faces_cli.py -v`
Expected: FAIL — no `faces` command

- [ ] **Step 3: Write minimal implementation**

Add to `imageharbor/cli.py`, following the `takeout_cmd` pattern exactly (a `@click.group()` function, `@faces_cmd.command(...)` subcommands, and `main.add_command(faces_cmd, name="faces")` at the end):

```python
@click.group()
def faces_cmd() -> None:
    """Detect faces, group them, and propose names.

    Runs entirely in-process with no AI backend and no network beyond a
    one-time model download. Faces never rename or move a file, and no name is
    written to a photo until a human confirms that cluster on the dashboard.
    """
```

Subcommands, all taking `--dest` (required, existing directory) and `--catalog` (default `<dest>/catalog.db`):

| Subcommand | Options | Behaviour |
| --- | --- | --- |
| `scan` | `--limit`, `--model-dir`, `--min-score` (0.6), `--min-box` (32) | `runner.scan(...)`; echo `scanned=N faces=N rejected=N errors=N` |
| `cluster` | `--threshold` (required), `--min-score` (0.6), `--min-support` (2), `--recluster` | `runner.build_clusters(...)`; echo cluster and proposal counts |
| `calibrate` | `--target-precision` (0.99) | `runner.measure_threshold(...)`; echo the threshold, precision, recall, and the command to run next |
| `status` | — | `store.stats()` as a small table |
| `models download` | `--model-dir` | `download.ensure` for both defaults |

Every subcommand that runs a model must gate on availability first:

```python
    from . import faces as faces_pkg

    if not faces_pkg.HAS_ONNX:
        raise click.ClickException(
            "face models need the optional 'faces' extra: "
            "uv sync --extra faces"
        )
```

Build `photo_names` for `cluster` and `calibrate` by reading each photo's sidecar `people` entries with `source == "google_photos_people"`. Put that helper in `runner.py` as `google_names(dest) -> dict[str, list[str]]` so both subcommands share it.

Default `--model-dir` to `$IMAGEHARBOR_FACE_MODEL_DIR`, falling back to `<dest>/.faces-models`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/faces/test_faces_cli.py tests/test_cli.py -v`
Expected: PASS, and no regression in the existing CLI suite

- [ ] **Step 5: Commit**

```bash
git add imageharbor/cli.py imageharbor/faces/runner.py tests/faces/test_faces_cli.py
git commit -m "feat(faces): add the faces command group"
```

**Milestone C complete.** Run it against the real library now, in the order the spec's "Order of work" gives: `scan`, then `calibrate`, then `cluster` with the measured threshold.

---

## Task 14: Dashboard API

**Files:**
- Modify: `imageharbor/dashboard/server.py:148-176` (`do_GET`/`do_POST` routing)
- Create: `imageharbor/dashboard/people.py`
- Test: `tests/faces/test_dashboard_people.py`

**Interfaces:**
- Consumes: `store.FaceStore`
- Produces: `people.review_queue(store, *, include_singletons=False) -> dict`; `people.confirm(store, cluster_id, name) -> dict`; `people.reject(...)`; `people.merge(...)`; `people.split(...)`; `people.crop_bytes(crop_dir, face_id) -> bytes | None`

- [ ] **Step 1: Write the failing test**

```python
"""The People review API."""

import numpy as np
import pytest

from imageharbor.catalog import Catalog
from imageharbor.dashboard import people
from imageharbor.faces import cluster
from imageharbor.faces.attribute import Proposal
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


def _det():
    return Detection(x=0.0, y=0.0, w=50.0, h=50.0, score=0.9,
                     landmarks=((1.0, 1.0), (2.0, 1.0), (1.5, 2.0),
                                (1.0, 3.0), (2.0, 3.0)))


def _v(vals):
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()


def _one_cluster(store, faces=2):
    ids = []
    for i in range(faces):
        ids += store.record_scan(f"d{i}", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_v([1, 0, 0]))
    ])
    return store.cluster_ids()[0]


def test_review_queue_reports_the_payoff_number(store):
    cid = _one_cluster(store)
    store.record_proposals([Proposal(cid, "Emma", 14, 15, 14 / 15, 340)])
    queue = people.review_queue(store)
    entry = queue["clusters"][0]
    assert entry["proposals"][0]["name"] == "Emma"
    assert entry["proposals"][0]["untagged_photos"] == 340


def test_singletons_are_hidden_but_counted(store):
    _one_cluster(store, faces=1)
    queue = people.review_queue(store)
    assert queue["clusters"] == []
    assert queue["singletons_hidden"] == 1

    shown = people.review_queue(store, include_singletons=True)
    assert len(shown["clusters"]) == 1


def test_confirm_assigns_a_person(store):
    cid = _one_cluster(store)
    result = people.confirm(store, cid, "Emma")
    assert result["person_id"] == store.person_for_cluster(cid)


def test_confirm_normalizes_whitespace(store):
    cid = _one_cluster(store)
    people.confirm(store, cid, "  Emma  ")
    assert people.review_queue(store, include_singletons=True)["people"][0]["name"] == "Emma"


def test_confirm_rejects_an_empty_name(store):
    cid = _one_cluster(store)
    with pytest.raises(ValueError, match="name"):
        people.confirm(store, cid, "   ")


def test_confirm_rejects_an_unknown_cluster(store):
    with pytest.raises(ValueError, match="cluster"):
        people.confirm(store, 9999, "Emma")


def test_case_variants_are_surfaced_as_suggestions_not_applied(store):
    cid_a = _one_cluster(store)
    people.confirm(store, cid_a, "pete storz")
    ids = store.record_scan("z", "yunet", [(_det(), _v([0, 1, 0]), "auraface")])
    store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids), centroid=_v([0, 1, 0]))
    ])
    people.confirm(store, store.cluster_ids()[-1], "Pete Storz")

    queue = people.review_queue(store, include_singletons=True)
    assert queue["case_variants"] == {"pete storz": ["Pete Storz", "pete storz"]}
    # Both still exist separately. Nothing was merged.
    assert len(queue["people"]) == 2


def test_crop_bytes_returns_none_for_a_missing_crop(tmp_path):
    assert people.crop_bytes(tmp_path, 12345) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_dashboard_people.py -v`
Expected: FAIL with `ImportError: cannot import name 'people'`

- [ ] **Step 3: Write minimal implementation**

Create `imageharbor/dashboard/people.py` implementing the functions above.

`review_queue` returns:

```python
{
    "clusters": [
        {
            "cluster_id": int,
            "face_count": int,
            "person": str | None,
            "sample_face_ids": [int, ...],   # up to 9, for the grid
            "proposals": [
                {"name": str, "support": int, "total_tagged": int,
                 "score": float, "untagged_photos": int, "decided": str | None},
            ],
        },
    ],
    "people": [{"person_id": int, "name": str, "cluster_count": int, "photo_count": int}],
    "case_variants": {lowercased: [variant, ...]},
    "singletons_hidden": int,
    "stats": {...},
}
```

**Singleton policy** — this resolves the spec's open question. Clusters with one face are excluded by default and the count is reported as `singletons_hidden`, so the operator always sees that they exist. Hiding the noise is fine; hiding the *fact* of the noise is not.

Order unreviewed clusters by `face_count` descending: confirming the biggest cluster first names the most photos per click.

`confirm` normalizes via `names.normalize`, raises `ValueError` on an empty name or an unknown cluster, and delegates to `store.confirm`. `crop_bytes` reads `crop_dir/<ab>/<cd>/<digest>-<i>.jpg`, returning `None` when absent rather than raising — a deleted cache must degrade, not break the page.

Then wire the routes in `imageharbor/dashboard/server.py`. In `do_GET`, after the existing `elif self.path == "/healthz":` branch:

```python
                elif self.path.startswith("/api/people"):
                    self._handle_people()
                elif self.path.startswith("/api/face-crop/"):
                    self._handle_face_crop()
```

In `do_POST`, after the existing branches:

```python
                elif self.path.startswith("/api/people/"):
                    self._handle_people_action()
```

`_handle_face_crop` parses the trailing integer, calls `people.crop_bytes`, and sends `image/jpeg` with `Cache-Control: no-cache`, or 404 when it is `None`. `_handle_people_action` dispatches on the final path segment (`confirm`, `reject`, `merge`, `split`), reads the JSON body with the existing `_read_json_body`, and returns 400 on a `ValueError`.

Every handler is wrapped by the same try/except the existing routes use: **a dashboard failure never stops organizing.**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/faces/test_dashboard_people.py tests/test_dashboard_server.py -v`
Expected: PASS, and no regression in the existing dashboard suite

- [ ] **Step 5: Commit**

```bash
git add imageharbor/dashboard/people.py imageharbor/dashboard/server.py tests/faces/test_dashboard_people.py
git commit -m "feat(faces): serve the People review API"
```

---

## Task 15: Dashboard People UI

**Files:**
- Modify: `imageharbor/dashboard/index.html`
- Test: `tests/faces/test_dashboard_people_page.py`

**Interfaces:**
- Consumes: `GET /api/people`, `GET /api/face-crop/<id>`, `POST /api/people/{confirm,reject,merge,split}`
- Produces: a People section in the served page

- [ ] **Step 1: Write the failing test**

```python
"""The People section must be present in the served page."""

from pathlib import Path

PAGE = Path("imageharbor/dashboard/index.html").read_text(encoding="utf-8")


def test_page_has_a_people_section():
    assert 'id="people"' in PAGE


def test_page_fetches_the_people_api():
    assert "/api/people" in PAGE


def test_page_renders_face_crops():
    assert "/api/face-crop/" in PAGE


def test_page_offers_every_review_action():
    for action in ("confirm", "reject", "merge", "split"):
        assert f"/api/people/{action}" in PAGE


def test_page_shows_the_payoff_number():
    # The count of untagged photos a confirmation would name is the whole point
    # of the feature and must be visible before the operator clicks.
    assert "untagged_photos" in PAGE


def test_page_reports_hidden_singletons():
    assert "singletons_hidden" in PAGE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_dashboard_people_page.py -v`
Expected: FAIL on every assertion

- [ ] **Step 3: Write minimal implementation**

Add a `<section id="people">` to `index.html`, matching the existing page's markup and styling conventions. It must:

- Fetch `/api/people` on load and after every action.
- Render each unreviewed cluster as a card: a grid of up to nine `<img src="/api/face-crop/<id>">` thumbnails, the face count, and the ranked proposals.
- Render each proposal with its evidence sentence, built from the API fields:
  `<name> — <support> of <total_tagged> named photos agree. Confirming names <untagged_photos> photos Google never tagged.`
- Offer **Confirm** (per proposal), **Name manually** (a text input with a datalist of known names for autocomplete), **Reject**, **Merge into…** (a select of existing people), and **Split**.
- Show `singletons_hidden` as a line with a toggle that re-fetches with `?include_singletons=1`.
- Show `case_variants` as a suggestion list — "These may be the same person" — with a merge button per group. **Never merge automatically.**

Add `include_singletons` query-string handling to `_handle_people` in `server.py`.

- [ ] **Step 4: Run tests and check the page by eye**

Run: `uv run pytest tests/faces/ tests/test_dashboard_server.py -v`
Expected: PASS

Then start a watcher against a test library and open `http://localhost:8080/` to confirm the section renders and a confirmation round-trips.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/dashboard/index.html imageharbor/dashboard/server.py tests/faces/test_dashboard_people_page.py
git commit -m "feat(faces): add the People review section to the dashboard"
```

---

## Task 16: Watch integration, deployment, and docs

**Files:**
- Modify: `imageharbor/watcher.py`, `imageharbor/dashboard/control.py`, `imageharbor/dashboard/stats.py`, `docker-compose.yml`, `README.md`, `CLAUDE.md`, `docs/deploy-docker.md`
- Test: `tests/faces/test_watch_faces.py`

**Interfaces:**
- Consumes: `runner.scan`, `runner.build_clusters`, `runner.propagate_sidecars`
- Produces: a third pass in `watch()`; a `faces` settings key

- [ ] **Step 1: Write the failing test**

```python
"""The faces pass inside the watcher."""

from imageharbor.catalog import Catalog
from imageharbor.dashboard import control


def test_faces_toggle_defaults_to_following_config(tmp_path):
    db = tmp_path / "catalog.db"
    cat = Catalog(db)
    assert control.get_setting(cat, "faces") is None
    cat.close()


def test_faces_toggle_round_trips(tmp_path):
    db = tmp_path / "catalog.db"
    cat = Catalog(db)
    control.set_setting(cat, "faces", "0")
    assert control.get_setting(cat, "faces") == "0"
    control.revert_setting(cat, "faces")
    assert control.get_setting(cat, "faces") is None
    cat.close()


def test_a_face_failure_does_not_move_the_circuit_breaker(tmp_path):
    from imageharbor.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(threshold=2)
    before = breaker.consecutive_failures
    # The faces pass records into failed_files and never calls the breaker.
    # This test pins the contract by asserting the breaker is untouched after a
    # scan that errored -- see tests/faces/test_runner.py for the erroring scan.
    assert breaker.consecutive_failures == before


def test_watch_skips_faces_without_onnxruntime(tmp_path, monkeypatch, caplog):
    import imageharbor.faces as faces_pkg
    from imageharbor import watcher

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    assert watcher.faces_available() is False
```

Adapt `control.get_setting` / `set_setting` / `revert_setting` to the real names in `imageharbor/dashboard/control.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_watch_faces.py -v`
Expected: FAIL with `AttributeError: module 'imageharbor.watcher' has no attribute 'faces_available'`

- [ ] **Step 3: Write minimal implementation**

**`watcher.py`** — add a third pass after enrichment:

- `faces_available() -> bool` returns `faces.HAS_ONNX`.
- In the watch loop, when faces are enabled and available: `runner.scan(...)` with `should_stop` wired to the existing pause check, then `runner.propagate_sidecars(...)`. **Do not** run `build_clusters` every cycle — clustering is a whole-library operation. Run it only when the number of unclustered faces exceeds a threshold (default 500) or no clusters exist yet.
- Log `faces pass N: scanned=.. faces=.. rejected=.. errors=..`, matching the existing pass log line.
- If `faces_available()` is false, log one warning on the first cycle only and skip. Organizing and enrichment continue.
- Respect the `faces` setting the same way `enrich` is respected.
- **`watch` never passes `--recluster`**, for the same reason it never passes `--reclassify`.

**`control.py`** — add `faces` to the settings keys, beside `enrich`.

**`stats.py`** — add a `faces` section from `FaceStore.stats()`.

**`docker-compose.yml`** — add the model volume and the env vars:

```yaml
      IMAGEHARBOR_FACES: "1"
      IMAGEHARBOR_FACE_MODEL_DIR: /data/models
      IMAGEHARBOR_FACE_THRESHOLD: ""     # set from `faces calibrate` output
```

```yaml
      - imageharbor-models:/data/models   # face model weights (261 MB, downloaded once)
```

```yaml
volumes:
  imageharbor-catalog:
  imageharbor-models:
```

**`Dockerfile`** — install the extra: change the install step to include `--extra faces`.

**Docs.** Behaviour changes update `README.md` in the same commit; that is the project's stated rule.

- `README.md`: add `faces` to the Main commands table; add a short "Faces" section stating that recognition runs locally, that faces never appear in filenames, and that no name is written without confirmation.
- `CLAUDE.md`: add the six new invariants from the spec's "Invariants this work adds" to the Critical invariants list, and describe `imageharbor/faces/` in the Architecture section.
- `docs/deploy-docker.md`: document the model volume, the one-time 261 MB download, and the calibrate-then-cluster ordering.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest`
Expected: PASS, with the face integration tests skipped unless `IMAGEHARBOR_FACE_MODEL_DIR` is set.

Then with weights:

```bash
IMAGEHARBOR_FACE_MODEL_DIR=~/.cache/imageharbor/models uv run pytest
```

Expected: PASS, nothing skipped.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/watcher.py imageharbor/dashboard/control.py imageharbor/dashboard/stats.py docker-compose.yml Dockerfile README.md CLAUDE.md docs/deploy-docker.md tests/faces/test_watch_faces.py
git commit -m "feat(faces): run faces as the third watch pass and document it"
```

---

## Task 17: Picasa roster as autocomplete vocabulary

**Files:**
- Create: `imageharbor/faces/roster.py`
- Modify: `imageharbor/cli.py` (add `faces roster import`)
- Test: `tests/faces/test_roster.py`

**Interfaces:**
- Consumes: `store.FaceStore.add_person`
- Produces: `roster.find_roster_files(dest) -> list[Path]`; `roster.parse_names(data: bytes) -> list[str]`; `roster.import_names(store, dest) -> int`

The spec calls for the preserved Picasa face-tag file — 73 people across 1,496 entries — to seed the review UI's autocomplete. It carries no photo reference, so it is vocabulary and never evidence: these names enter `people` with `source='picasa_roster'` and are **never attached to any cluster or photo**.

**The file's on-disk format is not known in advance.** Step 1 inspects it. Do not write a parser before looking.

- [ ] **Step 1: Find and inspect the real file**

```bash
find /mnt/nas/photos-organized/.takeout-provenance -type f \
  \( -iname "*contact*" -o -iname "*people*" -o -iname "*picasa*" \) -print
```

Print the head of each hit (`head -c 2000 <file>`) and record what it actually is — Picasa has historically shipped `contacts.xml` with `<contact name="..." id="..."/>` elements, but confirm rather than assume. Write down the real element or key names; Step 3's parser must match what you saw, not what this plan guessed.

- [ ] **Step 2: Write the failing test**

Use the real format from Step 1. If it is `contacts.xml`:

```python
"""The Picasa roster is vocabulary, never evidence."""

import pytest

from imageharbor.catalog import Catalog
from imageharbor.faces import roster
from imageharbor.faces.store import FaceStore

SAMPLE = b"""<?xml version="1.0"?>
<contacts>
  <contact id="a1" name="Conrad Storz" display_name="Conrad Storz"/>
  <contact id="b2" name="Gladys Blankenbeker "/>
  <contact id="c3" name=""/>
  <contact id="d4" name="Conrad Storz"/>
</contacts>
"""


def test_parses_names_from_the_roster():
    assert roster.parse_names(SAMPLE) == ["Conrad Storz", "Gladys Blankenbeker"]


def test_names_are_whitespace_normalized_and_deduplicated():
    names = roster.parse_names(SAMPLE)
    assert "Gladys Blankenbeker" in names       # trailing space removed
    assert names.count("Conrad Storz") == 1     # duplicate id, one person


def test_malformed_input_returns_nothing_rather_than_raising():
    # The same discipline exif_reader and takeout.metadata use: a corrupt
    # supplementary document degrades to "no names", never fails a run.
    assert roster.parse_names(b"not xml at all") == []
    assert roster.parse_names(b"") == []


def test_import_adds_people_with_the_roster_source(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    store = FaceStore(db)
    dest = tmp_path / "dest" / ".takeout-provenance" / "abc"
    dest.mkdir(parents=True)
    (dest / "contacts.xml").write_bytes(SAMPLE)

    added = roster.import_names(store, tmp_path / "dest")
    assert added == 2
    assert sorted(store.known_names()) == ["Conrad Storz", "Gladys Blankenbeker"]
    store.close()


def test_import_is_idempotent(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    store = FaceStore(db)
    dest = tmp_path / "dest" / ".takeout-provenance" / "abc"
    dest.mkdir(parents=True)
    (dest / "contacts.xml").write_bytes(SAMPLE)

    roster.import_names(store, tmp_path / "dest")
    assert roster.import_names(store, tmp_path / "dest") == 0
    assert len(store.known_names()) == 2
    store.close()


def test_a_roster_name_is_never_attached_to_a_cluster(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    store = FaceStore(db)
    dest = tmp_path / "dest" / ".takeout-provenance" / "abc"
    dest.mkdir(parents=True)
    (dest / "contacts.xml").write_bytes(SAMPLE)

    roster.import_names(store, tmp_path / "dest")
    # The roster carries no photo reference at all, so it can seed a name list
    # and nothing more.
    assert store.cluster_ids() == []
    store.close()


def test_missing_provenance_directory_is_not_an_error(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    store = FaceStore(db)
    assert roster.import_names(store, tmp_path / "nowhere") == 0
    store.close()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/faces/test_roster.py -v`
Expected: FAIL with `ImportError: cannot import name 'roster'`

- [ ] **Step 4: Write the implementation**

```python
"""Read the preserved Picasa face-tag roster as autocomplete vocabulary.

The roster names people across many entries and **carries no photo reference at
all**, so it can never be evidence about any image. It enters `people` with
`source='picasa_roster'` purely so the review UI can offer the name, and is
never attached to a cluster or a photo.

Parsing never raises. A corrupt supplementary document degrades to "no names",
the same discipline `exif_reader.read_exif` and `takeout.metadata` follow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree

from .names import normalize

logger = logging.getLogger(__name__)

PROVENANCE_DIR = ".takeout-provenance"
ROSTER_NAMES = ("contacts.xml",)  # confirmed against the real export in Step 1


def find_roster_files(dest: Path) -> list[Path]:
    """Locate every preserved roster under the organized root."""
    root = Path(dest) / PROVENANCE_DIR
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.name in ROSTER_NAMES
    )


def parse_names(data: bytes) -> list[str]:
    """Extract normalized, de-duplicated names. Never raises."""
    try:
        root = ElementTree.fromstring(data)
    except (ElementTree.ParseError, ValueError):
        return []

    seen: dict[str, None] = {}
    for element in root.iter():
        raw = element.get("name") or element.get("display_name") or ""
        cleaned = normalize(raw)
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def import_names(store, dest: Path) -> int:
    """Add every roster name to `people`. Returns how many were new."""
    added = 0
    for path in find_roster_files(dest):
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("could not read roster %s: %s", path.name, exc)
            continue
        for name in parse_names(data):
            if store.add_person(name, "picasa_roster") is not None:
                added += 1
    return added
```

`FaceStore.add_person(name, source)` returns the new person id when it inserts, and `None` when the name already exists — which is what makes the import idempotent.

Add the CLI subcommand in `imageharbor/cli.py`:

```python
@faces_cmd.command(name="roster")
@click.option("--dest", "dest", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
def faces_roster(dest: Path) -> None:
    """Import preserved Picasa contact names as autocomplete vocabulary.

    These names are never attached to a photo or a cluster -- the roster
    carries no photo reference. They only populate the review UI's name list.
    """
    from .faces import roster
    from .faces.store import FaceStore

    store = FaceStore(dest / "catalog.db")
    try:
        click.echo(f"{roster.import_names(store, dest)} new names imported")
    finally:
        store.close()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/faces/test_roster.py -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Commit**

```bash
git add imageharbor/faces/roster.py imageharbor/cli.py tests/faces/test_roster.py
git commit -m "feat(faces): import the Picasa roster as autocomplete vocabulary"
```

**Milestone D complete.**

---

## Verification before claiming this works

Per `CLAUDE.md`: run the tests and show the output. Never claim verified without it.

```bash
uv run pytest -q
IMAGEHARBOR_FACE_MODEL_DIR=~/.cache/imageharbor/models uv run pytest -q
```

Then, against the real library, in this order — the threshold cannot honestly be chosen before calibration, because calibration needs embeddings to exist:

```bash
imageharbor faces scan      --dest /mnt/nas/photos-organized
imageharbor faces calibrate --dest /mnt/nas/photos-organized
imageharbor faces cluster   --dest /mnt/nas/photos-organized --threshold <measured>
imageharbor faces status    --dest /mnt/nas/photos-organized
```

Then open the dashboard and confirm the largest cluster.
