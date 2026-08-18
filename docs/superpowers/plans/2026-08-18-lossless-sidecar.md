# Lossless Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the per-image JSON sidecar a complete, permanent record that may gain information and can never lose any — default-on, append-only, with source documents preserved verbatim.

**Architecture:** A new pure module `sidecar_schema.py` owns the merge policy (history, keyed lists, migration) with no I/O, so the never-lose guarantee can be tested exhaustively rather than illustratively. `sidecar.py` keeps its public surface and delegates all policy to it. Writers each contribute their own keys and never coordinate. Archive members that are not media are preserved verbatim in a library-level provenance room.

**Tech Stack:** Python 3, stdlib `json`/`os`, Click, pytest, `uv`.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-08-18-lossless-sidecar-design.md`. Every task's requirements implicitly include this section.

- **A sidecar may gain information. It may never lose any.** For any base `B` and updates `U`, `merge(B, U)` contains every value present in `B` and every value present in `U`. A superseded value is relocated into a history list, never dropped.
- **Idempotence:** `merge(merge(B, U), U)` is byte-identical to `merge(B, U)`. History dedupes on the *value*, never on a timestamp — a timestamp in the dedup key makes every re-run append.
- **Monotonic size:** the merged document is never smaller than the base.
- **`sidecar_schema.py` is pure:** no filesystem, no imports from the rest of `imageharbor` except `tiers` if needed for constants. `merge` never raises.
- **A sidecar failure must never fail an image** that is already copied, verified, and catalogued. Every sidecar write stays inside a try/except that logs and continues.
- **A corrupt existing sidecar is renamed, never overwritten** — `<name>.json.corrupt-<timestamp>` — and a fresh sidecar is written. Treating it as empty (today's behavior) would destroy data under the new rule.
- **`SCHEMA_VERSION = 2`.** v1 → v2 migration is itself a merge and obeys the same never-lose rule.
- **Unknown and hand-written keys are preserved untouched.**
- **Preserve everything that is not media** in the provenance room. No curation.
- **The Picasa face tags are preserved, never attached.** They carry no photo reference; `people[]` is populated only from a Google Photos export's inline `people` field.
- Python is managed with `uv` (`uv run pytest`); never pip or venv. Do not chain shell commands with `&&`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `imageharbor/sidecar_schema.py` | create | Pure merge policy: field shapes, history, keyed lists, v1 migration |
| `imageharbor/sidecar.py` | modify | Read, atomic write, corrupt-file quarantine; delegates policy |
| `imageharbor/pipeline.py` | modify | `sources[].folder` |
| `imageharbor/enrich.py` | modify | `classification.model_version` |
| `imageharbor/takeout/metadata.py` | modify | `AlbumMetadata` gains `access` and `date` |
| `imageharbor/takeout/ingest.py` | modify | `provenance[].raw`, `albums[]`, the provenance room |
| `imageharbor/takeout/provenance.py` | create | Write and index the provenance room |
| `imageharbor/cli.py` | modify | Flip four `--sidecar` defaults; add `sidecar backfill` |
| `imageharbor/backfill.py` | create | Rebuild sidecars from the catalog |
| `.gitignore` | modify | `.takeout-provenance/` |
| `tests/test_sidecar_schema.py` | create | The never-lose property, idempotence, keying, migration |
| `tests/test_sidecar.py` | create | Atomicity, corrupt-file quarantine |
| `tests/test_backfill.py` | create | Backfill correctness and idempotence |
| `tests/test_takeout_provenance.py` | create | Provenance room, orphaned sidecars |
| `tests/test_takeout_ingest.py` | modify | Raw capture, albums, accumulation across archives |
| `tests/test_pipeline.py`, `test_enrich.py`, `test_cli.py` | modify | Folder capture, model version, default flip, backfill verb |

---

### Task 1: `sidecar_schema.py` — the pure merge policy

The heart of the project. Everything else is plumbing.

**Files:**
- Create: `imageharbor/sidecar_schema.py`
- Test: `tests/test_sidecar_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SCHEMA_VERSION: int = 2`
  - `merge(base: dict, updates: dict, *, observed_at: str) -> dict` — pure, total, never raises
  - `migrate(doc: dict) -> dict` — v1 → v2, lossless
  - `KEYED_LISTS: dict[str, tuple[str, ...]]` — the identity key per list field
  - **Not** `is_noop(base, merged)`, which the spec's module sketch listed. The only caller that needs that answer is `sidecar backfill`, and comparing the file's bytes before and after the write is simpler and more truthful than asking the policy to predict its own output. Do not add it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sidecar_schema.py`:

```python
"""Tests for the sidecar merge policy.

The module's contract is a single sentence -- a sidecar may gain information
and may never lose any -- so these tests state that formally over generated
merge sequences rather than sampling it with hand-picked cases.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from imageharbor.sidecar_schema import SCHEMA_VERSION, merge, migrate

T0 = "2026-08-18T10:00:00+00:00"
T1 = "2026-08-19T10:00:00+00:00"


def _leaves(doc, out=None):
    """Every scalar leaf in a document, as a multiset-ish set of values."""
    out = set() if out is None else out
    if isinstance(doc, dict):
        for v in doc.values():
            _leaves(v, out)
    elif isinstance(doc, list):
        for v in doc:
            _leaves(v, out)
    elif doc is not None:
        out.add(repr(doc))
    return out


# --- the guarantee -------------------------------------------------------


def test_never_loses_a_value_over_a_random_merge_sequence() -> None:
    """The formal statement of the contract, over generated sequences.

    Every value ever written must be findable in the final document -- at top
    level or relocated into a history list.
    """
    rng = random.Random(20260818)
    doc: dict = {}
    written: set[str] = set()

    for i in range(60):
        update = {
            "date": {
                "value": f"20{10 + i % 15}-03-09T12:00:00",
                "tier": rng.choice([0, 10, 20, 30, 40]),
                "source": rng.choice(["exif_original", "external_sidecar", "filename_pattern"]),
            },
            "descriptor": {
                "value": rng.choice(["", "beach-trip", "emma-birthday"]),
                "tier": rng.choice([0, 20, 30]),
                "source": rng.choice(["none", "ai_subject", "human_filename"]),
            },
            "sources": [{"path": f"/src/{i}.jpg", "folder": f"folder-{i % 4}",
                         "first_seen": T0, "last_seen": T1}],
            "albums": [{"archive_id": f"A{i % 3}", "folder": f"album-{i % 3}",
                        "title": f"Album {i % 3}"}],
            "people": [{"name": rng.choice(["Emma", "Sam", "Judy"])}],
            "exif": {"Make": rng.choice(["Canon", "Nikon"]), f"Tag{i % 5}": i},
            "provenance": [{"kind": "takeout_media_json", "digest": f"D{i % 7}",
                            "raw": {"title": f"t{i}.jpg", "imageViews": str(i)}}],
        }
        written |= _leaves(update)
        doc = merge(doc, update, observed_at=T1)

    final = _leaves(doc)
    missing = written - final
    assert not missing, f"{len(missing)} values lost: {sorted(missing)[:10]}"


def test_merging_the_same_update_twice_is_byte_identical() -> None:
    """Idempotence. Without it, 'append-only' means 'grows on every run'."""
    update = {
        "date": {"value": "2015-03-09T12:56:32", "tier": 30, "source": "external_sidecar"},
        "sources": [{"path": "/a.jpg", "folder": "d", "first_seen": T0, "last_seen": T0}],
        "albums": [{"archive_id": "A1", "folder": "d", "title": "D"}],
        "provenance": [{"kind": "takeout_media_json", "digest": "D1", "raw": {"title": "a.jpg"}}],
        "exif": {"Make": "Canon"},
    }
    once = merge({}, update, observed_at=T0)
    twice = merge(once, update, observed_at=T1)
    assert json.dumps(twice, sort_keys=True) == json.dumps(once, sort_keys=True)


def test_merge_never_shrinks_the_document() -> None:
    rng = random.Random(7)
    doc = {}
    for i in range(25):
        before = len(json.dumps(doc))
        doc = merge(doc, {"exif": {f"K{i}": rng.randint(0, 99)}}, observed_at=T1)
        assert len(json.dumps(doc)) >= before


# --- the bug this project exists to fix ----------------------------------


def test_an_empty_list_does_not_erase_the_existing_one() -> None:
    """The live data-loss path in the old _deep_merge, pinned.

    Merging {"people": []} over two recorded names discarded both. Any caller
    passing a partial list triggered it.
    """
    base = merge({}, {"people": [{"name": "Judy"}, {"name": "Pete"}]}, observed_at=T0)
    after = merge(base, {"people": []}, observed_at=T1)
    assert {p["name"] for p in after["people"]} == {"Judy", "Pete"}


def test_a_partial_list_adds_without_removing() -> None:
    base = merge({}, {"sources": [{"path": "/a.jpg"}, {"path": "/b.jpg"}]}, observed_at=T0)
    after = merge(base, {"sources": [{"path": "/c.jpg"}]}, observed_at=T1)
    assert {s["path"] for s in after["sources"]} == {"/a.jpg", "/b.jpg", "/c.jpg"}


# --- tiered scalars ------------------------------------------------------


def test_a_higher_tier_wins_and_demotes_the_incumbent() -> None:
    base = merge({}, {"date": {"value": "2019-07-04", "tier": 10, "source": "filename_pattern"}},
                 observed_at=T0)
    after = merge(base, {"date": {"value": "2015-03-09", "tier": 30, "source": "external_sidecar"}},
                  observed_at=T1)
    assert after["date"]["value"] == "2015-03-09"
    assert after["date"]["tier"] == 30
    assert any(h["value"] == "2019-07-04" for h in after["date"]["history"])


def test_a_lower_tier_loses_but_is_still_recorded() -> None:
    """A rejected observation is data too."""
    base = merge({}, {"date": {"value": "2015-03-09", "tier": 30, "source": "external_sidecar"}},
                 observed_at=T0)
    after = merge(base, {"date": {"value": "2019-07-04", "tier": 10, "source": "filename_pattern"}},
                  observed_at=T1)
    assert after["date"]["value"] == "2015-03-09"
    assert any(h["value"] == "2019-07-04" for h in after["date"]["history"])


def test_an_equal_tier_with_the_same_value_adds_no_history() -> None:
    """This is what keeps a repeated run from growing the file."""
    block = {"date": {"value": "2015-03-09", "tier": 30, "source": "external_sidecar"}}
    base = merge({}, block, observed_at=T0)
    after = merge(base, block, observed_at=T1)
    assert after["date"]["history"] == []


# --- keyed lists ---------------------------------------------------------


def test_re_observing_a_source_updates_last_seen_only() -> None:
    base = merge({}, {"sources": [{"path": "/a.jpg", "folder": "d",
                                   "first_seen": T0, "last_seen": T0}]}, observed_at=T0)
    after = merge(base, {"sources": [{"path": "/a.jpg", "folder": "d",
                                      "first_seen": T1, "last_seen": T1}]}, observed_at=T1)
    assert len(after["sources"]) == 1
    assert after["sources"][0]["first_seen"] == T0   # written once, never moved
    assert after["sources"][0]["last_seen"] == T1


def test_albums_key_on_archive_and_folder() -> None:
    """The same folder name in two archives is two albums."""
    base = merge({}, {"albums": [{"archive_id": "A1", "folder": "2015", "title": "X"}]}, observed_at=T0)
    after = merge(base, {"albums": [{"archive_id": "A2", "folder": "2015", "title": "Y"}]}, observed_at=T1)
    assert len(after["albums"]) == 2


def test_provenance_keys_on_digest() -> None:
    doc = {"provenance": [{"kind": "takeout_media_json", "digest": "D1", "raw": {"a": 1}}]}
    base = merge({}, doc, observed_at=T0)
    after = merge(base, doc, observed_at=T1)
    assert len(after["provenance"]) == 1


def test_raw_provenance_is_stored_verbatim() -> None:
    raw = {"title": "x.jpg", "imageViews": "12", "height": "1600", "unknownFutureField": [1, 2]}
    doc = merge({}, {"provenance": [{"kind": "takeout_media_json", "digest": "D1", "raw": raw}]},
                observed_at=T0)
    assert doc["provenance"][0]["raw"] == raw


# --- flat maps -----------------------------------------------------------


def test_a_changed_exif_value_moves_the_old_one_to_history() -> None:
    base = merge({}, {"exif": {"Orientation": 1.0, "Make": "Canon"}}, observed_at=T0)
    after = merge(base, {"exif": {"Orientation": 6.0}}, observed_at=T1)
    assert after["exif"]["Orientation"] == 6.0
    assert after["exif"]["Make"] == "Canon"          # untouched sibling
    assert any(h["key"] == "Orientation" and h["value"] == 1.0 for h in after["exif_history"])


# --- unknown keys --------------------------------------------------------


def test_a_hand_written_key_survives_every_merge() -> None:
    base = merge({}, {"my_note": "keep this", "date": {"value": "2015-03-09", "tier": 30}},
                 observed_at=T0)
    after = merge(base, {"date": {"value": "2019-07-04", "tier": 40}}, observed_at=T1)
    assert after["my_note"] == "keep this"


def test_a_conflicting_unknown_key_keeps_the_incumbent_and_records_the_other() -> None:
    base = merge({}, {"note": "mine"}, observed_at=T0)
    after = merge(base, {"note": "theirs"}, observed_at=T1)
    assert after["note"] == "mine"
    assert any(c["key"] == "note" and c["value"] == "theirs" for c in after["conflicts"])


# --- migration -----------------------------------------------------------


V1 = {
    "schema_version": 1,
    "identity": {"sha256_b64url": "D" * 43, "size": 100, "ext": "jpg"},
    "sources": [{"path": "/a.jpg", "first_seen": T0, "last_seen": T0}],
    "date": {"value": "2015-03-09T12:56:32", "tier": 30, "source": "external_sidecar"},
    "descriptor": {"value": "beach", "tier": 30, "source": "human_filename"},
    "exif": {"Make": "Canon"},
    "takeout": {"archive": "t.zip", "archive_id": "A1", "member": "T/a.jpg",
                "album": "2015", "title": "a.jpg", "people": [], "favorited": False},
    "hand_written": "do not lose me",
}


def test_migration_preserves_every_v1_value() -> None:
    out = migrate(copy.deepcopy(V1))
    assert out["schema_version"] == SCHEMA_VERSION
    lost = _leaves(V1) - _leaves(out)
    assert not lost, f"migration lost: {sorted(lost)}"


def test_migration_moves_the_v1_takeout_block_into_provenance() -> None:
    out = migrate(copy.deepcopy(V1))
    kinds = {p.get("kind") for p in out["provenance"]}
    assert "imageharbor_v1_takeout_block" in kinds


def test_migration_is_idempotent() -> None:
    once = migrate(copy.deepcopy(V1))
    twice = migrate(copy.deepcopy(once))
    assert json.dumps(twice, sort_keys=True) == json.dumps(once, sort_keys=True)


def test_merge_migrates_a_v1_base_automatically() -> None:
    out = merge(copy.deepcopy(V1), {"exif": {"Model": "5D"}}, observed_at=T1)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["hand_written"] == "do not lose me"


# --- totality ------------------------------------------------------------


@pytest.mark.parametrize(
    "base, updates",
    [
        ({}, {}),
        (None, {"exif": {"a": 1}}),
        ({"date": "not a dict"}, {"date": {"value": "x", "tier": 1}}),
        ({"sources": "not a list"}, {"sources": [{"path": "/a"}]}),
        ({"exif": []}, {"exif": {"a": 1}}),
        ({"provenance": [None, 3]}, {"provenance": [{"digest": "D"}]}),
    ],
)
def test_merge_never_raises_on_malformed_input(base, updates) -> None:
    out = merge(base, updates, observed_at=T0)
    assert isinstance(out, dict)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sidecar_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'imageharbor.sidecar_schema'`.

- [ ] **Step 3: Write `sidecar_schema.py`**

Create `imageharbor/sidecar_schema.py`:

```python
"""The sidecar merge policy.

One rule governs this module: **a sidecar may gain information and may never
lose any.** A superseded value is relocated into a history list; it is never
overwritten and never dropped.

Pure by design. The policy is the part most likely to be wrong, so it takes no
filesystem and imports nothing from the rest of the package -- the same split
that made `takeout/metadata.py` and `takeout/pairing.py` exhaustively testable.
`sidecar.py` owns reading, atomic writing, and corrupt-file quarantine.

Two properties make the rule usable rather than merely true:

* **Idempotence.** Re-running any pass must leave a sidecar byte-identical.
  Every history list therefore dedupes on the *value*, never on a timestamp --
  a timestamp inside the dedup key would make every re-run append forever.
* **Totality.** `merge` never raises. A sidecar is a projection of the catalog,
  and a malformed one must degrade to "less is recorded", never fail an image
  that is already copied, verified, and catalogued.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 2

# Blocks carrying a quality tier, governed by `tiers.is_upgrade`'s logic:
# higher wins, equal is a no-op.
TIERED_BLOCKS: tuple[str, ...] = ("date", "descriptor")

# Blocks that replace on change but keep the old one.
VERSIONED_BLOCKS: tuple[str, ...] = ("classification",)

# Append-only lists and the fields that identify an entry within them.
KEYED_LISTS: dict[str, tuple[str, ...]] = {
    "sources": ("path",),
    "albums": ("archive_id", "folder"),
    "people": ("name",),
    "provenance": ("digest",),
}

# Key-by-key maps. A changed value is recorded in `<name>_history`.
FLAT_MAPS: tuple[str, ...] = ("identity", "exif")

# Fields excluded from a history entry's identity, so re-observing the same
# fact does not append a near-duplicate differing only in when it was seen.
_TIMESTAMP_FIELDS = frozenset({"observed_at", "superseded_at", "first_seen", "last_seen"})


def _core(block: Any) -> dict[str, Any]:
    """A block stripped of timestamps -- its identity for dedup purposes."""
    if not isinstance(block, dict):
        return {}
    return {k: v for k, v in block.items() if k not in _TIMESTAMP_FIELDS and k != "history"}


def _already_recorded(history: list, core: dict) -> bool:
    return any(_core(h) == core for h in history if isinstance(h, dict))


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _merge_tiered(base: Any, new: Any, observed_at: str) -> dict[str, Any]:
    """Higher tier wins; the loser is recorded rather than discarded.

    A *rejected* observation is recorded too. It is evidence about the photo
    even though it did not win, and discarding it would break the rule for the
    sake of a smaller file.
    """
    new = _as_dict(new)
    base = _as_dict(base)
    if not base:
        return {**_core(new), "observed_at": observed_at, "history": []}

    history = [h for h in _as_list(base.get("history")) if isinstance(h, dict)]
    old_core, new_core = _core(base), _core(new)

    if old_core == new_core:
        return base  # nothing observed that is not already on record

    old_tier = base.get("tier") or 0
    new_tier = new.get("tier") or 0

    if new_tier > old_tier:
        demoted = {**old_core, "superseded_at": observed_at}
        if not _already_recorded(history, old_core):
            history.append(demoted)
        return {**new_core, "observed_at": observed_at, "history": history}

    if not _already_recorded(history, new_core):
        history.append({**new_core, "observed_at": observed_at, "rejected": True})
    return {**base, "history": history}


def _merge_versioned(base: Any, new: Any, observed_at: str) -> dict[str, Any]:
    """No tiers; any change records the previous block."""
    new, base = _as_dict(new), _as_dict(base)
    if not base:
        return {**_core(new), "observed_at": observed_at, "history": []}
    history = [h for h in _as_list(base.get("history")) if isinstance(h, dict)]
    old_core, new_core = _core(base), _core(new)
    if old_core == new_core:
        return base
    if not _already_recorded(history, old_core):
        history.append({**old_core, "superseded_at": observed_at})
    return {**new_core, "observed_at": observed_at, "history": history}


def _merge_keyed_list(base: Any, new: Any, keys: tuple[str, ...]) -> list:
    """Append-only union. An existing entry gains fields; it never loses any."""
    out = [dict(e) for e in _as_list(base) if isinstance(e, dict)]
    index = {tuple(e.get(k) for k in keys): i for i, e in enumerate(out)}

    for entry in _as_list(new):
        if not isinstance(entry, dict):
            continue
        identity = tuple(entry.get(k) for k in keys)
        if identity in index:
            current = out[index[identity]]
            for field, value in entry.items():
                # Fill gaps and refresh only last_seen. Never overwrite a
                # recorded value -- that is what "never lose" means here.
                if field == "last_seen" or field not in current or current[field] in (None, "", [], {}):
                    current[field] = value
        else:
            index[identity] = len(out)
            out.append(dict(entry))
    return out


def _merge_flat_map(base: Any, new: Any, history: list, observed_at: str) -> dict:
    out = dict(_as_dict(base))
    for key, value in _as_dict(new).items():
        if key not in out:
            out[key] = value
        elif out[key] != value:
            core = {"key": key, "value": out[key]}
            if not _already_recorded(history, core):
                history.append({**core, "superseded_at": observed_at})
            out[key] = value
    return out


def migrate(doc: Any) -> dict[str, Any]:
    """Upgrade a v1 sidecar to v2. Itself lossless, and idempotent.

    The v1 `takeout` block becomes a provenance entry. Its `kind` says
    plainly that it is a reconstruction: the original Google document was not
    retained by v1, so this is the best record that exists for such a file.
    """
    doc = dict(_as_dict(doc))
    if doc.get("schema_version") == SCHEMA_VERSION:
        return doc

    for name in (*TIERED_BLOCKS, *VERSIONED_BLOCKS):
        block = doc.get(name)
        if isinstance(block, dict) and "history" not in block:
            doc[name] = {**block, "history": []}

    legacy = doc.pop("takeout", None)
    if isinstance(legacy, dict):
        provenance = [p for p in _as_list(doc.get("provenance")) if isinstance(p, dict)]
        digest = f"v1:{legacy.get('archive_id', '')}:{legacy.get('member', '')}"
        if not any(p.get("digest") == digest for p in provenance):
            provenance.append({
                "kind": "imageharbor_v1_takeout_block",
                "digest": digest,
                "archive_id": legacy.get("archive_id"),
                "archive": legacy.get("archive"),
                "member": legacy.get("member"),
                "raw": legacy,
            })
        doc["provenance"] = provenance

        folder = legacy.get("album")
        if folder:
            albums = [a for a in _as_list(doc.get("albums")) if isinstance(a, dict)]
            identity = (legacy.get("archive_id"), folder)
            if not any((a.get("archive_id"), a.get("folder")) == identity for a in albums):
                albums.append({"archive_id": legacy.get("archive_id"),
                               "folder": folder, "title": None})
            doc["albums"] = albums

    doc["schema_version"] = SCHEMA_VERSION
    return doc


def merge(base: Any, updates: Any, *, observed_at: str) -> dict[str, Any]:
    """Merge *updates* into *base*, losing nothing.

    Returns a new document containing every value present in either argument.
    Never raises.
    """
    out = migrate(base)
    updates = _as_dict(updates)

    exif_history = [h for h in _as_list(out.get("exif_history")) if isinstance(h, dict)]
    conflicts = [c for c in _as_list(out.get("conflicts")) if isinstance(c, dict)]

    for key, value in updates.items():
        if key == "schema_version":
            continue
        if key in TIERED_BLOCKS:
            out[key] = _merge_tiered(out.get(key), value, observed_at)
        elif key in VERSIONED_BLOCKS:
            out[key] = _merge_versioned(out.get(key), value, observed_at)
        elif key in KEYED_LISTS:
            out[key] = _merge_keyed_list(out.get(key), value, KEYED_LISTS[key])
        elif key in FLAT_MAPS:
            history = exif_history if key == "exif" else []
            merged = _merge_flat_map(out.get(key), value, history, observed_at)
            out[key] = merged
        elif key not in out:
            out[key] = value
        elif out[key] != value:
            # An unknown key already holds something else. The incumbent wins
            # -- it may be a hand edit -- and the newcomer is recorded rather
            # than dropped. No ImageHarbor pass writes unknown keys, so this
            # is reached only by a hand edit colliding with a future field.
            core = {"key": key, "value": value}
            if not _already_recorded(conflicts, core):
                conflicts.append({**core, "observed_at": observed_at})

    if exif_history:
        out["exif_history"] = exif_history
    if conflicts:
        out["conflicts"] = conflicts
    out["schema_version"] = SCHEMA_VERSION
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sidecar_schema.py -q`
Expected: PASS.

If `test_never_loses_a_value_over_a_random_merge_sequence` fails, read which values are missing before changing anything — a genuine loss is the bug this module exists to prevent, and the fix belongs in the merge policy, never in the test.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/sidecar_schema.py tests/test_sidecar_schema.py
git commit -m "feat: append-only sidecar merge policy"
```

---

### Task 2: `sidecar.py` delegates policy and quarantines corrupt files

**Files:**
- Modify: `imageharbor/sidecar.py`
- Test: `tests/test_sidecar.py`

**Interfaces:**
- Consumes: `sidecar_schema.merge`, `sidecar_schema.SCHEMA_VERSION` (Task 1).
- Produces: unchanged public surface — `sidecar_path_for`, `read_sidecar`, `merge_sidecar(organized_path, updates) -> Path`. `_deep_merge` is removed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sidecar.py`:

```python
"""Tests for sidecar I/O: atomicity and corrupt-file handling."""

from __future__ import annotations

import json
from pathlib import Path

from imageharbor.sidecar import merge_sidecar, read_sidecar, sidecar_path_for


def _img(tmp_path: Path) -> Path:
    p = tmp_path / "2015-03-09_abc.jpg"
    p.write_bytes(b"x")
    return p


def test_merge_creates_then_accretes(tmp_path: Path) -> None:
    img = _img(tmp_path)
    merge_sidecar(img, {"exif": {"Make": "Canon"}})
    merge_sidecar(img, {"exif": {"Model": "5D"}})
    doc = read_sidecar(img)
    assert doc["exif"] == {"Make": "Canon", "Model": "5D"}


def test_a_corrupt_sidecar_is_quarantined_not_overwritten(tmp_path: Path) -> None:
    """Treating a corrupt sidecar as empty would destroy data under the new rule.

    The old behavior logged and returned {}, so the next merge silently wrote a
    fresh document over whatever the unreadable bytes contained.
    """
    img = _img(tmp_path)
    path = sidecar_path_for(img)
    path.write_text('{"exif": {"Make": "Canon"} TRUNCATED', encoding="utf-8")

    merge_sidecar(img, {"exif": {"Model": "5D"}})

    quarantined = list(tmp_path.glob("*.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "TRUNCATED" in quarantined[0].read_text(encoding="utf-8")
    assert read_sidecar(img)["exif"] == {"Model": "5D"}


def test_a_hand_edit_survives_a_merge(tmp_path: Path) -> None:
    img = _img(tmp_path)
    merge_sidecar(img, {"exif": {"Make": "Canon"}})
    doc = read_sidecar(img)
    doc["my_note"] = "grandma's camera"
    sidecar_path_for(img).write_text(json.dumps(doc), encoding="utf-8")

    merge_sidecar(img, {"exif": {"Model": "5D"}})
    assert read_sidecar(img)["my_note"] == "grandma's camera"


def test_repeated_merge_is_byte_identical(tmp_path: Path) -> None:
    img = _img(tmp_path)
    update = {"sources": [{"path": "/a.jpg", "folder": "d", "first_seen": "T", "last_seen": "T"}]}
    merge_sidecar(img, update)
    first = sidecar_path_for(img).read_bytes()
    merge_sidecar(img, update)
    assert sidecar_path_for(img).read_bytes() == first


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    img = _img(tmp_path)
    merge_sidecar(img, {"exif": {"Make": "Canon"}})
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sidecar.py -q`
Expected: FAIL — `test_a_corrupt_sidecar_is_quarantined_not_overwritten` finds no quarantine file; `test_repeated_merge_is_byte_identical` may fail on the `last_seen` rewrite.

- [ ] **Step 3: Rewrite the merge path**

In `imageharbor/sidecar.py`: delete `_deep_merge` entirely, change `SIDECAR_SCHEMA_VERSION` to import from the schema module, and replace `read_sidecar`/`merge_sidecar`:

```python
from datetime import datetime, timezone

from .sidecar_schema import SCHEMA_VERSION as SIDECAR_SCHEMA_VERSION
from .sidecar_schema import merge as merge_documents


def _quarantine(path: Path, reason: str) -> None:
    """Move an unreadable sidecar aside instead of writing over it.

    Returning {} for a corrupt file -- the previous behavior -- meant the next
    merge silently replaced whatever those bytes held. Under the never-lose
    rule that is the one unacceptable outcome, so the bytes are preserved
    under a timestamped name and a fresh sidecar is built beside them.
    """
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        path.replace(target)
        logger.warning("Unreadable sidecar %s (%s); preserved as %s", path, reason, target.name)
    except OSError as exc:
        logger.error("Could not quarantine %s (%s); leaving it untouched", path, exc)


def read_sidecar(organized_path: Path) -> dict[str, Any]:
    """Return the existing sidecar contents, or ``{}`` if absent.

    An unreadable sidecar is quarantined (see :func:`_quarantine`) and reported
    as empty, so the caller proceeds with a fresh document while the original
    bytes survive on disk.
    """
    path = sidecar_path_for(organized_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        _quarantine(path, str(exc))
        return {}
    if not isinstance(data, dict):
        _quarantine(path, "top-level value is not an object")
        return {}
    return data


def merge_sidecar(organized_path: Path, updates: dict[str, Any]) -> Path:
    """Merge *updates* into the sidecar for *organized_path* and write it back.

    Merge policy lives in :mod:`imageharbor.sidecar_schema`; this function owns
    only reading, the atomic write (temp file then ``os.replace``), and the
    quarantine of an unreadable file.
    """
    path = sidecar_path_for(organized_path)
    observed_at = datetime.now(tz=timezone.utc).isoformat()
    merged = merge_documents(read_sidecar(organized_path), updates, observed_at=observed_at)

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

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS. Existing sidecar assertions in `tests/test_pipeline.py`, `tests/test_monotonicity.py`, and `tests/test_takeout_ingest.py` may need updating for the v2 shape — update assertions to match the new structure, but do **not** weaken any assertion about a value being present.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/sidecar.py tests/test_sidecar.py
git commit -m "feat: sidecar I/O delegates policy and quarantines corrupt files"
```

---

### Task 3: `pipeline.py` records the source folder

**Files:**
- Modify: `imageharbor/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:** Consumes Task 2. Produces `sources[].folder` in every sidecar the facts pass writes.

- [ ] **Step 1: Write the failing test**

```python
def test_sidecar_records_the_source_folder(tmp_path: Path, organized_dir: Path, catalog: Catalog) -> None:
    """The directory a photo was found in is a fact worth keeping.

    It is the only surviving trace of how the owner had organized things
    before ImageHarbor re-organized by date.
    """
    src = tmp_path / "src" / "Hawaii 2019"
    src.mkdir(parents=True)
    photo = _make_jpeg(src / "beach.jpg")

    result = Pipeline(tmp_path / "src", organized_dir, catalog, write_sidecars=True).process_file(photo)

    from imageharbor.sidecar import read_sidecar
    entry = read_sidecar(result.organized_path)["sources"][0]
    assert entry["folder"] == "Hawaii 2019"
    assert entry["path"] == str(photo)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_sidecar_records_the_source_folder -q`
Expected: FAIL with `KeyError: 'folder'`.

- [ ] **Step 3: Add the folder**

In `pipeline.py`, both places that build the `sources` list for a sidecar (`_write_sidecar` and the duplicate-upgrade re-merge) currently emit `{"path", "first_seen", "last_seen"}`. Add the folder, derived from the recorded source path:

```python
def _source_entry(row) -> dict[str, Any]:
    """A sidecar `sources[]` entry from a catalog `sources` row.

    `folder` is the immediate parent of the source path -- for a Takeout
    member that is its album directory, and for an ordinary `process` run it
    is whatever folder the owner had the photo in. Both are facts about the
    photo's history that the date-derived tree would otherwise erase.
    """
    raw = row["source_path"]
    member = raw.rsplit("!", 1)[-1]          # Takeout labels are <zip>!<member>
    folder = member.replace("\\", "/").rsplit("/", 2)[-2] if "/" in member.replace("\\", "/") else ""
    return {
        "path": raw,
        "folder": folder,
        "first_seen": row["first_seen_at"],
        "last_seen": row["last_seen_at"],
    }
```

Use it in both list comprehensions.

- [ ] **Step 4: Run the suite and commit**

Run: `uv run pytest -q` — expected PASS.

```bash
git add imageharbor/pipeline.py tests/test_pipeline.py
git commit -m "feat: record the source folder in the sidecar"
```

---

### Task 4: `enrich.py` records the model version

**Files:**
- Modify: `imageharbor/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:** Produces `classification.model_version` so a reclassification's history says which model produced each answer.

- [ ] **Step 1: Write the failing test**

```python
def test_classification_records_the_model_version(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """A classification without its model version cannot be re-evaluated later.

    When a better model arrives, history has to say which answer came from
    which model, or there is no way to tell an improvement from a regression.
    """
    from imageharbor.enrich import enrich_library
    from imageharbor.pipeline import Pipeline
    from imageharbor.sidecar import read_sidecar

    src = tmp_path / "src"
    src.mkdir()
    photo = _make_jpeg(src / "beach trip.jpg")
    result = Pipeline(src, organized_dir, catalog, write_sidecars=True).process_file(photo)

    enrich_library(organized_dir, catalog, classifier=StubClassifier(), write_sidecars=True)

    block = read_sidecar(result.organized_path)["classification"]
    assert block["model_version"]        # non-empty
    assert block["history"] == []        # first observation, nothing superseded
```

`_make_jpeg`, `organized_dir`, `catalog`, and `StubClassifier` already exist in
`tests/test_enrich.py`; match its current fixture and import style rather than
introducing new helpers. Check `enrich_library`'s real signature before writing
the call -- it is the one interface here this plan did not re-read.

- [ ] **Step 2: Run to verify it fails, then add the field**

In `enrich.py`'s `merge_sidecar` call, add `"model_version": content.model_version` to the `classification` dict. The block already carries `code`, `label`, and `folder_path`.

- [ ] **Step 3: Run the suite and commit**

```bash
git add imageharbor/enrich.py tests/test_enrich.py
git commit -m "feat: record the classifier's model version in the sidecar"
```

---

### Task 5: Takeout captures raw documents and real album titles

**Files:**
- Modify: `imageharbor/takeout/metadata.py`, `imageharbor/takeout/ingest.py`
- Test: `tests/test_takeout_metadata.py`, `tests/test_takeout_ingest.py`

**Interfaces:**
- Produces: `AlbumMetadata(title, description, access, date)`; `provenance[].raw` holding Google's document verbatim; `albums[]` carrying the title from `Albums.json`.
- **This task makes `parse_album_metadata` reachable for the first time** — it exists today and nothing calls it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_takeout_metadata.py`:

```python
def test_album_metadata_parses_access_and_date() -> None:
    raw = json.dumps({
        "title": "Hangout: Conrad Storz ● Herbie (Tony) Hughes",
        "access": "protected",
        "date": {"timestampSeconds": "1524674607", "formatted": "…"},
    }).encode()
    album = parse_album_metadata(raw)
    assert album.title.startswith("Hangout:")
    assert album.access == "protected"
    assert album.date == datetime(2018, 4, 25, 16, 43, 27)
```

In `tests/test_takeout_ingest.py`:

```python
def test_sidecar_preserves_googles_document_verbatim(dirs, catalog: Catalog) -> None:
    """Fields nobody modelled -- imageViews, height, width -- survive here.

    They come back without being parsed, and so will anything Google adds
    later, which is the whole reason the raw document is kept.
    """
    archives, dest = dirs
    payload = {
        "title": "a.jpg",
        "imageViews": "12",
        "height": "2432", "width": "4320",
        "photoTakenTime": {"timestampSeconds": "1425905792"},
        "someFieldFromTheFuture": {"nested": [1, 2, 3]},
    }
    _zip(archives / "t.zip", {f"{D}/a.jpg": _jpeg(60),
                              f"{D}/a.jpg.json": json.dumps(payload).encode()})

    ingest_archives(archives, dest, catalog, write_sidecars=True)

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    entry = next(p for p in read_sidecar(organized)["provenance"]
                 if p["kind"] == "takeout_media_json")
    assert entry["raw"] == payload


def test_sidecar_records_the_real_album_title(dirs, catalog: Catalog) -> None:
    """The directory name is not the album name."""
    archives, dest = dirs
    _zip(archives / "t.zip", {
        f"{D}/a.jpg": _jpeg(61),
        f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
        f"{D}/Albums.json": json.dumps({"title": "Hangout: Emma ● Sam",
                                        "access": "protected"}).encode(),
    })
    ingest_archives(archives, dest, catalog, write_sidecars=True)

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    album = read_sidecar(organized)["albums"][0]
    assert album["title"] == "Hangout: Emma ● Sam"
    assert album["access"] == "protected"
    assert album["folder"] == D.rsplit("/", 1)[-1]


def test_a_photo_in_two_archives_accumulates_both_albums(dirs, catalog: Catalog) -> None:
    """Duplicates stop being waste and become context."""
    archives, dest = dirs
    img = _jpeg(62)
    _zip(archives / "one.zip", {"Takeout/A/Album One/a.jpg": img,
                                "Takeout/A/Album One/Albums.json": json.dumps({"title": "One"}).encode()})
    _zip(archives / "two.zip", {"Takeout/A/Album Two/b.jpg": img,
                                "Takeout/A/Album Two/Albums.json": json.dumps({"title": "Two"}).encode()})

    stats = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert stats.ingested == 1
    assert stats.duplicates == 1

    from imageharbor.sidecar import read_sidecar
    organized = next(dest.rglob("*.jpg"))
    titles = {a["title"] for a in read_sidecar(organized)["albums"]}
    assert titles == {"One", "Two"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_takeout_metadata.py tests/test_takeout_ingest.py -q`
Expected: FAIL — `AlbumMetadata` has no `access`; `provenance` key absent; `albums[0]["title"]` is `None`.

- [ ] **Step 3: Extend `parse_album_metadata`**

In `takeout/metadata.py`, add `access: str | None = None` and `date: datetime | None = None` to `AlbumMetadata`, and populate them in `parse_album_metadata` using the existing `_text` and `_timestamp` helpers. Keep the never-raise contract.

- [ ] **Step 4: Capture raw and albums in `ingest.py`**

In `_ingest_image`, the sidecar merge currently writes a flat `takeout` block. Replace it so it writes the v2 shape:

- read the sidecar member's **bytes** (not just the parsed metadata) and compute their digest with `hashing.compute_sha256_b64url` over the bytes — add a small `_digest_bytes` helper rather than writing to a temp file;
- emit `provenance: [{kind, archive_id, archive, member, observed_at, digest, raw}]` where `raw` is `json.loads` of those bytes (falling back to omitting `raw` if it does not parse — the never-raise discipline);
- emit `albums: [{archive_id, folder, title, access, date}]`, where `folder` is the member's parent directory and the title comes from that directory's `Albums.json` if one exists in the batch;
- emit `people: [{"name": n, "source": "google_photos_people"} for n in meta.people]`.

Add an `_album_titles: dict[tuple[str, str], AlbumMetadata]` built during the survey, keyed by `(archive_id, folder)`, so the lookup is a dict hit rather than a zip read per photo.

**The duplicate branch must merge too.** A duplicate resolves its organized path from the catalog (already implemented); it must receive the same `provenance`/`albums` merge, because accumulating context from a second archive is exactly what makes duplicates valuable rather than wasted.

- [ ] **Step 5: Run the suite and commit**

Run: `uv run pytest -q` — expected PASS.

```bash
git add imageharbor/takeout/metadata.py imageharbor/takeout/ingest.py tests/test_takeout_metadata.py tests/test_takeout_ingest.py
git commit -m "feat: preserve Google's document verbatim and record real album titles"
```

---

### Task 6: The provenance room

**Files:**
- Create: `imageharbor/takeout/provenance.py`
- Modify: `imageharbor/takeout/ingest.py`, `.gitignore`
- Test: `tests/test_takeout_provenance.py`

**Interfaces:**
- Produces:
  - `ROOM_NAME = ".takeout-provenance"`
  - `preserve(organized_dir, identity, zf, members, *, orphaned) -> int` — writes every non-media member verbatim, returns how many were newly written
  - `manifest_path(organized_dir, archive_id) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_takeout_provenance.py` covering:

- every non-media member is written verbatim, byte-for-byte identical to the archive member;
- `archive_browser.html` is preserved (the uncurated rule — no judgement about which unknown file matters);
- `Albums.json` lands under `albums/<folder>/`;
- a media JSON with no media member in the batch lands under `orphaned/`;
- `manifest.json` lists every preserved document with its digest;
- **re-preserving the same archive writes nothing new** (assert file mtimes are unchanged, or count writes via a monkeypatched write helper);
- a write failure is logged and does not raise.

- [ ] **Step 2: Run to verify they fail**

Expected: `ModuleNotFoundError: No module named 'imageharbor.takeout.provenance'`.

- [ ] **Step 3: Write `provenance.py`**

The module's docstring must state the uncurated rule and why:

```python
"""Preserve every non-media archive member verbatim, under the organized root.

The rule is deliberately uncurated: anything that is not an image or a video
is kept. Deciding which unknown file is worth keeping is exactly where "never
lose" degrades into "lose the thing nobody thought about" -- so Google's
169 KB HTML viewer is preserved for the same reason its face-tag file is.

This is where data that cannot be attached to a photo lives. The Picasa face
tags name 73 people across 1,496 entries and carry no photo reference at all;
they are kept intact so that a future export supplying a join key can attach
them retroactively.
"""
```

Layout, keyed by `archive_id` so a renamed or moved archive resolves to the same room:

```
<organized_dir>/.takeout-provenance/<archive_id>/
    manifest.json
    albums/<folder>/Albums.json
    orphaned/<member basename>
    <every other non-media member, at its member path>
```

`manifest.json` records `{archive, archive_id, size, ingested_at, documents: [{member, digest, stored_as}]}`. A document whose digest is already in the manifest is skipped, which is what makes re-preserving a no-op.

- [ ] **Step 4: Call it from the survey**

In `ingest._survey`, after enumerating an archive's members, call `provenance.preserve(...)` with the non-media members and the set of media-JSON members whose media sibling is absent from the whole batch. Wrap the call so a failure is logged and the ingest continues — the archive itself is the original and remains intact.

Skip entirely when `dry_run` is set.

- [ ] **Step 5: Update `.gitignore`, run the suite, commit**

Add `.takeout-provenance/` beside `.takeout-staging/`.

```bash
git add imageharbor/takeout/provenance.py imageharbor/takeout/ingest.py .gitignore tests/test_takeout_provenance.py
git commit -m "feat: preserve non-media archive members in a provenance room"
```

---

### Task 7: Sidecars become the default

**Files:**
- Modify: `imageharbor/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:** `--sidecar/--no-sidecar` becomes `default=True` at all four command sites (`process`, `enrich`, `watch`, `takeout ingest`). No flag is renamed.

- [ ] **Step 1: Write the failing tests**

```python
def test_process_writes_a_sidecar_by_default(tmp_path: Path) -> None:
    """The flag flip, stated as behavior rather than as a default value."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"ÿØÿà" + b" " * 16 + b"ÿÙ")
    dest = tmp_path / "organized"

    result = CliRunner().invoke(main, ["process", "--source", str(src), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert [p for p in dest.rglob("*.json")], "no sidecar written without a flag"


def test_no_sidecar_still_suppresses(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "beach.jpg").write_bytes(b"ÿØÿà" + b"" * 16 + b"ÿÙ")
    dest = tmp_path / "organized"

    result = CliRunner().invoke(
        main, ["process", "--source", str(src), "--dest", str(dest), "--no-sidecar"]
    )
    assert result.exit_code == 0, result.output
    assert [p for p in dest.rglob("*.json")] == []


def test_takeout_ingest_writes_a_sidecar_by_default(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    _takeout_zip(archives / "t.zip", {
        "Takeout/A/a.jpg": b"ÿØÿà" + b"" * 16 + b"ÿÙ",
        "Takeout/A/a.jpg.json": json.dumps(
            {"title": "a.jpg", "photoTakenTime": {"timestampSeconds": "1425905792"}}
        ).encode(),
    })

    result = CliRunner().invoke(
        main, ["takeout", "ingest", "--archives", str(archives), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    assert organized.with_suffix(".json").exists()
```

- [ ] **Step 2: Run to verify they fail, then flip the four defaults**

Each site reads `default=False, show_default=True`; change to `default=True`. Update each help string to name the opt-out:

```python
    help="Write a JSON sidecar alongside each organized image. Use --no-sidecar to suppress.",
```

- [ ] **Step 3: Run the suite**

Run: `uv run pytest -q`

Several existing CLI tests assert on output or file counts from runs that previously wrote no sidecars. Update them to the new default; do not pass `--no-sidecar` merely to keep an old assertion green — that would hide the behavior change from the suite.

- [ ] **Step 4: Commit**

```bash
git add imageharbor/cli.py tests/test_cli.py
git commit -m "feat: write sidecars by default"
```

---

### Task 8: `sidecar backfill`

**Files:**
- Create: `imageharbor/backfill.py`
- Modify: `imageharbor/cli.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Produces:
  - `@dataclass BackfillStats: cataloged: int; written: int; unchanged: int; failed: int` — `failed` covers both a row whose organized file cannot be resolved and a write error; there is deliberately no separate `missing` counter, since the operator's remedy is the same either way
  - `backfill_sidecars(organized_dir: Path, catalog: Catalog, *, dry_run: bool = False) -> BackfillStats`
  - CLI: `imageharbor sidecar backfill --dest DEST [--catalog PATH] [--dry-run]`

- [ ] **Step 1: Write the failing tests**

Cover:

- every cataloged row with a resolvable organized file gains a sidecar carrying `identity`, `sources` (with folder), `date` and `descriptor` with tiers, and a freshly-read `exif`;
- `provenance` is empty — **assert this explicitly**, because it is the honest limit of the verb and a future reader should find it pinned rather than discovered;
- backfill **merges** into an existing thin sidecar rather than skipping it, and reports it as `written`;
- running backfill twice leaves every sidecar byte-identical and reports `unchanged`;
- `--dry-run` writes nothing and still reports accurate counts;
- a row whose organized file is missing is counted in `failed` and does not stop the run.

- [ ] **Step 2: Run to verify they fail**

Expected: `ModuleNotFoundError: No module named 'imageharbor.backfill'`.

- [ ] **Step 3: Write `backfill.py`**

Iterate `catalog.iter_all()`. For each row, resolve the organized path with `relocate.resolve_organized_path` (so a moved file is found by digest rather than reported missing), read EXIF from the organized copy, build the same update dict the facts pass builds, and call `merge_sidecar`. Count `written` when the file changed and `unchanged` when the merge was a no-op — compare the bytes before and after rather than trusting the merge to report.

The module docstring must state the limit plainly:

```python
"""Rebuild sidecars for an already-organized library from the catalog.

What this can write is bounded by what the catalog holds: identity, sources,
date and descriptor with their tiers, plus a fresh EXIF read from the
organized copy. **Google Takeout metadata is not recoverable this way** --
`provenance[]` stays empty for backfilled files, because the original archive
documents were never stored. Recovering those means re-ingesting the archives.
"""
```

- [ ] **Step 4: Add the CLI group**

A `sidecar` Click group with a `backfill` subcommand, following the `catalog list`/`takeout ingest` precedent already in `cli.py`. Exit non-zero when `stats.failed` is non-zero.

- [ ] **Step 5: Run the suite and commit**

```bash
git add imageharbor/backfill.py imageharbor/cli.py tests/test_backfill.py
git commit -m "feat: imageharbor sidecar backfill"
```

---

### Task 9: Documentation and real-archive verification

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/superpowers/specs/2026-08-18-lossless-sidecar-design.md`

- [ ] **Step 1: Update `CLAUDE.md`**

- Add `sidecar_schema.py` to the module list, stating the never-lose rule and the idempotence property that makes it usable.
- Rewrite the `sidecar.py` bullet: policy has moved out; corrupt files are quarantined, not treated as empty.
- Add a critical invariant: **a sidecar may gain information and may never lose any**, with the note that history dedupes on value rather than timestamp so repeated runs stay byte-identical.
- Note that `--sidecar` is now the default across `process`, `enrich`, `watch`, and `takeout ingest`.
- Add `sidecar backfill` to the command table, with its limit stated.
- Add `takeout/provenance.py` and the uncurated preservation rule.

- [ ] **Step 2: Update `README.md`** with the `sidecar backfill` command and one sentence on default-on sidecars.

- [ ] **Step 3: Mark the spec implemented**, recording any deliberate departures the same way the Takeout spec does.

- [ ] **Step 4: Verify against the real archive**

```bash
S="C:/Users/Conrad/AppData/Local/Temp/claude/D--Users-Conrad-Documents-programming-ImageHarbor/cc6a84f0-dc8d-4d50-8b62-f3d3e74d5d71/scratchpad/sidecar-verify"
mkdir -p "$S/archives"
cp imageharbor/takeout-20230618T004316Z-001.zip "$S/archives/"
uv run imageharbor takeout ingest --archives "$S/archives" --dest "$S/organized"
```

Report, with real output:

- the ingest summary, and that a sidecar exists for **every** organized file with no flag passed;
- that one sidecar's `provenance[0].raw` is byte-equal to the corresponding archive member's JSON;
- that `albums[0].title` is a real Google album title, not a directory name;
- the contents of `.takeout-provenance/<archive_id>/`, confirming the face-tag file, the album tags, the HTML viewer, and the 8 orphaned sidecars are all present;
- that a **second** ingest leaves every sidecar byte-identical (hash the tree before and after and compare);
- `uv run imageharbor verify "$S/organized"` — unchanged, 70 OK / 0 FAILED;
- `sha256sum` of the archive before and after, proving it was never modified.

- [ ] **Step 5: Run the full suite and commit**

```bash
git add CLAUDE.md README.md docs/superpowers/specs/2026-08-18-lossless-sidecar-design.md
git commit -m "docs: document lossless sidecars in the architecture reference"
```

---

## Notes for the implementer

**The one property that matters.** Every other test in this plan is a sample; `test_never_loses_a_value_over_a_random_merge_sequence` is the contract. If it fails, a value was genuinely lost — fix the merge policy, never the test.

**Idempotence is not a nice-to-have.** "Append-only" without value-based dedup means every `watch` cycle grows every sidecar forever. Any history list that dedupes on a timestamp is a bug, and it will look like it works until the second run.

**The duplicate branch is where the value is.** A photo appearing in three exports accumulating three provenance entries and the union of its albums is the feature, not an edge case. Tasks 5 and 6 both depend on the duplicate path merging exactly as the fresh-ingest path does.
