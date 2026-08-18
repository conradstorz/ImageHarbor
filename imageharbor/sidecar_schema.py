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

# Fields that annotate an observation rather than identify it. `_core` strips
# every one, because a history entry's identity is the VALUE it records --
# never when it was seen, nor why it lost. Adding a key here is how you make a
# new annotation safe; forgetting to is how unbounded growth gets reintroduced.
_ANNOTATION_FIELDS = frozenset({
    "observed_at", "superseded_at", "first_seen", "last_seen", "rejected", "history",
})


def _core(block: Any) -> dict[str, Any]:
    """A block stripped of annotations -- its identity for dedup purposes."""
    if not isinstance(block, dict):
        return {}
    return {k: v for k, v in block.items() if k not in _ANNOTATION_FIELDS}


def _already_recorded(history: list, core: dict) -> bool:
    return any(_core(h) == core for h in history if isinstance(h, dict))


def _relocate(history: list, core: dict, **annotations: Any) -> None:
    """Record *core* in *history* unless an entry with the same value is there.

    The single place the append-with-dedup pattern lives. Dedup compares
    `_core()` of each stored entry against *core*, so any annotation passed
    here MUST be in `_ANNOTATION_FIELDS` or the entry can never match itself
    and the list grows on every merge. That is not hypothetical: a `rejected`
    flag outside the stripped set is exactly how this module once grew a
    history entry per `watch` cycle, forever.
    """
    if not _already_recorded(history, core):
        history.append({**core, **annotations})


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
        _relocate(history, old_core, superseded_at=observed_at)
        return {**new_core, "observed_at": observed_at, "history": history}

    _relocate(history, new_core, observed_at=observed_at, rejected=True)
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
    _relocate(history, old_core, superseded_at=observed_at)
    return {**new_core, "observed_at": observed_at, "history": history}


def _merge_keyed_list(base: Any, new: Any, keys: tuple[str, ...], observed_at: str) -> list:
    """Append-only union. An existing entry gains fields; it never loses any.

    Three policies operate on an existing entry's fields:

    * `first_seen` keeps the *earliest* value seen so far -- a later
      re-report of the already-recorded first sighting is the same fact, not
      new information, so it is dropped without a history entry. But a
      genuinely *earlier* value arriving out of order (e.g. archive B is
      processed before archive A, even though A's copy actually came first)
      is real information and must not vanish: the later, now-superseded
      value is relocated to history exactly like any other differing field,
      never silently overwritten.
    * `last_seen` always advances to the newest observation; the superseded
      timestamp is redundant with the new one (both just say "still here"),
      so it is not recorded either. This -- along with `observed_at` on the
      tiered/versioned/flat-map blocks -- is the module's one deliberate
      exception to "never lose any": both are timestamps of *observation*,
      not facts about the photo, and recording their history would be
      unbounded by construction, one entry per merge, forever. The rule
      yields here to the idempotence rule that makes it usable at all.
    * Every other field (e.g. a takeout record's `raw` payload, which really
      can differ between observations of the "same" digest) keeps "newest
      wins" but relocates a genuinely different old value into the entry's
      own `history` list -- the same relocate-don't-drop rule `_merge_tiered`
      uses, just scoped to one list entry instead of one top-level block.
      This applies even when the recorded value is falsy (`""`, `[]`, `{}`,
      `None`): "recorded as empty" is a real observation, not the same as
      "never recorded", so only a field's outright *absence* from the
      current entry counts as a gap to fill silently.
    """
    out = [dict(e) for e in _as_list(base) if isinstance(e, dict)]
    index = {tuple(e.get(k) for k in keys): i for i, e in enumerate(out)}

    for entry in _as_list(new):
        if not isinstance(entry, dict):
            continue
        identity = tuple(entry.get(k) for k in keys)
        if identity in index:
            current = out[index[identity]]
            history = [h for h in _as_list(current.get("history")) if isinstance(h, dict)]
            for field, value in entry.items():
                if field == "history":
                    continue
                if field == "first_seen":
                    if field not in current:
                        current[field] = value
                    elif value != current[field]:
                        existing = current[field]
                        # Comparable ISO timestamps: keep the earlier one,
                        # relocate the later one. Anything not comparable
                        # (malformed input -- this module must stay total)
                        # falls back to keeping the incumbent and still
                        # records the newcomer, rather than guessing.
                        if isinstance(value, str) and isinstance(existing, str) and value < existing:
                            current[field], loser = value, existing
                        else:
                            loser = value
                        _relocate(history, {"field": field, "value": loser}, superseded_at=observed_at)
                    continue
                if field == "last_seen":
                    current[field] = value
                    continue
                if field not in current:
                    current[field] = value
                elif current[field] != value:
                    _relocate(history, {"field": field, "value": current[field]},
                              superseded_at=observed_at)
                    current[field] = value
            if history:
                current["history"] = history
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
            _relocate(history, {"key": key, "value": out[key]}, superseded_at=observed_at)
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

    # The old version number is itself a value on record (e.g. a v1 sidecar's
    # literal `1`) and the module-wide rule is "never lose any" -- so it is
    # relocated rather than clobbered by the new `schema_version` below.
    # `setdefault` makes this safe to reach twice: a document already carrying
    # this field (e.g. one migrated from v0 some day) keeps its original
    # value rather than being overwritten by an intermediate one.
    prev_version = doc.get("schema_version")

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

    if prev_version is not None and prev_version != SCHEMA_VERSION:
        doc.setdefault("migrated_from_schema_version", prev_version)
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
            out[key] = _merge_keyed_list(out.get(key), value, KEYED_LISTS[key], observed_at)
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
            _relocate(conflicts, core, observed_at=observed_at)

    if exif_history:
        out["exif_history"] = exif_history
    if conflicts:
        out["conflicts"] = conflicts
    out["schema_version"] = SCHEMA_VERSION
    return out
