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

# Where each flat map's superseded values are relocated. Every name in
# FLAT_MAPS must have an entry here -- `merge()` looks up each map's history
# list by this table rather than special-casing `exif`, so a changed
# `identity` value is routed to its OWN history list (`identity_history`)
# instead of a throwaway list nobody reads, which used to silently discard it.
FLAT_MAP_HISTORY_KEYS: dict[str, str] = {"exif": "exif_history", "identity": "identity_history"}

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

    An empty *core* (e.g. a demoted block that had no `value`/`tier`/`source`
    left after stripping annotations) is skipped outright -- relocating it
    would record an entry with nothing in it but annotations, which is noise,
    not provenance.
    """
    if not core:
        return
    if not _already_recorded(history, core):
        history.append({**core, **annotations})


def normalize_tier(value: Any) -> int:
    """Coerce a stored tier to an int; anything uncoercible reads as tier 0.

    A tier only ever needs to answer "which of these two is higher", so a
    value whose provenance cannot be established this way -- a hand-edited
    string, `None`, a float, a dict, a corrupted anything -- honestly
    outranks nothing. Comparing an un-normalized tier against another (e.g.
    `"30" > 40`) raises `TypeError` and breaks this module's totality
    guarantee; every tier comparison in this module (and any sibling module
    that reads a sidecar's stored tier, e.g. `backfill.py`) must route
    through this function rather than comparing the raw stored value.
    """
    if isinstance(value, bool):
        # bool is an int subclass but is never a legitimate tier -- treat it
        # the same as any other type that was never meant to hold one.
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_block(value: Any) -> dict[str, Any]:
    """Normalize a block-shaped field, preserving a bare value rather than dropping it.

    A v1 sidecar can hold a bare scalar where v2 expects a tiered block, and a
    hand edit can put anything anywhere. Coercing a non-dict to `{}` -- which is
    what a naive `isinstance` guard does -- silently discards it, and this
    module's whole contract is that nothing is silently discarded. Tier 0 is
    the honest reading: a value with no recorded provenance, which any real
    observation outranks.
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value, "tier": 0, "source": "unstructured"}


def _coerce_records(value: Any) -> list[dict]:
    """Normalize a list-of-records field, preserving a bare value or a bare
    entry rather than dropping it.

    Coercing any non-list to `[]` discards it outright, and a non-dict entry
    inside an otherwise-real list was once `continue`d past -- both are the
    same shape of loss `_coerce_block` fixes for tiered/versioned blocks,
    just one level down: a value with no home in the expected shape is still
    a value, not nothing. `None` is the one exception -- it means "field
    absent," the normal case for every list field on a fresh document, and
    must stay `[]` rather than becoming a stray `{"value": None}` entry on
    every single merge.
    """
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [item if isinstance(item, dict) else {"value": item} for item in items]


def _merge_tiered(base: Any, new: Any, observed_at: str) -> dict[str, Any]:
    """Higher tier wins; the loser is recorded rather than discarded.

    A *rejected* observation is recorded too. It is evidence about the photo
    even though it did not win, and discarding it would break the rule for the
    sake of a smaller file.
    """
    new = _coerce_block(new)
    base = _coerce_block(base)
    if not base:
        return {**_core(new), "observed_at": observed_at, "history": []}

    history = _coerce_records(base.get("history"))
    old_core, new_core = _core(base), _core(new)

    if old_core == new_core:
        return base  # nothing observed that is not already on record

    old_tier = normalize_tier(base.get("tier"))
    new_tier = normalize_tier(new.get("tier"))

    if new_tier > old_tier:
        _relocate(history, old_core, superseded_at=observed_at)
        return {**new_core, "observed_at": observed_at, "history": history}

    _relocate(history, new_core, observed_at=observed_at, rejected=True)
    return {**base, "history": history}


def _merge_versioned(base: Any, new: Any, observed_at: str) -> dict[str, Any]:
    """No tiers; any change records the previous block."""
    new, base = _coerce_block(new), _coerce_block(base)
    if not base:
        return {**_core(new), "observed_at": observed_at, "history": []}
    history = _coerce_records(base.get("history"))
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
    out = [dict(e) for e in _coerce_records(base)]
    index = {tuple(e.get(k) for k in keys): i for i, e in enumerate(out)}

    for entry in _coerce_records(new):
        identity = tuple(entry.get(k) for k in keys)
        if identity in index:
            current = out[index[identity]]
            history = _coerce_records(current.get("history"))
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
    """Merge *new* into *base* key by key.

    Both arguments are expected to be dicts (key -> scalar). A non-dict
    *base* or *new* -- a v1 sidecar with a malformed block, a hand edit --
    would silently vanish under a naive `isinstance` coercion to `{}`;
    instead it is relocated into *history* as a bare-value entry, the same
    "value with no home in the expected shape is still a value" treatment
    `_coerce_records` gives a stray list entry.
    """
    if not isinstance(base, dict):
        if base is not None:
            _relocate(history, {"value": base}, superseded_at=observed_at)
        base = {}
    out = dict(base)

    if not isinstance(new, dict):
        if new is not None:
            _relocate(history, {"value": new}, observed_at=observed_at, rejected=True)
        new = {}

    for key, value in new.items():
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
    if isinstance(doc, dict):
        doc = dict(doc)
    elif doc is None:
        doc = {}
    else:
        # A whole document that isn't dict-shaped at all -- a hand edit that
        # replaced the file with a bare value, or a caller error. Coercing
        # it to {} would drop it outright; instead it is kept as an
        # unstructured document, the same tier-0/bare-value treatment the
        # rest of this module gives a value that arrives in an unexpected
        # shape.
        doc = {"unstructured_document": doc}
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
        provenance = _coerce_records(doc.get("provenance"))
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
            albums = _coerce_records(doc.get("albums"))
            identity = (legacy.get("archive_id"), folder)
            if not any((a.get("archive_id"), a.get("folder")) == identity for a in albums):
                albums.append({"archive_id": legacy.get("archive_id"),
                               "folder": folder, "title": None})
            doc["albums"] = albums
    elif legacy is not None:
        # A `takeout` key that isn't dict-shaped -- a hand edit, a caller
        # error -- is still a value on record under the module-wide never-
        # lose rule. It cannot become a provenance entry (the reconstruction
        # above reads dict fields off it), so it is relocated into
        # `conflicts[]` rather than vanishing along with the `doc.pop` above
        # that removed it from `doc`. This is the same "unexpected shape is
        # still a value, not nothing" treatment `_coerce_block` and
        # `_coerce_records` give elsewhere in this module.
        conflicts = _coerce_records(doc.get("conflicts"))
        _relocate(conflicts, {"key": "takeout", "value": legacy}, observed_at="migration")
        doc["conflicts"] = conflicts

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

    flat_map_histories: dict[str, list] = {
        name: _coerce_records(out.get(hist_key))
        for name, hist_key in FLAT_MAP_HISTORY_KEYS.items()
    }
    conflicts = _coerce_records(out.get("conflicts"))

    if not isinstance(updates, dict):
        # A non-dict *updates* -- a caller error, or a hand-assembled patch
        # that lost its shape -- would vanish under a naive coercion to {}.
        # It has no key to merge under, but it is still evidence about the
        # photo, so it is recorded as a conflict rather than dropped.
        if updates is not None:
            _relocate(conflicts, {"key": "<updates>", "value": updates}, observed_at=observed_at)
        updates = {}

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
            history = flat_map_histories[key]
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

    for name, hist_key in FLAT_MAP_HISTORY_KEYS.items():
        history = flat_map_histories[name]
        if history:
            out[hist_key] = history
    if conflicts:
        out["conflicts"] = conflicts
    out["schema_version"] = SCHEMA_VERSION
    return out
