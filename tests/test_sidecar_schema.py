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


def _maybe_malform(rng: random.Random, value):
    """Occasionally swap a well-shaped update value for a bare scalar.

    All three Criticals found in this module were the same shape of bug: a
    value arriving where the code expected a dict (a tiered/versioned block)
    or a list (a keyed list), taking a code path the generator never
    produced. Mixing that shape into the random sequence -- a v1 sidecar
    scalar where v2 expects a block, a hand edit doing the same -- is the
    cheapest way to stop it recurring.
    """
    if rng.random() < 0.15:
        return f"malformed-{rng.randint(0, 999)}"
    return value


def test_never_loses_a_value_over_a_random_merge_sequence() -> None:
    """The formal statement of the contract, over generated sequences.

    Every value ever written must be findable in the final document -- at top
    level or relocated into a history list.
    """
    rng = random.Random(20260818)
    doc: dict = {}
    written: set[str] = set()
    prev_date: dict | None = None

    for i in range(60):
        date_block = {
            "value": f"20{10 + i % 15}-03-09T12:00:00",
            "tier": rng.choice([0, 10, 20, 30, 40]),
            "source": rng.choice(["exif_original", "external_sidecar", "filename_pattern"]),
        }
        # Every third iteration, re-report the previous date block verbatim --
        # the watch-loop case that exercises the reject-append dedup path
        # inside a longer sequence, not just in isolation.
        if prev_date is not None and i % 3 == 0:
            date_block = dict(prev_date)
        prev_date = date_block

        update = {
            "date": _maybe_malform(rng, date_block),
            "descriptor": _maybe_malform(rng, {
                "value": rng.choice(["", "beach-trip", "emma-birthday"]),
                "tier": rng.choice([0, 20, 30]),
                "source": rng.choice(["none", "ai_subject", "human_filename"]),
            }),
            "classification": _maybe_malform(rng, {
                "primary_subject": rng.choice(["beach", "dog", "sunset"]),
                "model_version": f"v{i % 3}",
            }),
            # path cycles (not strictly increasing) so the same `sources`
            # entry is re-observed later with a different `folder`, exercising
            # the gap-fill/relocate path for a genuinely empty recorded value.
            "sources": _maybe_malform(rng, [{"path": f"/src/{i % 20}.jpg",
                         "folder": rng.choice(["", f"folder-{i % 4}"]),
                         "first_seen": T0, "last_seen": T1}]),
            "albums": [{"archive_id": f"A{i % 3}", "folder": f"album-{i % 3}",
                        "title": rng.choice(["", None, f"Album {i % 3}"])}],
            "people": _maybe_malform(rng, [{"name": rng.choice(["", "Emma", "Sam", "Judy"])}]),
            "exif": {"Make": rng.choice(["Canon", "Nikon"]), f"Tag{i % 5}": i},
            "provenance": [{"kind": "takeout_media_json", "digest": f"D{i % 7}",
                            "raw": {"title": f"t{i}.jpg", "imageViews": str(i)}}],
        }
        written |= _leaves(update)
        doc = merge(doc, update, observed_at=T1)

        # Mid-sequence idempotence: re-applying the same update a second time,
        # at various points in the sequence (not just at the very end), must
        # never change the document.
        if i % 5 == 0:
            before = json.dumps(doc, sort_keys=True)
            doc = merge(doc, update, observed_at=T1)
            assert json.dumps(doc, sort_keys=True) == before

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


def test_a_bare_scalar_in_a_tiered_block_is_not_discarded() -> None:
    """A v1 sidecar can hold a scalar where v2 expects a block.

    Coercing it to {} loses it on the FIRST write, before any supersession
    logic runs -- the value never reaches the code that would have relocated it.
    """
    doc = merge({}, {"date": "2019-07-04"}, observed_at=T0)
    assert doc["date"]["value"] == "2019-07-04"
    assert doc["date"]["tier"] == 0

    later = merge(doc, {"date": {"value": "2015-03-09", "tier": 40, "source": "exif_original"}},
                  observed_at=T1)
    assert later["date"]["value"] == "2015-03-09"
    assert any(h.get("value") == "2019-07-04" for h in later["date"]["history"])


def test_a_bare_scalar_descriptor_is_not_discarded() -> None:
    doc = merge({}, {"descriptor": "beach-trip"}, observed_at=T0)
    assert doc["descriptor"]["value"] == "beach-trip"


def test_no_empty_history_entries_are_recorded() -> None:
    """An entry with no value is noise, not provenance."""
    doc = merge({}, {"date": {}}, observed_at=T0)
    doc = merge(doc, {"date": {"value": "2015-03-09", "tier": 40}}, observed_at=T1)
    assert all(
        {k: v for k, v in h.items()
         if k not in {"observed_at", "superseded_at", "first_seen", "last_seen", "rejected", "history"}}
        for h in doc["date"]["history"]
    ), doc["date"]["history"]


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


def test_a_repeated_losing_observation_does_not_grow_the_history() -> None:
    """The watch-loop case: a lower-tier source re-reporting the same fact.

    A filename-derived date keeps losing to EXIF on every scan. Recording the
    rejection is correct; recording it again on every scan is unbounded growth,
    and it looks like it works until the second run.
    """
    doc = merge({}, {"date": {"value": "2015-03-09", "tier": 30, "source": "exif_original"}},
                observed_at=T0)
    losing = {"date": {"value": "2019-07-04", "tier": 10, "source": "filename_pattern"}}

    doc = merge(doc, losing, observed_at=T1)
    after_first = json.dumps(doc, sort_keys=True)
    for _ in range(5):
        doc = merge(doc, losing, observed_at=T1)

    assert len(doc["date"]["history"]) == 1
    assert json.dumps(doc, sort_keys=True) == after_first


@pytest.mark.parametrize("malformed_tier", ["not-a-number", None, 25.0, {"nested": 1}])
def test_a_non_int_tier_never_raises(malformed_tier) -> None:
    """Finding 1: `_merge_tiered` compared `new_tier > old_tier` without
    normalizing types, so a hand-edited or corrupted sidecar carrying a
    non-int tier (a string, `None`, a float, a dict) raised `TypeError` --
    a crash, where the never-lose rule demands a degrade to "less is
    recorded" instead. `merge()` must stay total regardless of which side
    (base or update) carries the malformed value.
    """
    good = {"date": {"value": "2015-03-09", "tier": 40, "source": "exif_original"}}
    malformed = {"date": {"value": "2019-07-04", "tier": malformed_tier, "source": "hand_edit"}}

    base = merge({}, malformed, observed_at=T0)
    after = merge(base, good, observed_at=T1)  # malformed as the base -- must not raise
    assert after["date"]["value"] in ("2015-03-09", "2019-07-04")

    base2 = merge({}, good, observed_at=T0)
    after2 = merge(base2, malformed, observed_at=T1)  # malformed as the update -- must not raise
    assert after2["date"]["value"] in ("2015-03-09", "2019-07-04")


def test_a_coercible_string_tier_compares_as_its_int_value() -> None:
    """A numeric-string tier ("30") is real evidence, not automatically the
    loser -- it must normalize to 30 and win or lose on that value, exactly
    like the reproduction from finding 1.
    """
    base = merge({}, {"date": {"value": "2019-07-04", "tier": "30", "source": "x"}}, observed_at=T0)

    beats_lower = merge(base, {"date": {"value": "2015-03-09", "tier": 20, "source": "y"}},
                         observed_at=T1)
    assert beats_lower["date"]["value"] == "2019-07-04"

    loses_to_higher = merge(base, {"date": {"value": "2015-03-09", "tier": 40, "source": "exif_original"}},
                             observed_at=T1)
    assert loses_to_higher["date"]["value"] == "2015-03-09"


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


def test_a_recorded_empty_value_is_relocated_not_dropped() -> None:
    """"Recorded as empty" is not the same as "never recorded".

    Google ships blank album titles; treating one as absent silently discards a
    real observation with no trace anywhere in the document.
    """
    base = merge({}, {"albums": [{"archive_id": "A1", "folder": "d", "title": ""}]},
                 observed_at=T0)
    after = merge(base, {"albums": [{"archive_id": "A1", "folder": "d", "title": "Real Title"}]},
                  observed_at=T1)

    entry = after["albums"][0]
    assert entry["title"] == "Real Title"
    assert any(h.get("field") == "title" and h.get("value") == "" for h in entry["history"])


def test_last_seen_moves_without_recording_history() -> None:
    """The one deliberate exception, pinned so it is a choice and not a leak."""
    u1 = {"sources": [{"path": "/a.jpg", "first_seen": T0, "last_seen": T0}]}
    u2 = {"sources": [{"path": "/a.jpg", "first_seen": T0, "last_seen": T1}]}
    doc = merge(merge({}, u1, observed_at=T0), u2, observed_at=T1)
    entry = doc["sources"][0]
    assert entry["last_seen"] == T1
    assert not any(h.get("field") == "last_seen" for h in entry.get("history", []))


def test_last_seen_still_advances_without_history_across_repeated_merges() -> None:
    """Same guarantee as above, but over more than one supersession -- the
    generalized `_ANNOTATION_FIELDS` handling must not regress the field it
    was modelled on."""
    doc: dict = {}
    for i in range(5):
        doc = merge(
            doc,
            {"sources": [{"path": "/a.jpg", "first_seen": T0,
                          "last_seen": f"2026-08-31T00:00:0{i}+00:00"}]},
            observed_at=T1,
        )
    entry = doc["sources"][0]
    assert entry["last_seen"] == "2026-08-31T00:00:04+00:00"
    assert "history" not in entry


def test_first_seen_out_of_order_still_relocates_the_later_value() -> None:
    """`first_seen` is the one `_ANNOTATION_FIELDS` member that does NOT
    advance-and-drop like `last_seen`/`confirmed_at`: it keeps the EARLIEST
    value, and a genuinely earlier value arriving out of order still
    relocates the now-superseded later one. Pinned so the generalization
    added for `confirmed_at` cannot accidentally flatten this exception too.
    """
    base = merge({}, {"sources": [{"path": "/a.jpg", "first_seen": T1, "last_seen": T1}]},
                 observed_at=T1)
    after = merge(base, {"sources": [{"path": "/a.jpg", "first_seen": T0, "last_seen": T1}]},
                  observed_at=T1)
    entry = after["sources"][0]
    assert entry["first_seen"] == T0
    assert any(h.get("field") == "first_seen" and h.get("value") == T1 for h in entry["history"])


def test_a_non_annotation_keyed_list_field_still_relocates_to_history() -> None:
    """Guard against the `_ANNOTATION_FIELDS` generalization over-reaching:
    an ordinary field -- not in `_ANNOTATION_FIELDS` -- must still relocate
    its superseded value to the entry's own `history`, exactly as before.
    """
    base = merge({}, {"provenance": [{"digest": "D1", "raw": {"title": "a.jpg"}}]}, observed_at=T0)
    after = merge(base, {"provenance": [{"digest": "D1", "raw": {"title": "b.jpg"}}]}, observed_at=T1)
    entry = after["provenance"][0]
    assert entry["raw"] == {"title": "b.jpg"}
    assert any(h.get("field") == "raw" and h.get("value") == {"title": "a.jpg"} for h in entry["history"])


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


def test_a_changed_identity_value_moves_the_old_one_to_its_own_history() -> None:
    """Finding 4: `identity` is a FLAT_MAPS entry like `exif`, but `merge()`
    used to route its superseded values through a throwaway `[]` nobody
    read, silently discarding them. It must get a real history list, keyed
    separately from `exif_history` so the two never collide."""
    base = merge({}, {"identity": {"size": 10}}, observed_at=T0)
    after = merge(base, {"identity": {"size": 20}}, observed_at=T1)
    assert after["identity"]["size"] == 20
    assert any(h["key"] == "size" and h["value"] == 10 for h in after["identity_history"])
    # Must not leak into (or be confused with) exif's own history list.
    assert "size" not in {h.get("key") for h in after.get("exif_history", [])}


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


def test_a_repeated_identical_conflict_does_not_grow() -> None:
    """The unknown-key conflict recorder is dedup-guarded like every other list.

    It was the last append site outside `_relocate`, and it was safe only by
    coincidence -- `observed_at` happened to be a stripped annotation. Routing
    it through the helper makes that safety structural instead of lucky.
    """
    doc = merge({}, {"note": "mine"}, observed_at=T0)
    for _ in range(10):
        doc = merge(doc, {"note": "theirs"}, observed_at=T1)
    assert doc["note"] == "mine"
    assert len(doc["conflicts"]) == 1


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


def test_migrate_relocates_a_non_dict_takeout_value_into_conflicts() -> None:
    """Finding 3: `migrate()` used to `doc.pop("takeout", None)` and only
    handle the dict-shaped case -- a `takeout` key holding anything else (a
    hand edit, a caller error) was popped and silently dropped. It must
    survive, relocated into `conflicts[]` like every other value that
    arrives in an unexpected shape."""
    out = migrate({"schema_version": 1, "takeout": "a bare note"})
    assert "takeout" not in out
    assert any(
        c.get("key") == "takeout" and c.get("value") == "a bare note"
        for c in out.get("conflicts", [])
    )


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
