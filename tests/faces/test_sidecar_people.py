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


def test_repeated_reconfirmation_with_differing_confirmed_at_does_not_grow_history():
    """`propagate_sidecars` (Task 12) stamps a fresh `confirmed_at` on every
    run. `_merge_keyed_list` must treat `confirmed_at` the way it already
    treats `last_seen` -- advance to the newest observation, don't relocate
    the superseded one -- or the history list grows by one entry per run,
    forever. `_ANNOTATION_FIELDS` claiming `confirmed_at` is meaningless if
    the keyed-list merge path never consults it.
    """
    doc: dict = {}
    stable_keys: set[str] | None = None
    for i in range(5):
        updates = {
            "people": [
                {
                    "name": "Emma",
                    "source": "imageharbor_faces",
                    "cluster_ids": [3],
                    "confirmed_at": f"2026-08-31T00:00:0{i}+00:00",
                }
            ]
        }
        doc = sidecar_schema.merge(doc, updates, observed_at=f"2026-08-31T00:00:0{i}+00:00")
        entry = doc["people"][0]
        assert "history" not in entry
        # The entry's shape must be stable from the second merge onward --
        # not just history-free, but not silently accreting some other field.
        if i == 1:
            stable_keys = set(entry.keys())
        elif i > 1:
            assert set(entry.keys()) == stable_keys

    assert doc["people"][0]["confirmed_at"] == "2026-08-31T00:00:04+00:00"


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
