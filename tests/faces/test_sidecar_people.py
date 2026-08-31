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
