"""The Picasa roster is vocabulary, never evidence.

No roster file was present in the export this project was written against
(see `.superpowers/sdd/progress.md`, "FINDING: there is no Picasa roster in
this export"). The `contacts.xml` shape exercised here comes from Picasa's
documented format, not from an observed file -- see `roster.py`'s module
docstring.
"""

from __future__ import annotations

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


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "catalog.db"
    Catalog(db).close()
    s = FaceStore(db)
    yield s
    s.close()


def _seed_roster(dest, sample: bytes = SAMPLE) -> None:
    room = dest / ".takeout-provenance" / "abc"
    room.mkdir(parents=True)
    (room / "contacts.xml").write_bytes(sample)


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


def test_find_roster_files_returns_empty_list_when_nothing_matches(tmp_path):
    # The expected case for this export: no roster, no error.
    assert roster.find_roster_files(tmp_path / "dest") == []


def test_find_roster_files_locates_a_seeded_contacts_xml(tmp_path):
    dest = tmp_path / "dest"
    _seed_roster(dest)
    found = roster.find_roster_files(dest)
    assert len(found) == 1
    assert found[0].name == "contacts.xml"


def test_import_adds_people_with_the_roster_source(tmp_path, store):
    dest = tmp_path / "dest"
    _seed_roster(dest)

    added = roster.import_names(store, dest)
    assert added == 2
    assert sorted(store.known_names()) == ["Conrad Storz", "Gladys Blankenbeker"]


def test_import_is_idempotent(tmp_path, store):
    dest = tmp_path / "dest"
    _seed_roster(dest)

    roster.import_names(store, dest)
    assert roster.import_names(store, dest) == 0
    assert len(store.known_names()) == 2


def test_a_roster_name_is_never_attached_to_a_cluster(tmp_path, store):
    dest = tmp_path / "dest"
    _seed_roster(dest)

    roster.import_names(store, dest)
    # The roster carries no photo reference at all, so it can seed a name list
    # and nothing more.
    assert store.cluster_ids() == []


def test_missing_provenance_directory_is_not_an_error(tmp_path, store):
    assert roster.import_names(store, tmp_path / "nowhere") == 0
