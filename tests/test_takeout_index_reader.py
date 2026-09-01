"""Reading a Takeout_Inventory pairing index.

Databases here are built from a schema literal. That is a real seam between
two repositories: if Takeout_Inventory changes its schema these tests keep
passing while production breaks, which is why the reader asserts its expected
columns and the version on open.
"""
import sqlite3

import pytest

from imageharbor.takeout import index_reader

# Fixed archive tuple used by the "covered archive" fixtures below: a name,
# a size and mtime matching stats_for()'s defaults, a member count, and no
# error.
_ARCHIVE_1_COVERED = ("part-1.zip", 100, 5, 2, None)
_ARCHIVE_2_COVERED = ("part-2.zip", 100, 5, 1, None)
_ARCHIVE_2_UNCOVERED = ("part-2.zip", 999, 5, 1, None)


SCHEMA = """
CREATE TABLE sidecar (
  id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, name TEXT NOT NULL,
  title TEXT, taken_at TEXT, lat REAL, lon REAL, device TEXT,
  trashed INTEGER, archived INTEGER, from_partner INTEGER,
  parse_error TEXT, role TEXT);
CREATE TABLE media (
  id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, area TEXT NOT NULL,
  folder TEXT NOT NULL, name TEXT NOT NULL, ext TEXT, size INTEGER,
  actual_type TEXT, sidecar_id INTEGER REFERENCES sidecar(id),
  rule TEXT NOT NULL, confidence TEXT NOT NULL);
CREATE TABLE archive (
  name TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
  members INTEGER NOT NULL, error TEXT);
CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def make_index(path, *, version="1", archives=(("part-1.zip", 100, 5, 2, None),),
               media=(), sidecars=()):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for row in archives:
        con.execute("INSERT INTO archive VALUES (?,?,?,?,?)", row)
    for row in sidecars:
        con.execute("INSERT INTO sidecar (id, archive, path, name)"
                    " VALUES (?,?,?,?)", row)
    for row in media:
        con.execute("INSERT INTO media (archive, path, area, folder, name,"
                    " sidecar_id, rule, confidence) VALUES (?,?,?,?,?,?,?,?)", row)
    if version is not None:
        con.execute("INSERT INTO index_meta VALUES ('schema_version', ?)", (version,))
    con.commit()
    con.close()
    return path


def stats_for(size=100, mtime=5):
    class S:
        st_size = size
        st_mtime = mtime
    return S()


def test_open_reads_pairings(tmp_path):
    db = make_index(
        tmp_path / "i.sqlite",
        sidecars=[(1, "part-1.zip", "T/GP/a.jpg.json", "a.jpg.json")],
        media=[("part-1.zip", "T/GP/a.jpg", "GP", "GP", "a.jpg", 1, "exact", "own")])
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    p = idx.sidecar_for("T/GP/a.jpg")
    assert p.sidecar == "T/GP/a.jpg.json"
    assert p.confidence == "own"
    assert p.rule == "exact"


def test_orphan_media_reads_back_as_no_sidecar(tmp_path):
    db = make_index(
        tmp_path / "i.sqlite",
        media=[("part-1.zip", "T/GP/x.jpg", "GP", "GP", "x.jpg", None,
                "orphan", "none")])
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    p = idx.sidecar_for("T/GP/x.jpg")
    assert p.sidecar is None
    assert p.confidence == "none"


def test_unknown_member_returns_none(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    assert idx.sidecar_for("T/GP/never-seen.jpg") is None


def test_archive_with_a_different_size_is_not_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for(size=999)})
    assert not idx.covers("part-1.zip")


def test_archive_with_a_different_mtime_is_not_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for(mtime=999)})
    assert not idx.covers("part-1.zip")


def test_archive_absent_from_the_index_is_not_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-9.zip": stats_for()})
    assert not idx.covers("part-9.zip")


def test_matching_archive_is_covered(tmp_path):
    db = make_index(tmp_path / "i.sqlite")
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    assert idx.covers("part-1.zip")


def test_a_missing_file_is_unusable(tmp_path):
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(tmp_path / "nope.sqlite", {})


def test_a_newer_schema_version_is_unusable(tmp_path):
    db = make_index(tmp_path / "i.sqlite", version="2")
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})


def test_an_unversioned_index_is_unusable(tmp_path):
    db = make_index(tmp_path / "i.sqlite", version=None)
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})


def test_a_non_database_file_is_unusable(tmp_path):
    bad = tmp_path / "i.sqlite"
    bad.write_bytes(b"this is not a database")
    with pytest.raises(index_reader.IndexUnusable):
        index_reader.IndexPairings.open(bad, {})


def test_a_missing_media_column_is_unusable(tmp_path):
    """I5: the docstring on `_verify` says a schema change in the sibling
    repo "must surface here as a clear error, never as a wrong answer" --
    this is the only test that actually builds a schema-drifted index and
    checks that promise. Built by hand (not via `make_index`, which always
    creates the full, current schema) with a `media` table missing
    `confidence`, one of `_MEDIA_COLUMNS`. Matched on "table media is
    missing" specifically, not merely on the word "confidence" -- a
    downstream `sqlite3.OperationalError` from the SELECT in
    `_read_pairings` ("no such column: m.confidence") also contains
    "confidence" and would otherwise let this test pass even if the column
    check itself were disabled.
    """
    db = tmp_path / "i.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE sidecar (
          id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, name TEXT NOT NULL);
        CREATE TABLE media (
          id INTEGER PRIMARY KEY, archive TEXT, path TEXT NOT NULL, area TEXT NOT NULL,
          folder TEXT NOT NULL, name TEXT NOT NULL, sidecar_id INTEGER,
          rule TEXT NOT NULL);
        CREATE TABLE archive (
          name TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
          members INTEGER NOT NULL, error TEXT);
        CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    con.execute("INSERT INTO archive VALUES (?,?,?,?,?)", ("part-1.zip", 100, 5, 0, None))
    con.execute("INSERT INTO index_meta VALUES ('schema_version', '1')")
    con.commit()
    con.close()

    with pytest.raises(index_reader.IndexUnusable, match=r"table media is missing"):
        index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})


def test_open_percent_encodes_a_hash_in_the_path(tmp_path):
    # '#' starts a URI fragment; a bare f-string interpolation truncates the
    # connect URI there and SQLite opens a different (nonexistent) path --
    # surfacing as a misleading "no index_meta table" IndexUnusable for a
    # perfectly good file. A directory name containing '#' is legal on both
    # Windows and POSIX.
    special_dir = tmp_path / "has#hash"
    special_dir.mkdir()
    db = make_index(
        special_dir / "i.sqlite",
        sidecars=[(1, "part-1.zip", "T/GP/a.jpg.json", "a.jpg.json")],
        media=[("part-1.zip", "T/GP/a.jpg", "GP", "GP", "a.jpg", 1, "exact", "own")])
    idx = index_reader.IndexPairings.open(db, {"part-1.zip": stats_for()})
    p = idx.sidecar_for("T/GP/a.jpg")
    assert p.sidecar == "T/GP/a.jpg.json"
    assert p.confidence == "own"
    assert p.rule == "exact"


def test_uncovered_archive_does_not_leak_its_pairings(tmp_path):
    # `_read_pairings` filters on `m_archive not in covered` -- the guard
    # stopping a stale index from supplying pairings for an archive that
    # failed verification. This exercises both sides of that guard directly.
    db = make_index(
        tmp_path / "i.sqlite",
        archives=(_ARCHIVE_1_COVERED, _ARCHIVE_2_UNCOVERED),
        sidecars=[(1, "part-1.zip", "T/GP/a.jpg.json", "a.jpg.json")],
        media=[
            ("part-1.zip", "T/GP/a.jpg", "GP", "GP", "a.jpg", 1, "exact", "own"),
            ("part-2.zip", "T/GP/b.jpg", "GP", "GP", "b.jpg", None, "orphan", "none"),
        ])
    idx = index_reader.IndexPairings.open(
        db, {"part-1.zip": stats_for(), "part-2.zip": stats_for()})
    assert idx.covers("part-1.zip")
    assert not idx.covers("part-2.zip")
    assert idx.sidecar_for("T/GP/a.jpg") is not None
    assert idx.sidecar_for("T/GP/b.jpg") is None


def test_member_path_shared_by_two_covered_archives_is_excluded(tmp_path):
    # Mirrors `pairing.py`'s `PairingIndex.ambiguous_media`: a member path
    # keyed with no archive dimension can't tell two archives sharing that
    # path apart, so pairing either one risks dating one archive's bytes
    # with another archive's sidecar. The index reader must refuse this too,
    # not just the module it stands in for.
    db = make_index(
        tmp_path / "i.sqlite",
        archives=(_ARCHIVE_1_COVERED, _ARCHIVE_2_COVERED),
        sidecars=[
            (1, "part-1.zip", "T/GP/a.jpg.json", "a.jpg.json"),
            (2, "part-2.zip", "T/GP/a2.jpg.json", "a2.jpg.json"),
        ],
        media=[
            ("part-1.zip", "T/GP/a.jpg", "GP", "GP", "a.jpg", 1, "exact", "own"),
            ("part-2.zip", "T/GP/a.jpg", "GP", "GP", "a.jpg", 2, "exact", "own"),
        ])
    idx = index_reader.IndexPairings.open(
        db, {"part-1.zip": stats_for(), "part-2.zip": stats_for()})
    assert idx.covers("part-1.zip")
    assert idx.covers("part-2.zip")
    assert idx.sidecar_for("T/GP/a.jpg") is None
