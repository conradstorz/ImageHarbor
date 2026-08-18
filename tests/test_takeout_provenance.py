"""Behavioral tests for the Takeout provenance room.

Synthetic zips built in tmp_path replicate the real export's non-media
members: Albums.json, a Picasa face-tag file, an orphaned per-photo JSON, and
Google's own HTML viewer. No 79 MB fixture is committed -- the shapes are
what actually matter.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import pytest

from imageharbor.takeout import archive as archive_mod
from imageharbor.takeout import provenance

D = "Takeout/AlbumArchive/Hangouts/album"


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _identity(zip_path: Path, archive_id: str = "arc-1") -> archive_mod.ArchiveIdentity:
    stat = zip_path.stat()
    return archive_mod.ArchiveIdentity(
        archive_id=archive_id, path=zip_path, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
    )


@pytest.fixture()
def dirs(tmp_path: Path):
    archives = tmp_path / "archives"
    archives.mkdir()
    organized = tmp_path / "organized"
    organized.mkdir()
    return archives, organized


def _members(zf: zipfile.ZipFile) -> list[archive_mod.MemberInfo]:
    return list(archive_mod.iter_members(zf))


# --- verbatim preservation ---------------------------------------------------


def test_every_non_media_member_is_written_verbatim(dirs) -> None:
    archives, organized = dirs
    entries = {
        f"{D}/2015-03-09.jpg": b"\xff\xd8\xff\xe0fakejpegbytes\xff\xd9",
        f"{D}/2015-03-09.jpg.json": b'{"title": "2015-03-09.jpg"}',
        f"{D}/archive_browser.html": b"<html>Google offline viewer</html>",
        "Takeout/AlbumArchive/picasa_web_album_face_tags.json": (
            b'{"faces": [{"tag": "Alice"}]}'
        ),
    }
    zip_path = _zip(archives / "t.zip", entries)
    identity = _identity(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        written = provenance.preserve(
            organized, identity, zf, _members(zf), orphaned=set(),
        )

    assert written == 3  # every entry except the .jpg itself

    room = organized / provenance.ROOM_NAME / identity.archive_id
    for member_path in (
        f"{D}/2015-03-09.jpg.json",
        f"{D}/archive_browser.html",
        "Takeout/AlbumArchive/picasa_web_album_face_tags.json",
    ):
        preserved = room / member_path
        assert preserved.is_file()
        assert preserved.read_bytes() == entries[member_path]

    # The image itself is never mirrored into the provenance room.
    assert not (room / f"{D}/2015-03-09.jpg").exists()


def test_archive_browser_html_is_preserved_uncurated(dirs) -> None:
    """No judgement about which unknown file is worth keeping."""
    archives, organized = dirs
    zip_path = _zip(archives / "t.zip", {
        "Takeout/archive_browser.html": b"<html>viewer</html>",
    })
    identity = _identity(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        written = provenance.preserve(organized, identity, zf, _members(zf), orphaned=set())

    assert written == 1
    room = organized / provenance.ROOM_NAME / identity.archive_id
    assert (room / "Takeout/archive_browser.html").read_bytes() == b"<html>viewer</html>"


# --- special-cased layout ----------------------------------------------------


def test_albums_json_lands_under_albums_folder(dirs) -> None:
    archives, organized = dirs
    zip_path = _zip(archives / "t.zip", {
        f"{D}/Albums.json": b'{"title": "Vacation"}',
    })
    identity = _identity(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        provenance.preserve(organized, identity, zf, _members(zf), orphaned=set())

    room = organized / provenance.ROOM_NAME / identity.archive_id
    assert (room / "albums" / "album" / "Albums.json").read_bytes() == b'{"title": "Vacation"}'


def test_orphaned_media_json_lands_under_orphaned(dirs) -> None:
    """A media JSON whose photo is absent from the batch is not lost, and is
    not silently mixed in with normally-paired sidecars either."""
    archives, organized = dirs
    sidecar_path = f"{D}/P1010089.JPG(1).json"
    zip_path = _zip(archives / "t.zip", {
        sidecar_path: b'{"title": "P1010089.JPG"}',
    })
    identity = _identity(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        written = provenance.preserve(
            organized, identity, zf, _members(zf), orphaned={sidecar_path},
        )

    assert written == 1
    room = organized / provenance.ROOM_NAME / identity.archive_id
    assert (room / "orphaned" / "P1010089.JPG(1).json").read_bytes() == b'{"title": "P1010089.JPG"}'
    assert not (room / sidecar_path).exists()


# --- manifest -----------------------------------------------------------------


def test_manifest_lists_every_preserved_document_with_its_digest(dirs) -> None:
    archives, organized = dirs
    entries = {
        f"{D}/a.jpg.json": b'{"title": "a"}',
        f"{D}/archive_browser.html": b"<html></html>",
    }
    zip_path = _zip(archives / "t.zip", entries)
    identity = _identity(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        provenance.preserve(organized, identity, zf, _members(zf), orphaned=set())

    manifest_file = provenance.manifest_path(organized, identity.archive_id)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert manifest["archive"] == "t.zip"
    assert manifest["archive_id"] == identity.archive_id
    assert {doc["member"] for doc in manifest["documents"]} == set(entries)
    for doc in manifest["documents"]:
        assert isinstance(doc["digest"], str) and doc["digest"]
        assert doc["stored_as"]


# --- idempotency --------------------------------------------------------------


def test_re_preserving_the_same_archive_writes_nothing_new(dirs, monkeypatch) -> None:
    archives, organized = dirs
    zip_path = _zip(archives / "t.zip", {
        f"{D}/a.jpg.json": b'{"title": "a"}',
        f"{D}/archive_browser.html": b"<html></html>",
    })
    identity = _identity(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        first = provenance.preserve(organized, identity, zf, _members(zf), orphaned=set())
    assert first == 2

    room = organized / provenance.ROOM_NAME / identity.archive_id
    mtimes_before = {
        p: p.stat().st_mtime_ns for p in room.rglob("*") if p.is_file()
    }

    calls = []
    real = provenance._write_bytes
    monkeypatch.setattr(
        provenance, "_write_bytes",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        second = provenance.preserve(organized, identity, zf, _members(zf), orphaned=set())

    assert second == 0
    assert calls == []
    mtimes_after = {p: p.stat().st_mtime_ns for p in room.rglob("*") if p.is_file()}
    assert mtimes_after == mtimes_before


# --- failure isolation ---------------------------------------------------------


def test_a_write_failure_is_logged_and_does_not_raise(dirs, monkeypatch, caplog) -> None:
    archives, organized = dirs
    zip_path = _zip(archives / "t.zip", {
        f"{D}/archive_browser.html": b"<html></html>",
    })
    identity = _identity(zip_path)

    def _boom(*_a, **_k):
        raise OSError("disk is full")

    monkeypatch.setattr(provenance, "_write_bytes", _boom)

    with caplog.at_level(logging.WARNING):
        with zipfile.ZipFile(zip_path, "r") as zf:
            written = provenance.preserve(
                organized, identity, zf, _members(zf), orphaned=set(),
            )

    assert written == 0
    assert any(
        "archive_browser.html" in record.message or "Failed to preserve" in record.message
        for record in caplog.records
    )
