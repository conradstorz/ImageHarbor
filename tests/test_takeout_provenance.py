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


# --- containment (zip-slip defense-in-depth) -----------------------------------


def test_a_backslash_member_path_stays_inside_the_room(tmp_path: Path) -> None:
    # _safe_relpath drops "." and ".." SLASH components, but a component
    # containing backslashes ("..\\..\\..\\pwned.txt") survived whole and,
    # on Windows, pathlib expands it into separators -- writing outside the
    # room the docstring promises to contain. This is defense-in-depth, not
    # a live exploit: CPython's own zipfile read path normalizes "\" -> "/"
    # on Windows before a name ever reaches this module, so a real archive
    # cannot trigger it today. It guards a future archive reader, a
    # hand-rolled central-directory recovery, or a MemberInfo built from a
    # non-zipfile source.
    rel = provenance._safe_relpath("Takeout/..\\..\\..\\pwned.txt")
    room = tmp_path / "room"
    assert (room / rel).resolve().is_relative_to(room.resolve())


def test_safe_component_replaces_backslashes() -> None:
    assert "\\" not in provenance._safe_component("..\\..\\evil")


def test_a_hostile_member_path_is_neutralized_and_stays_in_the_room(
    dirs, tmp_path: Path, monkeypatch, caplog,
) -> None:
    # Mirrors Task 5's staging test: the malicious name is injected directly
    # into a hand-built MemberInfo (not round-tripped through a real zip
    # entry, since CPython's zipfile normalizes "\" -> "/" on read on
    # Windows and would mask the very bug this guards against) and zf.open
    # is patched to serve real bytes for it -- isolating exactly what
    # preserve()/_stored_path does with a hostile member.path. With the
    # sanitizer fixed, the hostile name is neutralized rather than rejected:
    # it is preserved safely under the room (never lost), just not at the
    # path it tried to claim, alongside an ordinary sibling member.
    archives, organized = dirs
    zip_path = _zip(archives / "t.zip", {
        "Takeout/archive_browser.html": b"<html>viewer</html>",
        "payload.txt": b"gotcha",
    })
    identity = _identity(zip_path)
    malicious_path = "Takeout/..\\..\\..\\pwned.txt"

    with zipfile.ZipFile(zip_path, "r") as zf:
        real_open = zf.open
        monkeypatch.setattr(
            zf, "open",
            lambda name, mode="r": real_open(
                "payload.txt" if name == malicious_path else name, mode,
            ),
        )
        legit = [m for m in _members(zf) if m.path == "Takeout/archive_browser.html"]
        hostile = archive_mod.MemberInfo(
            path=malicious_path, size=6, crc32=0, kind=archive_mod.KIND_METADATA,
        )
        with caplog.at_level(logging.WARNING):
            written = provenance.preserve(
                organized, identity, zf, legit + [hostile], orphaned=set(),
            )

    room = organized / provenance.ROOM_NAME / identity.archive_id
    assert (room / "Takeout" / "archive_browser.html").read_bytes() == b"<html>viewer</html>"
    hostile_dest = room / provenance._safe_relpath(malicious_path)
    assert hostile_dest.resolve().is_relative_to(room.resolve())
    assert hostile_dest.read_bytes() == b"gotcha"
    assert written == 2  # both members preserved; neither escaped nor was lost
    assert not (tmp_path / "pwned.txt").exists()


def test_a_stored_path_escape_is_isolated_and_does_not_abort_the_archive(
    dirs, monkeypatch, caplog,
) -> None:
    # Defense-in-depth for the containment guard itself: if a future
    # sanitization gap ever let `_stored_path` compute a path outside the
    # room, `_stored_path` raises ValueError (see the guard added at the end
    # of `_stored_path`) rather than returning it -- this proves `preserve()`
    # catches that ValueError, logs it, and keeps going, exactly like an
    # unreadable or unwritable member, instead of letting one hostile name
    # abort preservation of the rest of the archive.
    archives, organized = dirs
    zip_path = _zip(archives / "t.zip", {
        "Takeout/archive_browser.html": b"<html>viewer</html>",
        "Takeout/evil.json": b'{"still": "gotcha"}',
    })
    identity = _identity(zip_path)

    real_stored_path = provenance._stored_path

    def _fake_stored_path(room, member, *, orphaned):
        if member.path.endswith("evil.json"):
            raise ValueError("simulated sanitization gap")
        return real_stored_path(room, member, orphaned=orphaned)

    monkeypatch.setattr(provenance, "_stored_path", _fake_stored_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with caplog.at_level(logging.WARNING):
            written = provenance.preserve(organized, identity, zf, _members(zf), orphaned=set())

    room = organized / provenance.ROOM_NAME / identity.archive_id
    assert (room / "Takeout" / "archive_browser.html").read_bytes() == b"<html>viewer</html>"
    assert not (room / "Takeout" / "evil.json").exists()
    assert written == 1
    assert any(
        "evil.json" in record.message or "outside its room" in record.message
        for record in caplog.records
    )


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
