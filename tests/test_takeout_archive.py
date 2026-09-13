"""Tests for Takeout archive identity, enumeration, and extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.takeout import archive


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


# --- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "member, expected",
    [
        ("Takeout/AlbumArchive/Hangouts/a/2015-03-09.jpg", archive.KIND_IMAGE),
        ("Takeout/x/PHOTO.JPG", archive.KIND_IMAGE),
        ("Takeout/x/clip.mp4", archive.KIND_VIDEO),
        ("Takeout/x/clip.MOV", archive.KIND_VIDEO),
        ("Takeout/x/2015-03-09.jpg.json", archive.KIND_METADATA),
        ("Takeout/x/a.jpg.supplemental-metadata.json", archive.KIND_METADATA),
        ("Takeout/x/Albums.json", archive.KIND_ALBUM),
        ("Takeout/x/metadata.json", archive.KIND_ALBUM),
        ("Takeout/x/archive_browser.html", archive.KIND_OTHER),
        ("Takeout/x/notes.txt", archive.KIND_OTHER),
        ("Takeout/x/noextension", archive.KIND_OTHER),
    ],
)
def test_classify(member, expected) -> None:
    assert archive.classify(member) == expected


def test_classification_is_service_agnostic() -> None:
    """The real export is AlbumArchive, not 'Google Photos'. Never key on a path."""
    assert archive.classify("Takeout/AlbumArchive/Hangouts/x/a.jpg") == archive.KIND_IMAGE
    assert archive.classify("Takeout/Google Photos/2015/a.jpg") == archive.KIND_IMAGE
    assert archive.classify("some/unheard/of/service/a.jpg") == archive.KIND_IMAGE


@pytest.mark.parametrize(
    "member, expected",
    [
        ("Takeout/Google Photos/Trash/a.jpg", True),
        ("Takeout/Google Photos/trash/a.jpg", True),
        ("Trash/a.jpg", True),
        ("Takeout/Google Photos/Trashy Album/a.jpg", False),
        ("Takeout/Google Photos/2015/a.jpg", False),
    ],
)
def test_is_trash(member, expected) -> None:
    assert archive.is_trash(member) is expected


# --- enumeration -----------------------------------------------------------


def test_iter_members_reads_only_the_central_directory(tmp_path: Path) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa", "d/a.jpg.json": b"{}", "d/": b""})
    with zipfile.ZipFile(z, "r") as zf:
        members = list(archive.iter_members(zf))
    paths = {m.path for m in members}
    assert paths == {"d/a.jpg", "d/a.jpg.json"}   # the directory entry is skipped
    by_path = {m.path: m for m in members}
    assert by_path["d/a.jpg"].size == 3
    assert by_path["d/a.jpg"].kind == archive.KIND_IMAGE
    assert by_path["d/a.jpg"].crc32 != 0


def test_iter_members_does_not_decompress(tmp_path: Path, monkeypatch) -> None:
    """Enumeration must be central-directory only, even on a huge archive."""
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    with zipfile.ZipFile(z, "r") as zf:
        def _boom(*args, **kwargs):
            raise AssertionError("iter_members must not open a member")

        monkeypatch.setattr(zf, "open", _boom)
        assert len(list(archive.iter_members(zf))) == 1


# --- identity --------------------------------------------------------------


def test_identify_hashes_on_a_miss(tmp_path: Path, catalog: Catalog) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    identity = archive.identify(z, catalog)
    assert len(identity.archive_id) == 43
    assert identity.size == z.stat().st_size


def test_identify_uses_the_stat_fast_path(tmp_path: Path, catalog: Catalog, monkeypatch) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    identity = archive.identify(z, catalog)
    catalog.takeout_archive_upsert(
        archive_id=identity.archive_id,
        last_path=str(z),
        size=identity.size,
        mtime_ns=identity.mtime_ns,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("the fast path must not re-hash the archive")

    monkeypatch.setattr(archive, "compute_sha256_b64url", _boom)
    again = archive.identify(z, catalog)
    assert again.archive_id == identity.archive_id


def test_a_renamed_archive_resolves_to_the_same_id(tmp_path: Path, catalog: Catalog) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"aaa"})
    first = archive.identify(z, catalog)
    catalog.takeout_archive_upsert(
        archive_id=first.archive_id, last_path=str(z), size=first.size,
        mtime_ns=first.mtime_ns,
    )
    renamed = tmp_path / "renamed.zip"
    z.rename(renamed)
    assert archive.identify(renamed, catalog).archive_id == first.archive_id


# --- extraction ------------------------------------------------------------


def test_extract_to_preserves_the_member_basename(tmp_path: Path) -> None:
    """Downstream date/descriptor resolution reads the staged file's NAME."""
    z = _zip(tmp_path / "t.zip", {"d/2015-03-09.jpg": b"bytes"})
    staging = tmp_path / "staging"
    with zipfile.ZipFile(z, "r") as zf:
        member = next(archive.iter_members(zf))
        staged = archive.extract_to(zf, member, staging)
        assert staged.name == "2015-03-09.jpg"
        assert staged.read_bytes() == b"bytes"
    archive.discard_staged(staged)
    assert not staged.exists()


def test_extract_to_isolates_colliding_basenames(tmp_path: Path) -> None:
    z = _zip(tmp_path / "t.zip", {"a/x.jpg": b"one", "b/x.jpg": b"two"})
    staging = tmp_path / "staging"
    with zipfile.ZipFile(z, "r") as zf:
        members = list(archive.iter_members(zf))
        first = archive.extract_to(zf, members[0], staging)
        second = archive.extract_to(zf, members[1], staging)
        assert first != second
        assert {first.read_bytes(), second.read_bytes()} == {b"one", b"two"}


def test_a_corrupted_member_raises_rather_than_yielding_bad_bytes(tmp_path: Path) -> None:
    """zipfile verifies CRC on a full read; a bad member must fail loudly."""
    z = _zip(tmp_path / "t.zip", {"d/a.jpg": b"a" * 200})
    raw = bytearray(z.read_bytes())
    # Corrupt the compressed payload without touching the central directory.
    raw[60:70] = b"\x00" * 10
    z.write_bytes(bytes(raw))

    with zipfile.ZipFile(z, "r") as zf:
        member = next(archive.iter_members(zf))
        with pytest.raises(zipfile.BadZipFile):
            archive.extract_to(zf, member, tmp_path / "staging")


def test_read_member_returns_bytes(tmp_path: Path) -> None:
    z = _zip(tmp_path / "t.zip", {"d/a.json": b'{"title": "x"}'})
    with zipfile.ZipFile(z, "r") as zf:
        assert archive.read_member(zf, "d/a.json") == b'{"title": "x"}'


def test_a_backslash_member_name_cannot_escape_the_staging_dir(
    tmp_path: Path, monkeypatch
) -> None:
    # On Windows pathlib treats "\" as a separator, so a member basename of
    # r"..\..\..\pwned.txt" handed to `holder / name` walks OUT of the
    # holder. The sanitizer must neutralize backslashes like any other
    # illegal character.
    #
    # CPython's own zipfile (`ZipInfo.__init__` -> `_sanitize_filename`)
    # already rewrites any backslash to "/" the moment a name round-trips
    # through a real archive on Windows, which would mask the very bug this
    # test exists to catch. `member.path` is attacker-controlled input to
    # `extract_to` regardless of how a member name reaches it (a non-Python
    # zip writer, a different platform, a future `iter_members` change), so
    # the malicious name is injected directly into the `MemberInfo` and
    # `zf.open` is patched to serve real bytes for it -- isolating exactly
    # what `extract_to` does with a hostile `member.path`.
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("payload.txt", b"gotcha")
    staging = tmp_path / "staging"
    malicious_path = "Takeout/..\\..\\..\\pwned.txt"
    with zipfile.ZipFile(zip_path) as zf:
        real_open = zf.open
        monkeypatch.setattr(
            zf, "open", lambda name, mode="r": real_open("payload.txt", mode)
        )
        member = archive.MemberInfo(
            path=malicious_path,
            size=6,
            crc32=0,
            kind=archive.KIND_IMAGE,
        )
        staged = archive.extract_to(zf, member, staging)
    staged_resolved = staged.resolve()
    assert staged_resolved.is_relative_to(staging.resolve())
    assert "\\" not in staged.name
    assert not (tmp_path / "pwned.txt").exists()
    assert not (tmp_path.parent / "pwned.txt").exists()


def test_sanitizer_replaces_backslashes() -> None:
    assert "\\" not in archive._safe_name("..\\..\\evil.jpg")
