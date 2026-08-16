"""Behavioral tests for Takeout ingestion.

Synthetic zips built in tmp_path replicate the real export's name shapes. No
79 MB fixture is committed -- the shapes are what actually matter.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.takeout import archive as archive_mod
from imageharbor.takeout import ingest as ingest_mod
from imageharbor.takeout.ingest import ingest_archives

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"
D = "Takeout/AlbumArchive/Hangouts/album"


def _jpeg(n: int) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes([n]) * 16 + b"\xff\xd9"


def _sidecar(title: str, seconds: int) -> bytes:
    return json.dumps(
        {
            "title": title,
            "creationTime": {"timestampSeconds": str(seconds + 14836)},
            "photoTakenTime": {"timestampSeconds": str(seconds)},
            "geoData": {"latitude": 38.2768361, "longitude": -85.7357389},
        }
    ).encode()


def _zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


@pytest.fixture()
def dirs(tmp_path: Path):
    archives = tmp_path / "archives"
    archives.mkdir()
    dest = tmp_path / "organized"
    dest.mkdir()
    return archives, dest


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


# --- the happy path --------------------------------------------------------


def test_ingests_a_photo_with_its_sidecar(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(
        archives / "takeout-001.zip",
        {
            f"{D}/2015-03-09.jpg": _jpeg(1),
            f"{D}/2015-03-09.jpg.json": _sidecar("2015-03-09.jpg", 1425905792),
        },
    )
    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.failed == 0
    assert stats.missing_metadata == 0
    organized = list((dest / "2015" / "2015-03").glob("*.jpg"))
    assert len(organized) == 1


def test_a_photo_without_a_sidecar_is_still_fully_organized(dirs, catalog: Catalog) -> None:
    """No Google metadata is not a failure. It is a photo dated from itself."""
    archives, dest = dirs
    _zip(archives / "t.zip", {f"{D}/IMG_9999.jpg": _jpeg(2)})
    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.missing_metadata == 1
    assert len(list((dest / "Undated").glob("*.jpg"))) == 1


# --- idempotency and resume ------------------------------------------------


def test_a_second_run_extracts_zero_members(dirs, catalog: Catalog, monkeypatch) -> None:
    """Asserted by counting extractions, not by timing."""
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            f"{D}/a.jpg": _jpeg(3),
            f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
            f"{D}/b.jpg": _jpeg(4),
        },
    )
    ingest_archives(archives, dest, catalog)

    calls = []
    real = archive_mod.extract_to
    monkeypatch.setattr(
        archive_mod, "extract_to",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    stats = ingest_archives(archives, dest, catalog)

    assert calls == []
    assert stats.archives_skipped == 1
    assert stats.ingested == 0


def test_resume_after_a_mid_archive_crash_processes_only_the_remainder(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {f"{D}/a.jpg": _jpeg(5), f"{D}/b.jpg": _jpeg(6), f"{D}/c.jpg": _jpeg(7)},
    )

    real = archive_mod.extract_to
    seen: list[str] = []

    def _crash_on_third(zf, member, staging):
        seen.append(member.path)
        if len(seen) == 3:
            raise KeyboardInterrupt("simulated kill -9")
        return real(zf, member, staging)

    monkeypatch.setattr(archive_mod, "extract_to", _crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        ingest_archives(archives, dest, catalog)

    monkeypatch.setattr(archive_mod, "extract_to", real)
    extracted: list[str] = []
    monkeypatch.setattr(
        archive_mod, "extract_to",
        lambda zf, m, s: (extracted.append(m.path), real(zf, m, s))[1],
    )
    stats = ingest_archives(archives, dest, catalog)

    assert extracted == [f"{D}/c.jpg"]
    assert stats.ingested == 1


def test_a_corrupt_archive_does_not_stop_its_neighbours(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    (archives / "broken.zip").write_bytes(b"this is not a zip file")
    _zip(archives / "good.zip", {f"{D}/a.jpg": _jpeg(8)})

    stats = ingest_archives(archives, dest, catalog)

    assert stats.archives_corrupt == 1
    assert stats.ingested == 1


# --- videos, trash, duplicates ---------------------------------------------


def test_videos_are_deferred_with_a_date_and_no_bytes_copied(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            f"{D}/clip.mp4": b"not really an mp4",
            f"{D}/clip.mp4.json": _sidecar("clip.mp4", 1425905792),
        },
    )
    stats = ingest_archives(archives, dest, catalog)

    assert stats.deferred == 1
    assert stats.ingested == 0
    assert list(dest.rglob("*.mp4")) == []

    identity = catalog.takeout_archives_all()[0]["archive_id"]
    row = [m for m in catalog.takeout_members_all(identity) if m["kind"] == "video"][0]
    assert row["status"] == "deferred"
    assert row["taken_at"].startswith("2015-03-09")


def test_a_video_failure_does_not_abort_the_batch(dirs, catalog: Catalog, monkeypatch) -> None:
    """A video-processing failure must fail only that member, not the batch.

    `_defer_video` sits outside the image loop's isolation, so an unwrapped
    exception there would previously abort the whole archive. The injection
    point (`pairing.sidecar_for`) is also called from the image path, so the
    fake only raises for the video's member path -- this fixture's image has
    no sidecar of its own, so its (real) call to `sidecar_for` must keep
    succeeding for this test to prove isolation rather than a blanket outage.
    """
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            f"{D}/photo.jpg": _jpeg(30),
            f"{D}/clip.mp4": b"not really an mp4",
        },
    )

    real_sidecar_for = ingest_mod.pairing.sidecar_for

    def _boom(member_path, index):
        if member_path.endswith(".mp4"):
            raise RuntimeError("simulated video-path failure")
        return real_sidecar_for(member_path, index)

    monkeypatch.setattr(ingest_mod.pairing, "sidecar_for", _boom)

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.deferred == 0
    assert stats.failed == 1
    assert len(list((dest / "Undated").glob("*.jpg"))) == 1

    identity = catalog.takeout_archives_all()[0]["archive_id"]
    rows = {m["member_path"]: m for m in catalog.takeout_members_all(identity)}
    assert rows[f"{D}/clip.mp4"]["status"] == "failed"


def test_trash_is_enumerated_but_not_ingested(dirs, catalog: Catalog) -> None:
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {"Takeout/Google Photos/Trash/deleted.jpg": _jpeg(9), f"{D}/kept.jpg": _jpeg(10)},
    )
    stats = ingest_archives(archives, dest, catalog)

    assert stats.skipped_trash == 1
    assert stats.ingested == 1
    identity = catalog.takeout_archives_all()[0]["archive_id"]
    statuses = {m["member_path"]: m["status"] for m in catalog.takeout_members_all(identity)}
    assert statuses["Takeout/Google Photos/Trash/deleted.jpg"] == "skipped_trash"


def test_include_trash_ingests_previously_skipped_trash(dirs, catalog: Catalog) -> None:
    """A user who changes their mind must not be blocked by a terminal status."""
    archives, dest = dirs
    _zip(archives / "t.zip", {"Takeout/Google Photos/Trash/deleted.jpg": _jpeg(11)})

    ingest_archives(archives, dest, catalog)
    assert list(dest.rglob("*.jpg")) == []

    stats = ingest_archives(archives, dest, catalog, include_trash=True)
    assert stats.ingested == 1
    assert len(list(dest.rglob("*.jpg"))) == 1


def test_the_same_photo_in_two_archives_yields_one_file_and_two_sources(
    dirs, catalog: Catalog
) -> None:
    archives, dest = dirs
    _zip(archives / "t1.zip", {f"{D}/a.jpg": _jpeg(12)})
    _zip(archives / "t2.zip", {f"{D}/copy-of-a.jpg": _jpeg(12)})

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.duplicates == 1
    assert len(list(dest.rglob("*.jpg"))) == 1
    # catalog.iter_all() returns an iterator, not a list.
    row = next(iter(catalog.iter_all()))
    assert len(catalog.sources_for(row["sha256_b64url"])) == 2
    assert all("!" in r["source_path"] for r in catalog.sources_for(row["sha256_b64url"]))


# --- the late-sidecar case: the heart of the design ------------------------


def test_a_complete_archive_is_reopened_only_when_metadata_actually_arrives(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    """The reopen is targeted, not a blanket re-scan of every complete archive.

    This is what bounds the cost of the second survey pass. An archive whose
    members gained nothing must stay skipped and extract nothing -- otherwise
    every run would re-extract the entire library, and the four idempotency
    layers would be decorative.
    """
    archives, dest = dirs
    _zip(
        archives / "paired.zip",
        {
            f"{D}/a.jpg": _jpeg(20),
            f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
        },
    )
    # No sidecar exists for this one anywhere, so it can never become pairable.
    _zip(archives / "unpaired.zip", {f"{D}/b.jpg": _jpeg(21)})

    first = ingest_archives(archives, dest, catalog)
    assert first.ingested == 2
    assert first.archives_reopened == 0

    calls: list[str] = []
    real = archive_mod.extract_to
    monkeypatch.setattr(
        archive_mod, "extract_to",
        lambda zf, m, s: (calls.append(m.path), real(zf, m, s))[1],
    )
    second = ingest_archives(archives, dest, catalog)

    assert calls == []                      # nothing re-extracted
    assert second.archives_reopened == 0
    assert second.archives_skipped == 2
    assert second.ingested == 0


def test_a_sidecar_arriving_in_a_later_part_relocates_the_photo(
    dirs, catalog: Catalog
) -> None:
    """Google splits by size, so a photo and its sidecar land in different parts.

    Ingest part 1 alone -> Undated/. Add part 2 carrying the sidecar, re-run,
    and the existing monotonic-upgrade machinery relocates the file.
    """
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {f"{D}/IMG_1234.jpg": _jpeg(13)})

    ingest_archives(archives, dest, catalog)
    assert len(list((dest / "Undated").glob("*.jpg"))) == 1

    _zip(
        archives / "takeout-002.zip",
        {f"{D}/IMG_1234.jpg.json": _sidecar("IMG_1234.jpg", 1425905792)},
    )
    second = ingest_archives(archives, dest, catalog)

    # The stat must report that the reopen mechanism actually fired -- the
    # relocation below proves the effect, this proves the cause.
    assert second.archives_reopened == 1

    assert list((dest / "Undated").glob("*.jpg")) == []
    assert len(list((dest / "2015" / "2015-03").glob("*.jpg"))) == 1


def test_a_failed_retry_keeps_what_the_member_already_knew(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    """A reopened member's failed retry must not null out what the reopen preserved.

    `takeout_member_set` is a blind full-row UPDATE. The second survey pass
    carries `sha256_b64url` forward when it resets a stale member to
    `pending`; if a failure branch then wrote only `status`/`last_error`, that
    carried-forward digest would be wiped out by the very retry meant to
    upgrade it. This pins `_mark_failed` against that regression.
    """
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {f"{D}/IMG_1234.jpg": _jpeg(14)})

    ingest_archives(archives, dest, catalog)

    identity = catalog.takeout_archives_all()[0]["archive_id"]
    first_row = [
        m for m in catalog.takeout_members_all(identity)
        if m["member_path"] == f"{D}/IMG_1234.jpg"
    ][0]
    assert first_row["sha256_b64url"] is not None
    original_digest = first_row["sha256_b64url"]

    _zip(
        archives / "takeout-002.zip",
        {f"{D}/IMG_1234.jpg.json": _sidecar("IMG_1234.jpg", 1425905792)},
    )

    real = archive_mod.extract_to

    def _boom(zf, member, staging):
        if member.path == f"{D}/IMG_1234.jpg":
            raise RuntimeError("simulated retry failure")
        return real(zf, member, staging)

    monkeypatch.setattr(archive_mod, "extract_to", _boom)

    stats = ingest_archives(archives, dest, catalog)

    assert stats.archives_reopened == 1
    assert stats.failed == 1

    retried_row = [
        m for m in catalog.takeout_members_all(identity)
        if m["member_path"] == f"{D}/IMG_1234.jpg"
    ][0]
    assert retried_row["status"] == "failed"
    assert retried_row["sha256_b64url"] == original_digest
