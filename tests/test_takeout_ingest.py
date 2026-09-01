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


def _sidecar(title: str, seconds: int, people: tuple[str, ...] = ()) -> bytes:
    doc = {
        "title": title,
        "creationTime": {"timestampSeconds": str(seconds + 14836)},
        "photoTakenTime": {"timestampSeconds": str(seconds)},
        "geoData": {"latitude": 38.2768361, "longitude": -85.7357389},
    }
    if people:
        doc["people"] = [{"name": n} for n in people]
    return json.dumps(doc).encode()


def _read_sidecar(dest: Path, stem_contains: str) -> dict:
    """The JSON sidecar ImageHarbor wrote beside the one organized file whose
    name contains *stem_contains*.

    Two departures from a naive `dest.rglob("*.json")` were needed to make
    this find the right file:

    - Case-insensitive: `normalize_descriptor` lowercases every descriptor
      (and folds `_` into `-`), so an organized filename never contains a
      member's original mixed-case stem verbatim.
    - Excludes the provenance room (`.takeout-provenance/`): `_ingest_archive`
      preserves every non-media member verbatim there, under its ORIGINAL
      member name, regardless of `write_sidecars` -- so a photo's own Google
      JSON sidecar (e.g. "IMG_1.jpg.json") sits there too, an unrelated file
      that can share the same substring as the organized sidecar under test.
    """
    from imageharbor.takeout.provenance import ROOM_NAME

    needle = stem_contains.lower()
    hits = [
        p for p in dest.rglob("*.json")
        if ROOM_NAME not in p.parts and needle in p.name.lower()
    ]
    assert len(hits) == 1, [str(p.relative_to(dest)) for p in hits]
    return json.loads(hits[0].read_text(encoding="utf-8"))


def _make_stale_index(path: Path, *, name: str, size: int, mtime: int) -> Path:
    """A minimal Takeout_Inventory index describing one archive whose stats
    do not match what's actually on disk, so `covers()` refuses it and the
    ingest falls back to the built-in pairing for that archive.

    Built from the same schema literal `test_takeout_index_reader.py` uses --
    that module owns the schema, so the SQL is not duplicated here.
    """
    from tests.test_takeout_index_reader import make_index

    return make_index(path, archives=((name, size, mtime, 0, None),))


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


def test_include_trash_run_reaches_complete_and_stays_skipped_after(
    dirs, catalog: Catalog
) -> None:
    """A trash sidecar must not strand the archive in 'partial' forever.

    `takeout_members_unskip_trash` restores a trash image to 'pending' (a
    work item) but a trash metadata row to 'parsed' -- if it were reset to
    'pending' instead, nothing would ever drain it and the archive could
    never reach 'complete', so every future run would re-examine it and
    `takeout status` would permanently misreport pending work.
    """
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            "Takeout/Google Photos/Trash/deleted.jpg": _jpeg(12),
            "Takeout/Google Photos/Trash/deleted.jpg.json": _sidecar("deleted.jpg", 1425905792),
        },
    )

    ingest_archives(archives, dest, catalog)
    identity = catalog.takeout_archives_all()[0]["archive_id"]
    assert catalog.takeout_archive_get(identity)["status"] == "complete"

    second = ingest_archives(archives, dest, catalog, include_trash=True)
    assert second.ingested == 1
    assert catalog.takeout_archive_get(identity)["status"] == "complete"

    third = ingest_archives(archives, dest, catalog)
    assert third.ingested == 0
    assert third.archives_skipped == 1
    assert third.archives_reopened == 0


def test_trash_is_not_resurrected_by_the_reopen_pass(dirs, catalog: Catalog) -> None:
    """A second run of the SAME command must not ingest what the first skipped.

    Real exports give trash items sidecars, so a trash member is pairable --
    which made it eligible for the reopen pass until the status whitelist was
    added. The bug poured the entire deleted-photos tree into the library on
    the operator's second invocation.
    """
    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            "Takeout/Google Photos/Trash/deleted.jpg": _jpeg(30),
            "Takeout/Google Photos/Trash/deleted.jpg.json": _sidecar("deleted.jpg", 1425905792),
        },
    )

    first = ingest_archives(archives, dest, catalog)
    assert first.ingested == 0
    assert list(dest.rglob("*.jpg")) == []

    second = ingest_archives(archives, dest, catalog)
    assert second.ingested == 0
    assert second.archives_reopened == 0
    assert list(dest.rglob("*.jpg")) == []

    # And --include-trash still works, so this is a targeted exclusion.
    third = ingest_archives(archives, dest, catalog, include_trash=True)
    assert third.ingested == 1
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


def test_the_same_archive_at_two_paths_does_not_defeat_pairing(
    dirs, catalog: Catalog
) -> None:
    """A kept re-download must not collapse the batch's pairing.

    `ambiguous_media` exists to refuse a pairing when two DIFFERENT archives
    share a member path. It cannot see archive identity, so without a
    seen-identity guard in the survey, one archive listed twice looks exactly
    like that case and every photo in the batch loses its Google date.
    """
    import shutil

    archives, dest = dirs
    original = _zip(
        archives / "takeout-001.zip",
        {
            f"{D}/vacation.jpg": _jpeg(40),
            f"{D}/vacation.jpg.json": _sidecar("vacation.jpg", 1425905792),
        },
    )
    shutil.copy(original, archives / "takeout-001 (1).zip")

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.missing_metadata == 0
    assert len(list((dest / "2015" / "2015-03").glob("*.jpg"))) == 1
    assert list((dest / "Undated").glob("*.jpg")) == []


def test_two_different_archives_sharing_a_member_path_still_declines_pairing(
    dirs, catalog: Catalog
) -> None:
    """Two DIFFERENT archives that happen to share a member path is the real
    case `ambiguous_media` protects against, and the seen-identity guard added
    for the kept-re-download regression must not weaken it: these two zips
    have different bytes, so they get different `archive_id`s and are both
    surveyed and both contribute their member path to the index.
    """
    archives, dest = dirs
    _zip(
        archives / "takeout-001.zip",
        {
            f"{D}/dup.jpg": _jpeg(50),
            f"{D}/dup.jpg.json": _sidecar("dup.jpg", 1425905792),
        },
    )
    _zip(archives / "takeout-002.zip", {f"{D}/dup.jpg": _jpeg(51)})

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 2
    assert stats.missing_metadata == 2
    assert list((dest / "2015" / "2015-03").glob("*.jpg")) == []
    assert len(list((dest / "Undated").glob("*.jpg"))) == 2


def test_a_sidecar_shared_by_two_archives_does_not_misdate_a_photo(
    dirs, catalog: Catalog
) -> None:
    """The sidecar-side twin of the two tests above.

    Archive A holds a photo and its correct 2015 sidecar. Archive B is a
    different export that happens to carry only a sidecar at the SAME member
    path, dated 2019. The photo exists only in archive A. `self.owner` (the
    ingest layer's member-path -> owning-archive map) is last-writer-wins, so
    without `ambiguous_sidecars` the index would keep one sidecar entry and
    the photo could silently be dated from archive B's 2019 metadata, which
    does not describe it. It must instead land in Undated/ with its Google
    metadata reported missing -- costing only the metadata, never a wrong
    date.
    """
    archives, dest = dirs
    _zip(
        archives / "takeout-001.zip",
        {
            f"{D}/IMG_1234.jpg": _jpeg(60),
            f"{D}/IMG_1234.jpg.json": _sidecar("IMG_1234.jpg", 1425905792),  # 2015-03-09
        },
    )
    _zip(
        archives / "takeout-002.zip",
        {f"{D}/IMG_1234.jpg.json": _sidecar("IMG_1234.jpg", 1562252400)},  # 2019-07-04
    )

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1
    assert stats.missing_metadata == 1
    assert list((dest / "2019" / "2019-07").glob("*.jpg")) == []
    assert list((dest / "2015" / "2015-03").glob("*.jpg")) == []
    assert len(list((dest / "Undated").glob("*.jpg"))) == 1


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


def test_a_late_sidecar_relocation_still_records_takeout_provenance(
    dirs, catalog: Catalog
) -> None:
    """The late-sidecar reopen is a DUPLICATE branch, which must not lose the sidecar.

    `Pipeline.process_file` leaves `result.organized_path` None on the
    duplicate branch -- the file already lives wherever the first ingest put
    it. `_merge_takeout_sidecar` was guarded on that field being non-None, so
    the branch that is this module's headline case (a sidecar arriving in a
    later part, relocating an already-organized photo) relocated the bytes
    but silently never wrote the sidecar block explaining why.

    Updated for Task 5: the flat, unversioned `takeout` key is gone --
    `_merge_takeout_sidecar` now writes `provenance`/`albums` blocks the
    schema actually knows how to merge. The first ingest has no sidecar at
    all (no bytes to digest, so no provenance entry); the second, reopened
    ingest is the one that actually observes a Google document, and this
    pins that its provenance entry is not silently lost.
    """
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {f"{D}/IMG_1234.jpg": _jpeg(15)})

    ingest_archives(archives, dest, catalog, write_sidecars=True)

    _zip(
        archives / "takeout-002.zip",
        {f"{D}/IMG_1234.jpg.json": _sidecar("IMG_1234.jpg", 1425905792)},
    )
    second = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert second.archives_reopened == 1

    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    sidecar = organized.with_name(f"{organized.stem}.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    entry = next(p for p in payload["provenance"] if p["kind"] == "takeout_media_json")
    assert entry["archive"] == "takeout-001.zip"
    assert entry["member"] == f"{D}/IMG_1234.jpg.json"
    assert entry["digest"]
    assert entry["raw"]["title"] == "IMG_1234.jpg"


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


# --- Task 5: raw document preservation and real album titles ---------------


def test_sidecar_preserves_googles_document_verbatim(dirs, catalog: Catalog) -> None:
    """Fields nobody modelled -- imageViews, height, width -- survive here.

    They come back without being parsed, and so will anything Google adds
    later, which is the whole reason the raw document is kept.
    """
    archives, dest = dirs
    payload = {
        "title": "a.jpg",
        "imageViews": "12",
        "height": "2432", "width": "4320",
        "photoTakenTime": {"timestampSeconds": "1425905792"},
        "someFieldFromTheFuture": {"nested": [1, 2, 3]},
    }
    _zip(archives / "t.zip", {f"{D}/a.jpg": _jpeg(60),
                              f"{D}/a.jpg.json": json.dumps(payload).encode()})

    ingest_archives(archives, dest, catalog, write_sidecars=True)

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    entry = next(p for p in read_sidecar(organized)["provenance"]
                 if p["kind"] == "takeout_media_json")
    assert entry["raw"] == payload


def test_sidecar_records_the_real_album_title(dirs, catalog: Catalog) -> None:
    """The directory name is not the album name."""
    archives, dest = dirs
    _zip(archives / "t.zip", {
        f"{D}/a.jpg": _jpeg(61),
        f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
        f"{D}/Albums.json": json.dumps({"title": "Hangout: Emma ● Sam",
                                        "access": "protected"}).encode(),
    })
    ingest_archives(archives, dest, catalog, write_sidecars=True)

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    album = read_sidecar(organized)["albums"][0]
    assert album["title"] == "Hangout: Emma ● Sam"
    assert album["access"] == "protected"
    assert album["folder"] == D.rsplit("/", 1)[-1]


def test_a_photo_in_two_archives_accumulates_both_albums(dirs, catalog: Catalog) -> None:
    """Duplicates stop being waste and become context."""
    archives, dest = dirs
    img = _jpeg(62)
    _zip(archives / "one.zip", {"Takeout/A/Album One/a.jpg": img,
                                "Takeout/A/Album One/Albums.json": json.dumps({"title": "One"}).encode()})
    _zip(archives / "two.zip", {"Takeout/A/Album Two/b.jpg": img,
                                "Takeout/A/Album Two/Albums.json": json.dumps({"title": "Two"}).encode()})

    stats = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert stats.ingested == 1
    assert stats.duplicates == 1

    from imageharbor.sidecar import read_sidecar
    organized = next(dest.rglob("*.jpg"))
    titles = {a["title"] for a in read_sidecar(organized)["albums"]}
    assert titles == {"One", "Two"}


def test_album_metadata_lookup_never_fails_a_photo_when_albums_json_is_malformed(
    dirs, catalog: Catalog
) -> None:
    """A corrupt Albums.json degrades to 'no title', never fails the photo."""
    archives, dest = dirs
    _zip(archives / "t.zip", {
        f"{D}/a.jpg": _jpeg(63),
        f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
        f"{D}/Albums.json": b"{not json",
    })
    stats = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert stats.ingested == 1
    assert stats.failed == 0

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    album = read_sidecar(organized)["albums"][0]
    assert album["title"] is None
    assert album["folder"] == D.rsplit("/", 1)[-1]


def test_a_sidecar_member_whose_bytes_are_not_json_still_records_provenance(
    dirs, catalog: Catalog
) -> None:
    """`raw` is omitted, but the fact a document existed is not lost."""
    archives, dest = dirs
    _zip(archives / "t.zip", {
        f"{D}/a.jpg": _jpeg(64),
        f"{D}/a.jpg.json": b"not json at all",
    })
    stats = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert stats.ingested == 1
    assert stats.failed == 0

    from imageharbor.sidecar import read_sidecar
    # No parseable photoTakenTime -- the photo organizes into Undated/, same
    # as one with no sidecar at all. The point under test is that the
    # sidecar document's existence (digest) is still recorded even though it
    # failed to parse.
    organized = next((dest / "Undated").glob("*.jpg"))
    entry = next(p for p in read_sidecar(organized)["provenance"]
                 if p["kind"] == "takeout_media_json")
    assert "raw" not in entry
    assert entry["digest"]


def test_a_deleted_provenance_room_is_recreated_by_re_ingesting_a_complete_archive(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    """Finding 6: the provenance room is only ever created by
    `_ingest_archive` reopening the zip, which a `complete` archive with no
    stale sidecar work never reaches -- so a room deleted by hand (or never
    finished) stayed gone forever, and re-ingesting -- the documented
    recovery -- silently did nothing.

    `_survey` must now detect a missing manifest for a `complete` archive
    that has any non-media member and put it back in `todo`, WITHOUT
    resetting any member to `pending` -- so the room comes back but not one
    byte of media is re-extracted.
    """
    import shutil

    from imageharbor.takeout import provenance

    archives, dest = dirs
    _zip(
        archives / "t.zip",
        {
            f"{D}/a.jpg": _jpeg(70),
            f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
        },
    )

    first = ingest_archives(archives, dest, catalog)
    assert first.ingested == 1

    identity = catalog.takeout_archives_all()[0]["archive_id"]
    room = dest / provenance.ROOM_NAME / identity
    manifest = provenance.manifest_path(dest, identity)
    assert room.exists()
    assert manifest.exists()

    shutil.rmtree(room)
    assert not room.exists()

    calls: list[str] = []
    real = archive_mod.extract_to
    monkeypatch.setattr(
        archive_mod, "extract_to",
        lambda zf, m, s: (calls.append(m.path), real(zf, m, s))[1],
    )

    second = ingest_archives(archives, dest, catalog)

    assert calls == [], "re-ingesting a complete archive must not re-extract any member"
    assert second.ingested == 0
    assert second.duplicates == 0
    assert len(list(dest.rglob("*.jpg"))) == 1, "no photo was re-copied"
    assert room.exists(), "the provenance room must be recreated"
    assert manifest.exists(), "the manifest must be rewritten"
    assert any(room.rglob("a.jpg.json")), "the preserved document must be back"
    assert catalog.takeout_archive_get(identity)["status"] == "complete"


def test_a_photo_with_no_albums_json_in_its_directory_still_organizes(
    dirs, catalog: Catalog
) -> None:
    archives, dest = dirs
    _zip(archives / "t.zip", {
        f"{D}/a.jpg": _jpeg(65),
        f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
    })
    stats = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert stats.ingested == 1
    assert stats.failed == 0

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    data = read_sidecar(organized)
    assert data["albums"][0]["title"] is None


# --- Task 5: routing through the index, and the `related` policy -----------


def test_related_pairing_keeps_the_date_and_drops_title_and_people(
    dirs, catalog: Catalog
) -> None:
    """An -edited copy inherits its ORIGINAL's sidecar. The capture instant is
    this photograph's; the title and the people are the original's."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1-edited.jpg": _jpeg(2),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792,
                                        people=("Alice",)),
    })
    # `_read_sidecar` reads the JSON sidecar ingest writes beside the
    # organized file, which only happens with `write_sidecars=True` -- the
    # brief's own snippet omits this; every other provenance-block test in
    # this module passes it explicitly, and without it there is no file for
    # `_read_sidecar` to find.
    ingest_archives(archives, dest, catalog, write_sidecars=True)

    # "IMG_1" alone is a recognized camera pattern (see `descriptor.py`) and
    # is discarded, so it never survives into an organized filename; "edited"
    # does (the `-edited` suffix defeats the pattern), which is also exactly
    # the substring that distinguishes this file's sidecar from IMG_1.jpg's.
    edited = _read_sidecar(dest, "edited")
    prov = edited["provenance"][0]
    assert prov["confidence"] == "related"
    assert prov["pair_rule"]                       # recorded, never blank
    # The document is kept verbatim - deleting it would destroy the audit
    # trail - but it is labelled, so the coordinates in it are not silently
    # this photo's.
    assert prov["raw"]["geoData"]["latitude"] == 38.2768361
    assert "people" not in edited


def test_own_pairing_keeps_title_and_people(dirs, catalog: Catalog) -> None:
    """Fix pass 1, CRITICAL 2 row 4: the given fixture (`IMG_1.jpg`, a
    recognized camera pattern -- see `descriptor.py`) cannot actually prove
    the title survives, because `resolve_descriptor` discards a camera-
    generated verdict from EITHER the media's own stem OR the sidecar's
    title, so the descriptor comes out DESC_NONE whether `original_name` is
    threaded through or dropped entirely -- the `original_name=None` mutation
    is invisible against that fixture. The media filename here is changed to
    a non-camera-generated stem ("photo1"), and the sidecar's title to a
    different, also non-camera-generated human title ("Emma Birthday.jpg"),
    so the two are only equal if the title actually reaches the descriptor
    ladder -- the added assertion below fails if `original_name` is dropped,
    because the descriptor would then fall back to the media's own stem
    ("photo1") instead of the title's ("emma-birthday")."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/photo1.jpg": _jpeg(1),
        f"{D}/photo1.jpg.json": _sidecar("Emma Birthday.jpg", 1425905792,
                                          people=("Alice",)),
    })
    ingest_archives(archives, dest, catalog, write_sidecars=True)

    # This archive organizes exactly one photo, so the empty needle (matches
    # everything) still finds exactly one sidecar.
    own = _read_sidecar(dest, "")
    assert own["provenance"][0]["confidence"] == "own"
    assert own["people"] == [{"name": "Alice", "source": "google_photos_people"}]

    # The added assertion: the organized filename's descriptor came from the
    # sidecar's title ("Emma Birthday" -> "emma-birthday"), not from the
    # media's own filename ("photo1") -- proving `original_name` really
    # reached the descriptor ladder for an `own` pairing.
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    assert "emma-birthday" in organized.name
    assert "photo1" not in organized.name


def test_an_uncovered_archive_falls_back_and_is_counted(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """A stale index must never fail an ingest, and never be silent about it."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792),
    })
    # An index describing an archive with the right name and the wrong size.
    stale = _make_stale_index(tmp_path / "takeout-index.sqlite",
                              name="takeout-001.zip", size=1, mtime=1)
    stats = ingest_archives(archives, dest, catalog, index_path=stale)

    assert stats.index_archives_covered == 0
    assert stats.index_archives_fell_back == 1
    assert stats.ingested == 1        # identical to a no-index run
    assert stats.missing_metadata == 0


# --- Fix pass 1: CRITICAL 1 and CRITICAL 2 regression tests ----------------


def test_related_pairing_records_date_tier_25(dirs, catalog: Catalog) -> None:
    """Fix pass 1, CRITICAL 2 row 1: a `related` pairing's capture date must
    be resolved at DATE_RELATED_SIDECAR (25) / "related_sidecar", never at
    DATE_EXTERNAL_SIDECAR (30) -- the tier an `own` pairing gets. Kills the
    mutation that hardcodes `date_tier=tiers.DATE_EXTERNAL_SIDECAR` always."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1-edited.jpg": _jpeg(2),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792,
                                        people=("Alice",)),
    })
    ingest_archives(archives, dest, catalog)

    from imageharbor.hashing import compute_sha256_b64url_bytes
    sha = compute_sha256_b64url_bytes(_jpeg(2))
    row = catalog.get_by_sha256(sha)
    assert row is not None
    assert row["date_tier"] == 25
    assert row["date_source"] == "related_sidecar"


def _index_covering_weird1(tmp_path: Path, archives: Path) -> Path:
    """A Takeout_Inventory index that pairs `weird1.jpg` with a same-archive
    sidecar whose name is unrelated to it -- something the built-in ladder's
    naming rungs (exact, case-insensitive, truncation-prefix) cannot ever
    associate -- stamped with a distinctive rule name so `pair_rule`
    provenance can be checked for its actual value, not merely truthiness."""
    from tests.test_takeout_index_reader import make_index

    st = (archives / "takeout-001.zip").stat()
    return make_index(
        tmp_path / "index.sqlite",
        archives=(("takeout-001.zip", st.st_size, int(st.st_mtime), 0, None),),
        sidecars=[(1, "takeout-001.zip", f"{D}/totally-unrelated-name.json",
                   "totally-unrelated-name.json")],
        media=[("takeout-001.zip", f"{D}/weird1.jpg", "area", "folder",
                "weird1.jpg", 1, "index-only-rule", "own")],
    )


def test_index_supplied_pairing_differs_from_builtin_result(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """Fix pass 1, CRITICAL 2 row 2: a member the built-in ladder cannot pair
    (an unrelated sidecar name) is still paired when a Takeout_Inventory
    index supplies the answer -- proving the index is genuinely consulted,
    not merely present. Kills the mutation `if False and self.index...`."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/weird1.jpg": _jpeg(3),
        f"{D}/totally-unrelated-name.json": _sidecar("weird1.jpg", 1425905792),
    })

    # Baseline: the built-in ladder cannot associate these two unrelated names.
    baseline = ingest_archives(archives, dest, catalog, write_sidecars=True)
    assert baseline.missing_metadata == 1

    idx_path = _index_covering_weird1(tmp_path, archives)
    dest2 = tmp_path / "organized2"
    dest2.mkdir()
    cat2 = Catalog(tmp_path / "catalog2.db")
    try:
        stats = ingest_archives(archives, dest2, cat2, write_sidecars=True,
                                index_path=idx_path)
    finally:
        cat2.close()

    assert stats.index_archives_covered == 1
    # The observable difference from the built-in-only baseline: the index
    # supplied a pairing where the built-in ladder found none.
    assert stats.missing_metadata == 0


def test_index_pair_rule_is_recorded_verbatim(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """Fix pass 1, CRITICAL 2 row 3: the provenance entry's `pair_rule` must
    equal the index's own rule string, not merely be truthy. Kills the
    mutation that hardcodes `"pair_rule": "builtin"`."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/weird1.jpg": _jpeg(3),
        f"{D}/totally-unrelated-name.json": _sidecar("weird1.jpg", 1425905792),
    })
    idx_path = _index_covering_weird1(tmp_path, archives)
    stats = ingest_archives(archives, dest, catalog, write_sidecars=True,
                            index_path=idx_path)
    assert stats.index_archives_covered == 1

    from imageharbor.sidecar import read_sidecar
    organized = next((dest / "2015" / "2015-03").glob("*.jpg"))
    prov = read_sidecar(organized)["provenance"][0]
    assert prov["pair_rule"] == "index-only-rule"


def test_index_only_paired_sidecar_is_not_filed_as_an_orphan(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """I2: `_survey`'s claimed-sidecar accounting -- the set that gates
    `_preserve_provenance`'s orphaned/ bucket -- must route through the
    index too, not just the built-in ladder.

    The sidecar here (`some-other-photo.jpg.json`) is deliberately
    media-sidecar-SHAPED (`_looks_like_media_sidecar` requires its stem to
    classify as an image/video, or it is never even a candidate for
    orphaned/) but names a file that shares nothing with `weird1.jpg` --
    the built-in ladder's naming rungs (exact, case-insensitive,
    truncation-prefix) can never associate the two, only the index can (see
    `_index_covering_weird1`'s pattern). Without this fix that sidecar is
    filed under orphaned/ even though the index-only pairing DOES claim
    it -- overstating the residue that has to stay honest.
    """
    from tests.test_takeout_index_reader import make_index

    archives, dest = dirs
    sidecar_member = f"{D}/some-other-photo.jpg.json"
    _zip(archives / "takeout-001.zip", {
        f"{D}/weird1.jpg": _jpeg(3),
        sidecar_member: _sidecar("weird1.jpg", 1425905792),
    })
    st = (archives / "takeout-001.zip").stat()
    idx_path = make_index(
        tmp_path / "index.sqlite",
        archives=(("takeout-001.zip", st.st_size, int(st.st_mtime), 0, None),),
        sidecars=[(1, "takeout-001.zip", sidecar_member, "some-other-photo.jpg.json")],
        media=[("takeout-001.zip", f"{D}/weird1.jpg", "area", "folder",
                "weird1.jpg", 1, "index-only-rule", "own")],
    )

    stats = ingest_archives(archives, dest, catalog, index_path=idx_path)
    assert stats.index_archives_covered == 1

    from imageharbor.takeout import provenance

    identity = catalog.takeout_archives_all()[0]["archive_id"]
    room = dest / provenance.ROOM_NAME / identity
    orphaned = room / "orphaned" / "some-other-photo.jpg.json"
    claimed = room / D / "some-other-photo.jpg.json"
    assert not orphaned.exists(), "an index-only-paired sidecar must not be orphaned"
    assert claimed.exists(), "it must be preserved at its normal member path instead"


def test_index_null_sidecar_falls_back_to_builtin_pairing(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """Fix pass 1, CRITICAL 1: an index that covers the archive and knows the
    member, but reports NO sidecar for it (`sidecar_id` NULL), must fall
    through to the built-in ladder rather than silently overriding a pairing
    the built-in ladder CAN make. The built-in pairing is always a correct
    answer; every index problem falls back, counts, and reports."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792),
    })
    st = (archives / "takeout-001.zip").stat()
    from tests.test_takeout_index_reader import make_index
    idx_path = make_index(
        tmp_path / "index.sqlite",
        archives=(("takeout-001.zip", st.st_size, int(st.st_mtime), 0, None),),
        media=[("takeout-001.zip", f"{D}/IMG_1.jpg", "area", "folder",
                "IMG_1.jpg", None, "orphan", "none")],
    )
    stats = ingest_archives(archives, dest, catalog, index_path=idx_path)

    assert stats.index_archives_covered == 1
    assert stats.index_no_sidecar_fell_back == 1
    # The built-in ladder still found IMG_1.jpg.json -- missing_metadata must
    # NOT be incremented, and the photo must be dated and correctly foldered,
    # not dumped into Undated/.
    assert stats.missing_metadata == 0
    organized = list((dest / "2015" / "2015-03").glob("*.jpg"))
    assert len(organized) == 1
    assert not list(dest.glob("Undated/*.jpg"))

    from imageharbor.hashing import compute_sha256_b64url_bytes
    sha = compute_sha256_b64url_bytes(_jpeg(1))
    row = catalog.get_by_sha256(sha)
    assert row is not None
    assert row["date_tier"] == 30
    assert row["date_source"] == "external_sidecar"


def test_index_unrecognized_confidence_falls_back_and_is_counted(
    dirs, catalog: Catalog, tmp_path: Path
) -> None:
    """Fix pass 1, MINOR: an index row with a `confidence` value `pairing.py`
    does not recognize (e.g. "high") must not be trusted -- it would
    otherwise silently drop the title/people (neither "own" nor "related")
    and file the date at tier 30 regardless of what it actually was. It must
    fall back to the built-in ladder instead, and the fallback must be
    counted so a producer-side schema drift is visible, not invisible-safe."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792),
    })
    st = (archives / "takeout-001.zip").stat()
    from tests.test_takeout_index_reader import make_index
    idx_path = make_index(
        tmp_path / "index.sqlite",
        archives=(("takeout-001.zip", st.st_size, int(st.st_mtime), 0, None),),
        sidecars=[(1, "takeout-001.zip", f"{D}/IMG_1.jpg.json", "IMG_1.jpg.json")],
        media=[("takeout-001.zip", f"{D}/IMG_1.jpg", "area", "folder",
                "IMG_1.jpg", 1, "some-rule", "high")],
    )
    stats = ingest_archives(archives, dest, catalog, index_path=idx_path)

    assert stats.index_archives_covered == 1
    assert stats.index_bad_confidence_fell_back == 1
    # The built-in ladder still found IMG_1.jpg.json on its own.
    assert stats.missing_metadata == 0


# --- Fix pass 1: narrow the explicit-index exception catch -----------------


def test_an_unexpected_exception_from_an_explicit_index_open_is_not_hidden(
    dirs, catalog: Catalog, tmp_path: Path, monkeypatch
) -> None:
    """A bug in this code (or in `index_reader.py`) must surface as a real
    traceback when the index was named EXPLICITLY via `--takeout-index` --
    wrapping it into `IndexUnusable` tells the operator their FILE is bad
    when the fault is ImageHarbor's own. Only file-shaped failures
    (`OSError`, `sqlite3.Error`), plus a genuine `IndexUnusable`, may become
    `IndexUnusable`; anything else -- an `AttributeError` here, standing in
    for a real internal bug -- must propagate unwrapped."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {f"{D}/IMG_1.jpg": _jpeg(1)})
    idx_path = tmp_path / "index.sqlite"
    idx_path.write_bytes(b"")   # never actually read -- open() is replaced below

    def _boom(cls, path, archive_stats):
        raise AttributeError("'NoneType' object has no attribute 'sidecar_for_bug'")

    monkeypatch.setattr(ingest_mod.index_reader.IndexPairings, "open", classmethod(_boom))

    with pytest.raises(AttributeError, match="sidecar_for_bug"):
        ingest_archives(archives, dest, catalog, index_path=idx_path)


def test_an_unreadable_auto_detected_index_does_not_fail_the_run(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    """I1: `Path.is_file()` only swallows ENOENT/ENOTDIR/EBADF/ELOOP -- a
    PermissionError (EACCES), or a stale network handle, re-raises. That
    call used to sit OUTSIDE the try/except that treats an auto-detected
    index's failures as warn-and-fall-back, so an unreadable
    takeout-index.sqlite beside the archives -- no --takeout-index flag --
    failed the ENTIRE ingest with the raw PermissionError. A broken
    auto-detected index must never fail an ingest."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792),
    })
    index_candidate = archives / "takeout-index.sqlite"
    index_candidate.write_bytes(b"")

    real_is_file = Path.is_file

    def _boom(self):
        if self == index_candidate:
            raise PermissionError(13, "Access is denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", _boom)

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1                # the run completed, unfailed
    assert stats.index_path is None            # the index was never actually loaded
    assert stats.missing_metadata == 0         # built-in ladder still found the sidecar


def test_an_unexpected_exception_from_an_auto_detected_index_still_falls_back(
    dirs, catalog: Catalog, monkeypatch
) -> None:
    """The same bug, hit via AUTO-DETECTION (no explicit `--takeout-index`),
    must NOT fail the run. Narrowing the explicit path's catch must not
    change the auto-detect path's long-standing contract: any failure while
    opening an auto-detected index -- of any exception type -- only warns and
    falls back to the built-in pairing rungs for the whole run."""
    archives, dest = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/IMG_1.jpg": _jpeg(1),
        f"{D}/IMG_1.jpg.json": _sidecar("IMG_1.jpg", 1425905792),
    })
    (archives / "takeout-index.sqlite").write_bytes(b"")

    def _boom(cls, path, archive_stats):
        raise AttributeError("'NoneType' object has no attribute 'sidecar_for_bug'")

    monkeypatch.setattr(ingest_mod.index_reader.IndexPairings, "open", classmethod(_boom))

    stats = ingest_archives(archives, dest, catalog)

    assert stats.ingested == 1               # the run completed, unfailed
    assert stats.index_path is None           # the index was never actually loaded
    assert stats.missing_metadata == 0        # built-in ladder still found the sidecar
