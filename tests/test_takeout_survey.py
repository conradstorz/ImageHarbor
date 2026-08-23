"""The survey reads archives and never writes to them."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone

from imageharbor.takeout import survey

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64


def _sidecar(title, taken="2019-07-04T12:33:11Z"):
    # metadata.parse_photo_metadata reads the "timestamp" (epoch seconds) key,
    # not "formatted" -- it has to be derived from `taken`, not hardcoded, or
    # every call site would silently produce the same photo_taken_at.
    dt = datetime.strptime(taken, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    epoch = int(dt.timestamp())
    return json.dumps(
        {"title": title, "photoTakenTime": {"formatted": taken, "timestamp": str(epoch)}}
    ).encode()


def _archive(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_counts_members_by_kind(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/Photos from 2019/a.jpg": JPEG,
        "Takeout/Google Photos/Photos from 2019/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        "Takeout/Google Photos/Photos from 2019/b.mp4": MP4,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["image"] == 1
    assert inv.kind_counts["video"] == 1
    assert inv.kind_counts["metadata"] == 1


def test_misnamed_media_is_found_by_sniffing(tmp_path):
    """A .screen member holding JPEG bytes is a photograph, not a document."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/5427880241588018962.screen": JPEG,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.misnamed_counts[".screen"] == 1
    assert inv.kind_counts["other"] == 1


def test_recognized_extension_is_never_second_guessed(tmp_path):
    """A .jpg is trusted on its extension; its bytes are not read."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": b"this is not actually a jpeg",
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["image"] == 1
    assert inv.misnamed_counts == {}


def test_loose_part_filling_a_sequence_gap_is_identified(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/Google Photos/a.jpg": JPEG})
    _archive(tmp_path / "takeout-20260818T012414Z-2-003.zip", {"Takeout/Google Photos/c.jpg": JPEG})
    (tmp_path / "VID_20160529_175415-002.mp4").write_bytes(MP4)
    inv = survey.survey_archives(tmp_path)
    assert [lp.part for lp in inv.loose_parts] == ["002"]
    assert inv.loose_parts[0].kind == "video"


def test_timestamps_are_collected_for_clustering(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg", "1968-01-12T10:35:03Z"),
        "Takeout/Google Photos/b.jpg": JPEG,
        "Takeout/Google Photos/b.jpg.supplemental-metadata.json": _sidecar("b.jpg", "1968-01-12T10:35:03Z"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.timestamp_counts["1968-01-12T10:35:03"] == 2
    assert inv.year_counts["1968"] == 2


def test_descriptor_tiers_are_counted(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/IMG_1234.jpg": JPEG,          # camera-generated
        "Takeout/Google Photos/Scouts and Halloween 002.jpg": JPEG,  # human
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.descriptor_machine == 1
    assert inv.descriptor_human == 1


def test_media_without_a_sidecar_is_counted(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        "Takeout/Google Photos/lonely.jpg": JPEG,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.media_without_sidecar == 1


def test_orphan_sidecars_are_counted_across_the_whole_batch(tmp_path):
    """A sidecar is orphaned only if NO archive holds its media."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        "Takeout/Google Photos/ghost.jpg.supplemental-metadata.json": _sidecar("ghost.jpg"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.orphan_sidecars == 1


def test_a_sidecar_pairing_across_two_archives_is_not_an_orphan(tmp_path):
    """Google splits parts by size, so media and sidecar routinely separate."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
    })
    _archive(tmp_path / "takeout-20260818T012414Z-2-002.zip", {
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.orphan_sidecars == 0
    assert inv.media_without_sidecar == 0


def test_a_sidecar_with_no_usable_timestamp_is_counted(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.jpg": JPEG,
        "Takeout/Google Photos/a.jpg.supplemental-metadata.json": b'{"title": "a.jpg"}',
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.sidecars_without_timestamp == 1


def test_an_unreadable_archive_is_recorded_not_raised(tmp_path):
    (tmp_path / "takeout-20260818T012414Z-2-001.zip").write_bytes(b"not a zip at all")
    inv = survey.survey_archives(tmp_path)
    assert inv.unreadable_archives == 1
    assert inv.archives[0].error is not None


def test_survey_is_read_only(tmp_path):
    """The property that makes it safe to run against a live archive set."""
    paths = [
        _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
            "Takeout/Google Photos/a.jpg": JPEG,
            "Takeout/Google Photos/x.screen": JPEG,
            "Takeout/Google Photos/a.jpg.supplemental-metadata.json": _sidecar("a.jpg"),
        }),
    ]
    loose = tmp_path / "VID_20160529_175415-002.mp4"
    loose.write_bytes(MP4)
    paths.append(loose)

    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    survey.survey_archives(tmp_path)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after


def test_empty_directory_yields_an_empty_inventory(tmp_path):
    inv = survey.survey_archives(tmp_path)
    assert inv.archives == []
    assert sum(inv.kind_counts.values()) == 0
