"""The survey reads archives and never writes to them."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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


# --- M1: _extension agrees with archive.classify -------------------------

def test_a_member_named_only_dot_jpg_agrees_between_kind_and_extension(tmp_path):
    """archive.classify calls ".jpg" an image; _extension must not call it "".

    A member named literally ".jpg" exists in the real export. When the two
    disagree, by_kind and by_extension describe different files.
    """
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/.jpg": JPEG,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["image"] == 1
    assert inv.ext_counts[".jpg"] == 1
    assert inv.ext_counts[""] == 0


# --- I1a: the no-batch part naming form parses ---------------------------

def test_the_no_batch_zip_naming_form_yields_part_numbers(tmp_path):
    """Google's older, still-common naming has no batch segment."""
    _archive(tmp_path / "takeout-20260818T012414Z-001.zip", {"Takeout/a.jpg": JPEG})
    _archive(tmp_path / "takeout-20260818T012414Z-003.zip", {"Takeout/c.jpg": JPEG})
    inv = survey.survey_archives(tmp_path)
    assert inv.part_numbers == {"001", "003"}


def test_the_batched_zip_naming_form_still_yields_part_numbers(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    inv = survey.survey_archives(tmp_path)
    assert inv.part_numbers == {"001"}


def test_an_unrecognized_naming_scheme_reports_undetermined_not_none(tmp_path):
    from imageharbor.takeout import report as takeout_report

    _archive(tmp_path / "my-photos-backup-a.zip", {"Takeout/a.jpg": JPEG})
    inv = survey.survey_archives(tmp_path)
    assert inv.part_numbers == set()
    doc = takeout_report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] is None


# --- I3: a non-media file beside the archives is not a loose part ---------

def test_a_non_media_file_beside_the_archives_is_not_a_loose_part(tmp_path):
    """checksums.txt and a previous run's survey.json are not parts."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    (tmp_path / "checksums.txt").write_text("deadbeef  takeout-001.zip\n")
    (tmp_path / "survey.json").write_text('{"archives": {}}')
    inv = survey.survey_archives(tmp_path)
    assert inv.loose_parts == []
    assert inv.non_archive_files == 2


def test_a_media_file_beside_the_archives_is_still_a_loose_part(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    (tmp_path / "VID_20160529_175415-002.mp4").write_bytes(MP4)
    inv = survey.survey_archives(tmp_path)
    assert [lp.name for lp in inv.loose_parts] == ["VID_20160529_175415-002.mp4"]
    assert inv.non_archive_files == 0


def test_a_stray_text_file_does_not_erase_a_genuinely_missing_part(tmp_path):
    """C1 end to end: transfer-log-002.txt must not hide the absent part 002."""
    from imageharbor.takeout import report as takeout_report

    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    _archive(tmp_path / "takeout-20260818T012414Z-2-003.zip", {"Takeout/c.jpg": JPEG})
    (tmp_path / "transfer-log-002.txt").write_text("copied 002 ok\n")
    inv = survey.survey_archives(tmp_path)
    doc = takeout_report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == ["002"]
    assert "missing parts none" not in takeout_report.format_summary(doc)


# --- C2: a file that cannot be read is counted, never raised --------------

def test_an_unreadable_loose_file_is_counted_not_raised(tmp_path, monkeypatch):
    """A mid-download file under a byte-range lock must not abort the run.

    The module docstring and the CLI help both promise the command is safe
    against an archive set another process is still downloading.
    """
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    locked = tmp_path / "VID_20160529_175415-002.mp4"
    locked.write_bytes(MP4)

    real_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self.name == locked.name:
            raise PermissionError(32, "The process cannot access the file")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    inv = survey.survey_archives(tmp_path)
    assert inv.unreadable_loose_files == 1
    assert inv.loose_parts == []
    # The zip work already done is kept, not discarded.
    assert inv.kind_counts["image"] == 1


def test_an_unstattable_loose_file_is_counted_not_raised(tmp_path, monkeypatch):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    loose = tmp_path / "VID_20160529_175415-002.mp4"
    loose.write_bytes(MP4)

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name == loose.name and not args and not kwargs:
            raise PermissionError(32, "locked")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    inv = survey.survey_archives(tmp_path)
    assert inv.unreadable_loose_files == 1
    assert inv.kind_counts["image"] == 1


def test_an_archive_that_becomes_unreadable_between_passes_is_counted(tmp_path, monkeypatch):
    """Pass two reopens every zip; that reopen can fail like pass one's can."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {"Takeout/a.jpg": JPEG})
    _archive(tmp_path / "takeout-20260818T012414Z-2-002.zip", {"Takeout/b.jpg": JPEG})

    real_zipfile = zipfile.ZipFile
    seen: dict[str, int] = {}

    class FlakyZipFile(real_zipfile):
        def __init__(self, file, *args, **kwargs):
            name = getattr(file, "name", str(file))
            if name.endswith("002.zip"):
                seen[name] = seen.get(name, 0) + 1
                if seen[name] > 1:
                    raise PermissionError(32, "vanished between passes")
            super().__init__(file, *args, **kwargs)

    monkeypatch.setattr(survey.zipfile, "ZipFile", FlakyZipFile)

    inv = survey.survey_archives(tmp_path)
    assert inv.unreadable_archives == 1
    # The readable archive's members were still counted.
    assert inv.kind_counts["image"] == 1


def test_a_zip_whose_stat_fails_does_not_abort_the_run(tmp_path, monkeypatch):
    """The size lookup on an unreadable archive is itself unguarded today."""
    bad = tmp_path / "takeout-20260818T012414Z-2-001.zip"
    bad.write_bytes(b"not a zip at all")
    _archive(tmp_path / "takeout-20260818T012414Z-2-002.zip", {"Takeout/a.jpg": JPEG})

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name == bad.name and not args and not kwargs:
            raise PermissionError(32, "locked")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    inv = survey.survey_archives(tmp_path)
    assert inv.unreadable_archives == 1
    assert inv.kind_counts["image"] == 1


# --- C3: orphan counting matches what the pairing index calls media -------

def test_a_sidecar_for_a_misnamed_member_is_not_an_orphan(tmp_path):
    """pairing.build_index treats every non-.json member as media.

    A .screen file's sidecar has its media right there in the same archive, so
    reporting it orphaned inflates the count by every misnamed member in the
    set (up to 4,107 on the real export).
    """
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/5427880241588018962.screen": JPEG,
        "Takeout/Google Photos/5427880241588018962.screen.supplemental-metadata.json":
            _sidecar("5427880241588018962.screen"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["other"] == 1
    assert inv.orphan_sidecars == 0


def test_a_genuinely_orphaned_sidecar_is_still_counted_alongside_a_misnamed_pair(tmp_path):
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/x.screen": JPEG,
        "Takeout/Google Photos/x.screen.supplemental-metadata.json": _sidecar("x.screen"),
        "Takeout/Google Photos/ghost.jpg.supplemental-metadata.json": _sidecar("ghost.jpg"),
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.orphan_sidecars == 1


def test_the_misnamed_pairing_fix_does_not_change_the_media_tallies(tmp_path):
    """media_without_sidecar and the descriptor tiers stay scoped to real media."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/x.screen": JPEG,
        "Takeout/Google Photos/x.screen.supplemental-metadata.json": _sidecar("x.screen"),
        "Takeout/Google Photos/lonely.jpg": JPEG,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.media_without_sidecar == 1
    assert inv.descriptor_human + inv.descriptor_machine == 1


# --- I2: misnamed members are split by sniffed kind ----------------------

def test_misnamed_members_are_tallied_by_sniffed_kind(tmp_path):
    """The projection cannot separate the pipelines without this split."""
    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/a.screen": JPEG,
        "Takeout/Google Photos/b.screen": JPEG,
        "Takeout/Google Photos/c.tile": MP4,
    })
    inv = survey.survey_archives(tmp_path)
    assert inv.misnamed_kind_counts["image"] == 2
    assert inv.misnamed_kind_counts["video"] == 1
    assert inv.misnamed_kind_bytes["image"] == 2 * len(JPEG)
    assert inv.misnamed_kind_bytes["video"] == len(MP4)


# --- M7: a corrupt deflate stream raises zlib.error, not OSError ---------

def test_a_zlib_error_while_sniffing_is_swallowed(tmp_path, monkeypatch):
    """A corrupt deflate stream must not abort the survey."""
    import zlib

    _archive(tmp_path / "takeout-20260818T012414Z-2-001.zip", {
        "Takeout/Google Photos/x.screen": JPEG,
        "Takeout/Google Photos/a.jpg": JPEG,
    })

    real_open = zipfile.ZipFile.open

    def fake_open(self, name, *args, **kwargs):
        member = name if isinstance(name, str) else name.filename
        if member.endswith(".screen"):
            raise zlib.error("Error -3 while decompressing data: invalid block type")
        return real_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", fake_open)

    inv = survey.survey_archives(tmp_path)
    assert inv.kind_counts["other"] == 1
    assert inv.misnamed_counts == {}
