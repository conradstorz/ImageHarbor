"""imageharbor.util is the single home of the tiny helpers that used to be
copy-pasted across catalog.py, sidecar.py, and faces/ with a "keep the two
in sync" comment -- the class of hand-maintained invariant this codebase
otherwise refuses to have."""

import os
from datetime import datetime

import pytest

from imageharbor.util import fsync_file, json_default, now_iso


def test_fsync_file_syncs_and_propagates_oserror(tmp_path, monkeypatch):
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    called = {}
    monkeypatch.setattr(os, "fsync", lambda fd: called.setdefault("fd", fd))
    fsync_file(f)
    assert "fd" in called

    monkeypatch.setattr(
        os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk gone"))
    )
    with pytest.raises(OSError):
        fsync_file(f)


def test_now_iso_is_utc_aware_isoformat():
    value = now_iso()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_json_default_decodes_bytes_as_text_not_repr():
    # ExifVersion-style payload: must become "0230", never "b'0230'".
    assert json_default(b"0230") == "0230"


def test_json_default_replaces_undecodable_bytes_rather_than_raising():
    assert json_default(b"\xff\xfe") == "��"


def test_json_default_falls_back_to_str_for_exotic_types():
    assert json_default(3.5) == "3.5"
