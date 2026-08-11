"""Tests for target-path computation, safe renames, and self-healing."""

from datetime import datetime

import pytest

from imageharbor.date_resolver import ResolvedDate
from imageharbor.relocate import (
    apply_relocation,
    find_by_digest,
    resolve_organized_path,
    target_path,
)

_D = "qfQ8jnnXIdtn-juMY-1JDqyBLPF6j2MJlbh8sZOIfcI"


def _dated():
    return ResolvedDate(value=datetime(2019, 7, 4), tier=40, source="exif_original")


def _undated():
    return ResolvedDate(value=None, tier=0, source="none")


def test_target_path_for_a_dated_file(tmp_path):
    path = target_path(tmp_path, _dated(), "beach", _D, "jpg")
    assert path == tmp_path / "2019" / "2019-07" / f"2019-07-04-beach_{_D}.jpg"


def test_target_path_for_an_undated_file(tmp_path):
    path = target_path(tmp_path, _undated(), "beach", _D, "jpg")
    assert path == tmp_path / "Undated" / f"beach_{_D}.jpg"


def test_target_path_with_no_descriptor(tmp_path):
    path = target_path(tmp_path, _undated(), "", _D, "jpg")
    assert path == tmp_path / "Undated" / f"{_D}.jpg"


def test_apply_relocation_moves_the_file(tmp_path):
    old = tmp_path / "Undated" / f"{_D}.jpg"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"content")
    new = tmp_path / "2019" / "2019-07" / f"2019-07-04_{_D}.jpg"

    apply_relocation(old, new)

    assert not old.exists()
    assert new.read_bytes() == b"content"


def test_apply_relocation_is_a_no_op_when_paths_match(tmp_path):
    path = tmp_path / f"{_D}.jpg"
    path.write_bytes(b"content")
    apply_relocation(path, path)
    assert path.read_bytes() == b"content"


def test_apply_relocation_refuses_to_clobber_different_content(tmp_path):
    old = tmp_path / f"a_{_D}.jpg"
    old.write_bytes(b"content")
    new = tmp_path / f"b_{_D}.jpg"
    new.write_bytes(b"different")

    with pytest.raises(FileExistsError):
        apply_relocation(old, new)

    assert old.exists()
    assert new.read_bytes() == b"different"


def test_apply_relocation_tolerates_an_identical_destination(tmp_path):
    """A crash after rename but before the catalog update leaves this state."""
    old = tmp_path / f"a_{_D}.jpg"
    old.write_bytes(b"content")
    new = tmp_path / f"b_{_D}.jpg"
    new.write_bytes(b"content")

    apply_relocation(old, new)

    assert not old.exists()
    assert new.read_bytes() == b"content"


def test_find_by_digest_locates_a_moved_file(tmp_path):
    target = tmp_path / "2019" / "2019-07" / f"2019-07-04-beach_{_D}.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"content")
    assert find_by_digest(tmp_path, _D) == target


def test_find_by_digest_returns_none_when_absent(tmp_path):
    assert find_by_digest(tmp_path, _D) is None


def test_resolve_organized_path_prefers_the_recorded_path(tmp_path):
    recorded = tmp_path / "Undated" / f"{_D}.jpg"
    recorded.parent.mkdir(parents=True)
    recorded.write_bytes(b"content")
    assert resolve_organized_path(tmp_path, recorded, _D) == recorded


def test_resolve_organized_path_self_heals_after_an_interrupted_rename(tmp_path):
    """The catalog points at the old path; the file is already at the new one."""
    recorded = tmp_path / "Undated" / f"{_D}.jpg"
    actual = tmp_path / "2019" / "2019-07" / f"2019-07-04_{_D}.jpg"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"content")

    assert resolve_organized_path(tmp_path, recorded, _D) == actual


def test_resolve_organized_path_returns_none_when_truly_gone(tmp_path):
    assert resolve_organized_path(tmp_path, tmp_path / "gone.jpg", _D) is None
