"""Tests for imageharbor.discovery."""

from pathlib import Path

import pytest

from imageharbor.discovery import SUPPORTED_EXTENSIONS, discover_images


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ---------------------------------------------------------------------------


def test_supported_extensions_are_lowercase() -> None:
    for ext in SUPPORTED_EXTENSIONS:
        assert ext == ext.lower()
        assert ext.startswith(".")


def test_common_extensions_present() -> None:
    for ext in (".jpg", ".png", ".heic", ".cr2"):
        assert ext in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Nonexistent source
# ---------------------------------------------------------------------------


def test_nonexistent_source_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    # discover_images is a generator; must be consumed to trigger the raise.
    with pytest.raises(FileNotFoundError):
        list(discover_images(missing))


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------


def test_single_supported_file(tmp_path: Path) -> None:
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"")
    assert list(discover_images(f)) == [f]


def test_single_unsupported_file(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_bytes(b"")
    assert list(discover_images(f)) == []


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------


def test_directory_mixed_files(tmp_path: Path) -> None:
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"
    other = tmp_path / "readme.txt"
    for p in (img1, img2, other):
        p.write_bytes(b"")

    result = list(discover_images(tmp_path))
    assert result == [img1, img2]


def test_directory_recursive_finds_nested(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    nested_img = nested / "deep.jpg"
    nested_img.write_bytes(b"")

    assert list(discover_images(tmp_path, recursive=True)) == [nested_img]


def test_directory_non_recursive_skips_nested(tmp_path: Path) -> None:
    top_img = tmp_path / "top.jpg"
    top_img.write_bytes(b"")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.jpg").write_bytes(b"")

    assert list(discover_images(tmp_path, recursive=False)) == [top_img]


def test_extension_case_insensitive(tmp_path: Path) -> None:
    upper = tmp_path / "SHOUT.JPG"
    upper.write_bytes(b"")
    assert list(discover_images(tmp_path)) == [upper]


def test_subdirectories_not_yielded(tmp_path: Path) -> None:
    # A directory whose name ends in a supported extension must not be yielded.
    dir_like = tmp_path / "album.jpg"
    dir_like.mkdir()
    real_img = tmp_path / "real.png"
    real_img.write_bytes(b"")

    assert list(discover_images(tmp_path)) == [real_img]


def test_results_are_sorted(tmp_path: Path) -> None:
    # Create files out of alphabetical order.
    names = ["zebra.jpg", "apple.png", "mango.gif", "banana.bmp"]
    for name in names:
        (tmp_path / name).write_bytes(b"")

    result = list(discover_images(tmp_path))
    assert result == sorted(result, key=lambda p: p.as_posix())
    assert [p.name for p in result] == sorted(names)
