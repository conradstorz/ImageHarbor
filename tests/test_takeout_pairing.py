"""Tests for media -> sidecar pairing across Google's naming mutations.

The (N)-displacement rows are verbatim from the real export:
    2015-03-09.jpg       2015-03-09.jpg.json
    2015-03-09(1).jpg    2015-03-09.jpg(1).json
    2015-03-09(2).jpg    2015-03-09.jpg(2).json
"""

from __future__ import annotations

import pytest

from imageharbor.takeout.pairing import _MIN_TRUNCATION_PREFIX, build_index, sidecar_for

D = "Takeout/AlbumArchive/Hangouts/album"

MEMBERS = [
    f"{D}/2015-03-09.jpg",
    f"{D}/2015-03-09.jpg.json",
    f"{D}/2015-03-09(1).jpg",
    f"{D}/2015-03-09.jpg(1).json",
    f"{D}/2015-03-09(2).jpg",
    f"{D}/2015-03-09.jpg(2).json",
    f"{D}/IMG_1234.jpg",
    f"{D}/IMG_1234.jpg.supplemental-metadata.json",
    f"{D}/IMG_9999.jpg",                       # no sidecar anywhere
    f"{D}/edited-thing.jpg",
    f"{D}/edited-thing.jpg.json",
    f"{D}/edited-thing-edited.jpg",            # derivative: no sidecar of its own
    f"{D}/UPPER.JPG",
    f"{D}/UPPER.jpg.json",                     # extension case differs
    f"{D}/party ●●● 2015 + friends = fun.jpg",
    f"{D}/party ●●● 2015 + friends = fun.jpg.json",
    f"{D}/Albums.json",
]


@pytest.fixture()
def index():
    return build_index(MEMBERS)


@pytest.mark.parametrize(
    "media, expected",
    [
        (f"{D}/2015-03-09.jpg", f"{D}/2015-03-09.jpg.json"),
        (f"{D}/2015-03-09(1).jpg", f"{D}/2015-03-09.jpg(1).json"),
        (f"{D}/2015-03-09(2).jpg", f"{D}/2015-03-09.jpg(2).json"),
        (f"{D}/IMG_1234.jpg", f"{D}/IMG_1234.jpg.supplemental-metadata.json"),
        (f"{D}/edited-thing.jpg", f"{D}/edited-thing.jpg.json"),
        # Google emits no sidecar for an edited derivative; it inherits the
        # original's.
        (f"{D}/edited-thing-edited.jpg", f"{D}/edited-thing.jpg.json"),
        (f"{D}/UPPER.JPG", f"{D}/UPPER.jpg.json"),
        (
            f"{D}/party ●●● 2015 + friends = fun.jpg",
            f"{D}/party ●●● 2015 + friends = fun.jpg.json",
        ),
    ],
)
def test_pairing_table(index, media, expected) -> None:
    assert sidecar_for(media, index) == expected


def test_no_confident_match_returns_none(index) -> None:
    """Never guess: a photo without Google metadata is still fully organized."""
    assert sidecar_for(f"{D}/IMG_9999.jpg", index) is None


def test_a_sidecar_is_not_paired_with_itself(index) -> None:
    assert sidecar_for(f"{D}/2015-03-09.jpg.json", index) is None


def test_a_sidecar_in_another_directory_is_not_matched() -> None:
    """Pairing never crosses a directory boundary."""
    index = build_index(["a/x.jpg", "b/x.jpg.json"])
    assert sidecar_for("a/x.jpg", index) is None


def test_paren_form_is_not_shadowed_by_the_generic_rule() -> None:
    """`NAME(N).EXT.json` also exists in some exports; the displaced form wins."""
    index = build_index(
        ["d/p.jpg", "d/p.jpg.json", "d/p(1).jpg", "d/p.jpg(1).json", "d/p(1).jpg.json"]
    )
    assert sidecar_for("d/p(1).jpg", index) == "d/p.jpg(1).json"


def test_truncation_recovery_accepts_a_unique_prefix() -> None:
    long_media = "d/emma-graduation-ceremony-at-the-high-school-2.jpg"
    long_sidecar = "d/emma-graduation-ceremony-at-the-high-schoo.json"
    index = build_index([long_media, long_sidecar])
    assert sidecar_for(long_media, index) == long_sidecar


def test_truncation_recovery_refuses_an_ambiguous_prefix() -> None:
    media = "d/emma-graduation-ceremony-at-the-high-school-2.jpg"
    index = build_index(
        [
            media,
            "d/emma-graduation-ceremony-at-the-high-schoo.json",
            "d/emma-graduation-ceremony-at-the-high-scho.json",
        ]
    )
    assert sidecar_for(media, index) is None


def test_truncation_recovery_never_steals_a_claimed_sidecar() -> None:
    """A sidecar that exactly pairs with another member is off limits.

    The names are deliberately contrived so this test isolates the claimed-set
    guard and nothing else. Two conditions have to hold at once or the test is
    vacuous, which an earlier version of it was:

    * the sidecar's media-part must be at least ``_MIN_TRUNCATION_PREFIX``
      characters, or the length floor rejects the candidate first and the
      claimed-set guard is never reached;
    * the would-be thief's name must literally start with that whole
      media-part, extension included, or the prefix test fails first.

    With both satisfied, the ONLY thing returning ``None`` is the claimed-set
    exclusion -- so deleting that clause makes this test fail, which is the
    entire point of it.
    """
    owner = "d/photograph-of-a-sunset.jpg"
    sidecar = "d/photograph-of-a-sunset.jpg.json"
    thief = "d/photograph-of-a-sunset.jpg-extra.jpg"
    index = build_index([owner, sidecar, thief])

    # Preconditions, asserted so a future edit to the fixture cannot silently
    # make this test vacuous again.
    assert len(owner.rsplit("/", 1)[-1]) >= _MIN_TRUNCATION_PREFIX
    assert thief.startswith(owner)
    assert sidecar in index.claimed

    assert sidecar_for(owner, index) == sidecar
    assert sidecar_for(thief, index) is None


def test_case_colliding_sidecars_are_not_matched() -> None:
    """Two sidecars differing only in case make a case-insensitive hit a guess.

    Rung 5 exists to recover an exact match whose extension case differs. When
    two real members collide under lowercasing there is no longer one obvious
    answer, so the index poisons that key and the lookup must decline.
    """
    index = build_index(["d/PHOTO.JPG", "d/photo.jpg.json", "d/PHOTO.JPG.json"])
    assert index.sidecars_ci["d/photo.jpg.json"] is None
    assert sidecar_for("d/PHOTO.jpg", index) is None


def test_truncation_recovery_ignores_a_too_short_prefix() -> None:
    """A short sidecar name prefixes half the directory; that is a guess."""
    index = build_index(["d/abcdefgh.jpg", "d/abc.json"])
    assert sidecar_for("d/abcdefgh.jpg", index) is None


def test_root_level_members_pair() -> None:
    index = build_index(["x.jpg", "x.jpg.json"])
    assert sidecar_for("x.jpg", index) == "x.jpg.json"


def test_media_with_no_extension_does_not_crash() -> None:
    index = build_index(["d/noext", "d/noext.json"])
    assert sidecar_for("d/noext", index) == "d/noext.json"


def test_index_is_global_across_archives() -> None:
    """Google splits by size, so a photo and its sidecar land in different parts.

    The index is built from every member in the batch precisely so that the
    part boundary is invisible here.
    """
    index = build_index(["d/a.jpg", "d/b.jpg", "d/a.jpg.json", "d/b.jpg.json"])
    assert sidecar_for("d/a.jpg", index) == "d/a.jpg.json"
    assert sidecar_for("d/b.jpg", index) == "d/b.jpg.json"
