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


def test_a_media_path_shared_by_two_archives_is_never_paired() -> None:
    """Two exports sharing a member path is a natural user action.

    The index is keyed on bare member-path strings with no archive dimension,
    so a path present twice in the batch is indistinguishable from itself --
    pairing it would risk dating one archive's bytes with the OTHER archive's
    sidecar, silently. Declining is the only safe answer, even though one of
    the two occurrences here does have a sidecar.
    """
    members = ["d/a.jpg", "d/a.jpg", "d/a.jpg.json"]
    index = build_index(members)
    assert "d/a.jpg" in index.ambiguous_media
    assert sidecar_for("d/a.jpg", index) is None


def test_a_sidecar_path_shared_by_two_archives_is_never_paired() -> None:
    """A duplicated SIDECAR path is as ambiguous as a duplicated media path.

    The index has no archive dimension and the ingest layer's owner map is
    last-writer-wins, so a sidecar path present in two archives can hand a
    photo the wrong archive's bytes -- silently dating it from metadata that
    does not describe it. Declining costs only the Google metadata; a wrong
    date is permanent.
    """
    media = "d/IMG_1234.jpg"
    sidecar = "d/IMG_1234.jpg.json"
    # `sidecar` supplied twice: once by each archive in the batch.
    index = build_index([media, sidecar, sidecar])

    assert sidecar in index.ambiguous_sidecars
    assert sidecar_for(media, index) is None


def test_a_sidecar_appearing_once_still_pairs_normally() -> None:
    """The exclusion must be targeted -- the ordinary case is unaffected."""
    index = build_index(["d/IMG_1234.jpg", "d/IMG_1234.jpg.json"])
    assert index.ambiguous_sidecars == frozenset()
    assert sidecar_for("d/IMG_1234.jpg", index) == "d/IMG_1234.jpg.json"
