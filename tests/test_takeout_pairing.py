"""Tests for media -> sidecar pairing across Google's naming mutations.

The (N)-displacement rows are verbatim from the real export:
    2015-03-09.jpg       2015-03-09.jpg.json
    2015-03-09(1).jpg    2015-03-09.jpg(1).json
    2015-03-09(2).jpg    2015-03-09.jpg(2).json
"""

from __future__ import annotations

import pytest

from imageharbor.takeout.pairing import (
    NO_MATCH,
    OWN,
    RELATED,
    _MIN_TRUNCATION_PREFIX,
    build_index,
    sidecar_for,
)

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
    assert sidecar_for(media, index).sidecar == expected


def test_no_confident_match_returns_none(index) -> None:
    """Never guess: a photo without Google metadata is still fully organized."""
    assert sidecar_for(f"{D}/IMG_9999.jpg", index).sidecar is None


def test_a_sidecar_is_not_paired_with_itself(index) -> None:
    assert sidecar_for(f"{D}/2015-03-09.jpg.json", index).sidecar is None


def test_a_sidecar_in_another_directory_is_not_matched() -> None:
    """Pairing never crosses a directory boundary."""
    index = build_index(["a/x.jpg", "b/x.jpg.json"])
    assert sidecar_for("a/x.jpg", index).sidecar is None


def test_paren_form_is_not_shadowed_by_the_generic_rule() -> None:
    """`NAME(N).EXT.json` also exists in some exports; the displaced form wins."""
    index = build_index(
        ["d/p.jpg", "d/p.jpg.json", "d/p(1).jpg", "d/p.jpg(1).json", "d/p(1).jpg.json"]
    )
    assert sidecar_for("d/p(1).jpg", index).sidecar == "d/p.jpg(1).json"


def test_truncation_recovery_accepts_a_unique_prefix() -> None:
    long_media = "d/emma-graduation-ceremony-at-the-high-school-2.jpg"
    long_sidecar = "d/emma-graduation-ceremony-at-the-high-schoo.json"
    index = build_index([long_media, long_sidecar])
    assert sidecar_for(long_media, index).sidecar == long_sidecar


def test_truncation_recovery_refuses_an_ambiguous_prefix() -> None:
    media = "d/emma-graduation-ceremony-at-the-high-school-2.jpg"
    index = build_index(
        [
            media,
            "d/emma-graduation-ceremony-at-the-high-schoo.json",
            "d/emma-graduation-ceremony-at-the-high-scho.json",
        ]
    )
    assert sidecar_for(media, index).sidecar is None


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

    assert sidecar_for(owner, index).sidecar == sidecar
    assert sidecar_for(thief, index).sidecar is None


def test_case_colliding_sidecars_are_not_matched() -> None:
    """Two sidecars differing only in case make a case-insensitive hit a guess.

    Rung 5 exists to recover an exact match whose extension case differs. When
    two real members collide under lowercasing there is no longer one obvious
    answer, so the index poisons that key and the lookup must decline.
    """
    index = build_index(["d/PHOTO.JPG", "d/photo.jpg.json", "d/PHOTO.JPG.json"])
    assert index.sidecars_ci["d/photo.jpg.json"] is None
    assert sidecar_for("d/PHOTO.jpg", index).sidecar is None


def test_truncation_recovery_ignores_a_too_short_prefix() -> None:
    """A short sidecar name prefixes half the directory; that is a guess."""
    index = build_index(["d/abcdefgh.jpg", "d/abc.json"])
    assert sidecar_for("d/abcdefgh.jpg", index).sidecar is None


def test_root_level_members_pair() -> None:
    index = build_index(["x.jpg", "x.jpg.json"])
    assert sidecar_for("x.jpg", index).sidecar == "x.jpg.json"


def test_media_with_no_extension_does_not_crash() -> None:
    index = build_index(["d/noext", "d/noext.json"])
    assert sidecar_for("d/noext", index).sidecar == "d/noext.json"


def test_index_is_global_across_archives() -> None:
    """Google splits by size, so a photo and its sidecar land in different parts.

    The index is built from every member in the batch precisely so that the
    part boundary is invisible here.
    """
    index = build_index(["d/a.jpg", "d/b.jpg", "d/a.jpg.json", "d/b.jpg.json"])
    assert sidecar_for("d/a.jpg", index).sidecar == "d/a.jpg.json"
    assert sidecar_for("d/b.jpg", index).sidecar == "d/b.jpg.json"


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
    assert sidecar_for("d/a.jpg", index).sidecar is None


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
    assert sidecar_for(media, index).sidecar is None


def test_a_sidecar_appearing_once_still_pairs_normally() -> None:
    """The exclusion must be targeted -- the ordinary case is unaffected."""
    index = build_index(["d/IMG_1234.jpg", "d/IMG_1234.jpg.json"])
    assert index.ambiguous_sidecars == frozenset()
    assert sidecar_for("d/IMG_1234.jpg", index).sidecar == "d/IMG_1234.jpg.json"


# -- Rung 1b: NAME(N).EXT -> NAME.EXT.supplemental-metadata(N).json --------
#
# The newer supplemental-metadata spelling of the copy-suffix displacement,
# verified against a real 361 GiB export: the copy marker moves after the
# extension AND after the "supplemental-metadata" tag.

REAL_SUPPLEMENTAL_COPY_EXAMPLES = [
    ("2019_01_24_086(1).jpg", "2019_01_24_086.jpg.supplemental-metadata(1).json"),
    ("DSC_0002(1).JPG", "DSC_0002.JPG.supplemental-metadata(1).json"),
    ("IMG_425785091311895(1).jpeg", "IMG_425785091311895.jpeg.supplemental-metadata(1).json"),
    (
        "Copy (4) of scan0105a_edited(1).jpg",
        "Copy (4) of scan0105a_edited.jpg.supplemental-metadata(1).json",
    ),
]


@pytest.mark.parametrize("media_name, sidecar_name", REAL_SUPPLEMENTAL_COPY_EXAMPLES)
def test_supplemental_metadata_copy_suffix_pairs(media_name, sidecar_name) -> None:
    """Real examples from the export: the copy marker after the tag pairs."""
    media = f"{D}/{media_name}"
    sidecar = f"{D}/{sidecar_name}"
    index = build_index([media, sidecar])
    assert sidecar_for(media, index).sidecar == sidecar


def test_supplemental_copy_suffix_splits_on_the_last_parenthesised_group() -> None:
    """An earlier parenthesised group must not be mistaken for the copy marker.

    `_PAREN_RE` is anchored at the end with `$` -- that anchor, not
    greediness, is what forces it to take the LAST parenthesised group (the
    real copy suffix) rather than an earlier one embedded in the base name: a
    lazy `.*?` for the base group would backtrack to the exact same split,
    since the anchor is what pins the match to the string's end.
    """
    media = f"{D}/vacation (2019) day(1).jpg"
    sidecar = f"{D}/vacation (2019) day.jpg.supplemental-metadata(1).json"
    index = build_index([media, sidecar])
    assert sidecar_for(media, index).sidecar == sidecar


def test_edited_variant_with_copy_suffix_inherits_supplemental_sidecar() -> None:
    """`-edited` stripping (rung 4) must also see the supplemental-copy form."""
    original = "d/photo(1).jpg"
    edited = "d/photo(1)-edited.jpg"
    sidecar = "d/photo.jpg.supplemental-metadata(1).json"
    index = build_index([original, edited, sidecar])
    assert sidecar_for(original, index).sidecar == sidecar
    assert sidecar_for(edited, index).sidecar == sidecar


def test_supplemental_copy_sidecar_shared_by_two_archives_is_never_paired() -> None:
    """Ambiguity protection must cover the new candidate spelling too.

    A sidecar path duplicated across the batch (as if two different media --
    one from each archive -- both constructed it) is exactly the situation
    `ambiguous_sidecars` exists to refuse, no matter which rung constructed
    the candidate string.
    """
    media = "d/2019_01_24_086(1).jpg"
    sidecar = "d/2019_01_24_086.jpg.supplemental-metadata(1).json"
    index = build_index([media, sidecar, sidecar])
    assert sidecar in index.ambiguous_sidecars
    assert sidecar_for(media, index).sidecar is None


def test_media_with_copy_suffix_shared_by_two_archives_is_never_paired() -> None:
    """`ambiguous_media` protection also covers media using the new rung."""
    media = "d/2019_01_24_086(1).jpg"
    sidecar = "d/2019_01_24_086.jpg.supplemental-metadata(1).json"
    index = build_index([media, media, sidecar])
    assert media in index.ambiguous_media
    assert sidecar_for(media, index).sidecar is None


def test_supplemental_copy_suffix_declines_a_decoy_sidecar() -> None:
    """No false positive: only the copy's OWN sidecar may pair with it.

    An earlier version of this test built its index from one media member and
    zero sidecars, so `index.sidecars` was empty and any implementation --
    including a broken one -- returned `None`; it pinned nothing. This
    version supplies a decoy sidecar in each case, so it fails if the new
    candidate is constructed sloppily (e.g. if it ignored the copy index and
    matched any supplemental sidecar for the base name).
    """
    media = f"{D}/2019_01_24_086(1).jpg"

    # The non-copy's sidecar is present; the copy's own is not.
    index = build_index([media, f"{D}/2019_01_24_086.jpg.supplemental-metadata.json"])
    assert sidecar_for(media, index).sidecar is None

    # A different copy index's sidecar is present.
    index = build_index([media, f"{D}/2019_01_24_086.jpg.supplemental-metadata(2).json"])
    assert sidecar_for(media, index).sidecar is None


def test_plain_paren_json_spelling_still_pairs_no_regression() -> None:
    """The original `NAME.EXT(N).json` spelling must be unaffected."""
    index = build_index([f"{D}/2015-03-09(1).jpg", f"{D}/2015-03-09.jpg(1).json"])
    assert sidecar_for(f"{D}/2015-03-09(1).jpg", index).sidecar == f"{D}/2015-03-09.jpg(1).json"


def test_supplemental_paren_form_is_not_shadowed_by_the_generic_supplemental_rule() -> None:
    """`NAME(N).EXT.supplemental-metadata.json` also exists in some exports;
    the displaced form (`NAME.EXT.supplemental-metadata(N).json`) still wins.

    Mirrors `test_paren_form_is_not_shadowed_by_the_generic_rule` for the
    newer spelling: the new candidate is emitted at index 1 of `_candidates`,
    ahead of the generic supplemental rung for the same variant, so it must
    not be shadowed.

    NOTE: the real 361 GiB export this module was verified against contains
    zero directories where both spellings co-occur -- this test pins a
    precedence rule that is currently theoretical, not one observed in the
    wild.
    """
    index = build_index(
        [
            "d/p.jpg",
            "d/p.jpg.supplemental-metadata.json",
            "d/p(1).jpg",
            "d/p.jpg.supplemental-metadata(1).json",
            "d/p(1).jpg.supplemental-metadata.json",
        ]
    )
    assert sidecar_for("d/p(1).jpg", index).sidecar == "d/p.jpg.supplemental-metadata(1).json"


# -- Pairing confidence -------------------------------------------------
#
# `sidecar_for` returns a `Pairing`, not a bare path: an exact match and an
# `-edited` derivative inheriting its original's sidecar look identical as
# strings, but the sidecar's title and coordinates describe the ORIGINAL in
# the second case, not the file being paired. Confidence follows the name
# variant that produced the candidate (`_name_variants`'s `own` vs `related`),
# not the rung number, so the case-insensitive retry (rung 5) inherits
# whichever confidence its underlying rung would have carried.


def test_exact_match_is_own():
    index = build_index([
        "T/GP/2019/IMG_1.jpg", "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"])
    p = sidecar_for("T/GP/2019/IMG_1.jpg", index)
    assert p.sidecar == "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"
    assert p.confidence == OWN


def test_edited_derivative_is_related():
    # The sidecar names IMG_1.jpg, not IMG_1-edited.jpg. Its location and
    # title belong to a different file.
    index = build_index([
        "T/GP/2019/IMG_1-edited.jpg", "T/GP/2019/IMG_1.jpg",
        "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"])
    p = sidecar_for("T/GP/2019/IMG_1-edited.jpg", index)
    assert p.sidecar == "T/GP/2019/IMG_1.jpg.supplemental-metadata.json"
    assert p.confidence == RELATED


def test_case_insensitive_retry_keeps_the_underlying_confidence():
    # Rung 5 retries rungs 1-4 case-insensitively. It is NOT a confidence of
    # its own: a case-differing -edited file is still related.
    index = build_index([
        "T/GP/2019/IMG_1-EDITED.JPG", "T/GP/2019/img_1.jpg.json"])
    p = sidecar_for("T/GP/2019/IMG_1-EDITED.JPG", index)
    assert p.sidecar == "T/GP/2019/img_1.jpg.json"
    assert p.confidence == RELATED


def test_truncation_recovery_is_own():
    # Rung 6 resolves a truncated spelling of THIS file's own name. The
    # sidecar's truncated portion must be a genuine prefix of the media's
    # full basename (extension included) -- truncating only the stem and
    # then appending a full, untruncated extension (as an earlier version of
    # this test did) produces a "media part" that never prefix-matches, since
    # real Google truncation cuts the whole name string, not just the stem.
    long_name = "A_very_long_original_filename_that_google_truncated.jpg"
    index = build_index([
        f"T/GP/2019/{long_name}",
        f"T/GP/2019/{long_name[:40]}.supplemental-metadata.json"])
    p = sidecar_for(f"T/GP/2019/{long_name}", index)
    assert p.confidence == OWN


def test_no_match_is_none():
    index = build_index(["T/GP/2019/lonely.jpg"])
    p = sidecar_for("T/GP/2019/lonely.jpg", index)
    assert p.sidecar is None
    assert p.confidence == NO_MATCH
