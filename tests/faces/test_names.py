"""Name normalization: whitespace is fixed automatically, case never is."""

import pytest

from imageharbor.faces import names


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Gladys Blankenbeker ", "Gladys Blankenbeker"),   # 461 real occurrences
        (" Conrad Storz", "Conrad Storz"),
        ("Conrad  Storz", "Conrad Storz"),                 # collapsed internal run
        ("Conrad\tStorz", "Conrad Storz"),
        ("Conrad Storz", "Conrad Storz"),                  # already clean, unchanged
        ("", ""),
    ],
)
def test_normalize_fixes_whitespace(raw, expected):
    assert names.normalize(raw) == expected


def test_normalize_never_changes_case():
    # 1,539 photos say "pete storz". Case-folding is a judgement about identity,
    # not a formatting fix, so it is never applied automatically.
    assert names.normalize("pete storz") == "pete storz"
    assert names.normalize("claire Storz") == "claire Storz"


def test_case_variants_groups_only_case_differences():
    groups = names.case_variants(["pete storz", "Pete Storz", "Judy Storz"])
    assert groups == {"pete storz": ["Pete Storz", "pete storz"]}


def test_case_variants_never_groups_a_suffix_difference():
    # The whole reason fuzzy matching is banned: these are a father and a son.
    groups = names.case_variants(["Conrad Storz", "Conrad Storz III"])
    assert groups == {}


def test_case_variants_is_deterministic():
    a = names.case_variants(["b Smith", "B Smith", "B SMITH"])
    b = names.case_variants(["B SMITH", "B Smith", "b Smith"])
    assert a == b
    assert a["b smith"] == ["B SMITH", "B Smith", "b Smith"]


def test_case_variants_never_groups_more_than_case():
    # str.casefold() is Unicode-normalizing, not case-folding: 'Weiß'.casefold()
    # == 'Weiss'.casefold() even though they are different names (an extra 's',
    # not a case change of any character). Grouping these would surface a bogus
    # "these may be the same person" suggestion in the review UI.
    groups = names.case_variants(["Weiß", "Weiss"])
    assert groups == {}


def test_case_variants_still_groups_same_length_compatibility_characters():
    # Known, accepted residual: the Kelvin sign (U+212A) and 'K' are the same
    # length, and Unicode's own simple case mapping sends both to 'k' -- the
    # same target str.lower() gives plain 'K'. The length gate in _case_key
    # only excludes length-changing folds like Weiß/Weiss; it cannot and does
    # not separate this pair. This is accepted because case_variants only
    # ever suggests a merge for a human to confirm, never performs one --
    # do not "fix" this by trying to special-case Kelvin.
    kelvin_sign = "K"
    assert kelvin_sign != "K"  # distinct characters going in
    groups = names.case_variants([kelvin_sign, "K"])
    assert groups == {"k": ["K", kelvin_sign]}  # sorted: plain K (0x4B) before Kelvin sign (0x212A)
