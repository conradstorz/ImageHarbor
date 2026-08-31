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
