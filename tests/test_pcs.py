"""Tests for imageharbor.pcs."""

import pytest

from imageharbor.pcs import (
    PCS_CATEGORIES,
    VALID_CODES,
    PCS_VERSION,
    get_category,
    parent_folder_name,
    resolve_code,
    sub_folder_name,
)


def test_pcs_version_is_string() -> None:
    assert isinstance(PCS_VERSION, str)
    assert PCS_VERSION  # not empty


def test_all_top_level_codes_present() -> None:
    for code in (100, 200, 300, 400, 500, 600, 700, 800, 900):
        assert code in PCS_CATEGORIES, f"Missing top-level code {code}"


def test_all_codes_have_names() -> None:
    for code, cat in PCS_CATEGORIES.items():
        assert cat.name, f"Empty name for code {code}"
        assert cat.label, f"Empty label for code {code}"


def test_sub_categories_have_valid_parents() -> None:
    for code, cat in PCS_CATEGORIES.items():
        if cat.parent is not None:
            assert cat.parent in PCS_CATEGORIES, (
                f"Category {code} references unknown parent {cat.parent}"
            )


def test_valid_codes_sorted() -> None:
    assert VALID_CODES == sorted(VALID_CODES)


def test_get_category_known() -> None:
    cat = get_category(330)
    assert cat is not None
    assert cat.code == 330
    assert cat.name == "beach"


def test_get_category_unknown() -> None:
    assert get_category(999) is None


def test_resolve_code_known() -> None:
    assert resolve_code(330) == 330


def test_resolve_code_unknown_falls_back_to_900() -> None:
    assert resolve_code(999) == 900
    assert resolve_code(0) == 900


def test_parent_folder_name_places() -> None:
    result = parent_folder_name(330)
    assert result == "300-places"


def test_parent_folder_name_people() -> None:
    result = parent_folder_name(110)
    assert result == "100-people"


def test_sub_folder_name_beach() -> None:
    result = sub_folder_name(330)
    assert result == "330-beach"


def test_sub_folder_name_unknown_falls_back() -> None:
    result = sub_folder_name(999)
    assert result == "900-miscellaneous"


@pytest.mark.parametrize("code", [1500, 50, -50])
def test_parent_folder_name_out_of_range_falls_back_to_900(code: int) -> None:
    # Out-of-range codes are resolved to 900 first, so the prefix is always a
    # valid top-level code (consistent with sub_folder_name / resolve_code).
    assert parent_folder_name(code) == "900-miscellaneous"


@pytest.mark.parametrize("code", [1500, 50, -50, 999, 0])
def test_parent_and_sub_folder_agree_on_out_of_range(code: int) -> None:
    # Both helpers resolve out-of-range codes to 900 before deriving names.
    assert parent_folder_name(code) == "900-miscellaneous"
    assert sub_folder_name(code) == "900-miscellaneous"


def test_parent_folder_name_places_unchanged() -> None:
    # In-range codes are unaffected by the resolve step.
    assert parent_folder_name(330) == "300-places"


def test_parent_and_sub_folder_consistent() -> None:
    # For every sub-category, parent folder code prefix should match
    for code, cat in PCS_CATEGORIES.items():
        if cat.parent is not None:
            parent = parent_folder_name(code)
            parent_code = int(parent.split("-")[0])
            assert parent_code == (code // 100) * 100
