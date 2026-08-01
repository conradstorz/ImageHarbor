"""Tests for imageharbor.pcs."""

from imageharbor.pcs import (
    PCS_CATEGORIES,
    VALID_CODES,
    PCS_VERSION,
    get_category,
    resolve_code,
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
