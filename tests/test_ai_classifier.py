"""Tests for imageharbor.ai_classifier."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from imageharbor.ai_classifier import (
    AIClassifier,
    OpenAIClassifier,
    PhotoClassification,
    StubClassifier,
    _build_pcs_list,
)
from imageharbor.pcs import PCS_CATEGORIES, PCS_VERSION


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_image(path: Path) -> Path:
    """Write a tiny real JPEG so classify()'s base64 read works."""
    Image.new("RGB", (2, 2), "white").save(path, "JPEG")
    return path


@pytest.fixture()
def tiny_image(tmp_path: Path) -> Path:
    return _make_image(tmp_path / "sample.jpg")


# ---------------------------------------------------------------------------
# StubClassifier — determinism (a core project principle)
# ---------------------------------------------------------------------------


def test_stub_is_deterministic() -> None:
    stub = StubClassifier()
    path = Path("beach_sunset.jpg")
    first = stub.classify(path, {})
    second = stub.classify(path, {})
    assert first.pcs_code == second.pcs_code
    assert first.descriptor == second.descriptor


def test_stub_two_instances_agree() -> None:
    # Determinism must not depend on instance state.
    path = Path("my_dog.jpg")
    a = StubClassifier().classify(path, {})
    b = StubClassifier().classify(path, {})
    assert a.pcs_code == b.pcs_code
    assert a.descriptor == b.descriptor


# ---------------------------------------------------------------------------
# StubClassifier — keyword mapping (expected codes derived from keyword_map
# order; patterns are ORed regexes and the FIRST match wins).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem, expected_code",
    [
        # "beach" (330) is scanned before "sunset" (640).
        ("beach_sunset", 330),
        # "portrait" (110) is scanned before "family" (120).
        ("family_portrait", 110),
        ("my_dog", 210),
        ("eagle", 230),
        # "scan"/"receipt" first appear in the 710 pattern
        # ("...|receipt|scan"), which precedes the 730 "receipt|invoice|bill"
        # pattern -> 710 shadows the dedicated receipts code 730.
        ("receipt_scan", 710),
        # No pattern matches -> default miscellaneous.
        ("random_gibberish_xyz", 900),
    ],
)
def test_stub_keyword_mapping(stem: str, expected_code: int) -> None:
    stub = StubClassifier()
    result = stub.classify(Path(f"{stem}.jpg"), {})
    assert result.pcs_code == expected_code


def test_stub_receipt_alone_still_hits_710_not_730() -> None:
    # Documents the shadowing explicitly: a bare "receipt" resolves to 710,
    # never to the dedicated receipts category 730.
    stub = StubClassifier()
    assert stub.classify(Path("receipt.jpg"), {}).pcs_code == 710


# ---------------------------------------------------------------------------
# StubClassifier — descriptor construction
# ---------------------------------------------------------------------------


def test_stub_descriptor_first_two_words() -> None:
    stub = StubClassifier()
    result = stub.classify(Path("beach_sunset.jpg"), {})
    assert result.descriptor == "beach sunset"


def test_stub_descriptor_only_first_two_words() -> None:
    stub = StubClassifier()
    result = stub.classify(Path("red_car_on_road.jpg"), {})
    assert result.descriptor == "red car"


def test_stub_descriptor_skips_single_char_words() -> None:
    stub = StubClassifier()
    # "a" is a single char and is dropped; "dog" and "run" remain.
    result = stub.classify(Path("a_dog_run.jpg"), {})
    assert result.descriptor == "dog run"


def test_stub_descriptor_falls_back_to_photo_when_all_single_char() -> None:
    stub = StubClassifier()
    result = stub.classify(Path("a_b_c.jpg"), {})
    assert result.descriptor == "photo"


def test_stub_descriptor_falls_back_to_photo_when_empty_stem() -> None:
    stub = StubClassifier()
    result = stub.classify(Path("_.jpg"), {})
    assert result.descriptor == "photo"


# ---------------------------------------------------------------------------
# StubClassifier — returned object shape
# ---------------------------------------------------------------------------


def test_stub_returns_photo_classification_with_model_version() -> None:
    stub = StubClassifier()
    result = stub.classify(Path("eagle.jpg"), {})
    assert isinstance(result, PhotoClassification)
    assert result.model_version == "stub-1.0"
    assert result.pcs_version == PCS_VERSION


def test_stub_is_an_ai_classifier() -> None:
    assert isinstance(StubClassifier(), AIClassifier)


# ---------------------------------------------------------------------------
# StubClassifier — exif_data does not influence output
# ---------------------------------------------------------------------------


def test_stub_exif_data_does_not_affect_output() -> None:
    stub = StubClassifier()
    path = Path("mountain_peak.jpg")
    empty = stub.classify(path, {})
    populated = stub.classify(
        path,
        {"Make": "TestMake", "Model": "TestModel", "gps_lat": 41.5},
    )
    assert empty.pcs_code == populated.pcs_code
    assert empty.descriptor == populated.descriptor
    assert empty.caption == populated.caption


# ---------------------------------------------------------------------------
# _build_pcs_list
# ---------------------------------------------------------------------------


def test_build_pcs_list_non_empty_string() -> None:
    result = _build_pcs_list()
    assert isinstance(result, str)
    assert result.strip()


def test_build_pcs_list_contains_top_level_codes_and_labels() -> None:
    result = _build_pcs_list()
    assert "100: People" in result
    assert "900: Miscellaneous" in result


def test_build_pcs_list_indents_sub_categories() -> None:
    result = _build_pcs_list()
    # Sub-categories (parent is not None) are indented by two spaces.
    assert "  110: Portraits" in result
    assert "  330: Beach" in result
    # Top-level entries are not indented.
    assert "\n100: People" in ("\n" + result)


# ---------------------------------------------------------------------------
# OpenAIClassifier — missing package
# ---------------------------------------------------------------------------


def test_openai_missing_package_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setting the module to None in sys.modules makes `import openai` raise
    # ImportError without needing the package to actually be absent.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError) as excinfo:
        OpenAIClassifier(api_key="unused")
    assert "openai" in str(excinfo.value)
    assert "imageharbor[openai]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# OpenAIClassifier — mocked network
# ---------------------------------------------------------------------------


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake `openai` module so OpenAIClassifier can be constructed
    offline. The real client is replaced afterwards via clf._client."""
    fake = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key

    fake.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)


def _mock_response(content: str) -> Mock:
    resp = Mock()
    message = Mock()
    message.content = content
    choice = Mock()
    choice.message = message
    resp.choices = [choice]
    return resp


def test_openai_classify_success(monkeypatch: pytest.MonkeyPatch, tiny_image: Path) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused", model="gpt-4o-mini")

    payload = {
        "pcs_code": 330,
        "descriptor": "beach-sunset",
        "caption": "A sunset over the beach.",
        "objects": ["sun", "sand", "water"],
        "secondary_tags": ["golden-hour", "coast"],
        "ocr_text": "SEASIDE",
    }
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {})

    assert result.pcs_code == 330
    assert result.descriptor == "beach-sunset"
    assert result.caption == "A sunset over the beach."
    assert result.objects == ["sun", "sand", "water"]
    assert result.secondary_tags == ["golden-hour", "coast"]
    assert result.ocr_text == "SEASIDE"
    assert result.model_version == "gpt-4o-mini"
    # Network was actually invoked (mocked).
    assert clf._client.chat.completions.create.called


def test_openai_classify_invalid_json_falls_back(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("this is not json {")

    result = clf.classify(tiny_image, {})

    assert result.pcs_code == 900
    assert result.descriptor == "photo"
    assert result.caption == ""


def test_openai_classify_unknown_code_coerced_to_900(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {
        "pcs_code": 999,  # not a valid PCS code
        "descriptor": "mystery",
        "caption": "Unknown thing.",
    }
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {})

    assert 999 not in PCS_CATEGORIES
    assert result.pcs_code == 900
    # Non-code fields from the model are still preserved.
    assert result.descriptor == "mystery"
    assert result.caption == "Unknown thing."
