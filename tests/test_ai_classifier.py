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
    ContentDescription,
    OpenAIClassifier,
    PhotoClassification,
    StubClassifier,
    _build_pcs_list,
)
from imageharbor.pcs import PCS_VERSION


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
# StubClassifier — parent + label contract
# ---------------------------------------------------------------------------


def test_stub_returns_parent_and_label() -> None:
    c = StubClassifier().classify(Path("beach_sunset.jpg"), {}, taxonomy_snapshot="")
    assert c.top_parent == "300"
    assert c.label == "beach"
    assert c.sub_parent is None


def test_stub_unknown_is_misc() -> None:
    c = StubClassifier().classify(Path("random_xyz.jpg"), {}, taxonomy_snapshot="")
    assert c.top_parent == "900"
    assert c.label == "miscellaneous"


def test_stub_adjudicate_returns_none() -> None:
    assert StubClassifier().adjudicate("festivities", ["holidays"]) is None


# ---------------------------------------------------------------------------
# StubClassifier — determinism (a core project principle)
# ---------------------------------------------------------------------------


def test_stub_classify_is_deterministic() -> None:
    a = StubClassifier().classify(Path("my_dog.jpg"), {}, "")
    b = StubClassifier().classify(Path("my_dog.jpg"), {}, "")
    assert (a.top_parent, a.label, a.descriptor) == (b.top_parent, b.label, b.descriptor)


def test_stub_two_instances_agree() -> None:
    # Determinism must not depend on instance state.
    path = Path("my_dog.jpg")
    a = StubClassifier().classify(path, {}, "")
    b = StubClassifier().classify(path, {}, "")
    assert a.top_parent == b.top_parent
    assert a.label == b.label
    assert a.descriptor == b.descriptor


# ---------------------------------------------------------------------------
# StubClassifier — keyword mapping now expressed as (top_parent, label).
# Codes are scanned in keyword_map order; the FIRST match wins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem, top_parent, label",
    [
        # "beach" (330) is scanned before "sunset" (640).
        ("beach_sunset", "300", "beach"),
        # "portrait" (110) is scanned before "family" (120).
        ("family_portrait", "100", "portraits"),
        ("my_dog", "200", "pets"),
        ("eagle", "200", "birds"),
        # "receipt_scan" matches "scan" in the 710 (text) pattern.
        ("receipt_scan", "700", "text"),
        # Whole-word matching: "cathedral" -> 830 historic (not 210 via "cat").
        ("cathedral", "800", "historic"),
        # "texture" -> 930 abstract (not 710 via "text").
        ("texture", "900", "abstract"),
        # "oscar" no longer matches "car" -> falls through to 900 misc.
        ("oscar", "900", "miscellaneous"),
        # "location" no longer matches "cat" -> falls through to 900 misc.
        ("location", "900", "miscellaneous"),
        # No pattern matches -> default miscellaneous.
        ("random_gibberish_xyz", "900", "miscellaneous"),
    ],
)
def test_stub_keyword_mapping(stem: str, top_parent: str, label: str) -> None:
    result = StubClassifier().classify(Path(f"{stem}.jpg"), {}, "")
    assert result.top_parent == top_parent
    assert result.label == label


def test_stub_receipt_alone_hits_receipts_not_text() -> None:
    # With "receipt" removed from the 710 pattern, a bare "receipt" resolves to
    # the dedicated receipts category 730.
    result = StubClassifier().classify(Path("receipt.jpg"), {}, "")
    assert result.top_parent == "700"
    assert result.label == "receipts"


# ---------------------------------------------------------------------------
# StubClassifier — descriptor construction
# ---------------------------------------------------------------------------


def test_stub_descriptor_first_two_words() -> None:
    result = StubClassifier().classify(Path("beach_sunset.jpg"), {}, "")
    assert result.descriptor == "beach sunset"


def test_stub_descriptor_only_first_two_words() -> None:
    result = StubClassifier().classify(Path("red_car_on_road.jpg"), {}, "")
    assert result.descriptor == "red car"


def test_stub_descriptor_skips_single_char_words() -> None:
    # "a" is a single char and is dropped; "dog" and "run" remain.
    result = StubClassifier().classify(Path("a_dog_run.jpg"), {}, "")
    assert result.descriptor == "dog run"


def test_stub_descriptor_falls_back_to_photo_when_all_single_char() -> None:
    result = StubClassifier().classify(Path("a_b_c.jpg"), {}, "")
    assert result.descriptor == "photo"


def test_stub_descriptor_falls_back_to_photo_when_empty_stem() -> None:
    result = StubClassifier().classify(Path("_.jpg"), {}, "")
    assert result.descriptor == "photo"


# ---------------------------------------------------------------------------
# StubClassifier — returned object shape
# ---------------------------------------------------------------------------


def test_stub_returns_photo_classification_with_model_version() -> None:
    result = StubClassifier().classify(Path("eagle.jpg"), {}, "")
    assert isinstance(result, PhotoClassification)
    assert result.model_version == "stub-1.0"
    assert result.pcs_version == PCS_VERSION


def test_stub_is_an_ai_classifier() -> None:
    assert isinstance(StubClassifier(), AIClassifier)


# ---------------------------------------------------------------------------
# StubClassifier — exif_data does not influence output
# ---------------------------------------------------------------------------


def test_stub_exif_data_does_not_affect_output() -> None:
    path = Path("mountain_peak.jpg")
    empty = StubClassifier().classify(path, {}, "")
    populated = StubClassifier().classify(
        path,
        {"Make": "TestMake", "Model": "TestModel", "gps_lat": 41.5},
        "",
    )
    assert empty.top_parent == populated.top_parent
    assert empty.label == populated.label
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
        def __init__(
            self,
            api_key: str | None = None,
            base_url: str | None = None,
            timeout: float = 60.0,
        ) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout

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
        "top_parent": "300",
        "label": "beach",
        "descriptor": "beach-sunset",
        "caption": "A sunset over the beach.",
        "objects": ["sun", "sand", "water"],
        "secondary_tags": ["golden-hour", "coast"],
        "ocr_text": "SEASIDE",
    }
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, taxonomy_snapshot="300 places\n  330 beach")

    assert result.top_parent == "300"
    assert result.label == "beach"
    assert result.sub_parent is None
    assert result.descriptor == "beach-sunset"
    assert result.caption == "A sunset over the beach."
    assert result.objects == ["sun", "sand", "water"]
    assert result.secondary_tags == ["golden-hour", "coast"]
    assert result.ocr_text == "SEASIDE"
    assert result.model_version == "gpt-4o-mini"
    # Network was actually invoked (mocked).
    assert clf._client.chat.completions.create.called


def test_openai_classify_embeds_snapshot_in_system_prompt(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    # The live taxonomy snapshot must be passed to the model in the system
    # message so it can reuse existing labels.
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(
        json.dumps({"top_parent": "300", "label": "beach"})
    )

    snapshot = "300 places\n  330 beach\n  340 mountains"
    clf.classify(tiny_image, {}, taxonomy_snapshot=snapshot)

    _, kwargs = clf._client.chat.completions.create.call_args
    system_msg = next(m["content"] for m in kwargs["messages"] if m["role"] == "system")
    assert snapshot in system_msg


def test_openai_classify_parses_sub_parent(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {"top_parent": "500", "label": "holidays", "sub_parent": "540"}
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, "")
    assert result.top_parent == "500"
    assert result.label == "holidays"
    assert result.sub_parent == "540"


def test_openai_classify_invalid_json_falls_back(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("this is not json {")

    result = clf.classify(tiny_image, {}, "")

    assert result.top_parent == "900"
    assert result.label == "miscellaneous"
    assert result.descriptor == "photo"
    assert result.caption == ""


def test_openai_classify_missing_fields_default(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    # Well-formed JSON that omits top_parent/label falls back to misc defaults.
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {"descriptor": "mystery", "caption": "Unknown thing."}
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, "")

    assert result.top_parent == "900"
    assert result.label == "miscellaneous"
    # Non-code fields from the model are still preserved.
    assert result.descriptor == "mystery"
    assert result.caption == "Unknown thing."


# ---------------------------------------------------------------------------
# OpenAIClassifier — robustness against well-formed JSON with wrong types
# ---------------------------------------------------------------------------


def test_openai_top_parent_coerced_to_string(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    # A numeric top_parent must be coerced to a string (codes are strings now).
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {"top_parent": 300, "label": "beach"}
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, "")
    assert result.top_parent == "300"
    assert isinstance(result.top_parent, str)


def test_openai_objects_as_string_becomes_empty_list(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    # A bare string must not be char-split into a list of characters.
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {"top_parent": "300", "label": "beach", "objects": "sunset", "secondary_tags": "coast"}
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, "")

    assert result.label == "beach"
    assert result.objects == []
    assert result.secondary_tags == []


def test_openai_objects_as_non_iterable_becomes_empty_list(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    # A non-iterable (int) must not raise a TypeError.
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {"top_parent": "300", "label": "beach", "objects": 5, "secondary_tags": 7}
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, "")

    assert result.objects == []
    assert result.secondary_tags == []


def test_openai_objects_list_elements_stringified(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    # Valid list input is preserved (elements coerced to str).
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    payload = {"top_parent": "300", "label": "beach", "objects": ["sun", 42], "secondary_tags": ["coast"]}
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.classify(tiny_image, {}, "")

    assert result.objects == ["sun", "42"]
    assert result.secondary_tags == ["coast"]


# ---------------------------------------------------------------------------
# OpenAIClassifier — adjudicate
# ---------------------------------------------------------------------------


def test_openai_adjudicate_parses_matching_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("holidays")

    assert clf.adjudicate("festivities", ["sports", "holidays"]) == "holidays"
    assert clf._client.chat.completions.create.called


def test_openai_adjudicate_returns_none_when_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("NONE")

    assert clf.adjudicate("festivities", ["sports", "holidays"]) is None


def test_openai_classifier_passes_base_url_model_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("openai")
    fake.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)

    clf = OpenAIClassifier(
        api_key=None,
        model="llava",
        base_url="http://jetson.local:11434/v1",
        timeout=30.0,
    )

    assert captured["base_url"] == "http://jetson.local:11434/v1"
    assert captured["timeout"] == 30.0
    assert captured["api_key"] == "not-needed"  # placeholder when none supplied
    assert clf._model == "llava"
    assert clf.MODEL_VERSION == "llava"


# ---------------------------------------------------------------------------
# ContentDescription / describe / pick_class — perception contract (additive)
# ---------------------------------------------------------------------------


def test_content_description_and_stub_describe() -> None:
    c = StubClassifier().describe(Path("marching_band_2007.jpg"), {})
    assert isinstance(c, ContentDescription)
    assert c.primary_subject == "marching"  # first >1-char word of the stem
    # deterministic
    assert StubClassifier().describe(Path("marching_band_2007.jpg"), {}).primary_subject == "marching"


def test_stub_describe_returns_expected_fields() -> None:
    c = StubClassifier().describe(Path("marching_band_2007.jpg"), {})
    assert c.caption == "Stub description for marching_band_2007.jpg"
    assert c.tags == ["marching", "band", "2007"]
    assert c.model_version == "stub-1.0"


def test_stub_describe_falls_back_to_photo_when_empty_stem() -> None:
    c = StubClassifier().describe(Path("_.jpg"), {})
    assert c.primary_subject == "photo"


def test_stub_pick_class_default_900() -> None:
    c = ContentDescription(primary_subject="mystery")
    assert StubClassifier().pick_class(c, [("100", "people"), ("900", "miscellaneous")]) == "900"


def test_openai_describe_parses_content(monkeypatch: pytest.MonkeyPatch, tiny_image: Path) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")

    payload = {
        "primary_subject": "dog",
        "scene": "backyard",
        "objects": ["dog", "grass", "ball"],
        "caption": "A dog playing in a backyard.",
        "tags": ["pet", "outdoor"],
        "ocr_text": "",
    }
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response(json.dumps(payload))

    result = clf.describe(tiny_image, {})

    assert isinstance(result, ContentDescription)
    assert result.primary_subject == "dog"
    assert result.scene == "backyard"
    assert result.objects == ["dog", "grass", "ball"]
    assert result.caption == "A dog playing in a backyard."
    assert result.tags == ["pet", "outdoor"]
    assert result.ocr_text == ""
    assert result.model_version == "gpt-4o-mini"
    assert clf._client.chat.completions.create.called


def test_openai_describe_invalid_json_falls_back(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("not json {")

    result = clf.describe(tiny_image, {})

    assert result.primary_subject == "photo"
    assert result.objects == []
    assert result.tags == []


def test_openai_pick_class_returns_valid_code(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("500")

    content = ContentDescription(primary_subject="band", scene="stage")
    result = clf.pick_class(content, [("100", "people"), ("500", "events")])

    assert result == "500"
    assert clf._client.chat.completions.create.called


def test_openai_pick_class_invalid_reply_falls_back_to_900(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    _install_fake_openai(monkeypatch)
    clf = OpenAIClassifier(api_key="unused")
    clf._client = Mock()
    clf._client.chat.completions.create.return_value = _mock_response("I have no idea")

    content = ContentDescription(primary_subject="mystery")
    result = clf.pick_class(content, [("100", "people"), ("500", "events")])

    assert result == "900"
