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
    StubClassifier,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_image(path: Path) -> Path:
    """Write a tiny real JPEG so describe()'s base64 read works."""
    Image.new("RGB", (2, 2), "white").save(path, "JPEG")
    return path


@pytest.fixture()
def tiny_image(tmp_path: Path) -> Path:
    return _make_image(tmp_path / "sample.jpg")


# ---------------------------------------------------------------------------
# StubClassifier — base contract
# ---------------------------------------------------------------------------


def test_stub_is_an_ai_classifier() -> None:
    assert isinstance(StubClassifier(), AIClassifier)


def test_stub_adjudicate_returns_none() -> None:
    assert StubClassifier().adjudicate("festivities", ["holidays"]) is None


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
# ContentDescription / describe / pick_class — perception contract
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


def test_stub_describe_is_an_ai_classifier_method() -> None:
    # describe is the abstract method now; StubClassifier satisfies the ABC.
    assert isinstance(StubClassifier().describe(Path("dog.jpg"), {}), ContentDescription)


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
