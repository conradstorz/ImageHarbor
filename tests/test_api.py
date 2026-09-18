"""Tests for the public facade in imageharbor.api."""

from __future__ import annotations

from pathlib import Path

import pytest

from imageharbor import api
from imageharbor.ai_classifier import StubClassifier


def test_exception_hierarchy():
    assert issubclass(api.ConfigError, api.ImageHarborError)
    assert issubclass(api.Aborted, api.ImageHarborError)
    assert api.Aborted("x").report is None


def test_outcome_constants_are_their_own_names():
    for name in ("COPIED", "DUPLICATE", "SKIPPED", "ERROR", "ENRICHED",
                 "RENAMED", "TOTAL", "AI", "IO", "OK", "FAILED"):
        assert getattr(api, name) == name


def test_aiconfig_defaults_match_the_cli():
    cfg = api.AIConfig()
    assert (cfg.backend, cfg.base_url, cfg.model, cfg.timeout, cfg.api_key) == (
        "stub", None, "gpt-4o-mini", 60.0, None
    )


def test_build_classifier_stub_default():
    assert isinstance(api._build_classifier(api.AIConfig()), StubClassifier)


def test_build_classifier_unknown_backend_is_a_config_error():
    with pytest.raises(api.ConfigError):
        api._build_classifier(api.AIConfig(backend="nope"))


def test_guard_rejects_dest_inside_source(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    with pytest.raises(api.ConfigError) as ei:
        api._guard_dest_not_inside_source(src, src / "d")
    assert "--dest" in str(ei.value) and "--source" in str(ei.value)


def test_guard_allows_sibling_dest(tmp_path: Path):
    src = tmp_path / "s"
    src.mkdir()
    api._guard_dest_not_inside_source(src, tmp_path / "d")  # no raise
