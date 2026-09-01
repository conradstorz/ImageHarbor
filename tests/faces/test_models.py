"""The model registry declares what an ONNX graph cannot tell us."""

import pytest

from imageharbor.faces import models


def test_defaults_are_registered():
    assert models.DEFAULT_DETECTOR in models.DETECTORS
    assert models.DEFAULT_EMBEDDER in models.EMBEDDERS


def test_every_entry_declares_its_preprocessing_contract():
    # Channel order and normalization are NOT recoverable from an ONNX graph.
    # A wrong input shape raises; a wrong channel order loads, runs, and returns
    # plausible embeddings that are quietly worse. Every entry must state both.
    for info in {**models.DETECTORS, **models.EMBEDDERS}.values():
        assert info.channel_order in ("RGB", "BGR"), info.name
        assert len(info.input_size) == 2, info.name
        assert info.mean is not None and info.std is not None, info.name
        assert info.licence, info.name


def test_embedders_declare_a_dimension_and_detectors_do_not():
    for info in models.EMBEDDERS.values():
        assert info.embedding_dim and info.embedding_dim > 0, info.name
    for info in models.DETECTORS.values():
        assert info.embedding_dim is None, info.name


def test_filenames_are_disambiguated_by_publisher():
    # InsightFace's antelopev2 pack and fal's AuraFace both ship a file called
    # glintr100.onnx and they are different models. A name match is not an
    # artifact match, so our stored filename must not be the bare upstream one.
    auraface = models.EMBEDDERS["auraface"]
    assert auraface.filename != "glintr100.onnx"
    assert "auraface" in auraface.filename


def test_get_rejects_an_unknown_model():
    with pytest.raises(KeyError, match="unknown face model"):
        models.get("no-such-model")


def test_get_returns_registered_entries():
    assert models.get(models.DEFAULT_EMBEDDER).kind == "embedder"
    assert models.get(models.DEFAULT_DETECTOR).kind == "detector"
