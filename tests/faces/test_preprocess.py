"""The preprocessing contract each model declares must actually be applied.

These tests need no model weights, on purpose. A wrong channel order or
normalization does not raise -- it produces plausible embeddings that are
quietly worse -- so asserting the declared contract directly is the only cheap
way to catch it. Expectations are derived from `models.py` at run time rather
than hardcoded, so that changing the registry and forgetting the implementation
is what fails, not a correct registry change.
"""

import numpy as np
import pytest
from PIL import Image

from imageharbor.faces import models
from imageharbor.faces.preprocess import build_blob

AURAFACE = models.EMBEDDERS["auraface"]
YUNET = models.DETECTORS["yunet"]

# Channels deliberately distinct so a swap or a transposed axis cannot pass.
R, G, B = 10, 20, 30


def _solid(info, rgb=(R, G, B)):
    return Image.new("RGB", info.input_size, rgb)


def _expected(value, info):
    return (value - info.mean) / info.std


@pytest.mark.parametrize("info", [AURAFACE, YUNET], ids=lambda i: i.name)
def test_blob_is_nchw_with_the_declared_geometry(info):
    blob = build_blob([_solid(info)], info)
    width, height = info.input_size
    assert blob.shape == (1, 3, height, width)
    assert blob.dtype == np.float32


@pytest.mark.parametrize("info", [AURAFACE, YUNET], ids=lambda i: i.name)
def test_channel_axis_holds_channels_in_the_declared_order(info):
    # The whole point: a (1,3,H,W) array can have the right shape and the wrong
    # axis semantics. Each plane must be constant and equal to its own channel.
    blob = build_blob([_solid(info)], info)
    order = [R, G, B] if info.channel_order == "RGB" else [B, G, R]
    for plane, source_value in enumerate(order):
        expected = _expected(source_value, info)
        assert np.allclose(blob[0, plane], expected), (
            f"{info.name} plane {plane} should carry {source_value} "
            f"under {info.channel_order}"
        )


def test_rgb_and_bgr_models_disagree_on_the_same_pixels():
    # Guards against both registries being read but the swap never applied:
    # if this passes trivially, the channel-order branch is dead.
    rgb_blob = build_blob([_solid(AURAFACE)], AURAFACE)
    bgr_info = models.ModelInfo(**{**AURAFACE.__dict__, "channel_order": "BGR"})
    bgr_blob = build_blob([_solid(AURAFACE)], bgr_info)
    assert not np.allclose(rgb_blob, bgr_blob)
    # Specifically, the outer planes are swapped and the middle one is not.
    assert np.allclose(rgb_blob[0, 0], bgr_blob[0, 2])
    assert np.allclose(rgb_blob[0, 1], bgr_blob[0, 1])
    assert np.allclose(rgb_blob[0, 2], bgr_blob[0, 0])


@pytest.mark.parametrize("info", [AURAFACE, YUNET], ids=lambda i: i.name)
def test_normalization_uses_the_declared_mean_and_std(info):
    blob = build_blob([_solid(info, (0, 0, 0)), _solid(info, (255, 255, 255))], info)
    assert np.allclose(blob[0], _expected(0.0, info))
    assert np.allclose(blob[1], _expected(255.0, info))


def test_batching_keeps_each_image_in_its_own_row():
    # A future "optimization" must not reorder or reuse rows.
    colours = [(0, 0, 0), (255, 255, 255), (R, G, B)]
    blob = build_blob([_solid(AURAFACE, c) for c in colours], AURAFACE)
    assert blob.shape[0] == len(colours)
    for row, (r, _g, _b) in enumerate(colours):
        assert np.allclose(blob[row, 0], _expected(r, AURAFACE))


def test_a_wrongly_sized_image_is_resized_to_the_declared_input():
    oversized = Image.new("RGB", (400, 250), (R, G, B))
    width, height = AURAFACE.input_size
    assert build_blob([oversized], AURAFACE).shape == (1, 3, height, width)


def test_non_rgb_modes_convert_rather_than_producing_garbage():
    grey = Image.new("L", AURAFACE.input_size, 128)
    blob = build_blob([grey], AURAFACE)
    assert blob.shape == (1, 3, *reversed(AURAFACE.input_size))
    # Grey converts to equal R, G, B, so every plane must agree.
    assert np.allclose(blob[0, 0], blob[0, 1])
    assert np.allclose(blob[0, 1], blob[0, 2])


def test_empty_input_returns_an_empty_batch_of_the_right_shape():
    width, height = AURAFACE.input_size
    assert build_blob([], AURAFACE).shape == (0, 3, height, width)


def test_an_unknown_channel_order_raises_rather_than_guessing():
    bogus = models.ModelInfo(**{**AURAFACE.__dict__, "channel_order": "GRB"})
    with pytest.raises(ValueError, match="unknown channel order"):
        build_blob([_solid(AURAFACE)], bogus)
