"""Model acquisition verifies before it trusts."""

import hashlib

import pytest

from imageharbor.faces import download, models


def test_a_matching_file_is_accepted(tmp_path):
    art = tmp_path / "m.onnx"
    art.write_bytes(b"weights")
    info = models.ModelInfo(
        name="t", kind="detector", filename="m.onnx", url="http://example/m",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        input_size=(1, 1), channel_order="RGB", mean=0.0, std=1.0, licence="MIT",
    )
    assert download.ensure(info, tmp_path) == art


def test_a_mismatched_file_raises_rather_than_being_used(tmp_path):
    art = tmp_path / "m.onnx"
    art.write_bytes(b"tampered")
    info = models.ModelInfo(
        name="t", kind="detector", filename="m.onnx", url="http://example/m",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        input_size=(1, 1), channel_order="RGB", mean=0.0, std=1.0, licence="MIT",
    )
    with pytest.raises(download.ChecksumMismatch, match="m.onnx"):
        download.ensure(info, tmp_path)


def test_an_unpinned_model_refuses_to_verify_silently(tmp_path):
    art = tmp_path / "m.onnx"
    art.write_bytes(b"weights")
    info = models.ModelInfo(
        name="t", kind="detector", filename="m.onnx", url="http://example/m",
        sha256=None,
        input_size=(1, 1), channel_order="RGB", mean=0.0, std=1.0, licence="MIT",
    )
    with pytest.raises(download.ChecksumMismatch, match="no pinned checksum"):
        download.ensure(info, tmp_path)


def test_both_shipped_models_have_pinned_checksums():
    for info in {**models.DETECTORS, **models.EMBEDDERS}.values():
        assert info.sha256, f"{info.name} has no pinned checksum"
        assert len(info.sha256) == 64, info.name
