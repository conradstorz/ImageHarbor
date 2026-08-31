"""The faces package must import with or without onnxruntime installed."""

import imageharbor.faces as faces


def test_package_imports_without_onnxruntime():
    # Importing the package must never require the optional extra. Only the
    # modules that actually run a model may import onnxruntime, and they are
    # imported lazily by the runner.
    assert hasattr(faces, "HAS_ONNX")
    assert isinstance(faces.HAS_ONNX, bool)
