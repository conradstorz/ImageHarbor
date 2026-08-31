"""The faces package must import with or without onnxruntime installed."""

import sys
import unittest.mock

import imageharbor.faces as faces


def test_package_imports_without_onnxruntime():
    # Importing the package must never require the optional extra. Only the
    # modules that actually run a model may import onnxruntime, and they are
    # imported lazily by the runner.
    assert hasattr(faces, "HAS_ONNX")
    assert isinstance(faces.HAS_ONNX, bool)


def test_package_imports_when_onnxruntime_raises_non_import_error():
    # A broken install (ABI-mismatch, partial installation, etc.) may raise
    # non-ImportError exceptions during import. The probe must catch all
    # exceptions, not just ImportError, and set HAS_ONNX to False so the
    # package still imports successfully.

    # Clear the imageharbor.faces module from sys.modules to force re-execution
    # of its import probe.
    sys.modules.pop("imageharbor.faces", None)
    to_remove = [k for k in list(sys.modules.keys()) if k.startswith("imageharbor.faces.")]
    for k in to_remove:
        sys.modules.pop(k)

    # Patch builtins.__import__ to raise RuntimeError for onnxruntime,
    # simulating a broken C extension or ABI mismatch. The original __import__
    # is called for all other modules.
    import builtins
    original_import = builtins.__import__

    def mock_import_with_broken_onnx(name, *args, **kwargs):
        if name == "onnxruntime":
            raise RuntimeError("ONNX Runtime initialization failed: ABI version mismatch")
        return original_import(name, *args, **kwargs)

    with unittest.mock.patch("builtins.__import__", side_effect=mock_import_with_broken_onnx):
        # Import the package with the patched import. It should succeed
        # because the probe catches all exceptions.
        import importlib
        import imageharbor.faces as faces_reloaded

        assert hasattr(faces_reloaded, "HAS_ONNX")
        # If the exception handler only caught ImportError, this would fail
        # because RuntimeError would propagate. With the proper fix, it is False.
        assert faces_reloaded.HAS_ONNX is False
