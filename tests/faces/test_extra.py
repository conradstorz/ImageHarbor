"""The faces package must import with or without onnxruntime installed."""

import builtins
import sys

import imageharbor
import imageharbor.faces as faces


def test_package_imports_without_onnxruntime():
    # Importing the package must never require the optional extra. Only the
    # modules that actually run a model may import onnxruntime, and they are
    # imported lazily by the runner.
    assert hasattr(faces, "HAS_ONNX")
    assert isinstance(faces.HAS_ONNX, bool)


def test_package_imports_when_onnxruntime_raises_non_import_error(monkeypatch):
    # A broken install (ABI-mismatch, partial installation, etc.) may raise
    # non-ImportError exceptions during import. The probe must catch all
    # exceptions, not just ImportError, and set HAS_ONNX to False so the
    # package still imports successfully.

    # Clear the imageharbor.faces module from sys.modules to force re-execution
    # of its import probe. Use monkeypatch.delitem (not sys.modules.pop) so the
    # original module object is restored at teardown regardless of whether the
    # test passes or fails — a bare pop() here leaked the reloaded, ONNX-less
    # module into every later test in the session.
    #
    # This alone is NOT enough (found by Task 13's CLI tests, which failed
    # only when the full suite ran, never in isolation): `import
    # imageharbor.faces` below re-executes the package and, as a side effect
    # of the import machinery, overwrites `imageharbor`'s own `faces`
    # attribute -- not just the `sys.modules["imageharbor.faces"]` entry.
    # `imageharbor/cli.py`'s `from . import faces as faces_pkg` resolves
    # through that attribute, not through `sys.modules` directly, so without
    # the extra `monkeypatch.setattr` below, the poisoned HAS_ONNX=False
    # module stayed live for the rest of the session even though
    # `sys.modules` looked clean.
    original_faces_module = sys.modules["imageharbor.faces"]
    monkeypatch.setattr(imageharbor, "faces", original_faces_module)
    monkeypatch.delitem(sys.modules, "imageharbor.faces", raising=False)
    to_remove = [k for k in list(sys.modules.keys()) if k.startswith("imageharbor.faces.")]
    for k in to_remove:
        monkeypatch.delitem(sys.modules, k, raising=False)

    # Patch builtins.__import__ to raise RuntimeError for onnxruntime,
    # simulating a broken C extension or ABI mismatch. The original __import__
    # is called for all other modules.
    original_import = builtins.__import__

    def mock_import_with_broken_onnx(name, *args, **kwargs):
        if name == "onnxruntime":
            raise RuntimeError("ONNX Runtime initialization failed: ABI version mismatch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_with_broken_onnx)

    # Import the package with the patched import. It should succeed
    # because the probe catches all exceptions.
    import imageharbor.faces as faces_reloaded

    assert hasattr(faces_reloaded, "HAS_ONNX")
    # If the exception handler only caught ImportError, this would fail
    # because RuntimeError would propagate. With the proper fix, it is False.
    assert faces_reloaded.HAS_ONNX is False


def test_watch_no_faces_no_dashboard_survives_missing_numpy_and_onnxruntime(
    monkeypatch, tmp_path
):
    """`watch --no-faces --no-dashboard` must not require numpy or
    onnxruntime -- see this package's module docstring ("importing this
    package must never require the optional `faces` extra"), `cli.py`'s
    `--faces` help text, and CLAUDE.md's "a missing extra degrades to one
    warning, not a crash" invariant.

    Regression test for the whole-branch-review CRITICAL finding: `watch`
    imports `dashboard.server` unconditionally, before the `--no-dashboard`
    check further down; `dashboard/server.py` (and the `dashboard.people`/
    `dashboard.stats` modules it imports) pulled in
    `imageharbor.faces.store` at module scope, which does `import numpy as
    np` at module scope -- so numpy was required even with both `--no-faces`
    and `--no-dashboard` given.

    Blocks both `numpy` and `onnxruntime` at the same `builtins.__import__`
    patch point `test_package_imports_when_onnxruntime_raises_non_import_error`
    above uses. Every module in the numpy-reachable chain is first cleared
    from `sys.modules` (same technique as that test, for the same reason):
    this dev environment has the `faces` extra installed, so an earlier
    test's real numpy-backed import of these modules would otherwise stay
    cached and this test would pass even against the unfixed bug.
    """
    from imageharbor import watcher as _watcher
    from imageharbor.watcher import WatchStats

    original_faces_module = sys.modules["imageharbor.faces"]
    monkeypatch.setattr(imageharbor, "faces", original_faces_module)

    to_remove = [
        k
        for k in list(sys.modules.keys())
        if k == "imageharbor.faces"
        or k.startswith("imageharbor.faces.")
        or k
        in (
            "imageharbor.dashboard.server",
            "imageharbor.dashboard.people",
            "imageharbor.dashboard.stats",
        )
    ]
    for k in to_remove:
        monkeypatch.delitem(sys.modules, k, raising=False)

    original_import = builtins.__import__

    def mock_import_blocking_numpy_and_onnx(name, *args, **kwargs):
        if name in ("numpy", "onnxruntime"):
            raise ImportError(f"No module named {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_blocking_numpy_and_onnx)

    # Bound the watch loop to one pass -- same pattern as
    # tests/test_cli.py's `_fake_watch_cli` helper -- so this test doesn't
    # block on the real (looping) watcher.
    monkeypatch.setattr(_watcher, "watch", lambda **kwargs: WatchStats(passes=1))

    from click.testing import CliRunner

    from imageharbor.cli import main

    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "watch", "--source", str(src), "--dest", str(dest),
            "--no-faces", "--no-dashboard",
        ],
    )
    assert result.exit_code == 0, result.output
