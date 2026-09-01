"""The faces command group.

Every subcommand that runs a model gates on `imageharbor.faces.HAS_ONNX`.
`store.py` (and therefore `cluster`/`calibrate`/`status`, which all open a
`FaceStore`) imports numpy unconditionally, and numpy ships only inside the
`faces` extra in `pyproject.toml` -- not as a core dependency -- so none of
those three can actually run without the extra either, despite `scan` being
the only one that touches onnxruntime directly. `HAS_ONNX` is the only
importability signal the package exposes, and in practice `uv sync --extra
faces` installs numpy and onnxruntime together, so gating on it here turns a
raw `ModuleNotFoundError` into the same clear, actionable message `scan`
gives. Only `models download` (pure `hashlib`/`urllib`, no numpy) is
genuinely extra-free, so it alone is exempt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from imageharbor.catalog import Catalog
from imageharbor.cli import _faces_model_dir, main
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore

# These tests exercise the branch the *real* import state opens -- they do not
# monkeypatch `HAS_ONNX`, they need it to be genuinely True -- so they need the
# `faces` extra actually installed. Under the documented `uv sync --extra dev`
# they skip rather than fail, the same way tests/faces/test_detect.py and
# test_embed.py already do via `pytest.importorskip`. `find_spec` rather than
# `faces.HAS_ONNX`: a marker is evaluated at collection time, and reading the
# flag would make this sensitive to whichever test last monkeypatched it.
requires_faces_extra = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="needs the faces extra (uv sync --extra faces)",
)


def _det(x: float = 10.0) -> Detection:
    return Detection(
        x=x, y=10.0, w=50.0, h=50.0, score=0.9,
        landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0), (22.0, 42.0), (38.0, 42.0)),
    )


def _v(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


def _write_sidecar(dest: Path, digest: str, name: str) -> None:
    (dest / f"{digest}.json").write_text(
        json.dumps({
            "schema_version": 2,
            "identity": {"sha256_b64url": digest},
            "people": [{"name": name, "source": "google_photos_people"}],
        }),
        encoding="utf-8",
    )


def _seed_store(dest: Path) -> Path:
    """Create <dest>/catalog.db with schema for both Catalog and FaceStore."""
    catalog_path = dest / "catalog.db"
    Catalog(catalog_path).close()
    return catalog_path


# ---------------------------------------------------------------------------
# Brief's Step 1 tests, verbatim
# ---------------------------------------------------------------------------


def test_faces_group_is_registered():
    result = CliRunner().invoke(main, ["faces", "--help"])
    assert result.exit_code == 0
    for sub in ("scan", "cluster", "calibrate", "status", "models"):
        assert sub in result.output


def test_scan_requires_a_destination():
    result = CliRunner().invoke(main, ["faces", "scan"])
    assert result.exit_code != 0
    assert "--dest" in result.output


@requires_faces_extra
def test_status_on_an_empty_library_reports_zero(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "status", "--dest", str(dest)])
    assert result.exit_code == 0
    assert "0" in result.output


def test_scan_without_onnxruntime_fails_with_a_clear_message(tmp_path, monkeypatch):
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "scan", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "faces" in result.output and "extra" in result.output


# ---------------------------------------------------------------------------
# Additional coverage for the corrections documented in the task report
# ---------------------------------------------------------------------------


def test_cluster_requires_a_threshold(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "cluster", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "--threshold" in result.output


def test_cluster_without_onnxruntime_fails_clearly(tmp_path, monkeypatch):
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(
        main, ["faces", "cluster", "--dest", str(dest), "--threshold", "0.5"]
    )
    assert result.exit_code != 0
    assert "faces" in result.output and "extra" in result.output


def test_calibrate_without_onnxruntime_fails_clearly(tmp_path, monkeypatch):
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "calibrate", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "faces" in result.output and "extra" in result.output


def test_status_without_onnxruntime_also_fails_clearly(tmp_path, monkeypatch):
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "status", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "faces" in result.output and "extra" in result.output


def test_models_download_works_without_onnxruntime(tmp_path, monkeypatch):
    """The one subcommand that is genuinely extra-free: `download.py` never
    imports numpy or onnxruntime, only hashlib/urllib."""
    import imageharbor.faces as faces_pkg
    import imageharbor.faces.download as download_mod

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)

    calls = []

    def fake_ensure(info, model_dir):
        calls.append(info.name)
        path = Path(model_dir) / info.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        return path

    monkeypatch.setattr(download_mod, "ensure", fake_ensure)

    model_dir = tmp_path / "models"
    result = CliRunner().invoke(
        main, ["faces", "models", "download", "--model-dir", str(model_dir)]
    )
    assert result.exit_code == 0, result.output
    assert calls == ["yunet", "auraface"]


def test_models_download_requires_a_model_dir_or_dest(monkeypatch):
    monkeypatch.delenv("IMAGEHARBOR_FACE_MODEL_DIR", raising=False)
    result = CliRunner().invoke(main, ["faces", "models", "download"])
    assert result.exit_code != 0
    assert "model" in result.output.lower()


def test_models_download_falls_back_to_dest(tmp_path, monkeypatch):
    import imageharbor.faces.download as download_mod

    monkeypatch.delenv("IMAGEHARBOR_FACE_MODEL_DIR", raising=False)
    calls = []

    def fake_ensure(info, model_dir):
        calls.append(Path(model_dir))
        path = Path(model_dir) / info.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        return path

    monkeypatch.setattr(download_mod, "ensure", fake_ensure)

    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "models", "download", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert calls == [dest / ".faces-models", dest / ".faces-models"]


def test_faces_model_dir_precedence(tmp_path, monkeypatch):
    """explicit arg > env var > <dest>/.faces-models -- a reordering of the
    checks in _faces_model_dir survives every other CLI test in this file
    because they each only exercise one tier of the fallback at a time."""
    dest = tmp_path / "dest"
    env_dir = tmp_path / "env-models"
    monkeypatch.setenv("IMAGEHARBOR_FACE_MODEL_DIR", str(env_dir))

    resolved = _faces_model_dir(None, dest)
    assert resolved == env_dir
    assert resolved != dest / ".faces-models"

    explicit_dir = tmp_path / "explicit-models"
    assert _faces_model_dir(explicit_dir, dest) == explicit_dir


@requires_faces_extra
def test_cluster_builds_clusters_and_reports_proposal_counts(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    catalog_path = _seed_store(dest)

    store = FaceStore(catalog_path)
    for i in range(3):
        store.record_scan(f"d{i}", "yunet", [(_det(), _v([1, 0.01 * i, 0]), "auraface")])
    store.close()

    _write_sidecar(dest, "d0", "Emma")
    _write_sidecar(dest, "d1", "Emma")

    result = CliRunner().invoke(
        main, ["faces", "cluster", "--dest", str(dest), "--threshold", "0.5"]
    )
    assert result.exit_code == 0, result.output
    assert "clusters=1" in result.output
    assert "proposals=1" in result.output


@requires_faces_extra
def test_cluster_refuses_a_second_run_without_recluster(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    catalog_path = _seed_store(dest)
    store = FaceStore(catalog_path)
    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.close()

    cli = CliRunner()
    first = cli.invoke(main, ["faces", "cluster", "--dest", str(dest), "--threshold", "0.5"])
    assert first.exit_code == 0, first.output

    second = cli.invoke(main, ["faces", "cluster", "--dest", str(dest), "--threshold", "0.5"])
    assert second.exit_code != 0
    assert "--recluster" in second.output


@requires_faces_extra
def test_cluster_reclusters_when_the_flag_is_passed(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    catalog_path = _seed_store(dest)
    store = FaceStore(catalog_path)
    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.close()

    cli = CliRunner()
    cli.invoke(main, ["faces", "cluster", "--dest", str(dest), "--threshold", "0.5"])
    result = cli.invoke(
        main,
        ["faces", "cluster", "--dest", str(dest), "--threshold", "0.5", "--recluster"],
    )
    assert result.exit_code == 0, result.output


@requires_faces_extra
def test_calibrate_reports_a_measured_threshold(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    catalog_path = _seed_store(dest)

    store = FaceStore(catalog_path)
    rng = np.random.default_rng(0)
    for i in range(12):
        base = np.array([1.0, 0.0, 0.0]) if i < 6 else np.array([0.0, 1.0, 0.0])
        vec = base + rng.normal(0, 0.02, 3)
        store.record_scan(f"d{i}", "yunet", [(_det(), _v(vec), "auraface")])
        _write_sidecar(dest, f"d{i}", "Emma" if i < 6 else "Judy")
    store.close()

    result = CliRunner().invoke(main, ["faces", "calibrate", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "threshold=" in result.output
    assert "precision=" in result.output
    assert "recall=" in result.output
    assert "faces cluster" in result.output


@requires_faces_extra
def test_calibrate_reports_needs_two_names(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    catalog_path = _seed_store(dest)
    store = FaceStore(catalog_path)
    store.record_scan("d0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.close()
    _write_sidecar(dest, "d0", "Emma")

    result = CliRunner().invoke(main, ["faces", "calibrate", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "two distinct" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["faces", "--help"],
        ["faces", "scan", "--help"],
        ["faces", "cluster", "--help"],
        ["faces", "calibrate", "--help"],
        ["faces", "status", "--help"],
        ["faces", "roster", "--help"],
        ["faces", "models", "--help"],
        ["faces", "models", "download", "--help"],
    ],
)
def test_every_subcommand_help_renders(args):
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# faces roster: Picasa roster names as autocomplete vocabulary, never
# attached to a photo or cluster (Task 17).
# ---------------------------------------------------------------------------


def _seed_roster(dest: Path, sample: bytes) -> None:
    room = dest / ".takeout-provenance" / "abc"
    room.mkdir(parents=True)
    (room / "contacts.xml").write_bytes(sample)


_ROSTER_SAMPLE = b"""<?xml version="1.0"?>
<contacts>
  <contact id="a1" name="Conrad Storz"/>
  <contact id="b2" name="Gladys Blankenbeker "/>
</contacts>
"""


@requires_faces_extra
def test_roster_imports_names_and_reports_the_count(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    _seed_store(dest)
    _seed_roster(dest, _ROSTER_SAMPLE)

    result = CliRunner().invoke(main, ["faces", "roster", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "2" in result.output

    store = FaceStore(dest / "catalog.db")
    assert sorted(store.known_names()) == ["Conrad Storz", "Gladys Blankenbeker"]
    store.close()


@requires_faces_extra
def test_roster_import_is_idempotent_through_the_cli(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    _seed_store(dest)
    _seed_roster(dest, _ROSTER_SAMPLE)

    runner = CliRunner()
    runner.invoke(main, ["faces", "roster", "--dest", str(dest)])
    result = runner.invoke(main, ["faces", "roster", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "0" in result.output


def test_roster_without_onnxruntime_fails_clearly(tmp_path, monkeypatch):
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    dest = tmp_path / "dest"
    dest.mkdir()
    result = CliRunner().invoke(main, ["faces", "roster", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "faces" in result.output and "extra" in result.output


@requires_faces_extra
def test_roster_with_no_provenance_directory_reports_zero(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    _seed_store(dest)

    result = CliRunner().invoke(main, ["faces", "roster", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert "0" in result.output
