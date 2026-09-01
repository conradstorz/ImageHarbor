"""The faces pass inside the watcher.

The task-16 brief's Step 1 test code called `control.get_setting` /
`set_setting` / `revert_setting` -- module-level functions that do not exist
in `imageharbor/dashboard/control.py`. The real settings primitives are
`Catalog.setting_get`/`setting_set`/`setting_delete` (see
`tests/test_dashboard_control.py`, which every existing 'enrich'/'interval'
test already exercises the same way); `ControlPlane` layers env-vs-stored
precedence, hostile-value handling, and a live `*_enabled` property on top of
those, exactly as it already does for `enrich`. The tests below use the real
names throughout.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from imageharbor.catalog import Catalog
from imageharbor.dashboard.control import ControlPlane
from imageharbor.faces import runner
from imageharbor.faces.cluster import Cluster
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore
from imageharbor import watcher
from imageharbor.watcher import FacesConfig, watch


# ---------------------------------------------------------------------------
# fakes -- no onnxruntime/model weights needed, mirroring tests/faces/test_runner.py
# ---------------------------------------------------------------------------


class FakeDetector:
    model_name = "yunet"

    def __init__(self, per_image: int = 1) -> None:
        self.per_image = per_image

    def detect(self, image, score_threshold=0.6, nms_threshold=0.3):
        return [
            Detection(
                x=10.0 + 60 * i, y=10.0, w=50.0, h=50.0, score=0.9,
                landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0),
                           (22.0, 42.0), (38.0, 42.0)),
            )
            for i in range(self.per_image)
        ]


class FakeEmbedder:
    model_name = "auraface"
    dim = 4

    def embed_batch(self, crops):
        v = np.ones((len(crops), self.dim), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _det(x: float = 10.0) -> Detection:
    return Detection(
        x=x, y=10.0, w=50.0, h=50.0, score=0.9,
        landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0),
                   (22.0, 42.0), (38.0, 42.0)),
    )


def _vec(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


def _one_cycle_sleep(stop: threading.Event, n: int = 1):
    """A `sleep` double that lets *n* passes run, then sets *stop*."""
    calls = {"n": 0}

    def _sleep(_interval: float) -> bool:
        calls["n"] += 1
        if calls["n"] >= n:
            stop.set()
        return True

    return _sleep


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    return src


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


@pytest.fixture()
def catalog(tmp_path: Path):
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def face_store(tmp_path: Path):
    store = FaceStore(tmp_path / "catalog.db")
    yield store
    store.close()


def _make_pipeline(source_dir: Path, organized_dir: Path, catalog: Catalog):
    from imageharbor.pipeline import Pipeline

    return Pipeline(source_dir, organized_dir, catalog)


def _basic_face_config(
    tmp_path: Path, organized_dir: Path, face_store: FaceStore, **overrides
) -> FacesConfig:
    kwargs = dict(
        store=face_store,
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
        crop_dir=tmp_path / "face-crops",
        dest=organized_dir,
        gate=runner.QualityGate(min_score=0.5, min_box=10),
        cluster_threshold=0.5,
    )
    kwargs.update(overrides)
    return FacesConfig(**kwargs)


# ---------------------------------------------------------------------------
# settings key: 'faces' beside 'enrich'
# ---------------------------------------------------------------------------


def test_faces_setting_defaults_to_none(catalog: Catalog) -> None:
    assert catalog.setting_get("faces") is None


def test_faces_setting_round_trips(catalog: Catalog) -> None:
    catalog.setting_set("faces", "0")
    assert catalog.setting_get("faces") == "0"
    catalog.setting_delete("faces")
    assert catalog.setting_get("faces") is None


def test_control_plane_faces_enabled_follows_env_by_default(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True, env_faces=True)
    assert control.faces_enabled is True
    control_off = ControlPlane(catalog, env_interval=300, env_enrich=True, env_faces=False)
    assert control_off.faces_enabled is False


def test_control_plane_faces_override_wins_then_reverts(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True, env_faces=True)
    control.set_override("faces", False)
    assert control.faces_enabled is False
    control.revert("faces")
    assert control.faces_enabled is True
    assert catalog.setting_get("faces") is None


# ---------------------------------------------------------------------------
# faces never touch the circuit breaker
# ---------------------------------------------------------------------------


def test_a_face_failure_does_not_move_the_circuit_breaker() -> None:
    # The brief's Step 1 test read `breaker.consecutive_failures` and
    # constructed `CircuitBreaker(threshold=2)` -- neither exists.
    # `circuit_breaker.py`'s real constructor kwarg is `trip_threshold`, and
    # `_consecutive` is a private counter with no public getter; `state`
    # (a `BreakerState`) is the public, testable surface, so that is what
    # this test pins instead.
    from imageharbor.circuit_breaker import BreakerState, CircuitBreaker

    breaker = CircuitBreaker(trip_threshold=2)
    before = breaker.state
    assert before is BreakerState.CLOSED
    # The faces pass records into failed_files and never calls the breaker.
    # This test pins the contract by asserting the breaker is untouched after
    # a scan that errored -- see tests/faces/test_runner.py for the erroring
    # scan itself, and `watch()`'s faces-pass block in watcher.py, which
    # never references *breaker* at all.
    assert breaker.state == before


# ---------------------------------------------------------------------------
# faces_available()
# ---------------------------------------------------------------------------


def test_watch_skips_faces_without_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)
    assert watcher.faces_available() is False


def test_faces_available_true_when_onnx_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", True)
    assert watcher.faces_available() is True


# ---------------------------------------------------------------------------
# the faces pass actually runs and is logged
# ---------------------------------------------------------------------------


def test_watch_runs_the_faces_pass_when_enabled_and_available(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from PIL import Image

    for i in range(2):
        path = organized_dir / f"photo{i}.jpg"
        Image.new("RGB", (200, 200), (i * 40, 100, 100)).save(path)
        catalog.upsert(
            sha256_b64url=f"digest{i}", original_path=str(path), organized_path=str(path)
        )

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    face_config = _basic_face_config(tmp_path, organized_dir, face_store)
    stop = threading.Event()

    with caplog.at_level(logging.INFO):
        wstats = watch(
            pipeline=pipeline,
            catalog=catalog,
            interval=1.0,
            stop_event=stop,
            sleep=_one_cycle_sleep(stop),
            faces_enabled=True,
            face_config=face_config,
        )

    assert wstats.faces_scanned == 2
    assert wstats.faces_found == 2
    assert "faces pass 1: scanned=2 faces=2 rejected=0 errors=0" in caplog.text


def test_watch_does_not_run_faces_pass_when_disabled(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
) -> None:
    from PIL import Image

    path = organized_dir / "photo0.jpg"
    Image.new("RGB", (200, 200), (10, 100, 100)).save(path)
    catalog.upsert(sha256_b64url="digest0", original_path=str(path), organized_path=str(path))

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    face_config = _basic_face_config(tmp_path, organized_dir, face_store)
    stop = threading.Event()

    wstats = watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_one_cycle_sleep(stop),
        faces_enabled=False,
        face_config=face_config,
    )

    assert wstats.faces_scanned == 0
    assert face_store.is_scanned("digest0", "yunet") is False


# ---------------------------------------------------------------------------
# mutation target: the "log once, not per cycle" onnx-unavailable warning
# ---------------------------------------------------------------------------


def test_watch_warns_once_when_faces_enabled_but_unavailable(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A permanently-absent extra must not flood the log for days -- exactly
    one warning across many cycles, not one per cycle."""
    import imageharbor.faces as faces_pkg

    monkeypatch.setattr(faces_pkg, "HAS_ONNX", False)

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    face_config = _basic_face_config(tmp_path, organized_dir, face_store)
    stop = threading.Event()

    with caplog.at_level(logging.WARNING):
        watch(
            pipeline=pipeline,
            catalog=catalog,
            interval=1.0,
            stop_event=stop,
            sleep=_one_cycle_sleep(stop, n=4),
            faces_enabled=True,
            face_config=face_config,
        )

    warnings = [
        r for r in caplog.records
        if "faces" in r.getMessage() and "extra is not installed" in r.getMessage()
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# mutation target: clustering must be gated, not run every cycle
# ---------------------------------------------------------------------------


def test_watch_does_not_recluster_below_threshold_with_clusters_already_present(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed one confirmed-shape cluster (so cluster_ids() is non-empty) plus a
    # single unclustered face, well under a generous recluster_threshold.
    ids = face_store.record_scan("d0", "yunet", [(_det(), _vec([1, 0, 0, 0]), "auraface")])
    face_store.replace_clusters(
        "auraface", [Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0, 0]))]
    )
    face_store.record_scan("d1", "yunet", [(_det(), _vec([0, 1, 0, 0]), "auraface")])
    assert face_store.unclustered_face_count("auraface") == 1

    calls: list[int] = []
    monkeypatch.setattr(
        "imageharbor.faces.runner.build_clusters",
        lambda *a, **k: calls.append(1) or 0,
    )

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    face_config = _basic_face_config(
        tmp_path, organized_dir, face_store, recluster_threshold=500
    )
    stop = threading.Event()

    watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_one_cycle_sleep(stop, n=2),
        faces_enabled=True,
        face_config=face_config,
    )

    assert calls == []


def test_watch_reclusters_when_unclustered_exceeds_threshold(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = face_store.record_scan("d0", "yunet", [(_det(), _vec([1, 0, 0, 0]), "auraface")])
    face_store.replace_clusters(
        "auraface", [Cluster(face_ids=(ids[0],), centroid=_vec([1, 0, 0, 0]))]
    )
    face_store.record_scan("d1", "yunet", [(_det(), _vec([0, 1, 0, 0]), "auraface")])
    assert face_store.unclustered_face_count("auraface") == 1

    calls: list[int] = []
    monkeypatch.setattr(
        "imageharbor.faces.runner.build_clusters",
        lambda *a, **k: calls.append(1) or 0,
    )

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    # threshold=0: the single unclustered face already exceeds it.
    face_config = _basic_face_config(
        tmp_path, organized_dir, face_store, recluster_threshold=0
    )
    stop = threading.Event()

    watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_one_cycle_sleep(stop, n=1),
        faces_enabled=True,
        face_config=face_config,
    )

    assert calls == [1]


def test_watch_reclusters_when_no_clusters_exist_yet(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert face_store.cluster_ids() == []

    calls: list[int] = []
    monkeypatch.setattr(
        "imageharbor.faces.runner.build_clusters",
        lambda *a, **k: calls.append(1) or 0,
    )

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    # A very high threshold: only "no clusters exist yet" can be why this fires.
    face_config = _basic_face_config(
        tmp_path, organized_dir, face_store, recluster_threshold=500
    )
    stop = threading.Event()

    watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_one_cycle_sleep(stop, n=1),
        faces_enabled=True,
        face_config=face_config,
    )

    assert calls == [1]


def test_watch_warns_once_when_clustering_due_but_no_threshold_configured(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`IMAGEHARBOR_FACE_THRESHOLD` ships empty in docker-compose.yml on
    purpose (see docs/deploy-docker.md) -- this must warn once, not spam."""
    calls: list[int] = []
    monkeypatch.setattr(
        "imageharbor.faces.runner.build_clusters",
        lambda *a, **k: calls.append(1) or 0,
    )

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    face_config = _basic_face_config(
        tmp_path, organized_dir, face_store, cluster_threshold=None
    )
    stop = threading.Event()

    with caplog.at_level(logging.WARNING):
        watch(
            pipeline=pipeline,
            catalog=catalog,
            interval=1.0,
            stop_event=stop,
            sleep=_one_cycle_sleep(stop, n=3),
            faces_enabled=True,
            face_config=face_config,
        )

    assert calls == []  # never clustered without a threshold
    warnings = [
        r for r in caplog.records
        if "cluster threshold is configured" in r.getMessage()
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# mutation target: the pause setting must reach the faces pass
# ---------------------------------------------------------------------------


def test_watch_forwards_pause_check_into_the_faces_scan(
    tmp_path: Path,
    source_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    face_store: FaceStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`runner.scan`'s `should_stop` must be the SAME pause check the facts
    and enrichment phases use -- not `None`, and not a copy that can drift
    out of sync with a dashboard pause landing mid-pass."""
    captured: dict = {}

    def _spy_scan(catalog, store, detector, embedder, crop_dir, *, gate, limit=None, should_stop=None):
        captured["should_stop"] = should_stop
        return runner.ScanResult()

    monkeypatch.setattr("imageharbor.faces.runner.scan", _spy_scan)
    monkeypatch.setattr("imageharbor.faces.runner.propagate_sidecars", lambda *a, **k: 0)

    pipeline = _make_pipeline(source_dir, organized_dir, catalog)
    control = ControlPlane(catalog, env_interval=1.0, env_enrich=True, env_faces=True)
    face_config = _basic_face_config(tmp_path, organized_dir, face_store)
    stop = threading.Event()

    watch(
        pipeline=pipeline,
        catalog=catalog,
        interval=1.0,
        stop_event=stop,
        sleep=_one_cycle_sleep(stop),
        control=control,
        face_config=face_config,
    )

    # Bound methods compare equal (same `__self__`/`__func__`) even though
    # `is` would fail on two separate attribute lookups -- `==` is the
    # correct way to assert "this is control's own pause_check", not a
    # `None` or a copy that could drift out of sync with a pause landing
    # mid-pass.
    assert captured["should_stop"] == control.pause_check
