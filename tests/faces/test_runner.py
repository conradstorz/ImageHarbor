"""The scan pass: resumable, idempotent, and never a breaker failure."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from imageharbor.catalog import Catalog
from imageharbor.faces import runner
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore


class FakeDetector:
    model_name = "yunet"

    def __init__(self, per_image=1):
        self.per_image = per_image
        self.calls = 0

    def detect(self, image, score_threshold=0.6, nms_threshold=0.3):
        self.calls += 1
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


@pytest.fixture
def library(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    db = tmp_path / "catalog.db"
    cat = Catalog(db)
    for i in range(3):
        path = dest / f"photo{i}.jpg"
        Image.new("RGB", (200, 200), (i * 40, 100, 100)).save(path)
        # Catalog's real write method is `upsert`, not `record_photo` -- the
        # brief's fixture assumed an API that does not exist. See
        # imageharbor/catalog.py.
        cat.upsert(sha256_b64url=f"digest{i}", original_path=str(path),
                   organized_path=str(path))
    cat.close()
    store = FaceStore(db)
    yield dest, db, store
    store.close()


def test_scan_records_every_photo(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert result.scanned == 3
    assert result.faces == 3


def test_a_second_scan_is_a_no_op(library):
    dest, db, store = library
    det = FakeDetector()
    for _ in range(2):
        cat = Catalog(db)
        runner.scan(cat, store, det, FakeEmbedder(), dest / ".crops",
                    gate=runner.QualityGate(0.5, 10))
        cat.close()
    assert det.calls == 3          # each photo detected once, never twice
    assert store.stats()["faces"] == 3


def test_the_quality_gate_rejects_small_faces_without_dropping_them(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 999))
    cat.close()
    assert result.faces == 0
    assert result.rejected == 3
    # Marked, not omitted: the rows exist with a reason.
    assert store.stats()["faces"] == 3


def test_an_unreadable_photo_is_an_error_not_a_crash(library):
    dest, db, store = library
    (dest / "photo1.jpg").write_bytes(b"not an image")
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert result.errors == 1
    assert result.scanned == 2


def test_an_unreadable_photo_is_recorded_in_failed_files_not_the_breaker(library):
    # Task 11's own invariant: a face-scan failure is filesystem/decode
    # evidence, not AI-backend evidence, so it goes through the SAME
    # `failed_files` write the enrichment pass uses (`record_file_failure`)
    # and must never reach the circuit breaker. See CLAUDE.md invariant 3
    # and docs/superpowers/plans/2026-08-31-face-recognition.md Task 16's
    # `test_a_face_failure_does_not_move_the_circuit_breaker`, which pins the
    # other half of this same contract.
    dest, db, store = library
    bad_path = dest / "photo1.jpg"
    bad_path.write_bytes(b"not an image")
    cat = Catalog(db)
    runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                dest / ".crops", gate=runner.QualityGate(0.5, 10))
    row = cat._conn.execute(
        "SELECT * FROM failed_files WHERE source_path=?", (str(bad_path),)
    ).fetchone()
    cat.close()
    assert row is not None
    assert "faces" in row["last_error"].lower()


def test_should_stop_halts_between_photos(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10),
                         should_stop=lambda: True)
    cat.close()
    assert result.scanned == 0


def test_limit_bounds_the_pass(library):
    dest, db, store = library
    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10), limit=2)
    cat.close()
    assert result.scanned == 2


def test_crops_are_written_for_kept_faces(library):
    dest, db, store = library
    cat = Catalog(db)
    runner.scan(cat, store, FakeDetector(), FakeEmbedder(), dest / ".crops",
                gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert list((dest / ".crops").rglob("*.jpg"))


def test_draft_is_called_before_decode_for_every_photo(library, monkeypatch):
    # This is the single biggest win in the loop (a 12 MP JPEG decodes at
    # ~640x640 in the DCT domain instead of full-size) and nothing else in
    # this suite would notice its removal -- the fakes don't care how the
    # image got decoded. Spy on the real `Image.Image.draft` so a silent
    # regression here (a 10x slowdown at scale) actually fails a test.
    dest, db, store = library
    calls = []
    # The fixture's photos save as JPEG, and JpegImageFile overrides
    # Image.Image.draft with its own DCT-domain implementation -- patching
    # the base class method would silently miss every real call.
    original_draft = JpegImagePlugin.JpegImageFile.draft

    def spy_draft(self, mode, size):
        calls.append((mode, size))
        return original_draft(self, mode, size)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy_draft)
    cat = Catalog(db)
    runner.scan(cat, store, FakeDetector(), FakeEmbedder(), dest / ".crops",
                gate=runner.QualityGate(0.5, 10))
    cat.close()
    assert calls == [("RGB", runner.DECODE_SIZE)] * 3


def test_rejected_face_reasons_are_distinguishable(library):
    # A rejected-for-quality face and a rejected-for-degenerate-landmarks
    # face must not collapse into the same opaque marker -- see
    # imageharbor/faces/store.py's `faces.rejected` column, which exists
    # precisely to carry a reason.
    dest, db, store = library
    cat = Catalog(db)
    runner.scan(cat, store, FakeDetector(), FakeEmbedder(), dest / ".crops",
                gate=runner.QualityGate(0.5, 999), limit=1)
    cat.close()
    row = store._conn.execute("SELECT rejected FROM faces LIMIT 1").fetchone()
    assert row["rejected"] is not None
