"""The scan pass: resumable, idempotent, and never a breaker failure."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, JpegImagePlugin

from imageharbor.catalog import Catalog
from imageharbor.dashboard import people
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


def test_a_photo_that_fails_mid_decode_does_not_leak_its_file_handle(
    library, monkeypatch
):
    """A decode that raises must still close the file it opened.

    The success path is not where this can go wrong: Pillow's
    ``ImageFile.load`` closes an exclusively-opened fp itself once the raster
    is in memory, so a scan of a healthy library releases handles regardless
    of how ``_scan_one`` is written -- which is why a happy-path version of
    this test would pass against the leaking code and prove nothing.

    The window is a file that *opens* and then fails: a valid header with a
    truncated body. ``Image.open`` returns an object holding an open fp,
    ``load`` raises, ``scan`` catches it and moves on, and the fp survives
    until the garbage collector happens to run. Over a long ``watch --faces``
    loop across a library with a tail of damaged files -- the population this
    error path exists for -- that is an accumulating file-descriptor leak.

    ``b"not an image"`` (used by the tests above) does not reach this: it
    fails inside ``Image.open`` itself, so no image object is ever handed
    back to leak.
    """
    dest, db, store = library
    truncated = dest / "photo1.jpg"
    intact = truncated.read_bytes()
    truncated.write_bytes(intact[: len(intact) // 2])

    opened = []
    real_open = Image.open

    def spy(fp, *args, **kwargs):
        image = real_open(fp, *args, **kwargs)
        opened.append(image)
        return image

    monkeypatch.setattr(runner.Image, "open", spy)

    cat = Catalog(db)
    result = runner.scan(cat, store, FakeDetector(), FakeEmbedder(),
                         dest / ".crops", gate=runner.QualityGate(0.5, 10))
    cat.close()

    assert result.errors == 1, (
        "the truncated file did not fail to decode -- this test is not "
        "exercising the path it claims to"
    )
    assert opened, "Image.open was never called; the spy is not wired up"
    for image in opened:
        assert image.fp is None or image.fp.closed, (
            f"{image} still holds an open file handle after scan() returned"
        )


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


# ---------------------------------------------------------------------------
# crop-rank contract: runner.py's file-naming rank and people.py's
# id-derived rank must agree even when a gate-rejected face sits at a
# lower id than the kept faces around it. See CLAUDE.md invariant 7 and
# dashboard/people.py's crop_bytes docstring.
# ---------------------------------------------------------------------------


class _ThreeDetectionDetector:
    """One low-score (gate-rejected) detection plus two kept detections
    positioned over visually distinct regions of the fixture image, so each
    kept face's aligned crop is pixel-distinguishable from the other's."""

    model_name = "yunet"

    def detect(self, image):
        return [
            # Rejected by the quality gate -- occupies the *first* slot in
            # `detections`, so it lands at the lowest id once inserted,
            # ahead of both kept faces below.
            Detection(
                x=0.0, y=100.0, w=50.0, h=50.0, score=0.1,
                landmarks=((10.0, 110.0), (30.0, 110.0), (20.0, 120.0),
                           (12.0, 130.0), (28.0, 130.0)),
            ),
            # Kept face "A" -- over the red region.
            Detection(
                x=0.0, y=0.0, w=50.0, h=50.0, score=0.9,
                landmarks=((20.0, 20.0), (40.0, 20.0), (30.0, 30.0),
                           (22.0, 42.0), (38.0, 42.0)),
            ),
            # Kept face "B" -- over the blue region, translated +100 in x.
            Detection(
                x=100.0, y=0.0, w=50.0, h=50.0, score=0.9,
                landmarks=((120.0, 20.0), (140.0, 20.0), (130.0, 30.0),
                           (122.0, 42.0), (138.0, 42.0)),
            ),
        ]


def _crop_rank_fixture(tmp_path):
    """A photo with a gate-rejected face ahead of two visually distinct kept
    faces, scanned through the real `_scan_one` write path against a real
    `FaceStore` and crop directory."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 80, 100], fill=(220, 20, 20))     # red -- face A
    draw.rectangle([100, 0, 180, 100], fill=(20, 20, 220))  # blue -- face B
    photo_path = tmp_path / "photo.jpg"
    img.save(photo_path)

    db = tmp_path / "catalog.db"
    Catalog(db).close()
    store = FaceStore(db)
    digest = "crops_digest_0123456789"
    crop_dir = tmp_path / ".crops"
    gate = runner.QualityGate(min_score=0.6, min_box=10)

    kept, rejected = runner._scan_one(
        photo_path, _ThreeDetectionDetector(), FakeEmbedder(), gate, crop_dir,
        digest, store,
    )
    assert (kept, rejected) == (2, 1)
    return store, crop_dir, digest


def test_crop_bytes_matches_each_kept_face_s_own_crop_past_a_gate_rejection(tmp_path):
    # This is the regression the review flagged: runner.py names crop files
    # by rank-among-kept (assigned via enumerate over the post-gate,
    # post-align loop), and people.py's crop_bytes re-derives that same rank
    # by filtering `faces` to `rejected IS NULL` and indexing by ascending
    # id. Nothing before this test exercised both halves of that contract
    # together with a rejected face actually present ahead of the kept
    # ones -- two independently-broken mutations (reversing runner.py's
    # kept-face append order, or dropping `rejected IS NULL` from people.py's
    # rank query) both pass all 239 pre-existing tests.
    store, crop_dir, digest = _crop_rank_fixture(tmp_path)

    photo_dir = crop_dir / digest[:2] / digest[2:4]
    crop_a_bytes = (photo_dir / f"{digest}-0.jpg").read_bytes()
    crop_b_bytes = (photo_dir / f"{digest}-1.jpg").read_bytes()
    # Sanity: the fixture is actually distinguishable, or a swap bug could
    # coincidentally still produce a byte-identical "match".
    assert crop_a_bytes != crop_b_bytes

    # Recover which face id is physically which face via bbox_x -- a column
    # neither dangerous mutation touches, so this identification is a valid
    # oracle regardless of which side of the id<->rank contract regresses.
    rows = store._conn.execute(
        "SELECT id, bbox_x FROM faces WHERE sha256_b64url=? AND rejected IS NULL "
        "ORDER BY bbox_x",
        (digest,),
    ).fetchall()
    assert len(rows) == 2
    face_id_a, face_id_b = rows[0]["id"], rows[1]["id"]

    assert people.crop_bytes(crop_dir, face_id_a, store=store) == crop_a_bytes
    assert people.crop_bytes(crop_dir, face_id_b, store=store) == crop_b_bytes
    store.close()


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
