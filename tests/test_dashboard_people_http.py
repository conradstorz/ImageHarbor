"""HTTP wiring for the People routes on the dashboard server.

`tests/faces/test_dashboard_people.py` exercises `imageharbor.dashboard.people`
directly; this file exercises the routing added to
`imageharbor.dashboard.server` on top of it (`do_GET`/`do_POST` dispatch,
status codes, the `store is None` degrade path) using the same fake-socket
harness as `tests/test_dashboard_server.py` -- see that module's docstring for
why a `BytesIO`-backed fake socket is the standard way to unit test a
`BaseHTTPRequestHandler` subclass with no port and no thread.
"""

from __future__ import annotations

import io
import json
from http.client import HTTPResponse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from imageharbor.catalog import Catalog
from imageharbor.dashboard import server as dashboard_server
from imageharbor.dashboard.control import ControlPlane
from imageharbor.faces import cluster
from imageharbor.faces.decode import Detection
from imageharbor.faces.store import FaceStore

# ---------------------------------------------------------------------------
# Fake-socket harness (same shape as tests/test_dashboard_server.py)
# ---------------------------------------------------------------------------


class _FakeSocket:
    def __init__(self, request_bytes: bytes) -> None:
        self._rfile = io.BytesIO(request_bytes)
        self.sent = bytearray()

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> io.BytesIO:
        if mode == "rb":
            return self._rfile
        raise AssertionError(f"unexpected makefile mode {mode!r}")

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def settimeout(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def fileno(self) -> int:
        return -1

    def close(self) -> None:
        pass


class _DummyServer:
    pass


def _raw_request(
    method: str, path: str, *, body: bytes = b"", headers: dict[str, str] | None = None
) -> bytes:
    lines = [f"{method} {path} HTTP/1.1", "Host: test"]
    hdrs = dict(headers or {})
    if body and "Content-Length" not in hdrs:
        hdrs["Content-Length"] = str(len(body))
    for key, value in hdrs.items():
        lines.append(f"{key}: {value}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
    return head + body


class _ResponseSocketLike:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def makefile(self, *_args: Any, **_kwargs: Any) -> io.BytesIO:
        return io.BytesIO(self._data)


def _dispatch(handler_cls: type, method: str, path: str, **kwargs: Any):
    raw = _raw_request(method, path, **kwargs)
    sock = _FakeSocket(raw)
    handler_cls(sock, ("127.0.0.1", 54321), _DummyServer())
    resp = HTTPResponse(_ResponseSocketLike(bytes(sock.sent)))
    resp.begin()
    body = resp.read()
    return resp.status, dict(resp.getheaders()), body


def _dispatch_json(handler_cls: type, method: str, path: str, payload: Any = None, **kwargs: Any):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    headers = kwargs.pop("headers", {}) or {}
    if payload is not None:
        headers.setdefault("Content-Type", "application/json")
    status, resp_headers, resp_body = _dispatch(
        handler_cls, method, path, body=body, headers=headers, **kwargs
    )
    parsed = None
    if resp_body:
        try:
            parsed = json.loads(resp_body.decode("utf-8"))
        except json.JSONDecodeError:
            parsed = None
    return status, resp_headers, parsed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _det():
    return Detection(x=0.0, y=0.0, w=50.0, h=50.0, score=0.9,
                     landmarks=((1.0, 1.0), (2.0, 1.0), (1.5, 2.0),
                                (1.0, 3.0), (2.0, 3.0)))


def _v(vals):
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def control(catalog: Catalog) -> ControlPlane:
    return ControlPlane(catalog, env_interval=300, env_enrich=True)


@pytest.fixture()
def face_store(tmp_path: Path):
    s = FaceStore(tmp_path / "catalog.db")
    yield s
    s.close()


@pytest.fixture()
def crop_dir(tmp_path: Path) -> Path:
    d = tmp_path / "face-crops"
    d.mkdir()
    return d


@pytest.fixture()
def handler_cls_no_store(catalog: Catalog, control: ControlPlane):
    return dashboard_server.make_handler(catalog, control)


@pytest.fixture()
def handler_cls(catalog: Catalog, control: ControlPlane, face_store: FaceStore, crop_dir: Path):
    return dashboard_server.make_handler(
        catalog, control, store=face_store, crop_dir=crop_dir
    )


def _one_cluster(store: FaceStore, faces: int = 2) -> int:
    ids: list[int] = []
    for i in range(faces):
        ids += store.record_scan(f"d{i}", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    store.replace_clusters(
        "auraface", [cluster.Cluster(face_ids=tuple(ids), centroid=_v([1, 0, 0]))]
    )
    return store.cluster_ids()[0]


# ---------------------------------------------------------------------------
# GET /api/people -- degrades cleanly with no store wired in
# ---------------------------------------------------------------------------


def test_api_people_404s_when_no_store_is_wired_in(handler_cls_no_store) -> None:
    status, _, body = _dispatch_json(handler_cls_no_store, "GET", "/api/people")
    assert status == 404
    assert body is not None and "error" in body


def test_api_people_returns_the_review_queue(handler_cls, face_store: FaceStore) -> None:
    _one_cluster(face_store, faces=2)
    status, headers, body = _dispatch_json(handler_cls, "GET", "/api/people")
    assert status == 200
    assert headers.get("Content-Type", "").startswith("application/json")
    assert len(body["clusters"]) == 1


def test_api_people_include_singletons_query_param(handler_cls, face_store: FaceStore) -> None:
    _one_cluster(face_store, faces=1)
    status, _, body = _dispatch_json(handler_cls, "GET", "/api/people")
    assert body["clusters"] == []
    assert body["singletons_hidden"] == 1

    status, _, body = _dispatch_json(
        handler_cls, "GET", "/api/people?include_singletons=1"
    )
    assert len(body["clusters"]) == 1


# ---------------------------------------------------------------------------
# GET /api/face-crop/<id>
# ---------------------------------------------------------------------------


def test_face_crop_returns_404_for_a_missing_crop(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/api/face-crop/999999")
    assert status == 404


def test_face_crop_returns_the_jpeg_bytes(
    handler_cls, face_store: FaceStore, crop_dir: Path
) -> None:
    digest = "abcdef0123456789"
    ids = face_store.record_scan(digest, "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    photo_dir = crop_dir / digest[:2] / digest[2:4]
    photo_dir.mkdir(parents=True)
    (photo_dir / f"{digest}-0.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")

    status, headers, body = _dispatch(handler_cls, "GET", f"/api/face-crop/{ids[0]}")
    assert status == 200
    assert headers.get("Content-Type") == "image/jpeg"
    assert headers.get("Cache-Control") == "no-cache"
    assert body == b"\xff\xd8\xff\xe0fake"


def test_face_crop_non_integer_id_returns_404_not_500(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/api/face-crop/not-a-number")
    assert status == 404


def test_face_crop_404s_when_no_store_is_wired_in(handler_cls_no_store) -> None:
    status, _, _ = _dispatch(handler_cls_no_store, "GET", "/api/face-crop/1")
    assert status == 404


# ---------------------------------------------------------------------------
# POST /api/people/confirm|reject|merge|split
# ---------------------------------------------------------------------------


def test_post_people_confirm_writes_the_person(handler_cls, face_store: FaceStore) -> None:
    cid = _one_cluster(face_store)
    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/confirm",
        {"cluster_id": cid, "name": "Emma"},
    )
    assert status == 200
    assert body["person_id"] == face_store.person_for_cluster(cid)


def test_post_people_confirm_bad_input_returns_400_not_500(
    handler_cls, face_store: FaceStore
) -> None:
    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/confirm",
        {"cluster_id": 9999, "name": "Emma"},
    )
    assert status == 400
    assert "error" in body


def test_post_people_reject_marks_the_proposal(handler_cls, face_store: FaceStore) -> None:
    from imageharbor.faces.attribute import Proposal

    cid = _one_cluster(face_store)
    face_store.record_proposals([Proposal(cid, "Emma", 14, 15, 14 / 15, 340)])
    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/reject", {"cluster_id": cid, "name": "Emma"}
    )
    assert status == 200
    assert face_store.proposals_for(cid)[0]["decided"] == "rejected"


def test_post_people_reject_unmatched_name_returns_400(
    handler_cls, face_store: FaceStore
) -> None:
    cid = _one_cluster(face_store)
    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/reject", {"cluster_id": cid, "name": "Nobody"}
    )
    assert status == 400
    assert "error" in body


def test_post_people_merge_points_clusters_at_one_person(
    handler_cls, face_store: FaceStore
) -> None:
    ids_a = face_store.record_scan("a0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    ids_b = face_store.record_scan("b0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    face_store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids_a), centroid=_v([1, 0, 0])),
        cluster.Cluster(face_ids=tuple(ids_b), centroid=_v([1, 0, 0])),
    ])
    cid_a, cid_b = face_store.cluster_ids()
    person_id = face_store.confirm(cid_a, "Emma")

    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/merge",
        {"person_id": person_id, "cluster_ids": [cid_b]},
    )
    assert status == 200
    assert face_store.person_for_cluster(cid_b) == person_id


def test_post_people_split_creates_a_new_cluster(handler_cls, face_store: FaceStore) -> None:
    cid = _one_cluster(face_store, faces=3)
    face_ids = sorted(
        r["id"] for r in face_store._conn.execute(
            "SELECT id FROM faces WHERE cluster_id=?", (cid,)
        )
    )
    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/split",
        {"cluster_id": cid, "face_ids": [face_ids[-1]]},
    )
    assert status == 200
    assert body["new_cluster_id"] != cid


def test_post_people_split_face_from_another_cluster_returns_400(
    handler_cls, face_store: FaceStore
) -> None:
    ids_a = face_store.record_scan("a0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    ids_b = face_store.record_scan("b0", "yunet", [(_det(), _v([1, 0, 0]), "auraface")])
    face_store.replace_clusters("auraface", [
        cluster.Cluster(face_ids=tuple(ids_a), centroid=_v([1, 0, 0])),
        cluster.Cluster(face_ids=tuple(ids_b), centroid=_v([1, 0, 0])),
    ])
    cid_a, cid_b = face_store.cluster_ids()

    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/people/split",
        {"cluster_id": cid_b, "face_ids": [ids_a[0]]},
    )
    assert status == 400
    assert "error" in body
    row = face_store._conn.execute(
        "SELECT cluster_id FROM faces WHERE id=?", (ids_a[0],)
    ).fetchone()
    assert row["cluster_id"] == cid_a


def test_post_people_unknown_action_returns_404(handler_cls) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/people/bogus", {})
    assert status == 404


def test_post_people_action_404s_when_no_store_is_wired_in(handler_cls_no_store) -> None:
    status, _, _ = _dispatch_json(
        handler_cls_no_store, "POST", "/api/people/confirm", {"cluster_id": 1, "name": "Emma"}
    )
    assert status == 404


def test_post_people_confirm_malformed_json_returns_400_not_a_traceback(handler_cls) -> None:
    status, _, body = _dispatch(
        handler_cls, "POST", "/api/people/confirm",
        body=b"{not valid json", headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert b"Traceback" not in body


def test_post_people_confirm_wrong_shape_body_returns_400(handler_cls) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/people/confirm", [1, 2, 3])
    assert status == 400


# ---------------------------------------------------------------------------
# Existing routes must not regress alongside the new prefix-matched ones
# ---------------------------------------------------------------------------


def test_unrelated_routes_still_work_with_a_store_wired_in(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/healthz")
    assert status == 200
    status, _, _ = _dispatch_json(handler_cls, "GET", "/api/stats")
    assert status == 200
    status, _, _ = _dispatch(handler_cls, "GET", "/does/not/exist")
    assert status == 404
