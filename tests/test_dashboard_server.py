"""Tests for imageharbor.dashboard.server.

Per the task brief (docs/superpowers/specs/2026-08-19-dashboard-design.md,
"HTTP surface"), the handler is exercised directly against a fake socket --
no real port is bound -- except for the one test that specifically needs a
real, already-bound port to prove `serve()` degrades to a logged warning
instead of raising (see the design's "A dashboard failure must never stop
the watcher").

The fake-socket harness below is the standard way to unit test a
`BaseHTTPRequestHandler` subclass: `BaseRequestHandler.__init__` runs
`setup()` / `handle()` / `finish()` synchronously against whatever
request-like object it is given, so a `BytesIO`-backed fake socket lets a
single HTTP request/response round-trip happen with no socket, no thread,
and no port.
"""

from __future__ import annotations

import io
import json
import shutil
import socket
import tempfile
import threading
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
from imageharbor.faces.store import FaceStore, ScannedFace

# ---------------------------------------------------------------------------
# Fake-socket harness
# ---------------------------------------------------------------------------


class _FakeSocket:
    """A minimal stand-in for a connected socket.

    Supports exactly what `http.server.BaseHTTPRequestHandler` /
    `socketserver.StreamRequestHandler` touch during one request/response
    cycle: `makefile('rb', ...)` for reading the request, and `sendall` for
    writing the response (via `socketserver._SocketWriter`, which wraps the
    raw socket object rather than calling `makefile('wb', ...)` when
    `wbufsize == 0`, which is BaseHTTPRequestHandler's default).
    """

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
    """Placeholder for the `server` argument BaseRequestHandler stores.

    Nothing the handler does in this module reads attributes off it -- see
    module docstring -- so it is intentionally empty.
    """


def _raw_request(
    method: str,
    path: str,
    *,
    host: str | None = "localhost",
    body: bytes = b"",
    headers: dict[str, str | None] | None = None,
    extra_header_lines: list[str] | None = None,
) -> bytes:
    """Build a raw HTTP/1.1 request.

    A POST gets ``Content-Type: application/json`` by default (R2 Task 3 --
    every real POST in this API sends JSON) -- pass ``headers={"Content-Type":
    "text/plain"}`` to override it, or ``headers={"Content-Type": None}`` to
    omit the header entirely. ``extra_header_lines`` appends raw header lines
    verbatim, after the normal ones -- used by the duplicate-Host-header test,
    which needs two ``Host:`` lines and a plain ``dict`` can't hold that.
    """
    lines = [f"{method} {path} HTTP/1.1"]
    if host is not None:
        lines.append(f"Host: {host}")
    hdrs: dict[str, str | None] = dict(headers or {})
    if method == "POST" and "Content-Type" not in hdrs:
        hdrs["Content-Type"] = "application/json"
    resolved = {k: v for k, v in hdrs.items() if v is not None}
    if body and "Content-Length" not in resolved:
        resolved["Content-Length"] = str(len(body))
    for key, value in resolved.items():
        lines.append(f"{key}: {value}")
    for line in extra_header_lines or []:
        lines.append(line)
    # HTTP header bytes are ISO-8859-1 (Latin-1) on the wire -- that's what
    # `http.client`/`http.server` actually encode/decode with -- NOT UTF-8.
    # A header value within Latin-1's range (e.g. "café") round-trips
    # correctly only when encoded that way here; encoding as UTF-8 would
    # send two-byte sequences that the server's Latin-1 decode would then
    # mangle into mojibake, which is a bug in this harness, not in the
    # server under test.
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    return head + body


class _ResponseSocketLike:
    """Wraps raw response bytes so `http.client.HTTPResponse` can parse them."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def makefile(self, *_args: Any, **_kwargs: Any) -> io.BytesIO:
        return io.BytesIO(self._data)


def _dispatch(
    handler_cls: type,
    method: str,
    path: str,
    *,
    extra_header_lines: list[str] | None = None,
    **kwargs: Any,
):
    """Send one request through *handler_cls* with no socket, return the response.

    Returns (status, headers-dict, body-bytes).
    """
    raw = _raw_request(method, path, extra_header_lines=extra_header_lines, **kwargs)
    sock = _FakeSocket(raw)
    handler_cls(sock, ("127.0.0.1", 54321), _DummyServer())
    resp = HTTPResponse(_ResponseSocketLike(bytes(sock.sent)))
    resp.begin()
    body = resp.read()
    return resp.status, dict(resp.getheaders()), body


def _request(
    method: str,
    path: str,
    *,
    host: str | None = "localhost",
    allowed_hosts: frozenset[str] = frozenset(),
    body: Any = b"",
    headers: dict[str, str] | None = None,
    token_configured: str | None = None,
    token_header: str | None = None,
):
    """Dispatch one request through a *fresh, throwaway* handler.

    Used by the Host-header allowlist tests and the token-gate tests below,
    both of which care about a gate firing before any routing -- not about a
    seeded catalog -- so each call gets its own disposable in-memory
    catalog/control rather than reaching for the module's `handler_cls`
    fixture (which bakes in fixed `allowed_hosts=frozenset()`/`token=None`
    at fixture-construction time, before a test body gets to choose either).

    ``body`` accepts a JSON-serializable value in addition to raw bytes, for
    the token tests below that don't otherwise need `_dispatch_json`'s
    response-parsing.  ``token_configured`` becomes the handler's `token`;
    ``token_header`` -- when given -- becomes the request's
    `X-Dashboard-Token` header.
    """
    if not isinstance(body, (bytes, bytearray)):
        body = json.dumps(body).encode("utf-8")
    hdrs = dict(headers or {})
    if token_header is not None:
        hdrs["X-Dashboard-Token"] = token_header
    tmp_dir = tempfile.mkdtemp()
    cat = Catalog(Path(tmp_dir) / "catalog.db")
    try:
        ctrl = ControlPlane(cat, env_interval=300, env_enrich=True)
        handler_cls = dashboard_server.make_handler(
            cat, ctrl, allowed_hosts=allowed_hosts, token=token_configured
        )
        return _dispatch(
            handler_cls, method, path, host=host, body=body, headers=hdrs
        )
    finally:
        cat.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


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


@pytest.fixture()
def control(catalog: Catalog) -> ControlPlane:
    return ControlPlane(catalog, env_interval=300, env_enrich=True)


@pytest.fixture()
def handler_cls(catalog: Catalog, control: ControlPlane):
    return dashboard_server.make_handler(catalog, control)


@pytest.fixture()
def face_store(tmp_path: Path, catalog: Catalog):
    # Same db_path as `catalog` -- see tests/test_dashboard_people_http.py,
    # which wires catalog+face_store together the same way.
    s = FaceStore(tmp_path / "catalog.db")
    yield s
    s.close()


def _det() -> Detection:
    return Detection(
        x=0.0, y=0.0, w=50.0, h=50.0, score=0.9,
        landmarks=((1.0, 1.0), (2.0, 1.0), (1.5, 2.0), (1.0, 3.0), (2.0, 3.0)),
    )


def _vec(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


def _seed_faces(store: FaceStore) -> dict:
    """Same seeded shape as tests/test_dashboard_stats.py::_seed_faces --
    kept as a small local copy so this HTTP-level test doesn't reach across
    test modules for a fixture, per this suite's existing style of each test
    file being self-contained.

    faces=2, scanned=2, clusters=1, people=1, unreviewed=0, singletons=0.
    """
    ids = store.record_scan("d0", "yunet", [ScannedFace(_det(), _vec([1, 0, 0]), "auraface")])
    ids += store.record_scan("d1", "yunet", [ScannedFace(_det(), _vec([0, 1, 0]), "auraface")])
    store.replace_clusters(
        "auraface", [cluster.Cluster(face_ids=tuple(ids), centroid=_vec([1, 0, 0]))]
    )
    cid = store.cluster_ids()[0]
    store.confirm(cid, "Alice")
    return {
        "wired": True,
        "faces": 2,
        "scanned": 2,
        "clusters": 1,
        "people": 1,
        "unreviewed": 0,
        "singletons": 0,
    }


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------


def test_api_stats_on_empty_catalog_returns_200_and_valid_json(handler_cls) -> None:
    status, headers, body = _dispatch_json(handler_cls, "GET", "/api/stats")
    assert status == 200
    assert headers.get("Content-Type", "").startswith("application/json")
    assert body is not None
    for key in ("now", "library", "evidence", "queues", "history", "projection", "overrides"):
        assert key in body


def test_api_stats_faces_section_reflects_a_seeded_face_store(
    catalog: Catalog, control: ControlPlane, face_store: FaceStore
) -> None:
    """`/api/stats`'s `faces` key end to end, not merely present -- see
    CLAUDE.md's task-16 fix brief: the reviewer's `{"MUTATED": True}` swap of
    `_faces_section`'s whole body passed every existing test in this suite.
    """
    expected = _seed_faces(face_store)
    handler_cls = dashboard_server.make_handler(catalog, control, store=face_store)

    status, _, body = _dispatch_json(handler_cls, "GET", "/api/stats")

    assert status == 200
    assert body["faces"] == expected


def test_api_stats_faces_section_is_none_when_face_store_raises(
    catalog: Catalog, control: ControlPlane
) -> None:
    """A corrupt/missing face catalog must not 500 the whole `/api/stats`
    endpoint -- 'a dashboard failure never stops organizing' applies to the
    faces section exactly like every other one. `stats.collect()`'s `_safe`
    wrapper already covers this (see test_dashboard_stats.py); this pins the
    same guarantee at the HTTP boundary, through `make_handler`'s real
    `store=` wiring.
    """

    class _RaisingFaceStore:
        def stats(self) -> dict:
            raise RuntimeError("simulated corrupt face catalog")

    handler_cls = dashboard_server.make_handler(
        catalog, control, store=_RaisingFaceStore()  # type: ignore[arg-type]
    )

    status, _, body = _dispatch_json(handler_cls, "GET", "/api/stats")

    assert status == 200
    assert body["faces"] is None
    # The rest of the document must still be there -- one section raising
    # must not take the others down with it.
    for key in ("now", "library", "evidence", "queues", "history", "projection", "overrides"):
        assert body[key] is not None


# ---------------------------------------------------------------------------
# POST /api/pause
# ---------------------------------------------------------------------------


def test_post_pause_flips_the_flag(handler_cls, control: ControlPlane) -> None:
    assert control.paused is False
    status, _, body = _dispatch_json(handler_cls, "POST", "/api/pause", {"paused": True})
    assert status == 200
    assert control.paused is True
    assert body["paused"] is True

    status, _, body = _dispatch_json(handler_cls, "POST", "/api/pause", {"paused": False})
    assert status == 200
    assert control.paused is False
    assert body["paused"] is False


def test_post_pause_rejects_non_boolean(handler_cls, control: ControlPlane) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/pause", {"paused": "yes"})
    assert status == 400
    assert control.paused is False


# ---------------------------------------------------------------------------
# POST /api/settings
# ---------------------------------------------------------------------------


def test_post_settings_interval_applies_override(handler_cls, control: ControlPlane) -> None:
    status, _, body = _dispatch_json(handler_cls, "POST", "/api/settings", {"interval": 120})
    assert status == 200
    assert control.interval == 120
    assert body["overrides"]["interval"]["overridden"] is True


def test_post_settings_enrich_applies_override(handler_cls, control: ControlPlane) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/settings", {"enrich": False})
    assert status == 200
    assert control.enrich_enabled is False


def test_post_settings_faces_applies_override(handler_cls, control: ControlPlane) -> None:
    """`faces` must be reachable through the same HTTP boundary as
    `interval`/`enrich` -- `_SETTINGS_KEYS` omitting it made this 400 as an
    "unknown setting" even though `ControlPlane` already fully implements
    the storage/resolution side (`_resolve_faces`, `faces_enabled`,
    `set_override`'s `_FACES_KEY` branch, `overrides()`'s `faces` entry).
    """
    assert control.faces_enabled is False
    status, _, body = _dispatch_json(handler_cls, "POST", "/api/settings", {"faces": True})
    assert status == 200
    # Round-trip through the store, not just the POST's own response --
    # a second, independent read (control.faces_enabled) and the response
    # body's own overrides() both must reflect it.
    assert control.faces_enabled is True
    assert body["overrides"]["faces"]["value"] is True
    assert body["overrides"]["faces"]["overridden"] is True

    status, _, body = _dispatch_json(handler_cls, "POST", "/api/settings", {"faces": False})
    assert status == 200
    assert control.faces_enabled is False
    assert body["overrides"]["faces"]["value"] is False


def test_post_settings_rejects_negative_interval(handler_cls, control: ControlPlane) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/settings", {"interval": -5})
    assert status == 400
    # Must not have been stored -- the boundary rejects it, the reader is
    # never asked to defend against it.
    assert control.interval == 300


def test_post_settings_rejects_non_numeric_interval(handler_cls, control: ControlPlane) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/settings", {"interval": "abc"})
    assert status == 400
    assert control.interval == 300


def test_post_settings_rejects_unknown_key(handler_cls, control: ControlPlane) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/settings", {"bogus": 1})
    assert status == 400


# ---------------------------------------------------------------------------
# POST /api/settings -- IMPORTANT finding: multi-key requests must validate
# every key before applying any of them (reproduced: an earlier `enrich`
# key was persisted before a later, invalid `interval` key raised).
# ---------------------------------------------------------------------------


def test_post_settings_multi_key_failure_leaves_earlier_key_unchanged(
    handler_cls, control: ControlPlane
) -> None:
    """`{"enrich": false, "interval": -5}` must 400 AND leave `enrich` untouched.

    Dict (insertion) order puts the valid key first and the invalid key
    second -- exactly the order that exposed the bug when keys were
    validated and applied one at a time in the same loop.
    """
    assert control.enrich_enabled is True
    assert control.interval == 300
    status, _, _ = _dispatch_json(
        handler_cls, "POST", "/api/settings", {"enrich": False, "interval": -5}
    )
    assert status == 400
    # The store, not just the status code: `enrich` must not have been
    # applied even though it appears before the bad `interval` key.
    assert control.enrich_enabled is True
    assert control.interval == 300


def test_post_settings_multi_key_failure_other_key_order(
    handler_cls, control: ControlPlane
) -> None:
    """Same as above with the keys reversed, so the test does not depend on
    dict ordering -- the invalid key now comes first, the valid key second.
    """
    assert control.enrich_enabled is True
    assert control.interval == 300
    status, _, _ = _dispatch_json(
        handler_cls, "POST", "/api/settings", {"interval": -5, "enrich": False}
    )
    assert status == 400
    assert control.enrich_enabled is True
    assert control.interval == 300


def test_post_settings_valid_multi_key_request_applies_both(
    handler_cls, control: ControlPlane
) -> None:
    status, _, body = _dispatch_json(
        handler_cls, "POST", "/api/settings", {"enrich": False, "interval": 120}
    )
    assert status == 200
    assert control.enrich_enabled is False
    assert control.interval == 120
    assert body["overrides"]["interval"]["overridden"] is True
    assert body["overrides"]["enrich"]["overridden"] is True


# ---------------------------------------------------------------------------
# POST /api/settings -- CRITICAL finding #1: Infinity/-Infinity/NaN
# ---------------------------------------------------------------------------
#
# `json.dumps` cannot serialize `float("inf")` (raises `ValueError`), so
# `_dispatch_json` can't be used to send it -- but `json.loads` (Python's
# stdlib, used by `_read_json_body`) accepts the bare, non-standard-JSON
# tokens `Infinity`/`-Infinity`/`NaN` on the way IN. That asymmetry is
# exactly how a hostile/buggy client can hand this endpoint a value nothing
# in this codebase can ever produce by calling `json.dumps` itself. The raw
# bytes below reproduce that literally.


@pytest.mark.parametrize("token", ["Infinity", "-Infinity", "NaN"])
def test_post_settings_rejects_non_finite_interval(
    handler_cls, control: ControlPlane, token: str
) -> None:
    status, _, body = _dispatch(
        handler_cls,
        "POST",
        "/api/settings",
        body=("{\"interval\": %s}" % token).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert b"Traceback" not in body
    # Must not have been stored -- same boundary-rejects-it guarantee as the
    # negative-interval test above.
    assert control.interval == 300


# ---------------------------------------------------------------------------
# POST /api/settings/revert
# ---------------------------------------------------------------------------


def test_post_settings_revert_restores_env_value(handler_cls, control: ControlPlane) -> None:
    control.set_override("interval", 120)
    assert control.interval == 120
    status, _, body = _dispatch_json(handler_cls, "POST", "/api/settings/revert", {"key": "interval"})
    assert status == 200
    assert control.interval == 300
    assert body["overrides"]["interval"]["overridden"] is False


def test_post_settings_revert_rejects_unknown_key(handler_cls) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/settings/revert", {"key": "paused"})
    assert status == 400


# ---------------------------------------------------------------------------
# Malformed / hostile bodies -- must never 500
# ---------------------------------------------------------------------------


def test_malformed_json_body_returns_400_not_a_traceback(handler_cls) -> None:
    status, _, body = _dispatch(
        handler_cls,
        "POST",
        "/api/pause",
        body=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert status == 400
    assert b"Traceback" not in body


def test_wrong_shape_json_body_returns_400(handler_cls) -> None:
    status, _, _ = _dispatch_json(handler_cls, "POST", "/api/pause", [1, 2, 3])
    assert status == 400


def test_oversized_content_length_is_rejected_without_reading_body(handler_cls) -> None:
    # A huge declared Content-Length with a tiny actual body: if the server
    # tried to read `length` bytes from rfile it would hang/error against a
    # BytesIO short of data. Rejecting on the header alone must return
    # before any such read is attempted.
    status, _, body = _dispatch(
        handler_cls,
        "POST",
        "/api/pause",
        body=b'{"paused": true}',
        headers={"Content-Length": str(10 * 1024 * 1024)},
    )
    assert status in (400, 413)
    assert b"Traceback" not in body


def test_get_to_post_only_route_does_not_500(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/api/pause")
    assert status != 500


def test_post_to_get_only_route_does_not_500(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "POST", "/api/stats")
    assert status != 500


def test_path_traversal_attempt_returns_404(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/../../etc/passwd")
    assert status == 404


# ---------------------------------------------------------------------------
# Unknown path / healthz / index
# ---------------------------------------------------------------------------


def test_unknown_path_returns_404(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/does/not/exist")
    assert status == 404


def test_healthz_returns_200(handler_cls) -> None:
    status, _, body = _dispatch(handler_cls, "GET", "/healthz")
    assert status == 200
    assert body


def test_index_returns_200_html(handler_cls) -> None:
    status, headers, body = _dispatch(handler_cls, "GET", "/")
    assert status == 200
    assert headers.get("Content-Type", "").startswith("text/html")
    assert b"<html" in body.lower() or b"<!doctype" in body.lower()


# ---------------------------------------------------------------------------
# Host-header allowlist (DNS-rebinding defense)
# ---------------------------------------------------------------------------


def test_a_request_with_a_foreign_host_header_is_refused():
    # DNS-rebinding defense: evil.example resolves to this box, browser sends
    # Host: evil.example -- the server must refuse to serve it.
    status, headers, body = _request("GET", "/api/stats", host="evil.example")
    assert status == 403


def test_a_request_with_no_host_header_is_refused():
    status, headers, body = _request("GET", "/api/stats", host=None)
    assert status == 403


def test_loopback_hosts_are_always_allowed():
    for host in ("localhost", "localhost:8080", "127.0.0.1", "127.0.0.1:9999"):
        status, headers, body = _request("GET", "/healthz", host=host)
        assert status == 200, host


def test_a_configured_allowed_host_is_accepted_with_any_port():
    # handler built with allowed_hosts frozenset including "hpz440.tailnet"
    status, headers, body = _request(
        "GET", "/healthz", host="hpz440.tailnet:8087",
        allowed_hosts=frozenset({"hpz440.tailnet"}),
    )
    assert status == 200


def test_host_matching_is_case_insensitive_and_handles_ipv6_brackets():
    status, _, _ = _request("GET", "/healthz", host="LOCALHOST:8080")
    assert status == 200
    status, _, _ = _request("GET", "/healthz", host="[::1]:8080")
    assert status == 200


# ---------------------------------------------------------------------------
# Shared-secret token gate on POST (R2 Task 3)
# ---------------------------------------------------------------------------


def test_post_without_token_is_401_when_token_configured():
    status, _, body = _request(
        "POST", "/api/pause", body={"paused": True}, token_configured="s3cret",
    )
    assert status == 401


def test_post_with_wrong_token_is_401():
    status, _, _ = _request(
        "POST", "/api/pause", body={"paused": True},
        token_configured="s3cret", token_header="wrong",
    )
    assert status == 401


def test_post_with_the_right_token_succeeds():
    status, _, _ = _request(
        "POST", "/api/pause", body={"paused": True},
        token_configured="s3cret", token_header="s3cret",
    )
    assert status == 200


def test_get_never_requires_the_token():
    status, _, _ = _request("GET", "/api/stats", token_configured="s3cret")
    assert status == 200


def test_posts_work_unauthenticated_when_no_token_is_configured():
    status, _, _ = _request("POST", "/api/pause", body={"paused": True})
    assert status == 200


# ---------------------------------------------------------------------------
# serve() -- port binding
# ---------------------------------------------------------------------------


def test_serve_returns_thread_on_success(catalog: Catalog, control: ControlPlane) -> None:
    stop_event = threading.Event()
    # Port 0 asks the OS for any free port, avoiding flakiness from a
    # hardcoded port that might be busy in CI.
    thread = dashboard_server.serve(catalog, control, port=0, stop_event=stop_event)
    try:
        assert thread is not None
        assert thread.is_alive()
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=5)


def test_serve_on_already_bound_port_returns_none_and_does_not_raise(
    catalog: Catalog, control: ControlPlane
) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the SAME host ("127.0.0.1", `serve()`'s default) that `serve()`
    # binds. On Windows, a wildcard bind and a same-port specific-address
    # bind do not conflict with each other by default (verified: binding
    # 0.0.0.0:PORT succeeds even while 127.0.0.1:PORT is already listening,
    # and the reverse), so a "0.0.0.0" blocker would make this test pass for
    # the wrong reason -- no real conflict, `serve()` would bind fine, and
    # the assertion would only hold by accident.
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        stop_event = threading.Event()
        # Must not raise -- a dashboard failure must never stop the watcher.
        result = dashboard_server.serve(
            catalog, control, port=port, stop_event=stop_event
        )
        assert result is None
    finally:
        blocker.close()


def test_serve_binds_loopback_by_default(
    catalog: Catalog, control: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default bind must be loopback: a socket on 0.0.0.0 exposes every
    # mutating endpoint to the LAN, which is opt-in (compose) from R2 on.
    seen: dict[str, Any] = {}
    real_ctor = dashboard_server._DashboardHTTPServer.__init__

    def spy(self: Any, addr: Any, handler: Any) -> None:
        seen["addr"] = addr
        real_ctor(self, addr, handler)

    monkeypatch.setattr(dashboard_server._DashboardHTTPServer, "__init__", spy)

    stop_event = threading.Event()
    thread = dashboard_server.serve(catalog, control, port=0, stop_event=stop_event)
    try:
        assert thread is not None
        assert seen["addr"][0] == "127.0.0.1"
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=5)


def test_serve_binds_wildcard_only_on_request(
    catalog: Catalog, control: ControlPlane, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real_ctor = dashboard_server._DashboardHTTPServer.__init__

    def spy(self: Any, addr: Any, handler: Any) -> None:
        seen["addr"] = addr
        real_ctor(self, addr, handler)

    monkeypatch.setattr(dashboard_server._DashboardHTTPServer, "__init__", spy)

    stop_event = threading.Event()
    thread = dashboard_server.serve(
        catalog, control, port=0, host="0.0.0.0", stop_event=stop_event
    )
    try:
        assert thread is not None
        assert seen["addr"][0] == "0.0.0.0"
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Token comparison must never raise on non-ASCII input (R2 Task 1)
# ---------------------------------------------------------------------------
#
# `hmac.compare_digest` raises `TypeError` when given `str` arguments that
# contain non-ASCII characters (documented caveat: "only ascii characters are
# supported" for str/str comparisons) -- which would surface as a bare 500
# with a traceback, not the clean 401 a wrong token deserves.


def test_non_ascii_token_header_with_ascii_configured_token_is_401_not_500():
    status, _, body = _request(
        "POST", "/api/pause", body={"paused": True},
        token_configured="s3cret", token_header="café",
    )
    assert status == 401
    assert b"Traceback" not in body


def test_non_ascii_configured_token_with_matching_header_succeeds():
    status, _, _ = _request(
        "POST", "/api/pause", body={"paused": True},
        token_configured="café", token_header="café",
    )
    assert status == 200


def test_non_ascii_configured_token_with_wrong_header_is_401():
    status, _, body = _request(
        "POST", "/api/pause", body={"paused": True},
        token_configured="café", token_header="wrong",
    )
    assert status == 401
    assert b"Traceback" not in body


# ---------------------------------------------------------------------------
# Stalled body read -> clean 408 (R2 Task 2)
# ---------------------------------------------------------------------------


class _TimingOutRfile:
    """Stand-in for a socket-backed rfile whose read stalls past the
    handler's own `timeout` and raises, exactly like a real slowloris client
    would once `settimeout` fires."""

    def read(self, _length: int) -> bytes:
        raise TimeoutError("simulated stalled client")


def test_stalled_body_read_returns_408_not_a_traceback(handler_cls, caplog) -> None:
    handler = handler_cls.__new__(handler_cls)
    handler.headers = {"Content-Length": "16"}  # type: ignore[attr-defined]
    handler.rfile = _TimingOutRfile()  # type: ignore[attr-defined]

    value, status, message = handler._read_json_body()

    assert value is None
    assert status == 408
    assert message == "request body timed out"
    assert not any(r.levelname in ("ERROR", "CRITICAL") for r in caplog.records)


# ---------------------------------------------------------------------------
# CSRF: POSTs require Content-Type: application/json (R2 Task 3)
# ---------------------------------------------------------------------------


def test_post_with_text_plain_content_type_is_415(handler_cls) -> None:
    status, _, body = _dispatch(
        handler_cls,
        "POST",
        "/api/pause",
        body=b'{"paused": true}',
        headers={"Content-Type": "text/plain"},
    )
    assert status == 415
    assert b"Traceback" not in body


def test_post_with_application_json_and_charset_is_accepted(handler_cls) -> None:
    status, _, _ = _dispatch(
        handler_cls,
        "POST",
        "/api/pause",
        body=b'{"paused": true}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert status == 200


def test_post_with_missing_content_type_is_415(handler_cls) -> None:
    status, _, body = _dispatch(
        handler_cls,
        "POST",
        "/api/pause",
        body=b'{"paused": true}',
        headers={"Content-Type": None},
    )
    assert status == 415
    assert b"Traceback" not in body


# ---------------------------------------------------------------------------
# Host hygiene (R2 Task 4)
# ---------------------------------------------------------------------------


def test_duplicate_host_headers_are_refused(handler_cls) -> None:
    status, _, _ = _dispatch(
        handler_cls, "GET", "/healthz",
        extra_header_lines=["Host: evil.example"],
    )
    assert status == 403


def test_duplicate_host_headers_are_refused_on_post(handler_cls) -> None:
    status, _, _ = _dispatch(
        handler_cls, "POST", "/api/pause",
        body=b'{"paused": true}',
        extra_header_lines=["Host: evil.example"],
    )
    assert status == 403


def test_host_with_trailing_dot_is_allowed(handler_cls) -> None:
    status, _, _ = _dispatch(handler_cls, "GET", "/healthz", host="localhost.")
    assert status == 200


def test_forbidden_host_message_names_the_flag_and_env_var(handler_cls) -> None:
    status, _, body = _dispatch(handler_cls, "GET", "/healthz", host="evil.example")
    assert status == 403
    parsed = json.loads(body.decode("utf-8"))
    assert "--dashboard-allowed-hosts" in parsed["error"]
    assert "IMAGEHARBOR_DASHBOARD_ALLOWED_HOSTS" in parsed["error"]


# ---------------------------------------------------------------------------
# Handler timeout is pinned (R2 Task 5)
# ---------------------------------------------------------------------------


def test_handler_timeout_is_pinned_to_10_seconds(handler_cls) -> None:
    assert handler_cls.timeout == 10
