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
import socket
import threading
from http.client import HTTPResponse
from pathlib import Path
from typing import Any

import pytest

from imageharbor.catalog import Catalog
from imageharbor.dashboard import server as dashboard_server
from imageharbor.dashboard.control import ControlPlane

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
    body: bytes = b"",
    headers: dict[str, str] | None = None,
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
    """Wraps raw response bytes so `http.client.HTTPResponse` can parse them."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def makefile(self, *_args: Any, **_kwargs: Any) -> io.BytesIO:
        return io.BytesIO(self._data)


def _dispatch(handler_cls: type, method: str, path: str, **kwargs: Any):
    """Send one request through *handler_cls* with no socket, return the response.

    Returns (status, headers-dict, body-bytes).
    """
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


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


@pytest.fixture()
def control(catalog: Catalog) -> ControlPlane:
    return ControlPlane(catalog, env_interval=300, env_enrich=True)


@pytest.fixture()
def handler_cls(catalog: Catalog, control: ControlPlane):
    return dashboard_server.make_handler(catalog, control)


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
    # Bind the SAME host ("0.0.0.0", the wildcard) that `serve()` binds --
    # not "127.0.0.1". On Windows, a wildcard bind and a same-port
    # specific-address bind do not conflict by default (verified: binding
    # 0.0.0.0:PORT succeeds even while 127.0.0.1:PORT is already listening),
    # so a "127.0.0.1" blocker would make this test pass for the wrong
    # reason -- no real conflict, `serve()` would bind fine, and the
    # assertion would only hold by accident.
    blocker.bind(("0.0.0.0", 0))
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
