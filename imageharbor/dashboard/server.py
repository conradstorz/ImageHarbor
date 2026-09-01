"""HTTP surface for the operational dashboard: routing, the control POSTs,
and serving the single self-contained page.

Two entry points, per the design (``docs/superpowers/specs/2026-08-19-dashboard-design.md``,
"HTTP surface" and "CLI and Docker"):

- :func:`make_handler` builds a fresh ``BaseHTTPRequestHandler`` subclass
  closed over one ``(catalog, control, breaker)`` triple. A fresh class per
  call, rather than module-level globals a handler reaches for, is what
  makes the handler exercisable directly against a fake socket in tests --
  no port, no thread -- since ``BaseRequestHandler.__init__`` runs a whole
  request/response cycle synchronously against whatever request-like object
  it is given.
- :func:`serve` binds the socket and starts the daemon thread(s).

The rule that outranks everything else in this module: **a dashboard
failure must never stop the watcher.** ``serve()`` catches a bind failure
and returns ``None`` rather than letting it propagate, and every request
handler catches its own section/body-parsing failures rather than ever
producing a 500 with a traceback in the body -- an operator debugging a
crashed pass is exactly who is looking at this page, and it must not become
a second thing that is broken.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from imageharbor.catalog import Catalog
from imageharbor.circuit_breaker import CircuitBreaker
from imageharbor.dashboard import people, stats
from imageharbor.dashboard.control import ControlPlane

# FaceStore is only ever used in type annotations here (never instantiated
# or called), and this module has `from __future__ import annotations`
# above, so the annotation is never evaluated at runtime. Importing it at
# module scope would still require numpy (see imageharbor/faces/store.py's
# module-scope `import numpy as np`) even when `watch` starts with faces and
# the dashboard both disabled -- this module is imported unconditionally by
# `cli.py`'s `watch` command before its own `--no-dashboard` check runs. See
# CLAUDE.md's "a missing extra degrades to one warning" invariant.
if TYPE_CHECKING:
    from imageharbor.faces.store import FaceStore

logger = logging.getLogger(__name__)

_INDEX_HTML_PATH = Path(__file__).parent / "index.html"

# A defensive ceiling, not a realistic limit: the largest legitimate request
# body this API ever receives is a one-key JSON object (e.g.
# {"interval": 120}), a few dozen bytes. This exists solely so a hostile or
# broken Content-Length cannot make this process buffer an unbounded body --
# the check happens against the header, before any read of the body itself.
MAX_BODY_BYTES = 64 * 1024

# The only two keys `control.overrides()` knows about (see control.py's
# module docstring: 'paused' has no env counterpart and is not part of the
# override-precedence story). Validating against this set at the HTTP
# boundary -- rather than letting an arbitrary string reach
# ControlPlane.set_override/revert -- is what keeps a malformed request from
# ever being "defended against on read" instead of rejected outright.
_SETTINGS_KEYS = ("interval", "enrich")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def make_handler(
    catalog: Catalog,
    control: ControlPlane,
    *,
    breaker: CircuitBreaker | None = None,
    store: FaceStore | None = None,
    crop_dir: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a ``BaseHTTPRequestHandler`` subclass closed over one dashboard.

    Each route handler here is defensive by construction: a section that
    fails inside ``stats.collect`` already reports itself as ``None`` (see
    ``stats.py``'s ``_safe``), and anything unexpected that still escapes is
    caught here and turned into a JSON error response rather than an
    unhandled exception reaching ``http.server``'s own default (which would
    print a traceback to the client).

    ``store``/``crop_dir`` are optional: the face pipeline is not wired into
    every caller yet (``cli.py``'s ``watch`` command does not construct a
    ``FaceStore`` as of this task), so the People routes must degrade to a
    plain 404 rather than raising when either is absent -- the same
    "dashboard failure never stops organizing" rule, applied to "the feature
    isn't wired up here" rather than to a query that raised.
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "ImageHarborDashboard/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Route access logging through the project's logger rather than
            # stderr, and at debug level -- this is a status page polled
            # every few seconds, not an event worth INFO-level noise.
            logger.debug("dashboard: " + format, *args)

        # -- response helpers --------------------------------------------

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(
            self, status: HTTPStatus, text: str, content_type: str = "text/plain"
        ) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -- request body -------------------------------------------------

        def _read_json_body(self) -> tuple[Any, HTTPStatus | None, str | None]:
            """Read and parse a JSON request body defensively.

            Returns ``(value, None, None)`` on success -- ``value`` is ``{}``
            when there is no body at all -- or ``(None, status, message)``
            when the request should be rejected. Rejection happens in this
            order: an unparseable ``Content-Length`` (400), an oversized one
            (413, checked from the header alone and *before* any read so an
            oversized body is never pulled into memory), then a body that
            fails to decode as UTF-8 or parse as JSON (400). Never raises.
            """
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return {}, None, None
            try:
                length = int(raw_length)
            except (TypeError, ValueError):
                return None, HTTPStatus.BAD_REQUEST, "invalid Content-Length"
            if length < 0:
                return None, HTTPStatus.BAD_REQUEST, "invalid Content-Length"
            if length > MAX_BODY_BYTES:
                return (
                    None,
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request body too large",
                )
            if length == 0:
                return {}, None, None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8")), None, None
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None, HTTPStatus.BAD_REQUEST, "malformed JSON body"

        # -- routing --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 (http.server's naming convention)
            try:
                if self.path == "/":
                    self._handle_index()
                elif self.path == "/api/stats":
                    self._handle_stats()
                elif self.path == "/healthz":
                    self._send_text(HTTPStatus.OK, "ok")
                elif self.path.startswith("/api/people"):
                    self._handle_people()
                elif self.path.startswith("/api/face-crop/"):
                    self._handle_face_crop()
                else:
                    self._send_text(HTTPStatus.NOT_FOUND, "not found")
            except Exception:
                logger.exception("dashboard: unhandled error handling GET %s", self.path)
                self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/api/pause":
                    self._handle_pause()
                elif self.path == "/api/settings":
                    self._handle_settings()
                elif self.path == "/api/settings/revert":
                    self._handle_revert()
                elif self.path.startswith("/api/people/"):
                    self._handle_people_action()
                else:
                    self._send_text(HTTPStatus.NOT_FOUND, "not found")
            except Exception:
                logger.exception("dashboard: unhandled error handling POST %s", self.path)
                self._send_text(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

        # -- GET handlers ---------------------------------------------------

        def _handle_index(self) -> None:
            try:
                html = _INDEX_HTML_PATH.read_text(encoding="utf-8")
            except OSError:
                logger.exception("dashboard: could not read index.html")
                self._send_text(
                    HTTPStatus.INTERNAL_SERVER_ERROR, "dashboard page unavailable"
                )
                return
            self._send_text(HTTPStatus.OK, html, content_type="text/html")

        def _handle_stats(self) -> None:
            document = stats.collect(catalog, control, breaker=breaker, face_store=store)
            self._send_json(HTTPStatus.OK, document)

        def _handle_people(self) -> None:
            if store is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"error": "face review is not enabled"}
                )
                return
            query = parse_qs(urlsplit(self.path).query)
            include_singletons = query.get("include_singletons", ["0"])[0].lower() in (
                "1", "true",
            )
            document = people.review_queue(store, include_singletons=include_singletons)
            self._send_json(HTTPStatus.OK, document)

        def _handle_face_crop(self) -> None:
            if store is None or crop_dir is None:
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
                return
            raw_id = urlsplit(self.path).path[len("/api/face-crop/"):]
            try:
                face_id = int(raw_id)
            except ValueError:
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
                return
            data = people.crop_bytes(crop_dir, face_id, store=store)
            if data is None:
                self._send_text(HTTPStatus.NOT_FOUND, "not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            # Derived, deletable-at-any-time cache data (see
            # dashboard/people.py's crop_bytes docstring) -- never cached by
            # the browser, so a re-crop after a regeneration is visible on
            # the very next load rather than stuck behind a stale copy.
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        # -- POST handlers ----------------------------------------------------

        def _handle_pause(self) -> None:
            body, err_status, err_msg = self._read_json_body()
            if err_status is not None:
                self._send_json(err_status, {"error": err_msg})
                return
            if (
                not isinstance(body, dict)
                or "paused" not in body
                or not isinstance(body["paused"], bool)
            ):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": 'expected a JSON object: {"paused": true|false}'},
                )
                return
            control.set_paused(body["paused"])
            self._send_json(HTTPStatus.OK, {"paused": control.paused})

        def _handle_settings(self) -> None:
            body, err_status, err_msg = self._read_json_body()
            if err_status is not None:
                self._send_json(err_status, {"error": err_msg})
                return
            if not isinstance(body, dict) or not body:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "expected a JSON object with at least one of: "
                        f"{', '.join(_SETTINGS_KEYS)}"
                    },
                )
                return
            unknown = [k for k in body if k not in _SETTINGS_KEYS]
            if unknown:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": f"unknown setting(s): {unknown}"}
                )
                return

            # Validate every key in the request BEFORE applying any of them.
            # `body.items()` iterates in dict (insertion) order, and
            # `ControlPlane.set_override` persists each key as it is called
            # -- so a loop that validates-and-applies key-by-key can persist
            # an earlier, valid key and only then discover a later key is
            # bad, leaving the store partially updated even though the
            # caller receives nothing but a 400 (reproduced: `{"enrich":
            # false, "interval": -5}` left `enrich` applied and `interval`
            # untouched). Collecting every key's validation error into
            # `errors` first, and only calling `set_override` in a second,
            # separate loop once `errors` is empty, is what makes "the
            # boundary rejects a bad request before any write happens" true
            # for a multi-key body, not just a single-key one -- the shape
            # of the code (validate loop, then apply loop) is the guarantee,
            # not a try/except that would have to unwind a partial write
            # after the fact.
            errors: dict[str, str] = {}
            if "interval" in body:
                interval = body["interval"]
                # bool is a subclass of int -- reject it explicitly so
                # {"interval": true} cannot silently become interval=1.0 by
                # falling through to float(True).
                if isinstance(interval, bool) or not isinstance(interval, (int, float)):
                    errors["interval"] = "interval must be numeric"
                elif not math.isfinite(interval):
                    # `json.loads` accepts the bare (non-standard-JSON)
                    # tokens `Infinity`/`-Infinity`/`NaN` and hands back a
                    # real `float`, so `{"interval": Infinity}` reaches here
                    # as `body["interval"] == float("inf")`. Rejecting it
                    # explicitly at this boundary (rather than relying
                    # solely on `set_override`'s own ValueError) means
                    # non-finite junk never even reaches `set_override` --
                    # this was a Critical fix and must not regress.
                    errors["interval"] = "interval must be a finite number"
                elif interval <= 0:
                    errors["interval"] = (
                        "interval must be a positive, finite number"
                    )
            if "enrich" in body and not isinstance(body["enrich"], bool):
                errors["enrich"] = "enrich must be a boolean"

            if errors:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "invalid setting(s): " + ", ".join(sorted(errors)),
                        "details": errors,
                    },
                )
                return

            # Every key validated above -- apply all of them. `set_override`
            # can still raise for a key whose validation logic drifts out of
            # sync with the checks above; nothing has been written yet at
            # that point either, since this is the first write in the
            # request.
            for key, value in body.items():
                try:
                    control.set_override(key, value)
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
            self._send_json(HTTPStatus.OK, {"overrides": control.overrides()})

        def _handle_revert(self) -> None:
            body, err_status, err_msg = self._read_json_body()
            if err_status is not None:
                self._send_json(err_status, {"error": err_msg})
                return
            key = body.get("key") if isinstance(body, dict) else None
            if not isinstance(key, str) or key not in _SETTINGS_KEYS:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"key must be one of: {', '.join(_SETTINGS_KEYS)}"},
                )
                return
            control.revert(key)
            self._send_json(HTTPStatus.OK, {"overrides": control.overrides()})

        def _handle_people_action(self) -> None:
            if store is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"error": "face review is not enabled"}
                )
                return
            body, err_status, err_msg = self._read_json_body()
            if err_status is not None:
                self._send_json(err_status, {"error": err_msg})
                return
            if not isinstance(body, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": "expected a JSON object"}
                )
                return

            # dispatch on the final path segment, matching the routes in
            # the design doc's "Dashboard" table: confirm/reject/merge/split
            action = self.path.rsplit("/", 1)[-1]
            try:
                if action == "confirm":
                    result = people.confirm(
                        store, body.get("cluster_id"), body.get("name", "")
                    )
                elif action == "reject":
                    result = people.reject(
                        store, body.get("cluster_id"), body.get("name", "")
                    )
                elif action == "merge":
                    result = people.merge(
                        store, body.get("person_id"), body.get("cluster_ids") or []
                    )
                elif action == "split":
                    result = people.split(
                        store, body.get("cluster_id"), body.get("face_ids") or []
                    )
                else:
                    self._send_text(HTTPStatus.NOT_FOUND, "not found")
                    return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, result)

    return Handler


class _DashboardHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with reuse-address explicitly off.

    ``HTTPServer`` turns ``SO_REUSEADDR`` on by default. On POSIX that only
    affects sockets in ``TIME_WAIT`` and still refuses to bind a port with
    an active listener, but on Windows ``SO_REUSEADDR`` is permissive enough
    to let a second socket silently bind the same port -- which would make
    the already-bound-port case this server exists to degrade from
    (see the design's "A dashboard failure must never stop the watcher")
    impossible to observe. Turning it off makes "the port is already in
    use" fail the same way -- with ``OSError`` -- on every platform.
    """

    daemon_threads = True
    allow_reuse_address = False


def serve(
    catalog: Catalog,
    control: ControlPlane,
    *,
    port: int,
    breaker: CircuitBreaker | None = None,
    store: FaceStore | None = None,
    crop_dir: Path | None = None,
    stop_event: threading.Event,
) -> threading.Thread | None:
    """Start the dashboard on a daemon thread, sharing *stop_event* with the caller.

    Returns the serving thread, or ``None`` if the port could not be bound.
    **Never raises** -- a dashboard that cannot start must not take the
    watcher down with it (see the design's "A dashboard failure must never
    stop the watcher" and the module docstring above). The bind itself
    happens synchronously, before any thread is created, so a bind failure
    (e.g. the port is already in use) is caught and logged right here rather
    than surfacing asynchronously on a background thread where nothing
    would be watching for it.

    Two daemon threads are started on success: one runs
    ``ThreadingHTTPServer.serve_forever`` (the one returned to the caller),
    the other waits on *stop_event* and calls ``shutdown()`` -- `shutdown()`
    must be called from a different thread than the one running
    `serve_forever()`, per `socketserver`'s documented contract. Both are
    daemon threads, so a process exit is never blocked on either, and
    `docker stop` (which sets *stop_event*) still triggers a clean
    `server_close()`.
    """
    handler_cls = make_handler(
        catalog, control, breaker=breaker, store=store, crop_dir=crop_dir
    )
    try:
        httpd = _DashboardHTTPServer(("0.0.0.0", port), handler_cls)
    except OSError:
        logger.warning(
            "dashboard: could not bind port %d; continuing without a dashboard",
            port,
            exc_info=True,
        )
        return None

    def _serve_forever() -> None:
        try:
            httpd.serve_forever(poll_interval=0.5)
        except Exception:
            logger.exception("dashboard: server thread crashed")

    def _watch_stop() -> None:
        stop_event.wait()
        try:
            httpd.shutdown()
        except Exception:
            logger.exception("dashboard: error shutting down server")
        finally:
            httpd.server_close()

    serve_thread = threading.Thread(
        target=_serve_forever, name="dashboard-server", daemon=True
    )
    serve_thread.start()

    stop_thread = threading.Thread(
        target=_watch_stop, name="dashboard-stop-watcher", daemon=True
    )
    stop_thread.start()

    return serve_thread
