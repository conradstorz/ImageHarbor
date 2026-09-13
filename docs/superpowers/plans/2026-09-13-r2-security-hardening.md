# R2 "Security Hardening" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the quality review's two security findings — the unauthenticated wildcard-bound dashboard and the Windows zip-slip in Takeout handling — plus the deploy-posture items the R1 final review deferred here (CI Docker build, GHCR pull path, Dockerfile interpreter pins), shipping as v1.1.0.

**Architecture:** Dashboard hardening is three independent server-side layers (bind address, Host allowlist, POST token) plus a small page-JS change; each layer lands as its own task with its own tests. Zip-slip is two parallel fixes (staging extraction, provenance preservation) sharing one attack-shape test vector. Deploy items land last, then the ship task merges with a `[minor]` token so the automation cuts v1.1.0.

**Tech Stack:** Python http.server, click, zipfile/pathlib, GitHub Actions, Docker/compose.

## Global Constraints

- Use `uv` for ALL Python work; never pip. Never chain shell commands with `&&` — one command per Bash call. (User global CLAUDE.md.)
- Branch: `release/v2-hardening` off current `main`; one commit per task; commit prefixes as in repo history.
- After ANY `pyproject.toml` edit: `uv lock` and commit `uv.lock` (CI runs `uv sync --locked`). No pyproject changes are expected in this plan.
- Before every commit: `uv run pytest -q` (baseline 1198 passed / 19-or-20 skipped depending on `IMAGEHARBOR_REQUIRE_SIBLING_ORACLE`, 0 failures), `uv run ruff check .`, `uv run mypy imageharbor` — all clean.
- **Never-stop-the-watcher rule holds:** every new rejection path (403 host, 401 token, timeout) is a normal HTTP response, never an exception into `http.server`, and `serve()` still never raises.
- **The dashboard page must keep working end-to-end** for an operator with the token, and read-only (GET) behavior must not require a token.
- CLAUDE.md invariants untouched: no pipeline/tier/sidecar/breaker changes anywhere in this plan.
- Takeout invariant holds: archives opened `'r'` only; no member name is ever trusted as a path.

### Design constants (used across tasks — exact values)

| Thing | Value |
|---|---|
| Bind-host flag/env | `--dashboard-host` / `IMAGEHARBOR_DASHBOARD_HOST`, default `"127.0.0.1"` |
| Token flag/env | `--dashboard-token` / `IMAGEHARBOR_DASHBOARD_TOKEN`, default `None` (auth disabled when unset) |
| Allowlist flag/env | `--dashboard-allowed-hosts` / `IMAGEHARBOR_DASHBOARD_ALLOWED_HOSTS`, comma-separated hostnames, default empty |
| Always-allowed hostnames | `{"localhost", "127.0.0.1", "::1"}` (plus every configured entry, lowercased) |
| Token header | `X-Dashboard-Token` |
| Handler socket timeout | `timeout = 10` (class attribute on the request handler) |
| Attack vector for zip-slip tests | member name `Takeout/..\\..\\..\\pwned.txt` (single backslashes in the actual string) |

---

### Task 1: Handler timeout and Host-header allowlist

**Files:**
- Modify: `imageharbor/dashboard/server.py`
- Test: `tests/test_dashboard_server.py`

**Interfaces:**
- Produces: `make_handler(..., allowed_hosts: frozenset[str])` and `serve(..., allowed_hosts: Sequence[str] = ())` — Task 2/3 extend the same signatures further; `_host_allowed(header_value: str | None, allowed: frozenset[str]) -> bool` module-level pure helper.
- Consumes: the existing BytesIO fake-socket test harness in `tests/test_dashboard_server.py` (do not invent a new harness).

- [ ] **Step 1: Read the harness.** Read `tests/test_dashboard_server.py`'s request-builder helper(s). Existing fake requests likely send no `Host` header; this task makes an absent/foreign Host a 403, so the harness's default request must gain `Host: localhost`. Update the helper once, centrally — not per test.

- [ ] **Step 2: Write the failing tests** (add to `tests/test_dashboard_server.py`):

```python
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
```

(Adapt `_request(...)`'s exact shape to the harness found in Step 1 — the assertions and header values are the spec; the plumbing mirrors the file's existing style.)

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_dashboard_server.py -q`
Expected: new tests FAIL (no 403 logic exists; `_request` may need the `host=`/`allowed_hosts=` parameters added first).

- [ ] **Step 4: Implement.** In `imageharbor/dashboard/server.py`:

```python
_ALWAYS_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _host_allowed(header_value: str | None, allowed: frozenset[str]) -> bool:
    """Whether a request's Host header names this server.

    DNS-rebinding defense: a browser at an attacker's page can be pointed at
    a hostname that resolves to this machine; the Host header still carries
    the attacker's name. Absent Host is refused too -- every legitimate
    client (browser, curl, the compose healthcheck) sends one. Matching is
    on the hostname alone (port stripped, case-folded, IPv6 brackets
    removed): the port is already fixed by the socket we are serving on.
    """
    if not header_value:
        return False
    host = header_value.strip().lower()
    if host.startswith("["):          # [::1]:8080
        host = host[1:].partition("]")[0]
    else:
        host = host.partition(":")[0]
    return host in _ALWAYS_ALLOWED_HOSTS or host in allowed
```

`make_handler` gains a keyword parameter `allowed_hosts: frozenset[str]` (threaded into the handler class the same way `catalog`/`control` are). At the TOP of both `do_GET` and `do_POST`, inside the existing `try`, add the gate before any routing:

```python
                if not _host_allowed(self.headers.get("Host"), allowed_hosts):
                    self._send_json(
                        HTTPStatus.FORBIDDEN,
                        {"error": "host header not recognized; see "
                                  "--dashboard-allowed-hosts"},
                    )
                    return
```

On the handler class, next to the routing methods, add the socket timeout (closes the slowloris/unbounded-read DoS — `rfile.read` on a stalled client now raises `TimeoutError`, which the existing per-handler `except` turns into a closed connection, not a hung thread):

```python
        timeout = 10
```

`serve(...)` gains `allowed_hosts: Sequence[str] = ()` and passes `frozenset(h.strip().lower() for h in allowed_hosts if h.strip())` to `make_handler`.

- [ ] **Step 5: Run the file's tests, then the suite**

Run: `uv run pytest tests/test_dashboard_server.py -q` → all pass (pre-existing tests green via the harness's new default `Host: localhost`).
Run: `uv run pytest -q` → other dashboard-touching suites (`test_watcher.py` HTTP tests, `test_dashboard_people_http.py`) hit the server over real loopback sockets, so their `Host` header is `localhost:<port>` or `127.0.0.1:<port>` automatically — expect green; if any fails on 403, fix the TEST's request (it must send a loopback Host), never weaken the gate.

- [ ] **Step 6: mypy/ruff, commit**

```bash
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/dashboard/server.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): Host-header allowlist and request timeout"
```

---

### Task 2: Loopback bind by default, `--dashboard-host`

**Files:**
- Modify: `imageharbor/dashboard/server.py` (`serve()` — the bind at line ~509), `imageharbor/cli.py` (options at ~474-487, call site at ~658-671)
- Test: `tests/test_dashboard_server.py`

**Interfaces:**
- Produces: `serve(..., host: str = "127.0.0.1", ...)`; CLI `--dashboard-host` / env `IMAGEHARBOR_DASHBOARD_HOST` (default `127.0.0.1`), threaded to both `serve()` and the "Dashboard listening on" echo.
- Note: tests that call `serve(...)` today implicitly got `0.0.0.0`; with the default now loopback they keep working (they connect to `127.0.0.1`). Compose keeps working via Task 7's explicit `IMAGEHARBOR_DASHBOARD_HOST: 0.0.0.0`.

- [ ] **Step 1: Write the failing test:**

```python
def test_serve_binds_loopback_by_default_and_wildcard_only_on_request(tmp_path):
    # Default bind must be loopback: a socket on 0.0.0.0 exposes every
    # mutating endpoint to the LAN, which is opt-in (compose) from R2 on.
    import socket
    from imageharbor.dashboard import server as dashboard_server

    seen = {}
    real_ctor = dashboard_server._DashboardHTTPServer.__init__

    def spy(self, addr, handler):
        seen["addr"] = addr
        real_ctor(self, addr, handler)

    # monkeypatch _DashboardHTTPServer.__init__ with spy (use pytest monkeypatch)
    ...
    thread = dashboard_server.serve(catalog, control, port=0, stop_event=stop)
    assert seen["addr"][0] == "127.0.0.1"
```

(Fill in the fixture plumbing from this test file's existing `serve()` tests; `port=0` if an existing test already uses an ephemeral-port pattern, otherwise follow that pattern.)

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_dashboard_server.py -q` → new test FAILS (`seen["addr"][0] == "0.0.0.0"`).

- [ ] **Step 3: Implement.** `serve()` signature gains `host: str = "127.0.0.1"`; the bind becomes `_DashboardHTTPServer((host, port), handler_cls)`; the bind-failure log line includes the host. In `cli.py`, add after the `--dashboard-port` option:

```python
@click.option(
    "--dashboard-host",
    envvar="IMAGEHARBOR_DASHBOARD_HOST",
    default="127.0.0.1",
    show_default=True,
    help="Interface the dashboard binds. Loopback by default; set 0.0.0.0 "
    "to expose it beyond this machine (then set --dashboard-token and "
    "--dashboard-allowed-hosts -- see docs/deploy-docker.md).",
)
```

Thread `dashboard_host: str` through `watch(...)`'s parameters into `dashboard_server.serve(..., host=dashboard_host, ...)` and change the echo to `f"Dashboard listening on http://{dashboard_host}:{dashboard_port}/"`.

- [ ] **Step 4: Suite, lint, types** — `uv run pytest -q`; `uv run ruff check .`; `uv run mypy imageharbor` → all green/clean.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/dashboard/server.py imageharbor/cli.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): bind loopback by default, --dashboard-host to opt out"
```

---

### Task 3: Shared-secret token on every POST route

**Files:**
- Modify: `imageharbor/dashboard/server.py`, `imageharbor/cli.py`
- Test: `tests/test_dashboard_server.py`, `tests/test_dashboard_people_http.py` (if it POSTs — check)

**Interfaces:**
- Produces: `make_handler(..., token: str | None)` / `serve(..., token: str | None = None)`; CLI `--dashboard-token` / env `IMAGEHARBOR_DASHBOARD_TOKEN`. Contract: token configured → every `do_POST` request must carry header `X-Dashboard-Token` whose value matches (compared with `hmac.compare_digest`), else 401 `{"error": "missing or wrong X-Dashboard-Token"}`. Token unset → POSTs behave as today (loopback bind is the default mitigation). GETs never require the token.
- Consumes: Task 1's gate placement (the token check goes in `do_POST` immediately AFTER the Host gate).

- [ ] **Step 1: Write the failing tests:**

```python
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
```

(Again: assertions are the spec, plumbing mirrors the harness.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_dashboard_server.py -q`.

- [ ] **Step 3: Implement.** In `do_POST`, after the Host gate:

```python
                if token is not None:
                    supplied = self.headers.get("X-Dashboard-Token", "")
                    if not hmac.compare_digest(supplied, token):
                        self._send_json(
                            HTTPStatus.UNAUTHORIZED,
                            {"error": "missing or wrong X-Dashboard-Token"},
                        )
                        return
```

(`import hmac` at module top.) `make_handler`/`serve` thread `token` exactly like `allowed_hosts`. In `cli.py`:

```python
@click.option(
    "--dashboard-token",
    envvar="IMAGEHARBOR_DASHBOARD_TOKEN",
    default=None,
    help="Shared secret required (X-Dashboard-Token header) on every "
    "dashboard POST. Unset = POSTs are open; fine on the default loopback "
    "bind, set it whenever --dashboard-host is not 127.0.0.1.",
)
```

threaded through `watch` into `serve(..., token=dashboard_token)`. Also `--dashboard-allowed-hosts` (envvar `IMAGEHARBOR_DASHBOARD_ALLOWED_HOSTS`, default `""`, split on commas in `cli.py`, passed as the `allowed_hosts` sequence) — added here since Task 1 created only the server-side parameter.

- [ ] **Step 4: Suite, lint, types, commit**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy imageharbor
git add imageharbor/dashboard/server.py imageharbor/cli.py tests/test_dashboard_server.py
git commit -m "feat(dashboard): shared-secret token gate on mutating routes"
```

---

### Task 4: The page sends the token

**Files:**
- Modify: `imageharbor/dashboard/index.html`

**Interfaces:**
- Consumes: Task 3's contract (`X-Dashboard-Token`, 401 on missing/wrong).
- Produces: every JS `fetch` that POSTs attaches the header from `localStorage`; a 401 triggers one `prompt()` for the token, stores it (`localStorage` key `imageharbor-dashboard-token`), and retries once.

- [ ] **Step 1: Find every POSTing fetch.** `grep -n "fetch(" imageharbor/dashboard/index.html` and identify the POST call sites (pause, settings, revert, people confirm/reject/merge/split).

- [ ] **Step 2: Implement one wrapper near the top of the page script:**

```javascript
const TOKEN_KEY = "imageharbor-dashboard-token";

async function postJson(url, payload) {
  const attempt = () => {
    let token = null;
    try { token = localStorage.getItem(TOKEN_KEY); } catch (e) { /* storage unavailable */ }
    const headers = { "Content-Type": "application/json" };
    if (token) headers["X-Dashboard-Token"] = token;
    return fetch(url, { method: "POST", headers, body: JSON.stringify(payload) });
  };
  let resp = await attempt();
  if (resp.status === 401) {
    const entered = window.prompt(
      "This dashboard requires its access token (IMAGEHARBOR_DASHBOARD_TOKEN):");
    if (entered) {
      try { localStorage.setItem(TOKEN_KEY, entered.trim()); } catch (e) { /* ignore */ }
      resp = await attempt();
    }
  }
  return resp;
}
```

Convert every POSTing call site to `postJson(url, payload)`. Do not change GET calls. Preserve each call site's existing response handling.

- [ ] **Step 3: Verify server-side round trip.** The suite can't run page JS; verify the header name matches Task 3 by grep (`X-Dashboard-Token` appears in both files) and run the dashboard once locally against a scratch catalog to click Pause with a token set:

```bash
uv run imageharbor watch --source <scratch-src> --dest <scratch-dest> --dashboard-token testtoken --interval 300
```

(Use scratchpad directories; browse http://127.0.0.1:8080/, click pause, enter `testtoken` at the prompt, confirm the state flips; Ctrl+C the watcher.) Record what you observed in the report. If a headless environment makes the click impossible, verify with two `curl`s instead (401 without header, 200 with) and say so.

- [ ] **Step 4: Suite (unchanged expectations), commit**

```bash
uv run pytest -q
git add imageharbor/dashboard/index.html
git commit -m "feat(dashboard): page prompts for and sends the POST token"
```

---

### Task 5: Zip-slip — staging extraction (`takeout/archive.py`)

**Files:**
- Modify: `imageharbor/takeout/archive.py:53` (`_ILLEGAL_NAME_CHARS`), `:154-174` (`extract_to`)
- Test: `tests/test_takeout_archive.py`

**Interfaces:**
- Produces: backslash is an illegal name character (sanitized to `_`); `extract_to` refuses (raises `ValueError`) any dest that escapes its holder directory — defense-in-depth behind the sanitizer.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_takeout_archive.py`, reusing its existing zip-building helpers):

```python
def test_a_backslash_member_name_cannot_escape_the_staging_dir(tmp_path):
    # On Windows pathlib treats "\" as a separator, so a member basename of
    # r"..\..\..\pwned.txt" handed to `holder / name` walks OUT of the
    # holder. The sanitizer must neutralize backslashes like any other
    # illegal character.
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Takeout/..\\..\\..\\pwned.txt", b"gotcha")
    staging = tmp_path / "staging"
    with zipfile.ZipFile(zip_path) as zf:
        member = MemberInfo(path="Takeout/..\\..\\..\\pwned.txt", size=6,
                            crc32=0, kind=KIND_IMAGE)
        staged = extract_to(zf, member, staging)
    staged_resolved = staged.resolve()
    assert staged_resolved.is_relative_to(staging.resolve())
    assert "\\" not in staged.name
    assert not (tmp_path / "pwned.txt").exists()


def test_sanitizer_replaces_backslashes():
    assert "\\" not in _safe_name("..\\..\\evil.jpg")
```

(Match `MemberInfo` construction and constant names to the file's existing tests — read them first; `kind` values come from the module.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_takeout_archive.py -q`. On Windows the first test FAILS with the file landing outside `staging` (or an assertion on the resolved path). Expected proof of the bug.

- [ ] **Step 3: Fix.** Change line 53:

```python
_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"|?*\\\x00-\x1f]')
```

and extend its comment: backslash is included because `pathlib` on Windows treats it as a separator, so a backslash-bearing member basename would escape the staging holder — this is the zip-slip vector, not a cosmetic character. In `extract_to`, after computing `dest`:

```python
    if not dest.resolve().is_relative_to(holder.resolve()):
        raise ValueError(
            f"refusing to stage zip member outside its holder: {member.path!r}"
        )
```

- [ ] **Step 4: Run the file, then the suite** — `uv run pytest tests/test_takeout_archive.py -q` then `uv run pytest -q` → green. `uv run ruff check .`, `uv run mypy imageharbor` clean.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/archive.py tests/test_takeout_archive.py
git commit -m "fix(takeout): backslash zip-slip in staging extraction"
```

---

### Task 6: Zip-slip — provenance preservation (`takeout/provenance.py`)

**Files:**
- Modify: `imageharbor/takeout/provenance.py:58` (`_ILLEGAL_NAME_CHARS`), `:88-96` (`_stored_path`)
- Test: `tests/test_takeout_provenance.py` (or wherever `preserve`/`_safe_relpath` tests live — locate with `grep -rln "_safe_relpath\|def preserve" tests/`)

**Interfaces:**
- Produces: backslash sanitized in `_safe_component`; `_stored_path` guarantees containment (raises `ValueError` if the computed path escapes `room` — unreachable once sanitization works, kept as the invariant's enforcement).
- This module's own docstring already claims traversal-safety ("a member path is never allowed to write outside the room it was handed") — this task makes the claim true on Windows.

- [ ] **Step 1: Write the failing tests:**

```python
def test_a_backslash_member_path_stays_inside_the_room(tmp_path):
    # _safe_relpath drops "." and ".." SLASH components, but a component
    # containing backslashes ("..\\..\\..\\pwned.txt") survived whole and,
    # on Windows, pathlib expands it into separators -- writing outside the
    # room the docstring promises to contain. Verified against the real
    # preserve() write path, not just the helper.
    rel = _safe_relpath("Takeout/..\\..\\..\\pwned.txt")
    room = tmp_path / "room"
    assert (room / rel).resolve().is_relative_to(room.resolve())


def test_safe_component_replaces_backslashes():
    assert "\\" not in _safe_component("..\\..\\evil")
```

Plus one end-to-end test through `preserve()` with a real zip containing the attack member (mirror the module's existing `preserve` test setup): assert the preserved document lands under the room and `(tmp_path / "pwned.txt")` does not exist.

- [ ] **Step 2: Run to verify failure** — first test FAILS on Windows (`(room / rel).resolve()` escapes).

- [ ] **Step 3: Fix.** Line 58 gets the same character-class change as Task 5 (`\\` added), with the same comment rationale. In `_stored_path`, before returning, enforce containment on every branch — restructure to compute `candidate` then:

```python
    if not candidate.resolve().is_relative_to(room.resolve()):
        raise ValueError(
            f"refusing to preserve member outside its room: {member.path!r}"
        )
    return candidate
```

Then check `preserve()`'s per-member loop: if it has per-member error isolation, a `ValueError` from `_stored_path` should be caught there, logged, and counted as that member failing (find the existing failure counter/log used for unreadable members and reuse it). If the loop has NO per-member isolation, add `try/except ValueError` around the single member's preserve step — never let one hostile name abort the whole archive's preservation.

- [ ] **Step 4: Run file, suite, lint, types** — all green/clean.

- [ ] **Step 5: Commit**

```bash
git add imageharbor/takeout/provenance.py tests/<the test file>
git commit -m "fix(takeout): backslash zip-slip in provenance preservation"
```

---

### Task 7: Compose, threat model, GHCR pull path

**Files:**
- Modify: `docker-compose.yml`, `docs/deploy-docker.md`

**Interfaces:**
- Consumes: Tasks 2-3's env vars; R1's published image `ghcr.io/conradstorz/imageharbor` (`:latest` = v1.0.1 today, v1.1.0 after this ships).

- [ ] **Step 1: docker-compose.yml.** Replace lines 8-10 (`build: .` / `image: imageharbor:latest`) with:

```yaml
    # Pulls the released image from GHCR (published automatically on every
    # green merge -- see .github/workflows/release.yml). To run a local
    # build instead: docker build -t ghcr.io/conradstorz/imageharbor:latest .
    image: ghcr.io/conradstorz/imageharbor:latest
```

Add to `environment:` (after `IMAGEHARBOR_FACE_THRESHOLD`):

```yaml
      # Dashboard exposure (R2): inside the container the server must bind
      # 0.0.0.0 for the published port to work; exposure is bounded by the
      # host's port publish (tailnet-only on hpz440). The token gates every
      # mutating request; the allowlist is the DNS-rebinding defense and
      # must name every host/IP the dashboard is browsed as.
      IMAGEHARBOR_DASHBOARD_HOST: 0.0.0.0
      IMAGEHARBOR_DASHBOARD_TOKEN: ""          # REQUIRED: set a long random string before `up`
      IMAGEHARBOR_DASHBOARD_ALLOWED_HOSTS: ""  # e.g. "100.69.239.123,hpz440"
```

One wrinkle: an empty-string env var must mean "unset" for the token — check `cli.py`'s handling (click passes `""` through). In Task 3's cli option add `callback` or post-processing: `if not dashboard_token: dashboard_token = None` inside `watch` (and same "empty means none" for allowed-hosts split). If Task 3 already did this, nothing to do — verify and note.

- [ ] **Step 2: docs/deploy-docker.md.** Add a `## Security model` section (place it after the dashboard-access section, around the existing port discussion):

```markdown
## Security model

The dashboard is an operations console, not a public site. Its exposure
model, in one table:

| Layer | Default | On hpz440 |
|---|---|---|
| Bind address | `127.0.0.1` (loopback only) | `0.0.0.0` in-container; reachable only via the published port on the tailnet |
| Mutating routes (`POST /api/*`) | open on loopback | require `X-Dashboard-Token` = `IMAGEHARBOR_DASHBOARD_TOKEN` (the page prompts once and remembers it per-browser) |
| Host-header allowlist | loopback names only | must list every name/IP the dashboard is browsed as (`IMAGEHARBOR_DASHBOARD_ALLOWED_HOSTS`) — this is the DNS-rebinding defense |
| Read-only routes (`/api/stats`, face crops) | no token | reachable by anyone who can reach the port and passes the Host check — treat the tailnet as the trust boundary |

What this deliberately does NOT provide: TLS (the tailnet link is already
encrypted end-to-end), per-user accounts, or protection of read-only data
from someone already inside the tailnet. If any of those assumptions stop
holding (e.g. the port is ever published beyond the tailnet), put a real
reverse proxy with auth in front instead of extending this scheme.
```

Also add, near the build/run instructions, the pull path:

```markdown
Images are published automatically: `docker compose pull` fetches the
latest release from `ghcr.io/conradstorz/imageharbor` (public, no login
needed to pull). `docker compose up -d` after a pull upgrades in place.
```

- [ ] **Step 3: Sanity-check compose parses**

Run: `docker compose -f docker-compose.yml config --quiet`
Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml docs/deploy-docker.md
git commit -m "docs(deploy): GHCR pull path, dashboard security model, hardened compose env"
```

---

### Task 8: CI builds the Dockerfile; interpreter pins

**Files:**
- Modify: `.github/workflows/ci.yml`, `Dockerfile`

**Interfaces:**
- Produces: a `docker` CI job (build only, `push: false`) so Dockerfile regressions fail at PR time, not post-tag; `Dockerfile` pins `UV_PYTHON_DOWNLOADS=never` + `UV_PYTHON_PREFERENCE=only-system` so the shipped interpreter is provably the base image's 3.12.

- [ ] **Step 1: ci.yml.** Add a third job:

```yaml
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          build-args: |
            IMAGEHARBOR_VERSION=0.0.0
```

- [ ] **Step 2: Dockerfile.** In the existing `ENV` line that sets `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_IMAGEHARBOR`/`UV_PROJECT_ENVIRONMENT`, add two more (with a one-line comment: uv must use the base image's own CPython, never fetch a managed one — otherwise the `python:3.12-slim` tag is decorative):

```dockerfile
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
```

- [ ] **Step 3: Prove the pins locally**

Run: `docker build -t imageharbor:r2-pins-test --build-arg IMAGEHARBOR_VERSION=0.0.0 .`
Expected: build succeeds (uv finds the system 3.12).
Run: `docker run --rm --entrypoint python imageharbor:r2-pins-test -c "import sys; print(sys.version)"`
Expected: `3.12.x`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml Dockerfile
git commit -m "ci: build the image on every PR; pin the container interpreter"
```

---

### Task 9: Ship v1.1.0 and update the branch-protection contexts

This task is interactive with GitHub.

- [ ] **Step 1: Final local verification** — `uv run ruff check .`; `uv run mypy imageharbor`; `uv run pytest` (full output tail in report).

- [ ] **Step 2: Push, PR, CI green**

```bash
git push -u origin release/v2-hardening
gh pr create --title "R2: security hardening" --body "Implements docs/superpowers/plans/2026-09-13-r2-security-hardening.md (quality-roadmap Release 2): dashboard bind/token/Host-allowlist hardening, Windows zip-slip fixes, CI Docker build, GHCR pull path + threat model docs."
gh run watch <run-id> --exit-status
```

The new `docker` job is NOT yet a required check, so the PR can merge on the existing four — after merge, add it (Step 4).

- [ ] **Step 3: Merge with a minor bump** — new user-facing features, so v1.1.0:

```bash
gh pr merge <PR#> --merge --subject "Merge R2 security hardening [minor]"
```

Watch CI on main, then the Release workflow: it must create `v1.1.0` + Release + push `ghcr.io/conradstorz/imageharbor:v1.1.0` and `:latest`. Record the run URL and job conclusions.

- [ ] **Step 4: Add the docker job to branch protection**

```bash
gh api -X PUT repos/conradstorz/ImageHarbor/branches/main/protection --input <file>
```

with the same JSON as R1's but contexts now `["lint", "test (ubuntu-latest, 3.10)", "test (ubuntu-latest, 3.13)", "test (windows-latest, 3.13)", "docker"]`. Verify with the `--jq .required_status_checks.contexts` read-back.

- [ ] **Step 5: Deploy to hpz440.** Check for ssh access: `ssh hpz440 "docker --version"` (also try `ssh conrad@hpz440` / the tailnet IP `100.69.239.123` if the alias fails). If reachable: on the host, update the checkout's `docker-compose.yml` (git pull), set a real `IMAGEHARBOR_DASHBOARD_TOKEN` and `IMAGEHARBOR_DASHBOARD_ALLOWED_HOSTS: "100.69.239.123"` (plus any hostname used in the browser), then `docker compose pull` and `docker compose up -d`, and verify `curl -fsS http://127.0.0.1:8087/healthz` on the host plus a 401 on an unauthenticated `curl -X POST http://127.0.0.1:8087/api/pause -H "Host: 100.69.239.123" -d "{}"`. If ssh is NOT reachable from this machine: write the exact command sequence into the report and finish DONE_WITH_CONCERNS naming the deploy as the user's manual step.

- [ ] **Step 6: Update the memory/roadmap bookkeeping** — mark Release 2 done in `docs/quality-roadmap.md` is NOT needed (the roadmap is a plan, not a status board); instead note in the report: R2 shipped as v1.1.0, deploy status, and any deferred leftovers.

---

## Self-Review Notes

- Roadmap R2 coverage: dashboard exposure (Tasks 1-4, 7), zip-slip both modules (5-6), threat model doc (7), hpz440 deploy (9); R1-final-review deferrals folded in: CI docker build + interpreter pins (8), GHCR pull docs + compose image ref (7).
- The Host-allowlist default (loopback names only) composes with the loopback bind default: an unconfigured `watch` run behaves exactly as before for a local operator, and both hardening layers activate only where compose/env opts in — no breaking change for `watch` users off Docker.
- Known test-impact risk called out in Task 1 Step 5: real-socket dashboard tests send loopback Hosts automatically; only the fake-socket harness needs the explicit default.
- Empty-env-means-unset wrinkle for compose (`""` token) is explicitly assigned to Task 7 Step 1 to verify against Task 3's implementation.
