"""R2 HTTP API with the fake backend (no model, mic or network beyond localhost)."""

from __future__ import annotations

import http.client
import json
import stat
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from narrow_ai.r2.auth import ApiToken
from narrow_ai.r2.core import R2
from narrow_ai.r2.server import make_server
from narrow_ai.settings import Settings


class Api:
    def __init__(self, base: str, port: int, token: ApiToken, r2: R2) -> None:
        self.base, self.port, self.token, self.r2 = base, port, token, r2

    def request(
        self,
        method: str,
        path: str,
        body: dict | bytes | None = None,
        *,
        ctype: str = "application/json",
        auth: str | None = "default",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict]:
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
        hdrs = {"Content-Type": ctype} if method == "POST" else {}
        if auth == "default":
            hdrs["Authorization"] = f"Bearer {self.token.value}"
        elif auth is not None:
            hdrs["Authorization"] = auth
        hdrs.update(headers or {})
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path: str, body: dict | bytes | None = None, **kw) -> tuple[int, dict]:
        return self.request("POST", path, {} if body is None else body, **kw)

    def get(self, path: str, **kw) -> tuple[int, dict]:
        return self.request("GET", path, **kw)

    def raw(self, method: str, path: str, headers: dict[str, str], body: bytes = b"") -> int:
        """Send exact headers (urllib always rewrites Host)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for k, v in headers.items():
            conn.putheader(k, v)
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        code = conn.getresponse().status
        conn.close()
        return code


@pytest.fixture()
def api(tmp_path: Path):
    (tmp_path / "demo.py").write_text("print(1)\n")
    s = Settings()
    s.memory.db_path = str(tmp_path / "prefs.db")
    r2 = R2(s, backend="fake", workspace=tmp_path)
    token = ApiToken.create(tmp_path / "run" / "api-token")
    httpd = make_server(r2, token=token, port=0, speak=False)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    port = httpd.server_address[1]
    yield Api(f"http://127.0.0.1:{port}", port, token, r2)
    httpd.shutdown()
    httpd.server_close()


def test_health_status_and_ask(api):
    code, out = api.get("/health", auth=None)  # health is public
    assert code == 200 and out["ok"] is True
    assert api.get("/status")[1]["backend"] == "fake"
    code, out = api.post("/ask", {"text": "find python files", "speak": False})
    assert code == 200 and out["intent"] == "search_files" and "demo.py" in out["text"]


def test_bad_requests_are_rejected(api):
    assert api.post("/ask", {"text": ""})[0] == 400
    assert api.post("/ask_audio", b"not a wav file at all, definitely not RIFF data....",
                    ctype="audio/wav")[0] == 400
    assert api.post("/nope", {})[0] == 404


# --------------------------------------------------------------------- auth / origin
@pytest.mark.parametrize("auth", [None, "Bearer wrong-token", "Basic abc", "Bearer "])
def test_requests_without_valid_token_are_401(api, auth):
    assert api.get("/status", auth=auth)[0] == 401
    code, _ = api.post("/ask", {"text": "git status"}, auth=auth)
    assert code == 401
    assert api.post("/confirm", {}, auth=auth)[0] == 401
    assert api.post("/reset", {}, auth=auth)[0] == 401
    assert api.post("/listen/start", {}, auth=auth)[0] == 401


@pytest.mark.parametrize(
    "host", ["evil.test", "evil.test:8765", "127.0.0.1", "localhost:1", "", "127.0.0.1.nip.io"]
)
def test_foreign_host_header_is_403(api, host):
    """DNS rebinding: a browser sends the attacker's hostname in Host."""
    headers = {"Authorization": f"Bearer {api.token.value}"}
    if host:
        headers["Host"] = host
    assert api.raw("GET", "/status", headers) == 403
    assert api.raw("GET", "/health", headers) == 403


@pytest.mark.parametrize("host", ["127.0.0.1:{p}", "localhost:{p}", "LOCALHOST:{p}"])
def test_loopback_host_header_is_allowed(api, host):
    headers = {"Host": host.format(p=api.port), "Authorization": f"Bearer {api.token.value}"}
    assert api.raw("GET", "/status", headers) == 200


@pytest.mark.parametrize(
    "extra",
    [
        {"Origin": "https://evil.test"},
        {"Origin": "null"},
        {"Origin": "http://localhost:3000"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_browser_requests_are_403_even_with_token(api, extra):
    assert api.get("/status", headers=extra)[0] == 403
    assert api.post("/confirm", {}, headers=extra)[0] == 403


@pytest.mark.parametrize(
    "ctype", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data", ""]
)
def test_simple_request_content_types_are_415(api, ctype):
    """CSRF 'simple requests' can't set application/json without a preflight."""
    assert api.post("/ask", {"text": "git status"}, ctype=ctype)[0] == 415
    assert api.post("/ask_audio", b"RIFF" + b"\0" * 60, ctype=ctype or "text/plain")[0] == 415


def test_preflight_and_other_verbs_are_refused(api):
    headers = {
        "Host": f"127.0.0.1:{api.port}",
        "Access-Control-Request-Method": "POST",
    }
    assert api.raw("OPTIONS", "/ask", headers) == 405
    assert api.raw("DELETE", "/reset", headers) == 405


def test_malformed_content_length_is_400(api):
    headers = {
        "Host": f"127.0.0.1:{api.port}",
        "Authorization": f"Bearer {api.token.value}",
        "Content-Type": "application/json",
    }
    conn = http.client.HTTPConnection("127.0.0.1", api.port, timeout=10)
    conn.putrequest("POST", "/ask", skip_host=True)
    for k, v in {**headers, "Content-Length": "abc"}.items():
        conn.putheader(k, v)
    conn.endheaders()
    assert conn.getresponse().status == 400
    conn.close()


def test_token_file_is_private(api):
    mode = stat.S_IMODE(api.token.path.stat().st_mode)
    assert mode == 0o600
    assert api.token.path.read_text() == api.token.value
    assert len(api.token.value) >= 40


def test_non_loopback_bind_is_refused(tmp_path):
    s = Settings()
    s.memory.db_path = str(tmp_path / "prefs.db")
    r2 = R2(s, backend="fake", workspace=tmp_path)
    token = ApiToken.create(tmp_path / "tok")
    for host in ("0.0.0.0", "192.168.1.10", "example.com"):
        with pytest.raises(ValueError, match="non-loopback"):
            make_server(r2, token=token, host=host, port=0)


# --------------------------------------------------------------------- confirmations
def _park_remember(api) -> dict:
    code, out = api.post("/ask", {"text": "remember editor=Cursor", "speak": False})
    assert code == 200 and out["needs_confirmation"]
    assert out["confirmation_id"] and out["expires_in"] > 0
    assert api.r2.orch.preferences.get("editor") is None
    return out


def test_confirmation_with_id_survives_between_requests(api):
    out = _park_remember(api)
    code, done = api.post("/confirm", {"confirmation_id": out["confirmation_id"]})
    assert code == 200 and done["ok"] and api.r2.orch.preferences.get("editor") == "Cursor"
    # Single use: replaying the same id does nothing.
    code, again = api.post("/confirm", {"confirmation_id": out["confirmation_id"]})
    assert again["text"] == "Nothing to confirm."


def test_confirm_without_or_with_wrong_id_never_executes(api):
    _park_remember(api)
    code, out = api.post("/confirm", {})
    assert code == 200 and not out["ok"] and "cancelled" in out["text"]
    assert api.r2.orch.preferences.get("editor") is None
    assert api.r2.orch.pending is None  # fail closed: the attempt cancelled it

    _park_remember(api)
    code, out = api.post("/confirm", {"confirmation_id": "x" * 22})
    assert not out["ok"] and api.r2.orch.preferences.get("editor") is None


@pytest.mark.parametrize("word", ["yes", "Yes!", "do it", "ok"])
def test_typed_yes_without_id_does_not_confirm(api, word):
    _park_remember(api)
    code, out = api.post("/ask", {"text": word})
    assert not out["ok"] and api.r2.orch.preferences.get("editor") is None


def test_typed_yes_with_matching_id_confirms(api):
    """Hands-free path: the client echoes the id of the prompt it is showing."""
    out = _park_remember(api)
    code, done = api.post("/ask", {"text": "yes", "confirmation_id": out["confirmation_id"]})
    assert done["ok"] and api.r2.orch.preferences.get("editor") == "Cursor"


def test_old_id_cannot_approve_a_newer_action(api):
    first = _park_remember(api)
    code, second = api.post("/ask", {"text": "remember theme=dark"})
    assert second["confirmation_id"] != first["confirmation_id"]
    code, out = api.post("/confirm", {"confirmation_id": first["confirmation_id"]})
    assert not out["ok"]
    prefs = api.r2.orch.preferences
    assert prefs.get("theme") is None and prefs.get("editor") is None


def test_expired_confirmation_is_rejected(api):
    now = [1000.0]
    api.r2.orch.clock = lambda: now[0]
    out = _park_remember(api)
    now[0] += api.r2.settings.security.confirmation_ttl_sec + 1
    code, res = api.post("/confirm", {"confirmation_id": out["confirmation_id"]})
    assert not res["ok"] and "expired" in res["text"]
    assert api.r2.orch.preferences.get("editor") is None


def test_status_does_not_leak_confirmation_id(api):
    out = _park_remember(api)
    code, st = api.get("/status")
    assert st["pending"]["intent"] == "remember_preference"
    assert out["confirmation_id"] not in json.dumps(st)
