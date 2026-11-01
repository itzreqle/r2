"""R2 HTTP API with the fake backend (no model, mic or network beyond localhost)."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from narrow_ai.r2.core import R2
from narrow_ai.r2.server import make_server
from narrow_ai.settings import Settings


@pytest.fixture()
def api(tmp_path: Path):
    (tmp_path / "demo.py").write_text("print(1)\n")
    s = Settings()
    s.memory.db_path = str(tmp_path / "prefs.db")
    r2 = R2(s, backend="fake", workspace=tmp_path)
    httpd = make_server(r2, port=0, speak=False)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, r2
    httpd.shutdown()
    httpd.server_close()


def _post(url: str, body: dict | bytes, ctype: str = "application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_status_and_ask(api):
    base, _ = api
    with urllib.request.urlopen(base + "/health") as r:
        assert json.loads(r.read())["ok"] is True
    with urllib.request.urlopen(base + "/status") as r:
        assert json.loads(r.read())["backend"] == "fake"
    code, out = _post(base + "/ask", {"text": "find python files", "speak": False})
    assert code == 200 and out["intent"] == "search_files" and "demo.py" in out["text"]


def test_confirmation_survives_between_requests(api):
    base, r2 = api
    code, out = _post(base + "/ask", {"text": "remember editor=Cursor", "speak": False})
    assert out["needs_confirmation"] and r2.orch.preferences.get("editor") is None
    code, out = _post(base + "/confirm", {"speak": False})
    assert out["ok"] and r2.orch.preferences.get("editor") == "Cursor"
    code, out = _post(base + "/confirm", {"speak": False})
    assert out["text"] == "Nothing to confirm."


def test_bad_requests_are_rejected(api):
    base, _ = api
    assert _post(base + "/ask", {"text": ""})[0] == 400
    assert _post(base + "/ask_audio", b"not a wav file at all, definitely not RIFF data....",
                 "audio/wav")[0] == 400
    assert _post(base + "/nope", {})[0] == 404
