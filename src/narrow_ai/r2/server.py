"""Tiny local HTTP API for R2 (stdlib only, loopback only).

The model and KV cache stay warm in this one process, and pending confirmations
survive between requests.

Every request passes a guard before routing (fail closed):

1. ``Host`` must be a loopback name with this server's port (DNS rebinding).
2. Requests carrying ``Origin`` or a cross-/same-site ``Sec-Fetch-Site`` are
   refused: the native client sends neither, browsers always do.
3. All endpoints except ``/health`` need ``Authorization: Bearer <token>``
   (see :mod:`narrow_ai.r2.auth`).
4. POST bodies must be ``application/json`` (``audio/wav`` for ``/ask_audio``),
   which forces a CORS preflight that this server never answers.

Approving a privileged action requires the single-use ``confirmation_id`` that
was returned with the prompt, for ``/confirm`` and for a spoken/typed "yes".

GET  /health            -> {"ok": true}                      (no auth)
GET  /status            -> model / KV cache / state
POST /ask      {"text": "...", "speak": bool, "confirmation_id": "..."}
POST /confirm  {"confirmation_id": "...", "speak": bool}
POST /cancel
POST /reset
POST /listen/start                      (server-side mic, push-to-talk)
POST /listen/stop {"speak": bool, "confirmation_id": "..."} -> transcript + response
POST /ask_audio?speak=0&confirmation_id=...  (raw audio/wav body recorded by the client)
POST /tts      {"text": "...", "voice": "af_heart", "speed": 1.0} -> audio/wav (neural voice)
GET  /voices            -> available neural voices
POST /run      {"text": "..."}          -> bounded multi-step run (docs/AGENT_RUNTIME.md)
GET  /runs/<id>                         -> run status, steps, events
POST /runs/<id>/confirm {"confirmation_id": "..."}
POST /runs/<id>/cancel
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from narrow_ai.r2.auth import ApiToken

logger = logging.getLogger(__name__)

MAX_BODY = 64 * 1024
MAX_AUDIO = 16 * 1024 * 1024
PUBLIC_PATHS = {"/health"}
RUN_PATH = re.compile(r"^/runs/([A-Za-z0-9_-]{1,32})(?:/(confirm|cancel))?$")
WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}
BROWSER_SITES = {"cross-site", "same-site"}


class _BadRequest(ValueError):
    pass


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_handler(
    r2: Any, *, token: ApiToken, speak_default: bool = True
) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()
    tts_lock = threading.Lock()  # TTS runs beside (not behind) the LLM lock

    def speak_async(text: str, want: bool) -> None:
        if want:
            threading.Thread(target=r2.speak, args=(text,), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        server_version = "R2/0.3"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(
            self, code: int, payload: dict[str, Any], headers: dict[str, str] | None = None
        ) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------------ guard
        @property
        def route(self) -> str:
            return urlparse(self.path).path

        def _allowed_hosts(self) -> set[str]:
            port = self.server.server_address[1]
            return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

        def _guard(self) -> bool:
            """Return True if the request may proceed; otherwise reply and return False."""
            if self.headers.get("Host", "").lower() not in self._allowed_hosts():
                self._send(403, {"error": "forbidden host"})
                return False
            site = self.headers.get("Sec-Fetch-Site", "").lower()
            if self.headers.get("Origin") is not None or site in BROWSER_SITES:
                self._send(403, {"error": "browser requests are not allowed"})
                return False
            if self.route not in PUBLIC_PATHS and not token.verify(
                self.headers.get("Authorization")
            ):
                self._send(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
                return False
            if self.command == "POST":
                ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                ok = ctype in WAV_TYPES if self.route == "/ask_audio" else (
                    ctype == "application/json"
                )
                if not ok:
                    self._send(415, {"error": "unsupported content type"})
                    return False
            return True

        def _length(self, limit: int) -> int:
            raw = self.headers.get("Content-Length") or "0"
            try:
                n = int(raw)
            except ValueError as exc:
                raise _BadRequest("invalid Content-Length") from exc
            if n < 0:
                raise _BadRequest("invalid Content-Length")
            if n > limit:
                raise _BadRequest("request too large")
            return n

        def _body(self) -> dict[str, Any]:
            n = self._length(MAX_BODY)
            if n == 0:
                return {}
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise _BadRequest("JSON object expected")
            return data

        @staticmethod
        def _cid(value: Any) -> str | None:
            return value if isinstance(value, str) and value else None

        def _not_allowed(self) -> None:
            self._send(405, {"error": "method not allowed"}, {"Allow": "GET, POST"})

        # No CORS: preflights (OPTIONS) and other verbs are simply refused.
        do_OPTIONS = do_PUT = do_DELETE = do_PATCH = _not_allowed

        # ------------------------------------------------------------ routes
        def do_GET(self) -> None:  # noqa: N802
            if not self._guard():
                return None
            if self.route == "/health":
                return self._send(200, {"ok": True})
            if self.route == "/status":
                return self._send(200, r2.status())
            if self.route == "/voices":
                from narrow_ai.speech.neural_tts import VOICES

                return self._send(200, {"voices": VOICES})
            m = RUN_PATH.match(self.route)
            if m and m.group(2) is None:
                return self._run_call(lambda: r2.get_run(m.group(1)))
            return self._send(404, {"error": "not found"})

        def _run_call(self, fn: Any) -> None:
            try:
                with lock:
                    out = fn()
            except PermissionError as exc:
                return self._send(403, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                logger.exception("run request failed")
                return self._send(500, {"error": str(exc)})
            if out is None:
                return self._send(404, {"error": "no such run"})
            return self._send(200, out)

        def _ask_audio(self) -> None:
            import tempfile

            try:
                n = self._length(MAX_AUDIO)
            except _BadRequest as exc:
                return self._send(400, {"error": str(exc)})
            if n <= 44:
                return self._send(400, {"error": "WAV body required (max 16 MB)"})
            data = self.rfile.read(n)
            if data[:4] != b"RIFF":
                return self._send(400, {"error": "body must be a WAV file"})
            q = parse_qs(urlparse(self.path).query)
            want = q.get("speak", ["1" if speak_default else "0"])[0] == "1"
            cid = self._cid(q.get("confirmation_id", [""])[0])
            fd, name = tempfile.mkstemp(suffix=".wav", prefix="r2-client-")
            wav = Path(name)
            try:
                with open(fd, "wb") as fh:
                    fh.write(data)
                with lock:
                    transcript, resp = r2.ask_audio(
                        wav, confirmation_id=cid, require_confirmation_id=True
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("audio request failed")
                return self._send(500, {"error": str(exc), "state": "error"})
            finally:
                wav.unlink(missing_ok=True)
            if resp is None:
                return self._send(200, {"transcript": "", "text": "I didn't catch that.",
                                        "speech": "I didn't catch that.", "state": "idle"})
            payload = resp.to_dict()
            payload["transcript"] = transcript
            speak_async(resp.spoken_text(), want)
            return self._send(200, payload)

        def _tts(self) -> None:
            try:
                body = self._body()
                text = str(body.get("text", "")).strip()[:1500]
                if not text:
                    return self._send(400, {"error": "text required"})
                speed = min(2.0, max(0.5, float(body.get("speed", 1.0))))
                with tts_lock:
                    data = r2.tts_wav(text, voice=body.get("voice") or None, speed=speed)
            except (_BadRequest, json.JSONDecodeError, TypeError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                return self._send(503, {"error": str(exc)})
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if not self._guard():
                return None
            route = self.route
            if route == "/ask_audio":
                return self._ask_audio()
            if route == "/tts":
                return self._tts()
            try:
                body = self._body()
            except (_BadRequest, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})
            want_speech = bool(body.get("speak", speak_default))
            cid = self._cid(body.get("confirmation_id"))
            if route == "/run":
                text = str(body.get("text", "")).strip()[:2000]
                if not text:
                    return self._send(400, {"error": "text required"})
                return self._run_call(lambda: r2.run(text))
            m = RUN_PATH.match(route)
            if m and m.group(2) == "confirm":
                return self._run_call(lambda: r2.confirm_run(m.group(1), cid))
            if m and m.group(2) == "cancel":
                return self._run_call(lambda: r2.cancel_run(m.group(1)))
            try:
                with lock:
                    if route == "/ask":
                        text = str(body.get("text", "")).strip()[:2000]
                        if not text:
                            return self._send(400, {"error": "text required"})
                        resp = r2.ask(text, confirmation_id=cid, require_confirmation_id=True)
                    elif route == "/confirm":
                        resp = r2.confirm(cid, require_id=True)
                    elif route == "/cancel":
                        resp = r2.cancel()
                    elif route == "/reset":
                        r2.reset()
                        return self._send(200, {"ok": True})
                    elif route == "/listen/start":
                        r2.listen_start()
                        return self._send(200, {"state": "listening"})
                    elif route == "/listen/stop":
                        transcript, resp = r2.listen_stop(
                            confirmation_id=cid, require_confirmation_id=True
                        )
                        if resp is None:
                            return self._send(
                                200,
                                {"transcript": "", "text": "I didn't catch that.",
                                 "speech": "I didn't catch that.", "state": "idle"},
                            )
                        payload = resp.to_dict()
                        payload["transcript"] = transcript
                        speak_async(resp.spoken_text(), want_speech)
                        return self._send(200, payload)
                    else:
                        return self._send(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                logger.exception("request failed")
                return self._send(500, {"error": str(exc), "state": "error"})
            speak_async(resp.spoken_text(), want_speech)
            return self._send(200, resp.to_dict())

    return Handler


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_server(
    r2: Any,
    *,
    token: ApiToken,
    host: str = "127.0.0.1",
    port: int = 8765,
    speak: bool = True,
    allow_remote: bool = False,
) -> ThreadingHTTPServer:
    """Build the server. Non-loopback binds are refused unless ``allow_remote``."""
    if not allow_remote and not is_loopback_host(host):
        raise ValueError(f"refusing to bind R2 API to non-loopback address {host!r}")
    cls = _V6Server if ":" in host else ThreadingHTTPServer
    return cls((host, port), make_handler(r2, token=token, speak_default=speak))


def serve_forever(
    r2: Any,
    *,
    token_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    speak: bool = True,
) -> None:
    if not is_loopback_host(host):  # fail before creating a token
        raise ValueError(f"refusing to bind R2 API to non-loopback address {host!r}")
    token = ApiToken.create(token_path)
    try:
        httpd = make_server(r2, token=token, host=host, port=port, speak=speak)
    except Exception:
        token.remove()
        raise

    # The SwiftUI app stops the core with SIGTERM (Process.terminate); treat it like
    # Ctrl+C so the token file is always cleaned up.
    previous = None
    if threading.current_thread() is threading.main_thread():
        def _on_term(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, _on_term)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        token.remove()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
