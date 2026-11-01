"""Tiny local HTTP API for R2 (stdlib only, binds to 127.0.0.1).

The model and KV cache stay warm in this one process, and pending confirmations
survive between requests.

GET  /health            -> {"ok": true}
GET  /status            -> model / KV cache / state
POST /ask      {"text": "...", "speak": bool}
POST /confirm  {"speak": bool}
POST /cancel
POST /reset
POST /listen/start                      (server-side mic, push-to-talk)
POST /listen/stop {"speak": bool}       -> transcript + response
POST /ask_audio?speak=0  (raw audio/wav body, recorded by the client app)
POST /tts      {"text": "...", "voice": "af_heart", "speed": 1.0} -> audio/wav (neural voice)
GET  /voices            -> available neural voices
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)

MAX_BODY = 64 * 1024
MAX_AUDIO = 16 * 1024 * 1024


def make_handler(r2: Any, *, speak_default: bool = True) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()
    tts_lock = threading.Lock()  # TTS runs beside (not behind) the LLM lock

    def speak_async(text: str, want: bool) -> None:
        if want:
            threading.Thread(target=r2.speak, args=(text,), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        server_version = "R2/0.2"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            if n > MAX_BODY:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON object expected")
            return data

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if self.path == "/status":
                return self._send(200, r2.status())
            if self.path == "/voices":
                from narrow_ai.speech.neural_tts import VOICES

                return self._send(200, {"voices": VOICES})
            return self._send(404, {"error": "not found"})

        def _ask_audio(self) -> None:
            import tempfile
            from pathlib import Path
            from urllib.parse import parse_qs, urlparse

            n = int(self.headers.get("Content-Length") or 0)
            if n <= 44 or n > MAX_AUDIO:
                return self._send(400, {"error": "WAV body required (max 16 MB)"})
            data = self.rfile.read(n)
            if data[:4] != b"RIFF":
                return self._send(400, {"error": "body must be a WAV file"})
            q = parse_qs(urlparse(self.path).query)
            want = q.get("speak", ["1" if speak_default else "0"])[0] == "1"
            fd, name = tempfile.mkstemp(suffix=".wav", prefix="r2-client-")
            wav = Path(name)
            try:
                with open(fd, "wb") as fh:
                    fh.write(data)
                with lock:
                    transcript, resp = r2.ask_audio(wav)
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
                with tts_lock:
                    data = r2.tts_wav(
                        text, voice=body.get("voice") or None, speed=float(body.get("speed", 1.0))
                    )
            except Exception as exc:  # noqa: BLE001
                return self._send(503, {"error": str(exc)})
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.startswith("/ask_audio"):
                return self._ask_audio()
            if self.path == "/tts":
                return self._tts()
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})
            want_speech = bool(body.get("speak", speak_default))
            try:
                with lock:
                    if self.path == "/ask":
                        text = str(body.get("text", "")).strip()[:2000]
                        if not text:
                            return self._send(400, {"error": "text required"})
                        resp = r2.ask(text)
                    elif self.path == "/confirm":
                        resp = r2.confirm()
                    elif self.path == "/cancel":
                        resp = r2.cancel()
                    elif self.path == "/reset":
                        r2.reset()
                        return self._send(200, {"ok": True})
                    elif self.path == "/listen/start":
                        r2.listen_start()
                        return self._send(200, {"state": "listening"})
                    elif self.path == "/listen/stop":
                        transcript, resp = r2.listen_stop()
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


def make_server(r2: Any, *, host: str = "127.0.0.1", port: int = 8765, speak: bool = True):
    return ThreadingHTTPServer((host, port), make_handler(r2, speak_default=speak))


def serve_forever(r2: Any, *, host: str = "127.0.0.1", port: int = 8765, speak: bool = True):
    httpd = make_server(r2, host=host, port=port, speak=speak)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
