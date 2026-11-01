"""R2 session: one warm model + KV cache, text/voice turns, spoken replies.

Used by the terminal app, the HTTP server (SwiftUI client) and the notebooks.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from narrow_ai.assistant.orchestrator import AssistantOrchestrator
from narrow_ai.assistant.response import AssistantResponse
from narrow_ai.models.model_registry import create_backend
from narrow_ai.settings import Settings, load_settings

logger = logging.getLogger(__name__)


class R2:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: str | None = None,
        workspace: str | Path | None = None,
        cache_mode: str | None = None,
        warmup: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        if backend:
            self.settings.model.backend = backend
        if cache_mode:
            self.settings.model.cache_mode = cache_mode
        if workspace is not None:
            self.settings.workspace.path = str(Path(workspace).expanduser().resolve())
        t0 = time.perf_counter()
        self.backend = create_backend(self.settings)
        self.orch = AssistantOrchestrator(self.settings, self.backend)
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.warmup_ms = 0.0
        if warmup:
            t1 = time.perf_counter()
            self.orch.warmup()
            self.warmup_ms = (time.perf_counter() - t1) * 1000
        self._recorder: Any = None
        self._lock = threading.RLock()
        self.last_transcript = ""

    # ----------------------------------------------------------------- info
    @property
    def state(self) -> str:
        if self._recorder is not None and self._recorder.active:
            return "listening"
        return str(self.orch.state)

    def status(self) -> dict[str, Any]:
        st = self.backend.cache_stats()
        return {
            "name": self.settings.assistant.name,
            "state": self.state,
            "backend": self.backend.name,
            "model": self.settings.model.model_id if self.backend.name == "mlx" else "heuristic",
            "workspace": str(self.settings.workspace_path()),
            "load_ms": round(self.load_ms, 1),
            "warmup_ms": round(self.warmup_ms, 1),
            "kv_cache": {
                "mode": st.mode,
                "source": st.source,
                "prefix_tokens": st.prefix_tokens,
                "build_ms": round(st.build_ms, 1),
                "load_ms": round(st.load_ms, 1),
                "hits": st.hits,
                "misses": st.misses,
                "mb": round(st.extra.get("nbytes", 0) / 1e6, 1),
            },
            # The confirmation id is deliberately not exposed here.
            "pending": (
                {
                    **pa.decision.model_dump(mode="json"),
                    "expires_in": round(pa.expires_in, 1),
                }
                if (pa := self.orch.pending_action)
                else None
            ),
        }

    # ----------------------------------------------------------------- text
    def ask(
        self,
        text: str,
        *,
        speak: bool = False,
        confirmation_id: str | None = None,
        require_confirmation_id: bool = False,
    ) -> AssistantResponse:
        """One text turn. A "yes" approves the pending action only if allowed:
        the HTTP API passes ``require_confirmation_id=True`` plus the client's id."""
        with self._lock:
            resp = self.orch.handle_text(
                text,
                confirmation_id=confirmation_id,
                require_confirmation_id=require_confirmation_id,
            )
        if speak:
            self.speak(resp.spoken_text())
        return resp

    def confirm(
        self,
        confirmation_id: str | None = None,
        *,
        require_id: bool = False,
        speak: bool = False,
    ) -> AssistantResponse:
        """Approve the pending action (``require_id=False`` only for in-process UIs)."""
        with self._lock:
            resp = self.orch.confirm(confirmation_id, require_id=require_id)
        if speak:
            self.speak(resp.spoken_text())
        return resp

    def cancel(self) -> AssistantResponse:
        with self._lock:
            self.orch.cancel()
        return AssistantResponse(text="Okay, cancelled.", speech="Okay, cancelled.")

    def reset(self) -> None:
        with self._lock:
            self.orch.reset()

    # ----------------------------------------------------------------- voice
    def speak(self, text: str, *, block: bool = True) -> None:
        from narrow_ai.speech.synthesis import synthesize_speech

        try:
            neural = self.settings.speech.tts_backend in {"kokoro", "neural"}
            synthesize_speech(
                text,
                backend=self.settings.speech.tts_backend,
                voice=(self.settings.speech.neural_voice if neural else self.settings.speech.voice)
                or None,
                block=block,
            )
        except Exception as exc:  # noqa: BLE001 - TTS must never crash R2
            logger.warning("TTS failed: %s", exc)

    def tts_wav(self, text: str, *, voice: str | None = None, speed: float = 1.0) -> bytes:
        """Natural-voice WAV for clients (raises if the tts group is missing)."""
        from narrow_ai.speech.neural_tts import shared_tts

        tts = shared_tts()
        if tts is None:
            raise RuntimeError("Neural TTS not installed: uv sync --group tts")
        return tts.wav_bytes(text, voice=voice or self.settings.speech.neural_voice, speed=speed)

    def warmup_tts(self) -> bool:
        from narrow_ai.speech.neural_tts import shared_tts

        tts = shared_tts()
        if tts is None:
            return False
        try:
            tts.warmup()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("neural TTS unavailable: %s", exc)
            return False

    def chime(self, kind: str) -> None:
        if self.settings.speech.chimes:
            from narrow_ai.speech.synthesis import chime

            chime(kind)

    def transcribe(self, wav: Path) -> str:
        from narrow_ai.speech.transcription import transcribe_wav

        text = transcribe_wav(wav, model=self.settings.speech.whisper_model)
        self.last_transcript = text
        return text

    def listen_start(self) -> None:
        from narrow_ai.speech.input import Recorder

        if self._recorder is None:
            self._recorder = Recorder(sample_rate=self.settings.speech.sample_rate)
        self.chime("listen")
        self._recorder.start()

    def listen_stop(
        self,
        *,
        speak: bool = False,
        confirmation_id: str | None = None,
        require_confirmation_id: bool = False,
    ) -> tuple[str, AssistantResponse | None]:
        if self._recorder is None or not self._recorder.active:
            raise RuntimeError("not listening")
        wav = self._recorder.stop()
        self.chime("done")
        try:
            transcript = self.transcribe(wav)
        finally:
            wav.unlink(missing_ok=True)
        if not transcript.strip():
            return "", None
        return transcript, self.ask(
            transcript,
            speak=speak,
            confirmation_id=confirmation_id,
            require_confirmation_id=require_confirmation_id,
        )

    def ask_audio(
        self,
        wav: Path,
        *,
        speak: bool = False,
        confirmation_id: str | None = None,
        require_confirmation_id: bool = False,
    ) -> tuple[str, AssistantResponse | None]:
        transcript = self.transcribe(wav)
        if not transcript.strip():
            return "", None
        return transcript, self.ask(
            transcript,
            speak=speak,
            confirmation_id=confirmation_id,
            require_confirmation_id=require_confirmation_id,
        )
