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
        self._agent: Any = None
        self._planner: Any = None
        self._active_run: str | None = None  # run started from ask(), awaiting or done

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
        allow_runs: bool = False,
    ) -> AssistantResponse:
        """One text turn. A "yes" approves the pending action only if allowed:
        the HTTP API passes ``require_confirmation_id=True`` plus the client's id.

        ``allow_runs``: a request that plans to two or more steps becomes a bounded
        multi-step run (same gate and confirmation per step); everything else takes
        the normal single-step path."""
        with self._lock:
            resp = self._ask_locked(text, confirmation_id, require_confirmation_id, allow_runs)
        if speak:
            self.speak(resp.spoken_text())
        return resp

    def _awaiting_run(self) -> Any:
        if self._active_run is None or self._agent is None:
            return None
        from narrow_ai.agent.runner import RunStatus

        run = self._agent.get(self._active_run)
        return run if run is not None and run.status == RunStatus.AWAITING_CONFIRMATION else None

    def _confirm_active_run(self, run: Any, confirmation_id: str | None, require_id: bool):
        from narrow_ai.r2.runs import run_response

        since = run.cursor
        cid = confirmation_id if require_id else (run.pending.id if run.pending else None)
        return run_response(self.agent.confirm(run.id, cid), since=since)

    def _ask_locked(
        self, text: str, confirmation_id: str | None, require_id: bool, allow_runs: bool
    ) -> AssistantResponse:
        from narrow_ai.assistant.orchestrator import NO, YES, _norm

        run = self._awaiting_run()
        if run is not None:
            word = _norm(text)
            if word in YES:
                return self._confirm_active_run(run, confirmation_id, require_id)
            self.agent.cancel(run.id)  # "no", or a new request supersedes the paused run
            if word in NO:
                return AssistantResponse(text="Okay, cancelled.", speech="Okay, cancelled.")
        if allow_runs and self.settings.agent.enabled:
            started = self._maybe_start_run(text)
            if started is not None:
                return started
        return self.orch.handle_text(
            text, confirmation_id=confirmation_id, require_confirmation_id=require_id
        )

    def _maybe_start_run(self, text: str) -> AssistantResponse | None:
        """Plan once; start a run only for two or more steps. A plan with a refused
        part is refused as a whole (nothing runs)."""
        from pydantic import ValidationError

        from narrow_ai.agent.plan import PlanRejected, validate_plan
        from narrow_ai.agent.planner import split_clauses
        from narrow_ai.intents.classifier import IntentParseError
        from narrow_ai.r2.runs import FixedPlanner, looks_multistep, run_response

        if not looks_multistep(text):
            return None
        agent = self.agent
        try:
            plan = validate_plan(
                self._planner.plan(text), agent.registry, max_steps=self.settings.agent.max_steps
            )
        except (PlanRejected, IntentParseError, ValidationError, ValueError) as exc:
            if len(split_clauses(text)) < 2:
                return None  # one clause that isn't an action: the normal path answers it
            run = agent.start(text, FixedPlanner(rejection=exc))  # recorded; nothing runs
            self._active_run = run.id
            return run_response(run)
        if len(plan.steps) < 2:
            return None
        if self.orch.pending_action is not None:
            self.orch.cancel()  # a new request supersedes a pending single-step action
        run = agent.start(text, FixedPlanner(plan))
        self._active_run = run.id
        return run_response(run)

    def confirm(
        self,
        confirmation_id: str | None = None,
        *,
        require_id: bool = False,
        speak: bool = False,
    ) -> AssistantResponse:
        """Approve the pending action (``require_id=False`` only for in-process UIs).
        If a multi-step run is paused, this approves its current step instead."""
        with self._lock:
            run = self._awaiting_run()
            if run is not None:
                resp = self._confirm_active_run(run, confirmation_id, require_id)
            else:
                resp = self.orch.confirm(confirmation_id, require_id=require_id)
        if speak:
            self.speak(resp.spoken_text())
        return resp

    # ----------------------------------------------------------------- multi-step runs
    @property
    def agent(self) -> Any:
        """Bounded multi-step runtime (lazy; every step uses the same ToolExecutor path)."""
        if self._agent is None:
            from narrow_ai.agent.events import RunStore
            from narrow_ai.agent.planner import ClausePlanner, ModelPlanner
            from narrow_ai.agent.runner import AgentRunner

            if not self.settings.agent.enabled:
                raise PermissionError("agent runtime disabled (agent.enabled = false)")
            store = None
            if self.settings.agent.persist_runs:
                store = RunStore(self.settings.runs_db_path())
            self._agent = AgentRunner(
                self.settings,
                self.orch.registry,
                preferences=self.orch.preferences,
                answerer=self.orch.answer,
                explainer=self.orch.explain_tests,
                tool_state=self.orch.tool_state,
                store=store,
                audit=self.orch.audit,
                clock=lambda: self.orch.clock(),
            )
            if self.settings.agent.planner == "model" and self.backend.name != "fake":
                self._planner = ModelPlanner(self.settings, self.backend, self.orch.registry)
            else:  # with the fake backend the engine is the deterministic heuristic
                self._planner = ClausePlanner(
                    self.orch.engine,
                    self.settings.agent.max_steps,
                    split_and=self.settings.agent.split_and,
                    question_guard=self.settings.agent.question_guard,
                )
        return self._agent

    def run(self, text: str) -> dict[str, Any]:
        """Plan and start a multi-step run; includes the confirmation id if it paused."""
        with self._lock:
            agent = self.agent
            run = agent.start(text, self._planner)
            return {**run.to_dict(include_confirmation=True), "events": agent.events(run.id)}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self.agent.get(run_id)
            if run is None:
                return None
            return {**run.to_dict(), "events": self.agent.events(run_id)}

    def confirm_run(self, run_id: str, confirmation_id: str | None) -> dict[str, Any] | None:
        with self._lock:
            run = self.agent.confirm(run_id, confirmation_id)
            if run is None:
                return None
            return {**run.to_dict(include_confirmation=True), "events": self.agent.events(run_id)}

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self.agent.cancel(run_id)
            return None if run is None else {**run.to_dict(), "events": self.agent.events(run_id)}

    def cancel(self) -> AssistantResponse:
        with self._lock:
            if (run := self._awaiting_run()) is not None:
                self.agent.cancel(run.id)
            self.orch.cancel()
        return AssistantResponse(text="Okay, cancelled.", speech="Okay, cancelled.")

    def reset(self) -> None:
        with self._lock:
            if (run := self._awaiting_run()) is not None:
                self.agent.cancel(run.id)
            self._active_run = None
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
        allow_runs: bool = False,
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
            allow_runs=allow_runs,
        )

    def ask_audio(
        self,
        wav: Path,
        *,
        speak: bool = False,
        confirmation_id: str | None = None,
        require_confirmation_id: bool = False,
        allow_runs: bool = False,
    ) -> tuple[str, AssistantResponse | None]:
        transcript = self.transcribe(wav)
        if not transcript.strip():
            return "", None
        return transcript, self.ask(
            transcript,
            speak=speak,
            confirmation_id=confirmation_id,
            require_confirmation_id=require_confirmation_id,
            allow_runs=allow_runs,
        )
