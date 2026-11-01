"""Microphone capture: push-to-talk Recorder (start/stop) + blocking CLI helper."""

from __future__ import annotations

import tempfile
import threading
import wave
from pathlib import Path
from typing import Any


def _sd() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install speech extras: uv sync --group speech") from exc
    return sd


def write_wav(audio: Any, sample_rate: int, path: Path | None = None) -> Path:
    import numpy as np

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    out = path or Path(tempfile.mkstemp(suffix=".wav", prefix="r2-")[1])
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out


class Recorder:
    """Non-blocking push-to-talk recorder: start() … stop() -> wav path."""

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 15.0) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self._chunks: list[Any] = []
        self._stream: Any = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        sd = _sd()
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []

            def callback(indata, frames, time, status):  # noqa: ARG001
                self._chunks.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback
            )
            self._stream.start()

    def stop(self) -> Path:
        import numpy as np

        with self._lock:
            if self._stream is None:
                raise RuntimeError("Recorder was not started")
            self._stream.stop()
            self._stream.close()
            self._stream = None
            chunks, self._chunks = self._chunks, []
        if not chunks:
            raise RuntimeError("No audio captured (check microphone permission)")
        audio = np.concatenate(chunks, axis=0)[:, 0]
        audio = audio[: int(self.sample_rate * self.max_seconds)]
        return write_wav(audio, self.sample_rate)


def record_push_to_talk(*, sample_rate: int = 16000, max_seconds: float = 15.0) -> Path:
    """Record until Enter is pressed (blocking CLI helper)."""
    rec = Recorder(sample_rate=sample_rate, max_seconds=max_seconds)
    rec.start()
    try:
        input()
    finally:
        path = rec.stop()
    return path
