"""Natural on-device voice with Kokoro-82M (mlx-audio). ~12x faster than real time on M4.

Optional: needs the `tts` dependency group. Callers fall back to macOS `say`.
"""

from __future__ import annotations

import io
import logging
import threading
import wave
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"

# Curated Kokoro voices (prefix: a=American, b=British; f=female, m=male).
VOICES = {
    "af_heart": "Heart — warm, natural (default)",
    "af_bella": "Bella — bright, friendly",
    "af_nicole": "Nicole — soft, calm",
    "af_sarah": "Sarah — clear, neutral",
    "am_michael": "Michael — warm male",
    "am_fenrir": "Fenrir — deep male",
    "bf_emma": "Emma — British",
    "bm_george": "George — British male",
}


class NeuralTTS:
    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model

        from narrow_ai.models.hub import local_or_repo

        self._model = load_model(local_or_repo(self.model_id))

    def synthesize(self, text: str, *, voice: str = "af_heart", speed: float = 1.0) -> Any:
        """Return float32 mono audio at 24 kHz."""
        import numpy as np

        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        if voice not in VOICES:
            voice = "af_heart"
        speed = float(min(max(speed, 0.7), 1.3))
        with self._lock:
            self.load()
            chunks = [
                np.asarray(r.audio, dtype=np.float32).reshape(-1)
                for r in self._model.generate(text=text, voice=voice, speed=speed)
            ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def wav_bytes(self, text: str, *, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        import numpy as np

        audio = self.synthesize(text, voice=voice, speed=speed)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def warmup(self) -> None:
        """Load model + G2P and compile kernels so the first reply is fast."""
        self.synthesize("Ready.", voice="af_heart")


_shared: NeuralTTS | None = None


def shared_tts() -> NeuralTTS | None:
    """Process-wide instance, or None if the tts group isn't installed."""
    global _shared
    if _shared is None:
        try:
            import mlx_audio  # noqa: F401
        except ImportError:
            return None
        _shared = NeuralTTS()
    return _shared
