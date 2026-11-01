#!/usr/bin/env python3
"""Latency of R2's speech front and back ends (no microphone needed).

* STT: macOS `say` renders each phrase to a WAV, MLX Whisper (small) transcribes it;
  reports latency, real-time factor and word error rate against the phrase.
* TTS: Kokoro-82M synthesizes R2-style replies; reports latency and real-time factor.

    uv run python training/bench_voice.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrow_ai.settings import load_settings  # noqa: E402
from narrow_ai.speech.neural_tts import NeuralTTS  # noqa: E402
from narrow_ai.speech.transcription import synthesize_to_wav, transcribe_wav  # noqa: E402

PHRASES = [
    "What's my git status?",
    "Find all the Swift files.",
    "Show me the README file.",
    "Open Visual Studio Code.",
    "Remember that my shell is zsh.",
    "How do I undo my last commit?",
    "How big is my diff right now?",
    "What can you do?",
]
REPLIES = [
    "You're on main and the working tree is clean.",
    "I found eleven files matching star dot swift.",
    "Open Safari? Say yes or no.",
    "Git merge and rebase both combine changes from different branches, but rebase moves "
    "your commits so the history stays linear.",
]


def _norm(s: str) -> list[str]:
    s = s.lower().replace("readme", "read me").replace("zsh", "z s h")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def wav_seconds(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def main() -> int:
    s = load_settings()
    tmp = Path(tempfile.mkdtemp())
    transcribe_wav(synthesize_to_wav("warm up", tmp / "w.wav"), model=s.speech.whisper_model)

    stt = []
    for i, phrase in enumerate(PHRASES):
        wav = synthesize_to_wav(phrase, tmp / f"p{i}.wav")
        t0 = time.perf_counter()
        text = transcribe_wav(wav, model=s.speech.whisper_model)
        ms = (time.perf_counter() - t0) * 1000
        dur = wav_seconds(wav)
        stt.append({"phrase": phrase, "heard": text, "ms": ms, "audio_s": dur,
                    "rtf": ms / 1000 / dur, "wer": wer(phrase, text)})
        print(f"STT {ms:6.0f} ms  rtf {ms / 1000 / dur:.3f}  wer {stt[-1]['wer']:.2f}  {text!r}")

    tts_engine = NeuralTTS()
    tts_engine.load()
    tts_engine.warmup()
    tts = []
    for reply in REPLIES:
        t0 = time.perf_counter()
        data = tts_engine.wav_bytes(reply, voice=s.speech.neural_voice)
        ms = (time.perf_counter() - t0) * 1000
        p = tmp / "r.wav"
        p.write_bytes(data)
        dur = wav_seconds(p)
        tts.append({"text": reply, "chars": len(reply), "ms": ms, "audio_s": dur,
                    "rtf": ms / 1000 / dur})
        print(f"TTS {ms:6.0f} ms  audio {dur:4.1f} s  rtf {ms / 1000 / dur:.3f}")

    report = {
        "whisper_model": s.speech.whisper_model,
        "tts_model": tts_engine.model_id,
        "voice": s.speech.neural_voice,
        "stt": stt,
        "tts": tts,
        "summary": {
            "stt_ms_median": statistics.median(r["ms"] for r in stt),
            "stt_rtf_median": statistics.median(r["rtf"] for r in stt),
            "stt_wer_mean": statistics.mean(r["wer"] for r in stt),
            "tts_ms_median": statistics.median(r["ms"] for r in tts),
            "tts_rtf_median": statistics.median(r["rtf"] for r in tts),
        },
    }
    out = ROOT / "artifacts" / "voice_bench.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
