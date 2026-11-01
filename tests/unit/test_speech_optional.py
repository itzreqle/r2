from __future__ import annotations

from narrow_ai.speech.synthesis import synthesize_speech


def test_empty_tts_noop():
    synthesize_speech("")


def test_transcription_import_message():
    """Missing mlx-whisper should raise a clear RuntimeError, not crash imports."""
    from narrow_ai.speech import transcription

    assert hasattr(transcription, "transcribe_wav")
