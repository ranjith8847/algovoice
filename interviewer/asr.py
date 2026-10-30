"""Local speech recognition with faster-whisper.

The microphone pipeline feeds 16 kHz mono 16-bit PCM; this module turns a
completed utterance into text. The language is pinned (default English)
because Whisper's auto-detect frequently misclassifies short English
clips and outputs transliteration instead of translation.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache

from faster_whisper import WhisperModel

from interviewer import config


@lru_cache(maxsize=1)
def load_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_pcm(pcm_bytes: bytes) -> str:
    """Transcribe raw 16 kHz mono 16-bit PCM captured from the browser."""
    model = load_model(
        config.WHISPER_MODEL_SIZE,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE,
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as f:
            import contextlib
            import wave

            with contextlib.closing(wave.open(f, "wb")) as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(config.SAMPLE_RATE)
                wav_file.writeframes(pcm_bytes)
            temp_path = f.name

        segments, _info = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True,
            language=(
                config.TRANSCRIBE_LANGUAGE
                if config.TRANSCRIBE_LANGUAGE != "auto"
                else None
            ),
        )

        return " ".join(
            segment.text.strip() for segment in segments
        ).strip()

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
