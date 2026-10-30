"""Local text-to-speech with Piper (https://github.com/rhasspy/piper).

synthesize_wav() returns complete WAV file bytes for one sentence, which
the browser decodes and plays. The voice model is downloaded on first
use into models/piper/ unless already present.
"""

from __future__ import annotations

import io
import logging
import wave
from functools import lru_cache
from pathlib import Path

from piper import PiperVoice

from interviewer import config

logger = logging.getLogger(__name__)


def _voice_paths() -> tuple[Path, Path]:
    voice_name = config.PIPER_VOICE_NAME
    # en_US-amy-medium -> en/en_US/amy/medium/en_US-amy-medium.onnx
    lang_family, rest = voice_name.split("_", 1)
    lang_code, voice_name_part, quality = rest.split("-", maxsplit=2)
    url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang_family}/{lang_code}/{voice_name_part}/{quality}/"
        f"{voice_name}.onnx?download=true"
    )
    config.PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = config.PIPER_MODELS_DIR / f"{voice_name}.onnx"
    return onnx_path, url


def ensure_voice_downloaded() -> None:
    """Download the voice model from Hugging Face on first use."""
    onnx_path, url = _voice_paths()
    if onnx_path.exists() and onnx_path.stat().st_size > 0:
        return

    import urllib.request

    logger.info("Downloading Piper voice %s ...", config.PIPER_VOICE_NAME)
    tmp_path = onnx_path.with_suffix(".onnx.part")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(onnx_path)
        logger.info("Downloaded %s", onnx_path.name)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


@lru_cache(maxsize=1)
def get_voice() -> PiperVoice:
    ensure_voice_downloaded()
    onnx_path, _ = _voice_paths()
    return PiperVoice.load(
        onnx_path,
        config_path=str(onnx_path.with_suffix(".onnx.json"))
        if (onnx_path.with_suffix(".onnx.json")).exists()
        else None,
    )


def synthesize_wav(text: str) -> bytes:
    """Synthesize one sentence into complete WAV file bytes."""
    text = (text or "").strip()
    if not text:
        return b""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        from piper import SynthesisConfig

        get_voice().synthesize_wav(
            text,
            wav_file,
            syn_config=SynthesisConfig(
                length_scale=config.PIPER_LENGTH_SCALE
            ),
        )

    return buffer.getvalue()
