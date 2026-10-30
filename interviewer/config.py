"""Central configuration for the real-time voice interviewer.

Every value can be overridden with an environment variable so you can tune
the pipeline without editing code. See README in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"
PIPER_MODELS_DIR = PROJECT_ROOT / "models" / "piper"

# ---------------------------------------------------------------------------
# LM Studio (interview reasoning)
# ---------------------------------------------------------------------------
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
# Empty means "auto-detect": ask LM Studio's /v1/models endpoint at start.
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))

# Decision layer (Phase 4): background interview assessment.
DECISION_MAX_TOKENS = int(os.getenv("DECISION_MAX_TOKENS", "900"))

# Bounded queue between the LLM streaming thread and the TTS thread
# (backpressure so audio cannot accumulate unbounded).
TTS_QUEUE_MAX = int(os.getenv("TTS_QUEUE_MAX", "8"))

# ---------------------------------------------------------------------------
# Speech recognition (faster-whisper)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
# Pin the language so short clips are not misdetected (e.g. as Hindi).
TRANSCRIBE_LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "en")

# ---------------------------------------------------------------------------
# Text-to-speech (Piper)
# ---------------------------------------------------------------------------
PIPER_VOICE_NAME = os.getenv("PIPER_VOICE_NAME", "en_US-amy-medium")
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))

# ---------------------------------------------------------------------------
# Audio capture pipeline
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Hz, mono
BYTES_PER_SAMPLE = 2         # 16-bit PCM
# webrtcvad only accepts 10/20/30 ms frames.
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = SAMPLE_RATE * VAD_FRAME_MS // 1000          # 480
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * BYTES_PER_SAMPLE           # 960

VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "3"))   # 0..3

# How much speech (in consecutive VAD frames) before we consider the
# candidate to have started an utterance.
ONSET_FRAMES = int(os.getenv("ONSET_FRAMES", "4"))               # ~120 ms

# Automatic barge-in (interrupting the interviewer by just speaking) adds
# an echo risk: with weak echo cancellation the interviewer's own voice
# can trigger it. Off by default; use the browser's Interrupt button, or
# enable with AUTO_BARGE_IN=1 (works well with headphones).
AUTO_BARGE_IN = os.getenv("AUTO_BARGE_IN", "0") == "1"
# Sustained speech required to barge in while the interviewer is speaking.
BARGE_IN_CONFIRM_MS = int(os.getenv("BARGE_IN_CONFIRM_MS", "350"))

# Silence that ends an utterance. Interview candidates pause to think,
# so this deliberately errs on the long side.
END_OF_UTTERANCE_SILENCE_MS = int(os.getenv("END_OF_UTTERANCE_SILENCE_MS", "1100"))
MIN_UTTERANCE_MS = int(os.getenv("MIN_UTTERANCE_MS", "400"))
MAX_UTTERANCE_MS = int(os.getenv("MAX_UTTERANCE_MS", "60000"))

# Audio kept right before speech onset so word beginnings are not clipped.
PRE_ROLL_MS = int(os.getenv("PRE_ROLL_MS", "270"))

# Interview server port (LM Studio commonly uses 1234).
HOST = os.getenv("INTERVIEWER_HOST", "127.0.0.1")
PORT = int(os.getenv("INTERVIEWER_PORT", "8100"))
