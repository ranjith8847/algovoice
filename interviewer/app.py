"""FastAPI application for the real-time voice interviewer.

Run from the project root:
    python -m interviewer.app
Then open http://127.0.0.1:8100 in your browser.
"""

from __future__ import annotations

import logging
import threading

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from interviewer import config, controller, questions as questions_bank, tts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local DSA Voice Interviewer (real-time)")

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/questions")
def get_questions() -> dict:
    return questions_bank.QUESTION_BANK


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.websocket("/ws")
async def interview_socket(websocket: WebSocket) -> None:
    session = controller.InterviewSession()
    await session.run(websocket)


@app.on_event("startup")
def preload_models() -> None:
    """Warm up Piper and Whisper in background threads.

    First use downloads the Piper voice (~60 MB) and possibly the Whisper
    model, so preloading avoids a long stall on the first utterance.
    """

    def warm() -> None:
        try:
            tts.get_voice()
            logger.info("Piper voice warmed up")
        except Exception as exc:
            logger.warning("Piper warm-up failed: %s", exc)

    def warm_piper_in_thread() -> None:
        threading.Thread(target=warm, daemon=True).start()

    warm_piper_in_thread()
    # Whisper is loaded lazily on first transcription (it is the heaviest
    # dependency and the browser mic takes a moment anyway).


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
