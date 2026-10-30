"""Conversation controller: the interview state machine.

States
------
IDLE        no interview yet
LISTENING   microphone is live, waiting for speech (VAD tracks speech)
PROCESSING  speech ended, faster-whisper is transcribing
THINKING    LM Studio is generating the reply
SPEAKING    the interviewer is speaking (Piper sentences are streaming)
FINISHING   the ending workflow runs (feedback generation; audio rejected)
FINISHED    the interview has ended

Interruption is an EVENT, not a stored state: barge-in invalidates the
current generation, commits any already-spoken partial reply, stops client
playback, and returns to LISTENING while the candidate's new utterance is
captured.

Transitions
-----------
IDLE -> (start event) -> THINKING -> SPEAKING -> LISTENING -> ...
LISTENING --(end of utterance)--> PROCESSING -> THINKING -> SPEAKING -> LISTENING
SPEAKING/THINKING --(candidate speech detected: barge-in)--> PROCESSING -> ...
(end event) -> ... -> FINISHED
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as queue_mod
import threading
import time
from collections import deque
from enum import Enum

import webrtcvad
from fastapi import WebSocket, WebSocketDisconnect

from interviewer import asr, config, decision, llm_client, questions, tts
from interviewer.interview_state import InterviewState

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FINISHING = "finishing"
    FINISHED = "finished"


class InterviewSession:
    """Owns one connected browser client and its interview."""

    def __init__(self) -> None:
        self.ws: WebSocket | None = None
        self.state = State.IDLE
        self.interview_started = False
        self.messages: list[dict] = []

        # Generation counter: bumped to invalidate an in-flight LLM/TTS
        # pipeline (barge-in, end, reset). Only the CURRENT generation may
        # commit replies to the active conversation history.
        self._generation = 0
        # Partial speech per generation: a barge-in commits what was already
        # spoken BEFORE invalidating the generation (keeps history ordered).
        self._partial_spoken_by_gen: dict[int, list[str]] = {}
        # Monotonic revision of the committed conversation. The decision
        # layer compares this before applying results (staleness guard).
        self._conversation_revision = 0

        # Playback completion bookkeeping. Leaving SPEAKING requires BOTH
        # events, in any order: the pipeline finished sending audio
        # (_awaiting_playback) and the browser finished playing it
        # (_client_playback_idle). Idle events are generation-tagged to
        # reject stale reports.
        self._awaiting_playback = False
        self._speaking_generation: int | None = None
        self._client_playback_idle = False

        # Interview ending lifecycle: idempotent; incoming audio is
        # rejected while the final evaluation runs.
        self._ending = False

        # Phase 4: structured interview state + decision layer bookkeeping.
        self.interview_state: InterviewState | None = None
        self._base_system_content: str | None = None
        self._decision_busy = False

        self._submit_lock = asyncio.Lock()

        # --- continuous audio capture / VAD -----------------------------
        self._vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self._chunk_buffer = bytearray()          # bytes not yet framed
        self._pre_roll: deque[bytes] = deque(
            maxlen=max(1, config.PRE_ROLL_MS // config.VAD_FRAME_MS)
        )
        self._speech_run = 0        # consecutive speech frames (onset)
        self._silence_run = 0       # consecutive silence frames (offset)
        self._candidate_speaking = False
        self._utterance = bytearray()

        # True while the pipeline finished but the browser is still playing
        # the queued sentences. The server stays in SPEAKING until the
        # browser reports playback_idle.
        self._awaiting_playback = False
        self._speaking_generation = None

        # Latency instrumentation
        self._utterance_end_ts: float | None = None

    # ------------------------------------------------------------------
    # WebSocket plumbing
    # ------------------------------------------------------------------
    async def run(self, websocket: WebSocket) -> None:
        self.ws = websocket
        await websocket.accept()
        await self._send("state", state=self.state.value)
        logger.info("Client connected")
        try:
            while True:
                message = await websocket.receive()
                text = message.get("text")
                data = message.get("bytes")
                if message.get("type") == "websocket.disconnect":
                    break
                if text is not None:
                    await self._handle_event(text)
                elif data is not None:
                    self._handle_audio(data)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self._generation += 1  # invalidate pipelines
            self.ws = None
            logger.info("Client disconnected")

    async def _send(self, type_: str, **payload) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_text(json.dumps({"type": type_, **payload}))
        except Exception:  # pragma: no cover - client vanished
            pass

    async def _send_audio(self, wav_bytes: bytes) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_bytes(wav_bytes)
        except Exception:  # pragma: no cover - client vanished
            pass

    # ------------------------------------------------------------------
    # Client events
    # ------------------------------------------------------------------
    def _commit_partial(self) -> None:
        """Commit what the CURRENT generation has spoken so far.

        Must run while the generation is still valid and immediately
        before invalidating it, so the partial reply keeps the correct
        position in history (before the interruption-causing user turn).
        """
        key = self._generation
        parts = self._partial_spoken_by_gen.pop(key, None)
        if not parts:
            return
        text = "".join(parts).strip()
        if text:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": text.rstrip() + "\n[interrupted by the candidate]",
                }
            )
            self._conversation_revision += 1

    def _bump_generation(self, commit_partial: bool = True) -> None:
        """Invalidate the current generation (and any queued client work)."""
        if commit_partial:
            self._commit_partial()
        self._generation += 1
        self._awaiting_playback = False
        self._client_playback_idle = False
        self._speaking_generation = None

    async def _finish_playback_if_ready(self) -> None:
        """Return to LISTENING only when both conditions hold: the pipeline
        finished sending audio AND the browser finished playing it."""
        if not (self._awaiting_playback and self._client_playback_idle):
            return
        self._awaiting_playback = False
        self._client_playback_idle = False
        self._speaking_generation = None
        self._pre_roll.clear()
        self._speech_run = 0
        if self.state == State.SPEAKING:
            self.state = State.LISTENING
            await self._send("state", state=self.state.value)

    async def _handle_event(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return

        event_type = event.get("type")

        if event_type == "start":
            await self._start_interview(event)

        elif event_type == "end":
            await self._end_interview()

        elif event_type == "barge_in":
            # Manual interruption from the browser button.
            if self._ending:
                return
            if self.state in (State.SPEAKING, State.THINKING):
                await self._start_barge_in()

        elif event_type == "playback_idle":
            # All queued interviewer audio finished playing. Must carry the
            # generation it belongs to so stale reports are rejected.
            if (
                self._speaking_generation is not None
                and event.get("generation") == self._speaking_generation
            ):
                self._client_playback_idle = True
                client_first_audio_ms = event.get("client_first_audio_ms")
                if client_first_audio_ms is not None:
                    logger.info(
                        "Client audible latency: %s ms (server-side %s boundary)",
                        client_first_audio_ms,
                        "per-turn",
                    )
                await self._finish_playback_if_ready()

        # "reset" is handled client-side by reloading the page.

    async def _start_interview(self, event: dict) -> None:
        if self.interview_started:
            return

        question_id = event.get("question") or next(iter(questions.QUESTION_BANK))
        if question_id not in questions.QUESTION_BANK:
            question_id = next(iter(questions.QUESTION_BANK))
        name = str(event.get("name") or "Candidate").strip() or "Candidate"
        duration = int(event.get("duration") or 30)

        try:
            model = llm_client.resolve_model_name()
            logger.info("Using LM Studio model: %s", model)
        except Exception as exc:
            await self._send(
                "error",
                message=(
                    "Could not contact LM Studio. Confirm the server is "
                    f"running and a model is loaded.\n\n{llm_client.describe_api_error(exc)}"
                ),
            )
            return

        self.interview_started = True
        system_message = questions.build_system_message(question_id, duration, name)
        self.messages = [
            system_message,
            {"role": "user", "content": questions.GREETING_REQUEST},
        ]
        self._base_system_content = system_message["content"]
        self.interview_state = InterviewState(
            problem_id=question_id,
            topic=questions.QUESTION_BANK[question_id]["topic"],
            difficulty=questions.QUESTION_BANK[question_id]["difficulty"],
            duration_minutes=duration,
        )
        self.refresh_system_context()
        self._conversation_revision += 1
        await self._send(
            "interview_started",
            question=question_id,
            model=model,
        )
        await self._generate(
            self.messages,
            temperature=config.LLM_TEMPERATURE,
            end_of_utterance_ts=None,
        )

    async def _start_barge_in(self) -> None:
        """Stop the interviewer mid-speech and return to listening.

        The already-spoken partial reply is committed to history first
        (while its generation is still current), then the generation is
        invalidated, client playback is stopped, and the state moves to
        LISTENING so the new utterance can be captured.
        """
        if self._ending:
            return
        self._bump_generation(commit_partial=True)
        self._pre_roll.clear()
        self._speech_run = 0
        await self._send("stop_speaking")
        self.state = State.LISTENING
        await self._send("state", state=self.state.value)
        logger.info("Interruption accepted")

    def refresh_system_context(self) -> None:
        """Attach the live interview tracker to the system prompt.

        The system message must stay first (Qwen's template rejects system
        messages mid-conversation), so the tracker is baked into message[0]
        instead of injected as extra rows before each turn.
        """
        if not self.messages or self._base_system_content is None:
            return
        if self.interview_state is None:
            return
        directive = self.interview_state.directive_text()
        self.messages[0]["content"] = (
            self._base_system_content
            + "\n\nCurrent interview state (updated automatically; "
            "use it to steer your next turn):\n"
            + directive
        )

    async def _end_interview(self) -> None:
        """End the interview as a controlled lifecycle transition.

        Idempotent: a second end event is rejected. Audio processing is
        disabled for the rest of the session so nothing can interrupt the
        final evaluation.
        """
        if not self.interview_started or self._ending:
            await self._send(
                "error",
                message=(
                    "No active interview is available to end."
                    if self._ending
                    else "No interview is running."
                ),
            )
            return

        self._ending = True
        # Invalidate the active response (committing any partial reply)
        # before requesting feedback.
        self._bump_generation(commit_partial=True)
        await self._send("stop_speaking")

        self.state = State.FINISHING
        await self._send("state", state=self.state.value)

        self.messages.append(
            {
                "role": "user",
                "content": (
                    questions.FEEDBACK_REQUEST
                    + "\n\nInterview tracker (grounded only in this conversation):\n"
                    + (self.interview_state.summary_text() if self.interview_state else "")
                ),
            }
        )
        self._conversation_revision += 1
        await self._generate(
            self.messages,
            temperature=0.2,
            end_of_utterance_ts=None,
            finished=True,
        )

    # ------------------------------------------------------------------
    # Continuous audio capture + VAD
    # ------------------------------------------------------------------
    def _handle_audio(self, pcm: bytes) -> None:
        if (
            not self.interview_started
            or self._ending
            or self.state == State.FINISHED
        ):
            return

        self._chunk_buffer.extend(pcm)
        while len(self._chunk_buffer) >= config.VAD_FRAME_BYTES:
            frame = bytes(self._chunk_buffer[: config.VAD_FRAME_BYTES])
            del self._chunk_buffer[: config.VAD_FRAME_BYTES]
            self._vad_step(frame)

    def _vad_step(self, frame: bytes) -> None:
        # While the interviewer is speaking, mic capture is paused unless
        # auto barge-in is explicitly enabled. This prevents the
        # interviewer's own voice (through weak echo cancellation) from
        # being captured as candidate speech.
        if self.state == State.SPEAKING and not config.AUTO_BARGE_IN:
            return

        try:
            is_speech = self._vad.is_speech(frame, config.SAMPLE_RATE)
        except Exception:
            return

        if not self._candidate_speaking:
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0

            onset_needed = config.ONSET_FRAMES
            if self.state == State.SPEAKING and config.AUTO_BARGE_IN:
                onset_needed = max(
                    config.ONSET_FRAMES,
                    config.BARGE_IN_CONFIRM_MS // config.VAD_FRAME_MS,
                )

            if self._speech_run >= onset_needed:
                self._candidate_speaking = True
                self._speech_run = 0
                self._silence_run = 0
                self._utterance = bytearray(b"".join(self._pre_roll))
                self._pre_roll.clear()

                # Barge-in: the candidate started speaking while the
                # interviewer was speaking or thinking. Stop playback and
                # cancel the in-flight generation; the utterance being
                # captured now will be transcribed and answered. While
                # THINKING there is no audio playing, so echo is not a
                # risk and this is always allowed.
                if config.AUTO_BARGE_IN or self.state == State.THINKING:
                    asyncio.create_task(self._start_barge_in())
                    asyncio.create_task(self._send("interrupted"))
                    logger.info("Barge-in detected")
        else:
            self._utterance.extend(frame)
            if is_speech:
                self._silence_run = 0
            else:
                self._silence_run += 1

            duration_ms = len(self._utterance) // config.BYTES_PER_SAMPLE * 1000 // config.SAMPLE_RATE
            silence_ms = self._silence_run * config.VAD_FRAME_MS

            if (
                silence_ms >= config.END_OF_UTTERANCE_SILENCE_MS
                or duration_ms >= config.MAX_UTTERANCE_MS
            ):
                self._end_utterance()

    def _end_utterance(self) -> None:
        trim_frames = max(0, self._silence_run - config.PRE_ROLL_MS // config.VAD_FRAME_MS)
        if trim_frames:
            excess = trim_frames * config.VAD_FRAME_BYTES
            if excess < len(self._utterance):
                del self._utterance[len(self._utterance) - excess:]

        pcm = bytes(self._utterance)
        self._utterance = bytearray()
        self._candidate_speaking = False
        self._silence_run = 0
        self._pre_roll.clear()

        duration_ms = len(pcm) // config.BYTES_PER_SAMPLE * 1000 // config.SAMPLE_RATE
        if duration_ms < config.MIN_UTTERANCE_MS:
            logger.debug("Dropped short utterance (%s ms)", duration_ms)
            return

        self._utterance_end_ts = time.perf_counter()
        asyncio.create_task(self._submit_utterance(pcm))

    # ------------------------------------------------------------------
    # Utterance -> transcript -> LLM -> TTS -> client
    # ------------------------------------------------------------------
    async def _submit_utterance(self, pcm: bytes) -> None:
        async with self._submit_lock:
            self.state = State.PROCESSING
            await self._send("state", state=self.state.value)

            try:
                transcript = await asyncio.to_thread(asr.transcribe_pcm, pcm)
            except Exception as exc:
                logger.exception("Transcription failed")
                await self._send("error", message=f"Transcription failed: {exc}")
                if self.interview_started:
                    self.state = State.LISTENING
                    await self._send("state", state=self.state.value)
                return

            if not transcript:
                logger.info("No speech recognized; returning to listening")
                self.state = State.LISTENING
                await self._send("state", state=self.state.value)
                return

            await self._send("transcript", text=transcript)
            logger.info("Transcript: %s", transcript)

            if self.interview_state is not None and "```" in transcript:
                self.interview_state.note_code_submitted()
                self.refresh_system_context()

            self.messages.append({"role": "user", "content": transcript})
            self._conversation_revision += 1
            await self._generate(
                self.messages,
                temperature=config.LLM_TEMPERATURE,
                end_of_utterance_ts=self._utterance_end_ts,
            )

    async def _generate(
        self,
        messages: list[dict],
        temperature: float,
        end_of_utterance_ts: float | None,
        finished: bool = False,
    ) -> None:
        """Stream an LLM reply, speak it sentence by sentence.

        Two concurrent producers (Phase 2/6 audit fix): an LLM reader thread
        pushes completed sentences into a bounded queue, and a TTS thread
        synthesizes them while the LLM keeps generating. Only the current
        generation may touch conversation history or send audio.
        """
        generation = self._generation
        self.state = State.THINKING
        await self._send("state", state=self.state.value)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        # Bounded queue between the LLM reader and the TTS worker.
        sentence_queue: "queue_mod.Queue" = queue_mod.Queue(
            maxsize=config.TTS_QUEUE_MAX
        )

        spoken: list[str] = []
        self._partial_spoken_by_gen[generation] = spoken
        first_delta_ts: float | None = None
        first_audio_ts: float | None = None

        def llm_worker() -> None:
            """Reads the LLM stream; deltas go straight out, sentences
            are queued for the TTS thread (backpressure via maxsize)."""
            try:
                stream = llm_client.stream_sentences(
                    messages,
                    temperature=temperature,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
                for kind, value in stream:
                    if self._generation != generation:
                        return  # barge-in / reset: stop reading the stream
                    if kind == "delta":
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("delta", value, None)
                        )
                    else:  # sentence -> TTS thread
                        sentence_queue.put((generation, value))
            except Exception as exc:
                message = str(exc) if isinstance(exc, RuntimeError) else (
                    llm_client.describe_api_error(exc)
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", message, None)
                )
            finally:
                sentence_queue.put(None)  # close the TTS queue

        def tts_worker() -> None:
            """Synthesizes queued sentences while the LLM keeps generating.

            This thread always finishes last, so it owns the relay queue's
            "done" sentinel."""
            try:
                while True:
                    item = sentence_queue.get()
                    if item is None:
                        return  # LLM stream finished
                    gen_id, text = item
                    if self._generation != gen_id:
                        return  # stale generation: stop synthesizing
                    try:
                        wav = tts.synthesize_wav(text)
                    except Exception as exc:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("error", f"Speech synthesis failed: {exc}", None),
                        )
                        return
                    if self._generation != gen_id:
                        return
                    loop.call_soon_threadsafe(
                        queue.put_nowait, ("audio", text, wav)
                    )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None, None))

        threading.Thread(target=llm_worker, daemon=True).start()
        threading.Thread(target=tts_worker, daemon=True).start()

        error_message: str | None = None
        while True:
            kind, value, wav = await queue.get()

            if self._generation != generation:
                break  # stale generation: drop the rest silently

            if kind == "delta":
                if first_delta_ts is None:
                    first_delta_ts = time.perf_counter()
                spoken.append(value)
                await self._send("assistant_delta", text=value)

            elif kind == "audio":
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter()
                    self.state = State.SPEAKING
                    self._client_playback_idle = False
                    await self._send("turn_audio", generation=generation)
                    await self._send("state", state=self.state.value)
                await self._send_audio(wav)

            elif kind == "error":
                error_message = value
                break

            elif kind == "done":
                break

        # Commit bookkeeping. Only the CURRENT generation may modify
        # active history; partial replies from interrupted turns were
        # committed by the barge-in handler before invalidating the
        # generation, so anything left here is stale and discarded.
        reply_text = "".join(spoken).strip()
        self._partial_spoken_by_gen.pop(generation, None)
        if reply_text and self._generation == generation:
            messages.append({"role": "assistant", "content": reply_text})
            self._conversation_revision += 1

        if first_audio_ts is not None and end_of_utterance_ts is not None:
            await self._send(
                "metrics",
                llm_first_ms=(
                    int((first_delta_ts - end_of_utterance_ts) * 1000)
                    if first_delta_ts is not None
                    else None
                ),
                first_audio_ms=int((first_audio_ts - end_of_utterance_ts) * 1000),
            )

        if error_message:
            await self._send("error", message=error_message)

        if finished:
            self.state = State.FINISHED
        elif self._generation != generation:
            # Interrupted: the barge-in path already moved to LISTENING and
            # the new utterance gets its own pipeline.
            self._awaiting_playback = False
            self._client_playback_idle = False
            self._speaking_generation = None
            self.state = State.LISTENING
        elif first_audio_ts is not None:
            # Hold SPEAKING until the browser reports that all queued
            # sentences actually finished playing, otherwise the mic
            # would listen to the interviewer's own trailing audio.
            self._awaiting_playback = True
            self._speaking_generation = generation
            await self._finish_playback_if_ready()
        else:
            # Nothing was spoken (empty/error): return to listening.
            self.state = State.LISTENING
        await self._send("state", state=self.state.value)

        # Decision layer: assess the turn in the background so it never
        # adds latency to the voice loop; results shape the next turn.
        if (
            not finished
            and not self._ending
            and error_message is None
            and reply_text
            and self._generation == generation
        ):
            asyncio.create_task(self._run_decision())

    async def _run_decision(self) -> None:
        if (
            self._decision_busy
            or self._ending
            or self.interview_state is None
            or self.ws is None
        ):
            return
        self._decision_busy = True
        revision = self._conversation_revision
        started_ts = time.perf_counter()
        try:
            snapshot = list(self.messages)
            assessment = await asyncio.to_thread(decision.assess, self.interview_state, snapshot)
            elapsed_ms = int((time.perf_counter() - started_ts) * 1000)
            if assessment is not None and not self._ending and self.interview_started:
                if revision != self._conversation_revision:
                    logger.info(
                        "Discarded stale decision assessment "
                        "(snapshot revision %s, current %s)",
                        revision,
                        self._conversation_revision,
                    )
                else:
                    self.interview_state.apply_decision(assessment)
                    self.refresh_system_context()
                    await self._send("interview_update", **self.interview_state.to_ui())
                    logger.info(
                        "Decision layer (%d ms): action=%s hints=%s issues=%s",
                        elapsed_ms,
                        assessment.interviewer_action,
                        self.interview_state.hints_given,
                        self.interview_state.identified_issues,
                    )
        except Exception:
            logger.exception("Decision layer failed")
        finally:
            self._decision_busy = False
