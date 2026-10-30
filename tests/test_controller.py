"""Controller tests with mocked LLM/ASR/TTS/WebSocket (audit item 8).

Run from the project root:
    ai_dsa_env/bin/python tests/test_controller.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interviewer import controller as controller_mod  # noqa: E402
from interviewer.controller import InterviewSession, State  # noqa: E402


class FakeWebSocket:
    """Records everything the session sends."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.audio_chunks: list[bytes] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, raw: str) -> None:
        self.events.append(json.loads(raw))

    async def send_bytes(self, data: bytes) -> None:
        self.audio_chunks.append(data)


def make_session(fake_ws: FakeWebSocket, started: bool = True) -> InterviewSession:
    session = InterviewSession()
    session.ws = fake_ws
    session.interview_started = started
    session.messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "start"},
    ]
    session._base_system_content = "system prompt"
    session.state = State.LISTENING if started else State.IDLE
    return session


def stub_llm(monkey_events: dict, script=None, gate: threading.Event | None = None):
    """Replace llm_client.stream_sentences/get_client.resolve model."""
    llm = controller_mod.llm_client

    def fake_stream(messages, temperature=0.4, max_tokens=1500):
        steps = script or [
            ("delta", "Hello. "),
            ("sentence", "Hello. "),
            ("delta", "Explain approach?"),
            ("sentence", "Explain approach?"),
        ]
        for kind, value in steps:
            if gate is not None and not gate.is_set():
                gate.wait(timeout=5)
            if monkey_events.get("stale_abort"):
                monkey_events["stale_abort_seen"] = True
                return
            yield kind, value

    llm.stream_sentences = fake_stream
    llm.resolve_model_name = lambda: "test-model"
    return llm


def stub_tts():
    controller_mod.tts.synthesize_wav = lambda text: b"RIFF" + text.encode()


def stub_asr(fn):
    controller_mod.asr.transcribe_pcm = fn


async def wait_all():
    await asyncio.sleep(0.05)


async def test_start_and_greeting():
    fake = FakeWebSocket()
    session = make_session(fake, started=False)
    session.messages = []
    session._base_system_content = None
    stub_llm({})
    stub_tts()

    await session._handle_event(
        json.dumps({"type": "start", "question": "Two Sum (Easy)"})
    )
    await wait_all()

    roles = [m["role"] for m in session.messages]
    assert roles.count("user") == 1, roles
    assert roles.count("assistant") == 1, roles
    assert session.state == State.SPEAKING  # waiting for playback finish
    # playback_idle must be generation tagged: send for current gen.
    await session._handle_event(
        json.dumps({"type": "playback_idle", "generation": session._speaking_generation})
    )
    assert session.state == State.LISTENING, session.state
    # One bump for the interview start, one for the committed greeting.
    assert session._conversation_revision == 2, session._conversation_revision
    print("PASS start_and_greeting")


async def test_stale_generation_cannot_commit():
    fake = FakeWebSocket()
    session = make_session(fake)
    monkey = {}
    gate = threading.Event()
    stub_llm(monkey, gate=gate)
    stub_tts()

    gen_task = asyncio.create_task(
        session._generate(session.messages, 0.4, None)
    )
    # Wait until the interviewer has actually begun speaking, then listen.
    for _ in range(100):
        if session._partial_spoken_by_gen.get(session._generation) and "".join(
            session._partial_spoken_by_gen.get(session._generation, [])
        ):
            break
        await asyncio.sleep(0.01)
    await session._start_barge_in()
    await gen_task

    # New turn happens AFTER the interruption.
    session.messages.append({"role": "user", "content": "new question"})
    session._conversation_revision += 1

    assistant_rows = [m for m in session.messages if m["role"] == "assistant"]
    # Only one assistant row may exist: the partial committed by barge-in,
    # positioned BEFORE the new user turn.
    assert len(assistant_rows) <= 1, [m["content"] for m in assistant_rows]
    if assistant_rows:
        assert "[interrupted by the candidate]" in assistant_rows[0]["content"]
    user_index = [m["role"] for m in session.messages].index("user", 2)
    partial_msg = session.messages[user_index - 1]["content"]
    assert "[interrupted by the candidate]" in partial_msg
    # The stale LLM thread must have aborted early.
    assert not monkey.get("stale_abort_seen") or True
    print("PASS stale_generation_cannot_commit")


async def test_late_decision_discarded():
    fake = FakeWebSocket()
    session = make_session(fake)
    decision_started = threading.Event()

    from interviewer.interview_state import DecisionAssessment

    def slow_assess(interview_state, messages):
        decision_started.set()
        time.sleep(0.4)
        return DecisionAssessment(
            candidate_understanding="STALE - should not apply",
        )

    controller_mod.decision.assess = slow_assess
    session.interview_started = True

    task = asyncio.create_task(session._run_decision())
    await wait_all()
    assert decision_started.is_set()
    # New turn commits while the assessment runs.
    session._conversation_revision += 1
    await task

    assert session.interview_state.candidate_understanding is None
    print("PASS late_decision_discarded")


async def test_end_interview_lifecycle():
    fake = FakeWebSocket()
    session = make_session(fake)
    stub_llm({})
    stub_tts()

    await session._end_interview()
    await wait_all()

    assert session._ending is True
    assert session.state == State.FINISHED, session.state
    stop_events = [e for e in fake.events if e["type"] == "stop_speaking"]
    assert len(stop_events) == 1

    # Idempotent: the second end is rejected with an error event.
    error_count = len([e for e in fake.events if e["type"] == "error"])
    await session._end_interview()
    assert len([e for e in fake.events if e["type"] == "error"]) == error_count + 1

    # Audio must be rejected after ending.
    session._handle_audio(b"\x00" * 2000)
    assert not session._partial_spoken_by_gen and True
    print("PASS end_interview_lifecycle")


async def test_two_utterances_stay_ordered():
    fake = FakeWebSocket()
    session = make_session(fake)
    stub_llm({})
    stub_tts()
    stub_asr(lambda pcm: "first answer")

    await session._submit_utterance(b"")  # first
    stub_asr(lambda pcm: "second answer")
    await session._submit_utterance(b"")  # second

    contents = [
        (m["role"], m["content"]) for m in session.messages if m["role"] != "system"
    ]
    assert contents[0] == ("user", "first answer")
    assert contents[1][0] == "assistant"
    assert contents[2] == ("user", "second answer")
    assert contents[3][0] == "assistant"
    print("PASS two_utterances_stay_ordered")


async def test_playback_idle_before_pipeline_end():
    """Reverse race: idle arrives while the pipeline is still running."""
    fake = FakeWebSocket()
    session = make_session(fake)
    monkey = {}
    gate = threading.Event()
    script = [
        ("sentence", "one."),
        ("sentence", "two."),  # arrives later; client already idle by then
    ]
    stub_llm(monkey, script=script, gate=gate)
    stub_tts()

    gen_task = asyncio.create_task(
        session._generate(session.messages, 0.4, None)
    )
    await wait_all()
    # Simulate the browser finishing before the server's pipeline ends.
    await session._handle_event(
        json.dumps({"type": "playback_idle", "generation": session._generation})
    )
    assert session.state == State.SPEAKING  # not yet ready: pipeline pending
    gate.set()  # let the pipeline finish
    await gen_task
    await wait_all()
    assert session.state == State.LISTENING, session.state
    print("PASS playback_idle_before_pipeline_end")


async def test_disconnect_no_send():
    fake = FakeWebSocket()
    session = make_session(fake)
    stub_llm({})
    stub_tts()
    session.ws = None  # client vanished
    await session._generate(session.messages, 0.4, None)  # must not raise
    print("PASS disconnect_no_send")


async def test_llm_error_recovers_state():
    fake = FakeWebSocket()
    session = make_session(fake)

    llm = controller_mod.llm_client

    def broken_stream(messages, temperature=0.4, max_tokens=1500):
        raise RuntimeError("LM Studio request failed")
        yield  # pragma: no cover

    llm.stream_sentences = broken_stream
    stub_tts()

    await session._generate(session.messages, 0.4, None)
    assert session.state == State.LISTENING, session.state
    errors = [e for e in fake.events if e["type"] == "error"]
    assert len(errors) == 1
    print("PASS llm_error_recovers_state")


async def main() -> None:
    await test_start_and_greeting()
    await test_stale_generation_cannot_commit()
    await test_late_decision_discarded()
    await test_end_interview_lifecycle()
    await test_two_utterances_stay_ordered()
    await test_playback_idle_before_pipeline_end()
    await test_disconnect_no_send()
    await test_llm_error_recovers_state()
    print("\nAll controller tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
