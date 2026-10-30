"""
Local DSA Voice Interviewer
---------------------------
A single-file Streamlit prototype using:
- LM Studio's OpenAI-compatible local API for interview reasoning
- Faster-Whisper for local speech-to-text
- pyttsx3 for offline system text-to-speech
- Streamlit for the UI

Run:
    pip install -r requirements.txt
    streamlit run app.py

Before running:
1. Start the LM Studio local server (usually http://localhost:1234/v1).
2. Load a model in LM Studio.
3. Set LM_STUDIO_MODEL to the model identifier shown by LM Studio, or enter it in the sidebar.

Notes:
- This prototype does NOT execute submitted code. Do not run candidate code directly inside the web app.
- The first Faster-Whisper use downloads its model unless you point it to a local model directory.
- pyttsx3 uses the operating system's installed speech engine and voices.
"""

from __future__ import annotations

import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI

# Optional dependencies are imported with helpful errors so the UI can explain setup issues.
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


st.set_page_config(
    page_title="Local DSA Voice Interviewer",
    page_icon="🎙️",
    layout="wide",
)

SYSTEM_PROMPT = """
You are a realistic, supportive but rigorous senior software engineer conducting a DSA interview.
The candidate is practicing problem-solving and communication.

Rules:
- Speak naturally, as a human interviewer would. Keep each spoken response concise (usually 1-4 sentences).
- Ask only one main question at a time.
- Do not reveal the complete solution unless the candidate explicitly asks to end the interview and see the solution.
- If the candidate asks for a hint, provide a progressive hint, not the full answer.
- Ask the candidate to explain their approach before coding when appropriate.
- Ask about edge cases and time/space complexity at appropriate moments.
- Be encouraging, but do not falsely praise incorrect reasoning.
- If code has been pasted, review it conceptually, but never claim it was executed or passed tests.
- Do not claim to have run code. This app currently has no code execution engine.
- Keep track of the current question and previous discussion from the supplied conversation.
- The candidate may speak answers or type them.
"""

QUESTION_BANK = {
    "Two Sum (Easy)": {
        "topic": "Hash map",
        "difficulty": "Easy",
        "statement": (
            "Given an integer array nums and an integer target, return the indices of two "
            "distinct elements whose values add up to target. Assume exactly one solution exists, "
            "and you may not use the same element twice. Explain your approach before coding."
        ),
    },
    "Valid Parentheses (Easy)": {
        "topic": "Stack",
        "difficulty": "Easy",
        "statement": (
            "Given a string containing only (), {}, and [], determine whether the brackets are "
            "valid and correctly nested. Explain your approach and relevant edge cases."
        ),
    },
    "Longest Substring Without Repeating Characters (Medium)": {
        "topic": "Sliding window",
        "difficulty": "Medium",
        "statement": (
            "Given a string s, return the length of the longest substring without repeating "
            "characters. Discuss a brute-force approach and then aim for an efficient solution."
        ),
    },
    "Binary Search (Easy)": {
        "topic": "Binary search",
        "difficulty": "Easy",
        "statement": (
            "Given a sorted array of distinct integers and a target, return its index if present, "
            "or -1 otherwise. Aim for O(log n) time and explain why your search range is correct."
        ),
    },
    "Number of Islands (Medium)": {
        "topic": "Graph traversal",
        "difficulty": "Medium",
        "statement": (
            "Given a grid of '1' (land) and '0' (water), count the islands. Land connects "
            "horizontally and vertically. Explain your traversal strategy and complexity."
        ),
    },
}


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key or "lm-studio",
        timeout=120.0,
    )


def get_model_response(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=600,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str, device: str, compute_type: str):
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        )
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(
    audio_bytes: bytes,
    model_size: str,
    device: str,
    compute_type: str,
) -> str:
    model = load_whisper_model(model_size, device, compute_type)
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(audio_bytes)
        audio_path = temp_file.name

    try:
        segments, _info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def synthesize_speech(text: str) -> bytes | None:
    """Generate speech with the OS's local speech engine, returning WAV bytes."""
    if pyttsx3 is None:
        return None

    temp_path = None
    engine = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name

        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        engine.stop()

        # Some platform drivers finish writing just after runAndWait returns.
        for _ in range(20):
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 44:
                break
            time.sleep(0.1)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) <= 44:
            return None

        return Path(temp_path).read_bytes()
    except Exception as exc:
        st.warning(
            f"Local text-to-speech was unavailable ({exc}). The response is still shown as text."
        )
        return None
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def reset_session() -> None:
    for key in (
        "messages",
        "selected_question",
        "interview_started",
        "started_at",
        "last_audio",
        "last_transcript",
    ):
        st.session_state.pop(key, None)


with st.sidebar:
    st.header("⚙️ Local configuration")
    base_url = st.text_input("LM Studio API base URL", "http://localhost:1234/v1")
    model_name = st.text_input(
        "Model identifier",
        value=os.getenv("LM_STUDIO_MODEL", "local-model"),
        help="Use the model identifier displayed by LM Studio's local server.",
    )
    api_key = st.text_input(
        "API key (usually not required locally)",
        value=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        type="password",
    )
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.4, 0.05)

    st.divider()
    st.subheader("Speech recognition")
    whisper_size = st.selectbox(
        "Whisper model",
        ["base", "small", "medium", "large-v3"],
        index=1,
        help="Start with base or small for a faster first run. The model may download on first use.",
    )
    whisper_device = st.selectbox("Whisper device", ["cpu", "auto"], index=0)
    whisper_compute = st.selectbox(
        "Compute type",
        ["int8", "float16", "int8_float16"],
        index=0,
    )
    st.caption(
        "For a simple cross-platform start, use CPU + int8. Benchmark other settings on your Mac."
    )

    st.divider()
    if st.button("🧹 Reset interview", use_container_width=True):
        reset_session()
        st.rerun()


st.title("🎙️ Local DSA Voice Interviewer")
st.caption("A local-first mock interview powered by LM Studio, Faster-Whisper, and offline system TTS.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

left, right = st.columns([1.2, 1])

with left:
    st.subheader("1. Interview setup")
    selected_question = st.selectbox("Choose a DSA problem", list(QUESTION_BANK.keys()))
    duration = st.select_slider("Interview length", options=[15, 20, 30, 45, 60], value=30)
    candidate_name = st.text_input("Name (optional)", placeholder="Candidate")
    question = QUESTION_BANK[selected_question]

    st.info(f"**Topic:** {question['topic']}  \n**Difficulty:** {question['difficulty']}")
    with st.expander("Problem statement", expanded=True):
        st.write(question["statement"])

    if st.button("🚀 Start interview", type="primary", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + f"\nInterview duration: {duration} minutes."
                    + f"\nCandidate name: {candidate_name or 'Candidate'}."
                    + f"\nCurrent problem: {selected_question}."
                    + f"\nProblem statement: {question['statement']}"
                ),
            }
        ]
        st.session_state.selected_question = selected_question
        st.session_state.interview_started = True
        st.session_state.started_at = time.time()
        st.session_state.last_transcript = ""
        st.session_state.last_audio = None

        try:
            client = make_client(base_url, api_key)
            greeting = get_model_response(
                client,
                model_name,
                st.session_state.messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Start the interview. Briefly greet the candidate, introduce the "
                            "problem without solving it, and ask them to explain their initial approach."
                        ),
                    }
                ],
                temperature,
            )
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            st.session_state.last_audio = synthesize_speech(greeting)
            st.rerun()
        except Exception as exc:
            st.error(
                "Could not contact LM Studio. Confirm the local server is running, a model is "
                f"loaded, and the model identifier is correct.\n\nDetails: {exc}"
            )

with right:
    st.subheader("2. Interview conversation")

    if not st.session_state.interview_started:
        st.write("Start an interview to enable voice and text responses.")
    else:
        elapsed = int(time.time() - st.session_state.get("started_at", time.time()))
        remaining = max(0, duration * 60 - elapsed)
        st.metric("Time remaining", f"{remaining // 60:02d}:{remaining % 60:02d}")

        visible_messages = [
            msg for msg in st.session_state.messages if msg["role"] != "system"
        ]
        for msg in visible_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.get("last_audio"):
            st.audio(st.session_state.last_audio, format="audio/wav")

        st.markdown("#### Answer by voice")
        st.caption("Record your answer, then submit it for local transcription.")
        audio_input = st.audio_input("Record an answer", key="candidate_audio")

        voice_col1, voice_col2 = st.columns(2)
        with voice_col1:
            transcribe_clicked = st.button(
                "📝 Transcribe recording",
                disabled=audio_input is None,
                use_container_width=True,
            )
        with voice_col2:
            send_voice_clicked = st.button(
                "🎙️ Transcribe and send",
                disabled=audio_input is None,
                type="primary",
                use_container_width=True,
            )

        if transcribe_clicked or send_voice_clicked:
            try:
                with st.spinner("Transcribing locally..."):
                    transcript = transcribe_audio(
                        audio_input.getvalue(),
                        whisper_size,
                        whisper_device,
                        whisper_compute,
                    )
                st.session_state.last_transcript = transcript
                if transcript:
                    st.success("Transcription complete.")
                    st.text_area(
                        "Recognized speech (you can edit it below)",
                        value=transcript,
                        key="voice_transcript_edit",
                    )
                else:
                    st.warning("No speech was detected. Try recording again.")
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")

            if send_voice_clicked:
                transcript_to_send = st.session_state.get("last_transcript", "").strip()
                if transcript_to_send:
                    st.session_state.pending_candidate_message = transcript_to_send
                    st.rerun()

        if "voice_transcript_edit" in st.session_state:
            edited_transcript = st.text_area(
                "Edit transcription before sending",
                key="voice_transcript_final",
                value=st.session_state.get("voice_transcript_edit", ""),
            )
            if st.button("Send edited voice answer", use_container_width=True):
                if edited_transcript.strip():
                    st.session_state.pending_candidate_message = edited_transcript.strip()
                    st.rerun()

        typed_answer = st.text_area(
            "Or type an answer / paste code",
            key="typed_candidate_answer",
            height=120,
            placeholder="Explain your approach, ask for a hint, or paste your code here...",
        )

        send_typed = st.button("Send text answer", use_container_width=True)
        end_interview = st.button("End interview and get feedback", use_container_width=True)

        pending_message = st.session_state.pop("pending_candidate_message", None)
        if send_typed and typed_answer.strip():
            pending_message = typed_answer.strip()

        if pending_message and st.session_state.interview_started:
            st.session_state.messages.append(
                {"role": "user", "content": pending_message}
            )
            try:
                with st.spinner("Thinking locally..."):
                    client = make_client(base_url, api_key)
                    response_text = get_model_response(
                        client,
                        model_name,
                        st.session_state.messages,
                        temperature,
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
                st.session_state.last_audio = synthesize_speech(response_text)
                st.rerun()
            except Exception as exc:
                st.error(f"LM Studio request failed: {exc}")

        if end_interview:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "The interview is over. Give a concise but useful evaluation based only "
                        "on the conversation. Score problem understanding, algorithm selection, "
                        "correctness evidence, complexity analysis, edge-case awareness, and "
                        "communication from 1 to 5 each. Clearly state that code was not executed "
                        "by this prototype. Give three specific next steps."
                    ),
                }
            )
            try:
                with st.spinner("Preparing your feedback..."):
                    client = make_client(base_url, api_key)
                    feedback = get_model_response(
                        client,
                        model_name,
                        st.session_state.messages,
                        temperature=0.2,
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": feedback}
                )
                st.session_state.last_audio = synthesize_speech(feedback)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not generate final feedback: {exc}")

st.divider()
st.caption(
    "Privacy: audio is processed by local components in this app. Faster-Whisper may download "
    "model files on first use. LM Studio must be running locally. No submitted code is executed."
)
