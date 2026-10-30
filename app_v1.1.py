from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import streamlit as st
from openai import OpenAI

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


st.set_page_config(
    page_title="Local DSA Voice Interviewer",
    page_icon="🎙️",
    layout="wide",
)

SYSTEM_PROMPT = """
You are a realistic, supportive but rigorous senior software engineer
conducting a data structures and algorithms interview.

Rules:
- Keep spoken responses concise and natural.
- Ask one main question at a time.
- Do not reveal the complete solution unless asked for final feedback.
- Give progressive hints when requested.
- Ask the candidate to explain their approach before coding.
- Discuss edge cases and time and space complexity.
- Be supportive but do not falsely praise incorrect answers.
- Review pasted code conceptually. Never claim it was executed.
- Keep track of the conversation and current problem.
- The candidate may type answers or speak them.
"""

QUESTION_BANK = {
    "Two Sum (Easy)": {
        "topic": "Hash map",
        "difficulty": "Easy",
        "statement": (
            "Given an integer array nums and an integer target, return "
            "the indices of two distinct elements whose values add up "
            "to target. Assume exactly one solution exists, and you "
            "may not use the same element twice. Explain your approach "
            "before coding."
        ),
    },
    "Valid Parentheses (Easy)": {
        "topic": "Stack",
        "difficulty": "Easy",
        "statement": (
            "Given a string containing (), {}, and [], determine "
            "whether the brackets are valid and correctly nested. "
            "Explain your approach and edge cases."
        ),
    },
    "Longest Substring Without Repeating Characters (Medium)": {
        "topic": "Sliding window",
        "difficulty": "Medium",
        "statement": (
            "Given a string s, return the length of the longest "
            "substring without repeating characters. Discuss a "
            "brute-force approach and then an efficient solution."
        ),
    },
    "Binary Search (Easy)": {
        "topic": "Binary search",
        "difficulty": "Easy",
        "statement": (
            "Given a sorted array of distinct integers and a target, "
            "return its index if present, or -1 otherwise. Aim for "
            "O(log n) time and explain your search range."
        ),
    },
    "Number of Islands (Medium)": {
        "topic": "Graph traversal",
        "difficulty": "Medium",
        "statement": (
            "Given a grid of '1' (land) and '0' (water), count the "
            "islands. Land connects horizontally and vertically. "
            "Explain your traversal strategy and complexity."
        ),
    },
}


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key.strip() or "lm-studio",
        timeout=120.0,
        max_retries=0,
    )


def get_model_response(client, model, messages, temperature, max_tokens=1500):
    # Normalize messages and ensure a user query exists.
    normalized = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in ("system", "user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        normalized.append({
            "role": role,
            "content": content.strip(),
        })

    # Some model prompt templates reject requests without a user message.
    if not any(m["role"] == "user" for m in normalized):
        normalized.append({
            "role": "user",
            "content": "Please begin the DSA interview.",
        })

    # Some local model chat templates (e.g. Qwen 3.x served by LM Studio)
    # reject a history whose first non-system message is an assistant turn,
    # returning HTTP 400: "No user query found in messages."
    first_non_system = 0
    if normalized and normalized[0]["role"] == "system":
        first_non_system = 1
    if first_non_system < len(normalized) and normalized[first_non_system]["role"] != "user":
        normalized.insert(
            first_non_system,
            {"role": "user", "content": "Please continue the interview."},
        )

    response = client.chat.completions.create(
        model=model.strip(),
        messages=normalized,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    if isinstance(content, str) and content.strip():
        return content.strip()

    if content:
        return str(content).strip()

    raise RuntimeError(
        "The model returned an empty response. With reasoning ('thinking') "
        "models such as Qwen 3.x, the model can spend the entire token budget "
        "on reasoning. "
        "Increase the response token limit in the sidebar or disable "
        "thinking for this model in LM Studio, then send the answer again."
    )



@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size, device, compute_type):
    if WhisperModel is None:
        raise RuntimeError(
            "Faster-Whisper is not installed. Run: "
            "pip install faster-whisper"
        )

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )


def transcribe_audio(audio_bytes, model_size, device, compute_type, language="en"):
    model = load_whisper_model(
        model_size,
        device,
        compute_type,
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as f:
            f.write(audio_bytes)
            temp_path = f.name

        segments, _ = model.transcribe(
            temp_path,
            beam_size=5,
            vad_filter=True,
            # Whisper's auto-detect often misclassifies short English clips
            # as Hindi and outputs transliteration instead of translation.
            # Pinning the language fixes that; None means auto-detect.
            language=None if language == "auto" else language,
        )

        return " ".join(
            segment.text.strip() for segment in segments
        ).strip()

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def synthesize_speech(text):
    """
    Generate a WAV file using macOS `say` and `afconvert`.
    Return None if audio generation fails.
    """

    text = (text or "").strip()

    if not text:
        return None

    say_path = shutil.which("say")
    convert_path = shutil.which("afconvert")

    if not say_path or not convert_path:
        st.warning(
            "macOS speech commands were not found. "
            "The response is available as text, but audio "
            "could not be generated."
        )
        return None

    aiff_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".aiff",
        ) as f:
            aiff_path = f.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as f:
            wav_path = f.name

        subprocess.run(
            [say_path, "-o", aiff_path, text],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

        subprocess.run(
            [
                convert_path,
                "-f", "WAVE",
                "-d", "LEI16@22050",
                aiff_path,
                wav_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if not os.path.exists(wav_path):
            raise RuntimeError("The WAV file was not created.")

        if os.path.getsize(wav_path) <= 44:
            raise RuntimeError("The generated WAV file is empty.")

        with wave.open(wav_path, "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise RuntimeError("The WAV file contains no audio frames.")

        return Path(wav_path).read_bytes()

    except Exception as exc:
        st.warning(
            f"Speech generation failed: {exc}. "
            "The response is still available as text."
        )
        return None

    finally:
        for path in (aiff_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


def describe_api_error(exc):
    parts = [
        f"{type(exc).__name__}: {exc}",
    ]

    status = getattr(exc, "status_code", None)

    if status:
        parts.append(f"HTTP status: {status}")

    body = getattr(exc, "body", None)

    if body:
        parts.append(f"Server response body: {body}")

    response = getattr(exc, "response", None)

    if response is not None:
        try:
            parts.append(
                f"Response details: {response.text[:3000]}"
            )
        except Exception:
            pass

    return "\n\n".join(parts)


def reset_interview():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


with st.sidebar:
    st.header("⚙️ Local configuration")

    base_url = st.text_input(
        "LM Studio API base URL",
        value="http://localhost:1234/v1",
    )

    model_name = st.text_input(
        "Model identifier",
        value=os.getenv("LM_STUDIO_MODEL", "local-model"),
        help=(
            "Use the exact model ID shown by LM Studio's "
            "GET /v1/models endpoint."
        ),
    )

    if st.button("Check LM Studio models"):
        try:
            client = make_client(base_url, "lm-studio")
            model_response = client.models.list()

            ids = [
                item.id for item in model_response.data
            ]

            if ids:
                st.success("Models returned by LM Studio:")
                for model_id in ids:
                    st.code(model_id)
            else:
                st.warning("LM Studio returned no models.")

        except Exception as exc:
            st.error(describe_api_error(exc))

    api_key = st.text_input(
        "API key",
        value=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        type="password",
    )

    temperature = st.slider(
        "Response creativity",
        0.0,
        1.0,
        0.4,
        0.05,
    )

    max_tokens = st.slider(
        "Max response tokens",
        300,
        4000,
        1500,
        100,
        help=(
            "Reasoning models (Qwen 3.x and similar) spend tokens on hidden "
            "thinking before answering. If answers get cut off or come back "
            "empty, raise this."
        ),
    )

    st.divider()
    st.subheader("Speech recognition")

    whisper_size = st.selectbox(
        "Whisper model",
        ["base", "small", "medium", "large-v3"],
        index=1,
    )

    whisper_device = st.selectbox(
        "Whisper device",
        ["cpu", "auto"],
        index=0,
    )

    whisper_compute = st.selectbox(
        "Compute type",
        ["int8", "float16", "int8_float16"],
        index=0,
    )

    whisper_language = st.selectbox(
        "Transcription language",
        ["en", "hi", "auto"],
        index=0,
        help=(
            "Pin a language to avoid misdetection on short clips. "
            "'auto' lets Whisper detect it (less reliable)."
        ),
    )

    st.caption(
        "Whisper model files may need to download on first use."
    )

    st.divider()

    if st.button("Reset interview", use_container_width=True):
        reset_interview()
        st.rerun()


st.title("🎙️ Local DSA Voice Interviewer")

st.caption(
    "Local interview reasoning through LM Studio, "
    "local speech recognition, and macOS offline speech."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

left, right = st.columns([1.2, 1])

with left:
    st.subheader("1. Interview setup")

    selected_question = st.selectbox(
        "Choose a DSA problem",
        list(QUESTION_BANK.keys()),
    )

    duration = st.select_slider(
        "Interview length",
        options=[15, 20, 30, 45, 60],
        value=30,
    )

    candidate_name = st.text_input(
        "Name (optional)",
        placeholder="Candidate",
    )

    question = QUESTION_BANK[selected_question]

    st.info(
        f"**Topic:** {question['topic']}  \n"
        f"**Difficulty:** {question['difficulty']}"
    )

    with st.expander("Problem statement", expanded=True):
        st.write(question["statement"])

    if st.button(
        "🚀 Start interview",
        type="primary",
        use_container_width=True,
    ):
        system_message = {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + f"\nInterview duration: {duration} minutes."
                + f"\nCandidate name: {candidate_name or 'Candidate'}."
                + f"\nCurrent problem: {selected_question}."
                + f"\nProblem statement: {question['statement']}"
            ),
        }

        # Keep this request in the stored history (hidden from the chat UI).
        # Some local model chat templates (e.g. Qwen 3.x) reject histories
        # whose first non-system message is an assistant turn.
        start_request = {
            "role": "user",
            "hidden": True,
            "content": (
                "Start the DSA interview now. "
                "Briefly greet the candidate, introduce the problem, "
                "and ask them to explain their initial approach "
                "before writing code."
            ),
        }

        try:
            client = make_client(base_url, api_key)

            with st.spinner("Connecting to your local model..."):
                greeting = get_model_response(
                    client,
                    model_name,
                    [system_message, start_request],
                    temperature,
                    max_tokens,
                )

            st.session_state.messages = [
                system_message,
                start_request,
                {
                    "role": "assistant",
                    "content": greeting,
                },
            ]

            st.session_state.selected_question = selected_question
            st.session_state.interview_started = True
            st.session_state.started_at = time.time()
            st.session_state.last_audio = synthesize_speech(greeting)
            st.session_state.last_transcript = ""

            st.rerun()

        except Exception as exc:
            st.error(
                "Could not start the interview. Check that "
                "LM Studio is running, a model is loaded, and "
                "the model identifier is correct."
            )
            st.code(describe_api_error(exc))


with right:
    st.subheader("2. Interview conversation")

    if not st.session_state.interview_started:
        st.write("Start an interview to enable voice and text responses.")

    else:
        elapsed = int(
            time.time()
            - st.session_state.get("started_at", time.time())
        )

        remaining = max(0, duration * 60 - elapsed)

        st.metric(
            "Time remaining",
            f"{remaining // 60:02d}:{remaining % 60:02d}",
        )

        for msg in st.session_state.messages:
            if msg["role"] == "system" or msg.get("hidden"):
                continue

            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.last_audio:
            st.markdown("#### 🔊 Interviewer voice")

            st.audio(
                st.session_state.last_audio,
                format="audio/wav",
            )

            if st.button("Play interviewer response again"):
                st.audio(
                    st.session_state.last_audio,
                    format="audio/wav",
                )

        else:
            st.info(
                "No audio was generated for the latest response. "
                "The text conversation is still available."
            )

        st.markdown("#### Answer by voice")

        audio_input = st.audio_input("Record an answer")

        voice_col1, voice_col2 = st.columns(2)

        with voice_col1:
            transcribe_clicked = st.button(
                "Transcribe recording",
                disabled=audio_input is None,
                use_container_width=True,
            )

        with voice_col2:
            send_voice_clicked = st.button(
                "Transcribe and send",
                disabled=audio_input is None,
                use_container_width=True,
                type="primary",
            )

        if transcribe_clicked or send_voice_clicked:
            try:
                with st.spinner("Transcribing locally..."):
                    transcript = transcribe_audio(
                        audio_input.getvalue(),
                        whisper_size,
                        whisper_device,
                        whisper_compute,
                        whisper_language,
                    )

                st.session_state.last_transcript = transcript

                if transcript:
                    st.success("Transcription complete.")
                    st.text_area(
                        "Recognized speech",
                        value=transcript,
                        key="recognized_speech",
                    )
                else:
                    st.warning(
                        "No speech was detected. Please record again."
                    )

            except Exception as exc:
                st.error(f"Transcription failed: {exc}")

        if send_voice_clicked:
            transcript = st.session_state.last_transcript.strip()

            if transcript:
                st.session_state.pending_candidate_message = transcript
                st.rerun()

        typed_answer = st.text_area(
            "Or type an answer / paste code",
            key="typed_candidate_answer",
            height=120,
            placeholder=(
                "Explain your approach, ask for a hint, "
                "or paste your code here..."
            ),
        )

        send_typed = st.button("Send text answer")

        end_interview = st.button(
            "End interview and get feedback"
        )

        pending_message = st.session_state.pop(
            "pending_candidate_message",
            None,
        )

        if send_typed and typed_answer.strip():
            pending_message = typed_answer.strip()

        if pending_message and st.session_state.interview_started:
            candidate_messages = (
                st.session_state.messages
                + [
                    {
                        "role": "user",
                        "content": pending_message,
                    }
                ]
            )

            try:
                with st.spinner("Thinking locally..."):
                    client = make_client(base_url, api_key)

                    response_text = get_model_response(
                        client,
                        model_name,
                        candidate_messages,
                        temperature,
                        max_tokens,
                    )

                # Only commit the candidate answer after the model succeeds.
                st.session_state.messages = candidate_messages + [
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                ]

                st.session_state.last_audio = synthesize_speech(
                    response_text
                )

                st.rerun()

            except Exception as exc:
                st.error("LM Studio request failed.")
                st.code(describe_api_error(exc))
                st.info(
                    "The failed answer was not added to the conversation. "
                    "Check the model ID and LM Studio server, then retry."
                )

        if end_interview:
            feedback_request = {
                "role": "user",
                "content": (
                    "The interview is over. Give a concise evaluation "
                    "based on the conversation. Score problem understanding, "
                    "algorithm selection, correctness evidence, complexity, "
                    "edge cases, and communication from 1 to 5. State clearly "
                    "that code was not executed. Give three specific next steps."
                ),
            }

            feedback_messages = (
                st.session_state.messages + [feedback_request]
            )

            try:
                with st.spinner("Preparing feedback..."):
                    client = make_client(base_url, api_key)

                    feedback = get_model_response(
                        client,
                        model_name,
                        feedback_messages,
                        0.2,
                        max_tokens,
                    )

                st.session_state.messages = feedback_messages + [
                    {
                        "role": "assistant",
                        "content": feedback,
                    }
                ]

                st.session_state.last_audio = synthesize_speech(feedback)

                st.rerun()

            except Exception as exc:
                st.error("Could not generate final feedback.")
                st.code(describe_api_error(exc))


st.divider()

st.caption(
    "Privacy: interview reasoning uses your local LM Studio server. "
    "Speech recognition runs locally with Faster-Whisper. "
    "Speech output uses macOS's built-in say command. "
    "No submitted code is executed."
)
