"""LM Studio (OpenAI-compatible) client for the voice interviewer.

Two key behaviours for robust local-model integration:
- Messages are normalized before every request (some local model chat
  templates, e.g. Qwen 3.x, reject a history whose first non-system
  message is an assistant turn with HTTP 400 "No user query found").
- The language is streamed token-by-token (stream=True) so the UI can show
  text immediately and TTS can start synthesize sentences early.
"""

from __future__ import annotations

import os
import re
import threading

from openai import OpenAI

from interviewer import config

_client: OpenAI | None = None
_client_lock = threading.Lock()

_model_name: str | None = None


def get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                base_url=config.LM_STUDIO_BASE_URL.rstrip("/"),
                api_key=config.LM_STUDIO_API_KEY or "lm-studio",
                timeout=180.0,
                max_retries=0,
            )
        return _client


def resolve_model_name() -> str:
    """Return the model identifier to use for requests.

    Uses LM_STUDIO_MODEL if set, otherwise asks LM Studio for its
    currently available models and returns the first one.
    """
    global _model_name
    if _model_name:
        return _model_name

    env_model = getattr(config, "LM_STUDIO_MODEL", "") or os.getenv(
        "LM_STUDIO_MODEL", ""
    )
    if env_model.strip():
        _model_name = env_model.strip()
        return _model_name

    response = get_client().models.list()
    ids = [item.id for item in response.data]
    if not ids:
        raise RuntimeError(
            "LM Studio returned no models. Make sure a model is loaded."
        )
    _model_name = ids[0]
    return _model_name


def describe_api_error(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    status = getattr(exc, "status_code", None)
    if status:
        parts.append(f"HTTP status: {status}")
    body = getattr(exc, "body", None)
    if body:
        parts.append(f"Server response body: {body}")
    return "\n\n".join(parts)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Clean a message list for LM Studio's chat template.

    - Drops unknown roles and empty content.
    - Ensures at least one user message exists.
    - Ensures the first non-system message has role 'user' (Qwen 3.x and
      similar templates reject assistant-first histories).
    """
    normalized = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content.strip()})

    if not any(m["role"] == "user" for m in normalized):
        normalized.append(
            {"role": "user", "content": "Please begin the DSA interview."}
        )

    first_non_system = 0
    if normalized and normalized[0]["role"] == "system":
        first_non_system = 1
    if (
        first_non_system < len(normalized)
        and normalized[first_non_system]["role"] != "user"
    ):
        normalized.insert(
            first_non_system,
            {"role": "user", "content": "Please continue the interview."},
        )

    return normalized


# Sentence boundaries: . ! or ? followed by whitespace, but not an
# enumerated-list item like "1. " (protects spoken-style feedback lists).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Split *buffer* into complete sentences plus the trailing remainder.

    The last element of a re.split is always the remainder (possibly ''),
    so sentences = parts[:-1].
    """
    if not buffer:
        return [], ""
    parts = _SENTENCE_SPLIT.split(buffer)
    remainder = parts[-1]
    sentences = []
    for candidate in parts[:-1]:
        # Glue tiny fragments (e.g. "1.") to the next sentence instead
        # of speaking them alone.
        if candidate.strip() and len(candidate.strip()) >= 4:
            sentences.append(candidate)
        else:
            remainder = candidate + " " + remainder
    return sentences, remainder


def stream_sentences(
    messages: list[dict],
    temperature: float = 0.4,
    max_tokens: int = 1500,
):
    """Blocking generator over LM Studio's streaming chat API.

    Yields tuples:
        ("delta", text)    a raw token delta (for the live transcript UI)
        ("sentence", text) a complete sentence (for sentence-wise TTS)

    Raises RuntimeError with a readable message on failure or when the
    model returns no visible content (e.g. spent the whole token budget
    on chain-of-thought reasoning).
    """
    client = get_client()
    model = resolve_model_name()
    normalized = normalize_messages(messages)

    collected: list[str] = []
    sentence_buffer = ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=normalized,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except Exception as exc:
        raise RuntimeError(describe_api_error(exc)) from exc

    complete = False
    for chunk in stream:
        if not chunk.choices:
            continue
        finish_reason = chunk.choices[0].finish_reason
        if finish_reason is not None:
            complete = True

        delta = chunk.choices[0].delta
        token = getattr(delta, "content", None)
        if isinstance(token, str) and token:
            collected.append(token)
            yield ("delta", token)
            sentence_buffer += token
            sentences, sentence_buffer = _extract_sentences(sentence_buffer)
            for sentence in sentences:
                yield ("sentence", sentence)

    if sentence_buffer.strip():
        yield ("sentence", sentence_buffer.strip())

    visible = "".join(collected).strip()
    if not visible:
        raise RuntimeError(
            "The model returned an empty response. Reasoning ('thinking') "
            "models such as Qwen 3.x can spend the entire token budget on "
            "hidden reasoning. Raise LLM_MAX_TOKENS, or disable thinking "
            "for this model in LM Studio, then try again."
        )
