"""Interviewer decision layer (Phase 4B).

After the interviewer finishes speaking a turn, a separate non-streaming
call runs against the same LM Studio model to produce a structured,
validated assessment of the candidate and a recommended next action.

Design notes
------------
- The assessment runs asynchronously AFTER the reply has been spoken, so it
  never adds latency to the voice loop; its results shape the NEXT turn via
  the system prompt.
- Output is validated with Pydantic (DecisionAssessment). An invalid or
  empty response is logged and skipped: the assessment is treated as
  evidence to review, never as infallible ground truth.
- The decision prompt receives a compact transcript (heads of the system
  prompt excluded) plus the current interview state.
"""

from __future__ import annotations

import json
import logging
import re

from interviewer import config
from interviewer.interview_state import DecisionAssessment, InterviewState
from interviewer.llm_client import get_client, resolve_model_name

logger = logging.getLogger(__name__)

DECISION_SYSTEM_PROMPT = """You are the assessment layer of a live voice
technical interview. A separate interviewer voice talks to the candidate;
you never talk to the candidate yourself. Your job is to watch the
conversation and output structured assessment data.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{
  "candidate_understanding": "one sentence summarizing what the candidate has demonstrated so far",
  "identified_issues": ["short factual issue notes, empty if none"],
  "complexity": {"time": "as claimed by candidate or null", "space": "as claimed or null", "status": "correct|incorrect|not_assessed"},
  "interviewer_action": "one of: ask_clarification, ask_follow_up, give_hint, request_code, review_code, finish_interview",
  "next_question": "the concrete question the interviewer should ask next, or null",
  "hint_level": integer 0-4 (deeper only if the candidate is stuck),
  "suggested_phase": "one of: introduction, understanding, algorithm, complexity, edge_cases, code_review, evaluation",
  "brief_comment": "at most 12 words of context for the assessment log"
}

Rules:
- Base every field on the transcript. Do not invent claims.
- identified_issues must list ONLY issues that are still open after the
  latest exchange: remove issues the candidate has subsequently corrected,
  so resolved problems never resurface in later turns or the evaluation.
- Prefer ask_follow_up that moves the interview forward (approach ->
  algorithm -> complexity -> edge cases). Do not skip an assessment of
  correctness.
- If the candidate is stuck, prefer give_hint with a higher hint_level,
  guiding one step at a time. Never give the solution.
- Only choose finish_interview when the candidate has completed the
  problem or is clearly out of depth.
- request_code when it would help to see code; then review_code for the
  actual review (conceptual review only, code is never executed).
"""

MAX_TRANSCRIPT_MESSAGES = 16


def _transcript_brief(messages: list[dict]) -> str:
    """Compact transcript for the decision prompt.

    Excludes system messages (the current-state directive is attached
    separately) and hides raw token turns that are extremely long.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role not in ("user", "assistant"):
            continue
        text = content.strip()
        if len(text) > 1200:
            text = text[:1200] + " …"
        label = "INTERVIEWER" if role == "assistant" else "CANDIDATE"
        lines.append(f"{label}: {text}")
    return "\n\n".join(lines[-MAX_TRANSCRIPT_MESSAGES:])


def _extract_json(text: str) -> dict | None:
    """Find the first balanced JSON object in a possibly noisy reply."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    # Some models wrap JSON in code fences *inside* prose; try a stricter
    # fallback: first line that parses.
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def assess(interview_state: InterviewState, messages: list[dict]) -> DecisionAssessment | None:
    """Run one decision-layer call. Returns None on any failure."""
    client = get_client()
    model = resolve_model_name()

    decision_messages = [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Problem: {interview_state.problem_id} "
                f"({interview_state.topic}, {interview_state.difficulty}).\n"
                f"Current tracker:\n{interview_state.directive_text()}\n\n"
                f"Interview transcript (most recent last):\n"
                f"{_transcript_brief(messages)}\n\n"
                "Return the JSON assessment now."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=decision_messages,
            temperature=0.2,
            max_tokens=config.DECISION_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("Decision call failed: %s", exc)
        return None

    content = response.choices[0].message.content
    if isinstance(content, str):
        content = content.strip()
    if not content:
        return None

    payload = _extract_json(content)
    if payload is None:
        logger.info("Decision reply was not parseable JSON: %.200s", content)
        return None

    try:
        return DecisionAssessment.model_validate(payload)
    except Exception as exc:
        logger.info("Decision JSON failed schema validation: %s", exc)
        return None
