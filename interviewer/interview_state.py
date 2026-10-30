"""Structured interview state (Phase 4A/4C).

The conversation transcript (session.messages) answers "what was said";
InterviewState answers "where are we in the interview". The decision layer
(described in decision.py) updates this object after each interviewer turn,
and the current state is injected into the system prompt so the interviewer's
behaviour adapts to the candidate's actual progress.

Phases are advisory, not rigid: they transition on the decision layer's
suggestion, and the interviewer may move out of order as the candidate does.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    INTRODUCTION = "introduction"
    UNDERSTANDING = "understanding"
    ALGORITHM = "algorithm"
    COMPLEXITY = "complexity"
    EDGE_CASES = "edge_cases"
    CODE_REVIEW = "code_review"
    EVALUATION = "evaluation"


VALID_ACTIONS = {
    "ask_clarification",
    "ask_follow_up",
    "give_hint",
    "request_code",
    "review_code",
    "finish_interview",
}


class ComplexityAssessment(BaseModel):
    time: Optional[str] = None
    space: Optional[str] = None
    status: Optional[str] = None


class DecisionAssessment(BaseModel):
    """Validated output of the decision layer (Phase 4B).

    All fields are Optional-with-defaults so a partially malformed LLM
    response still validates; consumers must treat missing fields as
    "no new information", never as ground truth (Phase 4B guidance).
    """

    candidate_understanding: Optional[str] = None
    identified_issues: list[str] = Field(default_factory=list)
    complexity: Optional[ComplexityAssessment] = None
    interviewer_action: Optional[str] = None
    next_question: Optional[str] = None
    hint_level: Optional[int] = None
    suggested_phase: Optional[Phase] = None
    brief_comment: Optional[str] = None


class InterviewState:
    """Mutable tracker for one interview session."""

    def __init__(
        self,
        problem_id: str,
        topic: str,
        difficulty: str,
        duration_minutes: int,
    ) -> None:
        self.problem_id = problem_id
        self.topic = topic
        self.difficulty = difficulty
        self.duration_minutes = duration_minutes
        self.started_at = time.time()
        self.ended_at: float | None = None

        self.phase = Phase.INTRODUCTION
        self.candidate_approach: str | None = None
        self.candidate_understanding: str | None = None
        self.identified_issues: list[str] = []
        self.complexity: ComplexityAssessment | None = None

        self.hints_given = 0
        self.hint_level = 0
        # Code-path lifecycle kept explicit (audit 7A): asking for code is
        # not the same as the candidate submitting it, and review is the
        # conceptual pass afterwards.
        self.code_requested = False
        self.code_submitted = False
        self.code_reviewed = False
        self.actions: list[tuple[str, str]] = []  # (action, brief) pairs

    # ------------------------------------------------------------------
    def _next_question_brief(self) -> str | None:
        return self.actions[-1][1] if self.actions else None

    def apply_decision(self, decision: DecisionAssessment) -> None:
        """Merge a validated decision-layer assessment into the state."""
        if decision.candidate_understanding:
            self.candidate_understanding = decision.candidate_understanding
            if not self.candidate_approach and decision.brief_comment:
                self.candidate_approach = decision.brief_comment

        if decision.identified_issues is not None:
            # Open issues are the decision layer's current view, not an
            # accumulating log: issues the candidate has since resolved
            # must not linger into later turns or the final evaluation.
            self.identified_issues = [
                issue for issue in decision.identified_issues if issue
            ][:5]

        if decision.complexity is not None:
            self.complexity = decision.complexity

        if decision.hint_level is not None and decision.hint_level > self.hint_level:
            self.hint_level = int(decision.hint_level)

        action = decision.interviewer_action
        if action in VALID_ACTIONS:
            self.actions.append(
                (action, decision.brief_comment or decision.next_question or "")
            )

            if action == "give_hint":
                self.hints_given += 1
            elif action == "request_code":
                self.code_requested = True
            elif action == "review_code":
                # A review implies code was actually in the transcript.
                self.code_requested = True
                self.code_reviewed = True
            elif action == "ask_clarification":
                self.phase = max(self.phase, Phase.UNDERSTANDING, key=lambda p: list(Phase).index(p))

        if isinstance(decision.suggested_phase, Phase):
            # Advance only; never silently regress phases on its own.
            order = list(Phase)
            if order.index(decision.suggested_phase) > order.index(self.phase):
                self.phase = decision.suggested_phase

    def note_code_submitted(self) -> None:
        if not self.code_submitted:
            self.code_submitted = True
            self.phase = max(
                self.phase, Phase.CODE_REVIEW, key=lambda p: list(Phase).index(p)
            )

    # ------------------------------------------------------------------
    def directive_text(self) -> str:
        """Compact context injected into the system prompt (every turn)."""
        lines = [f"Current stage: {self.phase.value}"]

        if self.candidate_understanding:
            lines.append(f"Candidate understanding so far: {self.candidate_understanding}")
        if self.complexity is not None:
            lines.append(
                "Complexity discussed: "
                f"time={self.complexity.time or 'not stated'} "
                f"space={self.complexity.space or 'not stated'} "
                f"({self.complexity.status or 'assessed'})"
            )
        if self.identified_issues:
            lines.append("Open issues: " + "; ".join(self.identified_issues[-3:]))

        lines.append(f"Hints given so far: {self.hints_given} (deepest level {self.hint_level})")
        lines.append(
            "Hint policy: increase one level when the candidate is stuck; "
            "track independence; never reveal the full solution."
        )

        if self.code_requested or self.code_submitted:
            lines.append(
                "Code status: "
                + (
                    "reviewed conceptually"
                    if self.code_reviewed
                    else (
                        "candidate submitted code (review conceptually and "
                        "never claim it was executed)"
                        if self.code_submitted
                        else "you asked for code"
                    )
                )
            )

        last_action, last_brief = self.actions[-1] if self.actions else ("", "")
        if last_action:
            lines.append(f"Previous interviewer action: {last_action}")

        return "\n".join(lines)

    def summary_text(self) -> str:
        """Longer tracker view for the final evaluation."""
        parts = [
            f"Problem: {self.problem_id} ({self.topic}, {self.difficulty})",
            f"Phase reached: {self.phase.value}",
            f"Approach noted: {self.candidate_understanding or self.candidate_approach or 'none'}",
        ]
        if self.complexity is not None:
            parts.append(
                f"Complexity stated: time={self.complexity.time}, "
                f"space={self.complexity.space}, status={self.complexity.status}"
            )
        parts.append(f"Hints used: {self.hints_given} (deepest level {self.hint_level})")
        if self.identified_issues:
            parts.append("Issues observed: " + "; ".join(self.identified_issues))
        action_list = [a for a, _ in self.actions]
        parts.append(
            "Interviewer actions: " + (", ".join(action_list) if action_list else "none")
        )
        code_parts = [
            flag
            for flag, value in (
                ("asked for code", self.code_requested),
                ("candidate submitted code", self.code_submitted),
                ("code reviewed conceptually", self.code_reviewed),
            )
            if value
        ]
        if code_parts:
            parts.append("Code path: " + ", ".join(code_parts))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def to_ui(self) -> dict:
        return {
            "stage": self.phase.value,
            "hints": self.hints_given,
            "hint_level": self.hint_level,
            "action": self.actions[-1][0] if self.actions else None,
            "comment": self.actions[-1][1] if self.actions else None,
            "understanding": self.candidate_understanding,
            "complexity": (
                {
                    "time": self.complexity.time,
                    "space": self.complexity.space,
                    "status": self.complexity.status,
                }
                if self.complexity is not None
                else None
            ),
            "issues": list(self.identified_issues),
        }
