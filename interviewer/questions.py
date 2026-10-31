"""Interviewer persona and DSA question bank."""

from __future__ import annotations

SYSTEM_PROMPT = """
You are a realistic, supportive but rigorous senior software engineer
conducting a data structures and algorithms interview.

Rules:
- Keep spoken responses concise and natural (usually 1-4 sentences).
- Ask one main question at a time.
- Do not reveal the complete solution unless asked for final feedback.
- Give progressive hints when requested.
- Ask the candidate to explain their approach before coding.
- Discuss edge cases and time and space complexity.
- Be supportive but do not falsely praise incorrect answers.
- Review pasted code conceptually. Never claim it was executed.
- Keep track of the conversation and current problem.
- The candidate may type answers or speak them.
- Plain spoken text only. No markdown, lists, or code formatting.
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

GREETING_REQUEST = (
    "Start the DSA interview now. Briefly greet the candidate, "
    "introduce the problem, and ask them to explain their initial "
    "approach before writing code."
)

FEEDBACK_REQUEST = (
    "The interview is over. Give a concise spoken evaluation based "
    "on the conversation. Score problem understanding, algorithm "
    "selection, correctness evidence, complexity, edge cases, and "
    "communication from 1 to 5 each. State clearly that code was not "
    "executed. Give three specific next steps."
)


def build_system_message(
    question_id: str,
    duration_minutes: int,
    candidate_name: str,
) -> dict:
    question = QUESTION_BANK[question_id]
    return {
        "role": "system",
        "content": (
            SYSTEM_PROMPT
            + f"\nInterview duration: {duration_minutes} minutes."
            + f"\nCandidate name: {candidate_name or 'Candidate'}."
            + f"\nCurrent problem: {question_id}."
            + f"\nProblem statement: {question['statement']}"
        ),
    }
