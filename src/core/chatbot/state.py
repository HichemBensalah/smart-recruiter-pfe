from __future__ import annotations

from typing import Any, TypedDict


class RecruiterCopilotState(TypedDict, total=False):
    user_message: str
    job_description: str | None
    top_k: int
    target_role: str
    candidates: list[dict[str, Any]]
    decision_cards: list[dict[str, Any]]
    transferability: dict[str, Any]
    neo4j_available: bool
    answer: str | None
    sources: list[str]
    warnings: list[str]


def initial_state(user_message: str) -> RecruiterCopilotState:
    return {
        "user_message": user_message,
        "job_description": None,
        "top_k": 5,
        "target_role": "Backend Developer",
        "candidates": [],
        "decision_cards": [],
        "transferability": {},
        "neo4j_available": False,
        "answer": None,
        "sources": [],
        "warnings": [],
    }
