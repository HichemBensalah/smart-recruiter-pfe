from __future__ import annotations

from typing import Any, TypedDict


class RecruiterCopilotState(TypedDict, total=False):
    user_message: str
    session_id: str | None
    intent: str
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
    selected_candidate_id: str | None
    matching_completed: bool
    job_intake_state: dict[str, Any] | None
    structured_job_profile: dict[str, Any] | None
    routed_job_id: str | None
    matching_metadata: dict[str, Any]


def initial_state(user_message: str) -> RecruiterCopilotState:
    return {
        "user_message": user_message,
        "session_id": None,
        "intent": "search_candidates",
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
        "selected_candidate_id": None,
        "matching_completed": False,
        "job_intake_state": None,
        "structured_job_profile": None,
        "routed_job_id": None,
        "matching_metadata": {},
    }
