from __future__ import annotations

import re

from src.core.chatbot.state import RecruiterCopilotState


def understand_query_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    user_message = str(state.get("user_message") or "").strip()
    lowered = user_message.lower()
    target_role = "Backend Developer"
    if "data engineer" in lowered or "data engineering" in lowered:
        target_role = "Data Engineer"
    elif "data analyst" in lowered or "analyste data" in lowered:
        target_role = "Data Analyst"
    elif "machine learning" in lowered or "ml engineer" in lowered:
        target_role = "Machine Learning Engineer"
    elif "devops" in lowered:
        target_role = "DevOps Engineer"
    elif "frontend" in lowered or "front-end" in lowered:
        target_role = "Frontend Developer"
    elif "full stack" in lowered or "fullstack" in lowered:
        target_role = "Full Stack Developer"
    elif "backend" in lowered or "back-end" in lowered:
        target_role = "Backend Developer"

    top_k = _extract_top_k(user_message) or int(state.get("top_k") or 5)
    top_k = max(1, min(10, top_k))
    return {
        "job_description": user_message,
        "target_role": target_role,
        "top_k": top_k,
        "sources": _append_unique(state.get("sources", []), ["user_message"]),
    }


def _extract_top_k(message: str) -> int | None:
    match = re.search(r"\btop\s*(\d+)\b", message.lower())
    if match:
        return int(match.group(1))
    return None


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result
