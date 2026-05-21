from __future__ import annotations

import re

from src.core.chatbot.state import RecruiterCopilotState


def understand_query_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    user_message = str(state.get("user_message") or "").strip()
    lowered = user_message.lower()
    intent = _detect_intent(lowered)
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
        "intent": intent,
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


def _detect_intent(lowered_message: str) -> str:
    if _contains_any(lowered_message, ["pourquoi", "recommandé", "recommande", "premier candidat"]):
        return "explain_candidate"
    if _contains_any(lowered_message, ["à vérifier", "a verifier", "verifier", "review", "risque", "désaccord", "desaccord"]):
        return "review_needed"
    if _contains_any(lowered_message, ["gap", "gaps", "manque", "manquants", "compétences manquantes", "competences manquantes"]):
        return "gap_analysis"
    if _contains_any(lowered_message, ["compare", "comparer", "différence", "difference", "deux premiers"]):
        return "compare_candidates"
    if _contains_any(lowered_message, ["évoluer", "evoluer", "transition", "transférabilité", "transferabilite", "transferability"]):
        return "transferability"
    if _contains_any(lowered_message, ["cherche", "trouve", "profil", "candidat", "recrute"]):
        return "search_candidates"
    return "search_candidates"


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _append_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)
    for value in additions:
        if value not in result:
            result.append(value)
    return result
