from __future__ import annotations

from src.core.chatbot.state import RecruiterCopilotState
from src.core.chatbot.tools.candidate_tools import get_candidate_profile_tool
from src.core.chatbot.tools.decision_card_tools import get_decision_card_tool


FOLLOW_UP_INTENTS = {
    "explain_candidate",
    "review_needed",
    "gap_analysis",
    "compare_candidates",
    "transferability",
}


def fetch_decision_cards_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    warnings = list(state.get("warnings", []))
    sources = list(state.get("sources", []))
    existing_cards = state.get("decision_cards", [])
    if state.get("intent") in FOLLOW_UP_INTENTS and existing_cards:
        if "conversation_memory" not in sources:
            sources.append("conversation_memory")
        return {"decision_cards": existing_cards, "warnings": warnings, "sources": sources}

    cards: list[dict] = []
    enriched_candidates: list[dict] = []
    for candidate in state.get("candidates", [])[:5]:
        enriched_candidate = dict(candidate)
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            warnings.append("candidate without candidate_id skipped for decision card lookup")
            enriched_candidates.append(enriched_candidate)
            continue
        try:
            card = get_decision_card_tool.invoke({"candidate_id": candidate_id})
            if isinstance(card, dict):
                cards.append(card)
                _copy_candidate_name(enriched_candidate, card)
        except Exception as exc:
            warnings.append(f"decision card missing for {candidate_id}: {exc}")
        try:
            profile = get_candidate_profile_tool.invoke({"candidate_id": candidate_id})
            if isinstance(profile, dict):
                _copy_candidate_name(enriched_candidate, profile)
        except Exception as exc:
            warnings.append(f"candidate profile unavailable for {candidate_id}: {exc}")
        enriched_candidates.append(enriched_candidate)

    if len(state.get("candidates", [])) > len(enriched_candidates):
        enriched_candidates.extend(state.get("candidates", [])[len(enriched_candidates) :])
    if cards and "get_decision_card" not in sources:
        sources.append("get_decision_card")
    return {"candidates": enriched_candidates, "decision_cards": cards, "warnings": warnings, "sources": sources}


def _copy_candidate_name(candidate: dict, payload: dict) -> None:
    name = _extract_candidate_name(payload)
    if name and not candidate.get("candidate_name"):
        candidate["candidate_name"] = name


def _extract_candidate_name(payload: dict) -> str | None:
    for key in ("candidate_name", "full_name", "name"):
        value = payload.get(key)
        if value:
            return str(value)
    for key in ("candidate", "profile", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            nested_name = _extract_candidate_name(nested)
            if nested_name:
                return nested_name
    return None
