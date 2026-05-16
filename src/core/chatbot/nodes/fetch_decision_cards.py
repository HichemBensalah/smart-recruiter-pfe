from __future__ import annotations

from src.core.chatbot.state import RecruiterCopilotState
from src.core.chatbot.tools.decision_card_tools import get_decision_card_tool


def fetch_decision_cards_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    warnings = list(state.get("warnings", []))
    sources = list(state.get("sources", []))
    cards: list[dict] = []
    for candidate in state.get("candidates", [])[:5]:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            warnings.append("candidate without candidate_id skipped for decision card lookup")
            continue
        try:
            card = get_decision_card_tool.invoke({"candidate_id": candidate_id})
            if isinstance(card, dict):
                cards.append(card)
        except Exception as exc:
            warnings.append(f"decision card missing for {candidate_id}: {exc}")
    if cards and "get_decision_card" not in sources:
        sources.append("get_decision_card")
    return {"decision_cards": cards, "warnings": warnings, "sources": sources}
