from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from src.core.chatbot.tools.api_client import SmartRecruiterApiClient
from src.core.chatbot.tools.schemas import DecisionCardInput


def get_decision_card(candidate_id: str) -> dict[str, Any]:
    """Return the available Decision Card for a candidate."""
    client = SmartRecruiterApiClient()
    return client.get(f"/api/decision-cards/{candidate_id}")


get_decision_card_tool = StructuredTool.from_function(
    func=get_decision_card,
    name="get_decision_card",
    description="Fetch a candidate Decision Card by candidate_id, preserving official and experimental scores from the API.",
    args_schema=DecisionCardInput,
)
