from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from src.core.chatbot.tools.api_client import SmartRecruiterApiClient
from src.core.chatbot.tools.schemas import CandidateProfileInput


def get_candidate_profile(candidate_id: str) -> dict[str, Any]:
    """Return one candidate profile and its available card data."""
    client = SmartRecruiterApiClient()
    return client.get(f"/api/candidates/{candidate_id}")


get_candidate_profile_tool = StructuredTool.from_function(
    func=get_candidate_profile,
    name="get_candidate_profile",
    description="Fetch a candidate profile by candidate_id from the Smart Recruiter API.",
    args_schema=CandidateProfileInput,
)
