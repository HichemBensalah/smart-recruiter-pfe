from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from src.core.chatbot.tools.api_client import SmartRecruiterApiClient
from src.core.chatbot.tools.schemas import MatchCandidatesInput


def match_candidates(job_description: str, top_k: int = 10) -> dict[str, Any]:
    """Return Matching V3 candidate recommendations from the FastAPI facade."""
    client = SmartRecruiterApiClient()
    return client.post("/api/match", {"job_description": job_description, "top_k": top_k})


match_candidates_tool = StructuredTool.from_function(
    func=match_candidates,
    name="match_candidates",
    description=(
        "Find top candidate recommendations for a job description using the Smart Recruiter API. "
        "Returns existing Matching V3/ML/demo scores without inventing candidates."
    ),
    args_schema=MatchCandidatesInput,
)
