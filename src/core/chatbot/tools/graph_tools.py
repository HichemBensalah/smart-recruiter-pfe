from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import StructuredTool

from src.core.chatbot.tools.api_client import SmartRecruiterApiClient, SmartRecruiterApiError
from src.core.chatbot.tools.schemas import Neo4jTransferabilityInput, TransferabilityInput


def get_transferability(candidate_id: str) -> dict[str, Any]:
    """Return YAML/fallback transferability from Decision Cards with Transferability."""
    client = SmartRecruiterApiClient()
    return client.get(f"/api/graph/transferability/{candidate_id}")


def get_neo4j_transferability(candidate_id: str, target_role: str = "Backend Developer") -> dict[str, Any]:
    """Return Neo4j transferability if Neo4j is configured and available."""
    return _neo4j_guarded_call(
        lambda client: client.get(
            f"/api/graph/neo4j/transferability/{candidate_id}",
            params={"target_role": target_role},
        )
    )


def get_neo4j_gaps(candidate_id: str, target_role: str = "Backend Developer") -> dict[str, Any]:
    """Return Neo4j compensable/blocking gaps if Neo4j is configured and available."""
    return _neo4j_guarded_call(
        lambda client: client.get(
            f"/api/graph/neo4j/gaps/{candidate_id}",
            params={"target_role": target_role},
        )
    )


def _neo4j_guarded_call(callback: Callable[[SmartRecruiterApiClient], dict[str, Any]]) -> dict[str, Any]:
    client = SmartRecruiterApiClient()
    try:
        return callback(client)
    except SmartRecruiterApiError as exc:
        return {
            "available": False,
            "message": str(exc),
            "fallback_recommended": True,
        }


get_transferability_tool = StructuredTool.from_function(
    func=get_transferability,
    name="get_transferability",
    description="Fetch YAML-based transferability, gaps and transition analysis for a candidate.",
    args_schema=TransferabilityInput,
)

get_neo4j_transferability_tool = StructuredTool.from_function(
    func=get_neo4j_transferability,
    name="get_neo4j_transferability",
    description="Fetch Neo4j Graph-RAG transferability for a candidate and target role. Returns a clear fallback response if Neo4j is unavailable.",
    args_schema=Neo4jTransferabilityInput,
)

get_neo4j_gaps_tool = StructuredTool.from_function(
    func=get_neo4j_gaps,
    name="get_neo4j_gaps",
    description="Fetch Neo4j Graph-RAG compensable and blocking gaps for a candidate and target role. Returns a clear fallback response if Neo4j is unavailable.",
    args_schema=Neo4jTransferabilityInput,
)
