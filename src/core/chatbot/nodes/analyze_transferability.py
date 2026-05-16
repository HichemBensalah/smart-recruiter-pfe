from __future__ import annotations

from typing import Any

from src.core.chatbot.state import RecruiterCopilotState
from src.core.chatbot.tools.graph_tools import (
    get_neo4j_transferability_tool,
    get_transferability_tool,
)


def analyze_transferability_node(state: RecruiterCopilotState) -> RecruiterCopilotState:
    warnings = list(state.get("warnings", []))
    sources = list(state.get("sources", []))
    target_role = str(state.get("target_role") or "Backend Developer")
    transferability: dict[str, Any] = {}
    neo4j_available = False

    for candidate in state.get("candidates", [])[:5]:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            continue
        yaml_result: dict[str, Any] | None = None
        try:
            yaml_result = get_transferability_tool.invoke({"candidate_id": candidate_id})
        except Exception as exc:
            warnings.append(f"transferability YAML unavailable for {candidate_id}: {exc}")

        neo4j_result: dict[str, Any] | None = None
        try:
            neo4j_result = get_neo4j_transferability_tool.invoke(
                {"candidate_id": candidate_id, "target_role": target_role}
            )
            neo4j_available = neo4j_available or not _is_neo4j_unavailable(neo4j_result)
        except Exception as exc:
            neo4j_result = {"available": False, "message": str(exc), "fallback_recommended": True}

        transferability[candidate_id] = {
            "yaml": yaml_result,
            "neo4j": neo4j_result,
            "selected_source": "neo4j" if neo4j_result and not _is_neo4j_unavailable(neo4j_result) else "yaml",
        }

    if transferability and "get_transferability" not in sources:
        sources.append("get_transferability")
    if neo4j_available and "get_neo4j_transferability" not in sources:
        sources.append("get_neo4j_transferability")
    return {
        "transferability": transferability,
        "neo4j_available": neo4j_available,
        "warnings": warnings,
        "sources": sources,
    }


def _is_neo4j_unavailable(payload: dict[str, Any] | None) -> bool:
    return not payload or payload.get("available") is False or payload.get("fallback_recommended") is True
