from __future__ import annotations

from langchain_core.tools import BaseTool

from src.core.chatbot.tools.candidate_tools import get_candidate_profile_tool
from src.core.chatbot.tools.decision_card_tools import get_decision_card_tool
from src.core.chatbot.tools.demo_tools import (
    get_demo_executive_summary_tool,
    get_demo_top10_summary_tool,
    run_demo_tool,
)
from src.core.chatbot.tools.graph_tools import (
    get_neo4j_gaps_tool,
    get_neo4j_transferability_tool,
    get_transferability_tool,
)
from src.core.chatbot.tools.match_tools import match_candidates_tool


SMART_RECRUITER_TOOLS: list[BaseTool] = [
    match_candidates_tool,
    get_candidate_profile_tool,
    get_decision_card_tool,
    get_transferability_tool,
    get_neo4j_transferability_tool,
    get_neo4j_gaps_tool,
    get_demo_executive_summary_tool,
    get_demo_top10_summary_tool,
    run_demo_tool,
]


def get_smart_recruiter_tools() -> list[BaseTool]:
    return list(SMART_RECRUITER_TOOLS)
