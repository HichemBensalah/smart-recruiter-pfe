from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.core.chatbot.nodes.analyze_transferability import analyze_transferability_node
from src.core.chatbot.nodes.compose_answer import compose_answer_node
from src.core.chatbot.nodes.fetch_decision_cards import fetch_decision_cards_node
from src.core.chatbot.nodes.match_candidates import match_candidates_node
from src.core.chatbot.nodes.understand_query import understand_query_node
from src.core.chatbot.state import RecruiterCopilotState, initial_state


def build_recruiter_copilot_graph():
    graph = StateGraph(RecruiterCopilotState)
    graph.add_node("understand_query", understand_query_node)
    graph.add_node("match_candidates", match_candidates_node)
    graph.add_node("fetch_decision_cards", fetch_decision_cards_node)
    graph.add_node("analyze_transferability", analyze_transferability_node)
    graph.add_node("compose_answer", compose_answer_node)

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "match_candidates")
    graph.add_edge("match_candidates", "fetch_decision_cards")
    graph.add_edge("fetch_decision_cards", "analyze_transferability")
    graph.add_edge("analyze_transferability", "compose_answer")
    graph.add_edge("compose_answer", END)
    return graph.compile()


def run_recruiter_copilot(message: str) -> dict[str, Any]:
    app = build_recruiter_copilot_graph()
    final_state = app.invoke(initial_state(message))
    return {
        "answer": final_state.get("answer"),
        "candidates": final_state.get("candidates", []),
        "decision_cards": final_state.get("decision_cards", []),
        "transferability": final_state.get("transferability", {}),
        "sources": final_state.get("sources", []),
        "warnings": final_state.get("warnings", []),
    }
