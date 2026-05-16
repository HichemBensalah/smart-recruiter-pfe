from src.core.chatbot.nodes.analyze_transferability import analyze_transferability_node
from src.core.chatbot.nodes.compose_answer import compose_answer_node
from src.core.chatbot.nodes.fetch_decision_cards import fetch_decision_cards_node
from src.core.chatbot.nodes.match_candidates import match_candidates_node
from src.core.chatbot.nodes.understand_query import understand_query_node
from src.core.chatbot.state import initial_state


def test_understand_query_infers_target_role_without_llm() -> None:
    state = initial_state("Je cherche un data engineer top 3")
    update = understand_query_node(state)

    assert update["target_role"] == "Data Engineer"
    assert update["top_k"] == 3
    assert update["job_description"] == "Je cherche un data engineer top 3"


def test_compose_answer_handles_empty_candidates() -> None:
    state = initial_state("Je cherche un backend")
    state.update({"target_role": "Backend Developer", "candidates": []})

    update = compose_answer_node(state)

    assert "Aucun candidat" in update["answer"]
    assert "Matching V3" in update["answer"]


def test_compose_answer_uses_only_state_data() -> None:
    state = initial_state("Je cherche un backend")
    state.update(
        {
            "target_role": "Backend Developer",
            "candidates": [
                {
                    "candidate_id": "candidate_1",
                    "baseline_rank_v3": 1,
                    "baseline_score_v3": 0.8,
                    "rf_score": 0.9,
                    "xgboost_score": 0.7,
                    "recommendation_status": "agreement_high",
                }
            ],
            "decision_cards": [{"candidate_id": "candidate_1", "recommendation_status": "agreement_high"}],
            "transferability": {
                "candidate_1": {
                    "selected_source": "yaml",
                    "yaml": {
                        "transferability": {
                            "transferability_score": 0.5,
                            "gaps_bloquants": ["Django"],
                        }
                    },
                }
            },
        }
    )

    update = compose_answer_node(state)

    assert "candidate_1" in update["answer"]
    assert "score V3 0.8000" in update["answer"]
    assert "Django" in update["answer"]


def test_match_candidates_node_handles_tool_error(monkeypatch) -> None:
    class FakeTool:
        @staticmethod
        def invoke(payload):
            raise RuntimeError("api down")

    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeTool())
    state = initial_state("Backend Python")
    state.update({"job_description": "Backend Python"})

    update = match_candidates_node(state)

    assert update["candidates"] == []
    assert update["warnings"]


def test_fetch_decision_cards_node_continues_when_no_candidates() -> None:
    update = fetch_decision_cards_node(initial_state("Backend Python"))

    assert update["decision_cards"] == []


def test_analyze_transferability_node_continues_when_no_candidates() -> None:
    update = analyze_transferability_node(initial_state("Backend Python"))

    assert update["transferability"] == {}
    assert update["neo4j_available"] is False
