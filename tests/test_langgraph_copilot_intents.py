from src.core.chatbot.nodes.compose_answer import compose_answer_node
from src.core.chatbot.nodes.understand_query import understand_query_node
from src.core.chatbot.state import initial_state


def test_understand_query_detects_search_candidates() -> None:
    update = understand_query_node(initial_state("Je cherche un développeur backend"))
    assert update["intent"] == "search_candidates"


def test_understand_query_detects_explain_candidate() -> None:
    update = understand_query_node(initial_state("Pourquoi le premier candidat est recommandé ?"))
    assert update["intent"] == "explain_candidate"


def test_understand_query_detects_review_needed() -> None:
    update = understand_query_node(initial_state("Quels candidats sont à vérifier ?"))
    assert update["intent"] == "review_needed"


def test_understand_query_detects_gap_analysis() -> None:
    update = understand_query_node(initial_state("Quels sont les gaps du meilleur candidat ?"))
    assert update["intent"] == "gap_analysis"


def test_understand_query_detects_compare_candidates() -> None:
    update = understand_query_node(initial_state("Compare les deux premiers candidats"))
    assert update["intent"] == "compare_candidates"


def test_understand_query_detects_transferability() -> None:
    update = understand_query_node(initial_state("Est-ce que ce candidat peut évoluer vers Backend Developer ?"))
    assert update["intent"] == "transferability"


def test_compose_explain_candidate_answer_focuses_on_first_candidate() -> None:
    state = _mock_state("explain_candidate")
    answer = compose_answer_node(state)["answer"]

    assert "premier candidat" in answer
    assert "candidate_1" in answer
    assert "Top 3 candidats recommandés" not in answer


def test_compose_review_needed_answer_uses_review_title() -> None:
    state = _mock_state("review_needed")
    answer = compose_answer_node(state)["answer"]

    assert "Candidats à vérifier" in answer
    assert "Top candidats recommandés" not in answer
    assert "candidate_2" in answer


def test_compose_gap_analysis_answer_mentions_gaps() -> None:
    state = _mock_state("gap_analysis")
    answer = compose_answer_node(state)["answer"]

    assert "gaps" in answer.lower()
    assert "Django" in answer


def test_compose_compare_candidates_answer_mentions_comparison() -> None:
    state = _mock_state("compare_candidates")
    answer = compose_answer_node(state)["answer"]

    assert "Comparaison" in answer
    assert "candidate_1" in answer
    assert "candidate_2" in answer


def test_compose_transferability_answer_mentions_transferability() -> None:
    state = _mock_state("transferability")
    answer = compose_answer_node(state)["answer"]

    assert "transférabilité" in answer.lower() or "transition" in answer.lower()
    assert "candidate_1" in answer


def _mock_state(intent: str):
    state = initial_state("Question recruteur")
    state.update(
        {
            "intent": intent,
            "target_role": "Backend Developer",
            "candidates": [
                {
                    "candidate_id": "candidate_1",
                    "baseline_rank_v3": 1,
                    "baseline_score_v3": 0.82,
                    "rf_score": 0.91,
                    "xgboost_score": 0.77,
                    "recommendation_status": "agreement_high",
                },
                {
                    "candidate_id": "candidate_2",
                    "baseline_rank_v3": 2,
                    "baseline_score_v3": 0.71,
                    "rf_score": 0.55,
                    "xgboost_score": 0.22,
                    "recommendation_status": "review_needed",
                    "rank_shift_v3_vs_xgb": 20,
                },
            ],
            "decision_cards": [
                {"candidate_id": "candidate_1", "recommendation_status": "agreement_high"},
                {"candidate_id": "candidate_2", "recommendation_status": "review_needed"},
            ],
            "transferability": {
                "candidate_1": {
                    "selected_source": "yaml",
                    "yaml": {
                        "transferability": {
                            "fit_direct": False,
                            "transferability_score": 0.6,
                            "gaps_compensables": ["Docker"],
                            "gaps_bloquants": ["Django"],
                            "transitions_plausibles": [{"to_role": "Backend Developer"}],
                        }
                    },
                },
                "candidate_2": {
                    "selected_source": "yaml",
                    "yaml": {
                        "transferability": {
                            "fit_direct": False,
                            "transferability_score": 0.3,
                            "gaps_compensables": [],
                            "gaps_bloquants": ["Python", "SQL"],
                        }
                    },
                },
            },
        }
    )
    return state
