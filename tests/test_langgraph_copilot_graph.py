from src.core.chatbot.graph import (
    build_recruiter_copilot_graph,
    run_recruiter_copilot,
    run_recruiter_copilot_with_memory,
)
from src.core.chatbot.memory import SESSION_STORE


def test_build_recruiter_copilot_graph_compiles() -> None:
    graph = build_recruiter_copilot_graph()

    assert graph is not None


def test_run_recruiter_copilot_with_mocked_tools(monkeypatch) -> None:
    class FakeMatchTool:
        @staticmethod
        def invoke(payload):
            return {
                "items": [
                    {
                        "candidate_id": "candidate_1",
                        "baseline_rank_v3": 1,
                        "baseline_score_v3": 0.82,
                        "recommendation_status": "agreement_high",
                    }
                ]
            }

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "recommendation_status": "agreement_high"}

    class FakeTransferabilityTool:
        @staticmethod
        def invoke(payload):
            return {
                "candidate_id": payload["candidate_id"],
                "transferability": {
                    "transferability_score": 0.55,
                    "gaps_bloquants": [],
                },
            }

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {
                "available": False,
                "message": "Neo4j is not configured",
                "fallback_recommended": True,
            }

    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool())
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool", FakeDecisionCardTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_transferability_tool", FakeTransferabilityTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool", FakeNeo4jTool())

    result = run_recruiter_copilot("Je cherche un développeur backend Python FastAPI MongoDB")

    assert result["answer"]
    assert result["candidates"][0]["candidate_id"] == "candidate_1"
    assert result["decision_cards"][0]["candidate_id"] == "candidate_1"
    assert "candidate_1" in result["transferability"]
    assert "warnings" in result


def test_run_recruiter_copilot_with_memory_keeps_session_context(monkeypatch) -> None:
    class FakeMatchTool:
        calls = 0

        @classmethod
        def invoke(cls, payload):
            cls.calls += 1
            return {
                "items": [
                    {"candidate_id": "candidate_1", "baseline_rank_v3": 1, "baseline_score_v3": 0.82},
                    {"candidate_id": "candidate_2", "baseline_rank_v3": 2, "baseline_score_v3": 0.75},
                ]
            }

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "recommendation_status": "agreement_high"}

    class FakeTransferabilityTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "transferability": {"gaps_bloquants": []}}

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {"available": False, "fallback_recommended": True}

    SESSION_STORE.clear("session-test")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool)
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool", FakeDecisionCardTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_transferability_tool", FakeTransferabilityTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool", FakeNeo4jTool())

    memory = SESSION_STORE.get_or_create("session-test")
    memory.matching_completed = True
    memory.last_candidates = [
        {"candidate_id": "candidate_1", "baseline_rank_v3": 1, "baseline_score_v3": 0.82},
        {"candidate_id": "candidate_2", "baseline_rank_v3": 2, "baseline_score_v3": 0.75},
    ]
    memory.last_decision_cards = [{"candidate_id": "candidate_1", "recommendation_status": "agreement_high"}]
    memory.last_transferability = {"candidate_1": {"selected_source": "yaml", "yaml": {"gaps_bloquants": []}}}
    follow_up = run_recruiter_copilot_with_memory("Pourquoi le premier candidat ?", "session-test")

    assert follow_up["session_id"] == "session-test"
    assert follow_up["selected_candidate_id"] == "candidate_1"
    assert "candidate_1" in follow_up["answer"]
    assert FakeMatchTool.calls == 0


def test_run_recruiter_copilot_with_memory_compares_candidates_from_memory(monkeypatch) -> None:
    class FakeMatchTool:
        calls = 0

        @classmethod
        def invoke(cls, payload):
            cls.calls += 1
            return {"items": []}

    SESSION_STORE.clear("session-compare")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool)

    memory = SESSION_STORE.get_or_create("session-compare")
    memory.matching_completed = True
    memory.last_candidates = [
        {"candidate_id": "candidate_1", "baseline_rank_v3": 1, "baseline_score_v3": 0.82},
        {"candidate_id": "candidate_2", "baseline_rank_v3": 2, "baseline_score_v3": 0.75},
    ]
    memory.last_transferability = {
        "candidate_1": {"selected_source": "yaml", "yaml": {"gaps_bloquants": []}},
        "candidate_2": {"selected_source": "yaml", "yaml": {"gaps_bloquants": ["SQL"]}},
    }

    result = run_recruiter_copilot_with_memory("Compare le premier et le deuxième candidat", "session-compare")

    assert "Comparaison des deux premiers candidats" in result["answer"]
    assert "candidate_1" in result["answer"]
    assert "candidate_2" in result["answer"]
    assert FakeMatchTool.calls == 0


def test_run_recruiter_copilot_with_memory_resolves_best_candidate_gaps(monkeypatch) -> None:
    class FakeMatchTool:
        calls = 0

        @classmethod
        def invoke(cls, payload):
            cls.calls += 1
            return {"items": []}

    SESSION_STORE.clear("session-gaps")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool)

    memory = SESSION_STORE.get_or_create("session-gaps")
    memory.matching_completed = True
    memory.last_candidates = [
        {"candidate_id": "candidate_1", "baseline_rank_v3": 1, "baseline_score_v3": 0.82},
        {"candidate_id": "candidate_2", "baseline_rank_v3": 2, "baseline_score_v3": 0.75},
    ]
    memory.last_transferability = {
        "candidate_1": {"selected_source": "yaml", "yaml": {"gaps_bloquants": ["Docker"]}},
        "candidate_2": {"selected_source": "yaml", "yaml": {"gaps_bloquants": ["SQL"]}},
    }

    result = run_recruiter_copilot_with_memory("Quels sont les gaps du meilleur candidat ?", "session-gaps")

    assert result["selected_candidate_id"] == "candidate_1"
    assert "Docker" in result["answer"]
    assert FakeMatchTool.calls == 0


def test_run_recruiter_copilot_with_memory_handles_job_intake(monkeypatch) -> None:
    class FakeMatchTool:
        @staticmethod
        def invoke(payload):
            return {"items": [{"candidate_id": "candidate_1", "baseline_score_v3": 0.82}]}

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"]}

    class FakeTransferabilityTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "transferability": {}}

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {"available": False, "fallback_recommended": True}

    SESSION_STORE.clear("job-session")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool())
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool", FakeDecisionCardTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_transferability_tool", FakeTransferabilityTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool", FakeNeo4jTool())

    first = run_recruiter_copilot_with_memory("Je veux creer une nouvelle offre", "job-session")
    assert "titre du poste" in first["answer"]

    run_recruiter_copilot_with_memory("Backend Python Engineer", "job-session")
    run_recruiter_copilot_with_memory("We are looking for a backend engineer.", "job-session")
    run_recruiter_copilot_with_memory("You will design APIs.", "job-session")
    run_recruiter_copilot_with_memory("- Python\n- FastAPI\n- MongoDB", "job-session")
    run_recruiter_copilot_with_memory("- AWS", "job-session")
    summary = run_recruiter_copilot_with_memory("Mid-level with at least 3 years, Tunis, hybrid.", "job-session")

    assert "backend_python_fastapi_mongodb_aligned" in summary["answer"]
    assert "Voulez-vous lancer" in summary["answer"]

    matched = run_recruiter_copilot_with_memory("Oui lance la recherche", "job-session")
    assert matched["candidates"][0]["candidate_id"] == "candidate_1"
    assert matched["job_route"]["job_id"] == "backend_python_fastapi_mongodb_aligned"
