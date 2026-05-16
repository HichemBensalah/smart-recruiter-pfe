from src.core.chatbot.graph import build_recruiter_copilot_graph, run_recruiter_copilot


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
