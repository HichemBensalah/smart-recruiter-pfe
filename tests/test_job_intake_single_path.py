from __future__ import annotations

from src.core.chatbot.graph import run_recruiter_copilot_with_memory
from src.core.chatbot.memory import SESSION_STORE


def _patch_live_matching(monkeypatch, items, recorder=None):
    """Intercept the hybrid/live matching path so _run_live_matching_path
    (src/core/chatbot/graph.py) returns deterministic candidates without MongoDB
    or FAISS. That path imports LiveMatcher and create_mongo_repositories locally,
    so patching the source-module attributes is the correct injection point.
    """
    from src.core.matching.live_matcher import LiveMatchResult

    class _FakeRepos:
        def close(self):
            pass

    class _FakeLiveMatcher:
        def __init__(self, repositories, settings, **kwargs):
            pass

        def match(self, *, job_description, job_id=None, top_k=None, structured_job_profile=None):
            if recorder is not None:
                recorder["calls"] = recorder.get("calls", 0) + 1
                recorder["job_id"] = job_id
                recorder["structured_job_profile"] = structured_job_profile
            return LiveMatchResult(
                job_id=job_id,
                resolved_job_id=(structured_job_profile or {}).get("routed_base_job_id") or job_id,
                top_k=top_k or len(items),
                items=[dict(it) for it in items],
                warnings=[],
                data_source="mongodb:test.candidate_profiles",
                retrieval_source="faiss:test",
                dedup_info={},
            )

    monkeypatch.setattr(
        "src.core.storage.repositories.create_mongo_repositories",
        lambda uri, database: _FakeRepos(),
    )
    monkeypatch.setattr("src.core.matching.live_matcher.LiveMatcher", _FakeLiveMatcher)
    return recorder


def test_new_session_starts_wizard_and_does_not_match(monkeypatch) -> None:
    class FailingMatchTool:
        @staticmethod
        def invoke(payload):
            raise AssertionError("matching must not run before confirmation")

    SESSION_STORE.clear("single-path-1")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FailingMatchTool())

    result = run_recruiter_copilot_with_memory("Je cherche un backend Python FastAPI MongoDB", "single-path-1")

    assert "titre exact du poste" in result["answer"]
    assert result["candidates"] == []
    assert result["matching_completed"] is False
    assert result["job_intake_state"]["current_step"] == "job_title"


def test_follow_up_before_matching_is_blocked() -> None:
    SESSION_STORE.clear("single-path-2")

    result = run_recruiter_copilot_with_memory("Pourquoi le premier candidat ?", "single-path-2")

    assert "Je dois d'abord terminer l'offre" in result["answer"]
    assert result["candidates"] == []
    assert result["matching_completed"] is False


def test_wizard_collects_offer_and_matches_after_confirmation(monkeypatch) -> None:
    class FakeMatchTool:
        calls = 0
        last_payload = None

        @classmethod
        def invoke(cls, payload):
            cls.calls += 1
            cls.last_payload = payload
            return {
                "items": [
                    {
                        "candidate_id": "candidate_1",
                        "candidate_name": "Aziz Ben Ali",
                        "baseline_rank_v3": 1,
                        "baseline_score_v3": 0.7754,
                    }
                ]
            }

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "candidate_name": "Aziz Ben Ali"}

    class FakeCandidateProfileTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "full_name": "Aziz Ben Ali"}

    class FakeTransferabilityTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"], "transferability": {"gaps_bloquants": ["SQL"]}}

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {"available": False, "fallback_recommended": True}

    SESSION_STORE.clear("single-path-3")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool)
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool", FakeDecisionCardTool())
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_candidate_profile_tool", FakeCandidateProfileTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_transferability_tool", FakeTransferabilityTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool", FakeNeo4jTool())
    # Hybrid is the production mode: matching after confirmation runs the live path.
    live = _patch_live_matching(
        monkeypatch,
        [{"candidate_id": "candidate_1", "full_name": "Aziz Ben Ali", "rank": 1, "final_score": 0.7754}],
        recorder={},
    )

    run_recruiter_copilot_with_memory("Backend Python Engineer", "single-path-3")
    run_recruiter_copilot_with_memory("We are looking for a backend engineer.", "single-path-3")
    run_recruiter_copilot_with_memory("You will design APIs.", "single-path-3")
    run_recruiter_copilot_with_memory("- Python\n- FastAPI\n- MongoDB", "single-path-3")
    run_recruiter_copilot_with_memory("- AWS", "single-path-3")
    summary = run_recruiter_copilot_with_memory("Mid-level with at least 3 years, Tunis, hybrid, English.", "single-path-3")

    assert summary["structured_job_profile"]["required_skills"] == ["Python", "FastAPI", "MongoDB"]
    assert summary["routed_job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert live.get("calls", 0) == 0

    matched = run_recruiter_copilot_with_memory("Oui lance la recherche", "single-path-3")

    assert live.get("calls", 0) == 1
    # In live mode the matcher receives the generated job id; the base routing is
    # carried on the generated profile's routed_base_job_id.
    assert live["structured_job_profile"]["routed_base_job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert matched["matching_completed"] is True
    assert matched["candidates"][0]["candidate_name"] == "Aziz Ben Ali"
    # The answer is a short summary referencing the generated job id; the base
    # routing is preserved on job_route.
    assert matched["answer"]
    assert matched["job_route"]["job_id"] == "backend_python_fastapi_mongodb_aligned"
    assert matched["routed_job_id"].startswith("generated_")


def test_follow_up_after_matching_uses_memory() -> None:
    memory = SESSION_STORE.get_or_create("single-path-4")
    memory.matching_completed = True
    memory.last_candidates = [{"candidate_id": "candidate_1", "candidate_name": "Aziz Ben Ali", "baseline_score_v3": 0.8}]
    memory.last_decision_cards = [{"candidate_id": "candidate_1", "candidate_name": "Aziz Ben Ali"}]
    memory.last_transferability = {"candidate_1": {"selected_source": "yaml", "yaml": {"gaps_bloquants": []}}}

    result = run_recruiter_copilot_with_memory("Pourquoi le premier candidat ?", "single-path-4")

    assert result["selected_candidate_id"] == "candidate_1"
    assert "Aziz Ben Ali" in result["answer"]
