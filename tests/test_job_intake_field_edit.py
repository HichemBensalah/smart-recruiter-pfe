from __future__ import annotations

from src.core.chatbot.graph import run_recruiter_copilot_with_memory
from src.core.chatbot.job_intake import (
    apply_field_edit,
    detect_field_edit_request,
    extract_field_edit_value,
    request_field_edit,
    start_job_intake,
    update_job_intake,
)
from src.core.chatbot.memory import ConversationMemory, SESSION_STORE


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


def test_detect_field_edit_request_detects_required_skills() -> None:
    assert detect_field_edit_request("modifie les compétences obligatoires") == "required_skills"
    assert detect_field_edit_request("change required skills") == "required_skills"


def test_detect_field_edit_request_detects_job_title() -> None:
    assert detect_field_edit_request("corrige le titre") == "job_title"
    assert detect_field_edit_request("change title") == "job_title"


def test_detect_field_edit_request_detects_accented_and_location_variants() -> None:
    assert detect_field_edit_request("modifie les compétences obligatoires") == "required_skills"
    assert detect_field_edit_request("modifie la localisation") == "profile"


def test_extract_field_edit_value_from_inline_request() -> None:
    assert (
        extract_field_edit_value("change les compétences obligatoires en Python, FastAPI, MongoDB", "required_skills")
        == "Python, FastAPI, MongoDB"
    )
    assert extract_field_edit_value("corrige le titre : Backend Python FastAPI Engineer", "job_title") == "Backend Python FastAPI Engineer"


def test_apply_field_edit_replaces_field_and_rebuilds_profile() -> None:
    memory = _completed_memory()
    before = memory.job_intake["structured_job_profile"]

    request_field_edit(memory, "required_skills")
    apply_field_edit(memory, "- Python\n- Django\n- PostgreSQL")
    after = memory.job_intake["structured_job_profile"]

    assert before["required_skills"] == ["Python", "FastAPI", "MongoDB"]
    assert after["required_skills"] == ["Python", "Django", "PostgreSQL"]
    assert memory.job_intake["route"]["job_id"] == "backend_python_django_postgresql"
    assert memory.pending_confirmation == "launch_matching"
    assert memory.awaiting_field_replacement is False


def test_conversation_field_edit_does_not_launch_matching_before_reconfirmation(monkeypatch) -> None:
    class FakeMatchTool:
        calls = 0

        @classmethod
        def invoke(cls, payload):
            cls.calls += 1
            return {"items": [{"candidate_id": "candidate_1"}]}

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload["candidate_id"]}

    class FakeCandidateProfileTool:
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

    SESSION_STORE.clear("edit-session")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool)
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool", FakeDecisionCardTool())
    monkeypatch.setattr("src.core.chatbot.nodes.fetch_decision_cards.get_candidate_profile_tool", FakeCandidateProfileTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_transferability_tool", FakeTransferabilityTool())
    monkeypatch.setattr("src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool", FakeNeo4jTool())
    # Hybrid is the production mode: matching after confirmation goes through the
    # live path. Count live matcher invocations rather than the artifact tool.
    live = _patch_live_matching(monkeypatch, [{"candidate_id": "candidate_1"}], recorder={})

    _complete_conversation("edit-session")
    edit_request = run_recruiter_copilot_with_memory("modifie les compétences obligatoires", "edit-session")

    assert edit_request["awaiting_field_replacement"] is True
    assert edit_request["pending_field_edit"] == "required_skills"
    assert live.get("calls", 0) == 0

    edited = run_recruiter_copilot_with_memory("- Python\n- Django\n- PostgreSQL", "edit-session")

    assert edited["structured_job_profile"]["required_skills"] == ["Python", "Django", "PostgreSQL"]
    assert edited["routed_job_id"] == "backend_python_django_postgresql"
    assert "Voulez-vous lancer" in edited["answer"]
    assert live.get("calls", 0) == 0

    matched = run_recruiter_copilot_with_memory("oui lance la recherche", "edit-session")

    assert matched["matching_completed"] is True
    assert live.get("calls", 0) == 1


def test_inline_field_edit_rebuilds_without_launching_matching(monkeypatch) -> None:
    class FakeMatchTool:
        calls = 0

        @classmethod
        def invoke(cls, payload):
            cls.calls += 1
            return {"items": [{"candidate_id": "candidate_1"}]}

    SESSION_STORE.clear("inline-edit-session")
    monkeypatch.setattr("src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool)

    _complete_conversation("inline-edit-session")
    edited = run_recruiter_copilot_with_memory(
        "change les compétences obligatoires en Python, Django, PostgreSQL",
        "inline-edit-session",
    )

    assert edited["structured_job_profile"]["required_skills"] == ["Python", "Django", "PostgreSQL"]
    assert edited["routed_job_id"] == "backend_python_django_postgresql"
    assert "Voulez-vous lancer" in edited["answer"]
    assert edited["matching_completed"] is False
    assert FakeMatchTool.calls == 0


def _completed_memory() -> ConversationMemory:
    memory = ConversationMemory(session_id="field-edit")
    start_job_intake(memory)
    update_job_intake(memory, "Backend Python Engineer")
    update_job_intake(memory, "We are looking for a backend engineer.")
    update_job_intake(memory, "You will design APIs.")
    update_job_intake(memory, "- Python\n- FastAPI\n- MongoDB")
    update_job_intake(memory, "- AWS")
    update_job_intake(memory, "Mid-level with at least 3 years, Tunis, hybrid.")
    return memory


def _complete_conversation(session_id: str) -> None:
    run_recruiter_copilot_with_memory("Backend Python Engineer", session_id)
    run_recruiter_copilot_with_memory("We are looking for a backend engineer.", session_id)
    run_recruiter_copilot_with_memory("You will design APIs.", session_id)
    run_recruiter_copilot_with_memory("- Python\n- FastAPI\n- MongoDB", session_id)
    run_recruiter_copilot_with_memory("- AWS", session_id)
    run_recruiter_copilot_with_memory("Mid-level with at least 3 years, Tunis, hybrid.", session_id)
