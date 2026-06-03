# -*- coding: utf-8 -*-
"""
End-to-end test — Smart Recruiter main product scenario.

Full path:
  guided offer creation → structured_job_profile → routing job_id
  → recruiter confirmation → matching → candidate display → follow-up questions.

All tools that make HTTP or DB calls are stubbed.
No live server, MongoDB, Neo4j, FAISS, or LLM required.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.chatbot.memory import SESSION_STORE


# ── Stable fake candidates (anchor for anti-hallucination checks) ─────────────

_FAKE_CANDIDATES = [
    {
        "candidate_id": "candidate_1",
        "baseline_rank_v3": 1,
        "baseline_score_v3": 0.87,
        "rf_score": 0.82,
        "xgboost_score": 0.85,
        "recommendation_status": "agreement_high",
    },
    {
        "candidate_id": "candidate_2",
        "baseline_rank_v3": 2,
        "baseline_score_v3": 0.75,
        "rf_score": 0.70,
        "xgboost_score": 0.72,
        "recommendation_status": "agreement_high",
    },
]

_FAKE_CANDIDATE_IDS: set[str] = {c["candidate_id"] for c in _FAKE_CANDIDATES}


# ── Tool stubs ────────────────────────────────────────────────────────────────

def _patch_all_tools(monkeypatch) -> None:
    """Stub every tool that makes HTTP or external store calls."""

    class FakeMatchTool:
        @staticmethod
        def invoke(payload):
            return {
                "items": _FAKE_CANDIDATES,
                "job_id": "backend_python_fastapi_mongodb_aligned",
                "resolved_job_id": "backend_python_fastapi_mongodb_aligned",
                "matching_mode": "artifact",
                "artifact_source": "yaml",
                "fallback_used": False,
                "warnings": [],
            }

    class FakeDecisionCardTool:
        @staticmethod
        def invoke(payload):
            return {
                "candidate_id": payload.get("candidate_id", ""),
                "recommendation_status": "agreement_high",
            }

    class FakeCandidateProfileTool:
        @staticmethod
        def invoke(payload):
            return {"candidate_id": payload.get("candidate_id", "")}

    class FakeTransferabilityTool:
        @staticmethod
        def invoke(payload):
            return {
                "candidate_id": payload.get("candidate_id"),
                "transferability": {
                    "transferability_score": 0.65,
                    "gaps_bloquants": [],
                    "gaps_compensables": ["CI/CD"],
                },
            }

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {"available": False, "fallback_recommended": True}

    monkeypatch.setattr(
        "src.core.chatbot.nodes.match_candidates.match_candidates_tool",
        FakeMatchTool(),
    )
    monkeypatch.setattr(
        "src.core.chatbot.nodes.fetch_decision_cards.get_decision_card_tool",
        FakeDecisionCardTool(),
    )
    monkeypatch.setattr(
        "src.core.chatbot.nodes.fetch_decision_cards.get_candidate_profile_tool",
        FakeCandidateProfileTool(),
    )
    monkeypatch.setattr(
        "src.core.chatbot.nodes.analyze_transferability.get_transferability_tool",
        FakeTransferabilityTool(),
    )
    monkeypatch.setattr(
        "src.core.chatbot.nodes.analyze_transferability.get_neo4j_transferability_tool",
        FakeNeo4jTool(),
    )


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _chat(client: TestClient, message: str, session_id: str) -> dict:
    resp = client.post(
        "/api/chat",
        json={"message": message, "session_id": session_id},
    )
    assert resp.status_code == 200, (
        f"HTTP {resp.status_code} on message {message!r}\n{resp.text}"
    )
    return resp.json()


# ── Anti-hallucination helper ─────────────────────────────────────────────────

def _invented_candidate_ids(answer: str, valid_ids: set[str]) -> list[str]:
    """Return any candidate_... token in *answer* that is not a field label and not in *valid_ids*."""
    found = re.findall(r"candidate_[A-Za-z0-9_]+", answer)
    # "candidate_id" is a field label emitted by compose_answer nodes, not an id value
    _KNOWN_FIELD_LABELS = {"candidate_id"}
    return [t for t in found if t not in _KNOWN_FIELD_LABELS and t not in valid_ids]


# ═════════════════════════════════════════════════════════════════════════════
# Scenarios A–E: full main path (shared session)
# ═════════════════════════════════════════════════════════════════════════════

_SESSION_MAIN = "e2e-main-scenario-20240101"


def test_e2e_main_scenario(monkeypatch) -> None:
    """
    Covers:
      A. Start wizard
      B. Fill 6 fields (job_title → about_role → responsibilities →
                        required_skills → bonus_skills → profile)
      C. Verify structured_job_profile and routed_job_id after last field
      D. Confirm and launch matching
      E. Follow-up questions using session memory
    """
    _patch_all_tools(monkeypatch)
    SESSION_STORE.clear(_SESSION_MAIN)
    client = TestClient(app)

    # ── A. Start wizard ────────────────────────────────────────────────────
    r_a = _chat(client, "Je cherche un backend Python FastAPI MongoDB", _SESSION_MAIN)

    assert r_a["session_id"] == _SESSION_MAIN, "session_id must be preserved"
    assert r_a["matching_completed"] is False, "matching must not start before wizard"
    assert r_a["candidates"] == [], "candidates must be empty before wizard completion"
    assert r_a["answer"], "answer must not be empty"

    # Wizard starts — asks for job title
    answer_a = r_a["answer"].lower()
    assert "titre" in answer_a or "poste" in answer_a, (
        f"Expected wizard to ask for job title, got: {r_a['answer']!r}"
    )

    # ── B. Fill all 6 fields (one message per field) ───────────────────────
    _FIELD_MESSAGES = [
        # job_title
        "Backend Python Engineer",
        # about_role
        (
            "We are looking for a Backend Python Engineer to join our product engineering "
            "team working on a recruitment platform used by HR teams and hiring managers."
        ),
        # responsibilities
        (
            "Design, build and maintain REST APIs. Develop backend services with Python "
            "and FastAPI. Model and query application data in MongoDB. Collaborate with "
            "frontend, product and QA. Improve performance, monitoring and deployment."
        ),
        # required_skills
        "Python, FastAPI, MongoDB, Docker, REST API design",
        # bonus_skills
        "CI/CD, AWS",
        # profile (last field)
        "At least 3 years of experience, mid-level, English, Tunis, hybrid.",
    ]

    r_step = None
    for i, msg in enumerate(_FIELD_MESSAGES):
        r_step = _chat(client, msg, _SESSION_MAIN)
        assert r_step["matching_completed"] is False, (
            f"Matching must not launch before confirmation (field step {i + 1}/6)"
        )
        assert r_step["candidates"] == [], (
            f"Candidates must be empty before confirmation (field step {i + 1}/6)"
        )

    r_profile = r_step  # response after filling the last field (profile)
    assert r_profile is not None

    # ── C. After last field: MANDATORY assertions — structured profile & routing
    # Per spec: both fields MUST be populated after the 6th wizard field.
    # If either assertion fails, it is a regression in the applicative code
    # (job_intake.build_structured_job_profile / _job_intake_response), not in the test.
    sjp = r_profile.get("structured_job_profile")
    assert sjp is not None, (
        "SPEC VIOLATION: structured_job_profile must be present in the response after "
        "the 6th wizard field — check _job_intake_response / build_structured_job_profile"
    )
    assert isinstance(sjp, dict), "structured_job_profile must be a dict"
    skills_lower = [s.lower() for s in (sjp.get("required_skills") or [])]
    assert any("python" in s for s in skills_lower), (
        f"required_skills must contain Python: {sjp.get('required_skills')}"
    )
    assert any("fastapi" in s for s in skills_lower), (
        f"required_skills must contain FastAPI: {sjp.get('required_skills')}"
    )
    assert any("mongodb" in s for s in skills_lower), (
        f"required_skills must contain MongoDB: {sjp.get('required_skills')}"
    )
    assert sjp.get("min_years_experience") == 3, (
        f"min_years_experience must be 3, got {sjp.get('min_years_experience')}"
    )
    assert sjp.get("seniority") == "mid-level", (
        f"seniority must be 'mid-level', got {sjp.get('seniority')}"
    )

    routed_id_c = r_profile.get("routed_job_id")
    assert routed_id_c is not None, (
        "SPEC VIOLATION: routed_job_id must be present in the response after the 6th "
        "wizard field — check _job_intake_response / infer_job_route_from_structured_profile"
    )
    assert isinstance(routed_id_c, str) and routed_id_c, (
        "routed_job_id must be a non-empty string"
    )

    # Answer must ask for confirmation before launching matching
    answer_c = r_profile["answer"].lower()
    asks_confirmation = (
        "voulez" in answer_c
        or "lancer" in answer_c
        or "confirmation" in answer_c
        or "confirmer" in answer_c
    )
    assert asks_confirmation, (
        f"Expected confirmation prompt after wizard, got: {r_profile['answer'][:300]!r}"
    )

    # ── D. Confirmation → matching launched ────────────────────────────────
    r_d = _chat(client, "Oui lance la recherche", _SESSION_MAIN)

    assert r_d["matching_completed"] is True, (
        "matching_completed must be True after positive confirmation"
    )
    assert r_d["candidates"], "candidates must be non-empty after matching"

    returned_ids = [c["candidate_id"] for c in r_d["candidates"] if c.get("candidate_id")]
    assert returned_ids, "At least one candidate must have a candidate_id"

    # Anti-hallucination: every returned id must come from the fake fixture
    for cid in returned_ids:
        assert cid in _FAKE_CANDIDATE_IDS, (
            f"Candidate id '{cid}' was not provided by the fake tool — possible hallucination"
        )

    routed_id_d = r_d.get("routed_job_id")
    if routed_id_d is not None:
        assert isinstance(routed_id_d, str) and routed_id_d

    # ── E1. Follow-up: explain first candidate ─────────────────────────────
    r_e1 = _chat(client, "Pourquoi le premier candidat ?", _SESSION_MAIN)

    assert r_e1["answer"], "Follow-up answer must not be empty"
    assert r_e1["matching_completed"] is True, (
        "matching_completed must remain True during follow-up"
    )
    assert "candidate_1" in r_e1["answer"], (
        f"Expected 'candidate_1' referenced in explain answer, got: {r_e1['answer'][:300]!r}"
    )
    # Anti-hallucination
    hallucinated_e1 = _invented_candidate_ids(r_e1["answer"], _FAKE_CANDIDATE_IDS)
    assert not hallucinated_e1, (
        f"Hallucinated candidate_ids in explain answer: {hallucinated_e1}"
    )

    # ── E2. Follow-up: gap analysis ────────────────────────────────────────
    r_e2 = _chat(client, "Quels sont ses gaps ?", _SESSION_MAIN)

    assert r_e2["answer"], "Gap analysis answer must not be empty"
    assert r_e2["matching_completed"] is True, (
        "matching_completed must remain True during gap follow-up"
    )
    # Anti-hallucination
    hallucinated_e2 = _invented_candidate_ids(r_e2["answer"], _FAKE_CANDIDATE_IDS)
    assert not hallucinated_e2, (
        f"Hallucinated candidate_ids in gap analysis answer: {hallucinated_e2}"
    )
    # Answer mentions gaps or states data unavailability — both are valid
    answer_e2 = r_e2["answer"].lower()
    gap_content_present = (
        "gap" in answer_e2
        or "ci/cd" in answer_e2
        or "information non disponible" in answer_e2
        or "candidat" in answer_e2
    )
    assert gap_content_present, (
        f"Expected gap-related content in answer, got: {r_e2['answer'][:300]!r}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scenario F: follow-up BEFORE matching (fresh session)
# ═════════════════════════════════════════════════════════════════════════════

_SESSION_PRE_MATCH = "e2e-pre-match-followup-20240101"


def test_e2e_followup_before_matching(monkeypatch) -> None:
    """
    Scenario F — asking 'Pourquoi le premier candidat ?' on a brand-new session
    must be cleanly blocked: no crash, no candidates invented, clear guidance given.
    """
    _patch_all_tools(monkeypatch)
    SESSION_STORE.clear(_SESSION_PRE_MATCH)
    client = TestClient(app)

    r = _chat(client, "Pourquoi le premier candidat ?", _SESSION_PRE_MATCH)

    assert r["session_id"] == _SESSION_PRE_MATCH
    assert r["candidates"] == [], "No candidates must be returned before any matching"
    assert r["matching_completed"] is False
    assert r["answer"], "Answer must not be empty"

    # No hallucinated candidate ids
    hallucinated = _invented_candidate_ids(r["answer"], set())
    assert not hallucinated, (
        f"Follow-up before matching hallucinated candidate_ids: {hallucinated}"
    )

    # Answer must guide the recruiter to complete the offer first
    answer_lower = r["answer"].lower()
    informative = (
        "offre" in answer_lower
        or "recherche" in answer_lower
        or "terminer" in answer_lower
        or "lancer" in answer_lower
        or "titre" in answer_lower
        or "poste" in answer_lower
    )
    assert informative, (
        f"Expected answer to indicate offer/matching required first, got: {r['answer'][:300]!r}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scenario G: fallback routing (fresh session, exotic/unknown role)
# ═════════════════════════════════════════════════════════════════════════════

_SESSION_FALLBACK = "e2e-fallback-routing-20240101"


def test_e2e_fallback_routing(monkeypatch) -> None:
    """
    Scenario G — wizard filled with an exotic job title that matches no known
    routing rule.  Verifies: no crash, clean fallback job_id, matching only
    after confirmation, no hallucinated candidates.
    """
    _patch_all_tools(monkeypatch)
    SESSION_STORE.clear(_SESSION_FALLBACK)
    client = TestClient(app)

    # Start — exotic role that won't match any routing rule
    r0 = _chat(
        client,
        "Je veux creer une offre pour un poste Quantum Banana Architect",
        _SESSION_FALLBACK,
    )
    assert r0["candidates"] == []
    assert r0["matching_completed"] is False

    # Fill all 6 fields with exotic but valid content
    _EXOTIC_FIELDS = [
        "Quantum Banana Architect",
        (
            "We design quantum banana deployment pipelines for enterprise-grade "
            "banana peeling operations at scale."
        ),
        (
            "Peel bananas at quantum scale. Quantize banana models. "
            "Deploy banana pipelines to production."
        ),
        "Quantum Banana SDK, Python",
        "Docker",
        "At least 2 years of experience, junior, English, Tunis, remote.",
    ]

    r_last = None
    for msg in _EXOTIC_FIELDS:
        r_last = _chat(client, msg, _SESSION_FALLBACK)
        assert r_last["matching_completed"] is False, (
            "Matching must not fire before confirmation even with exotic role"
        )
        assert r_last["candidates"] == []

    assert r_last is not None

    # Fallback routed_job_id must be a non-empty string (may be the demo fallback)
    fallback_id = r_last.get("routed_job_id")
    if fallback_id is not None:
        assert isinstance(fallback_id, str) and fallback_id, (
            "Fallback routed_job_id must be a non-empty string"
        )

    # Confirm matching
    r_confirm = _chat(client, "Oui lance la recherche", _SESSION_FALLBACK)

    assert r_confirm["matching_completed"] is True, (
        "matching_completed must be True after confirmation even with fallback routing"
    )
    assert r_confirm["candidates"], "Matching must return candidates with fallback routing"

    # Anti-hallucination: all candidate_ids must come from the fake fixture
    candidate_ids = [
        c["candidate_id"] for c in r_confirm["candidates"] if c.get("candidate_id")
    ]
    for cid in candidate_ids:
        assert cid in _FAKE_CANDIDATE_IDS, (
            f"Hallucinated candidate_id in fallback scenario: {cid}"
        )

    confirmed_routed_id = r_confirm.get("routed_job_id")
    if confirmed_routed_id is not None:
        assert isinstance(confirmed_routed_id, str) and confirmed_routed_id
