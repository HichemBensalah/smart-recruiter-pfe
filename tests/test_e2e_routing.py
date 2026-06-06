# -*- coding: utf-8 -*-
"""
End-to-end routing test — Smart Recruiter job_router validation.

Real routing table derived from src/core/chatbot/job_router.py and data/job_profiles/:

  Signals                                          | job_id                              | conf
  ─────────────────────────────────────────────────┼─────────────────────────────────────┼─────
  python + fastapi + mongodb (required ∪ bonus)    | backend_python_fastapi_mongodb_aligned | 0.95
  django + postgresql (required ∪ bonus)           | backend_python_django_postgresql    | 0.90
  "data engineer" in title-text OR                 | data_engineer_python_sql_etl_aligned | 0.85
    ({"sql","etl"} & skills truthy AND python)     |                                     |
  "data analyst" in title-text OR                  | data_analyst_python_sql_powerbi     | 0.80
    powerbi/power bi/bi in skills                  |                                     |
  "machine learning" in title-text OR nlp in skills| machine_learning_python_nlp         | 0.80
  fallback (no rule matches)                       | backend_python_django_postgresql    | 0.45

Known routing bug (documented, not fixed here):
  `{"sql", "etl"} & skills` is truthy when only "sql" is in skills (no "etl").
  A Data Analyst profile with Python + SQL (no ETL) incorrectly routes to
  data_engineer_python_sql_etl_aligned.
  Fix: replace the condition with `{"sql", "etl"}.issubset(skills)`.
  Captured by test_route_data_analyst_with_python_routing_bug (marked xfail).

Architectural gap (reported, not fixed):
  job profiles frontend_react_nextjs, fullstack_react_node_mongodb, devops_docker_kubernetes
  exist in data/job_profiles/ but have no dedicated routing rule — they all fall back to
  backend_python_django_postgresql.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.chatbot.memory import SESSION_STORE


# ── Stable fake candidates (anti-hallucination anchor) ───────────────────────

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


# ── Live/hybrid matching stub ─────────────────────────────────────────────────

def _patch_live_matching(monkeypatch, items, recorder=None):
    """Intercept the hybrid/live matching path so _run_live_matching_path
    (src/core/chatbot/graph.py) returns deterministic candidates without MongoDB
    or FAISS. That path imports LiveMatcher and create_mongo_repositories locally,
    so patching the source-module attributes is the correct injection point in
    both hybrid and live modes.
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


# ── Tool stubs (same pattern as test_e2e_main_scenario.py) ───────────────────

def _patch_all_tools(monkeypatch) -> None:
    """Stub every external tool call. job_router is NOT mocked."""

    class FakeMatchTool:
        @staticmethod
        def invoke(payload):
            return {
                "items": _FAKE_CANDIDATES,
                "job_id": payload.get("job_id", "stub_job"),
                "resolved_job_id": payload.get("job_id", "stub_job"),
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
                    "gaps_compensables": [],
                },
            }

    class FakeNeo4jTool:
        @staticmethod
        def invoke(payload):
            return {"available": False, "fallback_recommended": True}

    monkeypatch.setattr(
        "src.core.chatbot.nodes.match_candidates.match_candidates_tool", FakeMatchTool()
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
    # Hybrid is the production mode: matching after confirmation goes through the
    # live path, not match_candidates_tool. Stub the live path too.
    _patch_live_matching(monkeypatch, _FAKE_CANDIDATES)


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

def _invented_ids(answer: str, valid_ids: set[str]) -> list[str]:
    found = re.findall(r"candidate_[A-Za-z0-9_]+", answer)
    return [t for t in found if t != "candidate_id" and t not in valid_ids]


# ── Wizard runner ─────────────────────────────────────────────────────────────

_FIELD_ORDER = [
    "job_title",
    "about_role",
    "responsibilities",
    "required_skills",
    "bonus_skills",
    "profile",
]


def _run_wizard(
    client: TestClient, starter: str, session_id: str, fields: dict
) -> dict:
    """
    Send wizard starter, then fill all 6 fields.
    Asserts matching_completed=False and candidates=[] at every intermediate step.
    Returns the response after the last (6th) field.
    """
    _chat(client, starter, session_id)
    r: dict = {}
    for key in _FIELD_ORDER:
        r = _chat(client, fields[key], session_id)
        assert r["matching_completed"] is False, (
            f"Matching fired early at field {key!r} — must wait for confirmation"
        )
        assert r["candidates"] == [], (
            f"Candidates appeared early at field {key!r} — must wait for matching"
        )
    return r


def _assert_routing(
    r_profile: dict,
    offer_type: str,
    expected_job_id: str,
    job_title: str,
    required_skills: str,
) -> None:
    """Assert structured_job_profile present and routed_job_id matches expected."""
    sjp = r_profile.get("structured_job_profile")
    assert sjp is not None, (
        f"SPEC VIOLATION [{offer_type}]: structured_job_profile must be present "
        f"in the response after the 6th wizard field"
    )
    actual_job_id = r_profile.get("routed_job_id")
    assert actual_job_id == expected_job_id, (
        f"\nROUTING MISMATCH\n"
        f"  offer_type      : {offer_type}\n"
        f"  expected_job_id : {expected_job_id}\n"
        f"  actual_job_id   : {actual_job_id}\n"
        f"  job_title       : {job_title}\n"
        f"  required_skills : {required_skills}"
    )


def _assert_post_confirmation(r_d: dict, offer_type: str) -> None:
    """Assert matching completed, candidates non-empty, no hallucinated ids."""
    assert r_d["matching_completed"] is True, (
        f"[{offer_type}] matching_completed must be True after confirmation"
    )
    assert r_d["candidates"], (
        f"[{offer_type}] candidates must not be empty after matching"
    )
    returned_ids = [c["candidate_id"] for c in r_d["candidates"] if c.get("candidate_id")]
    assert returned_ids, f"[{offer_type}] at least one candidate_id expected"
    for cid in returned_ids:
        assert cid in _FAKE_CANDIDATE_IDS, (
            f"[{offer_type}] Hallucinated candidate_id: {cid!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Test 1 — Backend FastAPI + MongoDB
# Routing rule: {"python", "fastapi", "mongodb"}.issubset(skills) → conf 0.95
# ═════════════════════════════════════════════════════════════════════════════

def test_route_backend_fastapi_mongodb(monkeypatch) -> None:
    """
    Signal  : {python, fastapi, mongodb} ⊆ (required_skills ∪ bonus_skills)
    Expected: backend_python_fastapi_mongodb_aligned
    File    : data/job_profiles/backend_python_fastapi_mongodb_aligned.json  ✓ exists
    """
    OFFER = "Backend FastAPI + MongoDB"
    SESSION = "e2e-route-fastapi-mongodb"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Backend Python Engineer",
        "about_role": (
            "We are building a high-performance recruitment platform. "
            "You will work on backend APIs consumed by HR teams and hiring managers."
        ),
        "responsibilities": (
            "Build and maintain REST APIs with FastAPI. "
            "Design and query data in MongoDB. "
            "Improve performance and reliability of backend services. "
            "Collaborate with frontend and product teams."
        ),
        "required_skills": "Python, FastAPI, MongoDB, Docker, REST API design",
        "bonus_skills": "CI/CD, AWS",
        "profile": "At least 3 years of experience, mid-level, English, Tunis, hybrid.",
    }

    r_profile = _run_wizard(
        client, "Je cherche un backend Python FastAPI MongoDB", SESSION, fields
    )
    _assert_routing(
        r_profile,
        OFFER,
        "backend_python_fastapi_mongodb_aligned",
        fields["job_title"],
        fields["required_skills"],
    )
    r_d = _chat(client, "Oui lance la recherche", SESSION)
    _assert_post_confirmation(r_d, OFFER)


# ═════════════════════════════════════════════════════════════════════════════
# Test 2 — Backend Django + PostgreSQL
# Routing rule: "django" in skills AND "postgresql" in skills → conf 0.90
# ═════════════════════════════════════════════════════════════════════════════

def test_route_backend_django_postgresql(monkeypatch) -> None:
    """
    Signal  : "django" ∈ skills AND "postgresql" ∈ skills
    Expected: backend_python_django_postgresql
    File    : data/job_profiles/backend_python_django_postgresql.json  ✓ exists
    """
    OFFER = "Backend Django + PostgreSQL"
    SESSION = "e2e-route-django-postgresql"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Backend Django Engineer",
        "about_role": (
            "We need a backend developer to build and maintain web APIs "
            "for a business application platform."
        ),
        "responsibilities": (
            "Develop and maintain REST APIs with Django REST Framework. "
            "Design and maintain relational schemas in PostgreSQL. "
            "Participate in code reviews and architecture decisions."
        ),
        "required_skills": "Python, Django, PostgreSQL, REST API, Docker",
        "bonus_skills": "Celery, Redis",
        "profile": "At least 3 years of experience, mid-level, French, Tunis, hybrid.",
    }

    r_profile = _run_wizard(
        client, "Je veux creer une offre pour un backend Django", SESSION, fields
    )
    _assert_routing(
        r_profile,
        OFFER,
        "backend_python_django_postgresql",
        fields["job_title"],
        fields["required_skills"],
    )
    r_d = _chat(client, "Oui lance la recherche", SESSION)
    _assert_post_confirmation(r_d, OFFER)


# ═════════════════════════════════════════════════════════════════════════════
# Test 3 — Data Engineer
# Routing rule: "data engineer" in title-text → conf 0.85
# (also covers: SQL + ETL + Python in skills — both signals active here)
# ═════════════════════════════════════════════════════════════════════════════

def test_route_data_engineer(monkeypatch) -> None:
    """
    Signal  : "data engineer" ∈ text (title = "Data Engineer")
              also: {"sql","etl"}.issubset(skills) AND "python" ∈ skills
    Expected: data_engineer_python_sql_etl_aligned
    File    : data/job_profiles/data_engineer_python_sql_etl_aligned.json  ✓ exists
    """
    OFFER = "Data Engineer"
    SESSION = "e2e-route-data-engineer"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Data Engineer",
        "about_role": (
            "We are looking for a Data Engineer to build and maintain our data infrastructure. "
            "You will design scalable ETL pipelines and data workflows."
        ),
        "responsibilities": (
            "Build ETL pipelines with Python and SQL. "
            "Process and transform large datasets. "
            "Optimize data workflows for reliability and performance. "
            "Collaborate with data scientists and analysts."
        ),
        "required_skills": "Python, SQL, ETL, data pipelines, Hadoop",
        "bonus_skills": "Spark, Airflow",
        "profile": "At least 3 years of experience, mid-level, English, Tunis, remote.",
    }

    r_profile = _run_wizard(
        client, "Je veux creer une offre pour un Data Engineer", SESSION, fields
    )
    _assert_routing(
        r_profile,
        OFFER,
        "data_engineer_python_sql_etl_aligned",
        fields["job_title"],
        fields["required_skills"],
    )
    r_d = _chat(client, "Oui lance la recherche", SESSION)
    _assert_post_confirmation(r_d, OFFER)


# ═════════════════════════════════════════════════════════════════════════════
# Test 4 — Data Analyst / BI  (title-based; Python excluded to avoid routing bug)
# Routing rule: "data analyst" in title-text → conf 0.80
# ═════════════════════════════════════════════════════════════════════════════

def test_route_data_analyst(monkeypatch) -> None:
    """
    Signal  : "data analyst" ∈ text (title = "Data Analyst")
    Expected: data_analyst_python_sql_powerbi
    File    : data/job_profiles/data_analyst_python_sql_powerbi.json  ✓ exists

    DESIGN NOTE — Python is intentionally excluded from required_skills and bonus_skills.
    If Python appears alongside SQL (without ETL), a routing bug in job_router.py causes
    Check 3 to fire: `{"sql","etl"} & skills` returns {"sql"} (truthy) even without ETL,
    combined with "python" in skills → incorrectly routes to data_engineer_python_sql_etl_aligned.
    The bug is documented and tested separately in test_route_data_analyst_with_python_routing_bug.
    """
    OFFER = "Data Analyst / BI"
    SESSION = "e2e-route-data-analyst"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Data Analyst",
        "about_role": (
            "We are looking for a Data Analyst to support our business teams "
            "with data-driven insights and reporting."
        ),
        "responsibilities": (
            "Analyze business data with SQL and BI tools. "
            "Build dashboards and reports with Power BI and Tableau. "
            "Present findings and recommendations to stakeholders."
        ),
        "required_skills": "SQL, Power BI, Tableau, Excel, data analysis",
        "bonus_skills": "Reporting, dashboards",
        "profile": "At least 2 years of experience, junior to mid-level, French, Tunis, onsite.",
    }

    r_profile = _run_wizard(
        client, "Je veux creer une offre pour un Data Analyst", SESSION, fields
    )
    _assert_routing(
        r_profile,
        OFFER,
        "data_analyst_python_sql_powerbi",
        fields["job_title"],
        fields["required_skills"],
    )
    r_d = _chat(client, "Oui lance la recherche", SESSION)
    _assert_post_confirmation(r_d, OFFER)


# ═════════════════════════════════════════════════════════════════════════════
# Test 5 — Machine Learning / NLP
# Routing rule: "machine learning" in title-text OR "nlp" in skills → conf 0.80
# ═════════════════════════════════════════════════════════════════════════════

def test_route_machine_learning(monkeypatch) -> None:
    """
    Signal  : "machine learning" ∈ text (title starts with "Machine Learning")
              also: "nlp" ∈ skills
    Expected: machine_learning_python_nlp
    File    : data/job_profiles/machine_learning_python_nlp.json  ✓ exists
    """
    OFFER = "Machine Learning / NLP"
    SESSION = "e2e-route-ml-engineer"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Machine Learning Engineer",
        "about_role": (
            "We are building ML-powered features for our recruitment platform. "
            "You will develop and evaluate models for matching and classification."
        ),
        "responsibilities": (
            "Develop machine learning models with Scikit-learn and TensorFlow. "
            "Work on NLP pipelines for text classification and entity extraction. "
            "Evaluate model performance and deploy models to production."
        ),
        "required_skills": "Python, Machine Learning, NLP, Scikit-learn, TensorFlow",
        "bonus_skills": "PyTorch, model deployment",
        "profile": "At least 3 years of experience, mid-level, English, Tunis, remote.",
    }

    r_profile = _run_wizard(
        client, "Je veux creer une offre Machine Learning", SESSION, fields
    )
    _assert_routing(
        r_profile,
        OFFER,
        "machine_learning_python_nlp",
        fields["job_title"],
        fields["required_skills"],
    )
    r_d = _chat(client, "Oui lance la recherche", SESSION)
    _assert_post_confirmation(r_d, OFFER)


# ═════════════════════════════════════════════════════════════════════════════
# Test 5b - Machine Learning Intern real offer
# ═════════════════════════════════════════════════════════════════════════════

def test_route_ml_intern_real_offer(monkeypatch) -> None:
    """
    Real interface regression:
    this ML internship contains FastAPI and MongoDB, but the ML intent must win
    before backend skill routing.
    """
    OFFER = "Machine Learning Engineer Intern"
    SESSION = "e2e-route-ml-intern-real-offer"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Machine Learning Engineer Intern",
        "about_role": (
            "We are looking for a Machine Learning Engineer Intern to build, "
            "train and deploy machine learning and deep learning models."
        ),
        "responsibilities": (
            "You will preprocess datasets and build ML training pipelines. "
            "You will train classification, anomaly detection and deep learning models. "
            "You will evaluate models and generate experiment reports. "
            "You will deploy models using FastAPI or Flask. "
            "You will use Docker, GitHub Actions and MLflow for MLOps workflows."
        ),
        "required_skills": (
            "Python, Machine Learning, Deep Learning, TensorFlow, Keras, FastAPI, "
            "Docker, GitHub Actions, MLflow, AWS"
        ),
        "bonus_skills": (
            "PyTorch, NLP, Computer Vision, Streamlit, PySpark, MongoDB, "
            "PostgreSQL, Power BI"
        ),
        "profile": (
            "Final-year Data Science and AI Engineering student, junior internship "
            "or PFE role, Tunis, hybrid, English and French."
        ),
    }

    r_profile = _run_wizard(
        client, "Je veux creer une offre Machine Learning Engineer Intern", SESSION, fields
    )
    _assert_routing(
        r_profile,
        OFFER,
        "machine_learning_python_nlp",
        fields["job_title"],
        fields["required_skills"],
    )
    assert r_profile.get("routed_job_id") != "backend_python_fastapi_mongodb_aligned"


# Test 6 - Fallback (no routing rule matches)

def test_route_fallback_unknown_role(monkeypatch) -> None:
    """
    Signal  : none of the 5 routing rules matches
    Expected: backend_python_django_postgresql  (fallback, confidence 0.45)
    Verifies: no crash, routed_job_id non-empty, matching only after confirmation.
    """
    OFFER = "Quantum Banana Architect (fallback)"
    SESSION = "e2e-route-fallback"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Quantum Banana Architect",
        "about_role": (
            "We design quantum banana deployment pipelines for enterprise-grade "
            "banana peeling operations at global scale."
        ),
        "responsibilities": (
            "Peel bananas at quantum scale. "
            "Quantize banana models and deploy them to production. "
            "Orchestrate banana orchestration layers."
        ),
        "required_skills": "Quantum Banana SDK, banana orchestration, quantum computing",
        "bonus_skills": "Docker",
        "profile": "At least 2 years of experience, junior, English, Tunis, remote.",
    }

    r_profile = _run_wizard(
        client,
        "Je veux creer une offre pour un Quantum Banana Architect",
        SESSION,
        fields,
    )

    sjp = r_profile.get("structured_job_profile")
    assert sjp is not None, (
        f"[{OFFER}] structured_job_profile must be present even for fallback routing"
    )

    fallback_id = r_profile.get("routed_job_id")
    assert fallback_id is not None and fallback_id, (
        f"[{OFFER}] routed_job_id must be non-empty even when no routing rule matches"
    )
    assert fallback_id == "backend_python_django_postgresql", (
        f"\nROUTING MISMATCH [{OFFER}]\n"
        f"  expected_job_id : backend_python_django_postgresql (fallback)\n"
        f"  actual_job_id   : {fallback_id!r}\n"
        f"  job_title       : {fields['job_title']}\n"
        f"  required_skills : {fields['required_skills']}"
    )

    r_d = _chat(client, "Oui lance la recherche", SESSION)
    _assert_post_confirmation(r_d, OFFER)


# ═════════════════════════════════════════════════════════════════════════════
# Test 7 — Data Analyst with Python + SQL routes correctly (was xfail, bug fixed)
# Bug was: {"sql","etl"} & skills truthy with only "sql" — fixed by .issubset()
# ═════════════════════════════════════════════════════════════════════════════

def test_route_data_analyst_with_python_routes_to_data_analyst(monkeypatch) -> None:
    """
    Signal  : "data analyst" ∈ text (title = "Data Analyst")
    Skills  : SQL + Power BI + Python + Tableau (realistic Data Analyst profile)
    Expected: data_analyst_python_sql_powerbi

    Previously marked xfail due to a routing bug in job_router.py line 19:
      `{"sql", "etl"} & skills and "python" in skills` was truthy with only "sql"
      present (no "etl" required), causing Python+SQL Data Analyst offers to route
      incorrectly to data_engineer_python_sql_etl_aligned.
    Fix applied: replaced with `{"sql", "etl"}.issubset(skills) and "python" in skills`.
    """
    OFFER = "Data Analyst with Python"
    SESSION = "e2e-route-data-analyst-with-python"
    SESSION_STORE.clear(SESSION)
    _patch_all_tools(monkeypatch)
    client = TestClient(app)

    fields = {
        "job_title": "Data Analyst",
        "about_role": (
            "We need a Data Analyst to derive business insights from our data warehouse "
            "using Python and SQL."
        ),
        "responsibilities": (
            "Analyze data with SQL and Python. "
            "Build dashboards with Power BI. "
            "Report findings to business stakeholders."
        ),
        "required_skills": "SQL, Power BI, Python, Tableau, reporting",
        "bonus_skills": "Excel, dashboards",
        "profile": "At least 2 years of experience, junior, French, Tunis, onsite.",
    }

    r_profile = _run_wizard(
        client, "Je veux creer une offre pour un Data Analyst", SESSION, fields
    )

    _assert_routing(
        r_profile,
        OFFER,
        "data_analyst_python_sql_powerbi",
        fields["job_title"],
        fields["required_skills"],
    )
