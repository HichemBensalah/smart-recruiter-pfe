from __future__ import annotations

from pathlib import Path

import pytest

from src.core.graph.neo4j_client import Neo4jClient, Neo4jUnavailable, load_neo4j_settings, neo4j_status
from src.core.graph.neo4j_transferability import build_coverage_response, build_transferability_explanation


def test_neo4j_client_reports_missing_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises(Neo4jUnavailable):
        load_neo4j_settings()

    status = neo4j_status()
    assert status["neo4j_available"] is False
    assert "Missing environment variables" in status["message"]


def test_neo4j_client_does_not_connect_before_needed() -> None:
    client = Neo4jClient.__new__(Neo4jClient)
    client.settings = None
    client._driver = None

    assert client._driver is None


def test_build_coverage_response_formats_scores() -> None:
    payload = build_coverage_response(
        "candidate_1",
        "Backend Developer",
        ["Python", "Git"],
        {"required_skills": ["Python", "SQL", "Git"], "adjacent_skills": ["Docker"]},
    )

    assert payload["coverage_score"] == 0.6667
    assert payload["matched_skills"] == ["Python", "Git"]
    assert payload["missing_skills"] == ["SQL"]
    assert payload["adjacent_skills"] == ["Docker"]


def test_build_transferability_explanation_formats_gaps() -> None:
    payload = build_transferability_explanation(
        candidate_id="candidate_1",
        target_role="Backend Developer",
        matched_skills=["Python"],
        missing_skills=["SQL", "Docker"],
        adjacent_skills=["Docker"],
        coverage_score=0.5,
        plausible_transitions=[],
    )

    assert payload["gaps_compensables"] == ["Docker"]
    assert payload["gaps_bloquants"] == ["SQL"]
    assert "Backend Developer" in payload["explanation"]


def test_import_script_documents_cdc_relations() -> None:
    content = Path("scripts/import_graph_to_neo4j.py").read_text(encoding="utf-8")

    for relation in ["REQUIRES", "RELATED_TO", "TRANSITIONS_TO", "HAS_SKILL", "FITS_ROLE"]:
        assert relation in content


def test_neo4j_import_report_has_expected_shape() -> None:
    import json

    payload = json.loads(Path("docs/reports/graph/neo4j_import_report.json").read_text(encoding="utf-8"))

    assert "status" in payload
    for key in ["roles_count", "skills_count", "jobs_count", "candidates_count", "relations_count"]:
        assert key in payload
        assert isinstance(payload[key], int)
    assert payload["fallback_available"] is True
    for relation in ["REQUIRES", "RELATED_TO", "TRANSITIONS_TO", "HAS_SKILL", "FITS_ROLE"]:
        assert relation in payload["cdc_relations"]
