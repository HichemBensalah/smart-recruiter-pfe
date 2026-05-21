from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCENARIOS_PATH = Path("data/evaluation/copilot_eval_scenarios.json")
SCRIPT_PATH = Path("scripts/evaluate_copilot.py")
REPORT_JSON = Path("docs/reports/copilot/copilot_evaluation.json")
REPORT_MD = Path("docs/reports/copilot/copilot_evaluation.md")


def test_copilot_eval_scenarios_exist_and_cover_phase_9_requirements() -> None:
    assert SCENARIOS_PATH.exists()
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    assert isinstance(scenarios, list)
    assert len(scenarios) >= 10
    ids = {scenario["id"] for scenario in scenarios}
    coverage = {tag for scenario in scenarios for tag in scenario.get("coverage", [])}

    assert "01_start_new_offer" in ids
    assert "09_confirm_matching" in ids
    assert "10_followup_explain_first" in ids
    assert "11_followup_compare_first_two" in ids
    assert "12_followup_best_candidate_gaps" in ids
    assert "13_graph_yaml_fallback_without_neo4j" in ids
    assert "14_match_unknown_job_id_fallback" in ids
    assert "new_offer_start" in coverage
    assert "six_field_collection" in coverage
    assert "pre_confirmation_correction" in coverage
    assert "matching_with_routed_job_id" in coverage
    assert "neo4j_yaml_fallback" in coverage


def test_evaluate_copilot_script_has_default_outputs() -> None:
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "DEFAULT_SCENARIOS" in content
    assert "TestClient" in content
    assert "tool_calling_accuracy" in content
    assert "average_chat_latency_ms" in content
    assert "memory_coherence_rate" in content


def test_evaluate_copilot_script_produces_valid_json(tmp_path) -> None:
    output_json = tmp_path / "copilot_evaluation.json"
    output_md = tmp_path / "copilot_evaluation.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--scenarios",
            str(SCENARIOS_PATH),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Copilot evaluation average_score" in completed.stdout
    assert output_json.exists()
    assert output_md.exists()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["total_scenarios"] >= 10
    assert "tool_calling_accuracy" in report
    assert "hallucination_free_rate" in report
    assert "average_chat_latency_ms" in report
    assert "memory_coherence_rate" in report
    assert "scenario_coverage" in report
    assert "fallback_coverage" in report


def test_copilot_evaluation_reports_are_generated() -> None:
    assert REPORT_JSON.exists()
    assert REPORT_MD.exists()


def test_copilot_evaluation_json_contains_required_metrics() -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    assert report["total_scenarios"] >= 10
    assert report["passed_scenarios"] + report["failed_scenarios"] == report["total_scenarios"]
    assert 0 <= report["average_score"] <= 1
    assert 0 <= report["tool_calling_accuracy"] <= 1
    assert 0 <= report["hallucination_free_rate"] <= 1
    assert report["average_chat_latency_ms"] is None or report["average_chat_latency_ms"] >= 0
    assert 0 <= report["memory_coherence_rate"] <= 1
    assert report["scenario_coverage"]["coverage_rate"] >= 0.9
    assert report["fallback_coverage"]["total"] >= 1
    assert isinstance(report["known_limitations"], list)


def test_copilot_evaluation_markdown_is_readable() -> None:
    content = REPORT_MD.read_text(encoding="utf-8")

    assert "Évaluation du Recruiter Copilot" in content
    assert "Métriques globales" in content
    assert "Tool calling accuracy" in content
    assert "Limites connues" in content
    assert "Conclusion" in content
