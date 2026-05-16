from __future__ import annotations

import json
from pathlib import Path


SCENARIOS_PATH = Path("data/evaluation/copilot_eval_scenarios.json")
SCRIPT_PATH = Path("scripts/evaluate_copilot.py")
REPORT_JSON = Path("docs/reports/copilot/copilot_evaluation.json")
REPORT_MD = Path("docs/reports/copilot/copilot_evaluation.md")


def test_copilot_eval_scenarios_exist() -> None:
    assert SCENARIOS_PATH.exists()
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 8
    assert all("message" in scenario for scenario in scenarios)
    assert all("intent" in scenario for scenario in scenarios)


def test_evaluate_copilot_script_exists() -> None:
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "run_recruiter_copilot" in content
    assert "scenario_score" in content


def test_copilot_evaluation_reports_are_generated() -> None:
    assert REPORT_JSON.exists()
    assert REPORT_MD.exists()


def test_copilot_evaluation_json_contains_scores() -> None:
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    assert "average_score" in report
    assert report["total_scenarios"] >= 8
    assert isinstance(report["results"], list)
    assert all("scenario_score" in result for result in report["results"])
    assert all(0 <= result["scenario_score"] <= 1 for result in report["results"])


def test_copilot_evaluation_markdown_is_readable() -> None:
    content = REPORT_MD.read_text(encoding="utf-8")

    assert "Évaluation du Recruiter Copilot" in content
    assert "Score global" in content
    assert "Conclusion" in content
