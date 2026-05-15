from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ml_experiment_interpretation_report_exists_and_documents_limits() -> None:
    md_path = ROOT / "docs/reports/ml/ml_experiment_interpretation.md"
    json_path = ROOT / "docs/reports/ml/ml_experiment_interpretation.json"
    assert md_path.exists()
    assert json_path.exists()

    text = md_path.read_text(encoding="utf-8").lower()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "pseudo-label" in text
    assert "circularité" in text
    assert "matching v3" in text
    assert "baseline officielle" in text
    assert "production-ready" not in text
    assert payload["target"] == "label_binary"
    assert "why_metrics_are_high" in payload


def test_interpretation_generator_runs() -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_ml_experiment_interpretation.py",
            "--training-report",
            "data/ranking/models/training_report.json",
            "--output-md",
            "docs/reports/ml/ml_experiment_interpretation.md",
            "--output-json",
            "docs/reports/ml/ml_experiment_interpretation.json",
        ],
        cwd=ROOT,
        check=True,
    )
