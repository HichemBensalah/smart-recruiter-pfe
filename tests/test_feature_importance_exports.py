from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_random_forest_feature_importance_is_generated_if_model_exists() -> None:
    rf_model = ROOT / "data/ranking/models/random_forest.joblib"
    assert rf_model.exists()
    subprocess.run(
        [
            sys.executable,
            "scripts/export_random_forest_feature_importance.py",
            "--model",
            "data/ranking/models/random_forest.joblib",
            "--feature-names",
            "data/ranking/models/feature_names.json",
            "--output-md",
            "docs/reports/ml/random_forest_feature_importance.md",
            "--output-json",
            "docs/reports/ml/random_forest_feature_importance.json",
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads((ROOT / "docs/reports/ml/random_forest_feature_importance.json").read_text(encoding="utf-8"))
    assert payload["top_features"]
    assert "pseudo-labels" in payload["methodology_warning"]


def test_xgboost_feature_importance_script_handles_missing_model(tmp_path: Path) -> None:
    output_md = tmp_path / "xgboost_feature_importance_missing_test.md"
    output_json = tmp_path / "xgboost_feature_importance_missing_test.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_xgboost_feature_importance.py",
            "--model",
            "data/ranking/models/does_not_exist_xgboost.joblib",
            "--feature-names",
            "data/ranking/models/feature_names.json",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "unavailable"


def test_xgboost_feature_importance_report_exists_only_if_model_exists() -> None:
    xgb_model = ROOT / "data/ranking/models/xgboost.joblib"
    xgb_report = ROOT / "docs/reports/ml/xgboost_feature_importance.json"
    if xgb_model.exists():
        assert xgb_report.exists()
        payload = json.loads(xgb_report.read_text(encoding="utf-8"))
        assert payload["top_features"]
    else:
        assert not xgb_report.exists() or json.loads(xgb_report.read_text(encoding="utf-8")).get("status") == "unavailable"
