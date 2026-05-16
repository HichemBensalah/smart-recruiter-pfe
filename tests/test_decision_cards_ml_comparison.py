from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FEATURES = ROOT / "data/ranking/features/backend_python_django_postgresql.jsonl"
RF_MODEL = ROOT / "data/ranking/models/random_forest.joblib"
FEATURE_NAMES = ROOT / "data/ranking/models/feature_names.json"
XGB_RANKING = ROOT / "docs/reports/ml/xgboost_primary_ranking.json"
SHAP_GLOBAL = ROOT / "docs/reports/ml/shap/shap_global_summary.json"
OFFICIAL_DECISION_CARDS = ROOT / "docs/reports/matching/v3/decision_cards_v3_normalized.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_decision_cards_ml_comparison_is_generated_without_modifying_inputs(tmp_path: Path) -> None:
    features_before = _sha256(FEATURES)
    official_cards_before = _sha256(OFFICIAL_DECISION_CARDS)
    output_json = tmp_path / "decision_cards_ml_comparison.json"
    output_md = tmp_path / "decision_cards_ml_comparison.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_decision_cards_ml_comparison.py",
            "--features",
            str(FEATURES),
            "--rf-model",
            str(RF_MODEL),
            "--feature-names",
            str(FEATURE_NAMES),
            "--xgb-ranking",
            str(XGB_RANKING),
            "--shap-global",
            str(SHAP_GLOBAL),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert output_json.exists()
    assert output_md.exists()
    assert payload["card_type"] == "ml_comparison"
    assert "pseudo-labels" in payload["methodological_note"]
    assert "Matching V3 reste la baseline officielle" in payload["methodological_note"]
    assert "Random Forest est le meilleur modèle ML actuel" in payload["methodological_note"]
    assert "XGBoost est conservé pour SHAP" in payload["methodological_note"]
    assert "Top 10 Random Forest" in markdown

    candidate = payload["candidates"][0]
    assert "baseline_score_v3" in candidate
    assert "rf_score" in candidate
    assert "xgboost_score" in candidate
    assert "rf_rank" in candidate
    assert "xgboost_rank" in candidate
    assert "rank_shift_v3_vs_rf" in candidate
    assert "rank_shift_v3_vs_xgb" in candidate
    assert "rank_shift_rf_vs_xgb" in candidate
    assert "shap_top_features" in candidate
    assert isinstance(candidate["shap_top_features"], list)
    assert "recommendation_status" in candidate

    assert _sha256(FEATURES) == features_before
    assert _sha256(OFFICIAL_DECISION_CARDS) == official_cards_before

