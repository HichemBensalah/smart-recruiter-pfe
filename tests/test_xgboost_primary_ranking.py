from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.ml_primary_ranker import run_primary_ranking  # noqa: E402


FEATURES = ROOT / "data/ranking/features/backend_python_django_postgresql.jsonl"
MODEL = ROOT / "data/ranking/models/xgboost.joblib"
FEATURE_NAMES = ROOT / "data/ranking/models/feature_names.json"
MATCHING_V3_REPORT = ROOT / "docs/reports/matching/v3/matching_report_v3_normalized.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primary_ranking_report_contains_ml_and_baseline_fields() -> None:
    report = run_primary_ranking(
        features_path=FEATURES,
        model_path=MODEL,
        feature_names_path=FEATURE_NAMES,
    )
    candidate = report["candidates"][0]
    assert report["ranking_mode"] == "xgboost_primary_with_matching_v3_features"
    assert "xgboost_score" in candidate
    assert "final_rank_ml" in candidate
    assert "baseline_rank_v3" in candidate
    assert "baseline_score_v3" in candidate
    assert "rank_shift" in candidate
    assert candidate["rank_shift"] == candidate["baseline_rank_v3"] - candidate["final_rank_ml"]
    assert "final_score_v3" in candidate["features"]


def test_candidates_are_sorted_by_xgboost_score_descending() -> None:
    report = run_primary_ranking(
        features_path=FEATURES,
        model_path=MODEL,
        feature_names_path=FEATURE_NAMES,
    )
    scores = [candidate["xgboost_score"] for candidate in report["candidates"]]
    ranks = [candidate["final_rank_ml"] for candidate in report["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert ranks == list(range(1, len(ranks) + 1))


def test_script_generates_reports_without_modifying_inputs(tmp_path: Path) -> None:
    features_before = _sha256(FEATURES)
    model_before = _sha256(MODEL)
    matching_v3_before = _sha256(MATCHING_V3_REPORT)
    output_json = tmp_path / "xgboost_primary_ranking.json"
    output_md = tmp_path / "xgboost_primary_ranking.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_xgboost_primary_ranking.py",
            "--features",
            str(FEATURES),
            "--model",
            str(MODEL),
            "--feature-names",
            str(FEATURE_NAMES),
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
    assert payload["ranking_mode"] == "xgboost_primary_with_matching_v3_features"
    assert payload["candidates"]
    assert "Matching V3" in markdown
    assert "baseline_score_v3" in markdown
    assert _sha256(FEATURES) == features_before
    assert _sha256(MODEL) == model_before
    assert _sha256(MATCHING_V3_REPORT) == matching_v3_before

