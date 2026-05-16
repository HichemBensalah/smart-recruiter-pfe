from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.ranking.ml_reranker import (  # noqa: E402
    METHODOLOGY_WARNING,
    build_reranking_report,
    extract_feature_matrix,
    extract_feature_vector,
    load_feature_names,
    load_jsonl,
    load_model,
)


FEATURES = ROOT / "data/ranking/features/backend_python_django_postgresql.jsonl"
MODEL = ROOT / "data/ranking/models/xgboost.joblib"
FEATURE_NAMES = ROOT / "data/ranking/models/feature_names.json"
MATCHING_V3_REPORT = ROOT / "docs/reports/matching/v3/matching_report_v3_normalized.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_and_feature_names_are_loaded() -> None:
    model = load_model(MODEL)
    feature_names = load_feature_names(FEATURE_NAMES)
    assert model is not None
    assert len(feature_names) == 12


def test_extracts_12_features_in_expected_order() -> None:
    rows = load_jsonl(FEATURES)
    feature_names = load_feature_names(FEATURE_NAMES)
    vector = extract_feature_vector(rows[0], feature_names)
    matrix = extract_feature_matrix(rows[:2], feature_names)
    assert len(vector) == 12
    assert matrix.shape == (2, 12)
    assert vector[0] == rows[0]["features"][feature_names[0]]
    assert vector[1] == rows[0]["features"][feature_names[1]]


def test_reranking_report_preserves_matching_v3_and_adds_ml_scores() -> None:
    rows = load_jsonl(FEATURES)
    model = load_model(MODEL)
    feature_names = load_feature_names(FEATURE_NAMES)
    report = build_reranking_report(rows=rows, model=model, feature_names=feature_names)
    candidate = report["candidates"][0]
    assert report["methodology_warning"] == METHODOLOGY_WARNING
    assert report["ranking_mode"] == "experimental_ml_reranking"
    assert 0.0 <= candidate["experimental_ml_score"] <= 1.0
    assert candidate["final_score_v3"] == rows[0]["features"]["final_score_v3"]
    assert candidate["rank_v3"] == rows[0]["rank"]
    assert isinstance(candidate["ml_rank"], int)
    assert np.isclose(
        candidate["score_delta"],
        candidate["experimental_ml_score"] - candidate["final_score_v3"],
    )


def test_script_generates_reports_without_modifying_inputs(tmp_path: Path) -> None:
    features_before = _sha256(FEATURES)
    model_before = _sha256(MODEL)
    matching_v3_before = _sha256(MATCHING_V3_REPORT)
    output_json = tmp_path / "ml_reranking_example.json"
    output_md = tmp_path / "ml_reranking_example.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_ml_reranking.py",
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
    assert output_json.exists()
    assert output_md.exists()
    assert payload["candidates"]
    assert "Matching V3 reste la baseline officielle" in payload["methodology_warning"]
    assert "Decision Cards ne sont pas modifiées" in markdown
    assert _sha256(FEATURES) == features_before
    assert _sha256(MODEL) == model_before
    assert _sha256(MATCHING_V3_REPORT) == matching_v3_before
