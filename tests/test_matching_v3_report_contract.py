from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHING_REPORT_PATH = ROOT / "docs/reports/matching/v3/matching_report_v3_normalized.json"
MULTI_OFFER_BACKEND_REPORT_PATH = (
    ROOT / "docs/reports/matching/v3/backend_python_fastapi_mongodb_matching_report_v3_normalized.json"
)
MATCHING_REPORT_PATH = MULTI_OFFER_BACKEND_REPORT_PATH if MULTI_OFFER_BACKEND_REPORT_PATH.exists() else DEFAULT_MATCHING_REPORT_PATH
FEATURE_JSONL_PATH = ROOT / "data/ranking/features/backend_python_fastapi_mongodb.jsonl"
VALID_SENIORITIES = {"junior", "mid_level", "senior", "lead", "principal"}
FORBIDDEN_SUPERVISED_FIELDS = {
    "label",
    "split",
    "xgboost_score",
    "shap_values",
    "shap_explanation",
    "model_prediction",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_recommendations() -> list[dict[str, Any]]:
    report = load_json(MATCHING_REPORT_PATH)
    return extract_recommendations(report)


def extract_recommendations(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("recommendations"), list):
                return [item for item in result["recommendations"] if isinstance(item, dict)]
    for key in ("recommendations", "top_10_candidates", "top_candidates", "candidates"):
        value = report.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def get_any_key(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise AssertionError(f"Missing one of keys {keys} in payload keys={sorted(payload)}")


def maybe_get_any_key(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def assert_bounded(value: Any, field_name: str) -> None:
    assert isinstance(value, (int, float)), f"{field_name} must be numeric, got {type(value).__name__}"
    assert 0.0 <= float(value) <= 1.0, f"{field_name}={value} is outside [0.0, 1.0]"


def load_feature_rows() -> list[dict[str, Any]]:
    rows = []
    for line in FEATURE_JSONL_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_matching_report_exists_and_has_recommendations() -> None:
    assert MATCHING_REPORT_PATH.exists()
    report = load_json(MATCHING_REPORT_PATH)
    recommendations = load_recommendations()
    assert len(recommendations) >= 1
    assert len(recommendations) == int(report.get("top_k") or len(recommendations))


def test_matching_recommendations_have_required_fields() -> None:
    for recommendation in load_recommendations():
        get_any_key(recommendation, ("rank",))
        get_any_key(recommendation, ("candidate_id",))
        get_any_key(recommendation, ("matched_profile_id", "profile_id"))
        get_any_key(recommendation, ("final_score",))
        get_any_key(recommendation, ("faiss_score", "vector_similarity"))
        get_any_key(recommendation, ("must_have_coverage",))
        get_any_key(recommendation, ("matched_skills", "matched_required_skills"))
        get_any_key(recommendation, ("missing_required_skills", "missing_skills"))
        get_any_key(recommendation, ("reliability_score",))
        get_any_key(recommendation, ("hallucination_risk",))
        get_any_key(recommendation, ("job_seniority",))
        get_any_key(recommendation, ("candidate_seniority",))
        get_any_key(recommendation, ("seniority_alignment",))


def test_scores_are_bounded() -> None:
    for recommendation in load_recommendations():
        assert_bounded(recommendation["final_score"], "final_score")
        assert_bounded(get_any_key(recommendation, ("faiss_score", "vector_similarity")), "semantic_similarity")
        assert_bounded(recommendation["must_have_coverage"], "must_have_coverage")
        assert_bounded(recommendation["reliability_score"], "reliability_score")
        assert_bounded(recommendation["seniority_alignment"], "seniority_alignment")


def test_seniority_fields_are_valid() -> None:
    for recommendation in load_recommendations():
        assert recommendation["job_seniority"] in VALID_SENIORITIES
        candidate_seniority = recommendation.get("candidate_seniority")
        assert candidate_seniority in VALID_SENIORITIES or candidate_seniority is None
        assert_bounded(recommendation["seniority_alignment"], "seniority_alignment")
        if candidate_seniority is None:
            assert recommendation["seniority_alignment"] == 0.0
            assert recommendation.get("seniority_warning")


def test_no_fake_supervised_fields_in_matching_report() -> None:
    for recommendation in load_recommendations():
        forbidden_present = FORBIDDEN_SUPERVISED_FIELDS.intersection(recommendation)
        assert forbidden_present == set()


def test_feature_jsonl_consistent_with_matching_seniority() -> None:
    assert FEATURE_JSONL_PATH.exists()
    recommendations_by_candidate = {
        str(recommendation["candidate_id"]): recommendation for recommendation in load_recommendations()
    }
    rows = load_feature_rows()
    assert rows

    for row in rows:
        recommendation = recommendations_by_candidate[str(row["candidate_id"])]
        features = row["features"]
        assert features["seniority_alignment"] == recommendation["seniority_alignment"]
        assert row["label"] is None
        assert row["split"] is None
        assert len(features) == 12
        assert maybe_get_any_key(features, ("seniority_alignment",)) is not None
