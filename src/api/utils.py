from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DECISION_CARDS_TRANSFERABILITY = PROJECT_ROOT / "docs/reports/decision_cards/decision_cards_with_transferability.json"
DECISION_CARDS_ML = PROJECT_ROOT / "docs/reports/decision_cards/decision_cards_ml_comparison.json"
DECISION_CARDS_OFFICIAL = PROJECT_ROOT / "docs/reports/matching/v3/decision_cards_v3_normalized.json"
DEMO_EXECUTIVE_SUMMARY = PROJECT_ROOT / "docs/reports/demo/demo_executive_summary.json"
DEMO_TOP10_SUMMARY = PROJECT_ROOT / "docs/reports/demo/demo_summary_top10.json"
DEMO_RUN_MANIFEST = PROJECT_ROOT / "docs/reports/demo/demo_run_manifest.json"


DEFAULT_DEMO_ARGS = {
    "--features": "data/ranking/features/backend_python_django_postgresql.jsonl",
    "--job": "data/job_profiles/backend_python_django_postgresql.json",
    "--profiles-dir": "data/profile_builder_module2_v2_grounded_all/profiles/grounded_profiles",
    "--graph": "data/graph/skills_roles_graph.yaml",
    "--rf-model": "data/ranking/models/random_forest.joblib",
    "--xgb-ranking": "docs/reports/ml/xgboost_primary_ranking.json",
    "--feature-names": "data/ranking/models/feature_names.json",
    "--cards-ml": "docs/reports/decision_cards/decision_cards_ml_comparison.json",
    "--output-dir": "docs/reports/demo",
}


def read_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found: {path.relative_to(PROJECT_ROOT)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"{label} must be a JSON object")
    return payload


def load_best_decision_cards() -> dict[str, Any]:
    for path, label in (
        (DECISION_CARDS_TRANSFERABILITY, "decision cards with transferability"),
        (DECISION_CARDS_ML, "decision cards ML comparison"),
        (DECISION_CARDS_OFFICIAL, "official decision cards"),
    ):
        if path.exists():
            return read_json_file(path, label)
    raise HTTPException(status_code=404, detail="No decision cards artifact found")


def get_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") or payload.get("decision_cards") or payload.get("items") or []
    if not isinstance(candidates, list):
        raise HTTPException(status_code=500, detail="Decision cards candidates field must be a list")
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    transferability = candidate.get("transferability") if isinstance(candidate.get("transferability"), dict) else {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile_id": candidate.get("profile_id"),
        "baseline_rank_v3": candidate.get("baseline_rank_v3") or candidate.get("rank_v3") or candidate.get("rank"),
        "baseline_score_v3": candidate.get("baseline_score_v3") or candidate.get("final_score_v3"),
        "rf_rank": candidate.get("rf_rank"),
        "rf_score": candidate.get("rf_score"),
        "xgboost_rank": candidate.get("xgboost_rank") or candidate.get("final_rank_ml"),
        "xgboost_score": candidate.get("xgboost_score"),
        "recommendation_status": candidate.get("recommendation_status"),
        "transferability_score": transferability.get("transferability_score"),
    }


def find_candidate(candidate_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    cards_payload = payload or load_best_decision_cards()
    for candidate in get_candidates(cards_payload):
        if candidate.get("candidate_id") == candidate_id or candidate.get("profile_id") == candidate_id:
            return candidate
    raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")


def read_profile_from_card(candidate: dict[str, Any]) -> dict[str, Any] | None:
    transferability = candidate.get("transferability")
    if not isinstance(transferability, dict):
        return None
    raw_path = transferability.get("profile_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def sort_by_v3_rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            item.get("baseline_rank_v3") is None,
            item.get("baseline_rank_v3") or item.get("rank_v3") or item.get("rank") or 10**9,
        ),
    )


def run_demo_script() -> dict[str, Any]:
    command = [sys.executable, "scripts/run_demo_end_to_end.py"]
    for key, value in DEFAULT_DEMO_ARGS.items():
        command.extend([key, value])
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Demo end-to-end script failed",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    return read_json_file(DEMO_RUN_MANIFEST, "demo run manifest")
