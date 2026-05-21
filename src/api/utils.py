from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
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
MATCHING_FEATURES_DIR = PROJECT_ROOT / "data/ranking/features"
DEFAULT_MATCHING_JOB_ID = "backend_python_django_postgresql"
DEMO_SCRIPT = PROJECT_ROOT / "scripts/run_demo_end_to_end.py"


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


@dataclass(frozen=True)
class MatchingArtifact:
    requested_job_id: str | None
    resolved_job_id: str
    artifact_path: Path
    fallback_used: bool
    warning: str | None = None

    @property
    def artifact_source(self) -> str:
        return _relative_source(self.artifact_path)


@dataclass(frozen=True)
class LoadedJsonArtifact:
    payload: dict[str, Any]
    source: str


def list_matching_artifacts() -> dict[str, Path]:
    if not MATCHING_FEATURES_DIR.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(MATCHING_FEATURES_DIR.glob("*.jsonl"))
        if path.is_file()
    }


def resolve_matching_artifact(job_id: str | None) -> MatchingArtifact:
    artifacts = list_matching_artifacts()
    requested_job_id = job_id or DEFAULT_MATCHING_JOB_ID
    if requested_job_id in artifacts:
        return MatchingArtifact(
            requested_job_id=job_id,
            resolved_job_id=requested_job_id,
            artifact_path=artifacts[requested_job_id],
            fallback_used=False,
        )

    fallback_path = artifacts.get(DEFAULT_MATCHING_JOB_ID)
    if fallback_path is None:
        fallback_path = MATCHING_FEATURES_DIR / f"{DEFAULT_MATCHING_JOB_ID}.jsonl"
    warning = (
        f"Matching artifact not found for job_id '{requested_job_id}'. "
        f"Fallback used: '{DEFAULT_MATCHING_JOB_ID}'."
    )
    return MatchingArtifact(
        requested_job_id=job_id,
        resolved_job_id=DEFAULT_MATCHING_JOB_ID,
        artifact_path=fallback_path,
        fallback_used=True,
        warning=warning,
    )


def read_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found: {_relative_source(path)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"{label} must be a JSON object")
    return payload


def _relative_source(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_best_decision_cards_artifact() -> LoadedJsonArtifact:
    for path, label in (
        (DECISION_CARDS_TRANSFERABILITY, "decision cards with transferability"),
        (DECISION_CARDS_ML, "decision cards ML comparison"),
        (DECISION_CARDS_OFFICIAL, "official decision cards"),
    ):
        if path.exists():
            return LoadedJsonArtifact(payload=read_json_file(path, label), source=_relative_source(path))
    raise HTTPException(status_code=404, detail="No decision cards artifact found")


def load_best_decision_cards() -> dict[str, Any]:
    return load_best_decision_cards_artifact().payload


def get_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") or payload.get("decision_cards") or payload.get("items") or []
    if not isinstance(candidates, list):
        raise HTTPException(status_code=500, detail="Decision cards candidates field must be a list")
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    transferability = candidate.get("transferability") if isinstance(candidate.get("transferability"), dict) else {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "profile_id": candidate.get("profile_id") or candidate.get("best_profile_id"),
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


def read_profile_from_card(candidate: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any] | None:
    transferability = candidate.get("transferability")
    if not isinstance(transferability, dict):
        if warnings is not None:
            warnings.append("Candidate card has no transferability block with a profile_path.")
        return None
    raw_path = transferability.get("profile_path")
    if not raw_path:
        if warnings is not None:
            warnings.append("Candidate card has no profile_path; returning the decision card only.")
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists() or not path.is_file():
        if warnings is not None:
            warnings.append(f"Candidate profile artifact not found: {_relative_source(path)}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append(f"Candidate profile artifact is not valid JSON: {_relative_source(path)} ({exc})")
        return None
    if not isinstance(payload, dict):
        if warnings is not None:
            warnings.append(f"Candidate profile artifact must be a JSON object: {_relative_source(path)}")
        return None
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
    script_path = DEMO_SCRIPT
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo script not found: {_relative_source(script_path)}")
    command = [sys.executable, _relative_source(script_path)]
    for key, value in DEFAULT_DEMO_ARGS.items():
        command.extend([key, value])
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Demo end-to-end script failed",
                "source": _relative_source(script_path),
                "fallback_used": False,
                "warnings": ["The demo script returned a non-zero exit code."],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    manifest = read_json_file(DEMO_RUN_MANIFEST, "demo run manifest")
    manifest.setdefault("source", _relative_source(script_path))
    manifest.setdefault("fallback_used", False)
    manifest.setdefault("warnings", [])
    return manifest
