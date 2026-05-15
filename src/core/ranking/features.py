from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.core.common.seniority import compute_seniority_alignment, normalize_seniority

from .schema import RankingFeatureRow, RankingFeatures


SOURCE_NAME = "matching_v3_normalized"


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def encode_hallucination_risk(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _clip01(float(value))
    normalized = str(value).strip().lower()
    if normalized in {"low", "faible"}:
        return 0.0
    if normalized in {"medium", "moyen"}:
        return 0.5
    if normalized in {"high", "eleve", "élevé"}:
        return 1.0
    return 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return default
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            result = float(stripped)
        except ValueError:
            return default
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    return default


def safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def extract_required_skills(job_profile: dict[str, Any]) -> list[str]:
    candidates = [
        job_profile.get("required_skills"),
        job_profile.get("must_have_skills"),
        _get_nested(job_profile, ("skills", "required")),
        _get_nested(job_profile, ("requirements", "required_skills")),
    ]
    for value in candidates:
        skills = _string_list(value)
        if skills:
            return skills
    return []


def load_candidate_profiles_by_source_path(profiles_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    if profiles_dir is None:
        return {}
    root = Path(profiles_dir)
    if not root.exists():
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        payload = load_json(path)
        source_path = _normalize_path_key(payload.get("source_path"))
        if source_path:
            profiles[source_path] = payload
    return profiles


def build_feature_row(
    job_profile: dict[str, Any],
    candidate_result: dict[str, Any],
    job_id: str,
    rank: int,
    candidate_profiles_by_source_path: dict[str, dict[str, Any]] | None = None,
) -> RankingFeatureRow:
    scores = candidate_result.get("scores") if isinstance(candidate_result.get("scores"), dict) else {}
    required_skills = extract_required_skills(job_profile)
    matched_skills = _string_list(_first_present(candidate_result, ("matched_skills", "matched_required_skills")))
    missing_skills = _string_list(_first_present(candidate_result, ("missing_required_skills", "missing_skills")))

    matched_required_count = len(matched_skills)
    missing_required_count = len(missing_skills)
    total_required = len(required_skills)

    if total_required and matched_required_count + missing_required_count != total_required:
        matched_required_count = len(_casefold_intersection(required_skills, matched_skills))
        missing_required_count = max(total_required - matched_required_count, 0)

    computed_coverage = 1.0 if total_required == 0 else matched_required_count / total_required
    must_have_coverage = _clip01(
        safe_float(
            _first_present(candidate_result, ("must_have_coverage", "required_skills_overlap")),
            computed_coverage,
        )
    )

    reliability_score = _clip01(
        safe_float(_first_present(candidate_result, ("reliability_score", "reliability")), 0.0)
    )
    profile_quality_score = _clip01(
        safe_float(
            _first_present(candidate_result, ("quality_score", "profile_quality_score")),
            safe_float(scores.get("score_profile_quality"), reliability_score),
        )
    )
    seniority_alignment = _resolve_seniority_alignment(
        job_profile,
        candidate_result,
        candidate_profiles_by_source_path or {},
    )

    features = RankingFeatures(
        vector_similarity=_clip01(
            safe_float(_first_present(candidate_result, ("similarity", "vector_similarity", "faiss_score")), 0.0)
        ),
        final_score_v3=_clip01(
            safe_float(_first_present(candidate_result, ("score", "final_score", "normalized_score")), 0.0)
        ),
        must_have_coverage=must_have_coverage,
        required_skills_overlap=must_have_coverage,
        nice_to_have_overlap=_clip01(
            safe_float(
                _first_present(candidate_result, ("nice_to_have_overlap", "optional_skills_overlap")),
                safe_float(scores.get("nice_to_have_overlap"), 0.0),
            )
        ),
        experience_match_score=_clip01(
            safe_float(
                _first_present(candidate_result, ("experience_match_score", "score_experience")),
                safe_float(scores.get("score_experience"), 0.0),
            )
        ),
        seniority_alignment=seniority_alignment,
        profile_quality_score=profile_quality_score,
        reliability_score=reliability_score,
        hallucination_risk_encoded=encode_hallucination_risk(
            _first_present(candidate_result, ("hallucination_risk", "risk"))
        ),
        missing_required_count=float(missing_required_count),
        matched_required_count=float(matched_required_count),
    )

    candidate_id = str(_first_present(candidate_result, ("candidate_id", "id")) or "unknown_candidate")
    profile_id = str(
        _first_present(candidate_result, ("profile_id", "candidate_profile_id", "matched_profile_id"))
        or candidate_id
    )

    return RankingFeatureRow(
        job_id=job_id,
        candidate_id=candidate_id,
        profile_id=profile_id,
        rank=rank,
        source=SOURCE_NAME,
        features=features,
        label=None,
        split=None,
    )


def build_feature_rows(
    job_profile: dict[str, Any],
    matching_report: dict[str, Any],
    job_id: str,
    candidate_profiles_by_source_path: dict[str, dict[str, Any]] | None = None,
) -> list[RankingFeatureRow]:
    candidates = _extract_candidate_results(matching_report)
    rows: list[RankingFeatureRow] = []
    for index, candidate in enumerate(candidates, start=1):
        rank = int(safe_float(candidate.get("rank"), float(index)))
        rows.append(build_feature_row(job_profile, candidate, job_id, rank, candidate_profiles_by_source_path))
    return rows


def write_jsonl(rows: list[RankingFeatureRow], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _extract_candidate_results(matching_report: dict[str, Any]) -> list[dict[str, Any]]:
    nested_results = safe_list(matching_report.get("results"))
    for result in nested_results:
        if isinstance(result, dict):
            recommendations = [item for item in safe_list(result.get("recommendations")) if isinstance(item, dict)]
            if recommendations:
                return recommendations
    for key in ("recommendations", "top_10_candidates", "top_candidates", "candidates"):
        candidates = [item for item in safe_list(matching_report.get(key)) if isinstance(item, dict)]
        if candidates:
            return candidates
    return []


def _resolve_seniority_alignment(
    job_profile: dict[str, Any],
    candidate_result: dict[str, Any],
    candidate_profiles_by_source_path: dict[str, dict[str, Any]],
) -> float:
    explicit = _first_present(candidate_result, ("seniority_alignment",))
    if explicit is not None:
        return _clip01(safe_float(explicit, 0.0))

    job_seniority = normalize_seniority(
        _first_present(candidate_result, ("job_seniority", "required_seniority"))
        or job_profile.get("seniority_level")
    )
    candidate_seniority = normalize_seniority(
        _first_present(candidate_result, ("candidate_seniority", "seniority", "seniority_level"))
    )

    if candidate_seniority is None:
        profile_payload = candidate_profiles_by_source_path.get(_normalize_path_key(candidate_result.get("source_path")))
        if profile_payload:
            profile = profile_payload.get("profile") if isinstance(profile_payload.get("profile"), dict) else profile_payload
            expertise = profile.get("expertise") if isinstance(profile.get("expertise"), dict) else {}
            candidate_seniority = normalize_seniority(expertise.get("experience_level"))

    return compute_seniority_alignment(job_seniority, candidate_seniority)


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _string_list(value: Any) -> list[str]:
    output: list[str] = []
    for item in safe_list(value):
        if isinstance(item, dict):
            item = item.get("name") or item.get("skill") or item.get("label")
        text = str(item).strip() if item is not None else ""
        if text:
            output.append(text)
    return output


def _casefold_intersection(left: list[str], right: list[str]) -> set[str]:
    right_values = {item.casefold().strip() for item in right}
    return {item for item in (skill.casefold().strip() for skill in left) if item in right_values}


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _normalize_path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").strip().lower()
