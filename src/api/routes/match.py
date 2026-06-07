from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import require_api_key
from src.api.config import DataSettings, MatchingSettings, load_data_settings, load_matching_settings
from src.api.schemas import MatchCandidate, MatchRequest, MatchResponse
from src.api.utils import (
    MatchingArtifact,
    compact_candidate,
    get_candidates,
    load_best_decision_cards,
    resolve_matching_artifact,
    sort_by_v3_rank,
)
from src.core.matching.live_matcher import LiveMatcher, LiveMatcherSettings, LiveMatchingUnavailable
from src.core.storage.repositories import RepositoryUnavailableError, create_mongo_repositories


router = APIRouter(prefix="/api/match", tags=["matching"], dependencies=[Depends(require_api_key)])


@router.get("/live-readiness")
def live_readiness() -> dict:
    """
    Diagnostic endpoint: checks whether MongoDB, FAISS, id_map and
    SentenceTransformer are available for live matching.
    Returns live_ready=true only when all dependencies are satisfied.
    """
    from src.core.chatbot.live_readiness import check_live_matching_readiness

    data_settings = _load_data_settings_or_error()
    matching_settings = _load_matching_settings_or_error()
    return check_live_matching_readiness(
        mongodb_uri=data_settings.mongodb_uri,
        mongodb_database=data_settings.mongodb_database,
        faiss_index_path=matching_settings.faiss_index_path,
        faiss_id_map_path=matching_settings.faiss_id_map_path,
    )


METHODOLOGICAL_NOTE = (
    "Matching V3 reste la baseline officielle. Cet endpoint utilise les artefacts Matching V3 deja generes "
    "pour le job_id demande quand ils existent. Si l'artefact n'existe pas, un fallback explicite est applique. "
    "L'endpoint ne relance pas FAISS, MongoDB, Matching V3 ou un entrainement ML."
)

LIVE_METHODOLOGICAL_NOTE = (
    "Matching V3 reste la baseline officielle. En mode live, l'endpoint utilise MongoDB comme source "
    "des profils candidats, FAISS comme retrieval, puis score_candidate() comme scoring Matching V3. "
    "RF, XGBoost et SHAP restent experimentaux et ne sont pas relances ici."
)


@router.post("", response_model=MatchResponse)
def match_candidates(request: MatchRequest) -> MatchResponse:
    data_settings = _load_data_settings_or_error()
    matching_settings = _load_matching_settings_or_error()
    if matching_settings.matching_mode == "artifact":
        return _match_from_artifacts(request, data_settings=data_settings)
    if matching_settings.matching_mode == "live":
        return _match_live_or_error(request, data_settings=data_settings, matching_settings=matching_settings)
    return _match_hybrid(request, data_settings=data_settings, matching_settings=matching_settings)


def _match_from_artifacts(
    request: MatchRequest,
    *,
    data_settings: DataSettings,
    extra_warnings: list[str] | None = None,
    force_fallback_used: bool = False,
    matching_mode_override: str | None = None,
) -> MatchResponse:
    artifact = resolve_matching_artifact(request.job_id)
    candidates, load_warnings, candidate_source = _load_candidates_for_artifact(artifact)
    candidates = candidates[: request.top_k]
    items = [_to_match_candidate(candidate) for candidate in candidates]
    matching_mode = matching_mode_override or _matching_mode(artifact, candidate_source)
    warnings = list(extra_warnings or [])
    if artifact.warning:
        warnings.append(artifact.warning)
    warnings.extend(load_warnings)
    if not items:
        warnings.append("No candidate could be returned for this matching request.")
    response = MatchResponse(
        job_description=request.job_description,
        job_id=request.job_id,
        resolved_job_id=artifact.resolved_job_id,
        artifact_source=artifact.artifact_source,
        data_backend=data_settings.data_backend,
        data_source=artifact.artifact_source,
        retrieval_source=artifact.artifact_source,
        scoring_source="matching_v3_artifact_features",
        top_k=request.top_k,
        matching_mode=matching_mode,
        fallback_used=artifact.fallback_used or force_fallback_used,
        warnings=warnings,
        methodological_note=METHODOLOGICAL_NOTE,
        items=items,
    )
    _try_save_matching_run(response, data_settings)
    return response


def _match_live_or_error(
    request: MatchRequest,
    *,
    data_settings: DataSettings,
    matching_settings: MatchingSettings,
) -> MatchResponse:
    try:
        return _match_live(request, data_settings=data_settings, matching_settings=matching_settings)
    except (LiveMatchingUnavailable, RepositoryUnavailableError) as exc:
        raise _live_matching_unavailable_http_error(str(exc), data_settings=data_settings, matching_settings=matching_settings) from exc


def _match_hybrid(
    request: MatchRequest,
    *,
    data_settings: DataSettings,
    matching_settings: MatchingSettings,
) -> MatchResponse:
    try:
        response = _match_live(request, data_settings=data_settings, matching_settings=matching_settings)
        response.matching_mode = "hybrid_live_mongodb_faiss_matching_v3"
        return response
    except (LiveMatchingUnavailable, RepositoryUnavailableError) as exc:
        live_error = str(exc)
        if data_settings.allow_artifact_fallback:
            try:
                return _match_from_artifacts(
                    request,
                    data_settings=data_settings,
                    extra_warnings=[f"Live matching failed; artifact fallback used: {live_error}"],
                    force_fallback_used=True,
                    matching_mode_override="hybrid_live_then_artifact",
                )
            except HTTPException as artifact_exc:
                if artifact_exc.status_code == 404:
                    # Both live and artifact failed — return a clear 503 instead of a
                    # confusing 404 "No decision cards artifact found".
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "matching_mode": "hybrid",
                            "fallback_used": False,
                            "warnings": [
                                f"Live matching unavailable: {live_error}",
                                "Artifact fallback also unavailable: no matching artifacts or decision cards found.",
                                "Ensure MongoDB is running with candidate profiles, or generate artifact files.",
                            ],
                        },
                    ) from artifact_exc
                raise
        raise _live_matching_unavailable_http_error(live_error, data_settings=data_settings, matching_settings=matching_settings) from exc


def _match_live(
    request: MatchRequest,
    *,
    data_settings: DataSettings,
    matching_settings: MatchingSettings,
) -> MatchResponse:
    repositories = create_mongo_repositories(data_settings.mongodb_uri, data_settings.mongodb_database)
    try:
        matcher = LiveMatcher(
            repositories=repositories,
            settings=LiveMatcherSettings(
                mongodb_database=data_settings.mongodb_database,
                top_n=matching_settings.live_matching_top_n,
                default_top_k=matching_settings.live_matching_top_k,
                faiss_index_path=Path(matching_settings.faiss_index_path),
                faiss_id_map_path=Path(matching_settings.faiss_id_map_path),
            ),
        )
        result = matcher.match(
            job_description=request.job_description,
            job_id=request.job_id,
            top_k=request.top_k,
        )
    finally:
        repositories.close()

    return MatchResponse(
        job_description=request.job_description,
        job_id=request.job_id,
        resolved_job_id=result.resolved_job_id,
        artifact_source=None,
        data_backend="mongodb",
        data_source=result.data_source,
        retrieval_source=result.retrieval_source,
        scoring_source=result.scoring_source,
        matching_run_id=result.matching_run_id,
        top_k=result.top_k,
        matching_mode="live_mongodb_faiss_matching_v3",
        fallback_used=False,
        warnings=result.warnings,
        methodological_note=LIVE_METHODOLOGICAL_NOTE,
        items=[_to_match_candidate(item) for item in result.items],
    )


def _matching_mode(artifact: MatchingArtifact, candidate_source: str) -> str:
    if candidate_source == "decision_cards_fallback":
        return "decision_cards_fallback_after_missing_matching_artifact"
    return "matching_v3_job_artifact_with_fallback" if artifact.fallback_used else "matching_v3_job_artifact"


def _load_candidates_for_artifact(artifact: MatchingArtifact) -> tuple[list[dict[str, Any]], list[str], str]:
    feature_rows, warnings = _load_feature_rows(artifact)
    if not feature_rows:
        payload = load_best_decision_cards()
        warnings.append("Matching artifact rows unavailable; candidates were loaded from decision cards instead.")
        return sort_by_v3_rank(get_candidates(payload)), warnings, "decision_cards_fallback"

    cards_by_candidate = _cards_by_candidate_id()
    candidates: list[dict[str, Any]] = []
    for row in sorted(feature_rows, key=lambda item: item.get("rank") or 10**9):
        candidate_id = row.get("candidate_id")
        card = cards_by_candidate.get(str(candidate_id), {}) if candidate_id else {}
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        merged = dict(card)
        merged.update(
            {
                "candidate_id": candidate_id,
                "profile_id": row.get("profile_id") or card.get("profile_id"),
                "baseline_rank_v3": row.get("rank") or card.get("baseline_rank_v3"),
                "baseline_score_v3": features.get("final_score_v3") or card.get("baseline_score_v3"),
                "features": features or card.get("features"),
                "job_id": row.get("job_id") or artifact.resolved_job_id,
                "source": row.get("source") or "matching_v3_normalized",
            }
        )
        candidates.append(merged)
    return candidates, warnings, "matching_v3_artifact"


def _load_feature_rows(artifact: MatchingArtifact) -> tuple[list[dict[str, Any]], list[str]]:
    path = artifact.artifact_path
    warnings: list[str] = []
    if not path.exists():
        warnings.append(f"Matching artifact file not found: {artifact.artifact_source}")
        return [], warnings
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Matching artifact contains invalid JSON at {artifact.artifact_source}:{line_number}: {exc.msg}",
            ) from exc
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            warnings.append(f"Matching artifact row ignored because it is not an object: {artifact.artifact_source}:{line_number}")
    return rows, warnings


def _cards_by_candidate_id() -> dict[str, dict[str, Any]]:
    try:
        payload = load_best_decision_cards()
    except Exception:
        return {}
    return {
        str(candidate.get("candidate_id")): candidate
        for candidate in get_candidates(payload)
        if candidate.get("candidate_id")
    }


def _to_match_candidate(candidate: dict[str, Any]) -> MatchCandidate:
    compact = compact_candidate(candidate)
    return MatchCandidate(
        candidate_id=compact["candidate_id"],
        profile_id=compact["profile_id"],
        rank=compact["baseline_rank_v3"],
        baseline_rank_v3=compact["baseline_rank_v3"],
        baseline_score_v3=compact["baseline_score_v3"],
        faiss_rank=candidate.get("faiss_rank"),
        faiss_score=candidate.get("faiss_score"),
        rf_rank=compact["rf_rank"],
        rf_score=compact["rf_score"],
        xgboost_rank=compact["xgboost_rank"],
        xgboost_score=compact["xgboost_score"],
        recommendation_status=compact["recommendation_status"],
        matched_skills=candidate.get("matched_skills") if isinstance(candidate.get("matched_skills"), list) else None,
        missing_required_skills=(
            candidate.get("missing_required_skills")
            if isinstance(candidate.get("missing_required_skills"), list)
            else None
        ),
        explanation=candidate.get("explanation") if isinstance(candidate.get("explanation"), str) else None,
        transferability=candidate.get("transferability") if isinstance(candidate.get("transferability"), dict) else None,
        cv_available=bool(candidate.get("cv_available")),
        has_original_cv=bool(candidate.get("has_original_cv") or candidate.get("cv_available")),
        cv_download_url=candidate.get("cv_download_url") if isinstance(candidate.get("cv_download_url"), str) else None,
        cv_url=candidate.get("cv_url") if isinstance(candidate.get("cv_url"), str) else None,
        cv_path=candidate.get("cv_path") if isinstance(candidate.get("cv_path"), str) else None,
        cv_filename=candidate.get("cv_filename") if isinstance(candidate.get("cv_filename"), str) else None,
        cv_mime_type=candidate.get("cv_mime_type") if isinstance(candidate.get("cv_mime_type"), str) else None,
        cv_source=candidate.get("cv_source") if isinstance(candidate.get("cv_source"), str) else None,
        cv_confidence=candidate.get("cv_confidence") if isinstance(candidate.get("cv_confidence"), str) else None,
        score_breakdown=candidate.get("score_breakdown") if isinstance(candidate.get("score_breakdown"), list) else None,
        base_score_before_penalty=candidate.get("base_score_before_penalty"),
        must_have_coverage=candidate.get("must_have_coverage"),
        must_have_penalty_multiplier=candidate.get("must_have_penalty_multiplier"),
        must_have_penalty_applied=candidate.get("must_have_penalty_applied"),
        quality_penalty_multiplier=candidate.get("quality_penalty_multiplier"),
    )


def _load_data_settings_or_error() -> DataSettings:
    try:
        return load_data_settings()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _load_matching_settings_or_error() -> MatchingSettings:
    try:
        return load_matching_settings()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _live_matching_unavailable_http_error(
    message: str,
    *,
    data_settings: DataSettings,
    matching_settings: MatchingSettings,
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": "Live matching is unavailable and artifact fallback was not used.",
            "matching_mode": matching_settings.matching_mode,
            "data_backend": data_settings.data_backend,
            "fallback_used": False,
            "warnings": [message],
        },
    )


def _try_save_matching_run(
    response: MatchResponse,
    data_settings: DataSettings,
) -> None:
    if data_settings.data_backend not in {"mongodb", "hybrid"}:
        return
    try:
        repositories = create_mongo_repositories(data_settings.mongodb_uri, data_settings.mongodb_database)
    except RepositoryUnavailableError as exc:
        response.warnings.append(f"Matching run was not persisted because MongoDB is unavailable: {exc}")
        return

    try:
        run_id = repositories.matching_runs.save_matching_run(
            {
                "job_id": response.job_id,
                "resolved_job_id": response.resolved_job_id,
                "job_description": response.job_description,
                "artifact_source": response.artifact_source,
                "matching_mode": response.matching_mode,
                "fallback_used": response.fallback_used,
                "top_k": response.top_k,
                "candidate_ids": [item.candidate_id for item in response.items],
                "scores": [
                    {
                        "candidate_id": item.candidate_id,
                        "profile_id": item.profile_id,
                        "rank": item.rank,
                        "final_score_v3": item.baseline_score_v3,
                    }
                    for item in response.items
                ],
                "warnings": list(response.warnings),
                "source_metadata": {
                    "data_source": response.data_source,
                    "retrieval_source": response.retrieval_source,
                    "scoring_source": response.scoring_source,
                },
                "response": response.model_dump(),
                "source": "api_match_artifact_result",
            }
        )
        response.matching_run_id = run_id
    except RepositoryUnavailableError as exc:
        response.warnings.append(f"Matching run was not persisted because MongoDB write failed: {exc}")
    finally:
        repositories.close()
