from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from src.api.auth import require_api_key
from src.api.config import DataSettings, load_data_settings
from src.api.schemas import CandidateDetailResponse, CandidateListItem, PaginatedCandidates
from src.api.utils import (
    compact_candidate,
    find_candidate,
    get_candidates,
    load_best_decision_cards_artifact,
    read_profile_from_card,
)
from src.core.chatbot.candidate_cv_resolver import PROJECT_ROOT, RAW_CV_ROOT, resolve_candidate_cv
from src.core.storage.repositories import (
    MongoRepositories,
    RepositoryUnavailableError,
    create_mongo_repositories,
    format_mongodb_source,
)


router = APIRouter(prefix="/api/candidates", tags=["candidates"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=PaginatedCandidates)
def list_candidates(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)) -> PaginatedCandidates:
    settings = _load_data_settings_or_error()
    if settings.data_backend == "artifacts":
        return _list_candidates_from_artifacts(limit, offset, settings=settings)

    try:
        repositories = create_mongo_repositories(settings.mongodb_uri, settings.mongodb_database)
    except RepositoryUnavailableError as exc:
        return _list_candidates_with_unavailable_mongodb_fallback(limit, offset, settings, exc)

    try:
        collection_name = getattr(repositories.candidates, "collection_name", "candidates")
        candidates = repositories.candidates.list_candidates(limit=limit, offset=offset)
        total = repositories.candidates.count_candidates()
    except RepositoryUnavailableError as exc:
        return _list_candidates_with_unavailable_mongodb_fallback(limit, offset, settings, exc)
    finally:
        repositories.close()

    if settings.data_backend == "hybrid" and not candidates and settings.allow_artifact_fallback:
        return _list_candidates_from_artifacts(
            limit,
            offset,
            settings=settings,
            fallback_used=True,
            extra_warnings=["MongoDB returned no candidates; artifact fallback was used in hybrid mode."],
        )

    warnings: list[str] = []
    if not candidates:
        warnings.append("MongoDB candidates collection is available but contains no candidate.")
    data_source = format_mongodb_source(settings.mongodb_database, collection_name)
    return PaginatedCandidates(
        total=total,
        limit=limit,
        offset=offset,
        source=data_source,
        data_backend=settings.data_backend,
        data_source=data_source,
        fallback_used=False,
        warnings=warnings,
        items=[CandidateListItem(**compact_candidate(candidate)) for candidate in candidates],
    )


def _list_candidates_from_artifacts(
    limit: int,
    offset: int,
    *,
    settings: DataSettings,
    fallback_used: bool = False,
    extra_warnings: list[str] | None = None,
) -> PaginatedCandidates:
    artifact = load_best_decision_cards_artifact()
    candidates = get_candidates(artifact.payload)
    warnings: list[str] = list(extra_warnings or [])
    if not candidates:
        warnings.append("Decision cards artifact is available but contains no candidate.")
    page = candidates[offset : offset + limit]
    return PaginatedCandidates(
        total=len(candidates),
        limit=limit,
        offset=offset,
        source=artifact.source,
        data_backend=settings.data_backend,
        data_source=artifact.source,
        fallback_used=fallback_used,
        warnings=warnings,
        items=[CandidateListItem(**compact_candidate(candidate)) for candidate in page],
    )


@router.get("/{candidate_id}", response_model=CandidateDetailResponse)
def get_candidate(candidate_id: str) -> CandidateDetailResponse:
    settings = _load_data_settings_or_error()
    if settings.data_backend == "artifacts":
        return _get_candidate_from_artifacts(candidate_id, settings=settings)

    try:
        repositories = create_mongo_repositories(settings.mongodb_uri, settings.mongodb_database)
    except RepositoryUnavailableError as exc:
        return _get_candidate_with_unavailable_mongodb_fallback(candidate_id, settings, exc)

    try:
        collection_name = getattr(repositories.candidates, "collection_name", "candidates")
        candidate = repositories.candidates.get_candidate(candidate_id)
        if candidate is None:
            if settings.data_backend == "hybrid" and settings.allow_artifact_fallback:
                return _get_candidate_from_artifacts(
                    candidate_id,
                    settings=settings,
                    fallback_used=True,
                    extra_warnings=["Candidate was not found in MongoDB; artifact fallback was used in hybrid mode."],
                )
            raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")
        profile = _read_profile_from_mongodb_candidate(candidate, repositories)
    except RepositoryUnavailableError as exc:
        return _get_candidate_with_unavailable_mongodb_fallback(candidate_id, settings, exc)
    finally:
        repositories.close()

    data_source = format_mongodb_source(settings.mongodb_database, collection_name)
    return CandidateDetailResponse(
        candidate=candidate,
        profile=profile,
        source=data_source,
        data_backend=settings.data_backend,
        data_source=data_source,
        fallback_used=False,
        warnings=[],
    )


@router.get("/{candidate_id}/cv")
def get_candidate_cv(candidate_id: str, job_id: str | None = Query(default=None)) -> FileResponse:
    cv = resolve_candidate_cv(candidate_id, job_id)
    if not cv.get("cv_available"):
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"CV original not found for candidate: {candidate_id}",
                "candidate_id": candidate_id,
                "job_id": job_id,
            },
        )

    file_path = _safe_cv_file_path(cv.get("cv_path"))
    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"CV original is unavailable or outside allowed raw CV directory: {candidate_id}",
                "candidate_id": candidate_id,
                "job_id": job_id,
            },
        )

    media_type = str(cv.get("cv_mime_type") or "application/octet-stream")
    filename = str(cv.get("cv_filename") or file_path.name)
    disposition = "inline" if media_type in {"application/pdf", "image/jpeg", "image/png", "text/plain"} else "attachment"
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
        content_disposition_type=disposition,
    )


def _get_candidate_from_artifacts(
    candidate_id: str,
    *,
    settings: DataSettings,
    fallback_used: bool = False,
    extra_warnings: list[str] | None = None,
) -> CandidateDetailResponse:
    artifact = load_best_decision_cards_artifact()
    candidate = find_candidate(candidate_id, artifact.payload)
    warnings: list[str] = list(extra_warnings or [])
    profile = read_profile_from_card(candidate, warnings)
    return CandidateDetailResponse(
        candidate=candidate,
        profile=profile,
        source=artifact.source,
        data_backend=settings.data_backend,
        data_source=artifact.source,
        fallback_used=fallback_used,
        warnings=warnings,
    )


def _safe_cv_file_path(cv_path: object) -> Path | None:
    if not cv_path:
        return None
    try:
        path = (PROJECT_ROOT / str(cv_path)).resolve(strict=True)
        raw_root = RAW_CV_ROOT.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not path.is_file() or not path.is_relative_to(raw_root):
        return None
    return path


def _read_profile_from_mongodb_candidate(
    candidate: dict,
    repositories: MongoRepositories,
) -> dict | None:
    profile_id = candidate.get("profile_id") or candidate.get("best_profile_id")
    if profile_id:
        profile = repositories.candidate_profiles.get_profile(str(profile_id))
        if profile:
            return profile
    candidate_id = candidate.get("candidate_id")
    if candidate_id:
        return repositories.candidate_profiles.get_profile_by_candidate_id(str(candidate_id))
    return None


def _list_candidates_with_unavailable_mongodb_fallback(
    limit: int,
    offset: int,
    settings: DataSettings,
    exc: RepositoryUnavailableError,
) -> PaginatedCandidates:
    warning = f"MongoDB unavailable: {exc}"
    if settings.allow_artifact_fallback:
        return _list_candidates_from_artifacts(
            limit,
            offset,
            settings=settings,
            fallback_used=True,
            extra_warnings=[warning, "Artifact fallback used because ALLOW_ARTIFACT_FALLBACK=true."],
        )
    raise _mongodb_unavailable_http_error(settings, warning)


def _get_candidate_with_unavailable_mongodb_fallback(
    candidate_id: str,
    settings: DataSettings,
    exc: RepositoryUnavailableError,
) -> CandidateDetailResponse:
    warning = f"MongoDB unavailable: {exc}"
    if settings.allow_artifact_fallback:
        return _get_candidate_from_artifacts(
            candidate_id,
            settings=settings,
            fallback_used=True,
            extra_warnings=[warning, "Artifact fallback used because ALLOW_ARTIFACT_FALLBACK=true."],
        )
    raise _mongodb_unavailable_http_error(settings, warning)


def _load_data_settings_or_error() -> DataSettings:
    try:
        return load_data_settings()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _mongodb_unavailable_http_error(settings: DataSettings, warning: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": "MongoDB data backend is unavailable and artifact fallback is disabled.",
            "data_backend": settings.data_backend,
            "fallback_used": False,
            "warnings": [warning],
        },
    )
