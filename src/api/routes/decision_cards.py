from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import require_api_key
from src.api.config import DataSettings, load_data_settings
from src.api.schemas import DecisionCardDetailResponse, DecisionCardsResponse
from src.api.utils import find_candidate, get_candidates, load_best_decision_cards_artifact
from src.core.storage.repositories import RepositoryUnavailableError, create_mongo_repositories, format_mongodb_source


router = APIRouter(prefix="/api/decision-cards", tags=["decision-cards"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=DecisionCardsResponse)
def list_decision_cards() -> dict:
    settings = _load_data_settings_or_error()
    if settings.data_backend == "artifacts":
        return _list_decision_cards_from_artifacts(settings=settings)

    try:
        repositories = create_mongo_repositories(settings.mongodb_uri, settings.mongodb_database)
    except RepositoryUnavailableError as exc:
        return _list_decision_cards_with_unavailable_mongodb_fallback(settings, exc)

    try:
        collection_name = getattr(repositories.decision_cards, "collection_name", "decision_cards")
        candidates = repositories.decision_cards.list_decision_cards()
    except RepositoryUnavailableError as exc:
        return _list_decision_cards_with_unavailable_mongodb_fallback(settings, exc)
    finally:
        repositories.close()

    if settings.data_backend == "hybrid" and not candidates and settings.allow_artifact_fallback:
        return _list_decision_cards_from_artifacts(
            settings=settings,
            fallback_used=True,
            extra_warnings=["MongoDB returned no decision cards; artifact fallback was used in hybrid mode."],
        )

    warnings: list[str] = []
    if not candidates:
        warnings.append("MongoDB decision_cards collection is available but contains no candidate.")
    data_source = format_mongodb_source(settings.mongodb_database, collection_name)
    return {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "source": data_source,
        "data_backend": settings.data_backend,
        "data_source": data_source,
        "fallback_used": False,
        "warnings": warnings,
    }


def _list_decision_cards_from_artifacts(
    *,
    settings: DataSettings,
    fallback_used: bool = False,
    extra_warnings: list[str] | None = None,
) -> dict:
    artifact = load_best_decision_cards_artifact()
    payload = dict(artifact.payload)
    candidates = get_candidates(payload)
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    warnings = [*(extra_warnings or []), *warnings]
    if not candidates:
        warnings = [*warnings, "Decision cards artifact is available but contains no candidate."]
    payload["candidates"] = candidates
    payload["candidate_count"] = payload.get("candidate_count") or len(candidates)
    payload["source"] = artifact.source
    payload["data_backend"] = settings.data_backend
    payload["data_source"] = artifact.source
    payload["fallback_used"] = fallback_used
    payload["warnings"] = [str(warning) for warning in warnings]
    return payload


@router.get("/{candidate_id}", response_model=DecisionCardDetailResponse)
def get_decision_card(candidate_id: str) -> dict:
    settings = _load_data_settings_or_error()
    if settings.data_backend == "artifacts":
        return _get_decision_card_from_artifacts(candidate_id, settings=settings)

    try:
        repositories = create_mongo_repositories(settings.mongodb_uri, settings.mongodb_database)
    except RepositoryUnavailableError as exc:
        return _get_decision_card_with_unavailable_mongodb_fallback(candidate_id, settings, exc)

    try:
        collection_name = getattr(repositories.decision_cards, "collection_name", "decision_cards")
        card = repositories.decision_cards.get_decision_card(candidate_id)
        if card is None:
            if settings.data_backend == "hybrid" and settings.allow_artifact_fallback:
                return _get_decision_card_from_artifacts(
                    candidate_id,
                    settings=settings,
                    fallback_used=True,
                    extra_warnings=["Decision Card was not found in MongoDB; artifact fallback was used in hybrid mode."],
                )
            raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")
    except RepositoryUnavailableError as exc:
        return _get_decision_card_with_unavailable_mongodb_fallback(candidate_id, settings, exc)
    finally:
        repositories.close()

    data_source = format_mongodb_source(settings.mongodb_database, collection_name)
    card = dict(card)
    card["source"] = data_source
    card["data_backend"] = settings.data_backend
    card["data_source"] = data_source
    card["fallback_used"] = False
    card["warnings"] = [str(warning) for warning in card.get("warnings", []) if warning]
    return card


def _get_decision_card_from_artifacts(
    candidate_id: str,
    *,
    settings: DataSettings,
    fallback_used: bool = False,
    extra_warnings: list[str] | None = None,
) -> dict:
    artifact = load_best_decision_cards_artifact()
    card = dict(find_candidate(candidate_id, artifact.payload))
    warnings = card.get("warnings") if isinstance(card.get("warnings"), list) else []
    warnings = [*(extra_warnings or []), *warnings]
    card["source"] = artifact.source
    card["data_backend"] = settings.data_backend
    card["data_source"] = artifact.source
    card["fallback_used"] = fallback_used
    card["warnings"] = [str(warning) for warning in warnings]
    return card


def _list_decision_cards_with_unavailable_mongodb_fallback(
    settings: DataSettings,
    exc: RepositoryUnavailableError,
) -> dict:
    warning = f"MongoDB unavailable: {exc}"
    if settings.allow_artifact_fallback:
        return _list_decision_cards_from_artifacts(
            settings=settings,
            fallback_used=True,
            extra_warnings=[warning, "Artifact fallback used because ALLOW_ARTIFACT_FALLBACK=true."],
        )
    raise _mongodb_unavailable_http_error(settings, warning)


def _get_decision_card_with_unavailable_mongodb_fallback(
    candidate_id: str,
    settings: DataSettings,
    exc: RepositoryUnavailableError,
) -> dict:
    warning = f"MongoDB unavailable: {exc}"
    if settings.allow_artifact_fallback:
        return _get_decision_card_from_artifacts(
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
