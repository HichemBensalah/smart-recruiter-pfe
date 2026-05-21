from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from src.api.config import load_api_settings


async def require_api_key(request: Request) -> None:
    settings = load_api_settings()
    if not settings.auth_enabled:
        return

    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "API key authentication is enabled but SMART_RECRUITER_API_KEY is not configured.",
                "header": settings.api_key_header,
                "auth_enabled": True,
            },
        )

    provided_api_key = request.headers.get(settings.api_key_header)
    if not provided_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "API key is required for this endpoint.",
                "header": settings.api_key_header,
                "auth_enabled": True,
            },
        )

    if not hmac.compare_digest(provided_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "API key is invalid.",
                "header": settings.api_key_header,
                "auth_enabled": True,
            },
        )
