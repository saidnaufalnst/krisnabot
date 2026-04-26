from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from app.core.config import settings


SERVICE_KEY_HEADER = "X-KRISNABOT-KEY"


def _service_key() -> str:
    return str(getattr(settings, "krisnabot_service_key", "") or "").strip()


def _is_production() -> bool:
    return str(getattr(settings, "environment", "") or "").strip().casefold() == "production"


def _verify_service_key(request: Request, *, required: bool) -> None:
    expected_key = _service_key()
    if not expected_key:
        if required and _is_production():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="KRISNABOT_SERVICE_KEY belum dikonfigurasi.",
            )
        return

    provided_key = (request.headers.get(SERVICE_KEY_HEADER) or "").strip()
    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Akses service KRISNABOT tidak valid.",
        )


def require_admin_service_key(request: Request) -> None:
    _verify_service_key(request, required=True)


def require_chat_service_key(request: Request) -> None:
    if not bool(getattr(settings, "krisnabot_require_chat_key", False)):
        return
    _verify_service_key(request, required=True)
