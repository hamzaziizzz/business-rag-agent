from __future__ import annotations

"""Authentication helpers."""

from dataclasses import dataclass

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.app.settings import settings


@dataclass(frozen=True)
class AuthContext:
    """Resolved authentication context for the current request."""
    api_key: str | None
    role: str


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    api_key_header: str | None = Security(_api_key_header),
) -> AuthContext:
    """Validate API key or allow anonymous access if configured."""
    api_key = api_key_header or _extract_api_key(request)
    key_map = settings.api_key_map
    allowed = settings.api_keys
    if not (key_map or allowed):
        if settings.allow_anonymous:
            return AuthContext(api_key=None, role="admin")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if key_map:
        entry = key_map.get(api_key)
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        role = entry.get("role", "reader").lower()
        return AuthContext(api_key=api_key, role=role)
    if api_key not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthContext(api_key=api_key, role="admin")


def _extract_api_key(request: Request) -> str | None:
    """Extract API key from headers."""
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()
    auth = request.headers.get("authorization")
    if not auth:
        return None
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
