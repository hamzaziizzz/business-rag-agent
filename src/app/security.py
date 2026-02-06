from __future__ import annotations

"""Authentication and tenant resolution helpers."""

from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from src.app.settings import settings


@dataclass(frozen=True)
class AuthContext:
    """Resolved authentication context for the current request."""
    api_key: str | None
    role: str
    tenant_id: str


async def require_api_key(request: Request) -> AuthContext:
    """Validate API key or allow anonymous access if configured."""
    api_key = _extract_api_key(request)
    key_map = settings.api_key_map
    allowed = settings.api_keys
    if not (key_map or allowed):
        if settings.allow_anonymous:
            return AuthContext(api_key=None, role="admin", tenant_id=settings.default_tenant_id)
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
        tenant_id = entry.get("tenant_id", settings.default_tenant_id)
        return AuthContext(api_key=api_key, role=role, tenant_id=tenant_id)
    if api_key not in allowed:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthContext(api_key=api_key, role="admin", tenant_id=settings.default_tenant_id)


def require_roles(auth: AuthContext, allowed: set[str]) -> None:
    """Enforce role-based access control."""
    if auth.role not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )


def resolve_tenant_id(request: Request, auth: AuthContext) -> str:
    """Resolve tenant ID, allowing admin override via header."""
    if auth.tenant_id == "*":
        return "*"
    header = request.headers.get("x-tenant-id")
    if header and auth.role == "admin":
        return header.strip()
    return auth.tenant_id


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
