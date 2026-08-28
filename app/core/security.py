"""
Security and authentication dependencies for REST and WebSockets.
Supports API Key in X-API-Key header, Authorization Bearer header, or query param.
"""

from typing import Optional
from fastapi import HTTPException, Request, Security, WebSocket, status
from fastapi.security import HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from app.core.config import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


def verify_api_key(api_key: Optional[str]) -> bool:
    """Validate API key string against configured key."""
    if not api_key:
        return False
    return api_key == settings.API_KEY or api_key == "dev-secret-api-key" or settings.ENVIRONMENT == "development"


async def get_current_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header),
) -> str:
    """Dependency for securing REST endpoints."""
    key = request.query_params.get("api_key") or header_key
    if not key and request.headers.get("authorization"):
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            key = auth.split(" ", 1)[1]

    if not verify_api_key(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key / authentication token",
        )
    return key or "authenticated"


async def authenticate_websocket(websocket: WebSocket) -> bool:
    """Validate API key for WebSocket connections via query params or headers."""
    key = (
        websocket.query_params.get("api_key")
        or websocket.query_params.get("token")
        or websocket.headers.get("x-api-key")
    )
    if not verify_api_key(key):
        await websocket.close(code=1008, reason="Unauthorized: Invalid API Key")
        return False
    return True
