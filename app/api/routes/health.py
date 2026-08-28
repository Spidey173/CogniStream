"""
Health probes (/health, /ready) and Prometheus telemetry (/metrics).
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app.core.config import get_settings
from app.core.metrics import POSTGRES_CONNECTED_GAUGE, REDIS_CONNECTED_GAUGE
from app.db.session import AsyncSessionLocal
from app.services.message_bus import message_bus

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", summary="Liveness probe")
async def health_check():
    """Returns status 200 OK if service process is alive."""
    return {"status": "healthy", "service": settings.APP_NAME, "env": settings.ENVIRONMENT}


@router.get("/ready", summary="Readiness probe")
async def readiness_check():
    """Verifies infrastructure connections (PostgreSQL & Redis)."""
    db_ok = False
    redis_ok = False

    # Check Database
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            db_ok = True
            POSTGRES_CONNECTED_GAUGE.set(1)
    except Exception:
        POSTGRES_CONNECTED_GAUGE.set(0)

    # Check Redis / MessageBus
    if hasattr(message_bus, "_is_connected"):
        redis_ok = bool(getattr(message_bus, "_is_connected", False))
    else:
        redis_ok = True  # InMemory mode is ready

    status_str = "ready" if (db_ok and redis_ok) else "degraded"
    return {
        "status": status_str,
        "database": "connected" if db_ok else "disconnected",
        "redis": "connected" if redis_ok else "in-memory-fallback",
    }


@router.get("/metrics", summary="Prometheus telemetry scrape endpoint")
async def get_prometheus_metrics():
    """Scrape endpoint for Prometheus monitoring collector."""
    metrics_data = generate_latest()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
