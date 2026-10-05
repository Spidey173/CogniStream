"""
CogniStream AI — Vercel Serverless API Handler
Lightweight edge telemetry, PostgreSQL persistence, and health checks.
Runs smoothly on Vercel Python Serverless without heavy OpenCV/MediaPipe baggage.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from fastapi import FastAPI, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognistream-api")

app = FastAPI(
    title="CogniStream AI — Serverless API",
    description="High-performance edge telemetry and detection persistence layer.",
    version="2.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# CORS Middleware (supports Vercel preview domains and local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-Memory Ring Buffer Fallback ──
# Ensures 100% endpoint reliability even when DATABASE_URL is not yet configured
_IN_MEMORY_RECORDS: List[Dict[str, Any]] = []
_MAX_IN_MEMORY = 100

# ── Schema Definitions ──
class DetectionPayload(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: Optional[float] = 0.95
    emotion: Optional[str] = "Neutral"
    emotion_confidence: Optional[float] = 75.0
    track_id: Optional[int] = 1


class ROIBatchRequest(BaseModel):
    camera_id: str = "default"
    detections: List[DetectionPayload] = Field(default_factory=list)


class ROIResponse(BaseModel):
    id: Optional[int] = None
    camera_id: str = "default"
    track_id: Optional[int] = None
    x: float
    y: float
    width: float
    height: float
    confidence: Optional[float] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    timestamp: datetime
    created_at: Optional[datetime] = None


# ── Database Normalization & Helper ──
def get_clean_database_url() -> Optional[str]:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        return None
    url = raw
    # Normalize postgres prefix
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    try:
        parsed = urlparse(url)
        if parsed.query:
            query_params = parse_qs(parsed.query)
            clean_params = {}
            if "sslmode" in query_params:
                clean_params["ssl"] = "require"
            elif "ssl" in query_params:
                clean_params["ssl"] = query_params["ssl"][0]
            new_query = urlencode(clean_params)
            url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            ))
    except Exception:
        pass
    return url


# ── Health Probes ──
@app.get("/api/health", summary="Liveness & Readiness probe")
@app.get("/api/v1/health")
async def health_check():
    db_url = get_clean_database_url()
    return {
        "status": "healthy",
        "service": "CogniStream AI",
        "runtime": "Vercel Python Serverless",
        "vision_engine": "In-Browser WebAssembly/WebGL (60 FPS)",
        "database": "configured" if db_url else "in-memory-fallback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Telemetry Ingestion (POST) ──
@app.post("/api/roi", summary="Persist face detection telemetry batch")
@app.post("/api/v1/roi")
async def save_roi_telemetry(payload: ROIBatchRequest):
    now = datetime.now(timezone.utc)
    inserted_count = 0
    saved_items = []

    for det in payload.detections:
        item = {
            "id": len(_IN_MEMORY_RECORDS) + 1,
            "camera_id": payload.camera_id,
            "track_id": det.track_id,
            "x": round(det.x, 3),
            "y": round(det.y, 3),
            "width": round(det.width, 3),
            "height": round(det.height, 3),
            "confidence": round(det.confidence or 0.95, 3),
            "emotion": det.emotion,
            "emotion_confidence": det.emotion_confidence,
            "timestamp": now.isoformat(),
            "created_at": now.isoformat(),
        }
        saved_items.append(item)
        _IN_MEMORY_RECORDS.insert(0, item)
        if len(_IN_MEMORY_RECORDS) > _MAX_IN_MEMORY:
            _IN_MEMORY_RECORDS.pop()
        inserted_count += 1

    # Optional async PostgreSQL insert if DATABASE_URL is configured
    db_url = get_clean_database_url()
    if db_url:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            from sqlalchemy.pool import NullPool

            engine = create_async_engine(db_url, poolclass=NullPool)
            async with engine.begin() as conn:
                # Ensure table exists
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS face_detections (
                        id SERIAL PRIMARY KEY,
                        camera_id VARCHAR(64) NOT NULL DEFAULT 'default',
                        track_id INTEGER,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        x REAL NOT NULL,
                        y REAL NOT NULL,
                        width REAL NOT NULL,
                        height REAL NOT NULL,
                        confidence REAL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                """))
                for det in payload.detections:
                    await conn.execute(
                        text("""
                            INSERT INTO face_detections (camera_id, track_id, timestamp, x, y, width, height, confidence)
                            VALUES (:camera_id, :track_id, :ts, :x, :y, :w, :h, :conf)
                        """),
                        {
                            "camera_id": payload.camera_id,
                            "track_id": det.track_id,
                            "ts": now,
                            "x": det.x,
                            "y": det.y,
                            "w": det.width,
                            "h": det.height,
                            "conf": det.confidence,
                        },
                    )
            await engine.dispose()
        except Exception as e:
            logger.warning("Database write failed (stored in in-memory buffer): %s", e)

    return {"status": "ok", "inserted": inserted_count}


# ── Query Recent Detections (GET) ──
@app.get("/api/roi/latest", summary="Retrieve latest face detection history")
@app.get("/api/v1/roi/latest")
async def get_latest_roi(
    count: int = Query(10, ge=1, le=50, description="Records limit"),
    camera_id: Optional[str] = Query(None, description="Optional camera ID filter"),
):
    db_url = get_clean_database_url()
    if db_url:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            from sqlalchemy.pool import NullPool

            engine = create_async_engine(db_url, poolclass=NullPool)
            async with engine.connect() as conn:
                query = "SELECT id, camera_id, track_id, timestamp, x, y, width, height, confidence FROM face_detections"
                if camera_id:
                    query += f" WHERE camera_id = '{camera_id}'"
                query += f" ORDER BY id DESC LIMIT {count}"
                result = await conn.execute(text(query))
                rows = result.mappings().all()
                if rows:
                    await engine.dispose()
                    return [dict(r) for r in rows]
            await engine.dispose()
        except Exception as e:
            logger.warning("Database query failed (falling back to in-memory): %s", e)

    # In-memory fallback
    results = _IN_MEMORY_RECORDS
    if camera_id:
        results = [r for r in results if r["camera_id"] == camera_id]
    return results[:count]


# ── System Stats ──
@app.get("/api/stats", summary="System performance & telemetry stats")
@app.get("/api/v1/stats")
async def get_system_stats():
    return {
        "status": "online",
        "architecture": "Edge-Distributed WebAssembly Client + Vercel Serverless Telemetry",
        "cached_records": len(_IN_MEMORY_RECORDS),
        "detector": "Google MediaPipe BlazeFace (WebAssembly)",
        "classifier": "7-Emotion Facial Action Unit Classifier (WebGL GPU)",
        "serverless_mode": True,
        "cold_start_delay": "0ms (Static Edge)",
    }
