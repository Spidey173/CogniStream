"""
Application settings loaded from environment variables using pydantic-settings.
Supports development, testing, and production environments.
"""

from typing import Any, List, Literal, Optional, Union
from functools import lru_cache
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Environment ---
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    APP_NAME: str = "CogniStream AI"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- Security & Auth ---
    API_KEY: str = "dev-secret-api-key"
    ALLOWED_ORIGINS: Union[List[str], str] = ["*"]
    MAX_FRAME_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/facedetect"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://postgres:postgres@db:5432/facedetect"

    # Database Batching (Ultra-fast 200ms real-time flush)
    BATCH_SIZE: int = 5
    BATCH_FLUSH_INTERVAL: float = 0.2  # seconds

    # --- Redis Pub/Sub ---
    REDIS_URL: Optional[str] = "redis://redis:6379/0"
    REDIS_FRAME_CHANNEL: str = "video_frames"
    REDIS_EVENT_CHANNEL: str = "detection_events"

    # --- Vision & Detection ---
    DETECTOR_TYPE: str = "mediapipe"  # "mediapipe" or "mock"
    DETECTION_CONFIDENCE: float = 0.35
    MAX_VIEWERS: int = 50

    # --- Emotion Preprocessing ---
    EMOTION_FACE_PADDING: float = 0.20          # Face crop padding ratio (0.15–0.25)
    EMOTION_MIN_FACE_SIZE: int = 48             # Minimum face dimension in pixels
    EMOTION_BLUR_THRESHOLD: float = 15.0        # Hard blur rejection (Laplacian variance)
    EMOTION_DEBUG: bool = False                 # Save debug crops to disk
    EMOTION_DEBUG_DIR: str = "./debug_crops"    # Debug output directory

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            return v
        url = v.strip()
        # Handle SQLite or in-memory
        if "sqlite" in url:
            return url
        # Convert postgres:// or postgresql:// to postgresql+asyncpg://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

        # Sanitize query parameters for asyncpg compatibility (e.g. Neon / Render)
        try:
            parsed = urlparse(url)
            if parsed.query:
                query_params = parse_qs(parsed.query)
                clean_params = {}
                # Translate sslmode to ssl for asyncpg
                if "sslmode" in query_params:
                    ssl_val = query_params["sslmode"][0]
                    if ssl_val in ["require", "verify-ca", "verify-full"]:
                        clean_params["ssl"] = "require"
                elif "ssl" in query_params:
                    clean_params["ssl"] = query_params["ssl"][0]
                
                # Keep other standard asyncpg supported params if any, drop unknown channel_binding
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

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v_trimmed = v.strip()
            if v_trimmed.startswith("[") and v_trimmed.endswith("]"):
                import json
                try:
                    return json.loads(v_trimmed)
                except Exception:
                    pass
            return [origin.strip() for origin in v_trimmed.split(",") if origin.strip()]
        return ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
