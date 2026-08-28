"""
FastAPI application entry point.
Clean lifespan setup for DB tables, vision pipeline, MessageBus, and ROIBatchProcessor worker.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.metrics import HTTP_REQUESTS_TOTAL
from app.db.base import Base
from app.db.session import engine
from app.services.batch_processor import roi_batch_processor
from app.services.message_bus import init_message_bus, message_bus
from app.api.routes import health as health_routes
from app.api.routes import roi as roi_routes
from app.api.routes import stream as stream_routes

settings = get_settings()

# Setup logging
setup_logging(log_level=settings.LOG_LEVEL, json_format=(settings.ENVIRONMENT == "production"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Prepare Database tables
      2. Initialize MessageBus (Redis Pub/Sub or In-Memory)
      3. Initialize Vision Pipeline & Detector
      4. Start ROIBatchProcessor async flush task

    Shutdown:
      1. Stop ROIBatchProcessor (flush remaining queue items to DB)
      2. Close MessageBus connections
      3. Dispose SQLAlchemy connection pool
    """
    logger.info("Starting %s in [%s] mode", settings.APP_NAME, settings.ENVIRONMENT)

    # 1. DB Init
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified")

    # 2. MessageBus Init
    await init_message_bus()

    # 3. Vision Pipeline Init
    stream_routes.init_detector()
    from app.services.emotion import emotion_classifier
    if emotion_classifier.is_loaded:
        logger.info(
            "🧠 Emotion Classifier Status: LOADED (model_path='%s')",
            emotion_classifier.model_path,
        )
    else:
        logger.error(
            "🚨 Emotion Classifier Status: FAILED TO LOAD (model_path='%s', error='%s')",
            emotion_classifier.model_path,
            emotion_classifier.load_error,
        )

    # 4. Batch Processor Init
    await roi_batch_processor.start()

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.APP_NAME)
    await roi_batch_processor.stop()
    if hasattr(message_bus, "close"):
        await message_bus.close()
    await engine.dispose()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Enterprise Real-Time Computer Vision Streaming Platform. "
            "Decoupled vision pipeline with Pluggable Detectors, Centroid Tracking, "
            "MessageBus/EventBus horizontal scaling, and Async ROI DB Batching."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS (supports wildcard regex for Vercel and Render dynamic preview domains)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS if settings.ALLOWED_ORIGINS != ["*"] else [],
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics Middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        response = await call_next(request)
        endpoint = request.url.path
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        return response

    # Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled global exception on %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "path": request.url.path},
        )

    # Include API Routers under /api/v1
    app.include_router(health_routes.router)  # /health, /ready, /metrics
    app.include_router(stream_routes.router, prefix="/api/v1")
    app.include_router(roi_routes.router, prefix="/api/v1")

    return app


app = create_app()
