"""
Shared test fixtures using in-memory SQLite and MockDetector.
"""

import io
import pytest
import pytest_asyncio
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.api.routes import stream as stream_routes
from app.api.routes import roi as roi_routes
from app.api.routes import health as health_routes
from app.core.config import get_settings
from app.services.face_detection import MockDetector

settings = get_settings()

_test_engine = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

_TestSession = async_sessionmaker(
    bind=_test_engine, class_=AsyncSession, expire_on_commit=False
)


async def _override_get_db():
    async with _TestSession() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest_asyncio.fixture
async def db_session():
    """Async session fixture for CRUD tests."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with _TestSession() as session:
        yield session


@pytest.fixture(scope="session")
def test_detector():
    """MockDetector fixture for testing."""
    return MockDetector(simulate_face=True, latency_ms=0.1)


@pytest.fixture(scope="session")
def app(test_detector):
    """Test FastAPI app instance."""
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        async with _test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        stream_routes.init_detector(test_detector)
        yield
        await _test_engine.dispose()

    app = FastAPI(lifespan=_lifespan)
    app.include_router(health_routes.router)
    app.include_router(stream_routes.router, prefix="/api/v1")
    app.include_router(roi_routes.router, prefix="/api/v1")

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dummy_jpeg() -> bytes:
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
