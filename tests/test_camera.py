"""Unit tests for camera source abstraction and CameraRegistry."""

import pytest
import numpy as np
from app.services.camera import (
    BrowserCameraSource,
    MockCameraSource,
    CameraRegistry,
)


@pytest.mark.asyncio
async def test_mock_camera_source():
    source = MockCameraSource(camera_id="cam_mock", width=320, height=240)
    await source.start()
    assert source.is_active

    frame = await source.read_frame()
    assert frame is not None
    assert frame.shape == (240, 320, 3)

    info = source.get_info()
    assert info.camera_id == "cam_mock"

    await source.stop()
    assert not source.is_active


@pytest.mark.asyncio
async def test_browser_camera_source():
    source = BrowserCameraSource(camera_id="cam_browser")
    await source.start()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    source.push_frame(img)

    frame = await source.read_frame()
    assert frame is not None
    assert frame.shape == (100, 100, 3)

    await source.stop()


def test_camera_registry():
    registry = CameraRegistry()
    source = MockCameraSource(camera_id="cam_reg")
    registry.register(source)

    assert registry.get("cam_reg") == source
    assert len(registry.list_cameras()) == 1

    registry.unregister("cam_reg")
    assert registry.get("cam_reg") is None
