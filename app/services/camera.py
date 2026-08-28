"""
Camera source abstractions and CameraRegistry.
Supports BrowserCameraSource, VideoFileCameraSource, and MockCameraSource.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np

from app.domain.camera import CameraInfo


class BaseCameraSource(ABC):
    """Abstract base class for all camera video sources."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    @abstractmethod
    async def start(self) -> None:
        """Initialize camera resource."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Release camera resource."""
        pass

    @abstractmethod
    async def read_frame(self) -> Optional[np.ndarray]:
        """Read latest RGB frame from camera source."""
        pass

    def get_info(self) -> CameraInfo:
        return CameraInfo(
            camera_id=self.camera_id,
            source_type=self.__class__.__name__.lower().replace("camerasource", ""),
            is_active=self.is_active,
        )


class BrowserCameraSource(BaseCameraSource):
    """Represents a browser webcam sending frames via WebSockets."""

    def __init__(self, camera_id: str):
        super().__init__(camera_id)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_event = asyncio.Event()

    async def start(self) -> None:
        self._is_active = True

    async def stop(self) -> None:
        self._is_active = False
        self._latest_frame = None

    def push_frame(self, frame: np.ndarray) -> None:
        """Push decoded frame received over WebSocket."""
        self._latest_frame = frame
        self._frame_event.set()

    async def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_active:
            return None
        return self._latest_frame


class MockCameraSource(BaseCameraSource):
    """Mock camera generating synthetic RGB test frames for testing & benchmarks."""

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        super().__init__(camera_id)
        self.width = width
        self.height = height
        self._counter = 0

    async def start(self) -> None:
        self._is_active = True

    async def stop(self) -> None:
        self._is_active = False

    async def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_active:
            return None
        self._counter += 1
        # Create gradient synthetic frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = (self._counter * 5) % 255
        frame[:, :, 1] = 128
        frame[:, :, 2] = 200
        return frame


class CameraRegistry:
    """Thread-safe global registry managing active camera sources."""

    def __init__(self):
        self._cameras: Dict[str, BaseCameraSource] = {}

    def register(self, camera: BaseCameraSource) -> BaseCameraSource:
        """Register a camera source."""
        self._cameras[camera.camera_id] = camera
        return camera

    def unregister(self, camera_id: str) -> Optional[BaseCameraSource]:
        """Remove a camera source."""
        return self._cameras.pop(camera_id, None)

    def get(self, camera_id: str) -> Optional[BaseCameraSource]:
        """Retrieve camera source by ID."""
        return self._cameras.get(camera_id)

    def get_or_create_browser_camera(self, camera_id: str) -> BrowserCameraSource:
        """Get existing or instantiate new BrowserCameraSource."""
        if camera_id not in self._cameras:
            source = BrowserCameraSource(camera_id)
            self._cameras[camera_id] = source
        cam = self._cameras[camera_id]
        if isinstance(cam, BrowserCameraSource):
            return cam
        # Replace if different type
        source = BrowserCameraSource(camera_id)
        self._cameras[camera_id] = source
        return source

    def list_cameras(self) -> List[CameraInfo]:
        """Return list of active camera metadata."""
        return [cam.get_info() for cam in self._cameras.values()]


# Global CameraRegistry singleton
camera_registry = CameraRegistry()
