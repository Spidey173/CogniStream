"""
StreamManager — Manages real-time video stream subscriber connections and telemetry stats.
Integrates with MessageBus for multi-instance distributed frame delivery.
"""

import asyncio
import logging
from typing import Dict, Optional, Set

from app.core.config import get_settings
from app.services.message_bus import message_bus

logger = logging.getLogger(__name__)
settings = get_settings()


class StreamManager:
    """Manages active viewer streams per camera_id."""

    def __init__(self):
        # camera_id -> Set of viewer_id
        self._viewers: Dict[str, Set[str]] = {}

    @property
    def total_viewers(self) -> int:
        return sum(len(v) for v in self._viewers.values())

    def add_viewer(self, camera_id: str, viewer_id: str) -> None:
        if self.total_viewers >= settings.MAX_VIEWERS:
            raise ConnectionError(f"Max viewers ({settings.MAX_VIEWERS}) limit reached.")
        if camera_id not in self._viewers:
            self._viewers[camera_id] = set()
        self._viewers[camera_id].add(viewer_id)
        logger.info("Viewer %s connected to camera %s (total: %d)", viewer_id, camera_id, self.total_viewers)

    def remove_viewer(self, camera_id: str, viewer_id: str) -> None:
        if camera_id in self._viewers:
            self._viewers[camera_id].discard(viewer_id)
            if not self._viewers[camera_id]:
                self._viewers.pop(camera_id, None)

    async def broadcast_frame(self, camera_id: str, frame_bytes: bytes) -> None:
        """Publish frame to message bus."""
        await message_bus.publish_frame(camera_id, frame_bytes)

    async def subscribe_camera(self, camera_id: str) -> asyncio.Queue[bytes]:
        """Subscribe viewer to camera frame stream queue."""
        return await message_bus.subscribe_frame(camera_id)

    async def unsubscribe_camera(self, camera_id: str, queue: asyncio.Queue[bytes]) -> None:
        """Unsubscribe viewer queue."""
        await message_bus.unsubscribe_frame(camera_id, queue)

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        """Get latest cached processed frame JPEG for snapshot API."""
        if hasattr(message_bus, "get_latest_frame"):
            return message_bus.get_latest_frame(camera_id)
        return None


# Global StreamManager singleton
stream_manager = StreamManager()
