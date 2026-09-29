"""
MessageBus (binary frame streaming) and EventBus (JSON telemetry events) with Redis Pub/Sub and in-memory fallback backends.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set

from app.core.config import get_settings
from app.core.metrics import REDIS_CONNECTED_GAUGE

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseMessageBus(ABC):
    """Interface for high-throughput binary JPEG frame distribution."""

    @abstractmethod
    async def publish_frame(self, camera_id: str, frame_bytes: bytes) -> None:
        pass

    @abstractmethod
    async def subscribe_frame(self, camera_id: str) -> asyncio.Queue[bytes]:
        pass

    @abstractmethod
    async def unsubscribe_frame(self, camera_id: str, queue: asyncio.Queue[bytes]) -> None:
        pass


class BaseEventBus(ABC):
    """Interface for structured JSON detection event distribution."""

    @abstractmethod
    async def publish_event(self, event_type: str, data: dict) -> None:
        pass

    @abstractmethod
    def subscribe_event(self, event_type: str, callback: Callable[[dict], None]) -> None:
        pass


class InMemoryMessageBus(BaseMessageBus, BaseEventBus):
    """In-memory asyncio queue pub/sub bus for local development and unit tests."""

    def __init__(self):
        # camera_id -> Set of subscriber asyncio.Queue
        self._frame_subscribers: Dict[str, Set[asyncio.Queue[bytes]]] = {}
        # event_type -> List of callback callables
        self._event_subscribers: Dict[str, List[Callable[[dict], None]]] = {}
        self._latest_frames: Dict[str, bytes] = {}

    async def publish_frame(self, camera_id: str, frame_bytes: bytes) -> None:
        self._latest_frames[camera_id] = frame_bytes
        subscribers = list(self._frame_subscribers.get(camera_id, set()))
        for q in subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(frame_bytes)
            except Exception as e:
                logger.warning("Failed pushing frame to subscriber: %s", e)

    async def subscribe_frame(self, camera_id: str) -> asyncio.Queue[bytes]:
        if camera_id not in self._frame_subscribers:
            self._frame_subscribers[camera_id] = set()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=5)
        self._frame_subscribers[camera_id].add(queue)
        # Pre-fill with latest frame if present
        if camera_id in self._latest_frames:
            queue.put_nowait(self._latest_frames[camera_id])
        return queue

    async def unsubscribe_frame(self, camera_id: str, queue: asyncio.Queue[bytes]) -> None:
        if camera_id in self._frame_subscribers:
            self._frame_subscribers[camera_id].discard(queue)

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        if camera_id in self._latest_frames:
            return self._latest_frames[camera_id]
        if self._latest_frames:
            return next(iter(self._latest_frames.values()))
        return None

    async def publish_event(self, event_type: str, data: dict) -> None:
        callbacks = self._event_subscribers.get(event_type, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(data)
                else:
                    cb(data)
            except Exception as e:
                logger.error("Error executing event subscriber callback for %s: %s", event_type, e)

    def subscribe_event(self, event_type: str, callback: Callable[[dict], None]) -> None:
        if event_type not in self._event_subscribers:
            self._event_subscribers[event_type] = []
        self._event_subscribers[event_type].append(callback)


class RedisMessageBus(BaseMessageBus, BaseEventBus):
    """Production Redis Pub/Sub implementation for multi-node horizontal scaling."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None
        self._pubsub = None
        self.fallback = InMemoryMessageBus()
        self._is_connected = False

    async def connect(self) -> bool:
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.redis_url, decode_responses=False)
            await self._redis.ping()
            self._is_connected = True
            REDIS_CONNECTED_GAUGE.set(1)
            logger.info("Connected to Redis Pub/Sub at %s", self.redis_url)
            return True
        except Exception as e:
            self._is_connected = False
            REDIS_CONNECTED_GAUGE.set(0)
            logger.warning("Redis connection failed (%s). Falling back to in-memory bus.", e)
            return False

    async def publish_frame(self, camera_id: str, frame_bytes: bytes) -> None:
        await self.fallback.publish_frame(camera_id, frame_bytes)
        if self._is_connected and self._redis:
            try:
                channel = f"{settings.REDIS_FRAME_CHANNEL}:{camera_id}"
                await self._redis.publish(channel, frame_bytes)
            except Exception as e:
                logger.error("Redis publish_frame error: %s", e)

    async def subscribe_frame(self, camera_id: str) -> asyncio.Queue[bytes]:
        return await self.fallback.subscribe_frame(camera_id)

    async def unsubscribe_frame(self, camera_id: str, queue: asyncio.Queue[bytes]) -> None:
        await self.fallback.unsubscribe_frame(camera_id, queue)

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        return self.fallback.get_latest_frame(camera_id)

    async def publish_event(self, event_type: str, data: dict) -> None:
        await self.fallback.publish_event(event_type, data)
        if self._is_connected and self._redis:
            try:
                channel = f"{settings.REDIS_EVENT_CHANNEL}:{event_type}"
                payload = json.dumps(data).encode("utf-8")
                await self._redis.publish(channel, payload)
            except Exception as e:
                logger.error("Redis publish_event error: %s", e)

    def subscribe_event(self, event_type: str, callback: Callable[[dict], None]) -> None:
        self.fallback.subscribe_event(event_type, callback)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._is_connected = False
            REDIS_CONNECTED_GAUGE.set(0)


# Instantiate global message bus singleton
message_bus = InMemoryMessageBus()


async def init_message_bus() -> None:
    """Initialize global message bus based on environment settings."""
    global message_bus
    if settings.REDIS_URL:
        bus = RedisMessageBus(settings.REDIS_URL)
        if await bus.connect():
            message_bus = bus
            return
    message_bus = InMemoryMessageBus()
    logger.info("Initialized InMemoryMessageBus")
