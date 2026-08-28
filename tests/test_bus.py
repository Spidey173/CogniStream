"""Unit tests for MessageBus and EventBus."""

import pytest
from app.services.message_bus import InMemoryMessageBus


@pytest.mark.asyncio
async def test_in_memory_message_bus_frame_pub_sub():
    bus = InMemoryMessageBus()
    queue = await bus.subscribe_frame("cam1")

    frame_data = b"\xff\xd8\xff\xe0test_jpeg"
    await bus.publish_frame("cam1", frame_data)

    received = queue.get_nowait()
    assert received == frame_data
    assert bus.get_latest_frame("cam1") == frame_data

    await bus.unsubscribe_frame("cam1", queue)


@pytest.mark.asyncio
async def test_in_memory_event_bus():
    bus = InMemoryMessageBus()
    received_events = []

    def callback(data):
        received_events.append(data)

    bus.subscribe_event("face_detected", callback)
    event = {"camera_id": "cam1", "faces": 1}
    await bus.publish_event("face_detected", event)

    assert len(received_events) == 1
    assert received_events[0] == event
