"""Unit tests for ROIBatchProcessor."""

import pytest
from app.services.batch_processor import ROIBatchProcessor


@pytest.mark.asyncio
async def test_roi_batch_processor_enqueue():
    processor = ROIBatchProcessor(batch_size=5, flush_interval=1.0)
    await processor.start()

    event = {
        "camera_id": "cam1",
        "detections": [
            {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50, "confidence": 0.95, "track_id": 1}
        ]
    }

    processor.enqueue(event)
    assert processor._queue.qsize() == 1

    await processor.stop()
