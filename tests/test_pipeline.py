"""Unit tests for Vision Processing Pipeline."""

import pytest
import io
import numpy as np
from PIL import Image

from app.domain.frame import Frame
from app.services.face_detection import MockDetector
from app.services.pipeline import DefaultVisionPipeline


@pytest.mark.asyncio
async def test_pipeline_execution():
    detector = MockDetector(simulate_face=True, latency_ms=0.1)
    pipeline = DefaultVisionPipeline(detector)

    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_jpeg = buf.getvalue()

    frame = Frame(camera_id="cam_test", frame_id=1, image=None)
    ctx = await pipeline.execute(frame, raw_jpeg=raw_jpeg)

    assert ctx.frame.image is not None
    assert ctx.detection_result is not None
    assert ctx.detection_result.faces_count == 1
    assert ctx.processed_jpeg is not None
