"""Unit tests for Face Detector implementations and registry."""

import pytest
import numpy as np
from app.services.face_detection import (
    MediaPipeDetector,
    MockDetector,
    get_detector,
    list_detectors,
    register_detector,
)


def test_mock_detector():
    detector = MockDetector(simulate_face=True, latency_ms=0.1)
    assert detector.name == "Mock Detector"
    assert detector.backend == "Synthetic"

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = detector.detect(frame, frame_id=1, camera_id="cam1")
    assert res.faces_count == 1
    assert len(res.detections) == 1
    assert res.detections[0].confidence == 0.95


def test_detector_registry():
    detectors = list_detectors()
    assert any(d["key"] == "mediapipe" for d in detectors)
    assert any(d["key"] == "mock" for d in detectors)

    det = get_detector("mock", simulate_face=False)
    assert isinstance(det, MockDetector)
