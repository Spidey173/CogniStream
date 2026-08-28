"""Unit tests for domain models."""

from datetime import datetime, timezone
import numpy as np
from app.domain.frame import Frame
from app.domain.detection import BoundingBox, DetectionResult
from app.domain.camera import CameraInfo
from app.domain.tracking import TrackedEntity


def test_frame_properties():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(camera_id="cam1", frame_id=1, image=img)
    assert frame.height == 480
    assert frame.width == 640
    assert frame.channels == 3
    assert frame.camera_id == "cam1"


def test_bounding_box_center_and_to_dict():
    bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=120, confidence=0.9, track_id=5)
    assert bbox.width == 100
    assert bbox.height == 100
    assert bbox.center == (60, 70)
    d = bbox.to_dict()
    assert d["track_id"] == 5
    assert d["confidence"] == 0.9


def test_detection_result_faces_count():
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.8)
    res = DetectionResult(frame_id=1, camera_id="cam1", detections=[bbox], inference_time_ms=5.2)
    assert res.faces_count == 1
    d = res.to_dict()
    assert d["faces_count"] == 1
    assert len(d["detections"]) == 1


def test_camera_info_to_dict():
    info = CameraInfo(camera_id="cam_test", source_type="browser", width=1280, height=720)
    d = info.to_dict()
    assert d["camera_id"] == "cam_test"
    assert d["resolution"] == "1280x720"


def test_tracked_entity():
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.8)
    entity = TrackedEntity(track_id=1, bbox=bbox, centroid=(5, 5))
    assert entity.track_id == 1
    assert entity.disappeared_frames == 0
