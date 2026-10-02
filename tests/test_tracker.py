"""Unit tests for CentroidTracker."""

from app.domain.detection import BoundingBox
from app.services.tracker import CentroidTracker


def test_centroid_tracker_assigns_persistent_ids():
    tracker = CentroidTracker(max_disappeared=5, max_distance=50.0)

    # Frame 1: Detection at (10, 10) to (30, 30) -> center (20, 20)
    bbox1 = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30, confidence=0.9)
    tracked1 = tracker.update([bbox1])

    assert len(tracked1) == 1
    track_id_1 = tracked1[0].track_id
    assert track_id_1 is not None

    # Frame 2: Slight movement to (12, 12) to (32, 32) -> center (22, 22)
    bbox2 = BoundingBox(x_min=12, y_min=12, x_max=32, y_max=32, confidence=0.9)
    tracked2 = tracker.update([bbox2])

    assert len(tracked2) == 1
    assert tracked2[0].track_id == track_id_1


def test_centroid_tracker_handles_empty_frame():
    tracker = CentroidTracker(max_disappeared=2)
    bbox = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30, confidence=0.9)
    tracker.update([bbox])

    # Disappear for 3 frames
    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert len(tracker.objects) == 0
