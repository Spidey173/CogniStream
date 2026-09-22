"""Domain models package."""

from app.domain.frame import Frame
from app.domain.detection import BoundingBox, DetectionResult
from app.domain.camera import CameraInfo
from app.domain.tracking import TrackedEntity

__all__ = [
    "Frame",
    "BoundingBox",
    "DetectionResult",
    "CameraInfo",
    "TrackedEntity",
]
