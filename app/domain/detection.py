"""Detection domain models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-coordinate bounding box."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    track_id: Optional[int] = None
    emotion: Optional[str] = "Neutral"
    emotion_confidence: Optional[float] = 88.0
    emotion_probabilities: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "confidence": float(self.confidence),
            "track_id": self.track_id,
            "emotion": self.emotion,
            "emotion_confidence": float(self.emotion_confidence) if self.emotion_confidence is not None else None,
            "emotion_probabilities": self.emotion_probabilities,
        }

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x_min + self.width // 2, self.y_min + self.height // 2)


@dataclass
class DetectionResult:
    """Contains face detection results for a frame."""

    frame_id: int
    camera_id: str
    detections: List[BoundingBox] = field(default_factory=list)
    inference_time_ms: float = 0.0
    detector_name: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def faces_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "faces_count": self.faces_count,
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detector_name": self.detector_name,
            "timestamp": self.timestamp.isoformat(),
        }
