"""TrackedEntity domain model."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple, Dict, Any
from app.domain.detection import BoundingBox


@dataclass
class TrackedEntity:
    """Represents a tracked face entity across frames."""

    track_id: int
    bbox: BoundingBox
    centroid: Tuple[int, int]
    disappeared_frames: int = 0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_dict(),
            "centroid": list(self.centroid),
            "disappeared_frames": self.disappeared_frames,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }
