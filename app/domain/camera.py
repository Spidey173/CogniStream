"""CameraInfo domain model."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any


@dataclass
class CameraInfo:
    """Domain model holding camera source metadata."""

    camera_id: str
    source_type: str = "browser"
    width: int = 640
    height: int = 480
    fps: float = 0.0
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "source_type": self.source_type,
            "resolution": f"{self.width}x{self.height}",
            "fps": round(self.fps, 1),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
