"""Frame domain model representing a single ingested video frame."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict
import numpy as np


@dataclass
class Frame:
    """Represents a video frame in the computer vision pipeline."""

    camera_id: str
    frame_id: int
    image: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0]) if self.image is not None and self.image.ndim >= 2 else 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1]) if self.image is not None and self.image.ndim >= 2 else 0

    @property
    def channels(self) -> int:
        return int(self.image.shape[2]) if self.image is not None and self.image.ndim == 3 else 1
