"""
SQLAlchemy ORM model for face detection ROI data.
Maps to the `face_detections` table in PostgreSQL.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Float, DateTime, String, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FaceDetection(Base):
    """Stores bounding box data and entity track IDs for detected faces."""

    __tablename__ = "face_detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)

    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("idx_face_detections_timestamp", "timestamp"),
        Index("idx_face_detections_camera_id", "camera_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceDetection id={self.id} camera={self.camera_id} "
            f"track_id={self.track_id} bbox=({self.x:.2f}, {self.y:.2f}, {self.width:.2f}, {self.height:.2f})>"
        )
