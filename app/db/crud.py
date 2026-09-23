"""
CRUD operations for face detection ROI data.
Supports single insert, bulk batch insertion, and paginated query filtering.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.roi import FaceDetection

logger = logging.getLogger(__name__)


async def insert_roi(
    db: AsyncSession,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    camera_id: str = "default",
    track_id: Optional[int] = None,
) -> FaceDetection:
    """Insert a single face detection record."""
    record = FaceDetection(
        timestamp=timestamp or datetime.now(timezone.utc),
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=confidence,
        camera_id=camera_id,
        track_id=track_id,
    )
    db.add(record)
    try:
        await db.commit()
        await db.refresh(record)
    except Exception:
        await db.rollback()
        logger.error("Failed inserting ROI record", exc_info=True)
        raise
    return record


async def insert_roi_batch(
    db: AsyncSession,
    items: List[Dict[str, Any]],
) -> int:
    """Bulk insert a batch of ROI detection records in a single transaction."""
    if not items:
        return 0

    records = [
        FaceDetection(
            camera_id=item.get("camera_id", "default"),
            track_id=item.get("track_id"),
            x=item["x"],
            y=item["y"],
            width=item["width"],
            height=item["height"],
            confidence=item.get("confidence"),
            timestamp=item.get("timestamp") or datetime.now(timezone.utc),
        )
        for item in items
    ]

    db.add_all(records)
    try:
        await db.commit()
        logger.debug("Successfully bulk inserted %d ROI records", len(records))
        return len(records)
    except Exception:
        await db.rollback()
        logger.error("Failed bulk inserting %d ROI records", len(records), exc_info=True)
        raise


async def fetch_latest_roi(
    db: AsyncSession,
    *,
    limit: int = 10,
    camera_id: Optional[str] = None,
) -> list[FaceDetection]:
    """Fetch the N most recent detection records."""
    query = select(FaceDetection)
    if camera_id:
        query = query.where(FaceDetection.camera_id == camera_id)
    query = query.order_by(FaceDetection.id.desc()).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def fetch_roi_paginated(
    db: AsyncSession,
    *,
    limit: int = 10,
    offset: int = 0,
    camera_id: Optional[str] = None,
) -> Tuple[int, list[FaceDetection]]:
    """Fetch total count and paginated list of detection records."""
    count_query = select(func.count(FaceDetection.id))
    data_query = select(FaceDetection)

    if camera_id:
        count_query = count_query.where(FaceDetection.camera_id == camera_id)
        data_query = data_query.where(FaceDetection.camera_id == camera_id)

    total = (await db.execute(count_query)).scalar_one()

    data_query = data_query.order_by(FaceDetection.timestamp.desc()).offset(offset).limit(limit)
    items = list((await db.execute(data_query)).scalars().all())

    return total, items
