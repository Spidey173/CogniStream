"""
ROI Query endpoints for retrieving face detection bounding box data.
Supports limit, offset, and camera_id filtering.
"""

import math
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import fetch_latest_roi, fetch_roi_paginated
from app.db.session import get_db
from app.schemas.roi import ROIListResponse, ROIPaginatedResponse, ROIResponse

router = APIRouter(tags=["roi"])


@router.get(
    "/roi/latest",
    response_model=List[ROIResponse],
    summary="Get N most recent face detections",
)
async def get_latest_roi(
    count: int = Query(10, ge=1, le=100, description="Number of recent records to return"),
    camera_id: Optional[str] = Query(None, description="Optional camera ID filter"),
    db: AsyncSession = Depends(get_db),
):
    """Fetch the N most recent face detection records from database."""
    records = await fetch_latest_roi(db, limit=count, camera_id=camera_id)
    return records


@router.get(
    "/roi",
    response_model=ROIPaginatedResponse,
    summary="Get paginated list of face detection records",
)
async def list_roi(
    limit: int = Query(10, ge=1, le=100, description="Page limit size"),
    offset: int = Query(0, ge=0, description="Offset position"),
    camera_id: Optional[str] = Query(None, description="Optional camera ID filter"),
    db: AsyncSession = Depends(get_db),
):
    """Query paginated detection history with metadata."""
    total, records = await fetch_roi_paginated(db, limit=limit, offset=offset, camera_id=camera_id)
    pages = math.ceil(total / limit) if limit > 0 else 0
    return ROIPaginatedResponse(
        total=total,
        limit=limit,
        offset=offset,
        pages=pages,
        results=records,
    )
