"""
Pydantic schemas for face detection ROI data and paginated API responses.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ROIBase(BaseModel):
    x: float = Field(..., description="X coordinate of bounding box origin")
    y: float = Field(..., description="Y coordinate of bounding box origin")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class ROICreate(ROIBase):
    timestamp: datetime
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    camera_id: str = "default"
    track_id: Optional[int] = None


class ROIResponse(ROIBase):
    id: int
    camera_id: str = "default"
    track_id: Optional[int] = None
    timestamp: datetime
    confidence: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ROIListResponse(BaseModel):
    count: int
    results: List[ROIResponse]


class ROIPaginatedResponse(BaseModel):
    total: int
    limit: int
    offset: int
    pages: int
    results: List[ROIResponse]
