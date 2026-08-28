"""Unit tests for CRUD operations."""

import pytest
from sqlalchemy import delete
from app.models.roi import FaceDetection
from app.db.crud import insert_roi, insert_roi_batch, fetch_latest_roi, fetch_roi_paginated


@pytest.mark.asyncio
async def test_crud_operations(db_session):
    try:
        # Single Insert
        rec = await insert_roi(
            db_session,
            x=0.1,
            y=0.1,
            width=0.4,
            height=0.4,
            confidence=0.9,
            camera_id="cam_crud",
            track_id=1,
        )
        assert rec.id is not None
        assert rec.camera_id == "cam_crud"

        # Fetch Latest
        latest = await fetch_latest_roi(db_session, limit=5, camera_id="cam_crud")
        assert len(latest) == 1

        # Bulk Batch Insert
        batch_items = [
            {"camera_id": "cam_crud", "x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3, "confidence": 0.85, "track_id": 2},
            {"camera_id": "cam_crud", "x": 0.3, "y": 0.3, "width": 0.2, "height": 0.2, "confidence": 0.88, "track_id": 3},
        ]
        inserted = await insert_roi_batch(db_session, batch_items)
        assert inserted == 2

        # Paginated Fetch
        total, items = await fetch_roi_paginated(db_session, limit=10, offset=0, camera_id="cam_crud")
        assert total == 3
        assert len(items) == 3

    finally:
        # Clean up database after test
        await db_session.execute(delete(FaceDetection))
        await db_session.commit()
