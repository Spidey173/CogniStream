"""
ROIBatchProcessor — Asynchronous event queue consumer for bulk PostgreSQL inserts.
Prevents per-frame database write bottlenecks by accumulating detection events
and executing bulk inserts based on configured BATCH_SIZE or BATCH_FLUSH_INTERVAL.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.metrics import DB_BATCH_INSERT_LATENCY
from app.db.crud import insert_roi_batch
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)
settings = get_settings()


class ROIBatchProcessor:
    """Async background worker accumulating ROI detection events and performing bulk PostgreSQL inserts."""

    def __init__(self, batch_size: int = 50, flush_interval: float = 2.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._worker_task: asyncio.Task | None = None
        self._is_running = False

    def enqueue(
        self,
        event_data: Dict[str, Any],
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        """Enqueue detection event for batch insertion with normalized coordinates."""
        if not self._is_running:
            return

        camera_id = event_data.get("camera_id", "default")
        detections = event_data.get("detections", [])

        w_denom = max(1, frame_width)
        h_denom = max(1, frame_height)

        for det in detections:
            x_min = float(det.get("x_min", 0))
            y_min = float(det.get("y_min", 0))
            x_max = float(det.get("x_max", 0))
            y_max = float(det.get("y_max", 0))

            # Normalize coordinates to 0.0 - 1.0 range
            norm_x = min(1.0, max(0.0, x_min / w_denom))
            norm_y = min(1.0, max(0.0, y_min / h_denom))
            norm_w = min(1.0, max(0.0, (x_max - x_min) / w_denom))
            norm_h = min(1.0, max(0.0, (y_max - y_min) / h_denom))

            roi_item = {
                "camera_id": camera_id,
                "x": norm_x,
                "y": norm_y,
                "width": norm_w,
                "height": norm_h,
                "confidence": float(det.get("confidence", 0.0)),
                "track_id": det.get("track_id"),
                "timestamp": datetime.now(timezone.utc),
            }
            try:
                self._queue.put_nowait(roi_item)
            except asyncio.QueueFull:
                logger.warning("ROI batch queue full, dropping item for camera %s", camera_id)

    async def start(self) -> None:
        """Start the background flush loop task."""
        if self._is_running:
            return
        self._is_running = True
        self._worker_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "ROIBatchProcessor started (batch_size=%d, interval=%.1fs)",
            self.batch_size,
            self.flush_interval,
        )

    async def stop(self) -> None:
        """Gracefully stop worker task and perform final flush."""
        if not self._is_running:
            return
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        logger.info("ROIBatchProcessor stopped and flushed remaining items")

    async def flush(self) -> int:
        """Flush currently queued ROI records in a single bulk database transaction."""
        items: List[Dict[str, Any]] = []
        while not self._queue.empty() and len(items) < self.batch_size:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not items:
            return 0

        start_time = time.perf_counter()
        try:
            async with AsyncSessionLocal() as db:
                inserted = await insert_roi_batch(db, items)
                elapsed = time.perf_counter() - start_time
                DB_BATCH_INSERT_LATENCY.observe(elapsed)
                logger.debug("Flushed %d ROI items to DB in %.3fs", inserted, elapsed)
                return inserted
        except Exception as e:
            logger.error("Failed executing ROI bulk DB insert batch: %s", e, exc_info=True)
            return 0

    async def _flush_loop(self) -> None:
        """Periodic loop flushing batch queue."""
        while self._is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in ROI batch flush loop: %s", e, exc_info=True)


# Global singleton instance
roi_batch_processor = ROIBatchProcessor(
    batch_size=settings.BATCH_SIZE,
    flush_interval=settings.BATCH_FLUSH_INTERVAL,
)
