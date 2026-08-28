"""
Video streaming and camera ingestion routes.
Includes camera-scoped WebSockets, MJPEG stream delivery, JPEG snapshots, and camera telemetry.
"""

import asyncio
import io
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.metrics import (
    ACTIVE_WS_CONNECTIONS,
    FPS_GAUGE,
    FRAMES_PROCESSED_TOTAL,
    FRAME_PROCESSING_LATENCY,
    TRACKED_FACES_GAUGE,
)
from app.core.security import authenticate_websocket, get_current_api_key
from app.domain.frame import Frame
from app.services.batch_processor import roi_batch_processor
from app.services.camera import camera_registry
from app.services.face_detection import BaseDetector, get_detector, list_detectors
from app.services.pipeline import DefaultVisionPipeline
from app.services.stream_manager import stream_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])
settings = get_settings()

# Default pipeline singleton
detector_instance: Optional[BaseDetector] = None
vision_pipeline: Optional[DefaultVisionPipeline] = None

# Frame counter & FPS tracking state per camera
_camera_frame_counters: Dict[str, int] = {}
_camera_fps_trackers: Dict[str, float] = {}


def init_detector(detector: Optional[BaseDetector] = None) -> None:
    """Initialize vision detector and pipeline."""
    global detector_instance, vision_pipeline
    if detector:
        detector_instance = detector
    else:
        detector_instance = get_detector(settings.DETECTOR_TYPE, min_confidence=settings.DETECTION_CONFIDENCE)
    vision_pipeline = DefaultVisionPipeline(detector_instance)
    logger.info("Vision pipeline initialized with detector: %s", detector_instance.name)


# ---------------------------------------------------------------------------
# WebSocket Ingestion Endpoint
# ---------------------------------------------------------------------------

@router.websocket("/stream")
@router.websocket("/stream/{camera_id}")
async def receive_stream(websocket: WebSocket, camera_id: str = "default"):
    """
    Ingest binary JPEG video frames over WebSocket from camera clients.
    Applies security validation, payload size limit, vision pipeline execution,
    MessageBus broadcasting, and async DB event queuing.
    """
    if not await authenticate_websocket(websocket):
        return

    await websocket.accept()
    ACTIVE_WS_CONNECTIONS.labels(type="camera", camera_id=camera_id).inc()

    camera_source = camera_registry.get_or_create_browser_camera(camera_id)
    await camera_source.start()

    global _camera_frame_counters, _camera_fps_trackers
    if camera_id not in _camera_frame_counters:
        _camera_frame_counters[camera_id] = 0

    fps_start_time = time.perf_counter()
    fps_frame_count = 0

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()

            # Payload size limit validation
            if len(raw_bytes) > settings.MAX_FRAME_SIZE_BYTES:
                logger.warning("Frame payload size (%d bytes) exceeds limit on camera %s", len(raw_bytes), camera_id)
                await websocket.close(code=1009, reason="Payload too large")
                break

            _camera_frame_counters[camera_id] += 1
            frame_id = _camera_frame_counters[camera_id]

            start_proc = time.perf_counter()

            # Construct Domain Frame
            frame = Frame(camera_id=camera_id, frame_id=frame_id, image=None)

            # Execute Vision Processing Pipeline
            if vision_pipeline:
                ctx = await vision_pipeline.execute(frame, raw_jpeg=raw_bytes)
                proc_elapsed = time.perf_counter() - start_proc
                FRAME_PROCESSING_LATENCY.labels(camera_id=camera_id, stage="pipeline").observe(proc_elapsed)

                # Push processed JPEG to stream manager & message bus
                if ctx.processed_jpeg:
                    await stream_manager.broadcast_frame(camera_id, ctx.processed_jpeg)

                # Queue detection result event for async DB batching
                if ctx.detection_result and ctx.detection_result.detections:
                    roi_batch_processor.enqueue(
                        ctx.detection_result.to_dict(),
                        frame_width=ctx.frame.width,
                        frame_height=ctx.frame.height,
                    )

                faces_found = ctx.detection_result.faces_count if ctx.detection_result else 0
                TRACKED_FACES_GAUGE.labels(camera_id=camera_id).set(faces_found)

            FRAMES_PROCESSED_TOTAL.labels(
                camera_id=camera_id,
                detector=detector_instance.name if detector_instance else "unknown",
            ).inc()

            # Update FPS Gauge
            fps_frame_count += 1
            now = time.perf_counter()
            if now - fps_start_time >= 1.0:
                fps = fps_frame_count / (now - fps_start_time)
                _camera_fps_trackers[camera_id] = fps
                FPS_GAUGE.labels(camera_id=camera_id).set(fps)
                fps_frame_count = 0
                fps_start_time = now

            # Extract live face analysis payload
            detections_payload = []
            if vision_pipeline and ctx and ctx.tracked_detections:
                w_denom = max(1, ctx.frame.width or 480)
                h_denom = max(1, ctx.frame.height or 360)
                for det in ctx.tracked_detections:
                    em = det.emotion or "Neutral"
                    em_conf = det.emotion_confidence if det.emotion_confidence is not None else 0.0
                    condition_map = {
                        "Happy": "Engaged & Positive 😊",
                        "Neutral": "Calm & Attentive 😐",
                        "Surprise": "High Alert / Alerted 😲",
                        "Sad": "Fatigue / Low Energy 😔",
                        "Angry": "Tension / Agitation 😠",
                        "Fear": "Distressed / High Stress 😨",
                        "Disgust": "Aversion / Disapproval 😒",
                        "Contempt": "Skeptical / Evaluating 🤔",
                    }
                    cond_desc = condition_map.get(em, "Analyzing State")
                    detections_payload.append({
                        "track_id": det.track_id,
                        "emotion": em,
                        "emotion_confidence": round(em_conf, 1),
                        "condition": cond_desc,
                        "probabilities": det.emotion_probabilities or {},
                        "x": round(det.x_min / w_denom, 3),
                        "y": round(det.y_min / h_denom, 3),
                        "width": round((det.x_max - det.x_min) / w_denom, 3),
                        "height": round((det.y_max - det.y_min) / h_denom, 3),
                        "confidence": round(det.confidence, 3),
                    })

            # Ack back to camera client with instant real-time AI face analysis
            await websocket.send_json({
                "status": "ok",
                "camera_id": camera_id,
                "frame_id": frame_id,
                "faces": len(detections_payload),
                "detections": detections_payload,
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    except WebSocketDisconnect:
        logger.info("Camera %s disconnected", camera_id)
    except Exception as e:
        logger.error("Error in stream ingestion for camera %s: %s", camera_id, e, exc_info=True)
    finally:
        ACTIVE_WS_CONNECTIONS.labels(type="camera", camera_id=camera_id).dec()
        await camera_source.stop()


# ---------------------------------------------------------------------------
# GET /video — MJPEG Stream for Viewers
# ---------------------------------------------------------------------------

async def _mjpeg_generator(camera_id: str):
    """Yield MJPEG boundary frames subscribed from stream_manager/MessageBus."""
    queue = await stream_manager.subscribe_camera(camera_id)
    try:
        while True:
            try:
                frame_bytes = await asyncio.wait_for(queue.get(), timeout=0.3)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            except asyncio.TimeoutError:
                cached = stream_manager.get_latest_frame(camera_id)
                if cached:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + cached + b"\r\n"
                    )
                await asyncio.sleep(0.03)
    finally:
        await stream_manager.unsubscribe_camera(camera_id, queue)


@router.get("/video", summary="MJPEG processed video stream (default camera)")
@router.get("/video/{camera_id}", summary="MJPEG processed video stream for camera_id")
async def video_feed(camera_id: str = "default"):
    """Serves continuous multipart MJPEG stream playable directly in <img> tags."""
    return StreamingResponse(
        _mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# GET /snapshot/{camera_id}
# ---------------------------------------------------------------------------

@router.get("/snapshot", summary="Get latest JPEG snapshot (default camera)")
@router.get("/snapshot/{camera_id}", summary="Get latest processed JPEG snapshot for camera_id")
async def get_snapshot(camera_id: str = "default"):
    """Returns the most recently processed video frame as a JPEG image."""
    frame_bytes = stream_manager.get_latest_frame(camera_id)
    if not frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active processed frames available for camera '{camera_id}'",
        )
    return Response(content=frame_bytes, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Telemetry & Info Endpoints
# ---------------------------------------------------------------------------

@router.get("/stats", summary="Real-time system telemetry statistics")
async def get_system_stats(api_key: str = Depends(get_current_api_key)):
    """Returns active streams, total viewers, FPS, and detector stats."""
    return {
        "active_cameras": camera_registry.list_cameras(),
        "total_viewers": stream_manager.total_viewers,
        "camera_fps": {k: round(v, 1) for k, v in _camera_fps_trackers.items()},
        "detector": {
            "name": detector_instance.name if detector_instance else "None",
            "version": detector_instance.version if detector_instance else "Unknown",
            "backend": detector_instance.backend if detector_instance else "Unknown",
        },
    }


@router.get("/cameras", summary="List active camera sources")
async def list_cameras(api_key: str = Depends(get_current_api_key)):
    """Returns list of registered active camera sources."""
    return {"cameras": camera_registry.list_cameras()}


@router.get("/detectors", summary="List available vision detector backends")
async def get_available_detectors(api_key: str = Depends(get_current_api_key)):
    """Returns list of pluggable detector backends registered in factory."""
    return {"detectors": list_detectors()}
