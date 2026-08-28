"""
Vision Processing Pipeline architecture.
Defines modular processing stages (Decode, Detect, Track, Emotion, Annotate, Publish).
"""

import cv2
import io
import time
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional
import numpy as np

from app.domain.frame import Frame
from app.domain.detection import DetectionResult, BoundingBox
from app.services.face_detection import BaseDetector
from app.services.tracker import CentroidTracker
from app.services.emotion import emotion_classifier, extract_padded_square_crop
from app.services.draw import draw_bbox

logger = logging.getLogger(__name__)


class PipelineContext:
    """Carries mutable state across pipeline stages for a single frame execution."""

    def __init__(self, frame: Frame):
        self.frame = frame
        self.raw_jpeg: Optional[bytes] = None
        self.processed_jpeg: Optional[bytes] = None
        self.detection_result: Optional[DetectionResult] = None
        self.tracked_detections: List[BoundingBox] = []
        self.metadata: Dict[str, str] = {}


class PipelineStage(ABC):
    """Abstract base class for a vision processing pipeline stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        pass


class DecodeStage(PipelineStage):
    """Stage 1: Fast OpenCV decode raw JPEG binary bytes into NumPy RGB image array."""

    @property
    def name(self) -> str:
        return "Decode"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.raw_jpeg is not None:
            nparr = np.frombuffer(ctx.raw_jpeg, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is not None:
                ctx.frame.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ctx


class DetectStage(PipelineStage):
    """Stage 2: Run face detection using configured BaseDetector."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector

    @property
    def name(self) -> str:
        return "Detect"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.frame.image is not None and ctx.frame.image.size > 0:
            res = self.detector.detect(
                frame=ctx.frame.image,
                frame_id=ctx.frame.frame_id,
                camera_id=ctx.frame.camera_id,
            )
            ctx.detection_result = res
        return ctx


class TrackStage(PipelineStage):
    """Stage 3: Run face entity tracking via CentroidTracker."""

    def __init__(self, tracker: CentroidTracker):
        self.tracker = tracker

    @property
    def name(self) -> str:
        return "Track"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.detection_result and ctx.detection_result.detections:
            tracked = self.tracker.update(ctx.detection_result.detections)
            ctx.tracked_detections = tracked
        else:
            self.tracker.update([])
            ctx.tracked_detections = []
        return ctx


class EmotionStage(PipelineStage):
    """Stage 4: Execute official HSEmotion ONNX Neural Network inference on padded square aligned face crops."""

    def __init__(self):
        self._stage_times: deque = deque(maxlen=100)

    @property
    def name(self) -> str:
        return "Emotion"

    def _get_stage_stats(self) -> Dict[str, float]:
        """Compute rolling average and p95 latency over the last 100 frames."""
        if len(self._stage_times) == 0:
            return {}
        arr = np.array(self._stage_times)
        return {
            "emotion_stage_avg_ms": round(float(np.mean(arr)), 2),
            "emotion_stage_p95_ms": round(float(np.percentile(arr, 95)), 2),
            "emotion_stage_samples": len(arr),
        }

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.frame.image is not None and ctx.tracked_detections:
            stage_start = time.perf_counter()
            updated_tracked: List[BoundingBox] = []

            for bbox in ctx.tracked_detections:
                bbox_tuple = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)

                # Extract padded, square aspect-ratio preserving crop
                face_crop = extract_padded_square_crop(
                    ctx.frame.image,
                    bbox_tuple,
                    padding_pct=emotion_classifier.padding_pct,
                    min_face_size=emotion_classifier.min_face_size,
                )

                if face_crop is not None:
                    # Run ONNX Runtime Deep Learning FER Inference with temporal track smoothing
                    res = emotion_classifier.predict(face_crop, track_id=bbox.track_id)
                else:
                    res = {"emotion": "Unknown", "confidence": 0.0}

                updated_bbox = BoundingBox(
                    x_min=bbox.x_min,
                    y_min=bbox.y_min,
                    x_max=bbox.x_max,
                    y_max=bbox.y_max,
                    confidence=bbox.confidence,
                    track_id=bbox.track_id,
                    emotion=res.get("emotion", "Unknown"),
                    emotion_confidence=res.get("confidence", 0.0),
                    emotion_probabilities=res.get("probabilities"),
                )
                updated_tracked.append(updated_bbox)

            # Record total stage time
            stage_ms = (time.perf_counter() - stage_start) * 1000.0
            self._stage_times.append(stage_ms)
            stage_stats = self._get_stage_stats()

            logger.debug(
                "EmotionStage: %d faces in %.2fms | avg=%.2fms p95=%.2fms (n=%d)",
                len(updated_tracked),
                stage_ms,
                stage_stats.get("emotion_stage_avg_ms", 0),
                stage_stats.get("emotion_stage_p95_ms", 0),
                stage_stats.get("emotion_stage_samples", 0),
            )

            ctx.tracked_detections = updated_tracked
            if ctx.detection_result:
                ctx.detection_result = DetectionResult(
                    frame_id=ctx.detection_result.frame_id,
                    camera_id=ctx.detection_result.camera_id,
                    detections=updated_tracked,
                    inference_time_ms=ctx.detection_result.inference_time_ms,
                    detector_name=ctx.detection_result.detector_name,
                )
        return ctx


class AnnotateStage(PipelineStage):
    """Stage 5: Draw bounding boxes, entity track IDs, and ONNX emotion confidence labels onto frame."""

    @property
    def name(self) -> str:
        return "Annotate"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.frame.image is not None and ctx.tracked_detections:
            for bbox in ctx.tracked_detections:
                bbox_tuple = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
                emotion = bbox.emotion or "Unknown"
                em_conf = bbox.emotion_confidence if bbox.emotion_confidence is not None else 0.0

                label_str = (
                    f"ID #{bbox.track_id} | {emotion} ({em_conf:.1f}%)"
                    if bbox.track_id
                    else f"Face | {emotion} ({em_conf:.1f}%)"
                )
                draw_bbox(
                    frame=ctx.frame.image,
                    bbox=bbox_tuple,
                    color=(0, 255, 0),
                    thickness=3,
                    label=label_str,
                )

        if ctx.frame.image is not None:
            bgr = cv2.cvtColor(ctx.frame.image, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                ctx.processed_jpeg = enc.tobytes()

        return ctx


class DefaultVisionPipeline:
    """Standard sequential execution pipeline: Decode -> Detect -> Track -> Emotion -> Annotate."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.tracker = CentroidTracker()
        self.stages: List[PipelineStage] = [
            DecodeStage(),
            DetectStage(self.detector),
            TrackStage(self.tracker),
            EmotionStage(),
            AnnotateStage(),
        ]

    async def execute(self, frame: Frame, raw_jpeg: Optional[bytes] = None) -> PipelineContext:
        ctx = PipelineContext(frame=frame)
        ctx.raw_jpeg = raw_jpeg

        for stage in self.stages:
            ctx = await stage.process(ctx)

        return ctx
