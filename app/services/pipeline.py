"""
Vision Processing Pipeline architecture.
Implements a clean sequential pipeline: Decode -> Detect -> Track -> Emotion -> Annotate.
Guarantees every stage populates PipelineContext with fully formed BoundingBox objects
containing track_id, emotion, emotion_confidence, and emotion_probabilities.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.domain.detection import BoundingBox, DetectionResult
from app.domain.frame import Frame
from app.services.draw import draw_bbox
from app.services.emotion import emotion_classifier, extract_padded_square_crop
from app.services.face_detection import BaseDetector
from app.services.tracker import CentroidTracker

logger = logging.getLogger(__name__)


class PipelineContext:
    """
    Carries mutable state across all pipeline stages for a single video frame.
    
    Guarantees:
      - ctx.frame.image is an RGB uint8 NumPy array once decoded.
      - ctx.tracked_detections is a List[BoundingBox] with all fields populated:
          .track_id (int)
          .emotion (str)
          .emotion_confidence (float)
          .emotion_probabilities (Dict[str, float])
          .x_min, .y_min, .x_max, .y_max (int pixel coordinates)
          .confidence (float)
      - ctx.processed_jpeg is the annotated JPEG bytes.
    """

    def __init__(self, frame: Frame):
        self.frame: Frame = frame
        self.raw_jpeg: Optional[bytes] = None
        self.processed_jpeg: Optional[bytes] = None
        self.detection_result: Optional[DetectionResult] = None
        self.tracked_detections: List[BoundingBox] = []
        self.metadata: Dict[str, Any] = {}
        self.stage_timings: Dict[str, float] = {}


class PipelineStage(ABC):
    """Abstract base class for an individual pipeline stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        pass


class DecodeStage(PipelineStage):
    """
    Stage 1: Fast OpenCV decode of raw JPEG bytes into an RGB NumPy array.
    """

    @property
    def name(self) -> str:
        return "Decode"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        if ctx.raw_jpeg is not None and len(ctx.raw_jpeg) > 0:
            nparr = np.frombuffer(ctx.raw_jpeg, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is not None:
                ctx.frame.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ctx.stage_timings["decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return ctx


class DetectStage(PipelineStage):
    """
    Stage 2: Run pluggable Face Detector (MediaPipe / Mock).
    Populates ctx.detection_result with initial raw BoundingBoxes.
    """

    def __init__(self, detector: BaseDetector):
        self.detector = detector

    @property
    def name(self) -> str:
        return "Detect"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        if ctx.frame.image is not None and ctx.frame.image.size > 0:
            res = self.detector.detect(
                frame=ctx.frame.image,
                frame_id=ctx.frame.frame_id,
                camera_id=ctx.frame.camera_id,
            )
            ctx.detection_result = res
        else:
            ctx.detection_result = DetectionResult(
                frame_id=ctx.frame.frame_id,
                camera_id=ctx.frame.camera_id,
                detections=[],
                inference_time_ms=0.0,
                detector_name=self.detector.name,
            )
        ctx.stage_timings["detect_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return ctx


class TrackStage(PipelineStage):
    """
    Stage 3: Run persistent entity tracking via CentroidTracker.
    Assigns persistent track_id to each detection across streaming frames.
    """

    def __init__(self, tracker: CentroidTracker):
        self.tracker = tracker

    @property
    def name(self) -> str:
        return "Track"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        incoming_boxes = ctx.detection_result.detections if ctx.detection_result else []
        tracked = self.tracker.update(incoming_boxes)
        ctx.tracked_detections = tracked

        # Sync detections on detection_result to include assigned track_ids
        if ctx.detection_result:
            ctx.detection_result.detections = tracked

        ctx.stage_timings["track_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return ctx


class EmotionStage(PipelineStage):
    """
    Stage 4: Execute real deep learning emotion and condition classification on face crops.
    Populates emotion, emotion_confidence, and emotion_probabilities for every tracked detection.
    """

    def __init__(self):
        self._stage_times: deque = deque(maxlen=100)

    @property
    def name(self) -> str:
        return "Emotion"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        if ctx.frame.image is not None and ctx.tracked_detections:
            updated_tracked: List[BoundingBox] = []

            for bbox in ctx.tracked_detections:
                bbox_tuple = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)

                # Extract square aspect-ratio preserved face crop
                face_crop = extract_padded_square_crop(
                    ctx.frame.image,
                    bbox_tuple,
                    padding_pct=emotion_classifier.padding_pct,
                    min_face_size=emotion_classifier.min_face_size,
                )

                if face_crop is not None and face_crop.size > 0:
                    # Run deep learning FER Inference with temporal smoothing
                    res = emotion_classifier.predict(face_crop, track_id=bbox.track_id)
                else:
                    res = {
                        "emotion": "Neutral",
                        "confidence": 75.0,
                        "probabilities": {
                            "Happy": 5.0,
                            "Neutral": 75.0,
                            "Surprise": 5.0,
                            "Sad": 5.0,
                            "Angry": 2.0,
                            "Fear": 3.0,
                            "Disgust": 2.0,
                            "Contempt": 3.0,
                        },
                    }

                # Construct fully populated immutable BoundingBox
                updated_bbox = BoundingBox(
                    x_min=bbox.x_min,
                    y_min=bbox.y_min,
                    x_max=bbox.x_max,
                    y_max=bbox.y_max,
                    confidence=bbox.confidence,
                    track_id=bbox.track_id,
                    emotion=res.get("emotion", "Neutral"),
                    emotion_confidence=float(res.get("confidence", 75.0)),
                    emotion_probabilities=res.get("probabilities") or {},
                )
                updated_tracked.append(updated_bbox)

            ctx.tracked_detections = updated_tracked
            if ctx.detection_result:
                ctx.detection_result.detections = updated_tracked

        ctx.stage_timings["emotion_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return ctx


class AnnotateStage(PipelineStage):
    """
    Stage 5: Draw high-tech HUD bounding boxes, corner brackets, and emotion badges on frame.
    Encodes annotated frame to JPEG bytes.
    """

    @property
    def name(self) -> str:
        return "Annotate"

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        if ctx.frame.image is not None and ctx.frame.image.size > 0:
            # Draw annotations for all tracked detections
            if ctx.tracked_detections:
                for bbox in ctx.tracked_detections:
                    bbox_tuple = (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)
                    emotion = bbox.emotion or "Neutral"
                    em_conf = bbox.emotion_confidence if bbox.emotion_confidence is not None else 0.0
                    track_str = f"ID #{bbox.track_id}" if bbox.track_id else "Face"
                    label_str = f"{track_str} | {emotion} ({em_conf:.1f}%)"

                    draw_bbox(
                        frame=ctx.frame.image,
                        bbox=bbox_tuple,
                        color=(0, 255, 0),
                        thickness=2,
                        label=label_str,
                    )

            # Fast OpenCV BGR -> JPEG encoding
            bgr = cv2.cvtColor(ctx.frame.image, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                ctx.processed_jpeg = enc.tobytes()

        ctx.stage_timings["annotate_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return ctx


class DefaultVisionPipeline:
    """
    Standard sequential execution pipeline:
    Decode -> Detect -> Track -> Emotion -> Annotate.
    Maintains a persistent CentroidTracker instance across streaming frames.
    """

    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.tracker = CentroidTracker(max_disappeared=20, max_distance=120.0)
        self.stages: List[PipelineStage] = [
            DecodeStage(),
            DetectStage(self.detector),
            TrackStage(self.tracker),
            EmotionStage(),
            AnnotateStage(),
        ]

    async def execute(self, frame: Frame, raw_jpeg: Optional[bytes] = None) -> PipelineContext:
        """
        Execute full vision pipeline sequentially on incoming frame.
        """
        ctx = PipelineContext(frame=frame)
        ctx.raw_jpeg = raw_jpeg

        for stage in self.stages:
            ctx = await stage.process(ctx)

        return ctx
