"""
Pluggable Face Detector Interface and implementations.
Includes MediaPipe Short-Range BlazeFace, MediaPipe Full-Range BlazeFace, MockDetector, and plugin registry.
"""

from abc import ABC, abstractmethod
import time
import logging
from typing import Dict, List, Optional, Type
import numpy as np

from app.domain.detection import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class interface for all face detector implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the detector implementation."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Detector version string."""
        pass

    @property
    @abstractmethod
    def backend(self) -> str:
        """Hardware/software backend (e.g. CPU, CUDA, MPS)."""
        pass

    @property
    def supports_gpu(self) -> bool:
        return False

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0, camera_id: str = "default") -> DetectionResult:
        """Detect faces in an RGB uint8 image array."""
        pass

    def close(self) -> None:
        """Release backend resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _ensure_model_file(filename: str, urls: List[str]) -> str:
    """Ensure a MediaPipe model asset exists locally, downloading if necessary."""
    import os
    import urllib.request

    base_dir = os.path.dirname(__file__)
    target_path = os.path.join(base_dir, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return target_path

    for url in urls:
        try:
            logger.info("Downloading MediaPipe model [%s] from %s...", filename, url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
                logger.info("Successfully downloaded model to %s (%d bytes)", target_path, os.path.getsize(target_path))
                return target_path
        except Exception as e:
            logger.warning("Failed to download model from %s: %s", url, e)

    return target_path


class MediaPipeDetector(BaseDetector):
    """MediaPipe BlazeFace Short-Range (<2m) face detector implementation using Tasks API."""

    def __init__(self, min_confidence: float = 0.35):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        self.min_confidence = min_confidence
        model_urls = [
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        ]
        model_path = _ensure_model_file("blaze_face_short_range.tflite", model_urls)

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_confidence,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = FaceDetector.create_from_options(options)

    @property
    def name(self) -> str:
        return "MediaPipe BlazeFace (Short-Range)"

    @property
    def version(self) -> str:
        return "1.0.x"

    @property
    def backend(self) -> str:
        return "CPU"

    def detect(self, frame: np.ndarray, frame_id: int = 0, camera_id: str = "default") -> DetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")

        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"Invalid frame format: shape={frame.shape}, dtype={frame.dtype}")

        start_time = time.perf_counter()
        h, w, _ = frame.shape

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self._detector.detect(mp_image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[BoundingBox] = []
        if results.detections:
            sorted_dets = sorted(results.detections, key=lambda d: d.categories[0].score, reverse=True)
            for d in sorted_dets:
                bbox = d.bounding_box
                x_min = max(0, int(bbox.origin_x))
                y_min = max(0, int(bbox.origin_y))
                x_max = min(w, int(bbox.origin_x + bbox.width))
                y_max = min(h, int(bbox.origin_y + bbox.height))

                detections.append(
                    BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        confidence=float(d.categories[0].score),
                    )
                )

        return DetectionResult(
            frame_id=frame_id,
            camera_id=camera_id,
            detections=detections,
            inference_time_ms=elapsed_ms,
            detector_name=self.name,
        )

    def close(self) -> None:
        if hasattr(self, "_detector") and self._detector:
            self._detector.close()


class MediaPipeFullRangeDetector(BaseDetector):
    """MediaPipe BlazeFace Full-Range (<5m) face detector task implementation."""

    def __init__(self, min_confidence: float = 0.35):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

        self.min_confidence = min_confidence
        model_urls = [
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
        ]
        model_path = _ensure_model_file("blaze_face_full_range.tflite", model_urls)

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_confidence,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._detector = FaceDetector.create_from_options(options)

    @property
    def name(self) -> str:
        return "MediaPipe BlazeFace (Full-Range)"

    @property
    def version(self) -> str:
        return "1.0.x"

    @property
    def backend(self) -> str:
        return "CPU"

    def detect(self, frame: np.ndarray, frame_id: int = 0, camera_id: str = "default") -> DetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")

        start_time = time.perf_counter()
        h, w, _ = frame.shape

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = self._detector.detect(mp_image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[BoundingBox] = []
        if results.detections:
            sorted_dets = sorted(results.detections, key=lambda d: d.categories[0].score, reverse=True)
            for d in sorted_dets:
                bbox = d.bounding_box
                x_min = max(0, int(bbox.origin_x))
                y_min = max(0, int(bbox.origin_y))
                x_max = min(w, int(bbox.origin_x + bbox.width))
                y_max = min(h, int(bbox.origin_y + bbox.height))

                detections.append(
                    BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        confidence=float(d.categories[0].score),
                    )
                )

        return DetectionResult(
            frame_id=frame_id,
            camera_id=camera_id,
            detections=detections,
            inference_time_ms=elapsed_ms,
            detector_name=self.name,
        )

    def close(self) -> None:
        if hasattr(self, "_detector") and self._detector:
            self._detector.close()


class MockDetector(BaseDetector):
    """Mock detector for ultra-fast isolated testing and benchmarking."""

    def __init__(self, simulate_face: bool = True, latency_ms: float = 1.0):
        self.simulate_face = simulate_face
        self.latency_ms = latency_ms

    @property
    def name(self) -> str:
        return "Mock Detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def backend(self) -> str:
        return "Synthetic"

    def detect(self, frame: np.ndarray, frame_id: int = 0, camera_id: str = "default") -> DetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")

        h, w = frame.shape[:2]
        time.sleep(self.latency_ms / 1000.0)

        detections: List[BoundingBox] = []
        if self.simulate_face:
            detections.append(
                BoundingBox(
                    x_min=int(w * 0.25),
                    y_min=int(h * 0.25),
                    x_max=int(w * 0.75),
                    y_max=int(h * 0.75),
                    confidence=0.95,
                )
            )

        return DetectionResult(
            frame_id=frame_id,
            camera_id=camera_id,
            detections=detections,
            inference_time_ms=self.latency_ms,
            detector_name=self.name,
        )


# --- Plugin Registry ---
_DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register_detector(name: str, detector_cls: Type[BaseDetector]) -> None:
    """Register a detector implementation class."""
    _DETECTOR_REGISTRY[name.lower()] = detector_cls
    logger.info("Registered face detector backend: '%s' -> %s", name, detector_cls.__name__)


def get_detector(name: str, **kwargs) -> BaseDetector:
    """Instantiate detector implementation from registry with resilient fallback."""
    key = name.lower()
    if key not in _DETECTOR_REGISTRY:
        logger.warning("Detector '%s' not found in registry. Defaulting to 'mediapipe'.", name)
        key = "mediapipe"
    try:
        return _DETECTOR_REGISTRY[key](**kwargs)
    except Exception as e:
        logger.error("Failed to initialize detector '%s': %s. Falling back to MockDetector.", key, e)
        return MockDetector(simulate_face=False)


def list_detectors() -> List[Dict[str, str]]:
    """Return available registered detector names and metadata."""
    result = []
    for key, cls in _DETECTOR_REGISTRY.items():
        result.append({"key": key, "class": cls.__name__})
    return result


# Register built-in detectors accurately
register_detector("mediapipe", MediaPipeDetector)
register_detector("mediapipe-full", MediaPipeFullRangeDetector)
register_detector("mock", MockDetector)
