"""
Official Pretrained HSEmotion (EfficientNet-B0) Neural Network Classifier.
Executes genuine deep learning inference via ONNX Runtime using official AffectNet weights.

Features & Optimizations:
- Pretrained HSEmotion EfficientNet-B0 (enet_b0_8_best_vgaf.onnx)
- AffectNet Benchmark Dataset (8 categories)
- Padded Square Aspect-Ratio Preserving Face Cropping (configurable 15–25%)
- Real Face Alignment via MediaPipe Face Mesh Landmarks (iris + 6-point eye contour)
- Two-Tier Blur Quality Gating (Laplacian Variance)
- Tiny Face Rejection (configurable min size, default 48px)
- Temporal Emotion Probability Smoothing per Track ID (EMA, alpha=0.35)
- Low-Confidence Rejection Threshold (< 35.0%) -> "Unknown" (0.0%)
- Debug mode: saves original crop, aligned crop, resized model input
- Runtime timing instrumentation (preprocessing + inference, measured via perf_counter)
"""

import os
import time
import urllib.request
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.face_align import FaceMeshAligner

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# Official HSEmotion AffectNet 8-class labels
HSEMOTION_CLASSES: List[str] = [
    "Angry",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

OFFICIAL_MODEL_URL = "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
DEFAULT_MODEL_FILENAME = "enet_b0_8_best_vgaf.onnx"

# Two-tier blur thresholds (Laplacian variance)
DEFAULT_BLUR_HARD_THRESHOLD = 15.0   # Below this → skip inference entirely
DEFAULT_BLUR_SOFT_THRESHOLD = 50.0   # Below this → still run inference (slightly blurry)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute exact Softmax probability distribution over model logit outputs."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def compute_laplacian_variance(face_crop: np.ndarray) -> float:
    """Compute Laplacian variance as a blur metric. Higher = sharper."""
    if face_crop is None or face_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_face_quality(
    face_crop: np.ndarray,
    min_face_size: int = 48,
    blur_hard_threshold: float = DEFAULT_BLUR_HARD_THRESHOLD,
) -> Tuple[bool, str]:
    """
    Quality gate for face crops. Checks minimum size and severe blur.

    Two-tier blur strategy:
    - Very blurry (variance < blur_hard_threshold) → reject
    - Slightly blurry → still passes (inference proceeds)

    Args:
        face_crop: RGB uint8 face crop.
        min_face_size: Minimum face dimension in pixels.
        blur_hard_threshold: Hard rejection threshold for Laplacian variance.

    Returns:
        (passes, reason): True if quality is acceptable, else (False, reason_string).
    """
    if face_crop is None or face_crop.size == 0 or face_crop.ndim != 3:
        return False, "invalid_crop"

    h, w = face_crop.shape[:2]

    # Tiny face rejection
    if h < min_face_size or w < min_face_size:
        return False, f"tiny_face ({w}x{h} < {min_face_size}px)"

    # Hard blur rejection (very blurry only)
    lap_var = compute_laplacian_variance(face_crop)
    if lap_var < blur_hard_threshold:
        return False, f"very_blurry (laplacian_var={lap_var:.1f} < {blur_hard_threshold})"

    return True, "ok"


def extract_padded_square_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_pct: float = 0.20,
    min_face_size: int = 48,
) -> Optional[np.ndarray]:
    """
    Extract a padded, square face crop from frame array preserving aspect ratio.
    Prevents clipping forehead or chin features.

    Args:
        frame: Full RGB frame array.
        bbox: (x_min, y_min, x_max, y_max) pixel-coordinate bounding box.
        padding_pct: Padding expansion ratio (clamped to 0.15–0.25 range).
        min_face_size: Minimum crop dimension to accept.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None

    # Clamp padding to valid range
    padding_pct = max(0.15, min(0.25, padding_pct))

    h_frame, w_frame = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)

    # Padding expansion
    pad_w = int(box_w * padding_pct)
    pad_h = int(box_h * padding_pct)

    x_min_pad = max(0, x_min - pad_w)
    y_min_pad = max(0, y_min - pad_h)
    x_max_pad = min(w_frame, x_max + pad_w)
    y_max_pad = min(h_frame, y_max + pad_h)

    crop_w = x_max_pad - x_min_pad
    crop_h = y_max_pad - y_min_pad

    # Make crop square to preserve aspect ratio
    max_side = max(crop_w, crop_h)
    center_x = (x_min_pad + x_max_pad) // 2
    center_y = (y_min_pad + y_max_pad) // 2

    square_x_min = max(0, center_x - max_side // 2)
    square_y_min = max(0, center_y - max_side // 2)
    square_x_max = min(w_frame, square_x_min + max_side)
    square_y_max = min(h_frame, square_y_min + max_side)

    crop = frame[square_y_min:square_y_max, square_x_min:square_x_max]
    if crop.size == 0 or crop.shape[0] < min_face_size or crop.shape[1] < min_face_size:
        return None

    return crop


class TemporalTrackSmoother:
    """Per-Track ID Temporal Emotion Probability Buffer (EMA & Rolling Window)."""

    def __init__(self, alpha: float = 0.35, max_history: int = 10):
        self.alpha = alpha
        self.max_history = max_history
        self.smoothed_probs: Dict[int, np.ndarray] = {}
        self.history: Dict[int, deque] = {}

    def smooth(self, track_id: int, current_probs: np.ndarray) -> np.ndarray:
        """Apply Exponential Moving Average smoothing across frames for track_id."""
        if track_id not in self.smoothed_probs:
            self.smoothed_probs[track_id] = current_probs.copy()
            self.history[track_id] = deque(maxlen=self.max_history)

        prev_probs = self.smoothed_probs[track_id]
        smoothed = self.alpha * current_probs + (1.0 - self.alpha) * prev_probs
        self.smoothed_probs[track_id] = smoothed
        self.history[track_id].append(smoothed)
        return smoothed

    def reset_track(self, track_id: int) -> None:
        self.smoothed_probs.pop(track_id, None)
        self.history.pop(track_id, None)


class PerformanceTracker:
    """Tracks rolling latency statistics for the emotion processing stage."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._preprocess_times: deque = deque(maxlen=window_size)
        self._inference_times: deque = deque(maxlen=window_size)
        self._total_times: deque = deque(maxlen=window_size)

    def record(
        self, preprocess_ms: float, inference_ms: float, total_ms: float
    ) -> None:
        self._preprocess_times.append(preprocess_ms)
        self._inference_times.append(inference_ms)
        self._total_times.append(total_ms)

    def get_stats(self) -> Dict[str, Any]:
        """Return measured performance statistics over the rolling window."""
        if len(self._total_times) == 0:
            return {}

        total_arr = np.array(self._total_times)
        preprocess_arr = np.array(self._preprocess_times)
        inference_arr = np.array(self._inference_times)

        return {
            "samples": len(total_arr),
            "avg_total_ms": round(float(np.mean(total_arr)), 2),
            "p95_total_ms": round(float(np.percentile(total_arr, 95)), 2),
            "avg_preprocess_ms": round(float(np.mean(preprocess_arr)), 2),
            "avg_inference_ms": round(float(np.mean(inference_arr)), 2),
            "p95_preprocess_ms": round(float(np.percentile(preprocess_arr, 95)), 2),
            "p95_inference_ms": round(float(np.percentile(inference_arr, 95)), 2),
        }


class HSEmotionONNXClassifier:
    """
    Official HSEmotion (EfficientNet-B0) Deep Neural Network FER Classifier.
    Executes genuine neural network inference via ONNX Runtime using official AffectNet weights.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_confidence_threshold: float = 35.0,
        padding_pct: float = 0.20,
        min_face_size: int = 48,
        blur_hard_threshold: float = DEFAULT_BLUR_HARD_THRESHOLD,
        debug_mode: bool = False,
        debug_dir: str = "./debug_crops",
        debug_max_frames_per_track: int = 20,
        debug_sample_interval: int = 5,
    ):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "input"
        self.output_name: str = "output"
        self.target_height: int = 224
        self.target_width: int = 224
        self.is_loaded: bool = False
        self.min_confidence_threshold = min_confidence_threshold

        # Configurable preprocessing parameters
        self.padding_pct = max(0.15, min(0.25, padding_pct))
        self.min_face_size = min_face_size
        self.blur_hard_threshold = blur_hard_threshold

        # Debug mode
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.debug_max_frames_per_track = debug_max_frames_per_track
        self.debug_sample_interval = debug_sample_interval
        self._debug_frame_counters: Dict[int, int] = {}
        self._debug_saved_counters: Dict[int, int] = {}

        # Face alignment (MediaPipe Face Mesh)
        self.aligner = FaceMeshAligner(refine_landmarks=True)

        # Temporal smoothing
        self.track_smoother = TemporalTrackSmoother(alpha=0.35, max_history=10)

        # Performance tracking
        self.perf_tracker = PerformanceTracker(window_size=100)

        if model_path is None:
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, DEFAULT_MODEL_FILENAME)

        self._ensure_model_exists(model_path)
        self._init_session(model_path)

    def _ensure_model_exists(self, model_path: str) -> None:
        """Download official model weights from GitHub if missing locally."""
        if not os.path.exists(model_path):
            try:
                logger.info("Downloading official HSEmotion ONNX model from GitHub to %s...", model_path)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve(OFFICIAL_MODEL_URL, model_path)
                logger.info("Downloaded HSEmotion model successfully (%d bytes).", os.path.getsize(model_path))
            except Exception as e:
                logger.error("Failed downloading official HSEmotion ONNX model: %s", e, exc_info=True)

    def _init_session(self, model_path: str) -> None:
        """Initialize ONNX Runtime Session and dynamically inspect input/output shapes."""
        if not _ONNX_AVAILABLE or not os.path.exists(model_path):
            logger.warning("ONNX Runtime or model file unavailable (%s).", model_path)
            return

        try:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(model_path, providers=providers)

            input_tensor = self.session.get_inputs()[0]
            output_tensor = self.session.get_outputs()[0]

            self.input_name = input_tensor.name
            self.output_name = output_tensor.name

            # Dynamic inspection of shape [N, C, H, W]
            shape = input_tensor.shape
            if len(shape) == 4:
                if isinstance(shape[2], int):
                    self.target_height = shape[2]
                if isinstance(shape[3], int):
                    self.target_width = shape[3]

            self.is_loaded = True
            logger.info(
                "Initialized HSEmotion ONNX Session (input='%s' [%dx%d], output='%s', providers=%s)",
                self.input_name,
                self.target_width,
                self.target_height,
                self.output_name,
                providers,
            )
        except Exception as e:
            logger.error("Failed initializing ONNX Session for HSEmotion: %s", e, exc_info=True)

    def _save_debug_crops(
        self,
        track_id: int,
        original: Optional[np.ndarray],
        aligned: Optional[np.ndarray],
        resized: Optional[np.ndarray],
    ) -> None:
        """Save debug crops for inspection. Limits output per track to avoid disk fill."""
        if not self.debug_mode or track_id is None:
            return

        # Initialize counters for this track
        if track_id not in self._debug_frame_counters:
            self._debug_frame_counters[track_id] = 0
            self._debug_saved_counters[track_id] = 0

        self._debug_frame_counters[track_id] += 1
        frame_num = self._debug_frame_counters[track_id]

        # Check if we've already saved enough for this track
        if self._debug_saved_counters[track_id] >= self.debug_max_frames_per_track:
            return

        # Sample every Nth frame (save first frame always)
        if frame_num > 1 and (frame_num % self.debug_sample_interval) != 0:
            return

        self._debug_saved_counters[track_id] += 1
        save_idx = self._debug_saved_counters[track_id]

        track_dir = os.path.join(self.debug_dir, f"track_{track_id}")
        os.makedirs(track_dir, exist_ok=True)

        try:
            if original is not None:
                path = os.path.join(track_dir, f"original_{save_idx:04d}.png")
                cv2.imwrite(path, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

            if aligned is not None:
                path = os.path.join(track_dir, f"aligned_{save_idx:04d}.png")
                cv2.imwrite(path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

            if resized is not None:
                path = os.path.join(track_dir, f"input_{save_idx:04d}.png")
                cv2.imwrite(path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.debug("Failed saving debug crops for track %d: %s", track_id, e)

    def preprocess(
        self,
        face_crop: np.ndarray,
        track_id: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Preprocessing pipeline for HSEmotion EfficientNet-B0:
        1. Quality gate (tiny face + hard blur rejection)
        2. Face alignment via MediaPipe Face Mesh landmarks
        3. Resize to dynamic model dimensions (224x224)
        4. Scale float32 [0.0, 1.0]
        5. Normalize with ImageNet mean/std
        6. Transpose HWC (224, 224, 3) -> NCHW (1, 3, 224, 224)

        Returns None if the crop fails quality checks.
        """
        if face_crop is None or face_crop.size == 0 or face_crop.ndim != 3:
            return None

        # Quality gate: tiny face + hard blur rejection
        passes, reason = check_face_quality(
            face_crop,
            min_face_size=self.min_face_size,
            blur_hard_threshold=self.blur_hard_threshold,
        )
        if not passes:
            logger.debug(
                "Face quality rejected (track_id=%s): %s", track_id, reason
            )
            return None

        # Store original for debug
        original_crop = face_crop.copy() if self.debug_mode else None

        # Face alignment via MediaPipe Face Mesh
        aligned, landmarks_found = self.aligner.align_face(face_crop)
        if not landmarks_found:
            logger.debug(
                "No landmarks found for face alignment (track_id=%s), using unaligned crop.",
                track_id,
            )

        # Resize to model input size (224, 224)
        resized = cv2.resize(
            aligned,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )

        # Save debug crops
        if self.debug_mode:
            self._save_debug_crops(track_id, original_crop, aligned, resized)

        # Scale float32 [0.0, 1.0]
        img_float = resized.astype(np.float32) / 255.0

        # ImageNet Normalization
        normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW -> NCHW (1, 3, 224, 224)
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0).astype(np.float32)
        return tensor

    def predict(
        self,
        face_crop: np.ndarray,
        track_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute genuine neural network inference with optional temporal smoothing.
        Returns prediction result dict with timing metrics.
        """
        total_start = time.perf_counter()

        if not self.is_loaded or self.session is None:
            return {"emotion": "Unknown", "confidence": 0.0}

        # --- Preprocessing (timed) ---
        preprocess_start = time.perf_counter()
        tensor = self.preprocess(face_crop, track_id=track_id)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        if tensor is None:
            return {"emotion": "Unknown", "confidence": 0.0}

        try:
            # --- ONNX Inference (timed) ---
            inference_start = time.perf_counter()
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            logits = outputs[0][0]  # Shape (8,)

            # Softmax Activation
            probs = softmax(logits)

            # Apply per-track temporal probability smoothing
            if track_id is not None:
                probs = self.track_smoother.smooth(track_id, probs)

            # Map raw probabilities to classes
            prob_dict: Dict[str, float] = {}
            for cls_name, p in zip(HSEMOTION_CLASSES, probs):
                if cls_name == "Happiness":
                    cls_name = "Happy"
                elif cls_name == "Sadness":
                    cls_name = "Sad"

                prob_dict[cls_name] = round(float(p * 100.0), 1)

            dominant_emotion = max(prob_dict, key=prob_dict.get)  # type: ignore
            confidence = prob_dict[dominant_emotion]

            # Total time
            total_ms = (time.perf_counter() - total_start) * 1000.0

            # Record performance metrics
            self.perf_tracker.record(preprocess_ms, inference_ms, total_ms)

            # Log timing, probabilities, confidence, and track ID
            logger.debug(
                "Emotion prediction [track_id=%s]: emotion=%s confidence=%.1f%% "
                "preprocess=%.2fms inference=%.2fms total=%.2fms probs=%s",
                track_id,
                dominant_emotion,
                confidence,
                preprocess_ms,
                inference_ms,
                total_ms,
                prob_dict,
            )

            # Reject low-confidence predictions
            if confidence < self.min_confidence_threshold:
                return {
                    "emotion": "Unknown",
                    "confidence": 0.0,
                    "probabilities": prob_dict,
                    "preprocessing_ms": round(preprocess_ms, 2),
                    "inference_ms": round(inference_ms, 2),
                    "total_ms": round(total_ms, 2),
                }

            return {
                "emotion": dominant_emotion,
                "confidence": confidence,
                "probabilities": prob_dict,
                "preprocessing_ms": round(preprocess_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "total_ms": round(total_ms, 2),
            }
        except Exception as e:
            logger.error("Error executing HSEmotion ONNX inference: %s", e, exc_info=True)
            return {"emotion": "Unknown", "confidence": 0.0}


def _create_emotion_classifier() -> HSEmotionONNXClassifier:
    """Create the global emotion classifier with settings from environment/config."""
    # Import here to avoid circular imports
    try:
        from app.core.config import get_settings
        settings = get_settings()
        return HSEmotionONNXClassifier(
            padding_pct=getattr(settings, "EMOTION_FACE_PADDING", 0.20),
            min_face_size=getattr(settings, "EMOTION_MIN_FACE_SIZE", 48),
            blur_hard_threshold=getattr(settings, "EMOTION_BLUR_THRESHOLD", DEFAULT_BLUR_HARD_THRESHOLD),
            debug_mode=getattr(settings, "EMOTION_DEBUG", False),
            debug_dir=getattr(settings, "EMOTION_DEBUG_DIR", "./debug_crops"),
        )
    except Exception:
        # Fall back to defaults if config isn't available
        return HSEmotionONNXClassifier()


# Global FER singleton instance
emotion_classifier = _create_emotion_classifier()
