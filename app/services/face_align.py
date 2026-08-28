"""
Face alignment using MediaPipe Face Mesh / Face Landmarker task.

Computes true eye centers from multiple landmark points (6-point eye contour
and iris landmarks), then applies an affine rotation to level the eyes
horizontally before emotion inference.
"""

import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# 6-point contour landmarks per eye (indices mapped to Landmarker result)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Iris center landmarks
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


class FaceMeshAligner:
    """
    Singleton-style face aligner using MediaPipe Face Landmarker.

    Extracts eye center coordinates from detected facial landmarks and applies
    an affine rotation to produce a horizontally-aligned face crop.
    """

    def __init__(self, refine_landmarks: bool = True):
        self._landmarker = None

        if not _MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available — face alignment will be disabled.")
            return

        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "face_landmarker.task")

        if not os.path.exists(model_path):
            try:
                import urllib.request
                logger.info("Downloading MediaPipe Face Landmarker task model...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    model_path
                )
            except Exception as e:
                logger.error("Failed to download Face Landmarker model: %s", e)
                return

        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            logger.info("FaceMeshAligner initialized successfully via Face Landmarker.")
        except Exception as e:
            logger.error("Failed to initialize FaceMeshAligner: %s", e, exc_info=True)
            self._landmarker = None

    @property
    def is_available(self) -> bool:
        return self._landmarker is not None

    def get_eye_centers(
        self, face_crop: np.ndarray
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Detect eye centers from a face crop image.
        """
        if self._landmarker is None:
            return None

        if face_crop is None or face_crop.size == 0 or face_crop.ndim != 3:
            return None

        h, w = face_crop.shape[:2]

        try:
            # Wrap image in MediaPipe Image container
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.debug("Face Landmarker processing failed: %s", e)
            return None

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        num_landmarks = len(landmarks)

        # Prefer iris centers if available (468+ landmarks)
        if num_landmarks > RIGHT_IRIS_CENTER:
            left_iris = landmarks[LEFT_IRIS_CENTER]
            right_iris = landmarks[RIGHT_IRIS_CENTER]
            left_eye = (left_iris.x * w, left_iris.y * h)
            right_eye = (right_iris.x * w, right_iris.y * h)
        else:
            # Fall back to eye contour centroids
            left_eye = self._compute_eye_centroid(landmarks, LEFT_EYE_LANDMARKS, w, h)
            right_eye = self._compute_eye_centroid(landmarks, RIGHT_EYE_LANDMARKS, w, h)

        if left_eye is None or right_eye is None:
            return None

        return (left_eye, right_eye)

    @staticmethod
    def _compute_eye_centroid(
        landmarks, indices: list, img_w: int, img_h: int
    ) -> Optional[Tuple[float, float]]:
        """Compute centroid of a set of landmark indices in pixel coordinates."""
        xs, ys = [], []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                xs.append(lm.x * img_w)
                ys.append(lm.y * img_h)

        if len(xs) < 3:
            return None

        return (float(np.mean(xs)), float(np.mean(ys)))

    def align_face(
        self, face_crop: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Align a face crop by rotating to level the eyes horizontally.
        """
        if face_crop is None or face_crop.shape[0] < 32 or face_crop.shape[1] < 32:
            return face_crop, False

        eye_centers = self.get_eye_centers(face_crop)
        if eye_centers is None:
            return face_crop, False

        left_eye, right_eye = eye_centers
        h, w = face_crop.shape[:2]

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]

        if abs(dx) < 1e-6:
            return face_crop, True

        angle = float(np.degrees(np.arctan2(dy, dx)))

        if abs(angle) < 1.0:
            return face_crop, True

        if abs(angle) > 45.0:
            logger.debug("Skipping alignment: extreme angle %.1f°", angle)
            return face_crop, True

        center_x = (left_eye[0] + right_eye[0]) / 2.0
        center_y = (left_eye[1] + right_eye[1]) / 2.0

        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        aligned = cv2.warpAffine(
            face_crop,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        return aligned, True

    def close(self) -> None:
        """Release landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            logger.info("FaceMeshAligner closed.")
