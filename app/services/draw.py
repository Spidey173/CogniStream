"""
Draw bounding boxes and HUD annotations on frames using high-performance OpenCV.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Draw a sleek bounding box and optional text banner on an RGB/BGR frame array in-place.
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, int(x_min)))
    y1 = max(0, min(h - 1, int(y_min)))
    x2 = max(0, min(w - 1, int(x_max)))
    y2 = max(0, min(h - 1, int(y_max)))

    if x2 <= x1 or y2 <= y1:
        return frame

    # Draw main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw corner brackets for tech HUD styling
    corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
    if corner_len > 3:
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        bg_y1 = max(0, y1 - th - 6)
        bg_y2 = y1
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 6, bg_y2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, max(th + 2, y1 - 4)),
            font,
            font_scale,
            (0, 0, 0),  # dark text on bright green background
            font_thick,
            cv2.LINE_AA,
        )

    return frame
