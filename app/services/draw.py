"""
Draw bounding boxes on frames using NumPy (fast in-place array slicing) or PIL (when text labels are requested).
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw


BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Draw a rectangle and optional text label on an RGB uint8 frame array. Modifies in-place.
    """
    x_min, y_min, x_max, y_max = bbox
    h, w = frame.shape[:2]
    t = thickness

    # Clamp coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Top edge
    frame[y_min : min(h, y_min + t), x_min:x_max] = color
    # Bottom edge
    frame[max(0, y_max - t) : y_max, x_min:x_max] = color
    # Left edge
    frame[y_min:y_max, x_min : min(w, x_min + t)] = color
    # Right edge
    frame[y_min:y_max, max(0, x_max - t) : x_max] = color

    if label:
        try:
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            draw.text((x_min + t + 2, max(0, y_min - 15)), label, fill=color)
            frame[:] = np.array(pil_img)[:]
        except Exception:
            pass

    return frame
