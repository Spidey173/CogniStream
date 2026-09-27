"""
CentroidTracker implementation for multi-face entity tracking across frames.
Assigns persistent tracking IDs (track_id) to faces across streaming video frames.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np

from app.domain.detection import BoundingBox
from app.domain.tracking import TrackedEntity


class CentroidTracker:
    """
    Lightweight Centroid Tracker for assigning persistent entity IDs across video frames.
    Uses Euclidean distance matrix matching between new detection centroids and existing tracked objects.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 100.0):
        self.next_object_id = 1
        self.objects: OrderedDict[int, TrackedEntity] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, bbox: BoundingBox, centroid: Tuple[int, int]) -> int:
        track_id = self.next_object_id
        self.objects[track_id] = TrackedEntity(
            track_id=track_id,
            bbox=bbox,
            centroid=centroid,
            disappeared_frames=0,
        )
        self.next_object_id += 1
        return track_id

    def _deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)

    def update(self, bboxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Update tracker with new bounding box detections for current frame.

        Returns:
            Updated list of BoundingBox objects with assigned track_id fields.
        """
        if len(bboxes) == 0:
            # Mark existing tracked objects as disappeared
            for object_id in list(self.objects.keys()):
                entity = self.objects[object_id]
                entity.disappeared_frames += 1
                if entity.disappeared_frames > self.max_disappeared:
                    self._deregister(object_id)
            return []

        # Compute centroids for incoming detections
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, bbox in enumerate(bboxes):
            input_centroids[i] = bbox.center

        # If currently tracking no objects, register all incoming detections
        if len(self.objects) == 0:
            result_bboxes = []
            for i, bbox in enumerate(bboxes):
                centroid = (int(input_centroids[i][0]), int(input_centroids[i][1]))
                track_id = self._register(bbox, centroid)
                result_bboxes.append(
                    BoundingBox(
                        x_min=bbox.x_min,
                        y_min=bbox.y_min,
                        x_max=bbox.x_max,
                        y_max=bbox.y_max,
                        confidence=bbox.confidence,
                        track_id=track_id,
                    )
                )
            return result_bboxes

        # Match existing tracked objects to new detection centroids using distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid].centroid for oid in object_ids])

        # Euclidean distance matrix (N x M)
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids, axis=2
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        result_bboxes = [None] * len(bboxes)

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))

            # Update tracked object
            entity = self.objects[object_id]
            entity.bbox = bboxes[col]
            entity.centroid = centroid
            entity.disappeared_frames = 0

            result_bboxes[col] = BoundingBox(
                x_min=bboxes[col].x_min,
                y_min=bboxes[col].y_min,
                x_max=bboxes[col].x_max,
                y_max=bboxes[col].y_max,
                confidence=bboxes[col].confidence,
                track_id=object_id,
            )

            used_rows.add(row)
            used_cols.add(col)

        # Register unassigned new detections
        unused_cols = set(range(0, len(bboxes))).difference(used_cols)
        for col in unused_cols:
            centroid = (int(input_centroids[col][0]), int(input_centroids[col][1]))
            track_id = self._register(bboxes[col], centroid)
            result_bboxes[col] = BoundingBox(
                x_min=bboxes[col].x_min,
                y_min=bboxes[col].y_min,
                x_max=bboxes[col].x_max,
                y_max=bboxes[col].y_max,
                confidence=bboxes[col].confidence,
                track_id=track_id,
            )

        # Deregister stale unassigned tracked objects
        unused_rows = set(range(0, len(object_ids))).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            entity = self.objects[object_id]
            entity.disappeared_frames += 1
            if entity.disappeared_frames > self.max_disappeared:
                self._deregister(object_id)

        return [b for b in result_bboxes if b is not None]
