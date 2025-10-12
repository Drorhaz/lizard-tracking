"""Domain objects representing pose detections and trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Optional, Tuple

import numpy as np

Pixel = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _midpoint(a: Pixel, b: Pixel) -> Pixel:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def compute_yaw(nose: Pixel, ear_left: Pixel, ear_right: Pixel) -> Optional[float]:
    """Return yaw angle in radians for the vector from nose to midpoint of ears."""
    ear_mid = _midpoint(ear_left, ear_right)
    dx = ear_mid[0] - nose[0]
    dy = ear_mid[1] - nose[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return atan2(dy, dx)


@dataclass
class PoseKeypoints:
    nose: Pixel
    ear_left: Pixel
    ear_right: Pixel

    def midpoint(self) -> Pixel:
        return _midpoint(self.ear_left, self.ear_right)

    def yaw(self) -> Optional[float]:
        return compute_yaw(self.nose, self.ear_left, self.ear_right)


@dataclass
class HeadPose:
    """Single detection describing the head pose for one individual lizard."""

    bbox_xyxy: BBox
    conf: float
    keypoints: PoseKeypoints
    yaw_rad: Optional[float]

    @property
    def nose(self) -> Pixel:
        return self.keypoints.nose

    @property
    def ear_left(self) -> Pixel:
        return self.keypoints.ear_left

    @property
    def ear_right(self) -> Pixel:
        return self.keypoints.ear_right

    def center(self) -> Pixel:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class PoseObservation:
    """Pose result packaged with frame metadata for downstream logging."""

    frame_index: int
    pose: Optional[HeadPose]

    def as_row(self) -> Tuple[float, ...]:
        if self.pose is None:
            return (
                float(self.frame_index),
                -1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        head = self.pose
        x1, y1, x2, y2 = head.bbox_xyxy
        cx, cy = head.center()
        yaw_deg = None if head.yaw_rad is None else head.yaw_rad * 180.0 / np.pi
        return (
            float(self.frame_index),
            float(head.conf),
            x1,
            y1,
            x2,
            y2,
            cx,
            cy,
            head.nose[0],
            head.nose[1],
            head.ear_left[0],
            head.ear_left[1],
            head.ear_right[0],
            head.ear_right[1],
            np.nan if yaw_deg is None else float(yaw_deg),
        )
