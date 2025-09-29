"""Utilities to integrate the pose model into a UI or live stream."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import PoseInferenceConfig
from ..core import HeadPose
from ..models.pogona_pose import PogonaHeadPoseModel


class ActivityEvent(str, Enum):
    ADVANCE = "advance"
    RETREAT = "retreat"
    STOP = "stop"


@dataclass(slots=True)
class LiveFrameOutput:
    frame: np.ndarray
    head: Optional[HeadPose]
    event: Optional[ActivityEvent]


class ActivityDetector:
    """Very small heuristic detector for high-level lizard activity."""

    def __init__(
        self,
        *,
        forward_axis: str = "y",
        advance_threshold: float = 8.0,
        retreat_threshold: float = -8.0,
        stop_delta: float = 2.0,
        stop_patience: int = 6,
    ):
        if forward_axis not in {"x", "y"}:
            raise ValueError("forward_axis must be 'x' or 'y'")
        self.forward_axis = forward_axis
        self.advance_threshold = advance_threshold
        self.retreat_threshold = retreat_threshold
        self.stop_delta = stop_delta
        self.stop_patience = stop_patience
        self._previous_center: Optional[Tuple[float, float]] = None
        self._still_frames: int = 0

    def reset(self) -> None:
        self._previous_center = None
        self._still_frames = 0

    def _axis_delta(self, prev: Tuple[float, float], curr: Tuple[float, float]) -> float:
        idx = 1 if self.forward_axis == "y" else 0
        return curr[idx] - prev[idx]

    def update(self, center: Optional[Tuple[float, float]]) -> Optional[ActivityEvent]:
        if center is None:
            self.reset()
            return None

        if self._previous_center is None:
            self._previous_center = center
            self._still_frames = 0
            return None

        delta_forward = self._axis_delta(self._previous_center, center)
        distance = np.linalg.norm(np.asarray(center) - np.asarray(self._previous_center))

        event: Optional[ActivityEvent] = None
        if delta_forward >= self.advance_threshold:
            event = ActivityEvent.ADVANCE
        elif delta_forward <= self.retreat_threshold:
            event = ActivityEvent.RETREAT
        else:
            if distance <= self.stop_delta:
                self._still_frames += 1
            else:
                self._still_frames = 0

            if self._still_frames >= self.stop_patience:
                event = ActivityEvent.STOP
                self._still_frames = 0

        self._previous_center = center
        return event


class LivePoseProcessor:
    """Wrap pose inference so the UI can process frames easily."""

    def __init__(
        self,
        cfg: PoseInferenceConfig,
        *,
        activity_detector: Optional[ActivityDetector] = None,
    ):
        self.cfg = cfg
        self.model = PogonaHeadPoseModel(
            cfg.weights,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            device=cfg.device,
        )
        self.activity_detector = activity_detector or ActivityDetector()

    def _draw_overlay(self, frame: np.ndarray, pose: HeadPose) -> np.ndarray:
        im = frame.copy()
        x1, y1, x2, y2 = map(int, pose.bbox_xyxy)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

        nose = (int(pose.nose[0]), int(pose.nose[1]))
        left = (int(pose.ear_left[0]), int(pose.ear_left[1]))
        right = (int(pose.ear_right[0]), int(pose.ear_right[1]))

        cv2.circle(im, nose, 4, (0, 0, 255), -1)
        cv2.circle(im, left, 4, (255, 0, 0), -1)
        cv2.circle(im, right, 4, (255, 0, 0), -1)

        mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
        cv2.line(im, nose, mid, (0, 255, 255), 2)
        return im

    def process_frame(self, frame: np.ndarray) -> LiveFrameOutput:
        poses = self.model.predict(frame)
        head = max(poses, key=lambda p: p.conf) if poses else None
        event: Optional[ActivityEvent] = None
        overlay = frame
        if head is not None:
            event = self.activity_detector.update(head.center())
            overlay = self._draw_overlay(frame, head)
        else:
            self.activity_detector.update(None)
        return LiveFrameOutput(frame=overlay, head=head, event=event)
