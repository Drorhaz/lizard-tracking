"""YOLO pose wrapper specialised for the pogona head keypoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from ..core import HeadPose, PoseKeypoints, compute_yaw


@dataclass
class ModelOutput:
    """Convenience container bundling raw YOLO tensor outputs."""

    boxes: np.ndarray
    confs: np.ndarray
    keypoints: np.ndarray


class PogonaHeadPoseModel:
    """Thin wrapper around a YOLO pose model fine-tuned for lizard head keypoints."""

    def __init__(
        self,
        weights: str,
        *,
        imgsz: int = 640,
        conf: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        self.model = YOLO(weights)
        if device is not None:
            self.model.to(device)
        self.imgsz = imgsz
        self.conf = conf

    def _extract(self, numpy_frame: np.ndarray) -> Optional[ModelOutput]:
        results = self.model.predict(
            source=numpy_frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )
        if not results:
            return None
        res = results[0]
        boxes = getattr(res.boxes, "xyxy", None)
        confs = getattr(res.boxes, "conf", None)
        kpts = getattr(res, "keypoints", None)
        if boxes is None or confs is None or kpts is None:
            return None

        H, W = numpy_frame.shape[:2]
        boxes_np = boxes.cpu().numpy()
        confs_np = confs.cpu().numpy()
        if hasattr(kpts, "xyn") and kpts.xyn is not None:
            keypoints_np = kpts.xyn.cpu().numpy() * np.array([W, H])[None, None, :]
        else:
            keypoints_np = kpts.xy.cpu().numpy()
        return ModelOutput(boxes=boxes_np, confs=confs_np, keypoints=keypoints_np)

    def predict(self, frame_bgr: np.ndarray) -> List[HeadPose]:
        """Return pose detections for a given OpenCV frame."""
        output = self._extract(frame_bgr)
        if output is None:
            return []

        detections: List[HeadPose] = []
        for box, conf, kp in zip(output.boxes, output.confs, output.keypoints):
            nose = (float(kp[0, 0]), float(kp[0, 1]))
            ear_left = (float(kp[1, 0]), float(kp[1, 1]))
            ear_right = (float(kp[2, 0]), float(kp[2, 1]))
            yaw = compute_yaw(nose, ear_left, ear_right)
            detections.append(
                HeadPose(
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=float(conf),
                    keypoints=PoseKeypoints(nose=nose, ear_left=ear_left, ear_right=ear_right),
                    yaw_rad=yaw,
                )
            )
        return detections
