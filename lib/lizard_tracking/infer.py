"""Convenience helpers for ad-hoc inference outside the full UI."""
from __future__ import annotations

from typing import Optional

import cv2

from .config import PoseInferenceConfig
from .models.pogona_pose import PogonaHeadPoseModel


def predict_image(path: str, cfg: Optional[PoseInferenceConfig] = None):
    cfg = cfg or PoseInferenceConfig(weights="runs/pose/pogona_head_pose/weights/best.pt")
    model = PogonaHeadPoseModel(
        cfg.weights,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        device=cfg.device,
    )
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return model.predict(image)
