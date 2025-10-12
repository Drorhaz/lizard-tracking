"""Helpers to load YAML configs into dataclasses."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yaml

from .settings import PoseTrainingConfig, PoseInferenceConfig, VideoTrackingConfig


def load_pipeline_config(path: str | Path) -> Tuple[PoseTrainingConfig, PoseInferenceConfig, VideoTrackingConfig]:
    data = yaml.safe_load(Path(path).read_text())

    train_cfg = PoseTrainingConfig(**data.get("training", {}))
    infer_cfg = PoseInferenceConfig(**data.get("inference", {}))
    track_cfg = VideoTrackingConfig(**data.get("tracking", {}))
    return train_cfg, infer_cfg, track_cfg
