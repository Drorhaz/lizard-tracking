"""Pose training wrapper around the Ultralytics YOLO API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from ..config import PoseTrainingConfig


class PoseTrainer:
    """High-level wrapper around Ultralytics YOLO pose training."""

    def __init__(self, config: PoseTrainingConfig):
        self.config = config
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.config.model)
        return self._model

    def train(self, overrides: Optional[Dict[str, Any]] = None):
        args = self.config.as_ultralytics_kwargs()
        if overrides:
            args.update(overrides)

        run_dir = self.config.run_directory
        run_dir.mkdir(parents=True, exist_ok=True)

        return self.model.train(
            task="pose",
            data=self.config.data_yaml,
            project=self.config.project,
            name=self.config.run_name,
            resume=self.config.resume,
            **args,
        )

    def best_checkpoint(self) -> Path:
        # path = self.config.run_directory / "weights" / "best.pt"
        run_dir = self.config.run_directory
        for folder in run_dir.glob("*"):
            candidate = folder / "weights" / "best.pt"
            if candidate.exists():
                path = candidate
                break
        
        if not path.exists():
            raise FileNotFoundError(
                "best.pt not found; run training or pass an explicit weights path"
            )
        return path

    def validate(self, weights: Optional[str] = None):
        weights_path = Path(weights) if weights else self.best_checkpoint()
        model = YOLO(str(weights_path))
        return model.val(task="pose", data=self.config.data_yaml)

    def export(self, weights: Optional[str] = None, fmt: str = "onnx", **kwargs):
        weights_path = Path(weights) if weights else self.best_checkpoint()
        model = YOLO(str(weights_path))
        return model.export(format=fmt, **kwargs)
