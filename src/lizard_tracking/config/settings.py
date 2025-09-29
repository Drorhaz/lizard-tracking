"""Configuration primitives for the lizard tracking pipelines.

The goal is to keep training/inference configuration in one place so that
scripts, notebooks, and the eventual UI can share the same objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PoseTrainingConfig:
    """High-level configuration for fine-tuning a YOLO pose model."""

    data_yaml: str = "data/pogona_head_pose.yaml"
    model: str = "yolo11s-pose.pt"
    project: str = "runs/pose"
    run_name: str = "pogona_head_pose"

    epochs: int = 150
    imgsz: int = 640
    batch: int = 16
    device: str | int = "0"
    lr0: float = 0.01
    weight_decay: float = 5e-4
    patience: Optional[int] = None
    resume: bool = False

    extra_overrides: Dict[str, Any] = field(default_factory=dict)

    def as_ultralytics_kwargs(self) -> Dict[str, Any]:
        """Translate the config into keyword arguments accepted by ``YOLO.train``."""
        overrides = {
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "lr0": self.lr0,
            "weight_decay": self.weight_decay,
        }
        if self.patience is not None:
            overrides["patience"] = self.patience
        overrides.update(self.extra_overrides)
        return overrides

    @property
    def run_directory(self) -> Path:
        return Path(self.project) / self.run_name


@dataclass
class PoseInferenceConfig:
    """Configuration for running pose inference on images or frames."""

    weights: str
    imgsz: int = 640
    conf: float = 0.25
    device: str | int = "0"
    max_det: int = 1000
    agnostic_nms: bool = False


@dataclass
class VideoTrackingConfig:
    """Configuration for tracking a video stream and logging trajectories."""

    source: str
    weights: str

    output_root: Path = Path("output")
    overlay_video: bool = True
    overlay_filename: str = "trajectory_overlay.mp4"
    csv_filename: str = "trajectory.csv"
    parquet_filename: Optional[str] = "trajectory.parquet"

    csv_path_override: Optional[Path] = None
    parquet_path_override: Optional[Path] = None
    overlay_path_override: Optional[Path] = None

    imgsz: int = 640
    conf: float = 0.25
    device: str | int = "0"

    def overlay_path(self) -> Optional[Path]:
        if not self.overlay_video:
            return None
        if self.overlay_path_override is not None:
            return self.overlay_path_override
        return self.output_root / self.overlay_filename

    def csv_path(self) -> Path:
        if self.csv_path_override is not None:
            return self.csv_path_override
        return self.output_root / self.csv_filename

    def parquet_path(self) -> Optional[Path]:
        if self.parquet_filename is None:
            return None
        if self.parquet_path_override is not None:
            return self.parquet_path_override
        return self.output_root / self.parquet_filename
