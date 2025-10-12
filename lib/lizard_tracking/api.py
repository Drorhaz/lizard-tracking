"""Public API surface for training, validation, and tracking."""
from __future__ import annotations

from typing import Optional

from ultralytics import YOLO

from .config import PoseTrainingConfig, VideoTrackingConfig
from .pipelines import PoseTrainer, VideoTracker


def train_pose(
    data_yaml: str = "data/pogona_head_pose.yaml",
    model: str = "yolo11s-pose.pt",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: int | str = 0,
    run_name: str = "pogona_head_pose",
    project: str = "runs/pose",
    lr0: float = 0.01,
    weight_decay: float = 5e-4,
    patience: Optional[int] = None,
    resume: bool = False,
    **extra,
):
    cfg = PoseTrainingConfig(
        data_yaml=data_yaml,
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        run_name=run_name,
        project=project,
        lr0=lr0,
        weight_decay=weight_decay,
        patience=patience,
        resume=resume,
        extra_overrides=extra,
    )
    trainer = PoseTrainer(cfg)
    return trainer.train()


def val_pose(weights: str, data_yaml: str = "data/pogona_head_pose.yaml", device: int | str = 0):
    # Ultralytics validation doesn't require the training config, but we keep parity with the trainer
    model = YOLO(weights)
    return model.val(task="pose", data=data_yaml, device=device)


def track_and_log_video(
    video_path: str,
    weights: str,
    out_csv: Optional[str] = "output/trajectory.csv",
    out_parquet: Optional[str] = "output/trajectory.parquet",
    out_video: Optional[str] = "output/trajectory_overlay.mp4",
    imgsz: int = 640,
    conf: float = 0.25,
    draw: bool = True,
):
    cfg = VideoTrackingConfig(
        source=video_path,
        weights=weights,
        imgsz=imgsz,
        conf=conf,
        overlay_video=draw,
    )

    if out_csv:
        from pathlib import Path

        cfg.csv_path_override = Path(out_csv)
        cfg.output_root = cfg.csv_path_override.parent
    if out_parquet:
        from pathlib import Path

        cfg.parquet_path_override = Path(out_parquet)
        cfg.output_root = cfg.parquet_path_override.parent
    if out_video and draw:
        from pathlib import Path

        cfg.overlay_path_override = Path(out_video)
        cfg.output_root = cfg.overlay_path_override.parent
    elif not draw:
        cfg.overlay_video = False

    tracker = VideoTracker(cfg)
    return tracker.run()
