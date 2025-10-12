"""Command line interface for the lizard tracking project."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import typer

from ..api import val_pose
from ..config import (
    PoseInferenceConfig,
    PoseTrainingConfig,
    VideoTrackingConfig,
    load_pipeline_config,
)
from ..pipelines import PoseTrainer, VideoTracker
from ..ui.stream import LivePoseProcessor

app = typer.Typer(help="Utilities for training and running the lizard pose models.")


def _load_config(path: Optional[Path]) -> tuple[PoseTrainingConfig, PoseInferenceConfig, VideoTrackingConfig]:
    if path is None:
        return PoseTrainingConfig(), PoseInferenceConfig(weights="yolo11s-pose.pt"), VideoTrackingConfig(
            source="", weights=""
        )
    return load_pipeline_config(path)


@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config describing the pipeline"),
    data: str = typer.Option("data/pogona_head_pose.yaml", help="Dataset YAML file"),
    model: str = typer.Option("yolo11s-pose.pt", help="Pretrained checkpoint"),
    epochs: int = typer.Option(150, help="Number of fine-tuning epochs"),
    imgsz: int = typer.Option(640, help="Training image size"),
    batch: int = typer.Option(16, help="Batch size"),
    device: str = typer.Option("0", help="Torch device"),
    lr0: float = typer.Option(0.01, help="Initial learning rate"),
    weight_decay: float = typer.Option(5e-4, help="Weight decay"),
    patience: Optional[int] = typer.Option(None, help="Early stopping patience"),
    resume: bool = typer.Option(False, help="Resume from last checkpoint"),
    project: str = typer.Option("runs/pose", help="Training project directory"),
    run_name: str = typer.Option("pogona_head_pose", help="Experiment run name"),
    extra: Optional[str] = typer.Option(None, help="JSON string of extra Ultralytics overrides"),
):
    """Train the pose model with the provided configuration."""
    base_train, _, _ = _load_config(config)
    overrides = json.loads(extra) if extra else {}
    cfg = replace(
        base_train,
        data_yaml=data,
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        weight_decay=weight_decay,
        patience=patience,
        resume=resume,
        project=project,
        run_name=run_name,
        extra_overrides=overrides,
    )
    trainer = PoseTrainer(cfg)
    typer.echo(f"[train] {cfg.model} → {cfg.run_directory}")
    trainer.train()


@app.command()
def validate(
    weights: str = typer.Argument(..., help="Path to the checkpoint (best.pt)"),
    data: str = typer.Option("data/pogona_head_pose.yaml", help="Dataset YAML"),
    device: str = typer.Option("0", help="Torch device"),
):
    """Validate a trained checkpoint against the dataset."""
    val_pose(weights=weights, data_yaml=data, device=device)


@app.command()
def track(
    source: Path = typer.Argument(..., help="Video file or camera index"),
    weights: str = typer.Option("runs/pose/pogona_head_pose/weights/best.pt", help="Pose model weights"),
    output_root: Path = typer.Option(Path("output"), help="Directory for logs"),
    overlay: bool = typer.Option(True, help="Save overlayed video"),
    overlay_name: str = typer.Option("trajectory_overlay.mp4", help="Overlay video filename"),
    csv_name: str = typer.Option("trajectory.csv", help="Trajectory CSV filename"),
    parquet_name: Optional[str] = typer.Option("trajectory.parquet", help="Trajectory parquet filename"),
    imgsz: int = typer.Option(640, help="Inference image size"),
    conf: float = typer.Option(0.25, help="Detection confidence threshold"),
    device: str = typer.Option("0", help="Torch device"),
):
    """Run offline tracking on a recorded video."""
    cfg = VideoTrackingConfig(
        source=str(source),
        weights=weights,
        output_root=output_root,
        overlay_video=overlay,
        overlay_filename=overlay_name,
        csv_filename=csv_name,
        parquet_filename=parquet_name,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    tracker = VideoTracker(cfg)
    csv_path, parquet_path, overlay_path = tracker.run()
    typer.echo(f"[track] csv={csv_path}")
    if parquet_path:
        typer.echo(f"[track] parquet={parquet_path}")
    if overlay_path:
        typer.echo(f"[track] overlay={overlay_path}")


@app.command()
def live(
    source: str = typer.Option("0", help="Camera index or video path"),
    weights: str = typer.Option("runs/pose/pogona_head_pose/weights/best.pt", help="Pose model weights"),
    imgsz: int = typer.Option(640, help="Inference image size"),
    conf: float = typer.Option(0.25, help="Detection confidence threshold"),
    device: str = typer.Option("0", help="Torch device"),
    display: bool = typer.Option(True, help="Open an OpenCV window showing the overlay"),
    max_frames: Optional[int] = typer.Option(None, help="Limit the number of frames processed"),
):
    """Stream live video, overlay detections, and emit activity events."""
    cfg = PoseInferenceConfig(
        weights=weights,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    processor = LivePoseProcessor(cfg)

    try:
        source_int = int(source)
    except ValueError:
        source_int = None
    cap = cv2.VideoCapture(source if source_int is None else source_int)
    if not cap.isOpened():
        raise typer.BadParameter(f"Could not open video source: {source}")

    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            output = processor.process_frame(frame)
            if output.event is not None:
                typer.echo(f"[event] frame={frame_idx} type={output.event.value}")
            if display:
                cv2.imshow("lizard-tracking", output.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
