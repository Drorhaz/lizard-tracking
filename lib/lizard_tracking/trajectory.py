"""Backward-compatible wrapper around the new video tracking pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from .config import VideoTrackingConfig
from .pipelines import VideoTracker


def track_video(
    video_path: str,
    weights_path: str,
    out_csv: Optional[str] = None,
    out_parquet: Optional[str] = None,
    out_video: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.25,
    draw: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Shim that preserves the old function signature used in tests and scripts."""
    config = VideoTrackingConfig(
        source=video_path,
        weights=weights_path,
        imgsz=imgsz,
        conf=conf,
    )

    if out_csv is not None:
        csv_path = Path(out_csv)
        config.csv_path_override = csv_path
        config.output_root = csv_path.parent
    if out_parquet is not None:
        parquet_path = Path(out_parquet)
        config.parquet_path_override = parquet_path
        config.output_root = parquet_path.parent
    if out_video is not None:
        overlay_path = Path(out_video)
        config.overlay_path_override = overlay_path if draw else None
        config.overlay_video = draw
        config.output_root = overlay_path.parent
    else:
        config.overlay_video = draw and out_video is not None

    tracker = VideoTracker(config)
    return tracker.run()
