import os
from pathlib import Path
import sys

# Add lib to path
ROOT_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import cv2
import numpy as np
import pytest

from lizard_tracking.trajectory import track_video


def test_track_video_writes_files(tmp_path):
    path = tmp_path / "toy.mp4"
    width, height, fps, frames = 320, 240, 10, 5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), np.uint8))
    writer.release()

    weights = Path("runs/pose/pogona_head_pose3/weights/best.pt")
    if not weights.exists():
        pytest.skip("pose weights not found; skipping trajectory smoke test")

    csv_path, parquet_path, overlay_path = track_video(str(path), str(weights), draw=False)

    assert csv_path and os.path.exists(csv_path)
    if parquet_path:
        assert os.path.exists(parquet_path)
    if overlay_path:
        assert os.path.exists(overlay_path)
