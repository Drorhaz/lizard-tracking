#!/usr/bin/env python3
"""Stream pose detections from a camera or video file and log trajectory data.

Outputs (per run):
- CSV with trajectory + keypoints + detected events
- Optional overlay video (when WRITE_VIDEO is True)
- Labeled frames (raw frame, overlay preview, YOLO pose label) every N frames

All artefacts are stored under ``output/detections/<run_name>_<timestamp>/``.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "lib"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lizard_tracking.config import PoseInferenceConfig
from lizard_tracking.core import HeadPose, PoseObservation
from lizard_tracking.ui.stream import ActivityDetector, LivePoseProcessor

# ---------------------------------------------------------------------------
# Configuration (edit to taste)
# ---------------------------------------------------------------------------
CONFIG = {
    "source": "videos/top_20250916T150021.mp4",   # path or camera index (string or int)
    "weights": "output/models/head_pose/best.pt",
    "output_root": "output/detections",
    "run_name": None,                 # optional explicit name; defaults to video stem
    "frame_stride": 1,                # process every Nth frame
    "save_every": 10,                 # save labeled frame every Nth processed frame
    "write_overlay_video": True,
    "display": False,
    "max_frames": None,
    "pose_imgsz": 640,
    "pose_conf": 0.25,
    "pose_device": 0,
    "fps_override": None,             # optional output fps override
    "activity": {
        "forward_axis": "y",
        "advance_threshold": 8.0,
        "retreat_threshold": -8.0,
        "stop_delta": 2.0,
        "stop_patience": 6,
    },
}


def _make_run_dir(root: Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = root / f"{name}_{timestamp}"
    (run_dir / "labeled_frames").mkdir(parents=True, exist_ok=True)
    return run_dir


def _center_from_pose(pose: HeadPose) -> tuple[float, float]:
    x1, y1, x2, y2 = pose.bbox_xyxy
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _write_pose_label(path: Path, pose: HeadPose, frame_shape: tuple[int, int, int]) -> None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = pose.bbox_xyxy
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    def _norm(pt):
        return pt[0] / w, pt[1] / h

    nose = _norm(pose.nose)
    ear_l = _norm(pose.ear_left)
    ear_r = _norm(pose.ear_right)

    with path.open("w") as f:
        f.write(
            "0 "
            f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{nose[0]:.6f} {nose[1]:.6f} 2 "
            f"{ear_l[0]:.6f} {ear_l[1]:.6f} 2 "
            f"{ear_r[0]:.6f} {ear_r[1]:.6f} 2\n"
        )


def main() -> None:
    cfg = CONFIG.copy()
    source = cfg["source"]
    try:
        src_index = int(source)
        cap = cv2.VideoCapture(src_index)
        run_base = f"camera{src_index}"
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(source)
        run_base = Path(source).stem
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {source}")

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("run_name") or run_base
    run_dir = _make_run_dir(output_root, run_name)
    frames_dir = run_dir / "labeled_frames"

    inference_cfg = PoseInferenceConfig(
        weights=cfg["weights"],
        imgsz=cfg["pose_imgsz"],
        conf=cfg["pose_conf"],
        device=cfg["pose_device"],
    )
    processor = LivePoseProcessor(
        inference_cfg,
        activity_detector=ActivityDetector(**cfg["activity"]),
    )

    fps = cfg.get("fps_override") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_writer = None
    if cfg["write_overlay_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_path = run_dir / "overlay.mp4"
        overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    csv_path = run_dir / "trajectory.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame_idx",
            "event",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "nose_x",
            "nose_y",
            "earL_x",
            "earL_y",
            "earR_x",
            "earR_y",
            "yaw_deg",
        ]
    )

    frame_stride = max(1, int(cfg["frame_stride"]))
    save_every = max(1, int(cfg["save_every"]))
    display = cfg.get("display", False)
    max_frames = cfg.get("max_frames")

    processed = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_idx % frame_stride != 0:
            continue

        output = processor.process_frame(frame)
        head = output.head
        event = output.event.value if output.event else ""
        if event:
            print(f"[EVENT] frame={frame_idx} -> {event}")

        if head is not None:
            obs = PoseObservation(frame_index=frame_idx, pose=head)
            row = list(obs.as_row())
            row.insert(1, event)
            csv_writer.writerow(row)
        else:
            csv_writer.writerow([frame_idx, event, -1] + [np.nan] * 13)

        overlay_frame = output.frame
        if overlay_writer is not None:
            overlay_writer.write(overlay_frame)

        if head is not None and (processed % save_every == 0):
            base = frames_dir / f"frame_{frame_idx:06d}"
            cv2.imwrite(str(base.with_suffix(".jpg")), frame)
            cv2.imwrite(str(base.with_suffix("_preview.jpg")), overlay_frame)
            _write_pose_label(base.with_suffix(".txt"), head, frame.shape)
            saved += 1

        if display:
            cv2.imshow("pogona-tracking", overlay_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed += 1
        if max_frames is not None and processed >= max_frames:
            break

    csv_file.close()
    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"[DONE] Processed {processed} frames ({saved} labeled frames)")
    print(f"[OUTPUT] Trajectory CSV: {csv_path.resolve()}")
    print(f"[OUTPUT] Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
