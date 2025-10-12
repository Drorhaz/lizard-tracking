"""Video tracking pipeline orchestrating frame inference and trajectory logging."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from ..config import VideoTrackingConfig
from ..core import HeadPose, PoseObservation
from ..models.pogona_pose import PogonaHeadPoseModel

CSV_HEADER = (
    "frame_idx",
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
)


# @dataclass(slots=True)
class FrameResult:
    """Single processed frame with optional overlay."""

    frame_index: int
    raw_frame: np.ndarray
    overlay_frame: np.ndarray
    observation: PoseObservation


class VideoTracker:
    """Run the pose model over a video stream and persist trajectory artefacts."""

    def __init__(self, config: VideoTrackingConfig):
        self.config = config
        self._model: Optional[PogonaHeadPoseModel] = None

    @property
    def model(self) -> PogonaHeadPoseModel:
        if self._model is None:
            self._model = PogonaHeadPoseModel(
                self.config.weights,
                imgsz=self.config.imgsz,
                conf=self.config.conf,
                device=self.config.device,
            )
        return self._model

    def _select_pose(self, poses: List[HeadPose]) -> Optional[HeadPose]:
        return max(poses, key=lambda pose: pose.conf) if poses else None

    def _draw_overlay(self, frame: np.ndarray, pose: HeadPose) -> np.ndarray:
        overlay = frame.copy()
        x1, y1, x2, y2 = map(int, pose.bbox_xyxy)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        nose = (int(pose.nose[0]), int(pose.nose[1]))
        left = (int(pose.ear_left[0]), int(pose.ear_left[1]))
        right = (int(pose.ear_right[0]), int(pose.ear_right[1]))
        mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)

        cv2.circle(overlay, nose, 4, (0, 0, 255), -1)
        cv2.circle(overlay, left, 4, (255, 0, 0), -1)
        cv2.circle(overlay, right, 4, (255, 0, 0), -1)
        cv2.line(overlay, nose, mid, (0, 255, 255), 2)
        return overlay

    def _open_writer(self, fps: float, size: Tuple[int, int]):
        overlay_path = self.config.overlay_path()
        if overlay_path is None:
            return None
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(overlay_path), fourcc, fps, size)

    def _frame_results(self, cap: cv2.VideoCapture) -> Generator[FrameResult, None, None]:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            poses = self.model.predict(frame)
            best = self._select_pose(poses)
            observation = PoseObservation(frame_index=frame_idx, pose=best)
            overlay = frame
            if best is not None and self.config.overlay_video:
                overlay = self._draw_overlay(frame, best)
            yield FrameResult(
                frame_index=frame_idx,
                raw_frame=frame,
                overlay_frame=overlay,
                observation=observation,
            )
            frame_idx += 1

    def iter_frames(self) -> Generator[FrameResult, None, None]:
        cap = cv2.VideoCapture(str(self.config.source))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.config.source}")
        try:
            yield from self._frame_results(cap)
        finally:
            cap.release()

    def _write_csv(self, rows: Iterable[Tuple[float, ...]], csv_path: Path) -> str:
        import csv

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        return str(csv_path)

    def _write_parquet(self, rows: List[Tuple[float, ...]], parquet_path: Path) -> Optional[str]:
        if not rows:
            return None
        try:
            import pandas as pd

            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows, columns=CSV_HEADER)
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[warn] failed to write parquet: {exc}")
            return None
        return str(parquet_path)

    def run(self) -> Tuple[str, Optional[str], Optional[str]]:
        cap = cv2.VideoCapture(str(self.config.source))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.config.source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)

        writer = self._open_writer(fps, size)
        rows: List[Tuple[float, ...]] = []
        overlay_target = self.config.overlay_path()
        overlay_written: Optional[str] = None
        parquet_target = self.config.parquet_path()

        try:
            for result in self._frame_results(cap):
                rows.append(result.observation.as_row())
                if writer is not None:
                    writer.write(result.overlay_frame)
                    overlay_written = str(overlay_target) if overlay_target is not None else None
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        csv_written = self._write_csv(rows, self.config.csv_path())
        parquet_written = (
            self._write_parquet(rows, parquet_target)
            if parquet_target is not None
            else None
        )

        return csv_written, parquet_written, overlay_written
