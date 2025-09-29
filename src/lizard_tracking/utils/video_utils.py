import cv2
from pathlib import Path

def open_video_writer(path: str | Path, fps: float, width: int, height: int):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))

def get_video_meta(cap: cv2.VideoCapture):
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nf    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fps": fps, "width": W, "height": H, "frames": nf}