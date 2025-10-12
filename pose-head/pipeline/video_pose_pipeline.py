#!/usr/bin/env python3
"""
pipeline/video_pose_pipeline.py

Step-by-step video pose pipeline with clean separation:
- inference for a single frame
- overlay rendering for live video
- per-frame label saving (YOLO .txt) + optional frame image
- cache playback: draw from saved labels without running inference

Behavior:
- If no detection in a frame → prints "no detection".
- If multiple detections → renders only the one with highest confidence.
- Supports 3 modes via env (no CLI args):
  - INFER_LIVE: run inference, display live overlay, save labels/frames
  - LABELS_ONLY: run inference, save labels/frames, no preview window
  - PLAYBACK_CACHE: read labels from disk and overlay on video (no inference)

Outputs per run:
  <OUTPUT_DIR>/<video-stem>-<timestamp>/
    - detections.csv                  (frame-wise best detection or NaN)
    - labeled_frames/frameXXXXXXXX.jpg (optional, every LABEL_EVERY_N frames)
    - labels/frameXXXXXXXX.txt         (YOLO format: class cx cy w h [conf])
    - run_config.json                 (snapshot of env config)
"""
from __future__ import annotations

import os, time, csv, json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# tqdm (optional)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Web interface (optional)
_web = None

def _maybe_start_web(cfg):
    global _web
    if _web is not None:
        return _web
    if cfg.get('WEB_PREVIEW', False):
        try:
            # Connect to shared web interface (local mode only)
            from pipeline.shared_web_interface import update_web_frame
            _web = update_web_frame  # Use the update function directly
            print('[WEB] Connected to shared web interface')
            return _web
        except Exception as e:
            print(f'[WEB][WARN] Could not connect to shared web interface: {e}')
            # Note: HPC mode streams via saved labeled frames, not real-time
            _web = None
    return _web

def _web_update(frame):
    if _web is not None:
        try:
            _web(frame)  # Call the update function directly (local mode only)
        except Exception as e:
            print(f'[WEB][WARN] Error updating web frame: {e}')

# optional dotenv
try:
    from dotenv import load_dotenv
    here = Path(__file__).resolve().parents[1]
    envpath = here / "config" / ".env"
    if envpath.exists():
        load_dotenv(envpath)
    else:
        load_dotenv()
except Exception:
    pass

# ────────────────────────────── Config helpers ────────────────────────────────

def _get_env(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return v if v != "" else default

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    return (v.lower() in ("1","true","yes","y")) if v != "" else default

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v) if v != "" else default
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v) if v != "" else default
    except Exception:
        return default
    
def _get_path(name: str, default: str) -> Path:
    v = os.getenv(name, "")
    if v != "":
        # Expand environment variables like $USER
        expanded = os.path.expandvars(v)
        return Path(expanded)
    else:
        # Also expand environment variables in default
        expanded_default = os.path.expandvars(default)
        return Path(expanded_default)

# ──────────────────────────────
# progress helpers
def _mk_pbar(total_frames: int, desc: str = "Frames"):
    use_ascii = _get_bool("TQDM_ASCII", False)
    if tqdm:
        # dynamic_ncols adapts to narrow terminals; ascii helps on flaky encodings
        return tqdm(total=total_frames if total_frames > 0 else None,
                    desc=desc, unit="frame", dynamic_ncols=True,
                    ascii=use_ascii, mininterval=0.2, leave=True)
    return None

def _fallback_ping(i: int, total: int, note: str = ""):
    every = _get_int("PROGRESS_EVERY", 250)
    if every <= 0:  # disable
        return
    if i % every == 0:
        if total > 0:
            pct = 100.0 * i / total
            print(f"[progress] {i}/{total} ({pct:.1f}%) {note}")
        else:
            print(f"[progress] {i} {note}")

# ────────────────────────────── Data types ────────────────────────────────────

@dataclass
class HeadPose:
    bbox_xyxy: Tuple[float,float,float,float]
    conf: float
    nose: Optional[Tuple[float,float]] = None
    ear_left: Optional[Tuple[float,float]] = None
    ear_right: Optional[Tuple[float,float]] = None

@dataclass
class PoseObservation:
    frame_index: int
    pose: Optional[HeadPose]  # None = no detection

    def as_row(self) -> Tuple:
        if self.pose is None:
            return (self.frame_index, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"))
        x1,y1,x2,y2 = self.pose.bbox_xyxy
        cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
        return (self.frame_index, self.pose.conf, x1, y1, x2, y2, cx, cy)

CSV_HEADER = ("frame_idx","conf","x1","y1","x2","y2","cx","cy")

# ────────────────────────────── Paths / output ────────────────────────────────

def now_tag():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def stem_for_source(src: str) -> str:
    p = Path(src)
    if p.exists():
        return p.stem
    return str(src).replace(":","_").replace("/","_")

def ensure_run_dir(output_base: str, source: str) -> Path:
    base = Path(output_base)
    base.mkdir(parents=True, exist_ok=True)
    run = base / f"{stem_for_source(source)}-{now_tag()}"
    (run / "labeled_frames").mkdir(parents=True, exist_ok=True)
    (run / "labels").mkdir(parents=True, exist_ok=True)
    return run

def save_run_config(run_dir: Path, cfg: dict):
    with open(run_dir / "run_config.json", "w") as fp:
        json.dump(cfg, fp, indent=2)

# ────────────────────────────── Model wrapper ─────────────────────────────────

class YOLOPoseModel:
    """Ultralytics YOLO wrapper returning a list of HeadPose objects."""
    def __init__(self, model_path: Path, imgsz: int = 960, conf: float = 0.25, iou: float = 0.5):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics is required (pip install ultralytics)") from e
        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def predict(self, frame: np.ndarray) -> List[HeadPose]:
        res = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]
        poses: List[HeadPose] = []
        if res is None or getattr(res, "boxes", None) is None or len(res.boxes) == 0:
            return poses
        # boxes
        boxes = res.boxes
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(boxes)):
            x1,y1,x2,y2 = map(float, xyxy[i])
            c = float(confs[i])
            poses.append(HeadPose(bbox_xyxy=(x1,y1,x2,y2), conf=c))
        # optional keypoints (if your model supports)
        try:
            if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
                kpts = res.keypoints.xy.cpu().numpy()
                for i, kp in enumerate(kpts):
                    # if your model indexes: 0=nose, 1=left ear, 2=right ear (adjust to your model!)
                    try:
                        nose = tuple(map(float, kp[0]))
                        earL = tuple(map(float, kp[1]))
                        earR = tuple(map(float, kp[2]))
                        poses[i].nose = nose
                        poses[i].ear_left = earL
                        poses[i].ear_right = earR
                    except Exception:
                        pass
        except Exception:
            pass
        return poses

# ────────────────────────────── Inference / overlay / labels ──────────────────

def select_best_pose(poses: List[HeadPose]) -> Optional[HeadPose]:
    return max(poses, key=lambda p: p.conf) if poses else None

def draw_overlay(frame: np.ndarray, pose: Optional[HeadPose]) -> np.ndarray:
    overlay = frame.copy()
    if pose is None:
        return overlay
    x1,y1,x2,y2 = map(int, pose.bbox_xyxy)
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Draw nose (red circle)
    if pose.nose:
        nose = (int(pose.nose[0]), int(pose.nose[1]))
        cv2.circle(overlay, nose, 6, (0,0,255), -1)  # Red circle for nose
    
    # Draw ears (blue circles) and connection line
    if pose.ear_left and pose.ear_right and pose.nose:
        left = (int(pose.ear_left[0]), int(pose.ear_left[1]))
        right = (int(pose.ear_right[0]), int(pose.ear_right[1]))
        mid = ((left[0]+right[0])//2, (left[1]+right[1])//2)
        
        # Draw ear circles
        cv2.circle(overlay, left, 5, (255,0,0), -1)  # Blue circle for left ear
        cv2.circle(overlay, right, 5, (255,0,0), -1)  # Blue circle for right ear
        cv2.line(overlay, (int(pose.nose[0]), int(pose.nose[1])), mid, (0,255,255), 2)  # Yellow line
    
    # Only show head confidence text on bounding box
    bbox_txt = f"HEAD {pose.conf:.3f}"
    cv2.putText(overlay, bbox_txt, (x1, max(15,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(overlay, bbox_txt, (x1, max(15,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
    return overlay

def save_yolo_label_txt(path_txt: Path, cls_id: int, bbox_xyxy: Tuple[float,float,float,float], img_w: int, img_h: int, conf: Optional[float] = None):
    x1,y1,x2,y2 = bbox_xyxy
    bw = x2-x1; bh = y2-y1
    cx = x1 + bw/2.0; cy = y1 + bh/2.0
    nx = cx / img_w; ny = cy / img_h; nw = bw / img_w; nh = bh / img_h
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(path_txt, "w") as f:
        if conf is None:
            f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {conf:.6f}\n")

def read_yolo_label_txt(path_txt: Path, img_w: int, img_h: int) -> Optional[HeadPose]:
    if not path_txt.exists():
        return None
    try:
        with open(path_txt, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        # class, cx, cy, w, h [, conf]
        if len(parts) < 5:
            return None
        _, nx, ny, nw, nh, *rest = parts
        nx=float(nx); ny=float(ny); nw=float(nw); nh=float(nh)
        cx = nx*img_w; cy = ny*img_h; bw = nw*img_w; bh = nh*img_h
        x1 = cx - bw/2.0; y1 = cy - bh/2.0; x2 = cx + bw/2.0; y2 = cy + bh/2.0
        conf = float(rest[0]) if rest else 1.0
        return HeadPose((x1,y1,x2,y2), conf)
    except Exception:
        return None

def imwrite_preview(dst: Path, img: np.ndarray, max_w: int):
    h, w = img.shape[:2]
    if w > max_w and w > 0:
        scale = max_w/float(w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)

# ────────────────────────────── Main modes ────────────────────────────────────

def run_infer_like(run_dir: Path, cfg: dict):
    """Shared loop for INFER_LIVE and LABELS_ONLY."""
    video = cfg["VIDEO_PATH"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # resolve model path
    model_path = cfg.get("MODEL_PATH","")
    if model_path == "":
        md = Path(cfg.get("MODEL_DIR","output/models/head_pose"))
        cands = sorted(list(md.glob("**/best*.pt"))) + sorted(list(md.glob("**/*.pt")))
        if not cands:
            raise FileNotFoundError(f"No .pt under {md}")
        model_path = str(cands[0])
    model = YOLOPoseModel(Path(model_path),
                          imgsz=int(cfg["IMG_SIZE"]),
                          conf=float(cfg["CONF_THRESH"]),
                          iou=float(cfg["IOU_THRESH"]))

    # CSV
    csv_path = run_dir / "detections.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv); writer.writerow(CSV_HEADER)

    preview = bool(cfg["PREVIEW"])
    win_name = cfg.get("WINDOW_NAME", "Pose Overlay")
    process_every = int(cfg["PROCESS_EVERY_N"])
    save_every = int(cfg["LABEL_EVERY_N"])
    fps_limit = float(cfg["FPS_LIMIT"])
    maxw = int(cfg["PREVIEW_MAX_W"])

    # Initialize web interface
    _maybe_start_web(cfg)

    # progress bar over **all frames read**, so it always advances smoothly
    pbar = _mk_pbar(total_frames, desc="Labeling")
    t0 = time.time()
    last_tick = time.time()

    frame_idx = 0       # frames read
    analyzed = 0        # frames actually inferred
    detections_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # progress (advance for every frame read)
        if pbar:
            pbar.update(1)
        else:
            _fallback_ping(frame_idx, total_frames)

        # cadence: skip inference on non-sampled frames
        if process_every > 1 and (frame_idx % process_every != 0):
            if preview:
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            _web_update(frame)  # Send raw frame to web interface
            continue

        analyzed += 1
        poses = model.predict(frame)
        best = select_best_pose(poses)
        if best is not None:
            detections_count += 1
        else:
            print("no detection")

        # log CSV
        obs = PoseObservation(frame_idx, best)
        with open(csv_path, "a", newline="") as fcsv:
            csv.writer(fcsv).writerow(obs.as_row())

        # overlay & persistence
        overlay = draw_overlay(frame, best)
        if best is not None:
            stem = f"frame{frame_idx:08d}"
            if save_every > 0 and (analyzed % save_every == 0):
                imwrite_preview(run_dir / "labeled_frames" / f"{stem}.jpg", overlay, maxw)
            save_yolo_label_txt(run_dir / "labels" / f"{stem}.txt", 0, best.bbox_xyxy, W, H, conf=best.conf)

        # live preview
        if preview:
            cv2.imshow(win_name, overlay if best is not None else frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # Update web interface with processed frame
        _web_update(overlay if best is not None else frame)

        # update postfix details if tqdm is active
        if pbar:
            det_rate = (detections_count / analyzed * 100) if analyzed else 0.0
            if best is not None:
                pbar.set_postfix(conf=f"{best.conf:.2f}", detections=f"{det_rate:.1f}%")
            else:
                pbar.set_postfix(detections=f"{det_rate:.1f}%")

        # optional FPS cap
        if fps_limit > 0:
            now = time.time()
            elapsed = now - last_tick
            min_dt = 1.0 / fps_limit
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
            last_tick = time.time()

    cap.release()
    if preview:
        cv2.destroyAllWindows()
    if pbar:
        pbar.close()

    elapsed = max(1e-6, time.time() - t0)
    fps_eff = analyzed / elapsed if analyzed else 0.0
    det_rate = (detections_count / analyzed * 100) if analyzed else 0.0
    print(f"Processed {analyzed} analyzed frames in {elapsed:.1f}s ({fps_eff:.2f} fps).")
    print(f"Found {detections_count} detections ({det_rate:.1f}% detection rate)")
    print(f"Wrote CSV: {csv_path}")
    
    
def run_playback_from_cache(run_dir: Path, cfg: dict):
    video = cfg["VIDEO_PATH"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    preview = bool(cfg["PREVIEW"])
    win_name = cfg.get("WINDOW_NAME", "Pose Overlay")

    pbar = _mk_pbar(total_frames, desc="Playback")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if pbar:
            pbar.update(1)
        else:
            _fallback_ping(frame_idx, total_frames)

        label_path = run_dir / "labels" / f"frame{frame_idx:08d}.txt"
        pose = read_yolo_label_txt(label_path, W, H)
        if pose is None:
            print("no detection")

        overlay = draw_overlay(frame, pose)

        if preview:
            cv2.imshow(win_name, overlay if pose is not None else frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if preview:
        cv2.destroyAllWindows()
    if pbar:
        pbar.close()
# ────────────────────────────── Entrypoint ────────────────────────────────────

def main():
    # read env config
    MODE = _get_env("MODE", "INFER_LIVE").upper()
    VIDEO_PATH = _get_env("VIDEO_PATH", None)
    if not VIDEO_PATH:
        raise SystemExit("Set VIDEO_PATH in config/.env")

    cfg = dict(
        MODE=MODE,
        VIDEO_PATH=_get_env("VIDEO_PATH", ""),
        MODEL_PATH=_get_env("MODEL_PATH", ""),
        MODEL_DIR=_get_env("MODEL_DIR", "output/models/head_pose"),
        OUTPUT_DIR=_get_env("OUTPUT_DIR", "output/detections"),
        PROCESS_EVERY_N=_get_int("PROCESS_EVERY_N", 1),
        LABEL_EVERY_N=_get_int("LABEL_EVERY_N", 10),
        PREVIEW=_get_bool("PREVIEW", True) if MODE=="INFER_LIVE" else False,  # force no window in LABELS_ONLY/PLAYBACK unless you flip it on purpose
        WINDOW_NAME=_get_env("WINDOW_NAME","Pose Overlay"),
        FPS_LIMIT=_get_float("FPS_LIMIT", 0.0),
        IMG_SIZE=_get_int("IMG_SIZE", 960),
        CONF_THRESH=_get_float("CONF_THRESH", 0.25),
        IOU_THRESH=_get_float("IOU_THRESH", 0.50),
        PREVIEW_MAX_W=_get_int("PREVIEW_MAX_W", 900),
        # Web interface settings
        WEB_PREVIEW=_get_bool("WEB_PREVIEW", False),
        WEB_HOST=_get_env("WEB_HOST", "0.0.0.0"),
        WEB_PORT=_get_int("WEB_PORT", 8765),
        HPC_MODE=_get_bool("HPC_MODE", False),  # Enable HPC web interface with job submission
    )

    # create run dir (even for playback: user points to an existing run dir)
    run_dir = ensure_run_dir(cfg["OUTPUT_DIR"], cfg["VIDEO_PATH"])

    # snapshot config
    save_run_config(run_dir, cfg)

    print(f"▶ MODE: {MODE}")
    print(f"▶ Video: {VIDEO_PATH}")
    print(f"▶ Writing to: {run_dir}")

    if MODE == "INFER_LIVE":
        run_infer_like(run_dir, cfg)
    elif MODE == "LABELS_ONLY":
        cfg["PREVIEW"] = False
        run_infer_like(run_dir, cfg)
    elif MODE == "PLAYBACK_CACHE":
        # For playback, you probably want to point run_dir to an existing run with labels.
        # Easiest: set OUTPUT_DIR to that existing run's parent; this creates a new timestamped dir.
        # Then manually set run_dir = Path(existing_run_dir) here if desired.
        # For simplicity, we re-use the freshly created dir; user can copy labels in.
        run_playback_from_cache(run_dir, cfg)
    else:
        raise SystemExit(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()
