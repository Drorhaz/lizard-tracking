#!/usr/bin/env python3
"""
pose_head_pipeline.py

Robust real-time (or file) inference loop for lizard head detection/pose using a YOLO model.
- Works WITH or WITHOUT calibration: if missing, it warns once and continues uncalibrated.
- Loads a YOLO model (trained for head/pose) from MODEL_PATH (or MODEL_DIR glob).
- Opens a camera stream or a video file.
- Processes frames at a configurable cadence.
- If calibration is available, computes trajectory distance/angle relative to a "screen line"
  and emits behavior events (advance+stop, retreat).
- Saves per-frame detections to CSV and saves labeled frames every N frames
  alongside YOLO-format label files for re-training.
- All "events" go through EventBus hooks (can be swapped with MQTT/UI).

Environment variables (defaults in parentheses):
  SOURCE (0) — camera index (e.g., "0") or path to video file
  MODEL_PATH ("") — direct .pt path
  MODEL_DIR ("output/models/head_pose") — directory to search for best*.pt if MODEL_PATH empty
  OUTPUT_DIR ("output/detections") — run outputs (csv/frames/labels)
  CALIB_PATH ("") — optional JSON/PKL calibration; if absent/invalid → uncalibrated
  PROCESS_EVERY_N (1) — analyze every Nth frame
  LABEL_EVERY_N (10) — save labeled frame every Nth analyzed frame
  PREVIEW (true) — show live window
  WINDOW_NAME ("PoseHead Live")
  FPS_LIMIT (0) — 0=unlimited; otherwise cap fps
  CONF_THRESH (0.25), IOU_THRESH (0.50), IMG_SIZE (960)
  EVENT_WINDOW_CM (8.0), STOP_SPEED_CM_S (1.0), RETREAT_CM_DELTA (6.0), VELOCITY_WIN (6)
  PREVIEW_MAX_W (900)
"""
from __future__ import annotations

import os, sys, time, math, json, csv, signal
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

# Optional: use dotenv if available (safe import)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ────────────────────────────── CONFIG (env-driven defaults) ───────────────────

@dataclass
class Config:
    # Source: path to video file OR integer camera index (e.g., "0")
    SOURCE: str = os.getenv("SOURCE", "videos/top_20250916T150021.mp4")

    # Where the trained model lives: either a direct .pt path or a directory to glob for best*.pt
    MODEL_PATH: str = os.getenv("MODEL_PATH", "")
    MODEL_DIR:  str = os.getenv("MODEL_DIR", "output/models/head_pose")

    # Output base
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output/detections")

    # Calibration json/pkl file path (optional)
    CALIB_PATH: str = os.getenv("CALIB_PATH", "")

    # Processing cadence
    PROCESS_EVERY_N: int = int(os.getenv("PROCESS_EVERY_N", "1"))   # analyze every Nth frame
    LABEL_EVERY_N:   int = int(os.getenv("LABEL_EVERY_N", "10"))    # save labeled frame every Nth analyzed frame

    # Runtime
    PREVIEW: bool = os.getenv("PREVIEW", "true").lower() in ("1","true","yes","y")
    WINDOW_NAME: str = os.getenv("WINDOW_NAME", "PoseHead Live")
    FPS_LIMIT: float = float(os.getenv("FPS_LIMIT", "0"))  # 0 = unlimited; otherwise sleep to cap

    # Detection
    CONF_THRESH: float = float(os.getenv("CONF_THRESH", "0.25"))
    IOU_THRESH:  float = float(os.getenv("IOU_THRESH", "0.50"))
    IMG_SIZE:    int   = int(os.getenv("IMG_SIZE", "960"))

    # Event logic thresholds
    WIN_CM:       float = float(os.getenv("EVENT_WINDOW_CM", "8.0"))   # min cm change to consider approach/retreat
    STOP_CM_S:    float = float(os.getenv("STOP_SPEED_CM_S", "1.0"))   # below this -> stop
    RETREAT_CM:   float = float(os.getenv("RETREAT_CM_DELTA", "6.0"))  # increase distance > threshold -> retreat
    VELOCITY_WIN: int   = int(os.getenv("VELOCITY_WIN", "6"))          # frames for velocity estimation

    # Labeled frame scaling for "preview"
    PREVIEW_MAX_W: int = int(os.getenv("PREVIEW_MAX_W", "900"))

cfg = Config()

# ────────────────────────────── Utilities ──────────────────────────────────────

def now_tag() -> str:
    # "date without separating chars": e.g., 20250929T191012
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def stem_for_source(src: str) -> str:
    try:
        p = Path(src)
        if p.exists():
            s = p.stem
        else:
            s = str(src).replace(":", "_").replace("/", "_")
    except Exception:
        s = str(src).replace(":", "_").replace("/", "_")
    return s

def ensure_model_path() -> Path:
    mp = Path(cfg.MODEL_PATH) if cfg.MODEL_PATH else None
    if mp and mp.is_file():
        return mp
    # Glob directory for a plausible best*.pt
    md = Path(cfg.MODEL_DIR)
    if not md.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {md}")
    cands = sorted(list(md.glob("**/best*.pt"))) + sorted(list(md.glob("**/*.pt")))
    if not cands:
        raise FileNotFoundError(f"No .pt files under {md}")
    return cands[0]

def ensure_output_dir(source: str) -> Path:
    base = Path(cfg.OUTPUT_DIR)
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{stem_for_source(source)}-{now_tag()}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "labeled_frames").mkdir(exist_ok=True)
    (target / "labels").mkdir(exist_ok=True)  # YOLO .txt files
    return target

# ────────────────────────────── Calibration ────────────────────────────────────
# Expected JSON format example:
# {
#   "screen_line": [[x0, y0], [x1, y1]],         # two points along the screen plane in image pixels
#   "pixels_per_cm": 4.2,                        # OR "cm_per_pixel": 0.238
#   "roi": [xmin, ymin, xmax, ymax]              # optional crop for inference/metrics
# }

@dataclass
class Calibration:
    p0: Tuple[float,float]
    p1: Tuple[float,float]
    px_per_cm: float
    roi: Optional[Tuple[int,int,int,int]] = None  # (xmin,ymin,xmax,ymax)

    @staticmethod
    def _from_mapping(data: Dict) -> Optional["Calibration"]:
        """Try to construct a Calibration from a generic mapping (JSON or PKL)."""
        if not isinstance(data, dict):
            return None

        # accept either pixels_per_cm or cm_per_pixel
        px_per_cm = data.get("pixels_per_cm", None)
        if px_per_cm is None:
            cm_per_px = data.get("cm_per_pixel", None)
            if cm_per_px and float(cm_per_px) != 0:
                px_per_cm = 1.0 / float(cm_per_px)

        # accept screen_line in a few common shapes
        sl = data.get("screen_line", None)
        if sl and isinstance(sl, (list, tuple)) and len(sl) == 2:
            p0, p1 = sl[0], sl[1]
            try:
                p0 = (float(p0[0]), float(p0[1]))
                p1 = (float(p1[0]), float(p1[1]))
            except Exception:
                p0 = p1 = None
        else:
            p0 = p1 = None

        if p0 and p1 and px_per_cm:
            roi = tuple(data["roi"]) if "roi" in data else None
            return Calibration(p0, p1, float(px_per_cm), roi)
        return None

    @staticmethod
    def load_optional(path: str) -> Optional["Calibration"]:
        """Load JSON or PKL calibration if available; return None otherwise."""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None

        try:
            if p.suffix.lower() == ".json":
                with p.open("r") as f:
                    data = json.load(f)
                return Calibration._from_mapping(data)
            elif p.suffix.lower() in (".pkl", ".pickle"):
                import pickle
                with p.open("rb") as f:
                    data = pickle.load(f)
                # Some PKLs store objects; try __dict__
                if not isinstance(data, dict) and hasattr(data, "__dict__"):
                    data = data.__dict__
                return Calibration._from_mapping(data)
            else:
                # try JSON by default
                with p.open("r") as f:
                    data = json.load(f)
                return Calibration._from_mapping(data)
        except Exception as e:
            print(f"[WARN] Failed to parse calibration at {path}: {e}")
            return None

    # geometry utilities
    def distance_cm_to_line(self, pt: Tuple[float,float]) -> float:
        (x0,y0), (x1,y1) = self.p0, self.p1
        x, y = pt
        A = y0 - y1
        B = x1 - x0
        C = x0*y1 - x1*y0
        dist_px = abs(A*x + B*y + C) / (math.hypot(A, B) + 1e-9)
        return dist_px / self.px_per_cm

    def angle_deg_relative_to_screen_normal(self, pt: Tuple[float,float]) -> float:
        (x0,y0), (x1,y1) = self.p0, self.p1
        x, y = pt
        vx, vy = (x1-x0), (y1-y0)
        nx, ny = -vy, vx
        nlen = math.hypot(nx, ny) + 1e-9
        nx, ny = nx/nlen, ny/nlen
        wx, wy = x-x0, y-y0
        dot = wx*nx + wy*ny
        wlen = math.hypot(wx, wy) + 1e-9
        cosang = max(-1.0, min(1.0, dot / wlen))
        return math.degrees(math.acos(cosang))

# ────────────────────────────── Event logic ────────────────────────────────────

class EventBus:
    """A super simple hook system you can replace with UI callbacks / MQTT later."""
    def on_event(self, name: str, info: Dict):
        ts = info.get("ts", time.time())
        human = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        print(f"[{human}] EVENT: {name} — {info}")

class BehaviorDetector:
    """
    Detects:
      - 'advance_and_stop': distance decreases by >= WIN_CM over a recent window, then velocity ~0
      - 'retreat': distance increases by >= RETREAT_CM over a recent window
    """
    def __init__(self, calib: Optional[Calibration], bus: EventBus):
        self.calib = calib
        self.bus = bus
        self.dist_hist = deque(maxlen=max(15, cfg.VELOCITY_WIN*3))
        self.time_hist = deque(maxlen=max(15, cfg.VELOCITY_WIN*3))

    def update(self, ts: float, head_xy: Tuple[float,float] | None):
        # no calibration => no behavior events
        if self.calib is None or head_xy is None:
            return
        d_cm = self.calib.distance_cm_to_line(head_xy)
        self.dist_hist.append(d_cm)
        self.time_hist.append(ts)

        if len(self.dist_hist) >= cfg.VELOCITY_WIN+1:
            dt = self.time_hist[-1] - self.time_hist[-(cfg.VELOCITY_WIN+1)]
            if dt <= 0:
                return
            dd = self.dist_hist[-1] - self.dist_hist[-(cfg.VELOCITY_WIN+1)]
            vel = dd / dt  # + = moving away; - = approaching

            # Advance+Stop
            total_win = min(len(self.dist_hist), 12)
            d_total = self.dist_hist[-(total_win)] - self.dist_hist[-1]
            if d_total >= cfg.WIN_CM and abs(vel) <= cfg.STOP_CM_S:
                self.bus.on_event("advance_and_stop", {"ts": ts, "d_approach_cm": d_total, "vel_cm_s": vel})

            # Retreat
            mid = min(len(self.dist_hist)-1, 8)
            if mid >= 2:
                d_mid = self.dist_hist[-1] - self.dist_hist[-mid]
                if d_mid >= cfg.RETREAT_CM and vel > 0:
                    self.bus.on_event("retreat", {"ts": ts, "d_retreat_cm": d_mid, "vel_cm_s": vel})

# ────────────────────────────── Drawing / Saving ───────────────────────────────

def draw_overlay(frame: np.ndarray,
                 head_xy: Optional[Tuple[int,int]],
                 bbox: Optional[Tuple[int,int,int,int]],
                 conf: Optional[float],
                 calib: Optional[Calibration],
                 traj: List[Tuple[int,int]]) -> np.ndarray:
    out = frame.copy()
    # draw screen line if available
    if calib is not None:
        cv2.line(out, (int(calib.p0[0]), int(calib.p0[1])),
                      (int(calib.p1[0]), int(calib.p1[1])),
                      (0, 255, 255), 2)

    # draw trajectory
    for i in range(1, len(traj)):
        cv2.line(out, (int(traj[i-1][0]), int(traj[i-1][1])),
                      (int(traj[i][0]),   int(traj[i][1])),
                      (255, 255, 0), 2)

    # head point + bbox
    if head_xy is not None:
        cv2.circle(out, (int(head_xy[0]), int(head_xy[1])), 5, (0, 255, 0), -1)
        if calib is not None:
            d_cm = calib.distance_cm_to_line(head_xy)
            ang  = calib.angle_deg_relative_to_screen_normal(head_xy)
            msg  = f"d={d_cm:.1f}cm ang={ang:.1f}deg"
        else:
            msg  = "uncalibrated"
        cv2.putText(out, msg, (int(head_xy[0])+8, int(head_xy[1])-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, msg, (int(head_xy[0])+8, int(head_xy[1])-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    if bbox is not None:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(out, (x1,y1), (x2,y2), (0, 200, 255), 2)
        if conf is not None:
            cv2.putText(out, f"{conf:.2f}", (x1, max(12,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(out, f"{conf:.2f}", (x1, max(12,y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def save_yolo_label_txt(path_txt: Path, cls_id: int, bbox_xyxy: Tuple[int,int,int,int], img_w: int, img_h: int):
    x1,y1,x2,y2 = bbox_xyxy
    bw = x2-x1
    bh = y2-y1
    cx = x1 + bw/2
    cy = y1 + bh/2
    # normalize
    nx = cx / img_w
    ny = cy / img_h
    nw = bw / img_w
    nh = bh / img_h
    with open(path_txt, "w") as f:
        f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

def imwrite_preview(dst: Path, img: np.ndarray, max_w: int):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        out = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        out = img
    cv2.imwrite(str(dst), out)

# ────────────────────────────── YOLO Loader ────────────────────────────────────

def load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics is required. pip install ultralytics") from e
    model = YOLO(str(model_path))
    return model

# ────────────────────────────── Main Loop ──────────────────────────────────────

def open_source(source: str) -> cv2.VideoCapture:
    # numeric => camera index
    cap: cv2.VideoCapture
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap

def analyze():
    # Resolve model + output dir
    model_path = ensure_model_path()
    out_dir    = ensure_output_dir(cfg.SOURCE)

    # Save a copy of config for provenance
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"▶ Using model: {model_path}")
    print(f"▶ Writing to: {out_dir}")

    # Load calibration (optional)
    calib = Calibration.load_optional(cfg.CALIB_PATH)
    if calib is None:
        print("─" * 80)
        print("[WARN] Calibration not found or could not be parsed.")
        print("       Continuing UNCALIBRATED: distances/angles will be NaN; behavior events disabled.")
        print("       (Set CALIB_PATH to a JSON with: screen_line, pixels_per_cm[, roi])")
        print("─" * 80)

    # CSV for detections
    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["ts", "frame_idx", "cx", "cy", "x1", "y1", "x2", "y2", "conf",
                         "dist_cm", "angle_deg"])  # will be NaN if uncalibrated

    # Event bus + behavior
    bus = EventBus()
    bdet = BehaviorDetector(calib, bus)

    # Open source
    cap = open_source(cfg.SOURCE)
    # Attempt to derive FPS (for info/sleep only)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps:
        print(f"Source FPS (reported): {src_fps:.2f}")

    model = load_yolo(model_path)

    traj: List[Tuple[int,int]] = []
    analyzed = 0
    frame_idx = 0
    keep_running = True

    def handle_sigint(signum, frame):
        nonlocal keep_running
        keep_running = False
    signal.signal(signal.SIGINT, handle_sigint)

    last_tick = time.time()

    while keep_running:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if cfg.PROCESS_EVERY_N > 1 and (frame_idx % cfg.PROCESS_EVERY_N != 0):
            # Preview raw if requested
            if cfg.PREVIEW:
                cv2.imshow(cfg.WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        ts = time.time()
        analyzed += 1

        img_h, img_w = frame.shape[:2]

        # Run detection (single-class head assumed; take the top conf box)
        res = load_yolo.cached_model.predict(source=frame, imgsz=cfg.IMG_SIZE, conf=cfg.CONF_THRESH,
                                             iou=cfg.IOU_THRESH, verbose=False)[0] if hasattr(load_yolo, "cached_model") \
              else model.predict(source=frame, imgsz=cfg.IMG_SIZE, conf=cfg.CONF_THRESH,
                                 iou=cfg.IOU_THRESH, verbose=False)[0]

        head_xy = None
        bbox_xyxy = None
        conf = None

        if res and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            i = int(np.argmax(res.boxes.conf.cpu().numpy()))
            b = res.boxes[i]
            xyxy = b.xyxy.cpu().numpy().reshape(-1)
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            conf = float(b.conf.cpu().numpy().reshape(-1)[0])
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            head_xy = (cx, cy)
            bbox_xyxy = (x1, y1, x2, y2)

            traj.append(head_xy)

            # save CSV row (NaNs if uncalibrated)
            if calib is not None:
                dist_cm = calib.distance_cm_to_line(head_xy)
                ang_deg = calib.angle_deg_relative_to_screen_normal(head_xy)
            else:
                dist_cm = float("nan")
                ang_deg = float("nan")

            with open(csv_path, "a", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([ts, frame_idx, cx, cy, x1, y1, x2, y2, conf, dist_cm, ang_deg])

            # event update
            bdet.update(ts, head_xy)

        # create overlay
        vis = draw_overlay(frame, head_xy, bbox_xyxy, conf, calib, traj)

        # save labeled frame + YOLO txt every LABEL_EVERY_N analyzed frames
        if cfg.LABEL_EVERY_N > 0 and (analyzed % cfg.LABEL_EVERY_N == 0) and bbox_xyxy is not None:
            stem = f"frame{frame_idx:08d}"
            img_path = out_dir / "labeled_frames" / f"{stem}.jpg"
            txt_path = out_dir / "labels"         / f"{stem}.txt"
            cv2.imwrite(str(img_path), vis)
            save_yolo_label_txt(txt_path, cls_id=0, bbox_xyxy=bbox_xyxy, img_w=img_w, img_h=img_h)
            # also write a preview
            imwrite_preview(out_dir / "labeled_frames" / f"{stem}.preview.jpg", vis, cfg.PREVIEW_MAX_W)

        # live preview
        if cfg.PREVIEW:
            cv2.imshow(cfg.WINDOW_NAME, vis if head_xy is not None else frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        # fps limit sleep
        if cfg.FPS_LIMIT > 0:
            now = time.time()
            elapsed = now - last_tick
            min_dt = 1.0 / cfg.FPS_LIMIT
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
            last_tick = time.time()

    cap.release()
    if cfg.PREVIEW:
        cv2.destroyAllWindows()
    print("Done.")

# Cache model at module scope if desired (optional micro-optimization)
try:
    from ultralytics import YOLO as _YOLO_
    if "_YOLO_" in globals():
        load_yolo.cached_model = _YOLO_(str(ensure_model_path()))
except Exception:
    pass

# ────────────────────────────── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    # Keep current working directory as-is (no chdir) to avoid surprises in env-based paths.
    analyze()
