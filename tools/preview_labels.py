#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview YOLO pose labels (bbox + 3 keypoints) without using any model.
- Assumes labels use 14 tokens per line:
  class cx cy w h  k1x k1y k1v  k2x k2y k2v  k3x k3y k3v
  (kpt order: nose, ear_left, ear_right; v in {0,1,2})
- Also supports 5-token detection-only lines (draws bbox only).

Output:
- Images saved to OUT_DIR with the SAME basename as source images.

Configure paths and options in the CONFIG block below (no CLI args).
"""

from __future__ import annotations
from pathlib import Path
import random
import cv2

# =======================
# CONFIG (edit here only)
# =======================
CONFIG = {
    # Where images live (any nested folders are fine)
    "IMAGES_ROOT": "dataset/images/train",

    # Where labels live (must mirror IMAGES_ROOT structure)
    "LABELS_ROOT": "dataset/labels/train",

    # Where to save annotated previews
    "OUT_DIR": "output/preview/train",

    # Optional sampling: set to None to render ALL, or an int to cap
    "LIMIT": None,      # e.g., 200 or None

    # Shuffle before limiting? (True/False)
    "SHUFFLE": False,

    # Keep subfolder structure under OUT_DIR? If False, save flat by filename
    "KEEP_SUBDIRS": False,

    # Drawing style
    "THICK": 2,
    "RADIUS": 4,
}

# Colors (BGR)
GREEN = (0, 255, 0)
RED   = (0,   0, 255)  # nose
BLUE  = (255, 0,   0)  # ears
YEL   = (0, 255, 255)  # nose -> ear-mid vector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def iter_image_label_pairs(images_root: Path, labels_root: Path):
    for img in images_root.rglob("*"):
        if img.suffix.lower() in IMG_EXTS:
            rel = img.relative_to(images_root).with_suffix(".txt")
            yield img, labels_root / rel

def draw_bbox(im, W, H, cx, cy, w, h, color=GREEN, thick=2):
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thick)

def draw_pose_line(im, W, H, tokens, thick=2, r=4):
    """
    tokens: list[str] of length 14:
      0=cls, 1..4=cx,cy,w,h, then (x,y,v)*3 in order: nose, ear_left, ear_right
    Only draws keypoints where v>0.
    """
    # bbox
    cx, cy, w, h = map(float, tokens[1:5])
    draw_bbox(im, W, H, cx, cy, w, h, GREEN, thick)

    # keypoints
    k = list(map(float, tokens[5:]))
    # unpack: nose, left, right
    nose = (int(k[0] * W), int(k[1] * H)); vn = int(k[2])
    le   = (int(k[3] * W), int(k[4] * H)); vl = int(k[5])
    re   = (int(k[6] * W), int(k[7] * H)); vr = int(k[8])

    if vn > 0: cv2.circle(im, nose, r, RED,  -1)
    if vl > 0: cv2.circle(im, le,   r, BLUE, -1)
    if vr > 0: cv2.circle(im, re,   r, BLUE, -1)

    # nose -> midpoint(ears) vector when both ears visible
    if vl > 0 and vr > 0 and vn > 0:
        mx = (le[0] + re[0]) // 2
        my = (le[1] + re[1]) // 2
        cv2.line(im, nose, (mx, my), YEL, thick)

def draw_det_line(im, W, H, tokens, thick=2):
    """5-token detection-only line: class cx cy w h"""
    cx, cy, w, h = map(float, tokens[1:5])
    draw_bbox(im, W, H, cx, cy, w, h, GREEN, thick)

def main():
    cfg = CONFIG
    images_root = Path(cfg["IMAGES_ROOT"]).resolve()
    labels_root = Path(cfg["LABELS_ROOT"]).resolve()
    out_dir     = Path(cfg["OUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather pairs and optionally shuffle/limit
    pairs = list(iter_image_label_pairs(images_root, labels_root))
    if cfg["SHUFFLE"]:
        random.shuffle(pairs)
    if cfg["LIMIT"] is not None:
        pairs = pairs[: int(cfg["LIMIT"])]

    kept = 0
    missing = 0
    bad = 0

    for img_path, lbl_path in pairs:
        if not lbl_path.exists():
            missing += 1
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            bad += 1
            continue
        H, W = im.shape[:2]

        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        drew_any = False
        for ln in lines:
            t = ln.split()
            if len(t) == 14:
                draw_pose_line(im, W, H, t, thick=cfg["THICK"], r=cfg["RADIUS"])
                drew_any = True
            elif len(t) == 5:
                draw_det_line(im, W, H, t, thick=cfg["THICK"])
                drew_any = True
            else:
                # Unsupported line length; skip but count
                continue

        # Save using SAME filename (optionally preserve subfolders)
        if drew_any:
            if cfg["KEEP_SUBDIRS"]:
                rel = img_path.relative_to(images_root)
                save_path = out_dir / rel
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = out_dir / img_path.name

            cv2.imwrite(str(save_path), im)
            kept += 1

    print(f"[done] wrote {kept} previews to {out_dir}")
    print(f"[stats] missing labels: {missing} | unreadable images: {bad} | total pairs scanned: {len(pairs)}")

if __name__ == "__main__":
    main()