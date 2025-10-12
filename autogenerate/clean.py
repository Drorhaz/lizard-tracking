#!/usr/bin/env python3
from pathlib import Path
SUB = "val"
IMG_DIR = Path(f"dataset/images/{SUB}")
LBL_DIR = Path(f"dataset/labels/{SUB}")

# extensions YOLO expects
exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

imgs = {p.stem: p for p in IMG_DIR.rglob("*") if p.suffix.lower() in exts}
lbls = {p.stem: p for p in LBL_DIR.rglob("*.txt")}

# find mismatches
only_imgs = imgs.keys() - lbls.keys()
only_lbls = lbls.keys() - imgs.keys()

print(f"[{SUB}] total imgs={len(imgs)}  labels={len(lbls)}")
print(f" -> images without label: {len(only_imgs)}")
print(f" -> labels without image: {len(only_lbls)}")

# delete the unpaired ones
for stem in only_imgs:
    print("deleting image:", imgs[stem])
    imgs[stem].unlink()

for stem in only_lbls:
    print("deleting label:", lbls[stem])
    lbls[stem].unlink()

print("[done] cleaned mismatches")




LBL_DIR = Path(f"dataset/labels/{SUB}")  # change to val/test if needed
deleted = 0
for p in LBL_DIR.rglob("*.txt"):
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:  # empty file
        p.unlink()
        deleted += 1
print(f"[done] deleted {deleted} empty label files from {LBL_DIR}")