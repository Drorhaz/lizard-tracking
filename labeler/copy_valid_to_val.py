#!/usr/bin/env python3
"""
Copy curated validation images (and their YOLO labels) into the official val split.

Reads a text file of image paths (one per line), copies each image into
DATASET_IMAGES_VAL, and attempts to locate/copy the corresponding label
into DATASET_LABELS_VAL.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

# ============================= CONFIG ============================= #
LOG_PATH = Path("labeler/output/data/valid/train_ok.txt")
DATASET_IMAGES_VAL = Path("dataset/images/val")
DATASET_LABELS_VAL = Path("dataset/labels/val")
OVERWRITE = True  # if False, existing targets are left untouched
VERBOSE = True    # emit per-file logging
# ================================================================== #

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image_paths(log_path: Path) -> list[Path]:
    lines = [ln.strip() for ln in log_path.read_text(encoding="utf-8").splitlines()]
    paths = [Path(ln).expanduser() for ln in lines if ln]
    return paths


def guess_label_path(img_path: Path) -> Path | None:
    stem_txt = img_path.with_suffix(".txt")
    if stem_txt.exists():
        return stem_txt

    parent = img_path.parent
    if parent.name == "labeled_frames":
        candidate = parent.parent / "labels" / (img_path.stem + ".txt")
        if candidate.exists():
            return candidate

    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts[:]
        label_parts[idx] = "labels"
        label_parts[-1] = img_path.stem + ".txt"
        candidate = Path(*label_parts)
        if candidate.exists():
            return candidate

    # fallback - check sibling "labels" directory if present
    candidate = parent / (img_path.stem + ".txt")
    if candidate.exists():
        return candidate

    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        if VERBOSE:
            print(f"[warn] missing source: {src}")
        return False
    if not OVERWRITE and dst.exists():
        if VERBOSE:
            print(f"[skip] already exists: {dst}")
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    if VERBOSE:
        print(f"[copy] {src} -> {dst}")
    return True


def main() -> int:
    if not LOG_PATH.exists():
        print(f"[error] log file not found: {LOG_PATH}", file=sys.stderr)
        return 1

    images = load_image_paths(LOG_PATH)
    if not images:
        print(f"[info] no image paths in {LOG_PATH}")
        return 0

    ensure_dir(DATASET_IMAGES_VAL)
    ensure_dir(DATASET_LABELS_VAL)

    copied = 0
    missing_labels = 0

    for img_path in images:
        if not img_path.exists():
            if VERBOSE:
                print(f"[warn] image not found, skipping: {img_path}")
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            if VERBOSE:
                print(f"[warn] not an image (skipped): {img_path}")
            continue

        dst_img = DATASET_IMAGES_VAL / img_path.name
        if copy_file(img_path, dst_img):
            copied += 1

        label_path = guess_label_path(img_path)
        if label_path is None:
            missing_labels += 1
            if VERBOSE:
                print(f"[warn] no label found for {img_path}")
            continue

        dst_lbl = DATASET_LABELS_VAL / label_path.name
        copy_file(label_path, dst_lbl)

    print(f"[done] copied {copied} image(s); labels missing for {missing_labels} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
