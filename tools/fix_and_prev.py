#!/usr/bin/env python3
"""
Fix pose keypoint order for a selected list of images, then preview the result.

- Hardcode IMAGES_TO_FIX below (basenames or paths relative to images root).
- Supports both dataset layouts:
    A) dataset/images/{split}/... + dataset/labels/{split}/...
    B) dataset/{split}/images/... + dataset/{split}/labels/...
- Writes fixed labels in-place (optionally makes .bak backup).
- Saves previews with the SAME image filename into --out (no subfolders).
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple
import shutil
import cv2

# ---------------------------
# EDIT THIS LIST:
IMAGES_TO_FIX = [
    "flir_above_28_05_124909_29.jpg",
    "flir_above_28_05_124909_46.jpg",
    "flir_above_28_05_124909_42.jpg",
    "flir_above_28_05_125209_97.jpg",
    "flir_above_28_05_125209_102.jpg",
    "flir_above_28_05_125209_103.jpg",
    "flir_above_28_05_132620_20.jpg",
    "flir_above_28_05_133630_41.jpg",
    "flir_above_28_05_134957_66.jpg",
    "flir_above_20200521-180046_19506475-0000_3.jpg",
    "flir_above_20200521-180046_19506475-0000_84.jpg",
    "flir_above_20200521-193850_19506481-0000_13.jpg",
    "flir_above_20200521-193850_19506481-0000_35.jpg",
    "flir_above_20200521-193850_19506481-0000_59.jpg",
    "flir_above_20200521-195012_19506481-0000_25.jpg",
    "flir_above_20200521-195103_19506481-0000_18.jpg",
    "flir_above_20200521-195103_19506481-0000_22.jpg",
]
# ---------------------------

# Drawing colors
GREEN=(0,255,0); RED=(0,0,255); BLUE=(255,0,0); YELLOW=(0,255,255)
EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def _is_relative_to(p: Path, base: Path) -> bool:
    """Py3.7/3.8 compatible replacement for Path.is_relative_to."""
    try:
        p.relative_to(base)
        return True
    except Exception:
        return False

def resolve_layout(dataset: Path):
    """
    Return tuple of ((images_root, labels_root) for layout A and list of (images_root, labels_root) for layout B).
    """
    a_images = dataset / "images"
    a_labels = dataset / "labels"
    layout_a = (a_images if a_images.exists() else None,
                a_labels if a_labels.exists() else None)

    layout_b = []
    for split in ("train","val","test"):
        b_images = dataset / split / "images"
        b_labels = dataset / split / "labels"
        if b_images.exists():
            layout_b.append((b_images, b_labels))
    return layout_a, layout_b

def find_image_in_layouts(dataset: Path, name_or_rel: str):
    """
    Try to find the image by:
      - direct path under dataset (if provided as relative like 'images/train/x.jpg' or 'train/images/x.jpg')
      - relative to each images_root
      - by basename search (last resort; may be slow)
    Returns (img_path, images_root, labels_root) or (None, None, None) if not found.
    """
    # Normalize path
    candidate = (dataset / name_or_rel).resolve()
    if candidate.exists() and candidate.suffix.lower() in EXTS:
        layout_a, layout_b = resolve_layout(dataset)
        # A
        if layout_a[0] and _is_relative_to(candidate, layout_a[0]):
            return candidate, layout_a[0], layout_a[1]
        # B
        for img_root, lbl_root in layout_b:
            if _is_relative_to(candidate, img_root):
                return candidate, img_root, lbl_root
        # Fallback
        if layout_a[0]:
            return candidate, layout_a[0], layout_a[1]
        if layout_b:
            return candidate, layout_b[0][0], layout_b[0][1]

    # If the user passed a relative path under images root (without 'images/' prefix)
    layout_a, layout_b = resolve_layout(dataset)
    # Try under A
    if layout_a[0]:
        p = (layout_a[0] / name_or_rel).resolve()
        if p.exists() and p.suffix.lower() in EXTS:
            return p, layout_a[0], layout_a[1]
    # Try under B
    for img_root, lbl_root in layout_b:
        p = (img_root / name_or_rel).resolve()
        if p.exists() and p.suffix.lower() in EXTS:
            return p, img_root, lbl_root

    # Basename scan (last resort)
    basename = Path(name_or_rel).name
    # A
    if layout_a[0]:
        hits = [q for q in layout_a[0].rglob(basename) if q.suffix.lower() in EXTS]
        if hits:
            return hits[0], layout_a[0], layout_a[1]
    # B
    for img_root, lbl_root in layout_b:
        hits = [q for q in img_root.rglob(basename) if q.suffix.lower() in EXTS]
        if hits:
            return hits[0], img_root, lbl_root

    return None, None, None

def permute_line(line: str, perm: Tuple[int,int,int]) -> str:
    """Permute YOLO-pose (15 tokens) keypoints by perm (tuple of 3 ints, 1-based)."""
    t = line.strip().split()
    if len(t) != 15:
        return line  # leave non-pose lines untouched
    head = t[:5]
    k = list(map(float, t[5:]))  # 9 numbers
    old = [(k[0],k[1],k[2]), (k[3],k[4],k[5]), (k[6],k[7],k[8])]
    new = [old[perm[0]-1], old[perm[1]-1], old[perm[2]-1]]
    tail = []
    for x,y,v in new:
        tail += [f"{x:.6f}", f"{y:.6f}", f"{v:.1f}"]
    return " ".join(head + tail)

def fix_label_for_image(img_path: Path, images_root: Path, labels_root: Path,
                        perm: Tuple[int,int,int], backup: bool=False) -> Optional[Path]:
    lblp = labels_root / img_path.relative_to(images_root).with_suffix(".txt")
    if not lblp.exists():
        print(f"[miss] label not found for {img_path.name}")
        return None

    lines = lblp.read_text(encoding="utf-8").splitlines()
    if not lines:
        print(f"[warn] empty label: {lblp}")
        return None

    # Backup if requested
    if backup:
        bak = lblp.with_suffix(".txt.bak")
        if not bak.exists():
            shutil.copyfile(lblp, bak)

    out = []
    mod = False
    for ln in lines:
        new_ln = permute_line(ln, perm)
        out.append(new_ln)
        mod |= (new_ln != ln)

    if mod:
        lblp.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[ok] fixed {lblp}")
    else:
        print(f"[skip] already in desired order: {lblp}")
    return lblp

def draw_preview(im, W, H, line: str) -> bool:
    t = line.strip().split()
    if len(t) != 15:
        return False
    cx,cy,w,h = map(float, t[1:5])
    x1 = int((cx - w/2)*W); y1 = int((cy - h/2)*H)
    x2 = int((cx + w/2)*W); y2 = int((cy + h/2)*H)
    cv2.rectangle(im, (x1,y1), (x2,y2), GREEN, 2)

    k = list(map(float, t[5:]))
    # assume normalized kpts (standard YOLO txt)
    nose  = (int(k[0]*W), int(k[1]*H)); vn = int(k[2])
    L     = (int(k[3]*W), int(k[4]*H)); vl = int(k[5])
    R     = (int(k[6]*W), int(k[7]*H)); vr = int(k[8])

    cv2.circle(im, nose, 4, RED, -1)
    cv2.circle(im, L, 4, BLUE, -1)
    cv2.circle(im, R, 4, BLUE, -1)
    if vn>0 and vl>0 and vr>0:
        mx, my = (L[0]+R[0])//2, (L[1]+R[1])//2
        cv2.line(im, nose, (mx,my), YELLOW, 2)
    return True

def preview_fixed(img_path: Path, label_path: Path, out_dir: Path) -> bool:
    im = cv2.imread(str(img_path))
    if im is None:
        print(f"[warn] cannot read image: {img_path}")
        return False
    H, W = im.shape[:2]
    drew = False
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        drew |= draw_preview(im, W, H, ln)
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / img_path.name   # same filename
    cv2.imwrite(str(outp), im)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset root")
    ap.add_argument("--perm", default="2,3,1",
                    help="new order as indices of old kpts (1-based), e.g., 2,3,1")
    ap.add_argument("--out", default="output/previews_fixed", help="folder to save previews")
    ap.add_argument("--backup", action="store_true", help="save .bak of original labels")
    args = ap.parse_args()

    perm = tuple(int(x) for x in args.perm.split(","))
    if sorted(perm) != [1,2,3]:
        raise SystemExit("perm must be a permutation of 1,2,3 (example: 2,3,1)")

    ds = Path(args.dataset)
    out = Path(args.out)

    if not IMAGES_TO_FIX:
        raise SystemExit("Please add image names/paths to IMAGES_TO_FIX near the top of this script.")

    fixed = 0; previewed = 0; missing = 0
    for name in IMAGES_TO_FIX:
        imgp, img_root, lbl_root = find_image_in_layouts(ds, name)
        if imgp is None:
            print(f"[miss] image not found for entry: {name}")
            missing += 1
            continue

        lblp = fix_label_for_image(imgp, img_root, lbl_root, perm, backup=args.backup)
        if lblp is None:
            missing += 1
            continue
        fixed += 1

        if preview_fixed(imgp, lblp, out):
            previewed += 1

    print(f"[done] fixed={fixed} previewed={previewed} missing={missing} out={out}")

if __name__ == "__main__":
    main()