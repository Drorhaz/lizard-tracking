#!/usr/bin/env python3
"""
Use a trained YOLO11 pose model to generate pose labels for images.
- Supports layouts:
    A) dataset/images/{train,val,test}/...
    B) dataset/{train,val,test}/images/...
- If the model finds nothing, existing labels are preserved (no erase).
- Optionally write to a separate labels dir via --out_labels (non-destructive).

Examples:
  python autogenerate/fill_pose_from_model.py \
      --weights autogenerate/best.pt \
      --dataset dataset --conf 0.25

  python autogenerate/fill_pose_from_model.py \
      --weights autogenerate/best.pt \
      --dataset dataset --conf 0.25 \
      --out_labels dataset/labels_auto
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def to_pose_line(cls, cx, cy, w, h, kpts):
    parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for x, y, v in kpts:
        parts += [f"{x:.6f}", f"{y:.6f}", f"{v:.1f}"]
    return " ".join(parts)

def find_images(dataset_root: Path):
    # Layout A
    images = dataset_root / "images"
    if images.exists():
        for p in images.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                yield p, p.relative_to(images), images
    # Layout B
    for split in ("train", "val", "test"):
        images = dataset_root / split / "images"
        if images.exists():
            for p in images.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    yield p, p.relative_to(images), images

def default_labels_root(images_root: Path):
    # images_root: .../dataset/images  -> .../dataset/labels
    # or          .../dataset/train/images -> .../dataset/train/labels
    return images_root.parent / "labels"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="pose model .pt")
    ap.add_argument("--dataset", required=True, help="dataset root")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--out_labels", default=None, help="optional labels root (non-destructive)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    ds_root = Path(args.dataset)

    processed = 0
    for imgp, rel, images_root in find_images(ds_root):
        # choose labels root
        labels_root = Path(args.out_labels) if args.out_labels else default_labels_root(images_root)
        labels_root.mkdir(parents=True, exist_ok=True)
        lblp = labels_root / rel.with_suffix(".txt")
        lblp.parent.mkdir(parents=True, exist_ok=True)

        im = cv2.imread(str(imgp))
        if im is None:
            print(f"[skip] cannot read {imgp}")
            continue
        H, W = im.shape[:2]

        r = model.predict(source=im, conf=args.conf, verbose=False)[0]
        if r.boxes is None or getattr(r, "keypoints", None) is None:
            # preserve existing (do not erase); skip writing
            if lblp.exists():
                print(f"[warn] no predictions; preserved existing {lblp}")
            else:
                print(f"[warn] no predictions; left unlabeled {lblp}")
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kobj  = r.keypoints
        if hasattr(kobj, "xyn") and kobj.xyn is not None:
            K = kobj.xyn.cpu().numpy()  # normalized
        else:
            K = kobj.xy.cpu().numpy()
            K = K / [W, H]

        order = confs.argsort()[::-1][: args.topk]
        lines = []
        for i in order:
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            # nose, ear_left, ear_right with v=2
            kpts = [
                (K[i, 0, 0], K[i, 0, 1], 2.0),
                (K[i, 1, 0], K[i, 1, 1], 2.0),
                (K[i, 2, 0], K[i, 2, 1], 2.0),
            ]
            lines.append(to_pose_line(0, cx, cy, bw, bh, kpts))

        if lines:
            lblp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[ok] wrote {lblp}")
            processed += 1
        else:
            # preserve
            print(f"[warn] empty predictions; preserved existing {lblp}")

    if processed == 0:
        print("[note] no images labeled. Check paths / confidence / weights.")

if __name__ == "__main__":
    main()