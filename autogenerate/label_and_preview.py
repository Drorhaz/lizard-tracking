#!/usr/bin/env python3
"""
Randomly sample N images, generate YOLO11 pose labels with a pose model,
save the .txt labels, and also save previews with bbox + nose/ears + vector.

Layouts supported (labels kept in the standard place, not new folders):
  A) dataset/images/{split}/...   + dataset/labels/{split}/...
  B) dataset/{split}/images/...   + dataset/{split}/labels/...

Example:
  python autogenerate/label_and_preview.py \
    --weights autogenerate/best.pt \
    --dataset dataset \
    --num 200 --conf 0.10 --topk 2 \
    --out output/previews
"""
import argparse, random
from pathlib import Path
import cv2
from ultralytics import YOLO

EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
GREEN=(0,255,0); RED=(0,0,255); BLUE=(255,0,0); YELLOW=(0,255,255)

def find_images(dataset_root: Path):
    # yield (img_path, images_root, labels_root)
    # Layout A
    A_images = dataset_root / "images"
    if A_images.exists():
        A_labels = dataset_root / "labels"
        for p in A_images.rglob("*"):
            if p.suffix.lower() in EXTS:
                yield p, A_images, A_labels
    # Layout B
    for split in ("train","val","test"):
        B_images = dataset_root / split / "images"
        if B_images.exists():
            B_labels = dataset_root / split / "labels"
            for p in B_images.rglob("*"):
                if p.suffix.lower() in EXTS:
                    yield p, B_images, B_labels

def to_pose_line(cls, cx, cy, w, h, kpts):
    parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for x, y, v in kpts:
        parts += [f"{x:.6f}", f"{y:.6f}", f"{v:.1f}"]
    return " ".join(parts)

def draw_preview(im, W, H, cx, cy, w, h, kpts):
    x1 = int((cx - w/2)*W); y1 = int((cy - h/2)*H)
    x2 = int((cx + w/2)*W); y2 = int((cy + h/2)*H)
    cv2.rectangle(im, (x1,y1), (x2,y2), GREEN, 2)
    # kpts are normalized
    nose  = (int(kpts[0][0]*W), int(kpts[0][1]*H))
    ear_l = (int(kpts[1][0]*W), int(kpts[1][1]*H))
    ear_r = (int(kpts[2][0]*W), int(kpts[2][1]*H))
    cv2.circle(im, nose, 4, RED, -1)
    cv2.circle(im, ear_l, 4, BLUE, -1)
    cv2.circle(im, ear_r, 4, BLUE, -1)
    mx, my = int((ear_l[0]+ear_r[0])/2), int((ear_l[1]+ear_r[1])/2)
    cv2.line(im, nose, (mx,my), YELLOW, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="pose model (.pt)")
    ap.add_argument("--dataset", required=True, help="dataset root")
    ap.add_argument("--num", type=int, default=100, help="random images to process")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--out", default="output/previews", help="single folder for previews")
    ap.add_argument("--seed", type=int, default=None, help="random seed (None=randomized)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite label file even if it exists (default: replace only if we have predictions)")
    args = ap.parse_args()

    if args.seed is None:
        # new randomness each run
        import os, time; random.seed((os.getpid(), time.time()))
    else:
        random.seed(args.seed)

    ds = Path(args.dataset)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # gather all images once, then sample N
    all_imgs = list(find_images(ds))
    if len(all_imgs) == 0:
        raise SystemExit("[error] no images found; check dataset layout")
    sample = random.sample(all_imgs, k=min(args.num, len(all_imgs)))

    wrote = 0; previewed = 0; skipped_empty = 0
    for imgp, images_root, labels_root in sample:
        labels_root.mkdir(parents=True, exist_ok=True)
        lblp = labels_root / imgp.relative_to(images_root).with_suffix(".txt")
        lblp.parent.mkdir(parents=True, exist_ok=True)

        im = cv2.imread(str(imgp))
        if im is None:
            print(f"[skip] cannot read {imgp}")
            continue
        H, W = im.shape[:2]

        r = model.predict(source=im, conf=args.conf, verbose=False)[0]
        if r.boxes is None or getattr(r, "keypoints", None) is None:
            skipped_empty += 1
            # do not erase existing files unless overwrite flag + we want to clear (we don't)
            print(f"[warn] no predictions for {imgp.name}")
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kobj  = r.keypoints
        if hasattr(kobj, "xyn") and kobj.xyn is not None:
            K = kobj.xyn.cpu().numpy()  # normalized
        else:
            K = kobj.xy.cpu().numpy() / [W, H]

        order = confs.argsort()[::-1][:args.topk]
        if len(order) == 0:
            skipped_empty += 1
            print(f"[warn] empty predictions for {imgp.name}")
            continue

        # write labels: one line per kept detection
        lines = []
        for i in order:
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            kpts = [
                (K[i, 0, 0], K[i, 0, 1], 2.0),
                (K[i, 1, 0], K[i, 1, 1], 2.0),
                (K[i, 2, 0], K[i, 2, 1], 2.0),
            ]
            lines.append(to_pose_line(0, cx, cy, bw, bh, kpts))

        if lines:
            if args.overwrite or not lblp.exists():
                lblp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                # if not overwriting, we still replace only if we actually have predictions (safe)
                lblp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            wrote += 1

            # make a preview in a single folder (no subdirs)
            # (to avoid name collisions, prefix with a counter)
            prev = im.copy()
            # draw only the *first* kept detection for clarity
            i = order[0]
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0 / W; cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W; bh = (y2 - y1) /  H
            kpts = [(K[i,0,0],K[i,0,1]), (K[i,1,0],K[i,1,1]), (K[i,2,0],K[i,2,1])]
            draw_preview(prev, W, H, cx, cy, bw, bh, kpts)
            outp = out / f"{imgp.name}"
            cv2.imwrite(str(outp), prev); previewed += 1
        else:
            skipped_empty += 1

    print(f"[done] labeled={wrote} | previews={previewed} | no-pred={skipped_empty} | out={out}")

if __name__ == "__main__":
    main()