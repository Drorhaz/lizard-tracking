#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import cv2

GREEN=(0,255,0); RED=(0,0,255); BLUE=(255,0,0); YELLOW=(0,255,255)

def parse_pose_line(t, W, H):
    cx, cy, w, h = map(float, t[1:5])
    x1 = int((cx - w/2.0) * W); y1 = int((cy - h/2.0) * H)
    x2 = int((cx + w/2.0) * W); y2 = int((cy + h/2.0) * H)
    k = list(map(float, t[5:]))
    kpts = [
        (int(k[0] * W), int(k[1] * H)),
        (int(k[3] * W), int(k[4] * H)),
        (int(k[6] * W), int(k[7] * H)),
    ]
    return (x1, y1, x2, y2), kpts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out_dir", default="qa_out")
    ap.add_argument("--num", type=int, default=20)
    args = ap.parse_args()

    ds = Path(args.dataset)
    img_root, lbl_root = ds / "images", ds / "labels"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    random.shuffle(imgs)
    picked = imgs[:args.num]

    count = 0
    for imgp in picked:
        lblp = lbl_root / imgp.relative_to(img_root).with_suffix(".txt")
        if not lblp.exists():
            print(f"[skip] no label for {imgp}")
            continue
        im = cv2.imread(str(imgp))
        if im is None:
            print(f"[skip] cannot read {imgp}")
            continue
        H, W = im.shape[:2]
        for line in lblp.read_text(encoding="utf-8").splitlines():
            t = line.strip().split()
            if not t: 
                continue
            if len(t) != 15:
                print(f"[warn] not pose format: {lblp}")
                continue
            bbox, kpts = parse_pose_line(t, W, H)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(im, (x1, y1), (x2, y2), GREEN, 2)
            cv2.circle(im, kpts[0], 4, RED, -1)
            cv2.circle(im, kpts[1], 4, BLUE, -1)
            cv2.circle(im, kpts[2], 4, BLUE, -1)
            mx = int(0.5 * (kpts[1][0] + kpts[2][0])); my = int(0.5 * (kpts[1][1] + kpts[2][1]))
            cv2.line(im, kpts[0], (mx, my), YELLOW, 2)
        outp = out / imgp.name
        outp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp), im)
        count += 1
    print(f"[done] wrote {count} QA images to {out}")

if __name__ == "__main__":
    main()
