#!/usr/bin/env python3
"""
Convert YOLO detection labels to YOLO pose labels with placeholder keypoints (v=0).
- Accepts both layouts:
    dataset/labels/{train,val,test}/...
    dataset/{train,val,test}/labels/...
- Adds three keypoints (nose, ear_left, ear_right) as 0.0 0.0 with visibility=0.
- Keeps existing pose labels untouched.

Example:
  python autogenerate/convert_box_to_pose_placeholders.py --labels_root dataset/labels
"""
import argparse
from pathlib import Path

def line_is_det(toks): return len(toks) == 5
def line_is_pose(toks): return len(toks) == 15

def convert_det_to_pose_line(toks):
    cls, cx, cy, w, h = toks
    # 3 keypoints (x, y, v=0) -> coords arbitrary when v=0; keep 0.0
    return f"{cls} {cx} {cy} {h} {h} 0.000000 0.000000 0 0.000000 0.000000 0 0.000000 0.000000 0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_root", required=True, help="root dir of labels")
    args = ap.parse_args()

    root = Path(args.labels_root)
    count = 0
    for p in root.rglob("*.txt"):
        lines = p.read_text(encoding="utf-8").splitlines()
        new_lines = []
        changed = False
        for ln in lines:
            t = ln.strip().split()
            if not t:
                continue
            if line_is_pose(t):
                new_lines.append(ln)     # keep as-is
            elif line_is_det(t):
                # convert detection line to pose placeholders
                cls, cx, cy, w, h = t
                new_lines.append(
                    f"{cls} {cx} {cy} {w} {h} 0.000000 0.000000 0 0.000000 0.000000 0 0.000000 0.000000 0"
                )
                changed = True
            else:
                # unknown line; keep to be safe
                new_lines.append(ln)
        if changed:
            p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            count += 1
            print(f"[ok] converted {p}")
    print(f"[done] converted {count} files to pose with v=0")

if __name__ == "__main__":
    main()