#!/usr/bin/env python3
import sys
from pathlib import Path

def is_pose_line(tokens):
    return len(tokens) == 15

def main(ds_root="dataset"):
    ds = Path(ds_root)
    lbl = ds / "labels"
    if not lbl.exists():
        print(f"[error] labels folder not found: {lbl}")
        sys.exit(1)

    det, pose, bad = 0, 0, 0
    files = list(lbl.rglob("*.txt"))
    for f in files:
        for line in f.read_text().splitlines():
            t = line.strip().split()
            if not t: 
                continue
            if len(t) == 5:
                det += 1
            elif is_pose_line(t):
                pose += 1
            else:
                bad += 1
    print(f"files: {len(files)} | det-lines: {det} | pose-lines: {pose} | bad-lines: {bad}")

if __name__ == "__main__":
    main(*sys.argv[1:])
