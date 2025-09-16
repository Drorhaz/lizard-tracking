#!/usr/bin/env python3
"""
Pogona Head Detection - Sanity Training Script
Goal: run a tiny, reliable training pass to verify learning and metrics > 0.
"""

from ultralytics import YOLO

def main():
    # 1) Load pretrained YOLOv11-small
    model = YOLO('yolo11s.pt')

    # 2) Minimal, safe training pass (CPU to avoid MPS issues)
    results = model.train(
        data='configs/pogona_head.yaml',
        epochs=3,          # short sanity run
        imgsz=640,
        batch=8,           # small fixed batch
        device='cpu',      # CPU avoids MPS training bugs on Mac
        patience=5,        # early-stop window (mostly inert here)
        optimizer='AdamW',
        lr0=0.001,
        workers=2,         # macOS-friendly
        plots=True,
        save=True,
        verbose=True,
        seed=42
    )

    print("\n[Sanity] Training completed.")
    print(f"[Sanity] Save dir: {results.save_dir}")
    print("[Sanity] Use the generated results.csv and best.pt to confirm non-zero losses/metrics.")

    return results

if __name__ == '__main__':
    main()
