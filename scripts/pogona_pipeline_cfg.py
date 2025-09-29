#!/usr/bin/env python3
"""Train the pogona head pose model using a simple config block."""
from __future__ import annotations

import shutil
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lizard_tracking.config import PoseTrainingConfig
from lizard_tracking.pipelines import PoseTrainer

# ---------------------------------------------------------------------------
# Adjust these values instead of passing CLI arguments.
# ---------------------------------------------------------------------------
CONFIG = {
    "data_yaml": "data/pogona_head_pose.yaml",
    "model": "yolo11s-pose.pt",
    "epochs": 150,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "lr0": 0.01,
    "weight_decay": 5e-4,
    "patience": None,
    "resume": False,
    "project": "runs/pose",
    "run_name": "pogona_head_pose",
    "extra_overrides": {},  # additional Ultralytics overrides
    "export_best_to": "output/models/head_pose/best.pt",
    "skip_training": True,
}


def main() -> None:
    config = CONFIG.copy()
    export_path = config.pop("export_best_to", None)
    skip_training = bool(config.pop("skip_training", False))
    cfg = PoseTrainingConfig(**config)

    trainer = PoseTrainer(cfg)
    print(f"[TRAIN] {cfg.model} → {cfg.run_directory}")

    if not skip_training:
        results = trainer.train()
        metrics = getattr(results, "results_dict", None)
        if metrics:
            print("[RESULTS]", metrics)
    else:
        print("[INFO] skip_training=True; training phase skipped")

    weights = None
    try:
        weights = trainer.best_checkpoint()
        print(f"[VAL] Using {weights}")
    except FileNotFoundError:
        print("[WARN] best.pt not found; skipping validation")

    if weights and Path(weights).exists():
        trainer.validate(str(weights))
        print(f"[INFO] Artifacts saved under {cfg.run_directory.resolve()}")
        if export_path:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights, export_path)
            print(f"[EXPORT] Copied best checkpoint to {export_path.resolve()}")
    else:
        print("[WARN] No checkpoint available to export")


if __name__ == "__main__":
    main()
