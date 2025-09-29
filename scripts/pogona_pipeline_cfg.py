#!/usr/bin/env python3
"""CLI wrapper around the PoseTrainer with sane defaults.

This replaces the loose training script with something that can be shared by
Optuna sweeps, SLURM jobs, and local experimentation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lizard_tracking.config import PoseTrainingConfig
from lizard_tracking.pipelines import PoseTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pogona head pose model")
    parser.add_argument("--data", default="data/pogona_head_pose.yaml", help="YAML describing the dataset")
    parser.add_argument("--model", default="yolo11s-pose.pt", help="Base checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project", default="runs/pose")
    parser.add_argument("--run-name", default="pogona_head_pose")
    parser.add_argument("--extra", type=str, help="JSON dict of additional Ultralytics overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_overrides = json.loads(args.extra) if args.extra else {}

    cfg = PoseTrainingConfig(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        patience=args.patience,
        resume=args.resume,
        project=args.project,
        run_name=args.run_name,
        extra_overrides=extra_overrides,
    )

    trainer = PoseTrainer(cfg)
    print(f"[TRAIN] {cfg.model} → {cfg.run_directory}")
    results = trainer.train()
    try:
        metrics = getattr(results, "results_dict", None)
        if metrics:
            print("[RESULTS]", json.dumps(metrics, indent=2))
    except Exception:  # pragma: no cover - logging only
        pass

    weights = None
    try:
        weights = trainer.best_checkpoint()
        print(f"[VAL] Using {weights}")
    except FileNotFoundError:
        print("[VAL] best.pt not found; skipping validation")

    if weights is not None and Path(weights).exists():
        trainer.validate(str(weights))


if __name__ == "__main__":
    main()
