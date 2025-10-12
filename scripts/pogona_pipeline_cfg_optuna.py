#!/usr/bin/env python3
"""Config-driven pipeline for YOLO pose training with Optuna tuning."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "lib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lizard_tracking.config import PoseTrainingConfig, VideoTrackingConfig
from lizard_tracking.pipelines import PoseTrainer, VideoTracker


CONFIG: Dict[str, Any] = {
    "DATA_YAML": "data/pogona_head_pose.yaml",
    "PROJECT": "runs/pose",
    "RUN_NAME": "pogona_head_pose",
    "BASE_MODELS": ["./yolo11s-pose.pt"],
    "TRAIN": {
        "epochs": 150,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "lr0": 0.01,
        "weight_decay": 0.0005,
    },
    "OPTUNA": {
        "enabled": True,
        "n_trials": 7,
        "epochs": 25,
        "search": {
            "lr0": [1e-4, 3e-2],
            "weight_decay": [1e-5, 1e-3],
            "imgsz": [512, 768],
        },
    },
    "TRACK": {
        "enabled": False,
        "video_path": "videos/exp1.mp4",
        "out_csv": "output/trajectory.csv",
        "out_video": "output/trajectory_overlay.mp4",
    },
}

_KNOWN_FIELDS = {"epochs", "imgsz", "batch", "device", "lr0", "weight_decay", "patience", "resume"}


def _score_from_results(results, run_dir: Optional[Path] = None) -> float:
    def extract(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        if not metrics:
            return None
        for key in ("metrics/pose/mAP50-95", "mAP50-95", "map50-95"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
        return None

    score = None
    score = extract(getattr(results, "results_dict", None))
    if score is None:
        score = extract(getattr(results, "metrics", None))

    if score is None and run_dir is not None:
        csv_path = Path(run_dir) / "results.csv"
        if csv_path.exists():
            try:
                import csv

                with csv_path.open(newline="") as fh:
                    rows = list(csv.DictReader(fh))
                if rows:
                    last = rows[-1]
                    for key in ("metrics/pose/mAP50-95", "mAP50-95", "map50-95"):
                        value = last.get(key)
                        if value not in (None, "", "nan"):
                            score = float(value)
                            break
            except Exception:
                pass

    return score if score is not None else float("-inf")


def _build_training_config(
    *,
    base_model: str,
    run_name: str,
    project: str,
    data_yaml: str,
    train_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> PoseTrainingConfig:
    merged = dict(train_cfg)
    if overrides:
        merged.update(overrides)
    main_kwargs = {k: merged.pop(k) for k in list(merged.keys()) if k in _KNOWN_FIELDS}
    return PoseTrainingConfig(
        data_yaml=data_yaml,
        model=base_model,
        project=project,
        run_name=run_name,
        epochs=main_kwargs.get("epochs", 150),
        imgsz=main_kwargs.get("imgsz", 640),
        batch=main_kwargs.get("batch", 16),
        device=main_kwargs.get("device", "0"),
        lr0=main_kwargs.get("lr0", 0.01),
        weight_decay=main_kwargs.get("weight_decay", 5e-4),
        patience=main_kwargs.get("patience"),
        resume=main_kwargs.get("resume", False),
        extra_overrides=merged,
    )


def _train_once(cfg: PoseTrainingConfig):
    trainer = PoseTrainer(cfg)
    return trainer.train()


def _validate(weights: str, data_yaml: str):
    trainer = PoseTrainer(PoseTrainingConfig(data_yaml=data_yaml, model=weights))
    return trainer.validate(weights)


def run_optuna(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    import optuna

    search_space = config["OPTUNA"]["search"]

    def objective(trial: "optuna.Trial") -> float:
        base = trial.suggest_categorical("base", config["BASE_MODELS"])
        hp_overrides: Dict[str, Any] = {}
        if "lr0" in search_space:
            lo, hi = search_space["lr0"]
            hp_overrides["lr0"] = trial.suggest_float("lr0", lo, hi, log=True)
        if "weight_decay" in search_space:
            lo, hi = search_space["weight_decay"]
            hp_overrides["weight_decay"] = trial.suggest_float("weight_decay", lo, hi, log=True)
        if "imgsz" in search_space:
            lo, hi = search_space["imgsz"]
            hp_overrides["imgsz"] = trial.suggest_int("imgsz", int(lo), int(hi), step=32)

        hp_overrides["epochs"] = config["OPTUNA"]["epochs"]
        run_name = f"optuna_{trial.number:03d}"
        train_cfg = _build_training_config(
            base_model=base,
            run_name=run_name,
            project=config["PROJECT"],
            data_yaml=config["DATA_YAML"],
            train_cfg=config["TRAIN"],
            overrides=hp_overrides,
        )
        _train_once(train_cfg)
        run_dir = Path(config["PROJECT"]) / run_name
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            return float("-inf")
        results = _validate(str(best_pt), config["DATA_YAML"])
        return _score_from_results(results, run_dir)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config["OPTUNA"]["n_trials"])
    params = study.best_params
    base = params.pop("base")
    print("[OPTUNA] best base", base, "params", params)
    return base, params


def maybe_run_tracking(config: Dict[str, Any], weights: Path) -> None:
    if not config["TRACK"]["enabled"]:
        return
    track_cfg = VideoTrackingConfig(
        source=config["TRACK"]["video_path"],
        weights=str(weights),
        overlay_video=True,
    )
    track_cfg.csv_path_override = Path(config["TRACK"]["out_csv"])
    track_cfg.overlay_path_override = Path(config["TRACK"]["out_video"])
    track_cfg.output_root = track_cfg.csv_path_override.parent
    tracker = VideoTracker(track_cfg)
    tracker.run()


def main() -> None:
    cfg = CONFIG
    base_model = cfg["BASE_MODELS"][0]
    overrides: Dict[str, Any] = {}

    if cfg["OPTUNA"]["enabled"]:
        base_model, overrides = run_optuna(cfg)

    train_cfg = _build_training_config(
        base_model=base_model,
        run_name=cfg["RUN_NAME"],
        project=cfg["PROJECT"],
        data_yaml=cfg["DATA_YAML"],
        train_cfg=cfg["TRAIN"],
        overrides=overrides,
    )

    run_dir = train_cfg.run_directory
    print(f"[TRAIN] {train_cfg.model} → {run_dir}")
    _train_once(train_cfg)

    best_pt = run_dir / "weights" / "best.pt"
    # best pt can be found in all run_dir + * folders
    for folder in run_dir.glob("*"):
        candidate = folder / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
            break
    
    if best_pt.exists():
        results = _validate(str(best_pt), cfg["DATA_YAML"])
        score = _score_from_results(results, run_dir)
        print("[VAL] score", score)
        maybe_run_tracking(cfg, best_pt)
    else:
        print("[WARN] best.pt not found; skipping validation and tracking")


if __name__ == "__main__":
    main()
