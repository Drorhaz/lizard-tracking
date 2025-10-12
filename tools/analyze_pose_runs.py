#!/usr/bin/env python3
"""Utility helpers to inspect YOLO pose training runs.

This script enables two main functions:
  * Compare multiple run directories (mAP, precision, recall, etc.).
  * Plot loss/metric curves for a specific run.

Examples
--------
Compare every run under ``runs/pose``:
    python tools/analyze_pose_runs.py compare runs/pose/*

Plot metrics for one run:
    python tools/analyze_pose_runs.py plot runs/pose/pogona_head_pose2 --out plots/pogona_head_pose2.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import pandas as pd

METRIC_KEYS = [
    "metrics/pose/mAP50",
    "metrics/pose/mAP50-95",
    "metrics/pose/P",
    "metrics/pose/R",
]
LOSS_COLUMNS = [
    "train/box_loss",
    "train/pose_loss",
    "train/kpt_loss",
    "val/box_loss",
    "val/pose_loss",
    "val/kpt_loss",
]
METRIC_PLOT_KEYS = [
    "metrics/pose/mAP50-95",
    "metrics/pose/mAP50",
    "metrics/pose/P",
    "metrics/pose/R",
]

CONFIG = {
    "compare": [
        "runs/pose/pogona_head_pose*",
        "runs/pose/optuna_*",
    ],
    "plots": [
        {"run": "runs/pose/pogona_head_pose2"},
    ],
}

OUTPUT_ROOT = Path("output/analytics")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class RunSummary:
    run_dir: Path
    epochs: int
    metrics: dict


def load_results(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing results.csv in {run_dir}")
    df = pd.read_csv(csv_path)
    df.index += 1  # epoch numbers start at 1
    return df


def summarize_run(run_dir: Path) -> RunSummary:
    df = load_results(run_dir)
    epochs = len(df)
    metrics = {}
    last_row = df.iloc[-1].to_dict()
    for key in METRIC_KEYS:
        metrics[key] = last_row.get(key)
    return RunSummary(run_dir=run_dir, epochs=epochs, metrics=metrics)


def compare_runs(run_dirs: Iterable[str]) -> None:
    rows: List[RunSummary] = []
    for entry in run_dirs:
        path = Path(entry)
        if path.is_dir():
            try:
                rows.append(summarize_run(path))
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
        else:
            print(f"[WARN] {path} not found", file=sys.stderr)
    if not rows:
        print("No runs to compare", file=sys.stderr)
        return

    data = []
    for row in rows:
        record = {
            "run": row.run_dir.name,
            "epochs": row.epochs,
        }
        for key, value in row.metrics.items():
            if value is None or value != value:
                record[key] = None
            else:
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = None
    data.append(record)

    df = pd.DataFrame(data)
    df = df.sort_values(by="metrics/pose/mAP50-95", ascending=False, na_position="last")
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False, justify="center"))

    # simple bar plot of mAP50-95 if possible
    if df["metrics/pose/mAP50-95"].notna().any():
        summary_csv = OUTPUT_ROOT / "run_comparison.csv"
        df.to_csv(summary_csv, index=False)
        print(f"[compare] Saved comparison table to {summary_csv.resolve()}")

        fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.5)))
        ax.barh(df["run"], df["metrics/pose/mAP50-95"], color="#3D7E9A")
        ax.set_xlabel("Pose mAP@0.5:0.95")
        ax.invert_yaxis()
        ax.set_title("Run comparison")
        fig.tight_layout()
        out_path = OUTPUT_ROOT / "run_comparison.png"
        fig.savefig(out_path)
        print(f"[compare] Saved comparison plot to {out_path.resolve()}")


def plot_run(run_dir: str, out_path: Optional[str]) -> None:
    path = Path(run_dir)
    df = load_results(path)
    metric_cols = [col for col in METRIC_PLOT_KEYS if col in df]
    if metric_cols:
        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    else:
        fig, ax_loss = plt.subplots(figsize=(10, 6))
        ax_metric = None

    for col in LOSS_COLUMNS:
        if col in df:
            ax_loss.plot(df.index, df[col], label=col)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Loss curves – {path.name}")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, linestyle='--', alpha=0.4)

    if ax_metric is not None:
        for col in metric_cols:
            ax_metric.plot(df.index, df[col], label=col)
        ax_metric.set_xlabel("Epoch")
        ax_metric.set_ylabel("Metric")
        ax_metric.set_ylim(0, 1.05)
        ax_metric.set_title("Validation metrics")
        ax_metric.legend(loc="lower right")
        ax_metric.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    if out_path:
        out_file = Path(out_path)
    else:
        out_file = OUTPUT_ROOT / f"{path.name}_curves.png"
    fig.savefig(out_file)
    print(f"[plot] Saved plot to {out_file.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pose training runs")
    sub = parser.add_subparsers(dest="command")

    cmp_parser = sub.add_parser("compare", help="Compare multiple runs")
    cmp_parser.add_argument("runs", nargs="+", help="Run directories to compare")

    plot_parser = sub.add_parser("plot", help="Plot metrics for a single run")
    plot_parser.add_argument("run", help="Run directory")
    plot_parser.add_argument("--out", help="Output image path")

    args = parser.parse_args()
    if args.command == "compare":
        compare_runs(args.runs)
    elif args.command == "plot":
        plot_run(args.run, args.out)
    elif args.command is None:
        # fall back to config-based execution
        expanded: List[str] = []
        for pattern in CONFIG.get("compare", []):
            matches = [str(Path(m)) for m in glob(pattern)]
            expanded.extend(matches)
        if expanded:
            print(f"[config] Comparing runs: {', '.join(expanded)}")
            compare_runs(expanded)
        else:
            print("[config] No runs matched compare globs", file=sys.stderr)

        for entry in CONFIG.get("plots", []):
            run = entry.get("run")
            if not run:
                continue
            out = entry.get("out")
            if out:
                out_path = Path(out)
                if not out_path.is_absolute():
                    out_path = OUTPUT_ROOT / out_path.name
            else:
                out_path = OUTPUT_ROOT / f"{Path(run).name}_curves.png"
            print(f"[config] Plotting {run} → {out_path}")
            try:
                plot_run(run, str(out_path))
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
