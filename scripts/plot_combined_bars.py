#!/usr/bin/env python3
"""
Generate the combined comparison bar chart directly from model_comparison.csv.
Useful when you already have the CSV and don't want to rerun the full comparison.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("output/compare/model_comparison.csv")
OUT_PATH = Path("output/compare/cmp_overview.png")


def _parse_float(row: dict, key: str) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _maybe_percent(value: float) -> float:
    if math.isnan(value):
        return value
    return value * 100.0 if value <= 1.5 else value


def _collect_rows() -> List[Tuple[str, float, float, float]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found. Run compare_runs_full.py first.")

    rows: List[Tuple[str, float, float, float]] = []
    with CSV_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row.get("run", "")
            if not run:
                continue

            acc = _parse_float(row, "best_map")
            if math.isnan(acc):
                acc = _parse_float(row, "final_map")
            acc = _maybe_percent(acc)

            det = _parse_float(row, "gpu0_det_rate")
            if math.isnan(det):
                det = _parse_float(row, "cpu_det_rate")
            det = _maybe_percent(det)

            speed = _parse_float(row, "gpu0_fps")
            if math.isnan(speed):
                speed = _parse_float(row, "cpu_fps")
            if math.isnan(speed):
                avg_ms = _parse_float(row, "gpu0_avg_ms")
                if math.isnan(avg_ms) or avg_ms <= 0:
                    avg_ms = _parse_float(row, "cpu_avg_ms")
                if not math.isnan(avg_ms) and avg_ms > 0:
                    speed = 1000.0 / avg_ms

            rows.append((run, acc, det, speed))
    return rows


def make_plot():
    rows = _collect_rows()
    if not rows:
        raise RuntimeError("No runs found in model_comparison.csv")

    runs = [r[0] for r in rows]
    accuracy = np.array([r[1] for r in rows], dtype=float)
    detection = np.array([r[2] for r in rows], dtype=float)
    speed = np.array([r[3] for r in rows], dtype=float)

    metrics = []
    if not np.isnan(accuracy).all():
        metrics.append(("Accuracy (%)", accuracy, "#1f77b4"))
    if not np.isnan(detection).all():
        metrics.append(("Detection Rate (%)", detection, "#ff7f0e"))
    if not np.isnan(speed).all():
        metrics.append(("Speed (FPS)", speed, "#2ca02c"))

    if not metrics:
        raise RuntimeError("No valid metrics found to plot.")

    x = np.arange(len(runs))
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    offsets = np.linspace(-bar_width * (n_metrics - 1) / 2, bar_width * (n_metrics - 1) / 2, n_metrics)

    fig_width = max(7.0, 1.2 * len(runs))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    for offset, (label, values, color) in zip(offsets, metrics):
        bars = ax.bar(x + offset, values, width=bar_width, label=label, color=color)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison overview")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=170)
    plt.close(fig)
    print(f"[ok] wrote {OUT_PATH}")


if __name__ == "__main__":
    make_plot()
