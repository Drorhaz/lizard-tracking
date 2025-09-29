#!/usr/bin/env python3
# Compare Ultralytics YOLO pose runs and generate plots + a summary table.
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

CONFIG: Dict[str, Any] = {
    "RUNS_DIR": "runs/pose",
    "OUT_DIR": "output/compare",
    "INCLUDE": [],
    "EXCLUDE": ["predict", "traincache"],
    "PER_RUN_CURVES": True,
    "MAX_EPOCHS": None,
    "SAVE_PNG_DPI": 170,
}

MAP_CANDIDATES = [
    "metrics/pose/mAP50-95","keypoints/mAP50-95","metrics/mAP50-95","mAP50-95","map50-95",
    "metrics/box/mAP50-95","box/mAP50-95",
]
TRAIN_LOSS_CANDS = ["train/box_loss","train/cls_loss","train/dfl_loss","train/kobj_loss","train/kpt_loss","train/seg_loss"]
VAL_LOSS_CANDS   = ["val/box_loss","val/cls_loss","val/dfl_loss","val/kobj_loss","val/kpt_loss","val/seg_loss"]

def _find_runs(root: Path) -> List[Path]:
    runs = []
    for p in root.glob("*"):
        if not p.is_dir(): continue
        csvp = p / "results.csv"
        if csvp.exists():
            s = str(p)
            inc, exc = CONFIG["INCLUDE"], CONFIG["EXCLUDE"]
            if inc and not any(tok in s for tok in inc): continue
            if exc and any(tok in s for tok in exc): continue
            runs.append(p)
    return sorted(runs)

def _pick_map_column(df: pd.DataFrame) -> str | None:
    for c in MAP_CANDIDATES:
        if c in df.columns: return c
    return None

def _available_losses(df: pd.DataFrame, cands: List[str]) -> List[str]:
    return [c for c in cands if c in df.columns]

def _clip_epochs(df: pd.DataFrame, max_epochs: int | None) -> pd.DataFrame:
    if max_epochs is None: return df
    if "epoch" in df.columns: return df[df["epoch"] < max_epochs].reset_index(drop=True)
    return df.iloc[:max_epochs, :].reset_index(drop=True)

def _safe_last(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.values[-1]) if len(s) else float("nan")

def compare_runs():
    runs_dir = Path(CONFIG["RUNS_DIR"])
    out_dir = Path(CONFIG["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run"
    if CONFIG["PER_RUN_CURVES"]: per_run_dir.mkdir(parents=True, exist_ok=True)

    runs = _find_runs(runs_dir)
    if not runs:
        print(f"[warn] no runs with results.csv found under {runs_dir}"); return

    records = []
    map_series = {}
    all_loss_series = {}

    for r in runs:
        name = r.name
        csvp = r / "results.csv"
        try:
            df = pd.read_csv(csvp)
        except Exception as e:
            print(f"[skip] cannot read {csvp}: {e}"); continue

        df = _clip_epochs(df, CONFIG["MAX_EPOCHS"])
        map_col = _pick_map_column(df)
        train_losses = _available_losses(df, TRAIN_LOSS_CANDS)
        val_losses   = _available_losses(df, VAL_LOSS_CANDS)

        if map_col and df[map_col].notna().any():
            xs = list(range(len(df[map_col])))
            ys = [float(v) if pd.notna(v) else float("nan") for v in df[map_col].values]
            map_series[name] = (xs, ys)

        for loss_name in set(train_losses + val_losses):
            ys = df[loss_name].values.tolist()
            xs = list(range(len(ys)))
            all_loss_series.setdefault(loss_name, {})
            all_loss_series[loss_name][name] = (xs, ys)

        row = {"run": name, "results_csv": str(csvp), "epochs": len(df)}
        if map_col:
            row["map_col"] = map_col
            row["final_map"] = _safe_last(df[map_col])
            row["best_map"]  = float(df[map_col].max())
            row["best_map_epoch"] = int(df[map_col].idxmax()) if df[map_col].notna().any() else -1
        for c in ["train/kpt_loss","val/kpt_loss","train/box_loss","val/box_loss"]:
            if c in df.columns: row[c] = _safe_last(df[c])
        records.append(row)

        if CONFIG["PER_RUN_CURVES"]:
            fig = plt.figure(figsize=(7,4))
            if map_col and df[map_col].notna().any():
                plt.plot(df[map_col].values, label=f"{map_col}")
            plotted = 0
            for c in (train_losses + val_losses):
                if plotted >= 4: break
                plt.plot(df[c].values, label=c, alpha=0.8); plotted += 1
            plt.title(name); plt.xlabel("epoch"); plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(per_run_dir / f"{name}_curves.png", dpi=CONFIG["SAVE_PNG_DPI"])
            plt.close(fig)

    if records:
        summary = pd.DataFrame.from_records(records)
        sort_cols = [c for c in ["final_map","best_map"] if c in summary.columns]
        if sort_cols: summary = summary.sort_values(by=sort_cols, ascending=False)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[ok] wrote {summary_path}")

    if map_series:
        fig = plt.figure(figsize=(9,5))
        for name, (xs, ys) in map_series.items():
            plt.plot(xs, ys, label=name)
        plt.xlabel("epoch"); plt.ylabel("mAP (auto-picked column)"); plt.title("Validation mAP vs. epoch")
        plt.grid(True, alpha=0.3); plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "map_curve.png", dpi=CONFIG["SAVE_PNG_DPI"])
        plt.close(fig)
        print(f"[ok] wrote {out_dir/'map_curve.png'}")
    else:
        print("[warn] no mAP-like column found in any results.csv")

    if all_loss_series:
        loss_names = list(all_loss_series.keys())
        chunk = 6
        for start in range(0, len(loss_names), chunk):
            fig = plt.figure(figsize=(10,6))
            for loss_name in loss_names[start:start+chunk]:
                series = all_loss_series[loss_name]
                for run_name, (xs, ys) in series.items():
                    plt.plot(xs, ys, label=f"{loss_name} ({run_name})", alpha=0.9)
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curves (train/val)")
            plt.grid(True, alpha=0.3); plt.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(out_dir / f"loss_curves_{start//chunk+1}.png", dpi=CONFIG["SAVE_PNG_DPI"])
            plt.close(fig)
        print(f"[ok] wrote loss curve figures to {out_dir}")
    else:
        print("[warn] no recognized loss columns found")

if __name__ == "__main__":
    compare_runs()
