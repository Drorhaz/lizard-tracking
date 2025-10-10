#!/usr/bin/env python3
# Compare Ultralytics YOLO pose runs and generate plots + CPU/GPU timings.
from __future__ import annotations
import os, sys, json, time, math, random, contextlib, subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


# ----------------------------- CONFIG (edit here) ----------------------------- #
CONFIG: Dict[str, Any] = {
    "RUNS_DIR": "runs/pose",
    "OUT_DIR": "output/compare",
    "INCLUDE": [],                    # substrings to include (empty => include all)
    "EXCLUDE": ["predict", "traincache"],
    "MAX_EPOCHS": None,               # cap epochs per run for plotting
    "SAVE_PNG_DPI": 170,

    # Per-model plots
    "PLOTS": {
        "per_model_map": True,
        "per_model_losses": True,
        "key_losses": ["train/kpt_loss","val/kpt_loss","train/box_loss","val/box_loss"],
        "max_losses_plotted": 4
    },

    # Timing (uses validation images)
    "DATA_YAML": "data/pogona_head_pose.yaml",  # used to auto-locate val images if VAL_IMAGES_DIR is None
    "VAL_IMAGES_DIR": "dataset/images/val",     # fallback folder for timing images
    "N_TIMING_IMAGES": 40,
    "CONF": 0.25,
    "IOU": 0.45,
    "SEED": 0,
    "SAVE_DEMOS": False,               # save prediction images during timing
    "DEMOS_SUBDIR": "inference_demos",

    # Which devices to time (order matters; GPU first avoids CUDA sticky -1)
    "DEVICES": ["gpu0"],        # "gpu0" means first visible GPU; add "gpu1" if needed.  ["gpu0", "cpu"]
}
# ----------------------------------------------------------------------------- #

# Candidate columns where various Ultralytics versions store mAP

MAP_CANDS = [
    "metrics/mAP50-95(P)",
    "metrics/mAP50(P)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/pose/mAP50-95",
    "metrics/pose/mAP@0.5:0.95",
    "keypoints/mAP50-95",
    "keypoints/mAP@0.5:0.95",
    "metrics/mAP50-95",
    "metrics/mAP@0.5:0.95",
    "mAP50-95",
    "map50-95",
    "pose/mAP50-95",
    "val/pose_map50-95",
    "metrics/box/mAP50-95",
    "box/mAP50-95",
]
LOSS_CANDS = [
    "train/box_loss","val/box_loss",
    "train/kpt_loss","val/kpt_loss",
    "train/cls_loss","val/cls_loss",
    "train/dfl_loss","val/dfl_loss",
    "train/kobj_loss","val/kobj_loss",
    "train/pose_loss","val/pose_loss"
]

# ------------------------------ helpers -------------------------------------- #
def _ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def _find_runs(root: Path) -> List[Path]:
    out = []
    for p in sorted(root.glob("*")):
        if p.is_dir() and (p / "results.csv").exists():
            s = str(p)
            if CONFIG["INCLUDE"] and not any(t in s for t in CONFIG["INCLUDE"]): 
                continue
            if CONFIG["EXCLUDE"] and any(t in s for t in CONFIG["EXCLUDE"]): 
                continue
            out.append(p)
    return out

def _clip(df: pd.DataFrame) -> pd.DataFrame:
    m = CONFIG["MAX_EPOCHS"]
    if m is None: return df
    return df[df["epoch"] < m] if "epoch" in df.columns else df.iloc[:m, :]

def _pick_map(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)

    # 1) exact priority list
    for c in MAP_CANDS:
        if c in cols:
            return c

    # 2) flexible regex: prefer pose '(P)' first, then any 0.5:0.95-ish
    #    e.g., "metrics/mAP50-95(P)" or "metrics/mAP@0.5:0.95 (P)"
    for c in df.columns:
        if re.search(r"mAP\s*50-?95.*\(P\)", c, flags=re.I) or re.search(r"mAP@?0\.5:?0\.95.*\(P\)", c, flags=re.I):
            return c

    # 3) any mAP50-95 style if nothing with (P) found
    for c in df.columns:
        if re.search(r"mAP\s*50-?95", c, flags=re.I) or re.search(r"mAP@?0\.5:?0\.95", c, flags=re.I):
            return c

    return None

def _key_losses(df: pd.DataFrame) -> List[str]:
    ks = [k for k in CONFIG["PLOTS"]["key_losses"] if k in df.columns]
    if not ks:
        ks = [k for k in LOSS_CANDS if k in df.columns]
    return ks[:CONFIG["PLOTS"]["max_losses_plotted"]]

def _weights(run: Path) -> Optional[Path]:
    for f in ["best.pt","last.pt"]:
        p = run / "weights" / f
        if p.exists(): 
            return p
    return None

def _val_dir_from_yaml(yaml_path: Path) -> Optional[Path]:
    try:
        import yaml
        d = yaml.safe_load(open(yaml_path,"r"))
        base = Path(d.get("path","."))
        v = d.get("val", None)
        if v is None: return None
        vp = Path(v)
        if not vp.is_absolute(): vp = base / v
        return vp if vp.exists() else None
    except Exception:
        return None

def _gather_val_images() -> List[Path]:
    vd = None
    if CONFIG.get("DATA_YAML") and Path(CONFIG["DATA_YAML"]).exists():
        vd = _val_dir_from_yaml(Path(CONFIG["DATA_YAML"]))
    if vd is None and CONFIG.get("VAL_IMAGES_DIR"):
        vd = Path(CONFIG["VAL_IMAGES_DIR"])
    imgs = []
    if vd and vd.exists():
        for p in vd.rglob("*"):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
                imgs.append(p)
    return sorted(imgs)

def _size_mb(p: Path) -> float:
    try: return p.stat().st_size / (1024*1024)
    except Exception: return float("nan")

# ---------- safe env contexts so CPU timing doesn’t kill CUDA for the process
@contextlib.contextmanager
def temp_env(**updates):
    old = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _subproc_time_model(weights: Path, img_paths: List[Path], device_label: str, conf: float, iou: float) -> Dict[str, float]:
    """
    Time a model in a clean Python subprocess so env changes (like CUDA_VISIBLE_DEVICES)
    can't leak between CPU and GPU timing runs.
    """
    code = f"""
import json, time, math
from ultralytics import YOLO
import pandas as pd
weights={json.dumps(str(weights))}
imgs={json.dumps([str(p) for p in img_paths])}
conf={conf}; iou={iou}
m = YOLO(weights)
# warmup
_ = m.predict(source=imgs[0], device={"'cpu'" if device_label=="cpu" else "'0'"},
              conf=conf, iou=iou, verbose=False, save=False)
times=[]; dets=[]; confs=[]
for p in imgs:
    t0=time.perf_counter()
    r = m.predict(source=p, device={"'cpu'" if device_label=="cpu" else "'0'"},
                  conf=conf, iou=iou, verbose=False, save=False)
    dt=(time.perf_counter()-t0)*1000.0
    times.append(dt)
    n=0; cs=[]
    if len(r):
        rr=r[0]
        if getattr(rr, "boxes", None) is not None:
            try:
                n=int(rr.boxes.shape[0])
                cs=rr.boxes.conf.detach().cpu().numpy().tolist()
            except: pass
    dets.append(n)
    confs.append(sum(cs)/len(cs) if cs else float("nan"))
s=pd.Series(times)
fps=1000.0/s.mean() if len(s) else float("nan")
det_rate=float((pd.Series(dets)>0).mean()) if len(dets) else float("nan")
avg_conf=float(pd.Series([x for x in confs if not math.isnan(x)]).mean()) if any(not math.isnan(x) for x in confs) else float("nan")
print(json.dumps({{
    "avg_ms": float(s.mean()), "med_ms": float(s.median()),
    "min_ms": float(s.min()), "max_ms": float(s.max()),
    "fps": fps, "det_rate": det_rate,
    "avg_dets": float(pd.Series(dets).mean()) if len(dets) else float("nan"),
    "avg_conf": avg_conf
}}))
"""
    env = os.environ.copy()
    if device_label == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # ensure the SLURM-assigned GPU is visible and indexed as '0' in child
        orig = env.get("SLURM_JOB_GPUS") or env.get("SLURM_STEP_GPUS") or env.get("CUDA_VISIBLE_DEVICES") or "0"
        env["CUDA_VISIBLE_DEVICES"] = str(orig)
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    return json.loads(out.decode("utf-8"))

def _param_count(weights: Path) -> float:
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights))
        return sum(p.numel() for p in m.model.parameters()) / 1e6
    except Exception:
        return float("nan")

def _bar_if_any(out_dir: Path, df: pd.DataFrame, cols: List[str], title: str, xlabel: str, fname: str, invert=False):
    d = df[["run"] + [c for c in cols if c in df.columns]].copy()
    d = d.dropna(how="all", subset=cols)
    if d.empty:
        print(f"[skip] bar '{title}': no finite values")
        return
    # plot side-by-side bars per run
    width = 0.35 if len(cols)==2 else 0.5
    fig = plt.figure(figsize=(9, max(4, 0.4*len(d))))
    y = range(len(d))
    off = (-width/2, width/2) if len(cols)==2 else (0.0,)
    for i, col in enumerate(cols):
        if col not in d.columns: 
            continue
        vals = pd.to_numeric(d[col], errors="coerce")
        ok = vals.notna()
        if ok.any():
            plt.barh([yy+off[i] for yy in y], vals[ok], height=width, label=col)
    plt.yticks(list(y), d["run"])
    plt.title(title); plt.xlabel(xlabel); plt.legend()
    fig.tight_layout(); fig.savefig(out_dir / fname, dpi=CONFIG["SAVE_PNG_DPI"]); plt.close(fig)
    print(f"[ok] wrote {out_dir/fname}")

def _combined_metrics_plot(out_dir: Path, cmp: pd.DataFrame):
    """Create grouped bars (accuracy, detection, speed) for each run."""
    def _pick_val(row, cols: List[str]) -> float:
        for c in cols:
            if c in row and pd.notna(row[c]):
                try:
                    return float(row[c])
                except (TypeError, ValueError):
                    continue
        return float("nan")

    def _maybe_percent(val: float) -> float:
        if math.isnan(val):
            return val
        return val * 100.0 if val <= 1.5 else val

    records = []
    for _, row in cmp.iterrows():
        run = row.get("run", "")
        if not run:
            continue
        acc = _pick_val(row, ["best_map", "final_map"])
        det = _pick_val(row, ["gpu0_det_rate", "cpu_det_rate"])
        speed = _pick_val(row, ["gpu0_fps", "cpu_fps"])
        if math.isnan(speed):
            avg_ms = _pick_val(row, ["gpu0_avg_ms", "cpu_avg_ms"])
            if not math.isnan(avg_ms) and avg_ms > 0:
                speed = 1000.0 / avg_ms
        acc = _maybe_percent(acc)
        det = _maybe_percent(det)
        records.append({"run": run, "accuracy": acc, "det_rate": det, "speed": speed})

    if not records:
        print("[skip] combined metrics plot: no comparable metrics available")
        return

    runs = [r["run"] for r in records]
    acc_vals = np.array([r["accuracy"] for r in records], dtype=float)
    det_vals = np.array([r["det_rate"] for r in records], dtype=float)
    speed_vals = np.array([r["speed"] for r in records], dtype=float)

    metrics = []
    if not np.isnan(acc_vals).all():
        metrics.append(("Accuracy (%)", acc_vals, "#1f77b4"))
    if not np.isnan(det_vals).all():
        metrics.append(("Detection Rate (%)", det_vals, "#ff7f0e"))
    if not np.isnan(speed_vals).all():
        metrics.append(("Speed (FPS)", speed_vals, "#2ca02c"))

    if not metrics:
        print("[skip] combined metrics plot: no comparable metrics available")
        return

    x = np.arange(len(runs))
    n_metrics = len(metrics)
    fig_width = max(7.0, 1.2 * len(runs))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    bar_width = 0.8 / n_metrics
    offsets = np.linspace(-bar_width * (n_metrics - 1) / 2, bar_width * (n_metrics - 1) / 2, n_metrics)

    for offset, (label, values, color) in zip(offsets, metrics):
        bars = ax.bar(x + offset, values, width=bar_width, label=label, color=color)
        for bx, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bx.get_x() + bx.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison overview")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / "cmp_overview.png"
    fig.savefig(out_path, dpi=CONFIG["SAVE_PNG_DPI"])
    plt.close(fig)
    print(f"[ok] wrote {out_path}")

# ------------------------------ main pipeline -------------------------------- #
def compare_models():
    runs_dir = Path(CONFIG["RUNS_DIR"])
    out_dir = Path(CONFIG["OUT_DIR"]); _ensure(out_dir)
    figs_dir = out_dir / "per_model"; _ensure(figs_dir)

    runs = _find_runs(runs_dir)
    if not runs:
        print(f"[warn] no runs found under {runs_dir}"); 
        return
    print("[info] runs:", ", ".join(r.name for r in runs))

    # Validation images for timing
    val_imgs = _gather_val_images()
    if not val_imgs:
        print("[warn] no validation images found; timing will be skipped.")
    else:
        random.Random(CONFIG["SEED"]).shuffle(val_imgs)
        val_imgs = val_imgs[:min(CONFIG["N_TIMING_IMAGES"], len(val_imgs))]
        print(f"[info] timing on {len(val_imgs)} images.")

    rows = []
    for r in runs:
        name = r.name
        csvp = r / "results.csv"
        try:
            df = pd.read_csv(csvp)
        except Exception as e:
            print(f"[skip] cannot read {csvp}: {e}"); 
            continue

        df = _clip(df.reset_index(drop=True))
        map_col = _pick_map(df)
        if not map_col:
            # helpful one-time columns print
            print(f"[cols] {name} ->", ", ".join(df.columns))

        # Per-model mAP curve
        if CONFIG["PLOTS"]["per_model_map"] and map_col and df[map_col].notna().any():
            fig = plt.figure(figsize=(6,4))
            plt.plot(df[map_col].values, label=map_col)
            plt.ylabel("mAP"); plt.xlabel("epoch"); plt.title(f"{name} – {map_col}")
            plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
            fig.tight_layout(); fp = figs_dir / f"{name}_map.png"
            fig.savefig(fp, dpi=CONFIG["SAVE_PNG_DPI"]); plt.close(fig)
            print(f"[ok] wrote {fp}")
        elif CONFIG["PLOTS"]["per_model_map"]:
            print(f"[skip] {name} accuracy curve: no valid column")

        # Per-model losses
        losses = _key_losses(df)
        if CONFIG["PLOTS"]["per_model_losses"] and losses:
            fig = plt.figure(figsize=(6,4))
            for c in losses:
                plt.plot(df[c].values, label=c, alpha=0.9)
            plt.ylabel("loss"); plt.xlabel("epoch"); plt.title(f"{name} – key losses")
            plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
            fig.tight_layout(); fp = figs_dir / f"{name}_losses.png"
            fig.savefig(fp, dpi=CONFIG["SAVE_PNG_DPI"]); plt.close(fig)
            print(f"[ok] wrote {fp}")

        # summary metrics
        final_map = float(df[map_col].dropna().iloc[-1]) if map_col and df[map_col].notna().any() else float("nan")
        best_map  = float(df[map_col].max()) if map_col and df[map_col].notna().any() else float("nan")

        w = _weights(r)
        size_mb = _size_mb(w) if w else float("nan")
        params_m = _param_count(w) if w else float("nan")

        # Timing per device in isolated subprocesses
        timing = {}
        if w and val_imgs:
            demo_dir = None
            if CONFIG["SAVE_DEMOS"]:
                demo_dir = out_dir / CONFIG["DEMOS_SUBDIR"] / name
                _ensure(demo_dir)
            for dev in CONFIG["DEVICES"]:
                try:
                    timing_dev = _subproc_time_model(w, val_imgs, dev, CONFIG["CONF"], CONFIG["IOU"])
                except Exception as e:
                    print(f"[timing skip] {name} on {dev}: {e}")
                    timing_dev = {k: float("nan") for k in ["avg_ms","med_ms","min_ms","max_ms","fps","det_rate","avg_dets","avg_conf"]}
                # prefix columns per device
                for k,v in timing_dev.items():
                    timing[f"{dev}_{k}"] = v

        # Collect row
        row = {
            "run": name, "results_csv": str(csvp), "epochs": len(df),
            "map_col": map_col, "final_map": final_map, "best_map": best_map,
            "weights": str(w) if w else "", "size_mb": size_mb, "params_m": params_m,
        }
        row.update(timing)
        rows.append(row)

    cmp = pd.DataFrame(rows)
    cmp_path = out_dir / "model_comparison.csv"
    cmp.to_csv(cmp_path, index=False)
    print(f"[ok] wrote {cmp_path}")

    # Bars: CPU vs GPU side-by-side if present
    # Accuracy (best mAP)
    if "best_map" in cmp.columns and pd.to_numeric(cmp["best_map"], errors="coerce").notna().any():
        _bar_if_any(out_dir, cmp, ["best_map"], "Accuracy (best mAP across runs)", "mAP", "cmp_best_map.png")

    # Speed (ms, FPS), Detection rate
    cpu_ms = "cpu_avg_ms";  gpu_ms = "gpu0_avg_ms"
    cpu_fps= "cpu_fps";     gpu_fps= "gpu0_fps"
    cpu_dr = "cpu_det_rate";gpu_dr = "gpu0_det_rate"

    if any(c in cmp.columns for c in [cpu_ms, gpu_ms]):
        _bar_if_any(out_dir, cmp, [c for c in [gpu_ms, cpu_ms] if c in cmp.columns],
                    "Speed (avg ms, lower is better)", "ms", "cmp_avg_ms_multi.png", invert=True)
    if any(c in cmp.columns for c in [cpu_fps, gpu_fps]):
        _bar_if_any(out_dir, cmp, [c for c in [gpu_fps, cpu_fps] if c in cmp.columns],
                    "Speed (FPS, higher is better)", "FPS", "cmp_fps_multi.png")
    if any(c in cmp.columns for c in [cpu_dr, gpu_dr]):
        _bar_if_any(out_dir, cmp, [c for c in [gpu_dr, cpu_dr] if c in cmp.columns],
                    "Detection rate (timed set)", "rate", "cmp_det_rate_multi.png")

    _combined_metrics_plot(out_dir, cmp)

    # Size / Params
    if "size_mb" in cmp.columns:
        _bar_if_any(out_dir, cmp, ["size_mb"], "Model size (MB)", "MB", "cmp_size_mb.png", invert=True)
    if "params_m" in cmp.columns:
        _bar_if_any(out_dir, cmp, ["params_m"], "Model parameters (Millions)", "M", "cmp_params_m.png", invert=True)

    print("[done] comparison complete.")

if __name__ == "__main__":
    # If SLURM gave us GPUs but someone set -1, restore visibility before imports that might read it.
    if os.environ.get("CUDA_VISIBLE_DEVICES", "-1") == "-1":
        for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
            if os.environ.get(k):
                os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[k]
                break
    compare_models()
