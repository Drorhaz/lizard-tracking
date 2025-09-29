#!/usr/bin/env python3
from __future__ import annotations
import time, math, random
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
    _HAVE_ULTRA = True
except Exception:
    _HAVE_ULTRA = False

CONFIG: Dict[str, Any] = {
    "RUNS_DIR": "runs/pose",
    "OUT_DIR": "output/compare_models",
    "INCLUDE": [],
    "EXCLUDE": ["predict", "traincache"],
    "MAX_EPOCHS": None,
    "SAVE_PNG_DPI": 170,
    "PLOTS": {
        "per_model_map": True,
        "per_model_losses": True,
        "key_losses": ["train/kpt_loss","val/kpt_loss","train/box_loss","val/box_loss"],
    },
    # timing
    "DATA_YAML": "data/pogona_head_pose.yaml", 
    "VAL_IMAGES_DIR": "dataset/images/val",        
    "N_TIMING_IMAGES": 40,
    "DEVICE": "cpu", # or "0" for GPU 0
    "CONF": 0.25,
    "IOU": 0.45,
    "SEED": 0,
    "SAVE_DEMOS": False,
    "DEMOS_SUBDIR": "inference_demos",
}

MAP_CANDS = ["metrics/pose/mAP50-95","keypoints/mAP50-95","metrics/mAP50-95","mAP50-95","map50-95",
             "metrics/box/mAP50-95","box/mAP50-95"]
LOSS_CANDS = ["train/box_loss","val/box_loss","train/kpt_loss","val/kpt_loss",
              "train/cls_loss","val/cls_loss","train/dfl_loss","val/dfl_loss",
              "train/kobj_loss","val/kobj_loss","train/pose_loss","val/pose_loss"]

def _ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def _find_runs(root: Path) -> List[Path]:
    out = []
    for p in sorted(root.glob("*")):
        if p.is_dir() and (p / "results.csv").exists():
            s = str(p)
            if CONFIG["INCLUDE"] and not any(t in s for t in CONFIG["INCLUDE"]): continue
            if CONFIG["EXCLUDE"] and any(t in s for t in CONFIG["EXCLUDE"]): continue
            out.append(p)
    return out

def _clip(df: pd.DataFrame) -> pd.DataFrame:
    m = CONFIG["MAX_EPOCHS"]; 
    if m is None: return df
    return df[df["epoch"] < m] if "epoch" in df.columns else df.iloc[:m, :]

def _pick_map(df: pd.DataFrame) -> Optional[str]:
    for c in MAP_CANDS:
        if c in df.columns: return c
    return None

def _key_losses(df: pd.DataFrame) -> List[str]:
    ks = [k for k in CONFIG["PLOTS"]["key_losses"] if k in df.columns]
    if not ks: ks = [k for k in LOSS_CANDS if k in df.columns]
    return ks[:3]

def _weights(run: Path) -> Optional[Path]:
    for f in ["best.pt","last.pt"]:
        p = run / "weights" / f
        if p.exists(): return p
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

def _param_count(m: YOLO) -> float:
    try:
        return sum(p.numel() for p in m.model.parameters()) / 1e6
    except Exception:
        return float("nan")

def _size_mb(p: Path) -> float:
    try: return p.stat().st_size / (1024*1024)
    except Exception: return float("nan")

def _time_model(weights: Path, images: List[Path], device, conf, iou, demo_dir: Optional[Path]) -> Dict[str, float]:
    model = YOLO(str(weights))
    # warmup
    _ = model.predict(source=str(images[0]), device=device, conf=conf, iou=iou, verbose=False, save=False)
    times, dets, confs = [], [], []
    for p in images:
        t0 = time.perf_counter()
        res = model.predict(source=str(p), device=device, conf=conf, iou=iou, verbose=False,
                            save=bool(demo_dir), project=str(demo_dir) if demo_dir else None,
                            name="", exist_ok=True)
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        n = 0; cs = []
        if len(res):
            r = res[0]
            if getattr(r, "boxes", None) is not None:
                try:
                    n = int(r.boxes.shape[0])
                    cs = r.boxes.conf.detach().cpu().numpy().tolist()
                except Exception: pass
        dets.append(n)
        confs.append(sum(cs)/len(cs) if cs else float("nan"))
    s_times = pd.Series(times)
    fps = 1000.0/s_times.mean() if len(s_times) else float("nan")
    det_rate = float((pd.Series(dets) > 0).mean()) if len(dets) else float("nan")
    avg_conf = float(pd.Series([x for x in confs if not math.isnan(x)]).mean()) if any(not math.isnan(x) for x in confs) else float("nan")
    return {"avg_ms": float(s_times.mean()), "med_ms": float(s_times.median()),
            "min_ms": float(s_times.min()), "max_ms": float(s_times.max()),
            "fps": fps, "det_rate": det_rate,
            "avg_dets": float(pd.Series(dets).mean()) if len(dets) else float("nan"),
            "avg_conf": avg_conf}

def _bar_if_any(out_dir: Path, df: pd.DataFrame, col: str, title: str, xlabel: str, fname: str, invert=False):
    if col not in df.columns: 
        print(f"[skip] bar '{title}': column '{col}' missing")
        return
    d = df[["run", col]].dropna()
    d = d[~pd.isna(d[col]) & pd.to_numeric(d[col], errors="coerce").notna()]
    if d.empty:
        print(f"[skip] bar '{title}': no finite values")
        return
    d = d.sort_values(col, ascending=invert)
    fig = plt.figure(figsize=(8,5))
    plt.barh(d["run"], d[col])
    plt.title(title); plt.xlabel(xlabel)
    for i,v in enumerate(d[col].values):
        try: plt.text(v, i, f" {float(v):.3f}")
        except Exception: pass
    fig.tight_layout(); fig.savefig(out_dir / fname, dpi=CONFIG["SAVE_PNG_DPI"]); plt.close(fig)
    print(f"[ok] wrote {out_dir/fname}")

def compare_models():
    runs_dir = Path(CONFIG["RUNS_DIR"])
    out_dir = Path(CONFIG["OUT_DIR"]); _ensure(out_dir)
    figs_dir = out_dir / "per_model"; _ensure(figs_dir)

    runs = _find_runs(runs_dir)
    if not runs:
        print(f"[warn] no runs found under {runs_dir}"); return
    print("[info] runs:", ", ".join(r.name for r in runs))

    # validation images (for timing)
    val_imgs = _gather_val_images()
    if not val_imgs:
        print("[warn] no validation images found; timing will be skipped (speed/fps NaN).")
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
        losses = _key_losses(df)

        # per-model: mAP
        if CONFIG["PLOTS"]["per_model_map"]:
            ok = bool(map_col and df[map_col].notna().any())
            if ok:
                fig = plt.figure(figsize=(6,4))
                plt.plot(df[map_col].values, label=map_col)
                plt.ylabel("mAP"); plt.xlabel("epoch"); plt.title(f"{name} – {map_col}")
                plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
                fig.tight_layout(); fp = figs_dir / f"{name}_map.png"
                fig.savefig(fp, dpi=CONFIG["SAVE_PNG_DPI"]); plt.close(fig)
                print(f"[ok] wrote {fp}")
            else:
                print(f"[skip] {name} mAP curve: no valid mAP column")

        # per-model: losses
        if CONFIG["PLOTS"]["per_model_losses"]:
            if losses:
                fig = plt.figure(figsize=(6,4))
                for c in losses:
                    plt.plot(df[c].values, label=c, alpha=0.9)
                plt.ylabel("loss"); plt.xlabel("epoch"); plt.title(f"{name} – key losses")
                plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
                fig.tight_layout(); fp = figs_dir / f"{name}_losses.png"
                fig.savefig(fp, dpi=CONFIG["SAVE_PNG_DPI"]); plt.close(fig)
                print(f"[ok] wrote {fp}")
            else:
                print(f"[skip] {name} losses: none found")

        # summary metrics
        final_map = float(df[map_col].dropna().iloc[-1]) if map_col and df[map_col].notna().any() else float("nan")
        best_map  = float(df[map_col].max()) if map_col and df[map_col].notna().any() else float("nan")

        w = _weights(r)
        size_mb = _size_mb(w) if w else float("nan")
        params_m = float("nan")
        timing = {k: float("nan") for k in ["avg_ms","med_ms","min_ms","max_ms","fps","det_rate","avg_dets","avg_conf"]}

        if _HAVE_ULTRA and w:
            try:
                params_m = _param_count(YOLO(str(w)))
            except Exception:
                pass
            if val_imgs:
                demo_dir = None
                if CONFIG["SAVE_DEMOS"]:
                    demo_dir = out_dir / CONFIG["DEMOS_SUBDIR"] / name
                    _ensure(demo_dir)
                try:
                    timing = _time_model(w, val_imgs, CONFIG["DEVICE"], CONFIG["CONF"], CONFIG["IOU"], demo_dir)
                except Exception as e:
                    print(f"[timing skip] {name}: {e}")

        rows.append({
            "run": name, "results_csv": str(csvp), "epochs": len(df),
            "map_col": map_col, "final_map": final_map, "best_map": best_map,
            "weights": str(w) if w else "", "size_mb": size_mb, "params_m": params_m,
            **timing
        })

    cmp = pd.DataFrame(rows)
    cmp_path = out_dir / "model_comparison.csv"
    cmp.to_csv(cmp_path, index=False)
    print(f"[ok] wrote {cmp_path}")

    # bars (only if we have finite values)
    _bar_if_any(out_dir, cmp, "best_map", "Accuracy (best mAP)", "mAP", "cmp_best_map.png")
    _bar_if_any(out_dir, cmp, "avg_ms",   "Speed (avg inference ms, lower is better)", "ms", "cmp_avg_ms.png", invert=True)
    _bar_if_any(out_dir, cmp, "fps",      "Speed (FPS, higher is better)", "FPS", "cmp_fps.png")
    _bar_if_any(out_dir, cmp, "size_mb",  "Model size (MB)", "MB", "cmp_size_mb.png", invert=True)
    _bar_if_any(out_dir, cmp, "params_m", "Model parameters", "Millions", "cmp_params_m.png", invert=True)
    _bar_if_any(out_dir, cmp, "det_rate", "Detection rate on timed set", "rate", "cmp_det_rate.png")

    print("[done] comparison complete.")
if __name__ == "__main__":
    compare_models()