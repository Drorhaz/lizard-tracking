#!/usr/bin/env python3
"""
tools/reconstruct_trajectory.py

Reconstructs trajectory from a detections.csv created by pose_head_pipeline.py,
computes basic kinematics, **heading** (unit vector) and saves an enriched CSV.
Also generates simple Matplotlib plots.

Usage:
  python tools/reconstruct_trajectory.py --csv /path/to/detections.csv --out-dir out_traj --smooth 5
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    by = "frame_idx" if "frame_idx" in df.columns else "ts"
    df = df.sort_values(by=[by]).reset_index(drop=True)
    return df

def infer_px_per_cm(df: pd.DataFrame) -> float | None:
    if "dist_px" in df.columns and "dist_cm" in df.columns:
        sub = df[~df["dist_cm"].isna() & np.isfinite(df["dist_cm"]) & np.isfinite(df["dist_px"]) & (df["dist_cm"] != 0)]
        if len(sub) >= 5:
            ratio = (sub["dist_px"] / sub["dist_cm"]).median()
            if np.isfinite(ratio) and ratio > 0:
                return float(ratio)
    return None

def smooth_series(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    kern = np.ones(win) / win
    xpad = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xpad, kern, mode="valid")

def compute_kinematics(df: pd.DataFrame, smooth: int = 1) -> pd.DataFrame:
    if "ts" not in df.columns:
        raise SystemExit("CSV must include a 'ts' column (seconds).")
    t = df["ts"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.nan  # avoid div-by-zero

    cx = df["cx"].to_numpy(dtype=float)
    cy = df["cy"].to_numpy(dtype=float)

    # raw deltas
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])

    # planar speed in px/s
    speed_px = np.hypot(dx, dy) / dt
    speed_px[~np.isfinite(speed_px)] = np.nan

    # heading unit vector (derived from dx,dy)
    mag = np.hypot(dx, dy)
    hx = np.divide(dx, mag, out=np.zeros_like(dx), where=mag>0)
    hy = np.divide(dy, mag, out=np.zeros_like(dy), where=mag>0)

    dist_px = df["dist_px"].to_numpy(dtype=float) if "dist_px" in df.columns else np.full_like(cx, np.nan, dtype=float)
    dist_cm = df["dist_cm"].to_numpy(dtype=float) if "dist_cm" in df.columns else np.full_like(cx, np.nan, dtype=float)

    v_norm_px = np.diff(dist_px, prepend=dist_px[0]) / dt
    v_norm_cm = np.diff(dist_cm, prepend=dist_cm[0]) / dt

    # smoothing (optional) for kinematics (not positions)
    if smooth and smooth > 1:
        speed_px_s = smooth_series(speed_px, smooth)
        v_norm_px_s = smooth_series(v_norm_px, smooth)
        v_norm_cm_s = smooth_series(v_norm_cm, smooth)
        # also lightly smooth heading to reduce jitter
        hx_s = smooth_series(hx, smooth)
        hy_s = smooth_series(hy, smooth)
        hmag = np.hypot(hx_s, hy_s); hmag[hmag==0] = 1.0
        hx_s, hy_s = hx_s / hmag, hy_s / hmag
    else:
        speed_px_s, v_norm_px_s, v_norm_cm_s = speed_px, v_norm_px, v_norm_cm
        hx_s, hy_s = hx, hy

    out = df.copy()
    out["dt_s"] = dt
    out["dx_px"] = dx
    out["dy_px"] = dy
    out["speed_xy_px_s"] = speed_px_s
    out["v_normal_px_s"] = v_norm_px_s
    out["v_normal_cm_s"] = v_norm_cm_s
    out["heading_x"] = hx_s  # unit vector
    out["heading_y"] = hy_s  # unit vector

    # try to infer px_per_cm
    px_per_cm = infer_px_per_cm(df)
    out.attrs["px_per_cm_inferred"] = px_per_cm
    if px_per_cm is not None:
        out["speed_xy_cm_s"] = out["speed_xy_px_s"] / px_per_cm
    else:
        out["speed_xy_cm_s"] = np.nan

    return out

def plot_trajectory(df: pd.DataFrame, out_path: Path):
    plt.figure()
    plt.plot(df["cx"], df["cy"])
    plt.gca().invert_yaxis()
    plt.title("Trajectory (cx, cy)")
    plt.xlabel("cx (px)")
    plt.ylabel("cy (px)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_distance(df: pd.DataFrame, out_path: Path):
    plt.figure()
    if "dist_px" in df.columns:
        plt.plot(df["dist_px"], label="dist_px")
    if "dist_cm" in df.columns:
        plt.plot(df["dist_cm"], label="dist_cm")
    plt.title("Distance to screen over time")
    plt.xlabel("frame")
    plt.ylabel("distance (px/cm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_speeds(df: pd.DataFrame, out_path: Path):
    plt.figure()
    plt.plot(df["speed_xy_px_s"], label="speed_xy_px_s")
    if "speed_xy_cm_s" in df.columns:
        plt.plot(df["speed_xy_cm_s"], label="speed_xy_cm_s")
    if "v_normal_px_s" in df.columns:
        plt.plot(df["v_normal_px_s"], label="v_normal_px_s")
    if "v_normal_cm_s" in df.columns:
        plt.plot(df["v_normal_cm_s"], label="v_normal_cm_s")
    plt.title("Speeds over time")
    plt.xlabel("frame")
    plt.ylabel("speed (px/s or cm/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="detections.csv from pose_head_pipeline.py")
    ap.add_argument("--out-dir", default="traj_out", help="output directory")
    ap.add_argument("--smooth", type=int, default=5, help="odd window for smoothing (>=1)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    out = compute_kinematics(df, smooth=args.smooth)

    enriched_csv = out_dir / "trajectory_enriched.csv"
    out.to_csv(enriched_csv, index=False)

    plot_trajectory(out, out_dir / "trajectory.png")
    plot_distance(out, out_dir / "distance.png")
    plot_speeds(out, out_dir / "speeds.png")

    px_per_cm = out.attrs.get("px_per_cm_inferred", None)
    with open(out_dir / "README.txt", "w") as f:
        f.write("Trajectory reconstruction outputs\n")
        if px_per_cm:
            f.write(f"Inferred px_per_cm (median from dist fields): {px_per_cm:.3f}\n")
        else:
            f.write("Could not infer px_per_cm (dist_cm likely NaN).\n")
        f.write("New columns include heading_x, heading_y (unit vectors).\n")
        f.write("Files:\n- trajectory_enriched.csv\n- trajectory.png\n- distance.png\n- speeds.png\n")

    print(f"Wrote: {enriched_csv}")
    print("Done.")

if __name__ == "__main__":
    main()
