#!/usr/bin/env python3
"""
tools/plot_arena_arrows_plotly.py

Plot trajectory as **arrows** over an arena-like grid using Plotly.
- Arrow direction is derived from heading (unit vector) if present; else from frame-to-frame delta.
- Arrow tip represents "nose" direction.
- Color encodes time (ts) or frame index.
- Saves interactive HTML.

Usage:
  python tools/plot_arena_arrows_plotly.py \
    --csv traj_out/trajectory_enriched.csv \
    --unit px \
    --arrow-len 30 \
    --arena-w 1920 --arena-h 1080 \
    --grid-step 120 \
    --color-by ts \
    --out arrows.html
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def build_grid_shapes(w, h, step):
    shapes = []
    if step and step > 0:
        # verticals
        for x in range(0, int(w)+1, int(step)):
            shapes.append(dict(type="line", x0=x, y0=0, x1=x, y1=h, line=dict(color="rgba(0,0,0,0.15)", width=1)))
        # horizontals
        for y in range(0, int(h)+1, int(step)):
            shapes.append(dict(type="line", x0=0, y0=y, x1=w, y1=y, line=dict(color="rgba(0,0,0,0.15)", width=1)))
    return shapes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="trajectory_enriched.csv (or detections.csv)")
    ap.add_argument("--out", default="arrows.html", help="output HTML path")
    ap.add_argument("--unit", choices=["px","cm"], default="px", help="units for arrow_len and arena size")
    ap.add_argument("--arrow-len", type=float, default=30.0, help="arrow length in chosen units")
    ap.add_argument("--arena-w", type=float, default=1920, help="arena width in chosen units")
    ap.add_argument("--arena-h", type=float, default=1080, help="arena height in chosen units")
    ap.add_argument("--grid-step", type=float, default=120, help="grid spacing in chosen units")
    ap.add_argument("--color-by", choices=["ts","frame_idx"], default="ts", help="color encoding")
    ap.add_argument("--stride", type=int, default=1, help="plot every Nth sample to reduce clutter")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # if enriched file exists, we expect heading_x/heading_y
    if "heading_x" not in df.columns or "heading_y" not in df.columns:
        # derive heading from deltas
        if "cx" in df.columns and "cy" in df.columns:
            dx = np.diff(df["cx"], prepend=df["cx"].iloc[0])
            dy = np.diff(df["cy"], prepend=df["cy"].iloc[0])
            mag = np.hypot(dx, dy); mag[mag==0] = 1
            df["heading_x"] = dx / mag
            df["heading_y"] = dy / mag
        else:
            raise SystemExit("CSV must include heading_x, heading_y or at least cx, cy to derive heading.")

    # choose base position fields
    if args.unit == "cm" and {"cx","cy","dist_cm"}.issubset(df.columns):
        # we still only have cx,cy in px; convert positions by inferring px_per_cm if available
        # prefer inferred px_per_cm if stored by our reconstructor
        if "px_per_cm" in df.attrs:
            px_per_cm = df.attrs["px_per_cm"]
        else:
            # try to infer from columns dist_px/dist_cm median
            if "dist_px" in df.columns and "dist_cm" in df.columns:
                sub = df[~df["dist_cm"].isna() & np.isfinite(df["dist_cm"]) & np.isfinite(df["dist_px"]) & (df["dist_cm"]!=0)]
                px_per_cm = (sub["dist_px"]/sub["dist_cm"]).median() if len(sub)>5 else None
            else:
                px_per_cm = None
        if px_per_cm and np.isfinite(px_per_cm) and px_per_cm>0:
            X = df["cx"].to_numpy() / px_per_cm
            Y = df["cy"].to_numpy() / px_per_cm
            scale = args.arrow_len
        else:
            # fall back to px
            X = df["cx"].to_numpy()
            Y = df["cy"].to_numpy()
            args.unit = "px"
            scale = args.arrow_len
    else:
        X = df["cx"].to_numpy()
        Y = df["cy"].to_numpy()
        scale = args.arrow_len

    HX = df["heading_x"].to_numpy()
    HY = df["heading_y"].to_numpy()

    # arrow tips (nose)
    X2 = X + HX * scale
    Y2 = Y + HY * scale

    # stride to declutter
    idx = np.arange(len(X))[::max(1, int(args.stride))]

    # color by time or frame
    if args.color_by in df.columns:
        C = df.loc[idx, args.color_by].to_numpy()
    else:
        C = np.arange(len(idx))

    # Build traces: a) segments as one scatter (with None separators), b) nose markers colored by time
    x_line = []
    y_line = []
    for i in idx:
        x_line += [X[i], X2[i], None]
        y_line += [Y[i], Y2[i], None]

    fig = go.Figure()

    # arena grid
    fig.update_layout(
        width=900, height=520,
        xaxis=dict(range=[0, args.arena_w], title=f"x ({args.unit})", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[args.arena_h, 0], title=f"y ({args.unit})"),  # image coordinates: y downwards
        shapes=build_grid_shapes(args.arena_w, args.arena_h, args.grid_step),
        template="plotly_white",
        margin=dict(l=60, r=10, t=40, b=60),
        title="Trajectory as arrows (nose at tip; color=time)",
    )

    # lines (uniform color)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="rgba(0,0,0,0.35)", width=2),
        name="heading arrows"
    ))

    # tips colored by time
    fig.add_trace(go.Scatter(
        x=X2[idx], y=Y2[idx], mode="markers",
        marker=dict(size=6, color=C, colorscale="Viridis", showscale=True, colorbar=dict(title=args.color_by)),
        name="nose tip"
    ))

    # optional: start point
    fig.add_trace(go.Scatter(
        x=[X[idx[0]]] if len(idx)>0 else [], y=[Y[idx[0]]] if len(idx)>0 else [], mode="markers",
        marker=dict(size=9, symbol="x"),
        name="start"
    ))

    out = Path(args.out)
    fig.write_html(out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
