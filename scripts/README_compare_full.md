# Compare YOLO Pose Runs (Full)

This script scans your Ultralytics runs (each with a `results.csv`) and produces:

- `summary.csv` with final & best mAP, some losses, and epochs
- `map_curve.png` overlay of validation mAP for all runs
- `loss_curves_*.png` overlays of train/val losses (grouped to avoid clutter)
- optional `per_run/<run>_curves.png` detailed curves per run

## Usage
1. Put the script here:
```
scripts/compare_runs_full.py
```
2. Edit the CONFIG block at the top.
3. Run:
```bash
python scripts/compare_runs_full.py
```
