# Real-Time Lizard Tracking — Head Detection & Behavioral Analysis

This repo contains the code and assets for **real-time lizard head detection and behavioral analysis** using YOLO (v11 small) with integrated trajectory analysis and activity detection capabilities.

## 🏆 **Current Results**
- **Model**: YOLOv11s (9.4M parameters)
- **Best Performance**: **98.3% mAP@0.5** (CPU training, 3 epochs)
- **Dataset**: 2,974 labeled images (thermal + visible)
- **Behavioral Analysis**: ✅ Real-time behavioral analysis and trajectory reconstruction
- **Libraries**: ✅ Modular `lib/lizard_tracking` and `lib/behavioral_analysis` 
- **Web Interface**: ✅ Interactive pose-head pipeline with video discovery
- **Status**: ✅ Dataset validated, ready for production CUDA training

## 🚨 **CRITICAL: MPS Training Issue**

**DO NOT use MPS (Apple GPU) for YOLO training on macOS!**

We discovered a PyTorch MPS backend bug that causes **silent training failure**:
- ❌ Training appears successful but **all losses stay at 0**
- ❌ **All metrics remain at 0** (mAP, Recall, Precision)  
- ❌ **No learning occurs** despite completing epochs
- ❌ Model files are created but **contain no learned features**

### ✅ **Confirmed Solutions**
- **Development**: Use `device='cpu'` (slower but works perfectly)
- **Production**: Use CUDA GPU (Google Colab, Kaggle) 
- **Always run sanity check first**: `python3 train_sanity.py`

### 📊 **Training Results Comparison**
| Training Run | Device | Epochs | Duration | mAP@0.5 | Status |
|-------------|--------|--------|----------|---------|--------|
| MPS Training | MPS (M1) | 10 | 43 min | **0.000** | ❌ Failed (backend bug) |
| CPU Sanity | CPU (M1) | 3 | 50 min | **0.983** | ✅ **Success** |
| Production | CUDA | TBD | TBD | TBD | 🔄 Planned |

---

## Quick Start

### 0) Environment Setup
**IMPORTANT**: Use the correct conda environment for all operations:
```bash
# Activate the LizardPose environment
conda activate /scratch200/bareketd1/LizardPose

# Or set up a new environment (if needed)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Prepare the dataset (YOLO format)
```
dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
```
> The `dataset/` folder is gitignored. Mount or link your data where appropriate.

### 2) Train
Use the same entry point we run on the cluster (defaults mirror `scripts/train_pose.sh`):
```bash
python scripts/pogona_pipeline_cfg.py \
  --data data/pogona_head_pose.yaml \
  --model yolo11s-pose.pt \
  --epochs 150 \
  --device 0
```
This script is a thin wrapper around `ultralytics.YOLO`. It honours any environment variables you export (`BATCH`, `IMGSZ`, `LR0`, …) and writes artefacts to `runs/pose/<run_name>/`.

### 3) Validate / inspect metrics
The quickest sanity check is to re-run validation on the best checkpoint:
```bash
python scripts/pogona_pipeline_cfg.py --data data/pogona_head_pose.yaml --model runs/pose/pogona_head_pose2/weights/best.pt --epochs 1 --batch 16 --device 0 --extra '{"task":"pose"}'
```
Every training run also drops a `results.csv`. You can plot the losses and validation curves with:
```bash
python tools/analyze_pose_runs.py plot runs/pose/pogona_head_pose2 --out plots/pogona_head_pose2.png
```

### 4) Real-time Behavioral Analysis
Use the integrated behavioral analysis library for advanced tracking:
```bash
# Real-time behavioral analysis with trajectory reconstruction
python lib/behavioral_analysis/detector.py

# Web interface for pose detection and behavioral analysis
cd pose-head/pipeline && python web_interface.py
```

### 5) Offline video tracking
```bash
python lib/lizard_tracking/api.py --track \
  --video videos/sample.mp4 \
  --weights runs/pose/pogona_head_pose2/weights/best.pt
```
This produces a CSV (and optional parquet) plus an overlay video under `output/`.

### 6) Live demo / UI smoke-test
```bash
python -m lizard_tracking.ui.stream --source 0 --weights runs/pose/pogona_head_pose2/weights/best.pt
```
Press `q` to exit the OpenCV preview window. Activity events are emitted to stdout.

---

## Repository Map
- **configs/** – Dataset definitions and end-to-end pipeline YAML bundles
- **scripts/** – Training scripts and HPC job wrappers
- **lib/lizard_tracking/** – Core pose detection library (training pipelines, models, configs)
- **lib/behavioral_analysis/** – Behavioral analysis library (trajectory reconstruction, activity detection)
- **pose-head/pipeline/** – Web interface for pose detection and behavioral analysis
- **tests/** – Smoke tests for inference and trajectory logging
- **output/**, **runs/** – Generated artefacts (ignored in Git)

### Key scripts
- `scripts/pogona_pipeline_cfg.py` – single-run training/validation harness.
- `scripts/pogona_pipeline_cfg_optuna.py` – Optuna hyper-parameter sweeps (writes `optuna_###` runs).
- `scripts/train_pose.sh` / `run_train_gpu.sh` – shell wrappers used on SLURM.
- `lib/behavioral_analysis/trajectory.py` – trajectory reconstruction and analysis.
- `lib/behavioral_analysis/detector.py` – real-time behavioral pattern detection.
- `pose-head/pipeline/web_interface.py` – web-based pose detection interface.
- `tools/analyze_pose_runs.py` – compares runs and plots loss/metric curves.
- `webapps/label_qc_web.py` – browser label editor (run `python webapps/label_qc_web.py`).

## Training vs validation metrics

Every YOLO run writes a `results.csv` and per-epoch images inside `runs/pose/<run_name>/`. The CSV columns follow Ultralytics’ naming:

- `train/*` columns capture training losses (box/kpt). Watch them trend down over epochs.
- `metrics/pose/*` columns are validation metrics (Precision `P`, Recall `R`, `mAP50`, `mAP50-95`).
- `val_batch*_labels/pred.jpg` offer quick qualitative snapshots of GT vs predictions.

Use the helper script to inspect them quickly:
```bash
# Compare all runs under runs/pose
python tools/analyze_pose_runs.py compare runs/pose/*

# Plot losses + validation curves for a particular run
python tools/analyze_pose_runs.py plot runs/pose/pogona_head_pose2 --out plots/pogona_head_pose2.png
```
When run without arguments it reads the config embedded at the top of the script, compares the configured runs, and saves everything under `output/analytics/` (CSV + PNGs). The helper also drops a bar chart highlighting the best `mAP@0.5:0.95`.

### 7) Live tracking + trajectory logging
Once you have a trained checkpoint copied to `output/models/head_pose/best.pt`, stream a video or camera feed and archive detections:
```bash
# Using the new behavioral analysis library
python lib/behavioral_analysis/detector.py

# Or using the legacy stream tool
python tools/run_pose_stream.py
```
Edit the config block at the top of the script to point at your video/camera and to adjust the logging cadence (`frame_stride`, `save_every`). Each run writes to `output/detections/<name>_<timestamp>/` with:
- `trajectory.csv` containing frame-by-frame positions/yaw plus the detected activity event
- `overlay.mp4` (optional) for a quick preview
- `labeled_frames/` with raw frames, overlays, and YOLO pose labels for retraining
Events (advance/retreat/stop) are echoed to stdout in real time.

### 8) Behavioral Analysis & Trajectory Reconstruction
Use the dedicated behavioral analysis library:
```bash
# Reconstruct trajectories from pose data
python lib/behavioral_analysis/trajectory.py

# Real-time behavioral pattern detection
python lib/behavioral_analysis/detector.py

# Web interface with integrated behavioral analysis
cd pose-head/pipeline && python web_interface.py
```

## Model selection workflow
1. Launch `scripts/pogona_pipeline_cfg_optuna.py` to explore learning-rate/weight-decay/imgsz combinations. Finished runs appear as `runs/pose/optuna_###`.
2. Review `tools/analyze_pose_runs.py compare runs/pose/optuna_*` to shortlist candidates.
3. Promote the best trial by copying its `weights/best.pt` into `runs/pose/pogona_head_poseX/` (or re-run with the chosen hyper-parameters).

---

## 🧪 **Inference Performance Testing**

**✅ BASELINE MODEL VALIDATED FOR PRODUCTION USE**

| Test Category | Result | Grade |
|---------------|--------|-------|
| **Speed** | 62.9ms avg (15.9 fps) | 🚀 **Excellent** |
| **Detection Rate** | 75% (15/20 images) | ⚠️ **Moderate** |
| **Confidence** | 68.6% average | 👍 **Good** |
| **Overall Assessment** | Production Ready | **B+ Grade** |

**📁 Testing Resources:**
- **[Inference Tests](tests/inference/)** - Complete performance testing suite
- **[Test Results](tests/inference/RESULTS.md)** - Detailed metrics and analysis
- **[Test Script](tests/inference/test_inference.py)** - Reusable performance testing
- **[Visual Demos](tests/inference/inference_demo/)** - Annotated detection examples

**🚀 Quick Test:**
```bash
cd tests/inference
python3 test_inference.py  # Run full performance test
```

## 📊 **Training Documentation & Results**

Detailed training documentation and results are available:
- **[Training Log](docs/training_log.md)** - Complete training history and analysis
- **[Training Results Summary](TRAINING_RESULTS.md)** - Performance comparison table
- **[Experiments](experiments/)** - Organized results by training run:
  - `sanity_check_cpu/` - Successful CPU training (98.3% mAP)
  - `mps_failed/` - Failed MPS training documentation  
  - `production/` - Future CUDA training results

## Known issues
- **MPS training on Mac:** Confirmed silent failure → always run `train_sanity.py` on CPU first.
- **Environment:** Use `/scratch200/bareketd1/LizardPose` conda environment for all operations.
- **Imports:** Fixed lizard_tracking module imports in scripts (config and pipelines).
- **Large files:** Don't commit datasets or weights (`*.pt`); use Git LFS if needed.

## 🧬 **Behavioral Analysis Features**

### Libraries
- **`lib/lizard_tracking/`** - Core pose detection and YOLO training pipelines
- **`lib/behavioral_analysis/`** - Dedicated behavioral analysis library with:
  - Real-time activity detection (advance/retreat/stop patterns)
  - Trajectory reconstruction from pose keypoints  
  - Event-driven behavioral metrics
  - Export capabilities for further analysis

### Web Interface
- **`pose-head/pipeline/web_interface.py`** - Interactive web interface featuring:
  - Automatic video discovery across project directories
  - Real-time pose detection with behavioral overlay
  - SLURM job submission for batch processing
  - Live streaming with activity event detection

---

## License
Add your chosen license here.
