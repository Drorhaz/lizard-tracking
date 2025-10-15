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

## 🔄 **Semi-Automatic Labeling Pipeline**

This project includes a complete **semi-automatic pose labeling pipeline** that combines automated model training, intelligent sample routing, and an interactive web-based labeler for efficient dataset creation. The pipeline operates in a continuous cycle: Train → Infer → Route → Review → Promote → Repeat.

**📖 Full Documentation**: See [`autogenerate/README.md`](autogenerate/README.md) for complete pipeline usage instructions.

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
conda activate LizardPose

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
Inference & FPS check
After training, run inference + latency profiling:


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

Every YOLO run writes a `results.csv` and per-epoch images inside `runs/pose/<run_name>/`. The CSV columns follow Ultralytics' naming:

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
- **Environment:** Use `LizardPose` conda environment for all operations.
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

## 📁 Directory Structure

```
dataset/
├── pose-seed/              # High-quality labeled training data
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── val/            # Validation images
│   └── labels/             # YOLO pose format labels
│       ├── train/
│       └── val/
├── head-detection-dataset/ # Source unlabeled images
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/            # (bbox labels, not used for pose)
└── review_queue/          # Low-confidence predictions for manual review
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
pip install ultralytics flask opencv-python numpy
```

### 1. Prepare Initial Seed Data

Create initial manually labeled samples in `dataset/pose-seed/`. You need at least 20-30 images with pose labels to bootstrap the process.

**Label format** (YOLO pose): Each `.txt` file contains one line per detection:
```
class_id center_x center_y width height kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v kpt3_x kpt3_y kpt3_v
```

Where:
- `class_id`: Always `0` (lizard_head)
- `center_x, center_y, width, height`: Bounding box (normalized 0-1)
- `kpt1_x, kpt1_y, kpt1_v`: Nose keypoint (x, y, visibility)
- `kpt2_x, kpt2_y, kpt2_v`: Left ear keypoint
- `kpt3_x, kpt3_y, kpt3_v`: Right ear keypoint
- Visibility: `0`=not labeled, `1`=occluded, `2`=visible

### 2. Run the Semi-Automatic Pipeline

```bash
cd autogenerate
python semi_auto_pose.py
```

**What happens:**
- Trains YOLO model on current seed data (20 epochs)
- Runs inference on all unlabeled images
- Routes predictions by confidence:
  - High confidence (box ≥ 0.8, all keypoints ≥ 0.7) → `pose-seed`
  - Low confidence (box < 0.4 or any keypoint < 0.3) → `review_queue`
  - Middle range → Skipped (not useful for training)
- Updates `autogenerate/best.pt` with latest trained model

**Configuration** (edit `semi_auto_pose.py`):
```python
CONFIG = PipelineConfig(
    epochs=20,                    # Training epochs per iteration
    high_conf_box=0.8,           # Box confidence threshold for auto-acceptance
    high_conf_kpt=0.7,           # Keypoint confidence threshold for auto-acceptance
    low_conf_box=0.4,            # Box confidence threshold for review
    low_conf_kpt=0.3,            # Keypoint confidence threshold for review
    # ... other settings
)
```

### 3. Review and Correct Low-Confidence Predictions

Use the web-based labeler to fix predictions that need manual review:

```bash
cd labeler
python label_qc_web.py
```

Then open http://localhost:5000 in your browser.

**Loading Review Queue:**
1. Click "📁 Change Path" 
2. Navigate to `dataset/review_queue/images/train` (or `val`)
3. Click "Load Images"

The labeler will show images with predicted labels that need correction.

## 🖱️ Using the Web Labeler

### Interface Overview

- **Canvas**: Displays current image with keypoints and bounding box
- **Navigation**: Previous/Next buttons, image counter
- **Tools**: Multiple editing modes and shortcuts

### Editing Modes

**Keypoint Mode (Default)**
- **Drag keypoints**: Move existing keypoints by dragging the colored dots
- **Add keypoints**: Double-click to add new keypoints (cycles: nose → left ear → right ear)
- **Toggle visibility**: Hold `T` and click a keypoint to toggle visible/occluded
- **Keyboard shortcuts**:
  - `R`: Reset keypoints to last used pattern
  - `T`: Hold to toggle keypoint visibility
  - `Escape`: Clear all keypoints

**Bounding Box Mode**
- **Toggle mode**: Press `Spacebar` to switch between keypoint and bbox editing
- **Move box**: Drag inside the bounding box to move it
- **Resize box**: Drag corner handles to resize
- **Create new box**: If no box exists, click and drag to draw one
- **Manual adjustment**: Use the coordinate fields (cx, cy, w, h) for precise control

### Pattern Memory

The labeler remembers your last keypoint placement pattern. When you place keypoints on one image and move to the next, pressing `R` will apply the same relative positions, making labeling faster for similar poses.

### Saving and Navigation

- **Auto-save**: Labels are saved automatically when you navigate between images
- **Mark as OK**: Click "✓ OK" to mark current image as reviewed
- **Skip**: Click "⏭ Skip" to skip current image
- **Resume**: The labeler automatically resumes at the first unreviewed image

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Navigate between images |
| `Enter` | Mark current image as OK |
| `Spacebar` | Toggle between keypoint and bbox editing modes |
| `R` | Reset keypoints to last used pattern |
| `T` | Hold + click keypoint to toggle visibility |
| `Escape` | Clear all keypoints |

## 🔄 Promoting Corrected Samples

After reviewing and correcting samples in the web labeler, promote them back to the seed dataset:

```bash
cd autogenerate
python promote_queue_samples.py
```

**What happens:**
- Selects corrected samples from the review queue
- Copies them to the pose-seed dataset
- Removes them from the review queue
- Ready for next training iteration

**Configuration** (edit `promote_queue_samples.py`):
```python
CONFIG = PromoteConfig(
    promote_count=50,           # Number of samples to promote per run
    random_selection=True,      # Random vs first N samples
    clear_remaining=True,       # Clean up remaining queue samples
)
```

## 🔧 Configuration and Customization

### Confidence Thresholds

The pipeline uses a three-tier confidence system:

```python
# High confidence → Automatic acceptance
high_conf_box = 0.8     # Bounding box confidence
high_conf_kpt = 0.7     # All keypoints must meet this threshold

# Low confidence → Manual review
low_conf_box = 0.4      # Below this triggers review
low_conf_kpt = 0.3      # Any keypoint below this triggers review

# Middle range → Skipped (not useful for training)
```

**Tuning tips:**
- **Stricter thresholds** (higher values): Fewer auto-accepted samples, more manual review
- **Looser thresholds** (lower values): More auto-accepted samples, faster pipeline

### Training Parameters

```python
epochs = 20                 # Training epochs per iteration
learning_rate = 0.035       # High LR for rapid adaptation to new data
batch = 16                  # Batch size
imgsz = 640                # Input image size
patience = 5               # Early stopping patience
```

### Dataset Behavior

```python
trim_seed = False          # Keep all samples (recommended)
overwrite_existing = False # Skip already processed images
```

## 🔄 Complete Workflow Example

1. **Initial Setup**
   ```bash
   # Start with 30 manually labeled seed images
   ls dataset/pose-seed/images/train/  # 20 images
   ls dataset/pose-seed/images/val/    # 10 images
   ```

2. **First Pipeline Run**
   ```bash
   cd autogenerate
   python semi_auto_pose.py
   # Output: processed=1000, sent_to_pose_seed=200, sent_to_review=50, skipped_middle_conf=750
   ```

3. **Review Low-Confidence Samples**
   ```bash
   cd labeler
   python label_qc_web.py
   # Fix 50 samples in web interface
   ```

4. **Promote Corrected Samples**
   ```bash
   cd autogenerate
   python promote_queue_samples.py
   # Output: promoted 50 samples to pose-seed
   ```

5. **Second Pipeline Run**
   ```bash
   python semi_auto_pose.py
   # Now training on 280 samples (original 30 + 200 auto + 50 corrected)
   # Better model → more high-confidence predictions
   ```

6. **Repeat** until dataset is complete

## 📊 Monitoring Progress

### Pipeline Output
```
[infer] processed=2974, sent_to_pose_seed=536, sent_to_review=36, skipped_middle_conf=513, skipped_existing=1889
```

### Dataset Growth
```bash
# Check seed dataset size
find dataset/pose-seed/images -name "*.jpg" | wc -l

# Check review queue size  
find dataset/review_queue/images -name "*.jpg" | wc -l
```

### Model Performance
Training logs are saved in `autogenerate/runs_pose_seed/` with metrics and visualizations.

## 🎯 Tips for Best Results

### Labeling Strategy
- **Start with diverse samples**: Include various poses, lighting, backgrounds
- **Quality over quantity**: Better to have fewer perfect labels than many imperfect ones
- **Consistent keypoint order**: Always label nose, left ear, right ear in order
- **Use pattern memory**: Press `R` to apply consistent relative positions

### Pipeline Optimization
- **Monitor confidence thresholds**: Adjust based on your model's performance
- **Regular promotion**: Don't let review queue grow too large
- **Iterative improvement**: Each cycle should improve model performance

### Common Issues
- **Empty review queue**: Lower confidence thresholds to capture more samples
- **Too many reviews**: Raise confidence thresholds or improve seed data quality
- **Inconsistent labels**: Use pattern memory and double-check keypoint order

## 🛠️ Advanced Features

### Negative Samples
Images with "non" in the filename are treated as negative samples (no lizard present) and automatically added to the seed dataset with empty labels.

### Custom Paths
Edit the configuration blocks in each script to customize directories:
```python
seed_dataset = Path("dataset/pose-seed")
unlabeled_dataset = Path("dataset/head-detection-dataset") 
review_queue = Path("dataset/review_queue")
```

### Dry Run Mode
Test changes without modifying files:
```python
dry_run = True  # Set in CONFIG
```

## 📝 File Formats

### YOLO Pose Label Format
```
0 0.5 0.3 0.2 0.4 0.45 0.25 2 0.35 0.15 2 0.55 0.15 2
```
- Class 0, bbox center (0.5, 0.3), size 0.2×0.4
- Nose at (0.45, 0.25), visible
- Left ear at (0.35, 0.15), visible  
- Right ear at (0.55, 0.15), visible

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Bitmap (.bmp)
- TIFF (.tif, .tiff)

---

This pipeline enables efficient creation of large-scale pose estimation datasets with minimal manual effort. The combination of automated routing and intelligent labeling tools accelerates the typically tedious process of pose annotation.
