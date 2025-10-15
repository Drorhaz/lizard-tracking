# Semi-Automatic Pose Labeling Pipeline

This directory contains a complete semi-automatic pipeline for training YOLO pose estimation models on lizard head detection with 3 keypoints (nose, left ear, right ear). The system combines automated model training, intelligent sample routing, and an interactive web-based labeler for efficient dataset creation and refinement.

## 🔄 Pipeline Overview

The semi-automatic pipeline operates in a continuous cycle:

1. **Train**: Train a YOLO pose model on manually labeled seed data
2. **Infer**: Run inference on unlabeled images from the dataset
3. **Route**: Automatically sort predictions based on confidence:
   - **High confidence** → Add to seed dataset (automatic training data)
   - **Low confidence** → Send to review queue (manual correction needed)
   - **Middle confidence** → Skip (not useful for training)
4. **Review**: Use web labeler to fix low-confidence predictions
5. **Promote**: Move corrected samples back to seed dataset
6. **Repeat**: Next iteration trains on expanded seed data

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
cd ../labeler
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
   cd ../labeler
   python label_qc_web.py
   # Fix 50 samples in web interface
   ```

4. **Promote Corrected Samples**
   ```bash
   cd ../autogenerate
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