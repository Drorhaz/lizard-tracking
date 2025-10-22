# 🦎 Lizard Pose Head Detection Pipeline

A complete pipeline for real-time lizard head pose detection using YOLO models with both web interface and command-line support.

---

## 🚀 Quick Start

### Prerequisites
```bash
# Navigate to your project directory (replace with your actual path)
cd ~/lizard-tracking/pose-head

# Activate your Python environment
conda activate LizardPose  # or your environment name
```

### **1. Web Interface (Interactive)**
```bash
cd ~/lizard-tracking/pose-head
python launch_web.py
# Access: http://localhost:8765
```

**Features:**
- 🎬 **Video Selection**: Choose from automatically discovered videos
- ⚡ **Execution Modes**: Local CPU, Local GPU, or HPC Cluster
- 🔴 **Live Detection**: Real-time pose inference with overlays
- 📁 **Offline Playback**: Load and view previously saved results
- 📊 **Real-time Stats**: FPS, detection rate, progress tracking

### 2. Command Line Interface
For automated processing without GUI:

```bash
# Set environment variables for your video and mode
export VIDEO_PATH="videos/your_video.mp4"
export MODE="INFER_LIVE"          # or LABELS_ONLY or PLAYBACK_CACHE
export OUTPUT_DIR="output/detections"

# Run the pipeline
python pipeline/video_pose_pipeline.py
```

---

## 📋 Execution Modes

### Web Interface Modes

| Mode | Device | Execution | Description |
|------|--------|-----------|-------------|
| **Local (CPU)** | CPU | Local process | Development and testing |
| **Local (GPU)** | GPU | Local process | Production processing |
| **HPC Cluster (GPU)** | HPC GPU | SLURM job |  Fast & accurate HPC inference |

### Command Line Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **INFER_LIVE** | Run inference + live preview + save results | Interactive development |
| **LABELS_ONLY** | Run inference + save results (no preview) | Batch processing |
| **PLAYBACK_CACHE** | Load saved labels + replay with overlays | Review previous results |

---

## 📁 Project Structure

```
pose-head/
├── pipeline/
│   ├── web_interface.py           # Main web interface
│   ├── video_pose_pipeline.py     # Core pipeline (standalone)
│   ├── shared_web_interface.py    # Video streaming
│   └── templates/                 # Web UI templates
├── hpc/
│   ├── submit_labels_gpu.sh       # SLURM job submission
│   └── run_labels_gpu.sbatch      # SLURM batch script
├── videos/                        # Video files (auto-discovered)
├── output/
│   └── detections/                # Results output
├── launch_web.py                  # Web interface launcher
└── README.md                      # This file
```

---

## 🔧 Configuration

### Environment Variables

#### Video Processing
```bash
export VIDEO_PATH="path/to/video.mp4"    # Input video file
export OUTPUT_DIR="output/detections"     # Results directory
export MODEL_PATH="yolo11s-pose.pt"      # YOLO model file
```

#### Detection Settings
```bash
export CONF_THRESH="0.1"                 # Confidence threshold (0.0-1.0)
export IMG_SIZE="640"                     # Input image size
export LABEL_EVERY_N="10"                # Save every N frames
```

#### Display Options
```bash
export MODE="INFER_LIVE"                  # Processing mode
export PREVIEW="true"                     # Show preview window
export WEB_PREVIEW="true"                 # Enable web streaming
export SAVE_LABELS="true"                 # Save detection labels
```

#### HPC Settings
```bash
export PARTITION="gpu"                    # SLURM partition
export TIME_LIMIT="02:00:00"              # Job time limit
export WEB_HOST="0.0.0.0"                 # Web interface host
export WEB_PORT="8765"                    # Web interface port
```

---

## 💻 Usage Examples

### Basic Local Inference
```bash
cd ~/lizard-tracking/pose-head

# Set your video
export VIDEO_PATH="videos/lizard_video.mp4"
export MODE="INFER_LIVE"

# Run with live preview
python pipeline/video_pose_pipeline.py
```

### Batch Processing (No GUI)
```bash
cd ~/lizard-tracking/pose-head

# Process multiple videos
for video in videos/*.mp4; do
    export VIDEO_PATH="$video"
    export MODE="LABELS_ONLY"
    export PREVIEW="false"
    python pipeline/video_pose_pipeline.py
done
```

### Review Previous Results
```bash
cd ~/lizard-tracking/pose-head

# Replay saved detections
export VIDEO_PATH="videos/processed_video.mp4"
export MODE="PLAYBACK_CACHE"
python pipeline/video_pose_pipeline.py
```

### HPC Cluster Processing
```bash
cd ~/lizard-tracking/pose-head

# Submit to GPU cluster
export VIDEO_PATH="videos/long_video.mp4"
export PARTITION="gpu"
export TIME_LIMIT="04:00:00"

# Submit job
bash hpc/submit_labels_gpu.sh
```

---

## 📊 Output Format

Each run creates a timestamped directory:
```
output/detections/video_name-20250101-120000/
├── detections.csv              # Frame-by-frame results
├── run_config.json            # Configuration snapshot
├── labels/                    # YOLO format labels
│   ├── frame00000001.txt
│   ├── frame00000002.txt
│   └── ...
└── labeled_frames/            # Annotated video frames
    ├── frame00000001.jpg
    ├── frame00000010.jpg      # Every LABEL_EVERY_N frames
    └── ...
```

### CSV Format
```csv
frame_number,bbox_x,bbox_y,bbox_w,bbox_h,confidence,keypoints_x,keypoints_y
1,320.5,240.2,45.8,52.1,0.87,"[315.2,345.8,...]","[235.1,248.9,...]"
2,,,,,,"",""  # No detection
```

---

## 🔍 Troubleshooting

### Common Issues

#### "No videos found"
- Check that video files exist in `videos/` directory
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`
- Videos are auto-discovered in: `videos/`, `scripts/`, `dataset/videos/`

#### "Model not found"
- Ensure YOLO model files exist in project root
- Auto-detected models: `yolo11s-pose.pt`, `yolo11n-pose.pt`, `best.pt`
- Or set: `export MODEL_PATH="path/to/your/model.pt"`

#### "Permission denied" on HPC
- Check SLURM partition access: `sinfo`
- Verify script permissions: `chmod +x hpc/submit_labels_gpu.sh`

#### Web interface not loading
- Check port availability: `netstat -ln | grep 8765`
- Try different port: `export WEB_PORT="8766"`
- Check firewall settings for remote access

---

## 🐛 Known Issues & Future Improvements

### Minor Issues (To Be Fixed)
- [ ] **Stream Stop**: Video streaming doesn't stop cleanly when switching videos
- [ ] **Restart Handling**: Interface needs better restart/reset functionality  
- [ ] **Error Recovery**: Better error handling for failed model loading
- [ ] **Memory Management**: Long videos may consume excessive memory
- [ ] **Progress Tracking**: HPC job progress reporting could be more accurate

### Enhancement Ideas
- [ ] **Multi-Video Processing**: Batch process multiple videos simultaneously
- [ ] **Real-time Camera**: Support live camera input
- [ ] **Model Comparison**: Side-by-side comparison of different models
- [ ] **Export Options**: Direct export to common formats (MOV, GIF)
- [ ] **Performance Profiling**: Built-in timing and performance metrics

---

## 🤝 Contributing

When developing:
1. Always use relative paths (project auto-detects root directory)
2. Test both web interface and command-line modes
3. Verify HPC compatibility if modifying batch scripts
4. Update this README for any new features or configuration options

---

## 📝 Version History

- **v2.0**: Portable paths, web interface, multi-execution modes
- **v1.0**: Basic command-line pipeline with YOLO detection

**Current Status**: ✅ Fully portable across systems, ready for production use
