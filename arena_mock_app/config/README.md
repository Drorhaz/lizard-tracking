# Configuration Guide

This folder contains configuration files for the Head Pose Detection Application.

## Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your values:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Update the paths** to match your system:
   - `MODEL_PATH` - Path to your YOLO model file
   - `VIDEO_PATH` - Path to the video you want to analyze
   - `OUTPUT_DIR` - Where to save results

## Configuration Sections

### 🤖 Model Configuration
Controls the YOLO pose detection model behavior:
- **MODEL_PATH**: Location of the trained `.pt` model file
- **CONFIDENCE_THRESHOLD**: How confident the model must be (0.0-1.0)
  - Lower = more detections but noisier
  - Higher = fewer but more accurate detections

### 📹 Video Input
Settings for video processing:
- **VIDEO_PATH**: Input video file location
- **PROCESSING_FPS**: How many frames to process per second
  - Lower = more stable, less CPU
  - Higher = more responsive but heavier load

### 💾 Output Configuration
Where and how often to save results:
- **OUTPUT_DIR**: Main output folder for all results
- **SAVE_EVERY_N_FRAMES**: Frequency of saving labeled frame images
  - Higher = saves disk space
  - Lower = more visual samples to review

### 🧠 Advanced Behavioral Analysis
Fine-tune the behavioral detection system:

**Spatial Zones:**
- **TARGET_LINE**: Which edge has the screen/target (`right`, `left`, `top`, `bottom`)
- **NEAR_MAX**: Distance considered "near" the target (fraction of frame)
- **MIDDLE_MAX**: Distance for middle buffer zone

**Movement Sensitivity:**
- **ADVANCE_THRESHOLD**: Sensitivity for detecting approach movements
- **RETREAT_THRESHOLD**: Sensitivity for detecting retreat movements
- **X_DIR_THRESH_NORM**: Minimum horizontal movement to detect
- **Y_DIR_THRESH_NORM**: Minimum vertical movement to detect

**Behavior Classification:**
- **HEAD_ONLY_THRESH_NORM**: Threshold for head-only movements (wiggling, orienting)
- **BODY_MOVE_THRESH_NORM**: Threshold for full-body locomotion (walking)
- **LOOKBACK_WINDOW**: How many frames to look back for missing detections

### 🔄 Simple Behavior Detection (Fallback)
Simpler behavior detection when advanced analysis isn't available:
- **MIN_MOVING_FRAMES**: Frames needed to classify as "moving"
- **STOP_THRESHOLD**: Pixel distance threshold for "stopped"
- **MIN_STATIONARY_FRAMES**: Frames needed to classify as "stationary"

### 🌐 Web Server
Flask web interface settings:
- **SERVER_HOST**: Network interface (`0.0.0.0` = all, `127.0.0.1` = localhost only)
- **SERVER_PORT**: Port number (default: 8078)
- **SERVER_DEBUG**: Debug mode (only for development!)
- **STREAM_FPS**: Frame rate for web video stream
- **JPEG_QUALITY**: Quality of streamed images (1-100)

## Example Configurations

### High Accuracy Mode
```env
CONFIDENCE_THRESHOLD=0.6
PROCESSING_FPS=5
ADVANCE_THRESHOLD=0.005
RETREAT_THRESHOLD=0.005
```

### Fast Processing Mode
```env
CONFIDENCE_THRESHOLD=0.3
PROCESSING_FPS=15
JPEG_QUALITY=70
```

### Sensitive Behavior Detection
```env
ADVANCE_THRESHOLD=0.001
RETREAT_THRESHOLD=0.001
HEAD_ONLY_THRESH_NORM=0.003
BODY_MOVE_THRESH_NORM=0.008
```

## Troubleshooting

**No detections showing up?**
- Lower `CONFIDENCE_THRESHOLD` (try 0.1)
- Check that `MODEL_PATH` and `VIDEO_PATH` are correct

**Too many false positives?**
- Raise `CONFIDENCE_THRESHOLD` (try 0.4 or 0.5)
- Increase movement thresholds

**Video stream is laggy?**
- Lower `STREAM_FPS` (try 10)
- Lower `JPEG_QUALITY` (try 70)
- Lower `PROCESSING_FPS`

**Behavioral events are too sensitive?**
- Increase `ADVANCE_THRESHOLD` and `RETREAT_THRESHOLD`
- Increase `HEAD_ONLY_THRESH_NORM` and `BODY_MOVE_THRESH_NORM`

## Security Note

⚠️ **Never commit `.env` to version control!** It may contain paths specific to your system.

The `.env.example` file is safe to commit and should be kept updated with new parameters.
