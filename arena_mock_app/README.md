# Head Pose Detection Application

A real-time web-based application for detecting and analyzing lizard head poses in video recordings. Features advanced behavioral analysis, trajectory tracking, and interactive visualizations.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Application

Copy the example configuration and edit with your paths:

```bash
cd config
cp .env.example .env
nano .env  # Edit with your model and video paths
```

**Key settings to update:**
- `MODEL_PATH` - Path to your trained YOLO model (.pt file)
- `VIDEO_PATH` - Path to the video you want to analyze
- `OUTPUT_DIR` - Where to save detection results

See `config/README.md` for detailed configuration options.

### 3. Run the Application

```bash
python api.py
```

The web interface will be available at: **http://localhost:8078**

## 📋 Features

### Real-Time Detection
- YOLO-based head pose detection
- Live video streaming with overlaid detections
- Configurable confidence thresholds

### Behavioral Analysis
- **Advanced Mode**: Arena mapping, approach/retreat detection, instruction grammar
- **Simple Mode**: Movement detection, stationary/moving classification
- Real-time behavioral event logging

### Data Export
- Frame-by-frame detection CSV (`detections.csv`)
- Detailed trajectory data (`trajectory.csv`)
- Behavioral events log (`behavioral_events.csv`)
- Interactive HTML plots (`nose_heading_map.html`)
- Labeled frame images for visual verification

### Web Interface
- Start/Stop detection controls
- Real-time video feed with detections
- Live behavioral event stream
- System status monitoring

## Output Structure

Detection results are saved to `output/detections/[video-name]-[timestamp]/`:
- **`detections.csv`** - Frame-by-frame detection data with nose coordinates
- **`trajectory.csv`** - Detailed movement tracking with behavioral events
- **`behavioral_events.csv`** - Detected approach/retreat instructions
- **`nose_heading_map.html`** - Interactive trajectory visualization
- **`labeled_frames/`** - Clean frames (NO drawings) saved periodically
- **`labels/`** - YOLO pose labels (bbox + keypoints) matching frames
- **`preview_frames/`** - Annotated frames WITH drawings for inspection

## ⚙️ Configuration

All settings are managed through the `config/.env` file:

### Model Settings
- `MODEL_PATH` - YOLO model location
- `CONFIDENCE_THRESHOLD` - Detection confidence (0.0-1.0)

### Video Settings
- `VIDEO_PATH` - Input video file
- `PROCESSING_FPS` - Processing frame rate

### Behavioral Analysis
- `TARGET_LINE` - Screen position (right/left/top/bottom)
- `NEAR_MAX`, `MIDDLE_MAX` - Distance zones
- Movement sensitivity thresholds
- Lookback window for missing detections

### Server Settings
- `SERVER_HOST` - Network interface (default: 0.0.0.0)
- `SERVER_PORT` - Port number (default: 8078)
- `STREAM_FPS` - Web stream frame rate
- `JPEG_QUALITY` - Stream image quality

See `config/README.md` for complete documentation.

## 🎯 Usage Examples

### High Accuracy Mode
```env
CONFIDENCE_THRESHOLD=0.6
PROCESSING_FPS=5
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
```

## 🔍 Troubleshooting

### No detections appearing?
- Lower `CONFIDENCE_THRESHOLD` in config/.env (try 0.1)
- Verify `MODEL_PATH` and `VIDEO_PATH` are correct
- Check model is compatible with your video resolution

### Video stream is laggy?
- Lower `STREAM_FPS` (try 10)
- Lower `JPEG_QUALITY` (try 60-70)
- Reduce `PROCESSING_FPS`

### Too many false positives?
- Raise `CONFIDENCE_THRESHOLD` (try 0.4-0.5)
- Increase behavioral movement thresholds

### Import errors?
Make sure the parent `lib` directory is accessible:
```bash
# From lizard-tracking root
ls lib/lizard_tracking/
ls lib/behavioral_analysis/
```

## 🏷️ Working with Labels and Annotations

The app saves both **clean frames** and **labels** for training/annotation workflows:

### Using Saved Frames + Labels

Frames and labels are saved to:
```
output/detections/[run-name]/
├── labeled_frames/   # Clean images (no drawings)
├── labels/           # YOLO pose format (.txt files)
└── preview_frames/   # Visual inspection (WITH drawings)
```

### Preview Labels Tool

Visualize labels on images WITHOUT running the model:

```bash
cd ../tools
python preview_labels.py
```

Configure in `preview_labels.py`:
```python
CONFIG = {
    "IMAGES_ROOT": "../arena_mock_app/output/detections/[run]/labeled_frames",
    "LABELS_ROOT": "../arena_mock_app/output/detections/[run]/labels",
    "OUT_DIR": "preview_output"
}
```

This draws keypoints and bounding boxes from saved labels for inspection.

### Interactive Labeler Tool

Fix/adjust labels interactively:

```bash
cd ../labeler
python labeler_app.py
```

Features:
- Load frames and labels together
- Visual editing of keypoints and bounding boxes  
- Save corrections back to YOLO format
- Perfect for quality control and refinement

### Frame Saving Configuration

Control how often frames are saved:

```env
# Save clean frames + labels every 10 detections
SAVE_EVERY_N_FRAMES=10

# Save preview frames every 30 detections  
SAVE_EVERY_N_PREVIEWS=30
```

**Tips:**
- Lower `SAVE_EVERY_N_FRAMES` = more training data
- Higher `SAVE_EVERY_N_PREVIEWS` = less disk usage
- Set to 0 to disable either type

## 📊 Analyzing Results

After running detection:

1. **Check CSV files** for raw data:
   ```bash
   # View detections
   column -t -s, output/detections/run_*/detections.csv | less
   
   # View trajectory
   column -t -s, output/detections/run_*/trajectory.csv | less
   ```

2. **Open interactive plot** in browser:
   ```bash
   # Open the HTML visualization
   firefox output/detections/run_*/nose_heading_map.html
   ```

3. **Review labeled frames**:
   ```bash
   # View sample frames with detections
   ls output/detections/run_*/labeled_frames/
   ```

## 🔧 Development

### File Structure
```
arena_mock_app/
├── api.py              # Main application
├── config/             # Configuration management
│   ├── .env           # Your settings (not in git)
│   ├── .env.example   # Template
│   ├── .gitignore     # Protect .env
│   └── README.md      # Config documentation
├── requirements.txt    # Python dependencies
├── videos/            # Sample videos
└── README.md          # This file
```

### Adding New Features

The application uses a modular structure:
- **SimpleHeadPoseDetector** - Main detection class
- **AppConfig** - Configuration management
- **Flask routes** - Web API endpoints
- **HTML_TEMPLATE** - Web interface

## 📝 License

Part of the lizard-tracking project.

## 🆘 Support

For issues or questions:
1. Check `config/README.md` for configuration help
2. Review troubleshooting section above
3. Check console output for error messages
4. Verify all file paths are absolute and correct
