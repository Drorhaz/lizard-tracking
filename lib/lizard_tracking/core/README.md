# Lizard Tracking - Unified Frame Processing Architecture

## Overview

This library provides a complete, plug-and-play system for real-time and offline animal tracking with behavioral analysis. Inspired by [PreyTouch](https://github.com/EvolutionaryNeuralCodingLab/PreyTouch), it's designed to be modular and easy to integrate into any application.

## Features

### 🎯 Core Capabilities
- **YOLO Pose Detection**: Head and keypoint detection with bounding boxes
- **Real-time Behavioral Analysis**: Approach, retreat, stop/movement detection
- **Trajectory Tracking**: Frame-by-frame position and speed logging
- **Flexible Output**: CSV trajectories, YOLO labels, annotated frames
- **Config-driven**: All settings via configuration object

### 📊 Data Outputs
- **Trajectory CSV**: `frame, timestamp, head_x, head_y, distance, speed, events`
- **YOLO Labels**: Normalized bbox coordinates for retraining
- **Annotated Frames**: Frames with detection overlays drawn
- **Event Logs**: Behavioral events with timestamps and metrics

### 🎛️ Configuration Options

```python
ProcessorConfig(
    # Detection settings
    model_path="path/to/yolo11.pt",
    conf_threshold=0.25,
    imgsz=640,
    
    # Output settings
    output_dir=Path("output"),
    save_trajectory=True,
    save_labels=True,
    save_frames=False,
    
    # Behavioral analysis
    enable_behavioral_analysis=True,
    screen_location="right",  # "left", "right", "top", "bottom"
    reference_point=(x, y),   # Or None for auto-calculation
    approach_threshold_px=300.0,
    retreat_threshold_px=300.0,
    stop_speed_threshold=2.0,
    
    # Processing
    process_every_n_frames=1,  # 1 = every frame
    fps_target=30.0
)
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/lizard-tracking.git
cd lizard-tracking

# Install dependencies
pip install ultralytics opencv-python numpy pandas

# Install behavioral analysis library
pip install -e lib/behavioral_analysis
```

### Basic Usage

#### Offline Video Processing

```python
from lizard_tracking.core.frame_processor import FrameProcessor, ProcessorConfig
import cv2

# Configure
config = ProcessorConfig(
    model_path="models/yolo11n-pose.pt",
    output_dir="output",
    save_trajectory=True,
    enable_behavioral_analysis=True,
    screen_location="right"
)

# Create processor
processor = FrameProcessor(config)

# Process video
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = processor.process_frame(frame)
    
    # Use results
    if result.pose:
        print(f"Detection at frame {result.frame_number}")
    
    for event in result.behavioral_events:
        print(f"Event: {event.event_type.value}")
    
    cv2.imshow("Tracking", result.display_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
processor.close()
```

#### Real-time Camera Processing

```python
# Same setup, just use camera as source
cap = cv2.VideoCapture(0)  # Camera ID

processor = FrameProcessor(config)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = processor.process_frame(frame)
    
    # Real-time event handling
    for event in result.behavioral_events:
        print(f"LIVE EVENT: {event.event_type.value}")
    
    cv2.imshow("Live", result.display_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
processor.close()
```

### Using the Example Scripts

```bash
# Process video file
python examples/process_with_frameprocessor.py \
    --video path/to/video.mp4 \
    --model models/yolo11n-pose.pt

# Process real-time camera
python examples/process_with_frameprocessor.py \
    --camera 0 \
    --model models/yolo11n-pose.pt
```

## Architecture

### Core Components

```
lib/lizard_tracking/core/
├── frame_processor.py      # Main unified processor
├── __init__.py

lib/behavioral_analysis/    # Behavioral analysis library
├── detector.py             # Event detection
├── config.py               # Configuration
├── events.py               # Event types and bus
├── metrics.py              # Live metrics tracking
└── export.py               # Data export utilities
```

### Processing Flow

```
Input Frame
    ↓
YOLO Pose Detection
    ↓
Best Pose Selection
    ↓
Behavioral Analysis
    ├→ Distance calculation
    ├→ Speed calculation
    └→ Event detection (approach/retreat/stop)
    ↓
Draw Overlays
    ↓
Save Outputs
    ├→ Trajectory CSV
    ├→ YOLO Labels (optional)
    └→ Frames (optional)
    ↓
Return FrameResult
```

## Output Format

### Trajectory CSV

```csv
frame,timestamp,elapsed_sec,head_x,head_y,confidence,distance_from_edge,speed_px_per_frame,event_type,event_name
1,2025-10-03T18:30:39.345,0.000,486.68,268.07,0.6791,152.32,0.00,approach_start,Approach Start
2,2025-10-03T18:30:40.378,1.033,492.15,270.32,0.7123,146.85,5.52,,
3,2025-10-03T18:30:41.411,2.066,,,,,,,
```

**Columns:**
- `frame`: Frame number
- `timestamp`: ISO timestamp
- `elapsed_sec`: Seconds since start
- `head_x`, `head_y`: Head position (nose keypoint or bbox center)
- `confidence`: Detection confidence
- `distance_from_edge`: Distance from reference point (pixels)
- `speed_px_per_frame`: Movement speed
- `event_type`, `event_name`: Behavioral events

### YOLO Labels

Format: `class cx cy w h conf` (normalized 0-1)

```
0 0.523456 0.456789 0.123456 0.234567 0.8531
```

## Integration with Web Interfaces

### Flask Example

```python
from flask import Flask, Response
from lizard_tracking.core.frame_processor import FrameProcessor, ProcessorConfig
import cv2

app = Flask(__name__)
processor = FrameProcessor(ProcessorConfig(
    model_path="models/best.pt",
    enable_behavioral_analysis=True
))

def generate_frames():
    cap = cv2.VideoCapture("video.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = processor.process_frame(frame)
        
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', result.display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def get_events():
    if processor.behavior_detector:
        events = processor.behavior_detector.event_bus.get_recent_events(10)
        return jsonify([e.to_dict() for e in events])
    return jsonify([])
```

## Screen Location Configuration

Specify where the interaction screen is located relative to the camera view:

```python
# Vertical screen on right side (default)
config.screen_location = "right"  # Reference: (width-1, height/2)

# Vertical screen on left side
config.screen_location = "left"   # Reference: (0, height/2)

# Horizontal screen at top
config.screen_location = "top"    # Reference: (width/2, 0)

# Horizontal screen at bottom
config.screen_location = "bottom" # Reference: (width/2, height-1)

# Custom reference point
config.reference_point = (800, 400)  # Explicit (x, y) coordinates
```

## Behavioral Events

The system detects the following events:

| Event | Description | Trigger Condition |
|-------|-------------|-------------------|
| `approach_start` | Animal starts moving towards screen | Distance decreases below approach threshold |
| `approach_end` | Animal stops approaching | Movement stops or distance increases |
| `retreat_start` | Animal starts moving away from screen | Distance increases above retreat threshold |
| `retreat_end` | Animal stops retreating | Movement stops or distance decreases |
| `stop_start` | Animal stops moving | Speed drops below stop threshold |
| `stop_end` | Animal resumes movement | Speed exceeds stop threshold |

## Performance Optimization

### For Real-time Processing

```python
config = ProcessorConfig(
    imgsz=640,  # Smaller for faster inference
    process_every_n_frames=2,  # Process every 2nd frame
    save_frames=False,  # Don't save frames in real-time
    save_labels=False   # Don't save labels in real-time
)
```

### For Offline Analysis

```python
config = ProcessorConfig(
    imgsz=960,  # Larger for better accuracy
    process_every_n_frames=1,  # Process all frames
    save_frames=True,   # Save annotated frames
    save_labels=True    # Save labels for retraining
)
```

## Comparison with PreyTouch

| Feature | PreyTouch | This Library |
|---------|-----------|--------------|
| Architecture | Multi-process with shared memory | Single/multi-process compatible |
| Detection | YOLO pose | YOLO pose |
| Behavioral Analysis | Integrated | Modular (lib/behavioral_analysis) |
| Output | CSV, videos, labels | CSV, videos, labels, events |
| Configuration | Config files | Python config objects |
| Integration | Arena-specific | Generic, pluggable |

## Troubleshooting

### Events not appearing in output

**Problem**: Events are detected but not showing in logs/CSV

**Solution**: Ensure behavioral analysis is enabled and configured:

```python
config.enable_behavioral_analysis = True
config.approach_threshold_px = 300.0  # Adjust based on your setup
```

### Detection but empty coordinates

**Problem**: CSV shows detections but `head_x`, `head_y` are empty

**Solution**: Check if model has keypoints. Fallback to bbox center:

```python
if pose.nose:
    x, y = pose.nose
else:
    x1, y1, x2, y2 = pose.bbox_xyxy
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
```

### Performance issues

**Problem**: Processing is too slow

**Solution**: 
1. Reduce `imgsz` (640 instead of 960)
2. Increase `process_every_n_frames` (process every 2nd or 3rd frame)
3. Disable frame saving in real-time mode

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

[Add your license here]

## Citation

If you use this library in your research, please cite:

```bibtex
@software{lizard_tracking_2025,
  title={Lizard Tracking: Unified Frame Processing for Animal Behavioral Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/lizard-tracking}
}
```
