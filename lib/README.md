# Lizard Tracking Library

This library provides reusable components for lizard pose detection and behavioral analysis.

## Video Streaming Utilities

The video streaming module provides frame-by-frame video playback with configurable FPS, timestamp display, and precise frame synchronization.

### Quick Start

```python
import sys
sys.path.append('lib')
from lizard_tracking.utils.video_stream import stream_video

# Simple streaming
with stream_video('video.mp4', fps=30.0) as streamer:
    streamer.play()
```

### VideoStreamConfig

Configure video streaming behavior:

```python
from lizard_tracking.utils.video_stream import VideoStreamConfig, VideoStreamer

config = VideoStreamConfig(
    video_path='pose-head/videos/top_20250916T150021.mp4',
    fps=30.0,                           # Playback FPS (default: 30.0)
    show_timestamp=True,                # Show timestamp overlay (default: True)
    timestamp_color=(0, 255, 0),        # Neon green timestamp (BGR)
    timestamp_position=(10, 30),        # Top-left position
    window_name="My Video Stream",      # Window title
    loop=False,                         # Loop video when finished
    start_time_seconds=0.0              # Start from specific time
)

with VideoStreamer(config) as streamer:
    streamer.play()
```

### Features

#### ⚡ Frame Rate Control
- **Configurable FPS**: Play at any desired frame rate (1-60+ FPS)
- **Precise Timing**: Frame-accurate playback timing
- **Original vs Playback**: Automatically handles different source/playback rates

#### 🟢 Timestamp Display  
- **Neon Green Color**: Highly visible timestamp overlay (configurable)
- **Format**: MM:SS.mmm display format
- **Position**: Configurable screen position
- **Outline**: Black outline for better visibility

#### 🎮 Interactive Controls
- **Space**: Pause/Resume playback
- **Q**: Quit streaming
- **R**: Restart from beginning

#### 🔄 Advanced Features
- **Looping**: Seamless video loops
- **Seeking**: Jump to specific time/frame
- **Callbacks**: Custom frame processing
- **Resizing**: Configurable window dimensions

### Example Usage

#### Basic Streaming
```python
from lizard_tracking.utils.video_stream import stream_video

# Stream at 30 FPS with neon green timestamp
with stream_video('video.mp4') as streamer:
    streamer.play()
```

#### Custom Frame Processing
```python
def process_frame(frame, timestamp, frame_number):
    """Add custom overlays to each frame"""
    import cv2
    
    # Add frame counter
    text = f"Frame: {frame_number}"
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame

config = VideoStreamConfig(
    video_path='video.mp4',
    fps=30.0,
    frame_callback=process_frame  # Custom processing
)

with VideoStreamer(config) as streamer:
    streamer.play()
```

#### Different Frame Rates
```python
# High-speed playback (60 FPS)
with stream_video('video.mp4', fps=60.0, timestamp_color=(255, 0, 255)) as streamer:
    streamer.play()

# Slow-motion playback (10 FPS)  
with stream_video('video.mp4', fps=10.0, timestamp_color=(0, 255, 255)) as streamer:
    streamer.play()
```

#### Programmatic Control
```python
from lizard_tracking.utils.video_stream import VideoStreamer, VideoStreamConfig

config = VideoStreamConfig('video.mp4', fps=30.0)
streamer = VideoStreamer(config)
streamer.open()

# Seek to 2 minutes
streamer.seek_to_time(120.0)

# Read individual frames
frame = streamer.read_frame()
processed_frame = streamer.process_frame(frame)

# Get video info
print(f"Duration: {streamer.total_frames / streamer.original_fps:.1f} seconds")
print(f"Current time: {streamer.format_timestamp(streamer.get_current_timestamp())}")

streamer.close()
```

### Integration with Pose Detection

```python
from lizard_tracking.utils.video_stream import stream_video
from lizard_tracking.utils.draw_utils import draw_head_pose_from_object

def add_pose_overlay(frame, timestamp, frame_num):
    """Add pose detection to video stream"""
    
    # Run pose detection (pseudo-code)
    detection = pose_model.predict(frame)
    
    # Add pose overlay
    if detection:
        frame = draw_head_pose_from_object(frame, detection)
    
    return frame

# Stream video with real-time pose detection
with stream_video('lizard_video.mp4', 
                  fps=30.0, 
                  frame_callback=add_pose_overlay) as streamer:
    streamer.play()
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | str | Required | Path to video file |
| `fps` | float | 30.0 | Playback frame rate |
| `loop` | bool | False | Loop video when finished |
| `start_time_seconds` | float | 0.0 | Start playback time |
| `show_timestamp` | bool | True | Show timestamp overlay |
| `timestamp_color` | tuple | (0,255,0) | Timestamp color (BGR) |
| `timestamp_position` | tuple | (10,30) | Timestamp position (x,y) |
| `timestamp_font_scale` | float | 1.0 | Timestamp font size |
| `window_name` | str | "Video Stream" | Window title |
| `window_width` | int | None | Resize window width |
| `window_height` | int | None | Resize window height |
| `frame_callback` | callable | None | Custom frame processor |

### Performance Notes

- **Frame Timing**: Uses precise sleep timing for accurate FPS
- **Memory Efficient**: Processes one frame at a time
- **Thread Safe**: Can be stopped from other threads
- **Headless Support**: Works in environments without GUI

## Drawing Utilities

The drawing utilities module provides functions for visualizing pose detections and behavioral events on video frames.

### Quick Start

```python
import sys
sys.path.append('lib')
from lizard_tracking.utils.draw_utils import draw_head_pose_from_object

# Draw pose detection on frame
labeled_frame = draw_head_pose_from_object(frame, detection_object)
```

### Functions Overview

#### `draw_head_pose_from_object(frame, head_pose_obj)`

**Purpose**: Draw complete pose detection overlay from a detection object

**Parameters**:
- `frame` (numpy.ndarray): Input video frame (BGR format)
- `head_pose_obj` (object): Detection object with pose attributes

**Returns**: Frame with pose overlay drawn

**Expected Object Attributes**:
The detection object should have these attributes (all optional):
- `bbox` or `bbox_xyxy`: Bounding box coordinates [x1, y1, x2, y2]
- `nose_tip` or `nose`: Nose keypoint coordinates [x, y]  
- `left_ear` or `ear_left`: Left ear coordinates [x, y]
- `right_ear` or `ear_right`: Right ear coordinates [x, y]
- `confidence` or `conf`: Detection confidence score (0.0-1.0)

**Example Usage**:

```python
import cv2
import sys
sys.path.append('lib')
from lizard_tracking.utils.draw_utils import draw_head_pose_from_object

# Load your frame
frame = cv2.imread('lizard_frame.jpg')

# Create a detection object (example from YOLO inference)
class Detection:
    def __init__(self):
        self.bbox = [100, 50, 200, 150]  # [x1, y1, x2, y2]
        self.nose_tip = [170, 90]        # [x, y]
        self.left_ear = [130, 70]        # [x, y] 
        self.right_ear = [130, 110]      # [x, y]
        self.confidence = 0.95           # float

detection = Detection()

# Draw the pose overlay
labeled_frame = draw_head_pose_from_object(frame, detection)

# Save or display result
cv2.imwrite('labeled_frame.jpg', labeled_frame)
```

**Visual Output**:
- 🟢 **Green bounding box** around the head
- 🔴 **Red circle** at nose tip position
- 🔵 **Blue circles** at ear positions
- 💛 **Yellow line** from nose to midpoint between ears (head direction)
- 📊 **Confidence text** showing detection score

#### `draw_head_pose(frame, bbox_xyxy, nose, ear_left, ear_right, confidence)`

**Purpose**: Low-level function to draw pose with individual coordinate parameters

**Parameters**:
- `frame`: Input frame
- `bbox_xyxy`: Bounding box [x1, y1, x2, y2]
- `nose`: Nose coordinates [x, y]
- `ear_left`: Left ear coordinates [x, y]
- `ear_right`: Right ear coordinates [x, y]
- `confidence`: Confidence score (optional)

#### `draw_no_detection(frame, text="No Detection")`

**Purpose**: Draw "no detection" overlay when no lizard is found

#### `draw_behavioral_event(frame, event_type, position, duration)`

**Purpose**: Draw behavioral event notifications (approach, retreat, stop)

#### `draw_processing_info(frame, mode, fps, frame_count, detection_count)`

**Purpose**: Draw processing statistics overlay

#### `draw_trajectory_line(frame, points, color, thickness)`

**Purpose**: Draw trajectory path connecting multiple points

### Integration Examples

#### With pose-head System
```python
from pipeline.video_pose_pipeline import YOLOPoseModel
from lizard_tracking.utils.draw_utils import draw_head_pose_from_object

model = YOLOPoseModel("best.pt")
detection = model.predict(frame)
labeled_frame = draw_head_pose_from_object(frame, detection)
```

#### With YOLO Results
```python
import torch
from ultralytics import YOLO
from lizard_tracking.utils.draw_utils import draw_head_pose_from_object

model = YOLO('best.pt')
results = model(frame)

for result in results:
    if result.keypoints is not None:
        # Create detection object from YOLO result
        class YOLODetection:
            def __init__(self, box, keypoints, conf):
                self.bbox = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
                kp = keypoints.xy[0].cpu().numpy()     # keypoints
                self.nose_tip = kp[0] if len(kp) > 0 else None
                self.left_ear = kp[1] if len(kp) > 1 else None  
                self.right_ear = kp[2] if len(kp) > 2 else None
                self.confidence = float(conf[0])
        
        for i, (box, kp, conf) in enumerate(zip(result.boxes, result.keypoints, result.boxes.conf)):
            detection = YOLODetection(box, kp, conf)
            frame = draw_head_pose_from_object(frame, detection)
```

#### Processing Video Files
```python
import cv2
from pathlib import Path
from lizard_tracking.utils.draw_utils import draw_head_pose_from_object

def process_video_with_labels(video_path, model, output_dir):
    """Process video and save frames with pose overlays"""
    cap = cv2.VideoCapture(str(video_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        detection = model.predict(frame)
        
        # Draw overlay
        if detection:
            labeled_frame = draw_head_pose_from_object(frame, detection)
        else:
            labeled_frame = draw_no_detection(frame)
        
        # Save frame
        output_path = output_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(output_path), labeled_frame)
        frame_count += 1
    
    cap.release()
```

### Color Scheme

- **Green** (`(0, 255, 0)`): Bounding boxes, confidence text
- **Red** (`(0, 0, 255)`): Nose tip
- **Blue** (`(255, 0, 0)`): Ear keypoints  
- **Yellow** (`(0, 255, 255)`): Head direction line, trajectory
- **White** (`(255, 255, 255)`): Text overlays
- **Black** (`(0, 0, 0)`): Text outlines for readability

### Error Handling

The drawing functions are designed to be robust:
- Missing attributes are handled gracefully
- Invalid coordinates are skipped
- Functions return the original frame if errors occur
- No exceptions are raised for missing pose data

### Performance Notes

- All drawing operations use OpenCV with anti-aliasing (`cv2.LINE_AA`)
- Functions modify frames in-place for efficiency
- Coordinate conversion to integers handled automatically
- Optimized for real-time video processing

## Other Modules

### Behavioral Analysis
Located in `behavioral_analysis/` - provides trajectory analysis and behavioral event detection.

### Core Tracking
Located in `lizard_tracking/core/` - core tracking algorithms and data structures.

## Installation

No separate installation needed. Add the lib directory to your Python path:

```python
import sys
sys.path.append('path/to/lib')
```

## Dependencies

- OpenCV (`cv2`)
- NumPy 
- Python 3.7+