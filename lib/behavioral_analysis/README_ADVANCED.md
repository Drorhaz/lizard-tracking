# Advanced Behavioral Analysis System

## Overview

This advanced trajectory analysis system provides **arena-mapped behavioral instructions** with sophisticated movement classification, head-vs-body motion detection, and interactive Plotly visualizations.

## ✅ Status: **FULLY TESTED & WORKING**

```
✅ Processed 150 frames
📊 Generated 150 instructions
📈 Collected 30 plot points
📊 Trajectory CSV saved
📊 Events CSV saved
📊 Interactive HTML plot created
```

---

## Key Features

### 1. **Arena Mapping with Target Line**
- Define arena orientation using a **target line** (`left`, `right`, `top`, `bottom`)
- Automatic **RIGHTSIDE/LEFTSIDE** classification based on target line
- **Near/Middle/Far** band detection with configurable thresholds

### 2. **Sophisticated Movement Classification**
- **Phase Detection**: `approaching`, `retreating`, `resting`
- **Directional Qualifiers**: `leftward`, `rightward` (when both axes change)
- **Motion Type Detection**: `[head-only]` vs `[whole-body]`

### 3. **Instruction Grammar**

Format: `<phase>[, <x-direction>] — <band> @ <arena-side> [<motion-type>]`

**Examples:**
```
approaching, rightward — near @ RIGHTSIDE [whole-body]
retreating, leftward — far @ LEFTSIDE [head-only]
resting — middle @ RIGHTSIDE
```

### 4. **Interactive Plotly Visualizations**
- Time-colored nose trajectory arrows
- Arena boundaries and target line visualization
- Near/middle/far band overlays
- Hover tooltips with frame information
- Exported as standalone HTML (viewable in any browser)

---

## Configuration

### `AdvancedBehaviorConfig`

```python
from behavioral_analysis import AdvancedBehaviorConfig

config = AdvancedBehaviorConfig(
    # 1) Target line (arena orientation)
    target_line='right',  # 'left' | 'right' | 'top' | 'bottom'
    
    # 2) Distance bands (normalized [0,1])
    near_max=0.20,      # ≤ 0.20 → near
    middle_max=0.30,    # 0.20-0.30 → middle (buffer zone)
                        # > 0.30 → far
    
    # 3) Motion thresholds
    advance_threshold=0.002,   # Δd < -0.002 → approaching
    retreat_threshold=0.002,   # Δd > +0.002 → retreating
    
    # 4) Directional thresholds (fraction of frame size)
    x_dir_thresh_norm=0.01,    # X-axis movement threshold
    y_dir_thresh_norm=0.01,    # Y-axis movement threshold
    
    # 5) Head vs body movement (fraction of frame diagonal)
    head_only_thresh_norm=0.005,   # Head wiggle threshold
    body_move_thresh_norm=0.010,   # Locomotion threshold
    
    # 6) Missing detection handling
    lookback_window=5,   # Frames to look back for valid detection
    
    # 7) Plotting
    arrow_length_norm=0.05,        # Arrow length for heading
    plot_colorscale='Viridis',     # Time color encoding
    plot_every_n_frames=5,         # Subsample for cleaner plot
)
```

---

## Usage

### Basic Example

```python
from behavioral_analysis import (
    AdvancedBehaviorConfig,
    AdvancedBehavioralDetector,
    create_nose_heading_map,
    save_trajectory_csv,
    save_events_csv
)

# 1. Create configuration
config = AdvancedBehaviorConfig(target_line='right')

# 2. Initialize detector
detector = AdvancedBehavioralDetector(
    config=config,
    frame_width=800,
    frame_height=600,
    fps=30.0
)

# 3. Process frames
for frame_idx in range(num_frames):
    # Get detection data (from YOLO or other source)
    nose = (x, y) or None
    ear_left = (x, y) or None
    ear_right = (x, y) or None
    bbox = (x1, y1, x2, y2) or None
    
    # Process frame
    instruction = detector.process_frame(
        frame_idx=frame_idx,
        nose=nose,
        ear_left=ear_left,
        ear_right=ear_right,
        bbox=bbox
    )
    
    if instruction:
        print(f"Frame {frame_idx}: {instruction.instruction}")

# 4. Save outputs
save_trajectory_csv(detector.get_plot_data(), 'trajectory.csv')
save_events_csv(detector.get_instructions_csv_format(), 'events.csv')
create_nose_heading_map(
    plot_data=detector.get_plot_data(),
    video_name='My Video',
    output_path=Path('nose_heading.html'),
    config=config,
    frame_width=800,
    frame_height=600
)
```

---

## Target Line Mapping

The **target line** defines arena orientation:

| Target Line | Direction | RIGHTSIDE         | LEFTSIDE         |
|-------------|-----------|-------------------|------------------|
| `right`     | →         | Bottom of screen  | Top of screen    |
| `left`      | ←         | Top of screen     | Bottom of screen |
| `top`       | ↑         | Right of screen   | Left of screen   |
| `bottom`    | ↓         | Left of screen    | Right of screen  |

**Intuition**: Stand on the target line facing toward it from the arena. RIGHTSIDE is on your right hand.

---

## Output Files

### 1. **Trajectory CSV** (`trajectory.csv`)
```csv
frame_idx,x_norm,y_norm,head_angle_deg,dist_to_target_norm
0,0.125,0.500,45.2,0.875
5,0.150,0.505,46.8,0.850
...
```

### 2. **Events CSV** (`events.csv`)
```csv
ts_ms,frame_idx,instruction,meta_json
0.0,0,"resting — far @ LEFTSIDE","{\"phase\":\"resting\",...}"
333.3,10,"approaching — middle @ RIGHTSIDE","{\"phase\":\"approaching\",...}"
...
```

### 3. **Interactive HTML Plot** (`nose_heading.html`)
- Time-colored trajectory with heading arrows
- Hover tooltips showing frame details
- Arena boundaries and band markers
- Target line visualization
- Fully interactive (zoom, pan, hover)

---

## Movement Classification Details

### Phase Detection (Approach/Retreat/Rest)
- Computed from **change in distance** to target line: `Δd = d(t) - d(t-1)`
- **approaching**: `Δd < -advance_threshold` (getting closer)
- **retreating**: `Δd > +retreat_threshold` (moving away)
- **resting**: Otherwise (stationary or parallel movement)

### Directional Qualifiers (Leftward/Rightward)
Applied **only when both axes change** and phase is `approaching` or `retreating`:
- `rightward` if `Δx > 0`
- `leftward` if `Δx < 0`

**Special cases:**
- **X-only movement**: No directional suffix (natural horizontal approach/retreat)
- **Y-only movement**: Label as `resting` (vertical displacement treated as stationary)

### Motion Type (Head-Only vs Whole-Body)
- **head-only**: Body stationary (`body_disp < body_thresh`) but nose moving (`head_disp ≥ head_thresh`)
- **whole-body**: Body displacement exceeds threshold (`body_disp ≥ body_thresh`)
- **unknown**: Ambiguous or insufficient displacement

---

## Missing Detection Handling

When a frame has no detection:
1. **Look back** up to `lookback_window` frames (default: 5)
2. Use most recent valid detection for position/angle
3. Continue band & arena side classification
4. If no valid detection in window → skip instruction generation

This maintains **temporal continuity** while handling brief detection failures.

---

## Integration with YOLO Pose Detection

```python
# In detection loop
results = model(frame)

if results and len(results) > 0:
    result = results[0]
    boxes = result.boxes
    keypoints = result.keypoints if hasattr(result, 'keypoints') else None
    
    # Extract data
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bbox = (float(x1), float(y1), float(x2), float(y2))
        
        nose, ear_left, ear_right = None, None, None
        if keypoints is not None and len(keypoints.xy) > 0:
            kpts = keypoints.xy[0].cpu().numpy()
            if len(kpts) >= 3:
                # Adjust indices based on your keypoint mapping
                nose = (float(kpts[2][0]), float(kpts[2][1]))
                ear_left = (float(kpts[0][0]), float(kpts[0][1]))
                ear_right = (float(kpts[1][0]), float(kpts[1][1]))
        
        # Process with advanced detector
        instruction = detector.process_frame(
            frame_idx=frame_idx,
            nose=nose,
            ear_left=ear_left,
            ear_right=ear_right,
            bbox=bbox
        )
```

---

## Testing

Run the included test suite:

```bash
cd lib/behavioral_analysis
python test_advanced.py
```

Expected output:
```
🧪 Testing Advanced Behavioral Detector
📍 Simulating trajectory: approaching from far-left to near-right

Frame   0: resting — far @ RIGHTSIDE
Frame  10: approaching — far @ RIGHTSIDE
...
Frame 140: approaching — near @ RIGHTSIDE

✅ Processed 150 frames
📊 Generated 150 instructions
📈 Collected 30 plot points
🎉 All tests passed!
```

---

## Dependencies

- **numpy** - Array operations
- **plotly** - Interactive visualizations (optional but recommended)

Install Plotly:
```bash
pip install plotly
```

---

## API Reference

### `AdvancedBehavioralDetector`

**Methods:**
- `process_frame(frame_idx, nose, ear_left, ear_right, bbox)` → `BehavioralInstruction | None`
- `get_plot_data()` → `List[dict]`
- `get_instructions_csv_format()` → `List[Tuple]`

**Attributes:**
- `instructions`: List of all generated instructions
- `plot_data`: Trajectory data for visualization

### `BehavioralInstruction`

**Attributes:**
- `frame_idx`: Frame number
- `timestamp_ms`: Timestamp in milliseconds
- `video_seconds`: Video time in seconds
- `phase`: 'approaching' | 'retreating' | 'resting'
- `band`: 'near' | 'middle' | 'far'
- `arena_side`: 'RIGHTSIDE' | 'LEFTSIDE'
- `motion_type`: 'head-only' | 'whole-body' | None
- `x_direction`: 'leftward' | 'rightward' | None
- `instruction`: Full formatted string

---

## Performance Notes

- **Efficient**: O(1) per-frame processing
- **Memory**: Bounded by `lookback_window` (default: 5 frames)
- **Real-time capable**: Processes frames faster than video FPS

---

## Future Enhancements

- [ ] Velocity-based smoothing for phase transitions
- [ ] Circular statistics for head angle analysis
- [ ] Dwell time analysis per band
- [ ] Heatmap overlays for position density
- [ ] Multi-animal tracking support
- [ ] Export to neurophysiology formats (NWB, etc.)

---

## License

Part of the lizard-tracking behavioral analysis library.

---

## Contact & Support

For issues or questions, please refer to the main repository documentation.
