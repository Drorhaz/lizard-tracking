# Behavioral Analysis Library

A standalone behavioral analysis library for real-time animal behavior detection and analysis. This library can integrate with any pose detection system and provides comprehensive behavioral event detection, metrics calculation, and data export capabilities.

## Features

### 🎯 Real-time Event Detection
- **Approach Detection**: Detect when animal moves towards reference point
- **Retreat Detection**: Detect when animal moves away from reference point  
- **Stop/Movement Detection**: Detect stationary periods and movement onset
- **Configurable Thresholds**: Pixel-based distance and speed thresholds
- **Hysteresis**: Prevent false triggering with configurable hysteresis zones

### 📊 Live Metrics
- Current position, speed, and direction
- Distance from reference points
- Total distance traveled
- Average speeds and trajectory analysis
- Direction stability and movement patterns
- Real-time trajectory reconstruction

### 📁 Data Export
- CSV export for trajectory data and events
- JSON export with full metadata
- Analysis software compatibility (EthoVision, BORIS)
- Session summaries with comprehensive statistics
- Raw data export for custom analysis

### 📈 Trajectory Analysis
- Interactive Plotly visualizations
- Speed profiles and movement patterns
- Path efficiency and tortuosity metrics
- Direction stability analysis
- Smoothing and subsampling tools

## Installation

```bash
# Install required dependencies
pip install numpy pandas plotly matplotlib

# The library is standalone - simply import from the lib/ directory
```

## Quick Start

```python
from lib.behavioral_analysis import BehaviorDetector, BehaviorConfig

# Configure behavior detection parameters
config = BehaviorConfig(
    approach_threshold=100,      # pixels
    retreat_threshold=300,       # pixels  
    stop_threshold=5,           # pixels/frame
    reference_point=(320, 240), # center of frame
    detect_approach=True,
    detect_retreat=True,
    detect_stop=True
)

# Create detector
detector = BehaviorDetector(config)

# Process pose coordinates frame by frame
pose_coordinates = [(x1, y1), (x2, y2), ...]  # from your pose detection
for frame_num, (x, y) in enumerate(pose_coordinates):
    events = detector.process_frame((x, y), frame_num)
    
    # Handle detected events
    for event in events:
        print(f"Frame {frame_num}: {event.event_type.value} at ({x:.1f}, {y:.1f})")
    
    # Get live metrics
    metrics = detector.metrics.to_dict()
    print(f"Current speed: {metrics['current_speed_px_per_frame']:.1f} px/frame")
```

## Web Interface Integration

The library is designed to integrate easily with web interfaces like the pose-head web application:

```python
# Example web interface integration
from lib.behavioral_analysis import BehaviorDetector, BehaviorConfig, BehaviorExporter

class BehaviorAnalysisWebInterface:
    def __init__(self):
        self.detector = None
        self.exporter = BehaviorExporter()
    
    def configure_behavior_detection(self, form_data):
        """Configure from web form checkboxes and sliders"""
        config = BehaviorConfig(
            detect_approach=form_data.get('enable_approach', False),
            detect_retreat=form_data.get('enable_retreat', False), 
            detect_stop=form_data.get('enable_stop', False),
            approach_threshold=float(form_data.get('approach_threshold', 100)),
            retreat_threshold=float(form_data.get('retreat_threshold', 300)),
            stop_threshold=float(form_data.get('stop_threshold', 5)),
            reference_point=(
                float(form_data.get('ref_x', 320)), 
                float(form_data.get('ref_y', 240))
            )
        )
        self.detector = BehaviorDetector(config)
    
    def process_frame_with_behavior(self, pose_result, frame_num):
        """Process pose detection result with behavioral analysis"""
        if self.detector and pose_result.get('keypoints'):
            # Extract head position (or use any keypoint)
            head_pos = pose_result['keypoints'][0]  # assuming head is first keypoint
            events = self.detector.process_frame(head_pos, frame_num)
            
            return {
                'pose_result': pose_result,
                'behavioral_events': [e.to_dict() for e in events],
                'live_metrics': self.detector.metrics.to_dict(),
                'current_state': self.detector.get_current_state()
            }
        return {'pose_result': pose_result}
    
    def export_session_data(self):
        """Export complete behavioral session"""
        if self.detector:
            events = self.detector.event_bus.get_all_events()
            return self.exporter.export_session_summary(
                events, 
                self.detector.metrics,
                self.detector.config.__dict__
            )
```

## Configuration Options

### BehaviorConfig Parameters

```python
@dataclass
class BehaviorConfig:
    # Event detection toggles
    detect_approach: bool = True
    detect_retreat: bool = True  
    detect_stop: bool = True
    
    # Distance thresholds (pixels)
    approach_threshold: float = 100.0
    retreat_threshold: float = 300.0
    hysteresis_px: float = 10.0
    
    # Speed thresholds
    stop_threshold: float = 5.0  # pixels/frame
    
    # Temporal filters
    min_stationary_frames: int = 10
    min_moving_frames: int = 5
    
    # Reference point for approach/retreat
    reference_point: Optional[Tuple[float, float]] = None
```

## Event Types

The library detects these behavioral events:

- `APPROACH_START`: Animal begins moving towards reference point
- `APPROACH_END`: Animal stops approaching reference point  
- `RETREAT_START`: Animal begins moving away from reference point
- `RETREAT_END`: Animal stops retreating from reference point
- `STOP_START`: Animal becomes stationary
- `STOP_END`: Animal starts moving after being stationary

## Trajectory Analysis

```python
from lib.behavioral_analysis import TrajectoryAnalyzer

# Create analyzer from coordinates
coordinates = [(x1, y1), (x2, y2), ...]
analyzer = TrajectoryAnalyzer.from_coordinates(coordinates)

# Calculate comprehensive metrics
metrics = analyzer.calculate_metrics(fps=30.0)
print(f"Path efficiency: {metrics['path_efficiency']:.3f}")
print(f"Total distance: {metrics['total_distance']:.1f} pixels")

# Create interactive visualization
fig = analyzer.plot_trajectory_plotly(title="Animal Movement")
fig.show()

# Export trajectory data
analyzer.export_trajectory("trajectory.csv", format_type="csv")
```

## Data Export

```python
from lib.behavioral_analysis import BehaviorExporter

exporter = BehaviorExporter(output_dir="behavioral_data")

# Export events to CSV
events_file = exporter.export_events_csv(detector.event_bus.get_all_events())

# Export trajectory
trajectory_file = exporter.export_trajectory_csv(detector.metrics)

# Export comprehensive session summary
summary_file = exporter.export_session_summary(
    detector.event_bus.get_all_events(),
    detector.metrics, 
    detector.config.__dict__
)

# Export for analysis software
ethovision_file = exporter.export_for_analysis_software(
    detector.event_bus.get_all_events(),
    detector.metrics,
    format_type="ethovision"
)
```

## Architecture

The library is designed with modularity and integration in mind:

- **`BehaviorConfig`**: Configuration and parameter management
- **`BehaviorDetector`**: Core detection logic and state management  
- **`EventBus`**: Event publishing and subscription system
- **`LiveMetrics`**: Real-time metrics calculation
- **`BehaviorExporter`**: Data export in multiple formats
- **`TrajectoryAnalyzer`**: Trajectory analysis and visualization

## Dependencies

- **numpy**: Numerical calculations
- **pandas**: Data export (CSV)
- **plotly**: Interactive visualizations (optional)
- **matplotlib**: Static plots (optional)

## Integration with Existing Tools

This library consolidates and modernizes functionality from existing tools:

- `tools/pose_head_pipeline.py`: Legacy pipeline with similar functionality
- `tools/reconstruct_trajectory.py`: Trajectory reconstruction
- `tools/plot_arena_arrows_plotly.py`: Visualization tools

The new library provides a cleaner API, better modularity, and easier integration with web interfaces.

## Example Applications

1. **Real-time behavioral monitoring** during experiments
2. **Post-hoc analysis** of recorded video data  
3. **Web interface integration** for live experiment control
4. **Research pipeline automation** with configurable parameters
5. **Export to analysis software** (EthoVision, BORIS, etc.)

## Future Enhancements

- Additional behavioral event types (circling, rearing, etc.)
- Machine learning-based behavior classification
- Multi-animal tracking support
- 3D pose analysis integration
- Real-time alerts and notifications