# Head Angle Tracking and No-Detection Handling Improvements

## Summary of Changes

I've successfully implemented the requested features for the lizard head pose detection application:

### 1. Head Angle Calculation and Kalman Filtering

**New Features Added:**
- **AngleKalmanFilter Class**: Implements a Kalman filter for smoothing angle measurements with configurable process and measurement noise
- **calculate_head_angle_to_target()**: Calculates the angle between the head direction and the target screen location
- **get_target_line_position()**: Helper function to get numerical position of target lines

**Key Capabilities:**
- Calculates angle from head pose keypoints (nose, left ear, right ear) to the target screen
- Supports all target line positions: 'right', 'left', 'top', 'bottom'
- Uses Kalman filtering for smooth angle tracking with angular velocity estimation
- Handles angle wraparound and discontinuities properly (-180° to 180° range)
- Robust to missing keypoints (works with nose + one ear, or nose + both ears)

### 2. No-Detection Handling Improvement

**Enhanced No-Detection Display:**
- When no head is detected, the system now displays the last detected behavioral event
- Added overlay text showing "NO DETECTION - Last: [last detected event]" in blue color
- Consistent with the requested pattern from `_handle_detection_output` method

### 3. Data Logging and Visualization

**Extended CSV Output:**
- Added three new columns to trajectory.csv:
  - `head_angle_raw`: Raw angle measurement in degrees
  - `head_angle_smoothed`: Kalman-filtered angle in degrees  
  - `angular_velocity`: Estimated angular velocity in degrees per frame

**Visual Overlays:**
- Real-time angle display on video feed: "Angle to [target]: [smoothed_angle]°"
- Direction indicator arrow in top-right corner showing head orientation relative to target
- Color-coded overlays (cyan for angle info, blue for no detection)

**Web Interface Updates:**
- Added angle tracking status section showing:
  - Target line position
  - Current raw angle
  - Smoothed angle value
  - Real-time updates every second

### 4. Technical Implementation Details

**Angle Calculation Method:**
- Uses head direction vector computed from ear-to-ear line (perpendicular to ears)
- Fallback to nose-to-ear vectors when only one ear is available
- Calculates angle between head direction and target direction using dot/cross products
- Returns 0° when pointing directly at target, ±180° when pointing away

**Kalman Filter Configuration:**
- Process noise: 1e-4 (low - assumes smooth head movement)
- Measurement noise: 1e-1 (higher - accounts for keypoint detection noise)
- State vector: [angle, angular_velocity]
- Prediction model: constant velocity

**Target Line Mapping:**
- 'right': Rightmost vertical line (x = frame_width)
- 'left': Leftmost vertical line (x = 0)  
- 'top': Topmost horizontal line (y = 0)
- 'bottom': Bottommost horizontal line (y = frame_height)

### 5. Configuration Integration

The angle tracking integrates seamlessly with existing configuration:
- Uses existing `TARGET_LINE` setting from config/.env
- Automatically initializes target line position when video loads
- Angle information included in status API for web interface
- All angle data saved to output files alongside existing metrics

### 6. Error Handling and Robustness

**Graceful Degradation:**
- Handles missing keypoints (nose or ears) gracefully
- Continues tracking when only partial keypoint data available
- Falls back to displaying last known behavioral event when detection fails
- Maintains angle tracking state across temporary detection failures

**Data Validation:**
- Validates keypoint coordinates (> 0 checks)
- Handles NaN values in CSV output appropriately
- Normalizes angles to standard [-180°, 180°] range
- Prevents division by zero in angle calculations

## Files Modified

1. **app.py**: Main application file with all new functionality
2. **test_angle_calculation.py**: Test file for verifying angle calculations (created)

## Usage

The application now provides rich angle tracking capabilities:

1. **Real-time angle monitoring** - See current head orientation relative to screen
2. **Smooth angle tracking** - Kalman filtering reduces noise and provides velocity estimates  
3. **Persistent behavioral display** - Always shows recent behavioral events even during detection gaps
4. **Comprehensive data logging** - Full angle history saved for offline analysis
5. **Visual feedback** - Clear on-screen indicators of head direction and angle

The angle tracking works automatically when keypoints are detected and integrates seamlessly with the existing behavioral analysis pipeline.