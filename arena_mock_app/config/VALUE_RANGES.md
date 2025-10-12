# ⚠️ IMPORTANT: Configuration Value Ranges

## Common Configuration Mistakes

### ❌ WRONG Values (will cause crashes or strange behavior):
```env
ADVANCE_THRESHOLD=300        # TOO BIG! Should be ~0.002
RETREAT_THRESHOLD=350        # TOO BIG! Should be ~0.002
CONFIDENCE_THRESHOLD=90      # Should be 0.0-1.0, not percentage!
DETECTION_IOU=50             # Should be 0.0-1.0, not percentage!
```

### ✅ CORRECT Values:
```env
ADVANCE_THRESHOLD=0.002      # Small decimal (normalized movement)
RETREAT_THRESHOLD=0.002      # Small decimal (normalized movement)
CONFIDENCE_THRESHOLD=0.2     # Decimal between 0.0 and 1.0
DETECTION_IOU=0.5            # Decimal between 0.0 and 1.0
```

## Value Ranges by Parameter

### Thresholds (0.0 to 1.0)
- `CONFIDENCE_THRESHOLD` → 0.0 to 1.0 (typically 0.2 to 0.6)
- `DETECTION_IOU` → 0.0 to 1.0 (typically 0.4 to 0.6)
- `NEAR_MAX` → 0.0 to 1.0 (fraction of frame width)
- `MIDDLE_MAX` → 0.0 to 1.0 (fraction of frame width)

### Normalized Movement (0.001 to 0.05)
These are **very small** because they're normalized to frame dimensions:
- `ADVANCE_THRESHOLD` → 0.001 to 0.01 (typically 0.002)
- `RETREAT_THRESHOLD` → 0.001 to 0.01 (typically 0.002)
- `X_DIR_THRESH_NORM` → 0.005 to 0.02 (typically 0.01)
- `Y_DIR_THRESH_NORM` → 0.005 to 0.02 (typically 0.01)
- `HEAD_ONLY_THRESH_NORM` → 0.001 to 0.01 (typically 0.005)
- `BODY_MOVE_THRESH_NORM` → 0.005 to 0.02 (typically 0.010)

### Pixel-based Threshold (large values OK)
- `STOP_THRESHOLD` → 100.0 to 500.0 pixels (typically 300.0)

### Integer Values
- `PROCESSING_FPS` → 5 to 60 (frames per second)
- `STREAM_FPS` → 10 to 30 (browser display FPS)
- `DETECTION_IMGSZ` → 320, 640, 1280 (YOLO image size)
- `JPEG_QUALITY` → 1 to 100 (typically 70-90)
- `LOOKBACK_WINDOW` → 3 to 10 frames
- `MIN_MOVING_FRAMES` → 2 to 10 frames
- `MIN_STATIONARY_FRAMES` → 2 to 10 frames
- `SERVER_PORT` → 1024 to 65535

### Boolean Values
- `VERBOSE` → true or false
- `SERVER_DEBUG` → true or false

## Why Small Movement Thresholds?

**Movement thresholds are NORMALIZED to frame dimensions:**

Example with 800x600 frame:
- `ADVANCE_THRESHOLD=0.002` means 0.2% of frame width
- 0.002 × 800 = **1.6 pixels** of movement toward target
- This is very sensitive!

If you set `ADVANCE_THRESHOLD=300`:
- That would mean 300% of frame width = 2400 pixels!
- The lizard would have to teleport to trigger it!

## Quick Tuning Guide

### More Sensitive (detect smaller movements):
```env
ADVANCE_THRESHOLD=0.001       # Half the default
RETREAT_THRESHOLD=0.001
HEAD_ONLY_THRESH_NORM=0.003
```

### Less Sensitive (only detect larger movements):
```env
ADVANCE_THRESHOLD=0.005       # 2.5x the default
RETREAT_THRESHOLD=0.005
HEAD_ONLY_THRESH_NORM=0.010
```

### Production Settings (balanced):
```env
CONFIDENCE_THRESHOLD=0.3
PROCESSING_FPS=30
DETECTION_IOU=0.5
ADVANCE_THRESHOLD=0.002
RETREAT_THRESHOLD=0.002
VERBOSE=false
```

### Debug Settings (detailed output):
```env
CONFIDENCE_THRESHOLD=0.2
PROCESSING_FPS=10
DETECTION_IOU=0.5
ADVANCE_THRESHOLD=0.002
RETREAT_THRESHOLD=0.002
VERBOSE=true
```

## Testing Your Config

Run this to validate your configuration loads correctly:
```bash
cd arena_mock_app
python -c "from api import CONFIG; CONFIG.print_config()"
```

You should see all your values printed clearly. If you get errors or values look wrong, check your `.env` file syntax.
