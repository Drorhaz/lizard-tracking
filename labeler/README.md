# Labeler - Interactive YOLO Pose Label Editor

A web-based application for fast previewing, quality control, and interactive editing of YOLO pose detection labels. This tool enables efficient review and correction of lizard head pose keypoint annotations with intuitive drag-and-drop editing capabilities.

## 🎯 Overview

The Labeler is a Flask-based web application specifically designed for:
- **Fast Label Preview**: Visualize YOLO pose labels overlaid on images
- **Interactive Editing**: Drag keypoints and adjust bounding boxes with mouse
- **Quality Control**: Mark images as OK or Skip for dataset curation
- **Batch Processing**: Navigate through entire datasets efficiently
- **Label Validation**: Ensure proper keypoint visibility and positioning

## ✨ Key Features

### Interactive Keypoint Editing
- **Drag & Drop**: Click and drag keypoints (nose, left ear, right ear) to adjust positions
- **Visibility Toggle**: Hold `T` and click on keypoints to toggle visibility states (0=not visible, 1=occluded, 2=visible)
- **Double-Click Seeding**: Double-click anywhere to automatically seed nose and ear keypoints at cursor position
- **Visual Feedback**: Real-time preview of changes with color-coded keypoints

### Bounding Box Management
- **Interactive Resize**: Toggle BBox edit mode to drag corners and resize boxes
- **Position Adjustment**: Drag inside bounding box to move entire region
- **Create New**: Click-drag in empty areas to create new bounding boxes
- **Manual Input**: Precise numeric control via input fields (normalized coordinates)
- **Smart Keypoint Mapping**: Automatically adjusts keypoints when resizing bounding boxes

### Dataset Navigation
- **Sequential Browse**: Navigate through images with Prev/Next buttons
- **Direct Jump**: Go to specific images by index or filename
- **Progress Tracking**: Real-time counter showing reviewed vs remaining images
- **Dataset Switching**: Change image/label directories and splits on-the-fly

### Quality Control Workflow
- **Mark as OK**: Approve labels for training data
- **Mark as Skip**: Flag problematic images for exclusion
- **Persistent Logs**: Maintains separate logs for approved and skipped images
- **Batch Review**: Efficiently process large datasets with keyboard shortcuts

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install flask opencv-python numpy pathlib waitress
```

### Quick Start
1. **Launch the application**:
   ```bash
   cd labeler/
   python label_qc_web.py
   ```

2. **Access the web interface**:
   - Open browser to `http://localhost:7860` (or displayed port)
   - The application will auto-find a free port if 7860 is occupied

3. **Configure dataset paths**:
   - Set Images dir: Path to your image dataset
   - Set Labels dir: Path to corresponding YOLO label files
   - Set Split: Dataset split name (train/val/test)

### Command Line Options
The application accepts dataset configuration via URL parameters:
```
http://localhost:7860/?img_dir=/path/to/images&lbl_dir=/path/to/labels&split=train
```

## 📋 Usage Guide

### Basic Workflow

1. **Load Dataset**:
   - Enter image and label directory paths
   - Specify dataset split (train, val, test)
   - Click "Load" to initialize

2. **Review Labels**:
   - Images display with overlay of current keypoints and bounding boxes
   - Green dots indicate visible keypoints, gray for occluded/invisible
   - Bounding box shown in green outline

3. **Edit Keypoints**:
   - **Drag**: Click and drag any keypoint to reposition
   - **Toggle Visibility**: Hold `T` key and click keypoint to cycle visibility (0→1→2→0)
   - **Quick Seed**: Double-click to place nose + ears at cursor location
   - **Manual Entry**: Use numeric input fields for precise coordinates

4. **Edit Bounding Boxes**:
   - Click "BBox Edit: Off" to enable edit mode (turns green when active)
   - **Move**: Drag inside box to reposition
   - **Resize**: Drag corners to adjust size
   - **Create**: Click-drag in empty area to create new box
   - **Delete**: Set width/height to 0 in manual fields

5. **Quality Control**:
   - **OK**: Mark image as approved for training
   - **Skip**: Flag image as problematic/excluded
   - **Navigate**: Use Prev/Next or direct image selection

### Keyboard Shortcuts
- `T` + Click: Toggle keypoint visibility
- Arrow navigation via Prev/Next buttons
- Direct image jumping via index input

### Advanced Features

#### Coordinate Systems
- **Display**: All editing happens in pixel coordinates for intuitive interaction
- **Storage**: Labels saved in YOLO format (normalized 0-1 coordinates)
- **Conversion**: Automatic conversion between display and storage formats

#### Label Format
YOLO pose format with 3 keypoints (nose, left ear, right ear):
```
class_id center_x center_y width height nose_x nose_y nose_v left_x left_y left_v right_x right_y right_v
```
Where visibility values: 0=not visible, 1=occluded, 2=visible

#### Dataset Organization
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 🔧 Configuration

### Dataset Path Configuration

The labeler needs to know where your images and labels are located. There are several ways to configure these paths:

#### Method 1: Web Interface (Recommended)
Use the form at the top of the web interface:
1. **Images dir**: Enter the full path to your image folder (e.g., `/path/to/your/dataset/images/train`)
2. **Labels dir**: Enter the full path to your label folder (e.g., `/path/to/your/dataset/labels/train`) 
3. **Split**: Specify dataset split name (`train`, `val`, `test`, or custom)
4. Click **Load** to apply changes

#### Method 2: URL Parameters
Configure paths directly in the browser URL:
```
http://localhost:7860/?img_dir=/absolute/path/to/images&lbl_dir=/absolute/path/to/labels&split=train
```

#### Method 3: Modify Source Code Defaults
Edit the default paths in `label_qc_web.py` (lines 29-31):
```python
DEFAULT_SPLIT = "train"  # Change default split
DEFAULT_IMAGES_ROOT = Path(f"your/custom/path/images/{DEFAULT_SPLIT}").resolve()
DEFAULT_LABELS_ROOT = Path(f"your/custom/path/labels/{DEFAULT_SPLIT}").resolve()
```

### Default Paths
The application includes these sensible defaults (relative to labeler directory):
- **Images**: `dataset/images/train/`
- **Labels**: `dataset/labels/train/`
- **Output**: `output/` (for logs and previews)

### Expected Dataset Structure
```
your-dataset-folder/
├── images/
│   ├── train/          # Training images (.jpg, .png, .bmp, .tif)
│   ├── val/            # Validation images
│   └── test/           # Test images
└── labels/
    ├── train/          # Training labels (.txt files)
    ├── val/            # Validation labels  
    └── test/           # Test labels
```

### Path Requirements
- **Absolute paths recommended**: Use full paths to avoid confusion
- **Matching structure**: Label files must have same names as images but with `.txt` extension
- **Supported formats**: Images (.jpg, .jpeg, .png, .bmp, .tif, .tiff)
- **YOLO format**: Labels must be in YOLO pose format (17 values per line)

### Environment Variables
- `PORT`: Override default port (7860)

### Canvas Settings
- **Max Width**: 800px (adjustable in code)
- **Handle Radius**: 6px for keypoint interaction
- **Auto-scaling**: Images scaled to fit canvas while preserving aspect ratio

## 📁 Output Files

### Quality Control Logs
- `output/data/valid/{split}_ok.txt`: List of approved images
- `output/data/skip/{split}_skip.txt`: List of skipped images

### Preview Images
- `output/preview/{split}/`: Rendered images with label overlays
- Useful for visual verification and documentation

## 🎨 Interface Overview

### Main Canvas
- **Image Display**: Scaled image with label overlays
- **Interactive Elements**: Draggable keypoints and bounding boxes
- **Visual Indicators**: Color-coded visibility states and edit modes

### Control Panel
- **Navigation**: Sequential browsing and direct image access
- **Actions**: Quality control marking (OK/Skip)
- **Edit Modes**: Toggle between keypoint and bounding box editing

### Manual Input Panel
- **Bounding Box**: Precise coordinate entry (normalized)
- **Keypoints**: Individual coordinate and visibility control
- **Label Info**: Current label file path and contents

## 🛠️ Technical Details

### Architecture
- **Backend**: Flask web application with RESTful API
- **Frontend**: HTML5 Canvas with vanilla JavaScript
- **Storage**: Direct YOLO label file manipulation
- **Deployment**: Built-in Waitress WSGI server for production

### API Endpoints
- `GET /`: Main interface
- `GET /raw/<int:i>.png`: Serve processed images
- `GET /state/<int:i>`: Get current label state
- `POST /hit_kp`: Keypoint hit detection
- `POST /hit_bbox`: Bounding box hit detection  
- `POST /action`: Label editing actions

### Performance
- **Image Loading**: Efficient OpenCV-based processing
- **Real-time Updates**: Immediate visual feedback for all edits
- **Memory Management**: Optimized for large datasets
- **Auto-port Discovery**: Handles multiple concurrent instances

## 🔍 Troubleshooting

### Common Issues

**No images found**:
- Verify image directory path contains supported formats (.jpg, .png, .bmp, .tif)
- Check that paths are accessible and properly formatted
- Ensure images directory structure matches expected layout

**Labels not loading**:
- Confirm label directory contains corresponding .txt files
- Verify YOLO format compliance (17 values per line for pose)
- Check file permissions and path accessibility

**Interface not responding**:
- Refresh browser page to reset state
- Check browser console for JavaScript errors
- Verify Flask application is running and accessible

**Port conflicts**:
- Application auto-finds free ports starting from 7860
- Set custom port via `PORT` environment variable
- Check firewall settings for port accessibility

### Performance Tips
- Use SSD storage for large datasets to improve loading times
- Close unused browser tabs to free memory for canvas operations
- Process datasets in smaller batches for memory efficiency
- Enable hardware acceleration in browser for smoother canvas rendering

## 🤝 Integration

### Pipeline Integration
The Labeler integrates seamlessly with the broader lizard-tracking pipeline:
- **Input**: YOLO pose labels from training/inference
- **Output**: Quality-controlled labels ready for model training
- **Workflow**: Fits between label generation and model training phases

### Batch Processing
Can be integrated into automated workflows:
- Programmatic dataset loading via URL parameters
- Quality control logs for automated filtering
- Export capabilities for processed datasets

---

*Part of the lizard-tracking project for bearded dragon head pose estimation and behavioral analysis.*