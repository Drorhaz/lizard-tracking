#!/usr/bin/env python3
"""
Head Pose Detection Application - Enhanced Version
=====================================
Features:
- Configurable model selection: Regular YOLO OR Embedding-enhanced YOLO
- Advanced behavioral analysis with arena mapping
- Head angle calculation with Kalman filtering
- Real-time visual overlays (angle, direction arrow)
- Comprehensive data logging and visualization
- No-detection handling with last event display
- Stable streaming with confidence-based feedback
- Embedding model gap-filling capabilities (when enabled)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Now loaded from config/.env
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
import time
import csv
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, Response, jsonify, request
import json
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv
import queue
from collections import deque
import math
import traceback

# Add the lib directory to the path for imports
import sys
import os
# Get the absolute path to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (lizard-tracking root)
parent_dir = os.path.dirname(script_dir)
# Add lib directory to path
lib_dir = os.path.join(parent_dir, 'lib')
sys.path.insert(0, lib_dir)
sys.path.insert(0, parent_dir)

print(f"🔧 Added to Python path: {lib_dir}")
print(f"🔧 Added to Python path: {parent_dir}")

# Try to import the embedding model
try:
    from lizard_tracking.models.embedding_pose import create_embedding_enhanced_model, EmbeddingOutput
    EMBEDDING_MODEL_AVAILABLE = True
    print("✅ Embedding model imported successfully")
except ImportError as e:
    EMBEDDING_MODEL_AVAILABLE = False
    print(f"⚠️ Embedding model not available: {e}")

# Load environment variables from config/.env
config_path = os.path.join(script_dir, 'config', '.env')
if os.path.exists(config_path):
    load_dotenv(config_path)
    print(f"✅ Configuration loaded from: {config_path}")
else:
    print(f"⚠️  No .env file found at: {config_path}")
    print("   Using default values. Copy config/.env.example to config/.env")

try:
    from lizard_tracking.utils.draw_utils import draw_head_pose
    DRAW_UTILS_AVAILABLE = True
    print("✅ Draw utils imported successfully")
except ImportError as e:
    print(f"⚠️ Draw utils import failed: {e}")
    DRAW_UTILS_AVAILABLE = False

# Import behavioral analysis for trajectory-based movement detection
try:
    from behavioral_analysis.detector import BehaviorDetector
    from behavioral_analysis.config import BehaviorConfig
    from behavioral_analysis.events import EventType
    from behavioral_analysis.metrics import LiveMetrics
    # Import advanced behavioral analysis
    from behavioral_analysis.config_advanced import AdvancedBehaviorConfig
    from behavioral_analysis.detector_advanced import AdvancedBehavioralDetector
    from behavioral_analysis.plotter import create_nose_heading_map, save_events_csv, save_trajectory_csv
    BEHAVIOR_ANALYSIS_AVAILABLE = True
    ADVANCED_BEHAVIOR_AVAILABLE = True
    print("✅ Behavioral analysis imported successfully (with LiveMetrics + Advanced)")
except ImportError as e:
    print(f"⚠️ Behavioral analysis import failed: {e}")
    BEHAVIOR_ANALYSIS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# Kalman Filter for Angle Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class AngleKalmanFilter:
    """Kalman filter for smoothing head angle measurements"""
    
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
        """
        Initialize Kalman filter for angle tracking
        
        Args:
            process_noise: Process noise variance (how much we expect the angle to change)
            measurement_noise: Measurement noise variance (how noisy our angle measurements are)
        """
        # State: [angle, angular_velocity]
        self.x = np.array([0.0, 0.0])  # Initial state (angle=0, velocity=0)
        
        # State covariance matrix (uncertainty in our state estimate)
        self.P = np.eye(2) * 1000  # Large initial uncertainty
        
        # State transition matrix (how state evolves)
        self.F = np.array([[1.0, 1.0],  # angle = angle + velocity * dt (dt=1 frame)
                           [0.0, 1.0]]) # velocity = velocity (assume constant velocity)
        
        # Measurement matrix (we only observe angle, not velocity)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance matrix
        self.Q = np.array([[process_noise, 0.0],
                           [0.0, process_noise]])
        
        # Measurement noise covariance matrix
        self.R = np.array([[measurement_noise]])
        
        self.initialized = False
    
    def update(self, measured_angle):
        """
        Update filter with new angle measurement
        
        Args:
            measured_angle: Measured angle in degrees
            
        Returns:
            smoothed_angle: Filtered angle in degrees
        """
        if not self.initialized:
            # Initialize with first measurement
            self.x[0] = measured_angle
            self.initialized = True
            return measured_angle
        
        # Handle angle wraparound (convert to 0-360 range)
        measured_angle = measured_angle % 360
        
        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Handle angle wraparound in prediction
        self.x[0] = self.x[0] % 360
        
        # Calculate innovation (difference between measurement and prediction)
        y = measured_angle - (self.H @ self.x)[0]
        
        # Handle angle wraparound in innovation (choose shortest path)
        if y > 180:
            y -= 360
        elif y < -180:
            y += 360
        
        # Update step
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ np.array([y])
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        # Handle angle wraparound in final result
        self.x[0] = self.x[0] % 360
        
        return self.x[0]

# ═══════════════════════════════════════════════════════════════════════════════
# Angle Calculation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_head_angle_to_target(nose, ear_left, ear_right, target_line, target_line_position, frame_width, frame_height):
    """
    Calculate the angle of the head relative to the target line (screen)
    
    Args:
        nose: (x, y) coordinates of nose keypoint
        ear_left: (x, y) coordinates of left ear keypoint
        ear_right: (x, y) coordinates of right ear keypoint
        target_line: 'right', 'left', 'top', or 'bottom' - location of screen
        target_line_position: numerical position of the target line
        frame_width: width of the frame
        frame_height: height of the frame
        
    Returns:
        tuple: (angle_degrees, head_direction_vector)
               angle_degrees: angle in degrees (0-360°)
               Convention: angle is measured from a vector that is
               parallel to the target line to the head vector (ears->nose).
               Therefore:
                   0°   = head vector is parallel to the target line
                   90°  = head vector is perpendicular to the target line (counterclockwise)
                   180° = head vector is parallel but opposite direction
                   270° = head vector is perpendicular to the target line (clockwise)
               head_direction_vector: (x, y) normalized vector showing head direction
    """
    if not nose or (not ear_left and not ear_right):
        return None, None
        
    # Calculate head direction vector using available keypoints
    head_direction = None
    
    if ear_left and ear_right:
        # Use both ears to determine head direction
        # The head points from the center of ears towards the nose
        ear_center_x = (ear_left[0] + ear_right[0]) / 2
        ear_center_y = (ear_left[1] + ear_right[1]) / 2
        head_direction = (nose[0] - ear_center_x, nose[1] - ear_center_y)
    elif ear_left:
        # Use nose-to-left-ear vector as approximation (from ear to nose)
        head_direction = (nose[0] - ear_left[0], nose[1] - ear_left[1])
    elif ear_right:
        # Use nose-to-right-ear vector as approximation (from ear to nose)
        head_direction = (nose[0] - ear_right[0], nose[1] - ear_right[1])
    
    if not head_direction:
        return None, None
    
    # Normalize head direction vector for consistent arrow display
    head_length = math.sqrt(head_direction[0]**2 + head_direction[1]**2)
    if head_length > 0:
        normalized_head_direction = (head_direction[0] / head_length, head_direction[1] / head_length)
    else:
        normalized_head_direction = (1, 0)  # Default to pointing right
        
    # Calculate target direction vector from nose to target line
    target_direction = None
    
    # IMPORTANT: use a vector that is PARALLEL to the target line (not the normal
    # pointing toward the target). This makes 0° correspond to the head being
    # aligned with the target line.
    if target_line in ('right', 'left'):
        # Right/left are vertical lines -> use a vertical unit vector (downwards)
        # Image coordinates: +Y is downward, so (0, 1) is a canonical vertical vector.
        target_direction = (0, 1)
    elif target_line in ('top', 'bottom'):
        # Top/bottom are horizontal lines -> use a horizontal unit vector (rightwards)
        target_direction = (1, 0)
    
    if not target_direction:
        return None, None
    
    # Calculate angle between head direction and target direction
    # Use dot product and cross product to get signed angle
    # dot = |a||b|cos(θ), cross = |a||b|sin(θ)
    dot_product = normalized_head_direction[0] * target_direction[0] + normalized_head_direction[1] * target_direction[1]
    cross_product = normalized_head_direction[0] * target_direction[1] - normalized_head_direction[1] * target_direction[0]
    
    # Calculate angle using atan2 for proper quadrant handling
    angle_rad = math.atan2(cross_product, dot_product)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to 0-360 range
    while angle_deg < 0:
        angle_deg += 360
    while angle_deg >= 360:
        angle_deg -= 360
    
    return angle_deg, normalized_head_direction

def get_target_line_position(target_line, frame_width, frame_height):
    """Get the numerical position of the target line"""
    if target_line == 'right':
        return frame_width
    elif target_line == 'left':
        return 0
    elif target_line == 'top':
        return 0
    elif target_line == 'bottom':
        return frame_height
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Management
# ═══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    """Application configuration loaded from .env file"""
    
    def __init__(self):
        # Get relative paths based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Model Configuration - use relative paths
        default_model_path = os.path.join(project_root, 'output', 'models', 'head_pose', 'best.pt')
        self.model_path = os.getenv('MODEL_PATH', default_model_path)
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.2'))
        
        # Video Input Configuration - use relative paths
        default_video_path = os.path.join(script_dir, 'videos', 'top_20250916T150021.mp4')
        self.video_path = os.getenv('VIDEO_PATH', default_video_path)
        self.processing_fps = int(os.getenv('PROCESSING_FPS', '60'))
        
        # Output Configuration
        self.output_dir = os.getenv('OUTPUT_DIR', '../output/detections')
        self.save_every_n_frames = int(os.getenv('SAVE_EVERY_N_FRAMES', '10'))
        self.save_every_n_previews = int(os.getenv('SAVE_EVERY_N_PREVIEWS', '30'))
        self.verbose = os.getenv('VERBOSE', 'false').lower() == 'true'
        self.detection_iou = float(os.getenv('DETECTION_IOU', '0.5'))
        self.detection_imgsz = int(os.getenv('DETECTION_IMGSZ', '640'))
        
        # Advanced Behavioral Analysis
        self.target_line = os.getenv('TARGET_LINE', 'right')
        self.near_max = float(os.getenv('NEAR_MAX', '0.20'))
        self.middle_max = float(os.getenv('MIDDLE_MAX', '0.30'))
        self.advance_threshold = float(os.getenv('ADVANCE_THRESHOLD', '0.002'))
        self.retreat_threshold = float(os.getenv('RETREAT_THRESHOLD', '0.002'))
        self.x_dir_thresh_norm = float(os.getenv('X_DIR_THRESH_NORM', '0.01'))
        self.y_dir_thresh_norm = float(os.getenv('Y_DIR_THRESH_NORM', '0.01'))
        self.head_only_thresh_norm = float(os.getenv('HEAD_ONLY_THRESH_NORM', '0.005'))
        self.body_move_thresh_norm = float(os.getenv('BODY_MOVE_THRESH_NORM', '0.010'))
        self.lookback_window = int(os.getenv('LOOKBACK_WINDOW', '5'))
        
        # Simple Behavior Detection (Fallback)
        self.min_moving_frames = int(os.getenv('MIN_MOVING_FRAMES', '3'))
        self.stop_threshold = float(os.getenv('STOP_THRESHOLD', '300.0'))
        self.min_stationary_frames = int(os.getenv('MIN_STATIONARY_FRAMES', '3'))
        
        # Web Server Configuration
        self.server_host = os.getenv('SERVER_HOST', '0.0.0.0')
        self.server_port = int(os.getenv('SERVER_PORT', '8079'))  # Different port from original version
        self.server_debug = os.getenv('SERVER_DEBUG', 'false').lower() == 'true'
        self.stream_fps = int(os.getenv('STREAM_FPS', '15'))
        self.jpeg_quality = int(os.getenv('JPEG_QUALITY', '85'))
        self.use_predict_stream = os.getenv('USE_PREDICT_STREAM', 'false').lower() == 'true'
        self.overlay_event_seconds = float(os.getenv('OVERLAY_EVENT_SECONDS', '2.5'))
        self.frame_queue_size = int(os.getenv('FRAME_QUEUE_SIZE', '2'))
        
        # Video saving with overlays
        self.save_video_with_overlays = os.getenv('SAVE_VIDEO_WITH_OVERLAYS', 'false').lower() == 'true'
        self.output_video_fps = float(os.getenv('OUTPUT_VIDEO_FPS', '15.0'))
        
        # Angle Tracking Configuration
        self.enable_angle_tracking = os.getenv('ENABLE_ANGLE_TRACKING', 'true').lower() == 'true'
        self.angle_kalman_process_noise = float(os.getenv('ANGLE_KALMAN_PROCESS_NOISE', '1e-4'))
        self.angle_kalman_measurement_noise = float(os.getenv('ANGLE_KALMAN_MEASUREMENT_NOISE', '1e-1'))
        self.show_angle_overlay = os.getenv('SHOW_ANGLE_OVERLAY', 'true').lower() == 'true'
        self.show_head_direction_arrow = os.getenv('SHOW_HEAD_DIRECTION_ARROW', 'true').lower() == 'true'
        
        # Embedding-enhanced pose model configuration
        self.use_embedding_model = os.getenv('USE_EMBEDDING_MODEL', 'false').lower() == 'true'
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '64'))
        self.embedding_memory_size = int(os.getenv('EMBEDDING_MEMORY_SIZE', '30'))
        self.embedding_min_confidence = float(os.getenv('EMBEDDING_MIN_CONFIDENCE', '0.3'))
        self.embedding_similarity_threshold = float(os.getenv('EMBEDDING_SIMILARITY_THRESHOLD', '0.7'))
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("📋 CURRENT CONFIGURATION - ENHANCED VERSION")
        print("="*60)
        print(f"🤖 Model: {self.model_path}")
        print(f"📹 Video: {self.video_path}")
        print(f"🎯 Confidence: {self.confidence_threshold}")
        print(f"⚡ Processing FPS: {self.processing_fps}")
        print(f"💾 Output: {self.output_dir}")
        print(f"🔊 Verbose: {self.verbose}")
        print(f"🌐 Server: {self.server_host}:{self.server_port}")
        print(f"📺 Stream FPS: {self.stream_fps}")
        print(f"🎬 Save Video with Overlays: {self.save_video_with_overlays}")
        if self.save_video_with_overlays:
            print(f"📼 Output Video FPS: {self.output_video_fps}")
        print(f"📐 Angle Tracking: {self.enable_angle_tracking}")
        if self.enable_angle_tracking:
            print(f"🎯 Target Line: {self.target_line}")
            print(f"📊 Show Angle Overlay: {self.show_angle_overlay}")
            print(f"🏹 Show Direction Arrow: {self.show_head_direction_arrow}")
        print(f"🧠 Embedding Model: {self.use_embedding_model}")
        if self.use_embedding_model:
            print(f"🔧 Embedding Dim: {self.embedding_dim}")
            print(f"💾 Memory Size: {self.embedding_memory_size}")
            print(f"🎯 Min Confidence: {self.embedding_min_confidence}")
            print(f"📏 Similarity Threshold: {self.embedding_similarity_threshold}")
        print("="*60 + "\n")

# Initialize global configuration
CONFIG = AppConfig()

# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes and File Management (from video_pose_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeadPose:
    bbox_xyxy: Tuple[float,float,float,float]
    conf: float
    nose: Optional[Tuple[float,float]] = None
    ear_left: Optional[Tuple[float,float]] = None
    ear_right: Optional[Tuple[float,float]] = None
    angle: Optional[float] = None  # Raw angle in degrees
    smoothed_angle: Optional[float] = None  # Kalman-filtered angle
    head_direction: Optional[Tuple[float,float]] = None  # Normalized direction vector

@dataclass
class PoseObservation:
    frame_index: int
    pose: Optional[HeadPose]

    def as_row(self) -> Tuple:
        if self.pose is None:
            return (self.frame_index, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        x1,y1,x2,y2 = self.pose.bbox_xyxy
        cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
        # Extract nose coordinates
        nose_x = self.pose.nose[0] if self.pose.nose else float("nan")
        nose_y = self.pose.nose[1] if self.pose.nose else float("nan")
        # Extract angle information if available
        angle = getattr(self.pose, 'angle', float("nan"))
        smoothed_angle = getattr(self.pose, 'smoothed_angle', float("nan"))
        return (self.frame_index, self.pose.conf, x1, y1, x2, y2, cx, cy, nose_x, nose_y, angle, smoothed_angle)

CSV_HEADER = ("frame_idx","conf","x1","y1","x2","y2","cx","cy","nose_x","nose_y","angle","smoothed_angle")

def now_tag():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def stem_for_source(src: str) -> str:
    p = Path(src)
    if p.exists():
        return p.stem
    return str(src).replace(":","_").replace("/","_")

def ensure_run_dir(output_base: str, source: str) -> Path:
    """Create organized output directory structure"""
    base = Path(output_base)
    base.mkdir(parents=True, exist_ok=True)
    run = base / f"{stem_for_source(source)}-{now_tag()}"
    (run / "labeled_frames").mkdir(parents=True, exist_ok=True)  # Clean frames for training
    (run / "labels").mkdir(parents=True, exist_ok=True)           # YOLO txt labels
    (run / "preview_frames").mkdir(parents=True, exist_ok=True)   # Frames WITH drawings
    return run

def save_run_config(run_dir: Path, cfg: dict):
    """Save configuration snapshot"""
    with open(run_dir / "run_config.json", "w") as fp:
        json.dump(cfg, fp, indent=2)

def save_yolo_label_txt(path_txt: Path, cls_id: int, bbox_xyxy: Tuple[float,float,float,float], 
                       img_w: int, img_h: int, conf: Optional[float] = None, 
                       keypoints: Optional[list] = None):
    """Save detection in YOLO pose format with keypoints"""
    x1,y1,x2,y2 = bbox_xyxy
    bw = x2-x1; bh = y2-y1
    cx = x1 + bw/2.0; cy = y1 + bh/2.0
    nx = cx / img_w; ny = cy / img_h; nw = bw / img_w; nh = bh / img_h
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(path_txt, "w") as f:
        # Start with bbox and confidence
        if conf is None:
            line = f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
        else:
            line = f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {conf:.6f}"
        
        # Add keypoints if available (YOLO pose format: x y visibility for each keypoint)
        if keypoints:
            for kpt in keypoints:
                if kpt is not None:
                    kpt_x, kpt_y = kpt
                    # Normalize keypoint coordinates
                    norm_x = kpt_x / img_w
                    norm_y = kpt_y / img_h
                    line += f" {norm_x:.6f} {norm_y:.6f} 2"  # visibility=2 (visible)
                else:
                    line += " 0.0 0.0 0"  # visibility=0 (not labeled)
        
        f.write(line + "\n")

def save_labeled_frame(path_img: Path, frame: np.ndarray, max_w: int = 900):
    """Save processed frame image"""
    h, w = frame.shape[:2]
    if w > max_w and w > 0:
        scale = max_w/float(w)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path_img), frame)

# ═══════════════════════════════════════════════════════════════════════════════

class SimpleHeadPoseDetector:
    def __init__(self, config: AppConfig):
        """Initialize with configuration from AppConfig"""
        self.config = config
        self.model_path = config.model_path
        self.video_path = config.video_path
        self.model = None
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.current_frame_number = 0
        self.total_frames = 0
        # Target processing FPS (can differ from source capture FPS)
        self.processing_target_fps = config.processing_fps
        self.fps = self.processing_target_fps  # Backward compatibility for existing references
        self.detection_count = 0
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.latest_raw_frame = None
        self.latest_processed_frame = None
        self.latest_raw_frame_time = 0.0
        self.latest_processed_frame_time = 0.0
        self.capture_fps = 0.0
        self.video_fps = 0.0
        self.capture_time_log = deque(maxlen=120)
        self.detection_time_log = deque(maxlen=120)
        self.stream_time_log = deque(maxlen=120)
        self.last_event_overlay = None
        self.last_event_time = 0.0
        self.frame_reader_thread = None
        self.detection_thread = None
        self.stream_target_fps = config.stream_fps
        self.detection_iou = config.detection_iou
        self.detection_imgsz = config.detection_imgsz
        
        # Control verbosity from config
        self.verbose = config.verbose
        self.use_predict_stream = config.use_predict_stream
        self.overlay_event_seconds = config.overlay_event_seconds
        
        # Trajectory logging for detailed CSV export
        self.trajectory_log = []  # Store detailed per-frame data
        self.start_time = time.time()
        
        # Initialize ADVANCED behavioral detector with arena mapping
        self.advanced_detector = None
        self.advanced_config = None
        self.behavior_detector = None
        self.live_metrics = None  # Always initialize to None
        
        if ADVANCED_BEHAVIOR_AVAILABLE:
            # Advanced configuration with arena mapping (from config)
            self.advanced_config = AdvancedBehaviorConfig(
                target_line=config.target_line,
                near_max=config.near_max,
                middle_max=config.middle_max,
                advance_threshold=config.advance_threshold,
                retreat_threshold=config.retreat_threshold,
                x_dir_thresh_norm=config.x_dir_thresh_norm,
                y_dir_thresh_norm=config.y_dir_thresh_norm,
                head_only_thresh_norm=config.head_only_thresh_norm,
                body_move_thresh_norm=config.body_move_thresh_norm,
                lookback_window=config.lookback_window,
            )
            # Initialize LiveMetrics for compatibility (even though advanced detector has its own tracking)
            self.live_metrics = LiveMetrics()
            # Note: Advanced detector will be fully initialized after video loads (need frame dimensions)
            print("✅ Advanced behavioral detector configured (arena mapping enabled)")
        elif BEHAVIOR_ANALYSIS_AVAILABLE:
            # Fallback to simple behavioral detector (from config)
            behavior_config = BehaviorConfig(
                min_moving_frames=config.min_moving_frames,
                stop_threshold=config.stop_threshold,
                min_stationary_frames=config.min_stationary_frames
            )
            self.behavior_detector = BehaviorDetector(behavior_config)
            self.live_metrics = LiveMetrics()
            print("✅ Simple behavioral detector initialized")
        else:
            print("⚠️ No behavioral analysis available")
        
        # Store behavioral events (newest first)
        self.behavioral_events = []
        self.events_lock = threading.Lock()
        
        # Detection settings (from config)
        self.CONFIDENCE_THRESHOLD = config.confidence_threshold
        
        # File organization (from config)
        self.output_dir = Path(config.output_dir)
        self.run_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.save_every_n_frames = config.save_every_n_frames
        self.save_every_n_previews = config.save_every_n_previews
        
        # Video saving with overlays 
        self.save_video_with_overlays = config.save_video_with_overlays
        self.output_video_fps = config.output_video_fps
        self.video_writer = None
        self.output_video_path = None
        
        # Angle tracking with Kalman filtering
        self.angle_kalman = AngleKalmanFilter(
            process_noise=config.angle_kalman_process_noise, 
            measurement_noise=config.angle_kalman_measurement_noise
        ) if config.enable_angle_tracking else None
        self.current_angle = None
        self.smoothed_angle = None
        self.head_direction = None  # Store normalized head direction vector for arrow display
        self.target_line_position = None
        self.last_detected = None  # Store last detected behavioral event for no-detection handling
        self.last_behavioral_instruction = None  # Store last meaningful behavioral instruction
        
        # Embedding model (if enabled)
        self.embedding_model = None
        
    def load_model(self):
        """Load YOLO model and optionally wrap with embedding model"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
                print(f"🚀 GPU detected: {device}")
            
            self.model = YOLO(self.model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
            print("✅ Model loaded on", "cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize embedding-enhanced model if enabled
            if self.config.use_embedding_model and EMBEDDING_MODEL_AVAILABLE:
                try:
                    # Create a simple pose model wrapper for the embedding model
                    class SimplePoseModel:
                        def __init__(self, yolo_model):
                            self.model = yolo_model
                        
                        def _extract(self, frame):
                            results = self.model.predict(frame, conf=CONFIG.confidence_threshold, verbose=False)
                            if results and len(results) > 0:
                                res = results[0]
                                if res.boxes is not None and len(res.boxes) > 0:
                                    from lizard_tracking.models.embedding_pose import ModelOutput
                                    return ModelOutput(
                                        boxes=res.boxes.xyxy.cpu().numpy() if res.boxes.xyxy is not None else None,
                                        confs=res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None,
                                        keypoints=res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
                                    )
                            return None
                    
                    base_model = SimplePoseModel(self.model)
                    self.embedding_model = create_embedding_enhanced_model(
                        weights_path=self.model_path,
                        embedding_dim=self.config.embedding_dim,
                        enable_gap_filling=True
                    )
                    self.embedding_model.base_model = base_model  # Override with our wrapper
                    print(f"✅ Embedding model enabled (dim={self.config.embedding_dim}, memory={self.config.embedding_memory_size})")
                except Exception as e:
                    print(f"⚠️ Embedding model failed to initialize: {e}")
                    self.embedding_model = None
            elif self.config.use_embedding_model and not EMBEDDING_MODEL_AVAILABLE:
                print("⚠️ Embedding model requested but not available - falling back to regular YOLO")
                self.embedding_model = None
            else:
                self.embedding_model = None
                print("📝 Using regular YOLO model (embedding disabled)")
            
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

    def predict_with_embeddings(self, frame):
        """Use embedding-enhanced model for prediction"""
        if self.embedding_model is None:
            return self.regular_predict(frame)
        
        try:
            # Get embedding-enhanced prediction
            embedding_output = self.embedding_model.predict(frame)
            
            if embedding_output and embedding_output.boxes is not None and len(embedding_output.boxes) > 0:
                # Convert to format expected by rest of pipeline
                # embedding_output should contain: boxes, confs, keypoints, embeddings
                
                # Get highest confidence detection
                best_idx = np.argmax(embedding_output.confs)
                
                bbox = embedding_output.boxes[best_idx]
                conf = embedding_output.confs[best_idx]
                
                if conf < self.config.embedding_min_confidence:
                    return None
                
                # Extract keypoints if available
                keypoints = None
                if embedding_output.keypoints is not None and len(embedding_output.keypoints) > best_idx:
                    keypoints = embedding_output.keypoints[best_idx]
                
                # Create HeadPose object
                pose = self.create_head_pose_from_prediction(bbox, conf, keypoints)
                return pose
                
        except Exception as e:
            print(f"⚠️ Embedding prediction failed: {e}")
            # Fallback to regular prediction
            return self.regular_predict(frame)
        
        return None

    def regular_predict(self, frame):
        """Use regular YOLO model for prediction"""
        try:
            results = self.model.predict(
                frame, 
                conf=self.CONFIDENCE_THRESHOLD, 
                iou=self.detection_iou,
                imgsz=self.detection_imgsz,
                verbose=False
            )
            
            if results and len(results) > 0:
                res = results[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    # Get highest confidence detection
                    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes.xyxy is not None else None
                    confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
                    keypoints = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
                    
                    if boxes is not None and confs is not None:
                        best_idx = np.argmax(confs)
                        bbox = boxes[best_idx]
                        conf = confs[best_idx]
                        
                        kpts = keypoints[best_idx] if keypoints is not None else None
                        pose = self.create_head_pose_from_prediction(bbox, conf, kpts)
                        return pose
        except Exception as e:
            print(f"⚠️ Regular prediction failed: {e}")
        
        return None

    def create_head_pose_from_prediction(self, bbox, conf, keypoints):
        """Create HeadPose object from prediction results"""
        try:
            # Extract keypoints (assuming nose=0, left_ear=3, right_ear=4 based on pose model)
            nose = None
            ear_left = None  
            ear_right = None
            
            if keypoints is not None and len(keypoints) >= 5:
                # Extract nose (index 0)
                if len(keypoints[0]) >= 2:
                    nose = (float(keypoints[0][0]), float(keypoints[0][1]))
                
                # Extract ears (indices 3 and 4)
                if len(keypoints) > 3 and len(keypoints[3]) >= 2:
                    ear_left = (float(keypoints[3][0]), float(keypoints[3][1]))
                
                if len(keypoints) > 4 and len(keypoints[4]) >= 2:
                    ear_right = (float(keypoints[4][0]), float(keypoints[4][1]))
            
            # Calculate head angle if angle tracking is enabled
            angle = None
            smoothed_angle = None
            head_direction = None
            
            if self.config.enable_angle_tracking and self.target_line_position is not None:
                frame_height, frame_width = self.latest_frame.shape[:2] if self.latest_frame is not None else (480, 640)
                
                angle, head_direction = calculate_head_angle_to_target(
                    nose, ear_left, ear_right, 
                    self.config.target_line, self.target_line_position,
                    frame_width, frame_height
                )
                
                if angle is not None and self.angle_kalman is not None:
                    smoothed_angle = self.angle_kalman.update(angle)
            
            pose = HeadPose(
                bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                conf=float(conf),
                nose=nose,
                ear_left=ear_left,
                ear_right=ear_right,
                angle=angle,
                smoothed_angle=smoothed_angle,
                head_direction=head_direction
            )
            
            # Store current angle information for display
            self.current_angle = angle
            self.smoothed_angle = smoothed_angle
            self.head_direction = head_direction
            
            return pose
            
        except Exception as e:
            print(f"⚠️ Error creating HeadPose: {e}")
            return None

    def predict(self, frame):
        """Main prediction method that chooses between embedding and regular YOLO"""
        if self.config.use_embedding_model and self.embedding_model is not None:
            return self.predict_with_embeddings(frame)
        else:
            return self.regular_predict(frame)
            
    def load_video(self):
        """Load video capture"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"❌ Could not open video: {self.video_path}")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Video loaded: {width}x{height} @ {self.video_fps:.1f} fps, {self.total_frames} frames")
            print(f"⚡ Processing target: {self.processing_target_fps} fps")
            
            # Initialize target line position for angle calculation
            if self.config.enable_angle_tracking:
                self.target_line_position = get_target_line_position(self.config.target_line, width, height)
                print(f"📐 Angle tracking enabled - Target line: {self.config.target_line} at position {self.target_line_position}")
            
            # Initialize advanced behavioral detector now that we have frame dimensions
            if ADVANCED_BEHAVIOR_AVAILABLE and self.advanced_config:
                self.advanced_detector = AdvancedBehavioralDetector(self.advanced_config, width, height)
                print(f"🧠 Advanced behavioral detector initialized for {width}x{height} arena")
            
            # Initialize output video writer if requested
            if self.save_video_with_overlays:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_video_path = str(self.run_dir / f"output_with_overlays_{now_tag()}.mp4")
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path, fourcc, self.output_video_fps, (width, height)
                )
                print(f"🎬 Video saving enabled: {self.output_video_path} @ {self.output_video_fps} fps")
            
            return True
            
        except Exception as e:
            print(f"❌ Video loading failed: {e}")
            return False

    def initialize_output(self):
        """Initialize output directory and CSV file"""
        try:
            # Create run directory
            self.run_dir = ensure_run_dir(str(self.output_dir), self.video_path)
            print(f"📁 Output directory: {self.run_dir}")
            
            # Save configuration
            config_dict = {
                'model_path': self.model_path,
                'video_path': self.video_path,
                'confidence_threshold': self.CONFIDENCE_THRESHOLD,
                'processing_fps': self.processing_target_fps,
                'detection_iou': self.detection_iou,
                'detection_imgsz': self.detection_imgsz,
                'use_embedding_model': self.config.use_embedding_model,
                'enable_angle_tracking': self.config.enable_angle_tracking,
                'target_line': self.config.target_line,
                'timestamp': now_tag()
            }
            save_run_config(self.run_dir, config_dict)
            
            # Initialize CSV for trajectory logging
            csv_path = self.run_dir / "trajectory.csv"
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(CSV_HEADER)
            print(f"📊 CSV logging: {csv_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Output initialization failed: {e}")
            return False

    def frame_reader_worker(self):
        """Background thread to read frames from video"""
        frame_interval = 1.0 / self.processing_target_fps if self.processing_target_fps > 0 else 0
        last_frame_time = 0
        
        while self.running and self.cap is not None:
            try:
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Store raw frame with timestamp
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                    self.latest_raw_frame_time = current_time
                
                # Add to detection queue (non-blocking)
                try:
                    self.frame_queue.put((self.current_frame_number, frame.copy()), timeout=0.001)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                
                last_frame_time = current_time
                
                # Log capture timing
                self.capture_time_log.append(current_time)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Frame reader error: {e}")
                break
        
        print("📹 Frame reader thread stopped")

    def detection_worker(self):
        """Background thread to process frames for detection"""
        last_save_frame = 0
        last_save_preview = 0
        
        while self.running:
            try:
                # Get frame from queue
                try:
                    frame_number, frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                detection_start = time.time()
                
                # Store frame for web interface
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Run pose detection
                pose = self.predict(frame)
                
                # Create observation
                observation = PoseObservation(frame_number, pose)
                
                # Log to CSV
                if self.csv_writer:
                    self.csv_writer.writerow(observation.as_row())
                
                # Add to trajectory log for behavioral analysis
                self.trajectory_log.append(observation)
                
                # Run behavioral analysis
                event = None
                instruction = None
                
                if self.advanced_detector and pose:
                    # Use advanced detector with arena mapping
                    event, instruction = self.advanced_detector.process_observation(observation)
                    if event:
                        with self.events_lock:
                            self.behavioral_events.insert(0, {
                                'frame': frame_number,
                                'time': time.time() - self.start_time,
                                'event': event,
                                'instruction': instruction or 'No instruction',
                                'pose': pose
                            })
                            # Keep only recent events
                            self.behavioral_events = self.behavioral_events[:100]
                        
                        # Store for no-detection overlay
                        self.last_detected = event
                        if instruction:
                            self.last_behavioral_instruction = instruction
                        self.last_event_time = time.time()
                        
                        if self.verbose:
                            print(f"🧠 Frame {frame_number}: {event} - {instruction}")
                
                elif self.behavior_detector and pose:
                    # Use simple behavioral detector
                    event = self.behavior_detector.process_observation(observation)
                    if event != EventType.UNKNOWN:
                        with self.events_lock:
                            self.behavioral_events.insert(0, {
                                'frame': frame_number,
                                'time': time.time() - self.start_time,
                                'event': event.name if hasattr(event, 'name') else str(event),
                                'instruction': 'Simple behavior detected',
                                'pose': pose
                            })
                            self.behavioral_events = self.behavioral_events[:100]
                        
                        self.last_detected = event.name if hasattr(event, 'name') else str(event)
                        self.last_event_time = time.time()
                        
                        if self.verbose:
                            print(f"🧠 Frame {frame_number}: {event}")
                
                # Update live metrics if available
                if self.live_metrics:
                    self.live_metrics.add_observation(observation)
                
                # Create annotated frame for preview
                preview_frame = self.create_preview_frame(frame, pose, event, instruction)
                
                # Store processed frame for web interface
                with self.frame_lock:
                    self.latest_processed_frame = preview_frame.copy()
                    self.latest_processed_frame_time = time.time()
                
                # Save video frame if enabled
                if self.video_writer is not None:
                    self.video_writer.write(preview_frame)
                
                # Save frames periodically
                if frame_number - last_save_frame >= self.save_every_n_frames:
                    if pose:  # Only save labeled frames when detection is present
                        # Save clean frame for training
                        frame_path = self.run_dir / "labeled_frames" / f"frame_{frame_number:06d}.jpg"
                        save_labeled_frame(frame_path, frame)
                        
                        # Save YOLO label
                        label_path = self.run_dir / "labels" / f"frame_{frame_number:06d}.txt"
                        keypoints = []
                        if pose.nose: keypoints.append(pose.nose)
                        if pose.ear_left: keypoints.append(pose.ear_left)  
                        if pose.ear_right: keypoints.append(pose.ear_right)
                        
                        save_yolo_label_txt(
                            label_path, 0, pose.bbox_xyxy, 
                            frame.shape[1], frame.shape[0], 
                            pose.conf, keypoints
                        )
                    
                    last_save_frame = frame_number
                
                # Save preview frames periodically
                if frame_number - last_save_preview >= self.save_every_n_previews:
                    preview_path = self.run_dir / "preview_frames" / f"preview_{frame_number:06d}.jpg"
                    save_labeled_frame(preview_path, preview_frame)
                    last_save_preview = frame_number
                
                self.detection_count += 1
                
                # Log detection timing
                detection_time = time.time() - detection_start
                self.detection_time_log.append(detection_time)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Detection worker error: {e}")
                    traceback.print_exc()
        
        print("🔍 Detection worker thread stopped")

    def create_preview_frame(self, frame, pose, event=None, instruction=None):
        """Create annotated frame with pose overlay and behavioral information"""
        annotated = frame.copy()
        
        if pose:
            # Draw bounding box
            x1, y1, x2, y2 = map(int, pose.bbox_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            conf_text = f"Conf: {pose.conf:.2f}"
            cv2.putText(annotated, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            if pose.nose:
                cv2.circle(annotated, (int(pose.nose[0]), int(pose.nose[1])), 3, (255, 0, 0), -1)
                cv2.putText(annotated, "nose", (int(pose.nose[0])+5, int(pose.nose[1])-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            if pose.ear_left:
                cv2.circle(annotated, (int(pose.ear_left[0]), int(pose.ear_left[1])), 3, (0, 255, 255), -1)
                cv2.putText(annotated, "L", (int(pose.ear_left[0])+5, int(pose.ear_left[1])-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            if pose.ear_right:
                cv2.circle(annotated, (int(pose.ear_right[0]), int(pose.ear_right[1])), 3, (255, 255, 0), -1)
                cv2.putText(annotated, "R", (int(pose.ear_right[0])+5, int(pose.ear_right[1])-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw angle information in top-right corner
            if self.config.enable_angle_tracking and self.config.show_angle_overlay:
                if pose.angle is not None and self.config.show_angle_overlay:
                    angle_text = f"Angle: {pose.angle:.1f}°"
                    if pose.smoothed_angle is not None:
                        angle_text += f" (smoothed: {pose.smoothed_angle:.1f}°)"
                    # Position in top-right corner
                    frame_height, frame_width = annotated.shape[:2]
                    text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = frame_width - text_size[0] - 10
                    text_y = 30
                    cv2.putText(annotated, angle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw head direction arrow in top-right corner
                if self.config.show_head_direction_arrow and pose.head_direction and pose.nose:
                    # Position arrow in top-right corner, below angle text
                    frame_height, frame_width = annotated.shape[:2]
                    arrow_center_x = frame_width - 60
                    arrow_center_y = 70
                    arrow_length = 40
                    arrow_start = (arrow_center_x, arrow_center_y)
                    end_point = (
                        int(arrow_center_x + pose.head_direction[0] * arrow_length),
                        int(arrow_center_y + pose.head_direction[1] * arrow_length)
                    )
                    cv2.arrowedLine(annotated, arrow_start, end_point, (0, 255, 255), 3, tipLength=0.3)
            
            # Draw target line for reference
            if self.config.enable_angle_tracking and self.target_line_position is not None:
                h, w = annotated.shape[:2]
                if self.config.target_line == 'right':
                    cv2.line(annotated, (w-1, 0), (w-1, h-1), (255, 0, 255), 2)
                elif self.config.target_line == 'left':
                    cv2.line(annotated, (0, 0), (0, h-1), (255, 0, 255), 2)
                elif self.config.target_line == 'top':
                    cv2.line(annotated, (0, 0), (w-1, 0), (255, 0, 255), 2)
                elif self.config.target_line == 'bottom':
                    cv2.line(annotated, (0, h-1), (w-1, h-1), (255, 0, 255), 2)
        
        # Behavioral event overlay
        if event:
            event_text = f"Behavior: {event}"
            if instruction:
                instruction_text = f"Instruction: {instruction}"
                cv2.putText(annotated, instruction_text, (10, annotated.shape[0]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated, event_text, (10, annotated.shape[0]-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Store for overlay persistence
            self.last_event_overlay = (event, instruction)
        
        # Handle no detection by displaying last detected behavioral event
        elif self.last_event_overlay and time.time() - self.last_event_time < self.overlay_event_seconds:
            event, instruction = self.last_event_overlay
            # Show with slightly reduced opacity (darker color)
            event_text = f"Last Behavior: {event}"
            cv2.putText(annotated, event_text, (10, annotated.shape[0]-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 64), 2)
            if instruction:
                instruction_text = f"Last Instruction: {instruction}"
                cv2.putText(annotated, instruction_text, (10, annotated.shape[0]-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (64, 128, 128), 2)
        
        # Add frame number and timestamp
        frame_text = f"Frame: {self.current_frame_number}/{self.total_frames}"
        timestamp = time.time() - self.start_time
        time_text = f"Time: {timestamp:.1f}s"
        
        cv2.putText(annotated, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated

    def start_processing(self):
        """Start video processing threads"""
        if not self.running:
            print("🚀 Starting video processing...")
            self.running = True
            
            # Start frame reader thread
            self.frame_reader_thread = threading.Thread(target=self.frame_reader_worker, daemon=True)
            self.frame_reader_thread.start()
            
            # Start detection thread  
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.detection_thread.start()
            
            print("✅ Processing threads started")

    def stop_processing(self):
        """Stop video processing"""
        if self.running:
            print("🛑 Stopping video processing...")
            self.running = False
            
            # Wait for threads to finish
            if self.frame_reader_thread:
                self.frame_reader_thread.join(timeout=2.0)
            if self.detection_thread:
                self.detection_thread.join(timeout=2.0)
            
            # Close video capture
            if self.cap:
                self.cap.release()
            
            # Close video writer
            if self.video_writer:
                self.video_writer.release()
                print(f"🎬 Video saved: {self.output_video_path}")
            
            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                print(f"📊 CSV saved: {self.run_dir / 'trajectory.csv'}")
            
            print("✅ Processing stopped")

    def get_latest_frame(self):
        """Get latest processed frame for web streaming"""
        with self.frame_lock:
            if self.use_predict_stream and self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
            elif self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
        return None

    def get_stats(self):
        """Get processing statistics"""
        current_time = time.time()
        
        # Calculate capture FPS
        if len(self.capture_time_log) >= 2:
            time_span = self.capture_time_log[-1] - self.capture_time_log[0]
            if time_span > 0:
                self.capture_fps = (len(self.capture_time_log) - 1) / time_span
        
        # Calculate detection timing
        avg_detection_time = np.mean(self.detection_time_log) if self.detection_time_log else 0
        
        # Calculate stream timing
        avg_stream_time = np.mean(self.stream_time_log) if self.stream_time_log else 0
        
        stats = {
            'frame_number': self.current_frame_number,
            'total_frames': self.total_frames,
            'progress_percent': (self.current_frame_number / self.total_frames * 100) if self.total_frames > 0 else 0,
            'video_fps': self.video_fps,
            'capture_fps': self.capture_fps,
            'processing_target_fps': self.processing_target_fps,
            'detection_count': self.detection_count,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'avg_stream_time_ms': avg_stream_time * 1000,
            'queue_size': self.frame_queue.qsize(),
            'uptime_seconds': current_time - self.start_time,
            'current_angle': self.current_angle,
            'smoothed_angle': self.smoothed_angle,
            'using_embedding_model': self.config.use_embedding_model and self.embedding_model is not None
        }
        
        # Add behavioral events
        with self.events_lock:
            stats['recent_events'] = self.behavioral_events[:10]  # Last 10 events
        
        return stats

# ═══════════════════════════════════════════════════════════════════════════════
# Flask Web Interface
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Global detector instance
detector = None

def generate_frames():
    """Generate video frames for streaming"""
    global detector
    
    while True:
        try:
            if detector is None:
                time.sleep(0.1)
                continue
                
            stream_start = time.time()
            
            frame = detector.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG.jpeg_quality])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Log stream timing
            stream_time = time.time() - stream_start
            detector.stream_time_log.append(stream_time)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control stream frame rate
            time.sleep(1.0 / CONFIG.stream_fps)
            
        except Exception as e:
            print(f"⚠️ Stream generation error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Main web interface"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Lizard Head Pose Detection - Enhanced Version</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.1); 
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            color: #333; 
        }
        .video-container { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .video-stream { 
            max-width: 100%; 
            height: auto; 
            border: 2px solid #007bff; 
            border-radius: 8px; 
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .stats-card { 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #007bff; 
        }
        .stats-title { 
            font-weight: bold; 
            color: #007bff; 
            margin-bottom: 10px; 
        }
        .stats-value { 
            font-size: 18px; 
            color: #333; 
        }
        .events-container { 
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
        }
        .event-item { 
            background-color: white; 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 5px; 
            border-left: 3px solid #28a745; 
        }
        .model-indicator {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .model-regular {
            background-color: #6c757d;
            color: white;
        }
        .model-embedding {
            background-color: #17a2b8;
            color: white;
        }
        .controls { 
            text-align: center; 
            margin-top: 20px; 
        }
        .btn { 
            background-color: #007bff; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 0 5px; 
        }
        .btn:hover { 
            background-color: #0056b3; 
        }
        .refresh-info { 
            text-align: center; 
            color: #666; 
            margin-top: 10px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦎 Lizard Head Pose Detection</h1>
            <h2>Enhanced Version - Configurable Model Selection</h2>
            <p>Real-time pose detection with behavioral analysis and angle tracking</p>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Video Stream">
        </div>
        
        <div class="stats-grid">
            <div class="stats-card">
                <div class="stats-title">📹 Video Progress</div>
                <div class="stats-value" id="progress">Loading...</div>
            </div>
            <div class="stats-card">
                <div class="stats-title">⚡ Performance</div>
                <div class="stats-value" id="performance">Loading...</div>
            </div>
            <div class="stats-card">
                <div class="stats-title">🔍 Detection Stats</div>
                <div class="stats-value" id="detection">Loading...</div>
            </div>
            <div class="stats-card">
                <div class="stats-title">📐 Angle Information</div>
                <div class="stats-value" id="angle">Loading...</div>
            </div>
        </div>
        
        <div class="events-container">
            <div class="stats-title">🧠 Recent Behavioral Events</div>
            <div id="events">Loading events...</div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="refreshStats()">🔄 Refresh Stats</button>
            <button class="btn" onclick="downloadData()">💾 Download Data</button>
        </div>
        
        <div class="refresh-info">
            Stats auto-refresh every 2 seconds
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Progress
                    document.getElementById('progress').innerHTML = 
                        `Frame ${data.frame_number}/${data.total_frames}<br>` +
                        `Progress: ${data.progress_percent.toFixed(1)}%<br>` +
                        `Uptime: ${data.uptime_seconds.toFixed(1)}s`;
                    
                    // Performance
                    document.getElementById('performance').innerHTML = 
                        `Video FPS: ${data.video_fps.toFixed(1)}<br>` +
                        `Capture FPS: ${data.capture_fps.toFixed(1)}<br>` +
                        `Target FPS: ${data.processing_target_fps}<br>` +
                        `Queue: ${data.queue_size} frames`;
                    
                    // Detection
                    const modelType = data.using_embedding_model ? 
                        '<span class="model-indicator model-embedding">EMBEDDING</span>' :
                        '<span class="model-indicator model-regular">REGULAR YOLO</span>';
                    document.getElementById('detection').innerHTML = 
                        `Model Type: ${modelType}<br>` +
                        `Detections: ${data.detection_count}<br>` +
                        `Avg Time: ${data.avg_detection_time_ms.toFixed(1)}ms<br>` +
                        `Stream Time: ${data.avg_stream_time_ms.toFixed(1)}ms`;
                    
                    // Angle
                    let angleText = 'Angle tracking: ';
                    if (data.current_angle !== null && data.current_angle !== undefined) {
                        angleText += `${data.current_angle.toFixed(1)}°<br>`;
                        if (data.smoothed_angle !== null && data.smoothed_angle !== undefined) {
                            angleText += `Smoothed: ${data.smoothed_angle.toFixed(1)}°`;
                        }
                    } else {
                        angleText += 'No detection';
                    }
                    document.getElementById('angle').innerHTML = angleText;
                    
                    // Events
                    let eventsHtml = '';
                    if (data.recent_events && data.recent_events.length > 0) {
                        data.recent_events.forEach(event => {
                            eventsHtml += `
                                <div class="event-item">
                                    <strong>Frame ${event.frame}</strong> (${event.time.toFixed(1)}s): 
                                    ${event.event}<br>
                                    <em>${event.instruction}</em>
                                </div>
                            `;
                        });
                    } else {
                        eventsHtml = '<div style="text-align: center; color: #666;">No recent behavioral events</div>';
                    }
                    document.getElementById('events').innerHTML = eventsHtml;
                })
                .catch(error => console.error('Error fetching stats:', error));
        }
        
        function refreshStats() {
            updateStats();
        }
        
        function downloadData() {
            window.open('/download_data', '_blank');
        }
        
        // Auto-refresh stats every 2 seconds
        updateStats();
        setInterval(updateStats, 2000);
    </script>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get current processing statistics"""
    global detector
    if detector:
        return jsonify(detector.get_stats())
    return jsonify({'error': 'Detector not initialized'})

@app.route('/download_data')
def download_data():
    """Download trajectory data as CSV"""
    global detector
    if detector and detector.run_dir:
        csv_path = detector.run_dir / "trajectory.csv"
        if csv_path.exists():
            return Response(
                open(csv_path, 'r').read(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=trajectory_{now_tag()}.csv'}
            )
    return "No data available", 404

# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    global detector
    
    # Print configuration
    CONFIG.print_config()
    
    # Check if embedding model is requested but not available
    if CONFIG.use_embedding_model and not EMBEDDING_MODEL_AVAILABLE:
        print("⚠️ WARNING: Embedding model requested but not available!")
        print("   Falling back to regular YOLO model")
        CONFIG.use_embedding_model = False
    
    # Initialize detector
    print("🔧 Initializing Enhanced Head Pose Detector...")
    detector = SimpleHeadPoseDetector(CONFIG)
    
    # Load model
    if not detector.load_model():
        print("❌ Failed to load model")
        return 1
    
    # Load video
    if not detector.load_video():
        print("❌ Failed to load video")
        return 1
    
    # Initialize output
    if not detector.initialize_output():
        print("❌ Failed to initialize output")
        return 1
    
    # Start processing
    detector.start_processing()
    
    try:
        print(f"🌐 Starting web server on {CONFIG.server_host}:{CONFIG.server_port}")
        print(f"📱 Open your browser to: http://{CONFIG.server_host}:{CONFIG.server_port}")
        print("🔧 Model type:", "Embedding-Enhanced" if CONFIG.use_embedding_model and detector.embedding_model else "Regular YOLO")
        print("📐 Angle tracking:", "Enabled" if CONFIG.enable_angle_tracking else "Disabled")
        
        app.run(
            host=CONFIG.server_host,
            port=CONFIG.server_port,
            debug=CONFIG.server_debug,
            threaded=True,
            use_reloader=False  # Disable reloader to avoid duplicate processes
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        # Cleanup
        detector.stop_processing()
        print("👋 Application terminated")

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code or 0)