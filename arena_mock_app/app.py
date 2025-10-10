#!/usr/bin/env python3
"""
Head Pose Detection Application
=====================================
- NoneType errors eliminated  
- Simple detection logic
- Stable streaming
- Visual feedback with confidence threshold
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
        
        # State covariance matrix
        self.P = np.eye(2) * 1000  # High initial uncertainty
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1.0, 1.0],  # angle = angle + velocity * dt (dt=1 frame)
                           [0.0, 1.0]])  # velocity = velocity
        
        # Measurement matrix (we only observe angle, not velocity)
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.Q = np.array([[process_noise, 0.0],
                           [0.0, process_noise]])
        
        # Measurement noise covariance
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
            self.x[1] = 0.0  # No initial velocity
            self.initialized = True
            return measured_angle
        
        # Handle angle wraparound (convert to 0-360 range)
        measured_angle = self._normalize_angle_360(measured_angle)
        
        # Handle angle discontinuity (e.g., 359° to 1°)
        angle_diff = measured_angle - self.x[0]
        if angle_diff > 180:
            measured_angle -= 360
        elif angle_diff < -180:
            measured_angle += 360
        
        # Prediction step
        self.x = self.F @ self.x  # Predict next state
        self.P = self.F @ self.P @ self.F.T + self.Q  # Predict covariance
        
        # Update step
        z = np.array([measured_angle])  # Measurement
        y = z - self.H @ self.x  # Innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y  # Update state
        self.P = (np.eye(2) - K @ self.H) @ self.P  # Update covariance
        
        # Return smoothed angle normalized to 0-360
        return self._normalize_angle_360(self.x[0])
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-180, 180] degrees"""
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360
        return angle
    
    def _normalize_angle_360(self, angle):
        """Normalize angle to [0, 360) degrees"""
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
    
    def get_angular_velocity(self):
        """Get estimated angular velocity in degrees per frame"""
        return self.x[1] if self.initialized else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Management
# ═══════════════════════════════════════════════════════════════════════════════

class AppConfig:
    """Application configuration loaded from .env file"""
    
    def __init__(self):
        # Model Configuration
        self.model_path = os.getenv('MODEL_PATH', '/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/output/models/head_pose/best.pt')
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.2'))
        
        # Video Input Configuration  
        self.video_path = os.getenv('VIDEO_PATH', '/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/arena_mock_app/videos/top_20250916T150021.mp4')
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
        self.server_port = int(os.getenv('SERVER_PORT', '8078'))
        self.server_debug = os.getenv('SERVER_DEBUG', 'false').lower() == 'true'
        self.stream_fps = int(os.getenv('STREAM_FPS', '15'))
        self.jpeg_quality = int(os.getenv('JPEG_QUALITY', '85'))
        self.use_predict_stream = os.getenv('USE_PREDICT_STREAM', 'false').lower() == 'true'
        self.overlay_event_seconds = float(os.getenv('OVERLAY_EVENT_SECONDS', '2.5'))
        self.frame_queue_size = int(os.getenv('FRAME_QUEUE_SIZE', '2'))
        
        # Video saving with overlays
        self.save_video_with_overlays = os.getenv('SAVE_VIDEO_WITH_OVERLAYS', 'false').lower() == 'true'
        self.output_video_fps = float(os.getenv('OUTPUT_VIDEO_FPS', '15.0'))
        
        # Embedding-enhanced pose model
        self.use_embedding_model = os.getenv('USE_EMBEDDING_MODEL', 'false').lower() == 'true'
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '64'))
        self.embedding_memory_size = int(os.getenv('EMBEDDING_MEMORY_SIZE', '30'))
        self.embedding_min_confidence = float(os.getenv('EMBEDDING_MIN_CONFIDENCE', '0.3'))
        self.embedding_similarity_threshold = float(os.getenv('EMBEDDING_SIMILARITY_THRESHOLD', '0.7'))
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("📋 CURRENT CONFIGURATION")
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
        print(f"🧠 Embedding Model: {self.use_embedding_model}")
        if self.use_embedding_model:
            print(f"📊 Embedding Dim: {self.embedding_dim}, Memory: {self.embedding_memory_size}")
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

@dataclass
class PoseObservation:
    frame_index: int
    pose: Optional[HeadPose]

    def as_row(self) -> Tuple:
        if self.pose is None:
            return (self.frame_index, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"))
        x1,y1,x2,y2 = self.pose.bbox_xyxy
        cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
        # Extract nose coordinates
        nose_x = self.pose.nose[0] if self.pose.nose else float("nan")
        nose_y = self.pose.nose[1] if self.pose.nose else float("nan")
        return (self.frame_index, self.pose.conf, x1, y1, x2, y2, cx, cy, nose_x, nose_y)

CSV_HEADER = ("frame_idx","conf","x1","y1","x2","y2","cx","cy","nose_x","nose_y")

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
                             0° = pointing directly at target
                             90° = pointing perpendicular to target (counterclockwise)
                             180° = pointing away from target
                             270° = pointing perpendicular to target (clockwise)
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
    
    if target_line == 'right':
        # Target is rightmost vertical line - pointing right towards target
        target_direction = (1, 0)  # Unit vector pointing right
    elif target_line == 'left':
        # Target is leftmost vertical line - pointing left towards target
        target_direction = (-1, 0)  # Unit vector pointing left
    elif target_line == 'top':
        # Target is topmost horizontal line - pointing up towards target
        target_direction = (0, -1)  # Unit vector pointing up (negative Y)
    elif target_line == 'bottom':
        # Target is bottommost horizontal line - pointing down towards target
        target_direction = (0, 1)  # Unit vector pointing down
    
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
        self.last_detected = None  # Store last detected behavioral event for low confidence display
        self.last_behavioral_instruction = None  # Store last meaningful behavioral instruction
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
        self.angle_kalman = AngleKalmanFilter(process_noise=1e-4, measurement_noise=1e-1)
        self.current_angle = None
        self.smoothed_angle = None
        self.head_direction = None  # Store normalized head direction vector for arrow display
        self.target_line_position = None
        self.last_detected = None  # Store last detected behavioral event for no-detection handling
        self.last_behavioral_instruction = None  # Store last meaningful behavioral instruction
        
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
            if hasattr(CONFIG, 'use_embedding_model') and CONFIG.use_embedding_model and EMBEDDING_MODEL_AVAILABLE:
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
                        embedding_dim=CONFIG.embedding_dim,
                        enable_gap_filling=True
                    )
                    self.embedding_model.base_model = base_model  # Override with our wrapper
                    print(f"✅ Embedding model enabled (dim={CONFIG.embedding_dim}, memory={CONFIG.embedding_memory_size})")
                except Exception as e:
                    print(f"⚠️ Embedding model failed to initialize: {e}")
                    self.embedding_model = None
            else:
                self.embedding_model = None
            
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def load_video(self):
        """Load video capture"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"❌ Failed to open video: {self.video_path}")
                return False
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.capture_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
            if self.capture_fps <= 0:
                fallback_fps = self.processing_target_fps if self.processing_target_fps > 0 else 15.0
                self.capture_fps = fallback_fps
            self.video_fps = self.capture_fps
            
            if self.processing_target_fps <= 0:
                self.processing_target_fps = self.capture_fps
            self.fps = self.processing_target_fps
            
            # Get frame dimensions for advanced detector
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = frame_width
            self.frame_height = frame_height
            self.video_fps = self.fps
            
            # Initialize advanced behavioral detector now that we have frame dimensions
            if ADVANCED_BEHAVIOR_AVAILABLE and self.advanced_config:
                self.advanced_detector = AdvancedBehavioralDetector(
                    config=self.advanced_config,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    fps=self.capture_fps
                )
                print(f"✅ Advanced detector initialized: {frame_width}x{frame_height} @ {self.capture_fps:.1f} FPS")
            
            # Initialize angle tracking target line position
            self.target_line_position = get_target_line_position(
                self.config.target_line, 
                frame_width, 
                frame_height
            )
            print(f"✅ Angle tracking initialized: target_line={self.config.target_line} at position={self.target_line_position}")
            
            print(f"✅ Video loaded: {self.total_frames} frames at {self.capture_fps:.1f} FPS")
            return True
        except Exception as e:
            print(f"❌ Video loading failed: {e}")
            return False
    
    def add_behavioral_event(self, event_type, description):
        """Add behavioral event to the list (newest first)"""
        current_time = datetime.now().strftime("%H:%M:%S")
        fps_for_time = self.video_fps or self.capture_fps or self.processing_target_fps or 1.0
        video_second = round(self.current_frame_number / fps_for_time, 1)
        
        description_ascii = (description
                              .replace("→", "->")
                              .replace("←", "<-")
                              .replace("—", "-")
                              .replace("–", "-")
                              .replace("•", "*")
                              )
        event = {
            "time": current_time,
            "video_second": video_second,
            "type": event_type,
            "description": description_ascii
        }
        
        with self.events_lock:
            # Insert at the beginning for newest-first ordering
            self.behavioral_events.insert(0, event)
            # Keep only last 50 events
            self.behavioral_events = self.behavioral_events[:50]
    
    def draw_simple_detection(self, frame, results):
        """Draw simple detection markers"""
        if not results or len(results) == 0:
            return frame
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes
            if len(boxes) == 0:
                continue
                
            for box in boxes:
                # Get confidence
                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0
                
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue
                
                # Get bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"Head: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw center dot
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                print(f"🎯 Detection found! Confidence: {conf:.3f} at ({center_x}, {center_y})")
        
        return frame
    
    def detect_poses(self, frame):
        """Run YOLO detection (with optional embedding enhancement) and draw overlays"""
        if self.model is None:
            return frame, []
        
        try:
            # Use embedding model if available
            if hasattr(self, 'embedding_model') and self.embedding_model is not None:
                try:
                    embedding_output = self.embedding_model.predict_with_embeddings(frame)
                    
                    if embedding_output.pose_output is not None:
                        # Convert back to YOLO format for drawing
                        class MockResult:
                            def __init__(self, pose_output):
                                self.boxes = MockBoxes(pose_output.boxes, pose_output.confs)
                                self.keypoints = MockKeypoints(pose_output.keypoints)
                        
                        class MockBoxes:
                            def __init__(self, boxes, confs):
                                self.xyxy = torch.from_numpy(boxes) if boxes is not None else None
                                self.conf = torch.from_numpy(confs) if confs is not None else None
                        
                        class MockKeypoints:
                            def __init__(self, keypoints):
                                self.xy = torch.from_numpy(keypoints) if keypoints is not None else None
                        
                        results = [MockResult(embedding_output.pose_output)]
                        
                        # Add embedding info to status
                        if embedding_output.filled_from_embedding:
                            self.last_event_overlay = f"🧠 EMBEDDING FILL: conf {embedding_output.confidence:.2f}"
                            self.last_event_time = time.time()
                        
                        if self.verbose and embedding_output.filled_from_embedding:
                            print(f"🧠 Frame {self.current_frame_number}: Filled from embedding (conf: {embedding_output.confidence:.2f})")
                        
                        # Continue with normal drawing
                        frame_with_detection = self.draw_simple_detection(frame, results)
                        return frame_with_detection, results
                    else:
                        results = []
                        frame_with_detection = frame.copy()
                        return frame_with_detection, results
                        
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Embedding model error: {e}, falling back to standard detection")
                    # Fall through to standard detection
            
            # Standard YOLO detection
            device_target = 'cuda' if torch.cuda.is_available() else 'cpu'
            detect_iou = self.detection_iou
            detect_imgsz = self.detection_imgsz
            if self.use_predict_stream:
                results_iter = self.model.predict(
                    frame,
                    stream=True,
                    verbose=False,
                    conf=self.CONFIDENCE_THRESHOLD,
                    iou=detect_iou,
                    imgsz=detect_imgsz,
                    device=device_target
                )
                results = list(results_iter)
            else:
                results = self.model(
                    frame, 
                    verbose=False, 
                    conf=self.CONFIDENCE_THRESHOLD,
                    iou=detect_iou,  # Match offline defaults
                    imgsz=detect_imgsz,
                    device=device_target
                )
                if not isinstance(results, list):
                    results = [results]
            
            if results and len(results) > 0:
                self.detection_count += 1
                if self.verbose:
                    print(f"🔍 Frame {self.current_frame_number}: {len(results)} results found")
                
                # Try to use proper drawing function first
                if DRAW_UTILS_AVAILABLE:
                    try:
                        # Extract bbox and keypoints from YOLO results
                        boxes = results[0].boxes
                        keypoints = results[0].keypoints if hasattr(results[0], 'keypoints') else None
                        
                        if boxes is not None and len(boxes) > 0:
                            box = boxes[0]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # Extract keypoints if available
                            nose, ear_left, ear_right = None, None, None
                            if keypoints is not None and len(keypoints.xy) > 0:
                                kpts = keypoints.xy[0].cpu().numpy()  # First detection keypoints
                                if self.verbose:
                                    print(f"🔍 DEBUG: Found {len(kpts)} keypoints: {kpts}")
                                if len(kpts) >= 3:
                                    # Extract keypoint coordinates
                                    kpt0 = (kpts[0][0], kpts[0][1]) if kpts[0][0] > 0 and kpts[0][1] > 0 else None
                                    kpt1 = (kpts[1][0], kpts[1][1]) if kpts[1][0] > 0 and kpts[1][1] > 0 else None
                                    kpt2 = (kpts[2][0], kpts[2][1]) if kpts[2][0] > 0 and kpts[2][1] > 0 else None
                                    
                                    if self.verbose:
                                        print(f"🔍 DEBUG: kpt0={kpt0}, kpt1={kpt1}, kpt2={kpt2}")
                                    
                                    # CORRECT mapping for /output/models/head_pose/best.pt (matches lib)
                                    nose = kpt0       # First keypoint is nose
                                    ear_left = kpt1   # Second keypoint is left ear
                                    ear_right = kpt2  # Third keypoint is right ear
                            
                            # Draw with proper parameters
                            frame = draw_head_pose(frame, (x1, y1, x2, y2), nose, ear_left, ear_right, confidence)
                            if self.verbose:
                                print(f"✅ Drew keypoints: nose={nose is not None}, ears={ear_left is not None and ear_right is not None}")
                        else:
                            frame = self.draw_simple_detection(frame, results)
                            if self.verbose:
                                print("⚠️ No boxes found, using simple drawing")
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ draw_head_pose failed: {e}, using simple drawing")
                        frame = self.draw_simple_detection(frame, results)
                else:
                    frame = self.draw_simple_detection(frame, results)
                
                # Add simple detection event occasionally
                if self.detection_count % 30 == 1:  # Every ~2 seconds at 15fps
                    self.add_behavioral_event("detection", f"Head detected (conf: {self.CONFIDENCE_THRESHOLD})")
            
            else:
                if self.verbose:
                    print(f"❌ Frame {self.current_frame_number}: No detections")
                
                # Handle no detection by displaying last detected behavioral event
                last_instruction = self.last_behavioral_instruction or self.last_detected
                if last_instruction:
                    # Add overlay text showing last detected behavioral event
                    overlay_text = f"CONF < {self.CONFIDENCE_THRESHOLD:.2f} - Last: {last_instruction}"
                    safe_overlay = overlay_text.replace("→", "->").replace("←", "<-")
                    cv2.putText(frame, safe_overlay, (10, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)  # Blue color for low confidence
            
            return frame, results
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return frame, []
    
    def process_behavioral_detection(self, results, frame):
        """Process detection with ADVANCED behavioral analysis (arena mapping, instruction grammar)"""
        instruction = None
        if not results or len(results) == 0:
            # Process frame with no detection (lookback will handle)
            if self.advanced_detector:
                instruction = self.advanced_detector.process_frame(
                    frame_idx=self.current_frame_number,
                    nose=None,
                    ear_left=None,
                    ear_right=None,
                    bbox=None
                )
            return instruction
        
        try:
            # Extract detection data from YOLO results
            result = results[0]
            boxes = result.boxes
            keypoints = result.keypoints if hasattr(result, 'keypoints') else None
            
            nose, ear_left, ear_right, bbox = None, None, None, None
            
            if boxes is not None and len(boxes) > 0:
                # Get bounding box
                box = boxes[0]
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = tuple(xyxy)  # numpy array already contains floats
                
                # Extract keypoints
                if keypoints is not None and len(keypoints.xy) > 0:
                    kpts = keypoints.xy[0].cpu().numpy()
                    if len(kpts) >= 3:
                        # Map keypoints: kpt0=nose, kpt1=ear_left, kpt2=ear_right
                        if kpts[0][0] > 0 and kpts[0][1] > 0:
                            nose = tuple(kpts[0])
                        if kpts[1][0] > 0 and kpts[1][1] > 0:
                            ear_left = tuple(kpts[1])
                        if kpts[2][0] > 0 and kpts[2][1] > 0:
                            ear_right = tuple(kpts[2])
                
                # Calculate head angle to target screen
                if nose and (ear_left or ear_right):
                    angle_result = calculate_head_angle_to_target(
                        nose=nose,
                        ear_left=ear_left,
                        ear_right=ear_right,
                        target_line=self.config.target_line,
                        target_line_position=self.target_line_position,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height
                    )
                    
                    if angle_result[0] is not None:
                        self.current_angle, self.head_direction = angle_result
                        # Apply Kalman filtering for smooth angle tracking
                        self.smoothed_angle = self.angle_kalman.update(self.current_angle)
                        
                        if self.verbose:
                            angular_velocity = self.angle_kalman.get_angular_velocity()
                            print(f"🧭 Angle to {self.config.target_line}: {self.current_angle:.1f}° → {self.smoothed_angle:.1f}° (vel: {angular_velocity:.2f}°/frame)")
                    else:
                        self.current_angle = None
                        self.head_direction = None
                else:
                    self.current_angle = None
                    self.head_direction = None
                
                # Process with ADVANCED detector
                if self.advanced_detector:
                    instruction = self.advanced_detector.process_frame(
                        frame_idx=self.current_frame_number,
                        nose=nose,
                        ear_left=ear_left,
                        ear_right=ear_right,
                        bbox=bbox
                    )
                    
                    # Add behavioral event with new instruction format
                    if instruction:
                        self.add_behavioral_event(
                            instruction.phase,  # 'approaching', 'retreating', or 'resting'
                            instruction.instruction  # Full instruction string
                        )
                        if self.verbose:
                            print(f"📍 {instruction.instruction}")
                
                # Update LiveMetrics (legacy, for compatibility)
                if hasattr(self, 'live_metrics') and self.live_metrics and bbox:
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    reference_point = (800, 300)
                    self.live_metrics.update_position((center_x, center_y), reference_point)
                
                # Log detailed trajectory data for CSV export
                if nose and bbox:
                    elapsed_sec = time.time() - self.start_time
                    fps_for_time = self.video_fps or self.capture_fps or self.processing_target_fps
                    timestamp_ms = self.current_frame_number * (1000.0 / fps_for_time) if fps_for_time and fps_for_time > 0 else 0
                    conf = result.boxes[0].conf[0].item() if hasattr(result.boxes[0], 'conf') else 0.0
                    
                    # Calculate distance from right edge (screen position)
                    frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 800
                    distance_from_edge = frame_width - nose[0]
                    
                    # Get speed from LiveMetrics
                    speed = self.live_metrics.current_speed_px_per_frame if self.live_metrics else 0.0
                    
                    # Get current event type from instruction
                    event_type = instruction.phase if instruction else ''
                    event_name = instruction.instruction if instruction else ''
                    
                    self.trajectory_log.append({
                        'frame': self.current_frame_number,
                        'timestamp': timestamp_ms,
                        'elapsed_sec': elapsed_sec,
                        'head_x': nose[0],
                        'head_y': nose[1],
                        'confidence': conf,
                        'distance_from_edge': distance_from_edge,
                        'speed_px_per_frame': speed,
                        'event_type': event_type,
                        'event_name': event_name,
                        'head_angle_raw': self.current_angle if self.current_angle is not None else float('nan'),
                        'head_angle_smoothed': self.smoothed_angle if self.smoothed_angle is not None else float('nan'),
                        'angular_velocity': self.angle_kalman.get_angular_velocity() if self.angle_kalman.initialized else float('nan')
                    })
                        
        except Exception as e:
            print(f"❌ Behavioral detection error: {e}")
            import traceback
            traceback.print_exc()
        
        return instruction
    
    def get_position_description(self, x, y):
        """Get position description based on screen at rightmost of frame"""
        frame_width = 800   # Screen width reference
        frame_height = 600  # Screen height reference
        
        # Screen is on the rightmost part of frame
        rightmost_line = frame_width  # Rightmost vertical line
        
        # FAR/NEAR based on distance from rightmost vertical line
        # far = more than 300 pixels from rightmost line
        distance_from_right = rightmost_line - x
        horizontal = "FAR" if distance_from_right > 300 else "NEAR"
        
        # RIGHT/LEFT where right="bottom", left="top" 
        # (assuming screen orientation where bottom=right, top=left)
        vertical = "RIGHT" if y > frame_height / 2 else "LEFT"
        
        return f"{horizontal}-{vertical}"
    
    def _compute_fps(self, time_log: deque) -> float:
        """Compute rolling FPS based on timestamps in deque"""
        if len(time_log) < 2:
            return 0.0
        duration = time_log[-1] - time_log[0]
        if duration <= 0:
            return 0.0
        return (len(time_log) - 1) / duration
    
    def _record_capture_timestamp(self):
        self.capture_time_log.append(time.time())
    
    def _record_detection_timestamp(self):
        self.detection_time_log.append(time.time())
    
    def _record_stream_timestamp(self):
        self.stream_time_log.append(time.time())
    
    def _get_event_overlay_text(self) -> Optional[str]:
        """Return event text for overlay if still fresh"""
        if not self.last_event_overlay:
            return None
        if (time.time() - self.last_event_time) > self.overlay_event_seconds:
            self.last_event_overlay = None
            return None
        return self.last_event_overlay
    
    def _handle_detection_output(self, frame_index: int, clean_frame: np.ndarray, frame_with_detection: np.ndarray,
                                 results, frame_width: int, frame_height: int, video_fps: float,
                                 detection_start_time: float):
        """Common post-processing for a detected frame (behavior, overlays, saving, metrics)."""
        instruction = self.process_behavioral_detection(results, clean_frame)
        self._record_detection_timestamp()
        
        if instruction:
            overlay_text = f"{instruction.phase.upper()}: {instruction.instruction}"
            overlay_text_ascii = (overlay_text
                                   .replace("→", "->")
                                   .replace("←", "<-")
                                   .replace("—", "-")
                                   .replace("–", "-")
                                   .replace("•", "*")
                                   )
            self.last_event_overlay = overlay_text_ascii
            self.last_event_time = time.time()
            self.last_detected = overlay_text_ascii
            self.last_behavioral_instruction = overlay_text_ascii  # Track meaningful instructions
        elif results:
            # Ensure periodic detection updates even without behavioral instructions
            if self.detection_count % 10 == 1:
                primary_conf = None
                try:
                    first_result = results[0]
                    if hasattr(first_result, 'boxes') and first_result.boxes is not None and len(first_result.boxes) > 0:
                        primary_conf = float(first_result.boxes[0].conf[0])
                except Exception:
                    primary_conf = None
                conf_text = f"{primary_conf:.2f}" if primary_conf is not None else f"&lt;{self.CONFIDENCE_THRESHOLD:.2f}"
                self.add_behavioral_event("detection", f"Head detected (conf: {conf_text})")
                last_detection = self.last_detected
                self.last_event_overlay = f"DETECTION: conf {conf_text}. Last detection: {last_detection}"
                self.last_event_time = time.time()
        
        # Always show status overlay in yellow at bottom-left
        event_overlay = self._get_event_overlay_text()
        if event_overlay:
            status_text = event_overlay
        else:
            # Fallback status when no behavioral events
            status_text = f"Conf: <{self.CONFIDENCE_THRESHOLD:.2f}. Latest detection: {self.last_detected or 'None'}>"
        
        # Clean up text for display
        safe_overlay = status_text.replace("→", "->").replace("←", "<-")
        cv2.putText(frame_with_detection, safe_overlay, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        fps_for_display = self.video_fps or video_fps or self.capture_fps or self.processing_target_fps or 1.0
        video_time = f"Video: {frame_index / fps_for_display:.1f}s"
        cv2.putText(frame_with_detection, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_detection, video_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add angle information overlay
        if self.smoothed_angle is not None:
            angle_text = f"Angle to {self.config.target_line}: {self.smoothed_angle:.1f}`"
            cv2.putText(frame_with_detection, angle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Cyan color
            
            # Draw angle direction indicator - arrow showing actual head direction
            if hasattr(self, 'head_direction') and self.head_direction is not None:
                # Draw small arrow in top-right corner showing actual head direction
                arrow_center = (frame_width - 60, 40)
                arrow_length = 30
                # Use the actual head direction vector for the arrow
                arrow_end = (
                    int(arrow_center[0] + arrow_length * self.head_direction[0]),
                    int(arrow_center[1] + arrow_length * self.head_direction[1])
                )
                cv2.arrowedLine(frame_with_detection, arrow_center, arrow_end, (0, 255, 255), 2, tipLength=0.3)
        
        # Overlay real-time FPS metrics
        detection_fps = self._compute_fps(self.detection_time_log)
        capture_fps = self._compute_fps(self.capture_time_log) or self.capture_fps
        # fps_text = f"Detect FPS: {detection_fps:.1f} | Capture FPS: {capture_fps:.1f}"
        # cv2.putText(frame_with_detection, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        # Save detection artifacts and metrics
        frame_count = frame_index + 1
        self.save_detection_data(results, frame_count, frame_width, frame_height, video_fps, clean_frame, frame_with_detection)
        self.save_metrics_snapshot(frame_count)
        
        with self.frame_lock:
            self.latest_processed_frame = frame_with_detection
            self.latest_frame = frame_with_detection
            self.latest_processed_frame_time = time.time()
        
        # Save frame to video with overlays if enabled
        if self.save_video_with_overlays and self.video_writer is not None:
            self.video_writer.write(frame_with_detection)
        
        detection_duration = time.time() - detection_start_time
        if self.verbose:
            print(f"⏱️ Detection frame {frame_index} took {detection_duration*1000:.1f} ms")
    
    def start_detection(self):
        """Start the detection process"""
        if not self.load_model() or not self.load_video():
            return False
        
        self.running = True
        self.fps = self.processing_target_fps  # maintain legacy attribute for target processing FPS
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.capture_time_log.clear()
        self.detection_time_log.clear()
        self.stream_time_log.clear()
        self.last_event_overlay = None
        self.last_detected = None
        self.last_behavioral_instruction = None
        self.last_event_time = 0.0
        self.latest_processed_frame = None
        self.latest_frame = None
        self.latest_raw_frame = None
        self.latest_processed_frame_time = 0.0
        self.latest_raw_frame_time = 0.0
        print("🎬 Detection started")
        
        # Setup file organization (from video_pose_pipeline.py)
        self.setup_file_organization()
        
        # Add initial event
        self.add_behavioral_event("system", "Detection started")
        
        target_fps_for_capture = self.capture_fps if self.capture_fps and self.capture_fps > 0 else self.processing_target_fps
        if not target_fps_for_capture or target_fps_for_capture <= 0:
            target_fps_for_capture = self.processing_target_fps if self.processing_target_fps > 0 else self.capture_fps
        if target_fps_for_capture and self.capture_fps and target_fps_for_capture > self.capture_fps:
            # Avoid racing ahead of the actual video frame rate to keep motion smooth
            target_fps_for_capture = self.capture_fps
        frame_interval = 1.0 / target_fps_for_capture if target_fps_for_capture and target_fps_for_capture > 0 else 0.0
        
        def frame_reader_loop():
            frame_index = 0
            next_frame_time = time.time()
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 End of video reached")
                    self.running = False
                    break
                
                clean_frame = frame.copy()
                with self.frame_lock:
                    self.latest_raw_frame = clean_frame
                    self.latest_raw_frame_time = time.time()
                
                self._record_capture_timestamp()
                
                try:
                    while self.frame_queue.full():
                        self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
                try:
                    self.frame_queue.put_nowait((clean_frame, frame_index))
                except queue.Full:
                    if self.verbose:
                        print("⚠️ Frame queue full, dropping frame")
                
                frame_index += 1
                
                if frame_interval > 0:
                    next_frame_time += frame_interval
                    sleep_time = next_frame_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            # Signal detection loop to exit
            try:
                self.frame_queue.put_nowait((None, None))
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((None, None))
                except Exception:
                    pass
        
        def detection_loop():
            frame_width = getattr(self, 'frame_width', None) or int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = getattr(self, 'frame_height', None) or int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = getattr(self, 'video_fps', None) or (self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
            
            try:
                while self.running or not self.frame_queue.empty():
                    try:
                        clean_frame, frame_index = self.frame_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    
                    if clean_frame is None:
                        break
                    
                    self.current_frame_number = frame_index
                    frame_for_detection = clean_frame.copy()
                    detection_start = time.time()
                    frame_with_detection, results = self.detect_poses(frame_for_detection)
                    self._handle_detection_output(frame_index, clean_frame, frame_with_detection, results,
                                                   frame_width, frame_height, video_fps, detection_start)
            finally:
                self.cleanup_files()
        
        # Start frame reader and detection threads
        self.frame_reader_thread = threading.Thread(target=frame_reader_loop, daemon=True, name="FrameReader")
        self.frame_reader_thread.start()
        self.detection_thread = threading.Thread(target=detection_loop, daemon=True, name="DetectionWorker")
        self.detection_thread.start()
        return True
    
    def setup_file_organization(self):
        """Initialize file organization structure (from video_pose_pipeline.py)"""
        # Create run directory
        self.run_dir = ensure_run_dir(str(self.output_dir), self.video_path)
        
        # Save configuration
        config = {
            "video_path": self.video_path,
            "model_path": self.model_path,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "fps": self.processing_target_fps,
            "capture_fps": self.capture_fps,
            "stream_fps": self.stream_target_fps,
            "device": "GPU (CUDA)" if torch.cuda.is_available() else "CPU",
            "timestamp": now_tag(),
            "total_frames": self.total_frames
        }
        save_run_config(self.run_dir, config)
        
        # Initialize CSV file
        csv_path = self.run_dir / "detections.csv"
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(CSV_HEADER)
        
        print(f"📁 Output directory: {self.run_dir}")
        print(f"📊 CSV file: {csv_path}")
        
        # Initialize video writer if enabled
        if self.save_video_with_overlays:
            self.output_video_path = self.run_dir / "processed_video_with_overlays.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc,
                self.output_video_fps,
                (self.frame_width, self.frame_height)
            )
            print(f"🎬 Video with overlays will be saved to: {self.output_video_path}")
    
    def save_detection_data(self, results, frame_count, frame_width, frame_height, video_fps, clean_frame, drawn_frame):
        """Save detection data: CSV + clean frames + labels + preview frames"""
        if not self.run_dir:
            return
        
        # Extract pose from YOLO results
        pose = None
        if results and len(results) > 0:
            # Get first result (YOLO returns list of Results objects)
            result = results[0]
            boxes = result.boxes
            keypoints = result.keypoints if hasattr(result, 'keypoints') else None
            
            if boxes is not None and len(boxes) > 0:
                # Get first (best) detection
                box = boxes[0]
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy  # numpy array already contains floats
                conf = box.conf[0].item() if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                
                # Create HeadPose object
                pose = HeadPose(
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=conf
                )
                
                # Extract keypoints (nose=kpt0, ear_left=kpt1, ear_right=kpt2)
                if keypoints is not None and len(keypoints.xy) > 0:
                    kpts = keypoints.xy[0].cpu().numpy()  # First detection keypoints
                    if len(kpts) >= 3:
                        # Extract keypoints with correct mapping
                        kpt0 = kpts[0]  # nose
                        kpt1 = kpts[1]  # ear_left  
                        kpt2 = kpts[2]  # ear_right
                        
                        # Save nose position (kpt0) - THIS IS WHAT USER WANTS
                        if kpt0[0] > 0 and kpt0[1] > 0:
                            pose.nose = tuple(kpt0)
                        
                        # Save ear positions
                        if kpt1[0] > 0 and kpt1[1] > 0:
                            pose.ear_left = tuple(kpt1)
                        if kpt2[0] > 0 and kpt2[1] > 0:
                            pose.ear_right = tuple(kpt2)
        
        # Log detection to CSV (includes nose x,y if available)
        obs = PoseObservation(frame_count, pose)
        self.csv_writer.writerow(obs.as_row())
        self.csv_file.flush()  # Ensure data is written
        
        # Save files if detection found
        if pose is not None:
            frame_name = f"frame{frame_count:08d}.jpg"
            label_name = f"frame{frame_count:08d}.txt"
            
            # Save CLEAN frame + YOLO label (for training/annotation)
            if self.detection_count % self.save_every_n_frames == 0:
                save_labeled_frame(self.run_dir / "labeled_frames" / frame_name, clean_frame)
                
                # Save YOLO label with keypoints
                keypoints = [pose.ear_left, pose.ear_right, pose.nose]
                save_yolo_label_txt(
                    self.run_dir / "labels" / label_name, 
                    0, pose.bbox_xyxy, frame_width, frame_height, pose.conf,
                    keypoints=keypoints
                )
                if self.verbose:
                    print(f"💾 Saved clean frame + label: {frame_name}")
            
            # Save PREVIEW frame with drawings (for visual inspection)
            if self.detection_count % self.save_every_n_previews == 0:
                save_labeled_frame(self.run_dir / "preview_frames" / frame_name, drawn_frame)
                if self.verbose:
                    print(f"🖼️  Saved preview: {frame_name}")
    
    def save_metrics_snapshot(self, frame_number):
        """Save periodic snapshot of LiveMetrics"""
        if not self.run_dir or not self.live_metrics:
            return
        
        # Save every 100 frames
        if frame_number % 100 == 0:
            metrics_dict = self.live_metrics.to_dict()
            metrics_dict['frame_number'] = frame_number
            
            # Convert NumPy types to Python types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            metrics_dict = convert_to_json_serializable(metrics_dict)
            
            # Append to metrics log file (JSON lines format)
            metrics_log_path = self.run_dir / "live_metrics_log.jsonl"
            with open(metrics_log_path, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
    
    def cleanup_files(self):
        """Cleanup file handles and save LiveMetrics summary"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.output_video_path:
                print(f"🎬 Video with overlays saved to: {self.output_video_path}")
        
        # Save LiveMetrics summary
        if self.run_dir and self.live_metrics:
            metrics_dict = self.live_metrics.to_dict()
            
            # Convert NumPy types to Python types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            metrics_dict = convert_to_json_serializable(metrics_dict)
            
            metrics_path = self.run_dir / "live_metrics_summary.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"📊 LiveMetrics saved to: {metrics_path}")
            
            # Print summary
            print(f"\n📈 LIVE METRICS SUMMARY:")
            print(f"   Frames processed: {metrics_dict['frames_processed']}")
            print(f"   Total distance traveled: {metrics_dict['total_distance_traveled']:.1f} px")
            print(f"   Average speed: {metrics_dict['average_speed']:.2f} px/frame")
            print(f"   Events detected: {metrics_dict['events_detected']}")
            print(f"   Trajectory length: {metrics_dict['trajectory_length']:.1f} px")
            print(f"   Is stationary: {metrics_dict['is_stationary']}")
            print(f"   Direction stability: {metrics_dict['direction_stability']:.2f}")
        
        # Save detailed trajectory log to CSV
        if self.run_dir and self.trajectory_log:
            trajectory_csv = self.run_dir / "trajectory.csv"
            print(f"\n📊 Saving detailed trajectory data ({len(self.trajectory_log)} frames)...")
            
            with open(trajectory_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'frame', 'timestamp', 'elapsed_sec', 'head_x', 'head_y', 
                    'confidence', 'distance_from_edge', 'speed_px_per_frame', 
                    'event_type', 'event_name', 'head_angle_raw', 'head_angle_smoothed', 'angular_velocity'
                ])
                writer.writeheader()
                writer.writerows(self.trajectory_log)
            
            print(f"📝 Trajectory CSV saved to: {trajectory_csv}")
        
        # Save ADVANCED behavioral analysis outputs
        if self.run_dir and self.advanced_detector:
            print(f"\n🎯 ADVANCED BEHAVIORAL ANALYSIS:")
            print(f"   Generated {len(self.advanced_detector.instructions)} instructions")
            
            # Save behavioral instructions CSV
            events_csv = self.run_dir / "behavioral_events.csv"
            save_events_csv(
                self.advanced_detector.get_instructions_csv_format(),
                events_csv
            )
            print(f"📝 Behavioral instructions saved to: {events_csv}")
            
            # Create interactive nose-heading map
            if hasattr(self, 'cap') and self.cap:
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                frame_width, frame_height = 800, 600
            
            plot_html = self.run_dir / "nose_heading_map.html"
            video_name = Path(self.video_path).stem
            
            create_nose_heading_map(
                plot_data=self.advanced_detector.get_plot_data(),
                video_name=video_name,
                output_path=plot_html,
                config=self.advanced_config,
                frame_width=frame_width,
                frame_height=frame_height
            )
            print(f"📈 Interactive plot saved to: {plot_html}")
        
        if self.run_dir:
            print(f"\n✅ Processing complete!")
            print(f"📊 Total detections: {self.detection_count}")
            print(f"📁 Results saved to: {self.run_dir}")
    
    def stop_detection(self):
        """Stop detection and save all outputs"""
        self.running = False
        
        try:
            if self.frame_queue:
                self.frame_queue.put_nowait((None, None))
        except Exception:
            pass
        
        if hasattr(self, 'frame_reader_thread') and self.frame_reader_thread:
            self.frame_reader_thread.join(timeout=2)
        
        if hasattr(self, 'detection_thread') and self.detection_thread:
            self.detection_thread.join(timeout=4)
        
        if self.cap:
            self.cap.release()
        self.add_behavioral_event("system", "Detection stopped")
    
    def get_latest_frame(self):
        """Get the latest processed frame"""
        freshness_window = 0.5
        if self.stream_target_fps > 0:
            freshness_window = max(freshness_window, 2.0 / self.stream_target_fps)
        
        with self.frame_lock:
            processed = self.latest_processed_frame
            processed_time = self.latest_processed_frame_time
            raw = self.latest_raw_frame
        
        now = time.time()
        if processed is not None and (now - processed_time) <= freshness_window:
            return processed.copy()
        if raw is not None:
            return raw.copy()
        return None
    
    def get_status(self):
        """Get current detection status"""
        device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        memory_info = f"GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "CPU Mode"
        capture_fps_measured = self._compute_fps(self.capture_time_log)
        detection_fps_measured = self._compute_fps(self.detection_time_log)
        stream_fps_measured = self._compute_fps(self.stream_time_log)
        
        if capture_fps_measured <= 0 and self.capture_fps:
            capture_fps_measured = self.capture_fps
        if detection_fps_measured < 0:
            detection_fps_measured = 0.0
        if stream_fps_measured <= 0 and self.stream_target_fps:
            stream_fps_measured = self.stream_target_fps
        
        return {
            "running": self.running,
            "device": device_info,
            "memory": memory_info,
            "fps": self.processing_target_fps,
            "processing_fps": self.processing_target_fps,
            "source_fps": self.capture_fps,
            "stream_target_fps": self.stream_target_fps,
            "capture_fps": capture_fps_measured,
            "detection_fps": detection_fps_measured,
            "stream_fps": stream_fps_measured,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "output_dir": str(self.run_dir.resolve()) if self.run_dir else "Not started yet",
            "predict_stream": self.use_predict_stream,
            "current_angle": self.current_angle,
            "smoothed_angle": self.smoothed_angle,
            "target_line": self.config.target_line,
            "target_line_position": self.target_line_position,
            "angle_tracking_enabled": True
        }
    
    def get_behavioral_events(self):
        """Get behavioral events (already ordered newest first)"""
        with self.events_lock:
            return self.behavioral_events.copy()

# Flask App
app = Flask(__name__)

# Disable Flask request logging (stops the 127.0.0.1 spam)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only show errors, not every request

# Global detector instance
detector = None

# HTML Template with improved UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🦎 FINAL Head Pose Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f0f0f0;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
            margin-bottom: 0px;
        }
        .main-content {
            display: flex;
            gap: 20px;
        }
        .video-section {
            flex: 2;
        }
        .controls-section {
            flex: 1;
            min-width: 300px;
        }
        .video-container {
            text-align: center;
            background: #000;
            border-radius: 10px;
            padding: 10px;
        }
        .controls-status-row {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .controls {
            flex: 0.6;
        }
        .controls-status-row .status {
            flex: 1.4;
        }
        .controls {
            background: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
        .controls h3 {
            margin: 0 0 8px 0;
            font-size: 1em;
        }
        .status {
            background: #e7f5e7;
            padding: 10px;
            border-radius: 10px;
        }
        .status h3 {
            margin: 0 0 8px 0;
            font-size: 1em;
        }
        .behavioral-log {
            background: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            max-height: calc(100vh - 250px);
            overflow-y: auto;
        }
        .behavioral-log h3 {
            margin: 0 0 10px 0;
            font-size: 1.1em;
            font-weight: bold;
        }
        .event-item {
            background: white;
            margin: 5px 0;
            padding: 8px;
            border-radius: 5px;
            border-left: 3px solid #4CAF50;
            font-size: 0.9em;
            transition: background-color 0.3s ease;
        }
        .event-item.new-event {
            animation: flashGreen 1s ease;
        }
        @keyframes flashGreen {
            0% { background-color: #c8e6c9; }
            100% { background-color: white; }
        }
        .event-time {
            font-weight: bold;
            color: #666;
        }
        .event-video-time {
            color: #999;
            font-size: 0.8em;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        }
        button:hover { background: #45a049; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        .stop-btn { background: #f44336; }
        .stop-btn:hover { background: #da190b; }
        video, img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 5px;
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦎 Head Pose Detection Interface</h1>
            <p>Real-time lizard head pose tracking with behavioral analysis</p>
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="video-stream" src="/video_feed" alt="Video Stream" style="width: 100%; max-width: 800px;">
                </div>
            </div>
            
            <div class="controls-section">
                <div class="controls-status-row">
                    <div class="controls">
                        <h3>🎮 Controls</h3>
                        <button id="start-btn" onclick="startDetection()">▶️ Start</button>
                        <button id="stop-btn" onclick="stopDetection()" class="stop-btn" disabled>⏹️ Stop</button>
                    </div>
                    
                    <div class="status">
                        <h3>📊 Status</h3>
                        <div id="status-info" style="font-size: 0.9em;">
                            <p>🔄 Ready to start...</p>
                        </div>
                    </div>
                </div>
                
                <div class="behavioral-log">
                    <h3>🎯 Behavioral Events</h3>
                    <div id="behavioral-events">
                        <p>No events yet...</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        
        function startDetection() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRunning = true;
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('stop-btn').disabled = false;
                        updateStatus();
                    }
                });
        }
        
        function stopDetection() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRunning = false;
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('stop-btn').disabled = true;
                        updateStatus();
                    }
                });
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status-info');
                    if (data.running) {
                        const outputDir = data.output_dir || 'Not started yet';
                        const processingTarget = Number((data.processing_fps ?? data.fps) || 0).toFixed(1);
                        const streamTarget = Number((data.stream_target_fps ?? data.stream_fps) || 0).toFixed(1);
                        const sourceFps = Number((data.source_fps ?? data.capture_fps) || 0).toFixed(1);
                        const captureFps = Number(data.capture_fps || 0).toFixed(1);
                        const detectFps = Number(data.detection_fps || 0).toFixed(1);
                        const streamFps = Number(data.stream_fps || 0).toFixed(1);
                        
                        // Angle information
                        const currentAngle = data.current_angle !== null ? Number(data.current_angle).toFixed(1) : 'N/A';
                        const smoothedAngle = data.smoothed_angle !== null ? Number(data.smoothed_angle).toFixed(1) : 'N/A';
                        const targetLine = data.target_line || 'N/A';
                        
                        // Create inline status layout
                        statusDiv.innerHTML = `
                            <div style="display: flex; flex-wrap: wrap; gap: 12px; line-height: 1.6; align-items: center;">
                                <span>✅ <strong>RUNNING</strong></span>
                                <span>🖥️ ${data.device || 'Unknown'}</span>
                                <span>💾 ${data.memory || 'Unknown'}</span>
                                <div style="display: flex; flex-direction: column; gap: 4px;">
                                    <span>⚙️ Processing target: ${processingTarget} fps</span>
                                    <span>📺 Stream target: ${streamTarget} fps</span>
                                    <span>🎞️ Source video: ${sourceFps} fps</span>
                                    <span>🎥 Capture: ${captureFps} | 🧠 Detect: ${detectFps} | 📺 Stream: ${streamFps}</span>
                                </div>
                                <span>🔍 Conf: ${data.confidence_threshold}</span>
                                <span>🛰️ YOLO stream: ${data.predict_stream ? 'ON' : 'OFF'}</span>
                                <div style="display: flex; flex-direction: column; gap: 4px; padding: 8px; background: #f0f8ff; border-radius: 4px;">
                                    <span>🧭 <strong>Head Angle Tracking</strong></span>
                                    <span>📍 Target: ${targetLine} edge</span>
                                    <span>📐 Current: ${currentAngle}° | Smoothed: ${smoothedAngle}°</span>
                                </div>
                            </div>
                            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
                                <div style="font-size: 0.85em; color: #666;">
                                    <strong>📁 Output:</strong><br>
                                    <a href="file://${outputDir}" style="color: #4CAF50; text-decoration: none; word-break: break-all; font-size: 0.8em;">${outputDir}</a>
                                </div>
                                <div style="font-size: 0.75em; color: #999; margin-top: 5px;">
                                    Files: trajectory.csv, behavioral_events.csv, nose_heading_map.html
                                </div>
                            </div>
                        `;
                    } else {
                        statusDiv.innerHTML = '<p>⏸️ Status: <strong>STOPPED</strong></p>';
                    }
                });
        }
        
        let previousEventCount = 0;
        
        function updateEvents() {
            fetch('/api/events')
                .then(response => response.json())
                .then(data => {
                    const eventsDiv = document.getElementById('behavioral-events');
                    if (data.length === 0) {
                        eventsDiv.innerHTML = '<p>No events yet...</p>';
                        return;
                    }
                    
                    const hasNewEvent = data.length > previousEventCount;
                    
                    eventsDiv.innerHTML = data.map((event, index) => {
                        // Events are newest-first, so index 0 is the newest (just entered)
                        const isNew = hasNewEvent && index === 0;
                        return `
                            <div class="event-item ${isNew ? 'new-event' : ''}">
                                <div style="display: flex; gap: 8px; align-items: baseline;">
                                    <span class="event-time">${event.time}</span>
                                    <span class="event-video-time">(${event.video_second}s)</span>
                                </div>
                                <div><strong>${event.type}:</strong> ${event.description}</div>
                            </div>
                        `;
                    }).join('');
                    
                    previousEventCount = data.length;
                })
                .catch(error => {
                    console.error('Error fetching events:', error);
                });
        }
        
        // Auto-update status and events
        setInterval(() => {
            if (isRunning) {
                updateStatus();
                updateEvents();
            }
        }, 1000);
        
        // Initial update
        updateStatus();
        updateEvents();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        base_target_fps = CONFIG.stream_fps
        print(f"✅ Video streaming started at {base_target_fps} FPS")
        last_target_fps = base_target_fps
        frame_interval = 1.0 / last_target_fps if last_target_fps > 0 else 0.0
        next_frame_time = time.time()
        while True:
            frame = None
            active_detector = detector
            if active_detector:
                frame = active_detector.get_latest_frame()
                current_target_fps = getattr(active_detector, 'stream_target_fps', last_target_fps)
            else:
                current_target_fps = base_target_fps
            
            if current_target_fps != last_target_fps:
                last_target_fps = current_target_fps
                frame_interval = 1.0 / last_target_fps if last_target_fps > 0 else 0.0
                next_frame_time = time.time()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG.jpeg_quality])
            if ret:
                if active_detector:
                    active_detector._record_stream_timestamp()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            if frame_interval > 0:
                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.001)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_detection():
    global detector
    try:
        if detector and detector.running:
            return jsonify({"success": False, "message": "Already running"})
        
        # Initialize detector with configuration
        detector = SimpleHeadPoseDetector(CONFIG)
        success = detector.start_detection()
        
        return jsonify({"success": success, "message": "Detection started" if success else "Failed to start"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    global detector
    try:
        if detector:
            detector.stop_detection()
        return jsonify({"success": True, "message": "Detection stopped"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/status')
def get_status():
    if detector:
        return jsonify(detector.get_status())
    return jsonify({"running": False})

@app.route('/api/events')
def get_events():
    if detector:
        return jsonify(detector.get_behavioral_events())
    return jsonify([])

if __name__ == '__main__':
    print("✅ All imports successful")
    print("🚀 Starting FINAL Head Pose Detection UI")
    
    # Print current configuration
    CONFIG.print_config()
    
    print(f"🌐 Server will be available at: http://localhost:{CONFIG.server_port}")
    print(f"🌐 Or access from network: http://{CONFIG.server_host}:{CONFIG.server_port}")
    
    # Start Flask app with configured settings
    app.run(host=CONFIG.server_host, port=CONFIG.server_port, debug=CONFIG.server_debug, threaded=True)
