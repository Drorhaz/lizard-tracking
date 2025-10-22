#!/usr/bin/env python3
"""
Head Pose Detection Application
=====================================
Features:
- Regular YOLO pose detection
- Advanced behavioral analysis with arena mapping
- Head angle calculation with Kalman filtering
- Real-time visual overlays (angle, direction arrow)
- Comprehensive data logging and visualization
- No-detection handling with last event display
- Stable streaming with confidence-based feedback
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - loaded from config/.env
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

# Import optical flow tracking
try:
    from lizard_tracking.core.optical_flow_tracker import OpticalFlowTracker, TrackingConfig, TrackedPose
    OPTICAL_FLOW_AVAILABLE = True
    print("✅ Optical flow tracking imported successfully")
except ImportError as e:
    print(f"⚠️ Optical flow tracking import failed: {e}")
    OPTICAL_FLOW_AVAILABLE = False

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

def run_performance_analysis(run_dir_path, config):
    """Run the performance analysis script on the completed run"""
    if not config.run_performance_analysis:
        print("📊 Performance analysis disabled in configuration")
        return None
    
    script_path = os.path.join(os.path.dirname(__file__), config.analysis_script_path)
    script_path = os.path.abspath(script_path)
    
    if not os.path.exists(script_path):
        print(f"⚠️ Performance analysis script not found: {script_path}")
        return None
    
    print(f"📊 Running performance analysis on: {run_dir_path}")
    print(f"🔧 Using script: {script_path}")
    print(f"🎯 Min confidence: {config.analysis_min_confidence}")
    print(f"📈 Generate plots: {config.analysis_generate_plots}")
    print(f"⏱️ Seconds per bin: {config.analysis_seconds_per_bin}")
    
    try:
        import subprocess
        import sys
        
        # Build command arguments
        cmd = [
            sys.executable,
            script_path,
            str(run_dir_path),
            "--min-confidence", str(config.analysis_min_confidence),
            "--seconds-per-bin", str(config.analysis_seconds_per_bin)
        ]
        
        # Add --no-plots flag if plots are disabled
        if not config.analysis_generate_plots:
            cmd.append("--no-plots")
        
        # Run the analysis script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Performance analysis completed successfully")
            if result.stdout:
                # Print relevant output lines
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['benchmark', 'summary', 'coverage', 'confidence', 'fps', '✅', '📊', '📈']):
                        print(f"📊 {line}")
            return True
        else:
            print(f"❌ Performance analysis failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
            if result.stdout:
                print(f"Standard output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Performance analysis timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running performance analysis: {e}")
        return False

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
        self.good_confidence_threshold = float(os.getenv('GOOD_CONFIDENCE_THRESHOLD', '0.8'))
        
        # Video Input Configuration - use relative paths
        default_video_path = os.path.join(script_dir, 'videos', 'top_20250916T150021.mp4')
        self.video_path = os.getenv('VIDEO_PATH', default_video_path)
        # Video directory for selection dropdown
        default_video_dir = os.path.join(script_dir, 'videos')
        self.video_dir = os.getenv('VIDEO_DIR', default_video_dir)
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
        
        # Optical Flow Tracking Configuration
        self.enable_optical_flow = os.getenv('ENABLE_OPTICAL_FLOW', 'true').lower() == 'true'
        self.optical_flow_confidence_threshold = float(os.getenv('OPTICAL_FLOW_CONF_THRESHOLD', '0.4'))
        self.optical_flow_max_track_length = int(os.getenv('OPTICAL_FLOW_MAX_TRACK_LENGTH', '8'))
        self.optical_flow_confidence_boost = float(os.getenv('OPTICAL_FLOW_CONF_BOOST', '0.2'))
        
        # Optical Flow Frame Saving Configuration
        self.save_optical_flow_frames = os.getenv('SAVE_OPTICAL_FLOW_FRAMES', 'true').lower() == 'true'
        self.save_every_n_optical_flow = int(os.getenv('SAVE_EVERY_N_OPTICAL_FLOW', '1'))  # Save every frame by default
        
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
        
        # Angle Tracking Configuration
        self.enable_angle_tracking = os.getenv('ENABLE_ANGLE_TRACKING', 'true').lower() == 'true'
        self.angle_kalman_process_noise = float(os.getenv('ANGLE_KALMAN_PROCESS_NOISE', '1e-4'))
        self.angle_kalman_measurement_noise = float(os.getenv('ANGLE_KALMAN_MEASUREMENT_NOISE', '1e-1'))
        self.show_angle_overlay = os.getenv('SHOW_ANGLE_OVERLAY', 'true').lower() == 'true'
        self.show_head_direction_arrow = os.getenv('SHOW_HEAD_DIRECTION_ARROW', 'true').lower() == 'true'
        
        # Performance Analysis Configuration
        self.run_performance_analysis = os.getenv('RUN_PERFORMANCE_ANALYSIS', 'true').lower() == 'true'
        self.analysis_script_path = os.getenv('ANALYSIS_SCRIPT_PATH', '../scripts/analyze_realworld_benchmark.py')
        self.analysis_min_confidence = float(os.getenv('ANALYSIS_MIN_CONFIDENCE', '0.05'))
        self.analysis_generate_plots = os.getenv('ANALYSIS_GENERATE_PLOTS', 'true').lower() == 'true'
        self.analysis_seconds_per_bin = int(os.getenv('ANALYSIS_SECONDS_PER_BIN', '15'))
    
    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("📋 CURRENT CONFIGURATION")
        print("="*60)
        print(f"🤖 Model: {self.model_path}")
        print(f"📹 Video: {self.video_path}")
        print(f"🎯 Confidence: {self.confidence_threshold}")
        print(f"⭐ Good Confidence: {self.good_confidence_threshold}")
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
        print(f"📊 Performance Analysis: {self.run_performance_analysis}")
        if self.run_performance_analysis:
            print(f"📈 Analysis Script: {self.analysis_script_path}")
            print(f"🎯 Analysis Min Confidence: {self.analysis_min_confidence}")
            print(f"📈 Generate Plots: {self.analysis_generate_plots}")
            print(f"⏱️ Seconds Per Bin: {self.analysis_seconds_per_bin}")
        print(f"🔄 Optical Flow Tracking: {self.enable_optical_flow}")
        if self.enable_optical_flow:
            print(f"🎯 OF Confidence Threshold: {self.optical_flow_confidence_threshold}")
            print(f"📏 OF Max Track Length: {self.optical_flow_max_track_length} frames")
            print(f"⚡ OF Confidence Boost: {self.optical_flow_confidence_boost}")
            print(f"💾 Save OF Frames: {self.save_optical_flow_frames}")
            if self.save_optical_flow_frames:
                print(f"📁 Save Every N OF Frames: {self.save_every_n_optical_flow}")
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
                    float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"))
        x1,y1,x2,y2 = self.pose.bbox_xyxy
        cx = (x1+x2)/2.0; cy=(y1+y2)/2.0
        # Extract nose coordinates
        nose_x = self.pose.nose[0] if self.pose.nose else float("nan")
        nose_y = self.pose.nose[1] if self.pose.nose else float("nan")
        # Extract ear coordinates
        ear_left_x = self.pose.ear_left[0] if self.pose.ear_left else float("nan")
        ear_left_y = self.pose.ear_left[1] if self.pose.ear_left else float("nan")
        ear_right_x = self.pose.ear_right[0] if self.pose.ear_right else float("nan")
        ear_right_y = self.pose.ear_right[1] if self.pose.ear_right else float("nan")
        # Extract angle information if available
        angle = getattr(self.pose, 'angle', float("nan"))
        smoothed_angle = getattr(self.pose, 'smoothed_angle', float("nan"))
        return (self.frame_index, self.pose.conf, x1, y1, x2, y2, cx, cy, nose_x, nose_y, 
                ear_left_x, ear_left_y, ear_right_x, ear_right_y, angle, smoothed_angle)

CSV_HEADER = ("frame_idx","conf","x1","y1","x2","y2","cx","cy","nose_x","nose_y",
              "ear_left_x","ear_left_y","ear_right_x","ear_right_y","angle","smoothed_angle")

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
    # Confidence-based directories
    (run / "frames_to_review" / "images").mkdir(parents=True, exist_ok=True)  # Low confidence frames
    (run / "frames_to_review" / "labels").mkdir(parents=True, exist_ok=True)  # Low confidence labels
    (run / "high_conf" / "images").mkdir(parents=True, exist_ok=True)         # High confidence frames
    (run / "high_conf" / "labels").mkdir(parents=True, exist_ok=True)         # High confidence labels
    return run

def save_run_config(run_dir: Path, cfg: dict):
    """Save configuration snapshot"""
    with open(run_dir / "run_config.json", "w") as fp:
        json.dump(cfg, fp, indent=2)

def save_optical_flow_frame_and_label(run_dir: Path, frame: np.ndarray, tracked_pose, 
                                     frame_count: int, is_rescue: bool = False):
    """Save optical flow tracked frame and generate YOLO label"""
    if not tracked_pose or not tracked_pose.bbox:
        return
    
    # Determine subdirectory based on tracking type
    subdir = "optical_flow_rescue" if is_rescue else "optical_flow"
    
    # Save frame
    frame_name = f"frame{frame_count:08d}.jpg"
    frame_path = run_dir / subdir / "images" / frame_name
    cv2.imwrite(str(frame_path), frame)
    
    # Generate YOLO label from tracked pose
    frame_height, frame_width = frame.shape[:2]
    label_path = run_dir / subdir / "labels" / f"frame{frame_count:08d}.txt"
    
    # Create keypoints list from tracked pose as tuples (x, y, vis)
    keypoints = []
    if tracked_pose.nose:
        keypoints.append((tracked_pose.nose[0], tracked_pose.nose[1], 2))  # 2 = visible
    else:
        keypoints.append((0, 0, 0))  # 0 = not visible

    if tracked_pose.ear_left:
        keypoints.append((tracked_pose.ear_left[0], tracked_pose.ear_left[1], 2))
    else:
        keypoints.append((0, 0, 0))

    if tracked_pose.ear_right:
        keypoints.append((tracked_pose.ear_right[0], tracked_pose.ear_right[1], 2))
    else:
        keypoints.append((0, 0, 0))
    
    # Save label with tracking metadata in comment
    save_yolo_label_txt(
        label_path, 
        cls_id=0,  # Head pose class
        bbox_xyxy=tracked_pose.bbox,
        img_w=frame_width,
        img_h=frame_height,
        conf=tracked_pose.confidence,
        keypoints=keypoints
    )
    
    # Add tracking metadata as comment in label file
    with open(label_path, 'a') as f:
        f.write(f"\n# Optical Flow Tracking Metadata\n")
        f.write(f"# is_tracked: {tracked_pose.is_tracked}\n")
        f.write(f"# tracking_quality: {tracked_pose.tracking_quality:.4f}\n")
        f.write(f"# frame_number: {tracked_pose.frame_number}\n")
        f.write(f"# is_rescue: {is_rescue}\n")
        f.write(f"# of_confidence: {tracked_pose.confidence:.4f}\n")

        # If the tracked_pose contains an original YOLO confidence or bbox, include it
        orig_conf = getattr(tracked_pose, 'original_confidence', None)
        orig_bbox = getattr(tracked_pose, 'original_bbox', None)
        if orig_conf is not None:
            f.write(f"# original_yolo_confidence: {orig_conf:.4f}\n")
        if orig_bbox is not None:
            f.write(f"# original_yolo_bbox_xyxy: {orig_bbox}\n")

    # Also save a raw (no overlays) image copy for the same frame so user has clean frames
    raw_frame_name = f"frame{frame_count:08d}_raw.jpg"
    raw_frame_path = run_dir / subdir / "images" / raw_frame_name
    try:
        cv2.imwrite(str(raw_frame_path), frame if frame is not None else frame)
    except Exception:
        # If frame contains overlays, try to reconstruct raw from latest_raw_frame on detector
        pass
    
    return frame_path, label_path

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
        self.video_completed = False  # Track if video reached end
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
        
        # Initialize optical flow tracker
        self.optical_flow_tracker = None
        if OPTICAL_FLOW_AVAILABLE and config.enable_optical_flow:
            tracking_config = TrackingConfig(
                max_track_length=config.optical_flow_max_track_length,
                confidence_boost=config.optical_flow_confidence_boost,
                min_track_quality=0.2,  # Minimum tracking quality threshold
                max_displacement=80.0,  # Maximum allowed displacement per frame  
                keypoint_names=['nose', 'ear_left', 'ear_right']
            )
            self.optical_flow_tracker = OpticalFlowTracker(tracking_config)
            print(f"✅ Optical flow tracker initialized (conf_threshold={config.optical_flow_confidence_threshold})")
        else:
            print("⚠️ Optical flow tracking disabled or unavailable")
        
    def load_model(self):
        """Load YOLO model"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
                print(f"🚀 GPU detected: {device}")
            
            self.model = YOLO(self.model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
            print("✅ Model loaded on", "cuda" if torch.cuda.is_available() else "cpu")
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
            if self.config.enable_angle_tracking:
                self.target_line_position = get_target_line_position(
                    self.config.target_line, frame_width, frame_height
                )
                print(f"✅ Angle tracking enabled: target line '{self.config.target_line}' at position {self.target_line_position}")
            
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
        """Run YOLO detection on the provided frame with optical flow fallback and draw overlays"""
        if self.model is None:
            return frame, []
        
        # Update optical flow tracker with current frame
        if self.optical_flow_tracker:
            self.optical_flow_tracker.update_frame(frame)
        
        try:
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
                            
                            # Check if we should use optical flow or YOLO detection
                            use_optical_flow = False
                            tracked_pose = None
                            
                            if (self.optical_flow_tracker and 
                                confidence < self.config.optical_flow_confidence_threshold):
                                # Try optical flow tracking for low confidence detections
                                if self.verbose:
                                    print(f"🔍 Low confidence ({confidence:.3f} < {self.config.optical_flow_confidence_threshold:.3f}), attempting optical flow")
                                
                                tracked_pose = self.optical_flow_tracker.track_pose(
                                    frame_number=self.current_frame_number,
                                    fallback_confidence=confidence
                                )
                                
                                if tracked_pose is not None:
                                    use_optical_flow = True
                                    if self.verbose:
                                        print(f"✅ Optical flow tracking successful (quality: {tracked_pose.tracking_quality:.3f})")
                            
                            # Determine final pose data to use
                            if use_optical_flow and tracked_pose:
                                # Use optical flow tracked pose
                                nose = tracked_pose.nose
                                ear_left = tracked_pose.ear_left
                                ear_right = tracked_pose.ear_right
                                final_bbox = tracked_pose.bbox or (x1, y1, x2, y2)
                                final_confidence = tracked_pose.confidence
                                is_tracked = True
                            else:
                                # Use YOLO detection or initialize tracking for high confidence
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
                                
                                final_bbox = (x1, y1, x2, y2)
                                final_confidence = confidence
                                is_tracked = False
                                
                                # Initialize or reinitialize optical flow tracking for high confidence detections
                                if (self.optical_flow_tracker and 
                                    self.optical_flow_tracker.should_reinitialize(confidence, self.config.optical_flow_confidence_threshold)):
                                    
                                    pose_data = {
                                        'nose': nose,
                                        'ear_left': ear_left,
                                        'ear_right': ear_right,
                                        'bbox': final_bbox,
                                        'confidence': confidence
                                    }
                                    
                                    try:
                                        self.optical_flow_tracker.start_tracking(pose_data, self.current_frame_number)
                                        if self.verbose:
                                            print(f"🔄 Optical flow tracking (re)initialized with confidence {confidence:.3f}")
                                    except Exception as e:
                                        if self.verbose:
                                            print(f"⚠️ Failed to initialize optical flow tracking: {e}")
                            
                            # Calculate head angle if angle tracking is enabled
                            angle, head_direction = None, None
                            if self.config.enable_angle_tracking and nose and (ear_left or ear_right):
                                angle, head_direction = calculate_head_angle_to_target(
                                    nose, ear_left, ear_right,
                                    self.config.target_line, 
                                    self.target_line_position,
                                    self.frame_width, self.frame_height
                                )
                                
                                if angle is not None and self.angle_kalman:
                                    # Apply Kalman filtering for smoother angle tracking
                                    self.smoothed_angle = self.angle_kalman.update(angle)
                                    self.current_angle = angle
                                    self.head_direction = head_direction
                                    
                                    if self.verbose:
                                        print(f"📐 Angle: {angle:.1f}° → {self.smoothed_angle:.1f}° (filtered)")
                            
                            # Draw with proper parameters (use final pose data)
                            frame = draw_head_pose(frame, final_bbox, nose, ear_left, ear_right, final_confidence)
                            
                            # Add tracking status overlay if using optical flow
                            if is_tracked and self.optical_flow_tracker:
                                tracking_info = self.optical_flow_tracker.get_tracking_info()
                                tracking_text = f"OF: {tracking_info['frames_since_detection']}f, Q:{tracked_pose.tracking_quality:.2f}"
                                cv2.putText(frame, tracking_text, (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Cyan for optical flow
                                
                                # Save optical flow tracked frame and label
                                if (self.config.save_optical_flow_frames and self.run_dir and 
                                    self.current_frame_number % self.config.save_every_n_optical_flow == 0):
                                    try:
                                        # Attach original YOLO info for metadata
                                        try:
                                            tracked_pose.original_confidence = confidence
                                            tracked_pose.original_bbox = (x1, y1, x2, y2)
                                        except Exception:
                                            pass
                                        frame_path, label_path = save_optical_flow_frame_and_label(
                                            self.run_dir, frame, tracked_pose, self.current_frame_number, is_rescue=False
                                        )
                                        if self.verbose:
                                            print(f"💾 Saved optical flow frame: {frame_path.name}")
                                    except Exception as e:
                                        if self.verbose:
                                            print(f"⚠️ Failed to save optical flow frame: {e}")
                            
                            # Add angle overlay in top-right corner if enabled
                            if self.config.enable_angle_tracking and self.config.show_angle_overlay and self.smoothed_angle is not None:
                                angle_text = f"Angle: {self.smoothed_angle:.1f}`"
                                # Position in top-right corner
                                frame_height, frame_width = frame.shape[:2]
                                text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                text_x = frame_width - text_size[0] - 10
                                text_y = 30
                                cv2.putText(frame, angle_text, (text_x, text_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
                            # Draw head direction arrow in top-right corner if enabled
                            if (self.config.enable_angle_tracking and self.config.show_head_direction_arrow 
                                and nose and self.head_direction):
                                # Position arrow in top-right corner, below angle text
                                frame_height, frame_width = frame.shape[:2]
                                arrow_center_x = frame_width - 60
                                arrow_center_y = 70
                                arrow_length = 40
                                arrow_start = (arrow_center_x, arrow_center_y)
                                arrow_end = (
                                    int(arrow_center_x + self.head_direction[0] * arrow_length),
                                    int(arrow_center_y + self.head_direction[1] * arrow_length)
                                )
                                cv2.arrowedLine(frame, arrow_start, arrow_end, 
                                              (0, 255, 255), 3, tipLength=0.3)
                            
                            if self.verbose:
                                print(f"✅ Drew keypoints: nose={nose is not None}, ears={ear_left is not None and ear_right is not None}")
                                if angle is not None:
                                    print(f"📐 Head angle: {angle:.1f}° (smoothed: {self.smoothed_angle:.1f}°)")
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
                
                # # Add simple detection event occasionally
                # if self.detection_count % 30 == 1:  # Every ~2 seconds at 15fps
                #     detection_msg = f"conf: >{self.CONFIDENCE_THRESHOLD:.2f}"
                #     self.add_behavioral_event("detection", detection_msg)
                #     self.last_detected = detection_msg
            
            else:
                if self.verbose:
                    print(f"❌ Frame {self.current_frame_number}: No detections")
                
                # Try optical flow tracking even when YOLO has no detections
                tracked_pose = None
                if self.optical_flow_tracker:
                    tracked_pose = self.optical_flow_tracker.track_pose(
                        frame_number=self.current_frame_number,
                        fallback_confidence=0.1  # Very low confidence for no-detection tracking
                    )
                    
                    if tracked_pose is not None:
                        if self.verbose:
                            print(f"✅ Optical flow rescued frame (quality: {tracked_pose.tracking_quality:.3f})")
                        
                        # Draw tracked pose
                        if DRAW_UTILS_AVAILABLE and tracked_pose.bbox:
                            frame = draw_head_pose(frame, tracked_pose.bbox, 
                                                 tracked_pose.nose, tracked_pose.ear_left, tracked_pose.ear_right, 
                                                 tracked_pose.confidence)
                            
                            # Add tracking status overlay
                            tracking_info = self.optical_flow_tracker.get_tracking_info()
                            tracking_text = f"OF RESCUE: {tracking_info['frames_since_detection']}f, Q:{tracked_pose.tracking_quality:.2f}"
                            cv2.putText(frame, tracking_text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)  # Orange for rescue tracking
                            
                            # Save rescue tracked frame and label
                            if (self.config.save_optical_flow_frames and self.run_dir and 
                                self.current_frame_number % self.config.save_every_n_optical_flow == 0):
                                try:
                                    frame_path, label_path = save_optical_flow_frame_and_label(
                                        self.run_dir, frame, tracked_pose, self.current_frame_number, is_rescue=True
                                    )
                                    if self.verbose:
                                        print(f"💾 Saved rescue tracking frame: {frame_path.name}")
                                except Exception as e:
                                    if self.verbose:
                                        print(f"⚠️ Failed to save rescue tracking frame: {e}")
                            
                            # Create fake results for behavioral analysis
                            results = [type('obj', (object,), {
                                'boxes': type('obj', (object,), {
                                    'xyxy': [np.array([tracked_pose.bbox])],
                                    'conf': [tracked_pose.confidence]
                                })(),
                                'keypoints': type('obj', (object,), {
                                    'xy': [np.array([[
                                        tracked_pose.nose or [0, 0],
                                        tracked_pose.ear_left or [0, 0], 
                                        tracked_pose.ear_right or [0, 0]
                                    ]])]
                                })() if tracked_pose.nose else None
                            })()]
                        
                        return frame, results
                
                # No tracking possible - handle no detection by displaying last detected behavioral event
                last_instruction = self.last_behavioral_instruction or self.last_detected
                if last_instruction:
                    # Add overlay text showing last detected behavioral event
                    overlay_text = f"CONF < {self.CONFIDENCE_THRESHOLD:.2f} - Last: {last_instruction}"
                    safe_overlay = overlay_text.replace("→", "->").replace("←", "<-").replace("—", "-").replace("–", "-")
                    cv2.putText(frame, safe_overlay, (10, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)  # Blue color for low confidence
                else:
                    # No previous detections available
                    overlay_text = f"CONF < {self.CONFIDENCE_THRESHOLD:.2f} - No previous detections"
                    cv2.putText(frame, overlay_text, (10, frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
            
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
                        # Map keypoints: kpt0=ear_left, kpt1=ear_right, kpt2=nose
                        if kpts[2][0] > 0 and kpts[2][1] > 0:
                            nose = tuple(kpts[2])
                        if kpts[0][0] > 0 and kpts[0][1] > 0:
                            ear_left = tuple(kpts[0])
                        if kpts[1][0] > 0 and kpts[1][1] > 0:
                            ear_right = tuple(kpts[1])
                
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
                        # Update last behavioral instruction for no-detection display
                        self.last_behavioral_instruction = instruction.instruction
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
                        'event_name': event_name
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
                conf_text = f"{primary_conf:.2f}" if primary_conf is not None else f"< {self.CONFIDENCE_THRESHOLD:.2f}"
                # detection_msg = f"conf: {conf_text}"
                # self.add_behavioral_event("detection", detection_msg)
                # self.last_detected = detection_msg
                last_detection = self.last_detected
                self.last_event_overlay = f"DETECTION: conf {conf_text}. Last detection: {last_detection}"
                self.last_event_time = time.time()
        
        # Always show status overlay in yellow at bottom-left
        event_overlay = self._get_event_overlay_text()
        if event_overlay:
            status_text = event_overlay
        else:
            # Fallback status when no behavioral events
            status_text = f"MONITORING | Frame: {frame_index} | Conf: >{self.CONFIDENCE_THRESHOLD:.2f}"
        
        # Clean up text for display
        safe_overlay = status_text.replace("→", "->").replace("←", "<-")
        cv2.putText(frame_with_detection, safe_overlay, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        fps_for_display = self.video_fps or video_fps or self.capture_fps or self.processing_target_fps or 1.0
        video_time = f"Video: {frame_index / fps_for_display:.1f}s"
        cv2.putText(frame_with_detection, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_detection, video_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        self.video_completed = False  # Reset video completion flag
        self.fps = self.processing_target_fps  # maintain legacy attribute for target processing FPS
        self.frame_queue = queue.Queue(maxsize=self.config.frame_queue_size)
        self.capture_time_log.clear()
        self.detection_time_log.clear()
        self.stream_time_log.clear()
        self.last_event_overlay = None
        self.last_detected = None
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
                    self.video_completed = True
                    self.running = False
                    self.add_behavioral_event("system", "Video completed")
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
                print("\n🔄 Finalizing outputs and cleaning up...")
                self.cleanup_files()
                
                # Run performance analysis after cleanup
                if self.run_dir:
                    print(f"\n📊 Running performance analysis...")
                    try:
                        analysis_success = run_performance_analysis(self.run_dir, self.config)
                        if analysis_success:
                            print("✅ Performance analysis completed")
                        else:
                            print("⚠️ Performance analysis failed or was skipped")
                    except Exception as e:
                        print(f"❌ Error during performance analysis: {e}")
                
                # Store the final output directory for status reporting
                if hasattr(self, 'run_dir') and self.run_dir:
                    self.final_output_dir = str(self.run_dir)
                    print(f"\n📁 Final output directory: {self.final_output_dir}")
                else:
                    self.final_output_dir = None
                
                # Ensure running state is properly set to False after all processing
                self.running = False
                print("🏁 All processing completed - detection fully stopped")
        
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
            "good_confidence_threshold": self.config.good_confidence_threshold,
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
                
                # Extract keypoints (nose=kpt2, ear_left=kpt0, ear_right=kpt1)
                if keypoints is not None and len(keypoints.xy) > 0:
                    kpts = keypoints.xy[0].cpu().numpy()  # First detection keypoints
                    if len(kpts) >= 3:
                        # Extract keypoints with correct mapping
                        kpt0 = kpts[0]  # ear_left
                        kpt1 = kpts[1]  # ear_right  
                        kpt2 = kpts[2]  # nose
                        
                        # Save nose position (kpt2) - THIS IS WHAT USER WANTS
                        if kpt2[0] > 0 and kpt2[1] > 0:
                            pose.nose = tuple(kpt2)
                        
                        # Save ear positions
                        if kpt0[0] > 0 and kpt0[1] > 0:
                            pose.ear_left = tuple(kpt0)
                        if kpt1[0] > 0 and kpt1[1] > 0:
                            pose.ear_right = tuple(kpt1)
                
                # Add angle information if available
                if hasattr(self, 'current_angle') and self.current_angle is not None:
                    pose.angle = self.current_angle
                if hasattr(self, 'smoothed_angle') and self.smoothed_angle is not None:
                    pose.smoothed_angle = self.smoothed_angle
                if hasattr(self, 'head_direction') and self.head_direction is not None:
                    pose.head_direction = self.head_direction
        
        # Log detection to CSV (includes nose x,y if available)
        obs = PoseObservation(frame_count, pose)
        self.csv_writer.writerow(obs.as_row())
        self.csv_file.flush()  # Ensure data is written
        
        # Save frames with no detection to frames_to_review (every N frames to avoid spam)
        if pose is None and frame_count % self.save_every_n_frames == 0:
            frame_name = f"frame{frame_count:08d}.jpg"
            save_labeled_frame(self.run_dir / "frames_to_review" / "images" / frame_name, clean_frame)
            if self.verbose:
                print(f"📝 Saved no-detection frame: {frame_name}")
        
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
            
            # Save frames based on confidence thresholds
            self._save_confidence_based_frames(pose, clean_frame, frame_name, label_name, frame_width, frame_height)
    
    def _save_confidence_based_frames(self, pose, clean_frame, frame_name, label_name, frame_width, frame_height):
        """Save frames to confidence-based directories"""
        if not pose or not self.run_dir:
            return
        
        conf = pose.conf
        keypoints = [pose.ear_left, pose.ear_right, pose.nose]
        
        # Low confidence frames (< threshold) → frames_to_review
        if conf < self.CONFIDENCE_THRESHOLD:
            # Save image
            save_labeled_frame(self.run_dir / "frames_to_review" / "images" / frame_name, clean_frame)
            # Save label
            save_yolo_label_txt(
                self.run_dir / "frames_to_review" / "labels" / label_name,
                0, pose.bbox_xyxy, frame_width, frame_height, conf,
                keypoints=keypoints
            )
            if self.verbose:
                print(f"📝 Saved low-conf frame (conf={conf:.3f}): {frame_name}")
        
        # High confidence frames (> good_conf threshold) → high_conf
        elif conf > self.config.good_confidence_threshold:
            # Save image
            save_labeled_frame(self.run_dir / "high_conf" / "images" / frame_name, clean_frame)
            # Save label
            save_yolo_label_txt(
                self.run_dir / "high_conf" / "labels" / label_name,
                0, pose.bbox_xyxy, frame_width, frame_height, conf,
                keypoints=keypoints
            )
            if self.verbose:
                print(f"⭐ Saved high-conf frame (conf={conf:.3f}): {frame_name}")
    
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
        
        # Ensure video writer is properly closed and saved
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                self.video_writer = None
                if self.output_video_path and os.path.exists(self.output_video_path):
                    print(f"🎬 Video with overlays saved to: {self.output_video_path}")
                    # Store video path for status reporting
                    self.saved_video_path = self.output_video_path
                else:
                    print("⚠️ Video file was not created or is missing")
                    self.saved_video_path = None
            except Exception as e:
                print(f"⚠️ Error releasing video writer: {e}")
                self.saved_video_path = None
        else:
            self.saved_video_path = None
        
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
                    'event_type', 'event_name'
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
            # Count confidence-based saves
            frames_to_review_count = len(list((self.run_dir / "frames_to_review" / "images").glob("*.jpg"))) if (self.run_dir / "frames_to_review" / "images").exists() else 0
            high_conf_count = len(list((self.run_dir / "high_conf" / "images").glob("*.jpg"))) if (self.run_dir / "high_conf" / "images").exists() else 0
            
            print(f"\n✅ Processing complete!")
            print(f"📊 Total detections: {self.detection_count}")
            print(f"📝 Frames to review (conf < {self.CONFIDENCE_THRESHOLD}): {frames_to_review_count}")
            print(f"⭐ High confidence frames (conf > {self.config.good_confidence_threshold}): {high_conf_count}")
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
        
        self.add_behavioral_event("system", "Detection stopped - cleanup will complete in background")
        
        # Note: cleanup_files() and performance analysis will be called 
        # by the detection_loop's finally block, so we don't duplicate it here
    
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
        
        # Determine output directory to display
        output_dir_display = "Not started yet"
        if hasattr(self, 'final_output_dir') and self.final_output_dir:
            # Show final output directory if processing is complete
            output_dir_display = self.final_output_dir
        elif self.run_dir:
            # Show current run directory if processing is ongoing
            output_dir_display = str(self.run_dir.resolve())
        
        status = {
            "running": self.running,
            "video_completed": self.video_completed,
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
            "good_confidence_threshold": self.config.good_confidence_threshold,
            "output_dir": output_dir_display,
            "predict_stream": self.use_predict_stream,
            # Video output information
            "saved_video_path": getattr(self, 'saved_video_path', None),
            "final_output_dir": getattr(self, 'final_output_dir', None),
            # Angle tracking information
            "angle_tracking_enabled": self.config.enable_angle_tracking,
            "target_line": self.config.target_line if self.config.enable_angle_tracking else None,
            "current_angle": self.current_angle if self.config.enable_angle_tracking else None,
            "smoothed_angle": self.smoothed_angle if self.config.enable_angle_tracking else None,
            "show_angle_overlay": self.config.show_angle_overlay if self.config.enable_angle_tracking else False,
            "show_direction_arrow": self.config.show_head_direction_arrow if self.config.enable_angle_tracking else False
        }
    
    def get_behavioral_events(self):
        """Get behavioral events (already ordered newest first)"""
        with self.events_lock:
            return self.behavioral_events.copy()
    
    def set_video_path(self, video_path):
        """Update the video path (only when not running)"""
        if not self.running:
            self.video_path = video_path
            self.video_completed = False
            return True
        return False

# Flask App
app = Flask(__name__)

# Disable Flask request logging (stops the 127.0.0.1 spam)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only show errors, not every request

# Global detector instance
detector = None

# HTML Template with UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🦎 Head Pose Detection - Original Version</title>
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
        .video-selector {
            background: #f0f8ff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .video-selector h3 {
            margin: 0 0 8px 0;
            font-size: 1em;
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
            <h1>🦎 Head Pose Detection - Original Version</h1>
            <p>Real-time lizard head pose tracking with behavioral analysis and angle measurement</p>
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="video-stream" src="/video_feed" alt="Video Stream" style="width: 100%; max-width: 800px; display: none;">
                    <img id="video-preview" alt="Video Preview" style="width: 100%; max-width: 800px; display: block;">
                    <div id="preview-info" style="color: #999; font-size: 0.9em; margin-top: 10px;">
                        <span id="preview-text">Loading preview...</span>
                    </div>
                </div>
            </div>
            
            <div class="controls-section">
                <div class="video-selector">
                    <h3>📹 Video Selection</h3>
                    <select id="video-select" onchange="selectVideo()" style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px;">
                        <option value="">Loading videos...</option>
                    </select>
                    <div id="video-info" style="font-size: 0.8em; color: #666; margin-bottom: 10px;">
                        <span id="current-video">No video selected</span>
                    </div>
                </div>
                
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
        
        // Initialize isRunning based on server state
        function initializeRunningState() {
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                isRunning = data.running || false;
                console.log('Initialized isRunning state:', isRunning);
                // Force immediate UI sync
                updateStatus();
            })
            .catch(error => {
                console.error('Error initializing running state:', error);
                isRunning = false;
            });
        }
        
        function loadVideos() {
            console.log('Loading videos...');
            fetch('/api/videos')
                .then(response => {
                    console.log('Response status:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('API response:', data);
                    const select = document.getElementById('video-select');
                    const currentVideoSpan = document.getElementById('current-video');
                    
                    // Clear existing options
                    select.innerHTML = '';
                    
                    if (data.error) {
                        select.innerHTML = '<option value="">Error loading videos</option>';
                        currentVideoSpan.textContent = `Error: ${data.error}`;
                        return;
                    }
                    
                    if (data.videos.length === 0) {
                        select.innerHTML = '<option value="">No videos found</option>';
                        currentVideoSpan.textContent = `No videos in ${video_dir}`;
                        return;
                    }
                    
                    // Add videos to dropdown
                    data.videos.forEach(video => {
                        const option = document.createElement('option');
                        option.value = video.path;
                        option.textContent = `${video.name} (${(video.size / 1024 / 1024).toFixed(1)} MB)`;
                        if (video.path === data.current) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });
                    
                    // Update current video display
                    const currentVideo = data.videos.find(v => v.path === data.current);
                    if (currentVideo) {
                        currentVideoSpan.textContent = `Current: ${currentVideo.name}`;
                        // Load preview for current video
                        loadPreview();
                    } else {
                        currentVideoSpan.textContent = 'No video selected';
                    }
                })
                .catch(error => {
                    console.error('Error loading videos:', error);
                    const select = document.getElementById('video-select');
                    const currentVideoSpan = document.getElementById('current-video');
                    select.innerHTML = '<option value="">Error loading videos</option>';
                    currentVideoSpan.textContent = `Error: ${error.message}`;
                });
        }
        
        function selectVideo() {
            const select = document.getElementById('video-select');
            const videoPath = select.value;
            
            if (!videoPath) return;
            
            fetch('/api/set_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video_path: videoPath})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const currentVideoSpan = document.getElementById('current-video');
                    const videoName = videoPath.split('/').pop();
                    currentVideoSpan.textContent = `Current: ${videoName}`;
                    console.log(data.message);
                    // Update preview when video changes
                    loadPreview();
                } else {
                    alert(`Error: ${data.message}`);
                    loadVideos(); // Reload to reset selection
                }
            })
            .catch(error => {
                console.error('Error setting video:', error);
                loadVideos(); // Reload to reset selection
            });
        }
        
        function loadPreview() {
            const previewImg = document.getElementById('video-preview');
            const previewText = document.getElementById('preview-text');
            
            previewText.textContent = 'Loading preview...';
            
            fetch('/api/preview_frame')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        previewImg.src = data.image;
                        previewText.textContent = `Preview: ${data.video_info.name} (${data.video_info.dimensions})`;
                    } else {
                        previewImg.src = '';
                        previewText.textContent = `Preview error: ${data.message}`;
                    }
                })
                .catch(error => {
                    console.error('Error loading preview:', error);
                    previewImg.src = '';
                    previewText.textContent = 'Error loading preview';
                });
        }
        
        function startDetection() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        isRunning = true;
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('stop-btn').disabled = false;
                        // Switch from preview to live stream
                        document.getElementById('video-preview').style.display = 'none';
                        document.getElementById('video-stream').style.display = 'block';
                        document.getElementById('preview-text').textContent = 'Live detection running...';
                        updateStatus();
                    } else {
                        alert(`Error starting detection: ${data.message}`);
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
                        // Switch back to preview
                        document.getElementById('video-stream').style.display = 'none';
                        document.getElementById('video-preview').style.display = 'block';
                        loadPreview();
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
                        
                        // Create inline status layout
                        let angleInfo = '';
                        if (data.angle_tracking_enabled) {
                            const currentAngle = data.current_angle ? data.current_angle.toFixed(1) : 'N/A';
                            const smoothedAngle = data.smoothed_angle ? data.smoothed_angle.toFixed(1) : 'N/A';
                            angleInfo = `
                                <div style="display: flex; flex-direction: column; gap: 4px;">
                                    <span>📐 Target: ${data.target_line || 'N/A'}</span>
                                    <span>🎯 Angle: ${currentAngle}° → ${smoothedAngle}° (filtered)</span>
                                    <span>📊 Overlays: ${data.show_angle_overlay ? '✅' : '❌'} angle, ${data.show_direction_arrow ? '✅' : '❌'} arrow</span>
                                </div>
                            `;
                        }
                        
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
                                ${angleInfo}
                                <span>🔍 Conf: ${data.confidence_threshold}</span>
                                <span>🛰️ YOLO stream: ${data.predict_stream ? 'ON' : 'OFF'}</span>
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
                        // Handle stopped or completed state
                        let statusText = '⏸️ Status: <strong>STOPPED</strong>';
                        if (data.video_completed) {
                            statusText = '✅ Status: <strong>VIDEO COMPLETED</strong>';
                        }
                        
                        // Show final output information if available
                        let outputInfo = '';
                        if (data.final_output_dir || data.output_dir !== 'Not started yet') {
                            const finalDir = data.final_output_dir || data.output_dir;
                            outputInfo = `
                                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
                                    <div style="font-size: 0.85em; color: #666;">
                                        <strong>📁 Final Output Directory:</strong><br>
                                        <a href="file://${outputDir}" style="color: #4CAF50; text-decoration: none; word-break: break-all; font-size: 0.8em;">${finalDir}</a>
                                    </div>
                                    <div style="font-size: 0.75em; color: #999; margin-top: 5px;">
                                        Files: detections.csv, trajectory.csv, behavioral_events.csv, nose_heading_map.html, analysis_plots/
                                    </div>
                                    ${data.saved_video_path ? `
                                        <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                                            <strong>🎬 Saved Video:</strong><br>
                                            <a href="file://${data.saved_video_path}" style="color: #4CAF50; text-decoration: none; word-break: break-all; font-size: 0.75em;">${data.saved_video_path}</a>
                                        </div>
                                    ` : ''}
                                </div>
                            `;
                        }
                        
                        statusDiv.innerHTML = `<p>${statusText}</p>${outputInfo}`;
                    }
                    
                    // Update button states based on server running status (always sync with server)
                    if (data.running && !isRunning) {
                        // Server says running but client doesn't think so - sync up
                        console.log('Syncing: Server running, client stopped - enabling detection mode');
                        isRunning = true;
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('stop-btn').disabled = false;
                    } else if (!data.running && isRunning) {
                        // Server says stopped but client thinks it's running - sync up
                        console.log('Syncing: Server stopped, client running - disabling detection mode');
                        isRunning = false;
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('stop-btn').disabled = true;
                        // Switch back to preview
                        document.getElementById('video-stream').style.display = 'none';
                        document.getElementById('video-preview').style.display = 'block';
                        loadPreview();
                    } else if (!data.running && !isRunning) {
                        // Both agree it's stopped - make sure buttons are in correct state
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('stop-btn').disabled = true;
                        // Ensure we're showing preview mode
                        document.getElementById('video-stream').style.display = 'none';
                        document.getElementById('video-preview').style.display = 'block';
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
            // Always update status to catch when backend stops
            updateStatus();
            // Only update events when running (events only happen during detection)
            if (isRunning) {
                updateEvents();
            }
        }, 1000);
        
        // Initial update - initialize running state first
        initializeRunningState();
        updateEvents();  
        loadVideos();
        // Don't load preview immediately - wait for video selection
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

@app.route('/api/videos')
def get_videos():
    """Get list of available videos from the video directory"""
    try:
        video_dir = CONFIG.video_dir
        if not os.path.exists(video_dir):
            return jsonify({"videos": [], "error": f"Video directory not found: {video_dir}"})
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        videos = []
        
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(video_dir, filename)
                videos.append({
                    "name": filename,
                    "path": video_path,
                    "size": os.path.getsize(video_path) if os.path.exists(video_path) else 0
                })
        
        # Sort by name
        videos.sort(key=lambda x: x['name'])
        
        # Get current video
        current_video = CONFIG.video_path if detector is None else detector.video_path
        
        return jsonify({
            "videos": videos,
            "current": current_video,
            "video_dir": video_dir
        })
    except Exception as e:
        return jsonify({"videos": [], "error": str(e)})

@app.route('/api/set_video', methods=['POST'])
def set_video():
    """Set the video path for the next detection run"""
    global detector
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        
        if not video_path:
            return jsonify({"success": False, "message": "No video path provided"})
        
        if not os.path.exists(video_path):
            return jsonify({"success": False, "message": f"Video file not found: {video_path}"})
        
        if detector and detector.running:
            return jsonify({"success": False, "message": "Cannot change video while detection is running"})
        
        # Update global config
        CONFIG.video_path = video_path
        
        # Update detector if it exists
        if detector:
            detector.set_video_path(video_path)
        
        return jsonify({"success": True, "message": f"Video set to: {os.path.basename(video_path)}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/preview_frame')
def get_preview_frame():
    """Get the first frame of the current video as a preview"""
    try:
        video_path = CONFIG.video_path if detector is None else detector.video_path
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({"success": False, "message": "No valid video selected"})
        
        # Open video and get first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Cannot open video file"})
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"success": False, "message": "Cannot read first frame"})
        
        # Resize frame for preview (max width 800px)
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800.0 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return jsonify({"success": False, "message": "Cannot encode frame"})
        
        # Return as base64 encoded image
        import base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True, 
            "image": f"data:image/jpeg;base64,{frame_base64}",
            "video_info": {
                "path": video_path,
                "name": os.path.basename(video_path),
                "dimensions": f"{width}x{height}"
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    print("✅ All imports successful")
    print("🚀 Starting Head Pose Detection UI - Original Version (Regular YOLO)")
    
    # Print current configuration
    CONFIG.print_config()
    
    print(f"🌐 Server will be available at: http://localhost:{CONFIG.server_port}")
    print(f"🌐 Or access from network: http://{CONFIG.server_host}:{CONFIG.server_port}")
    
    # Start Flask app with configured settings
    app.run(host=CONFIG.server_host, port=CONFIG.server_port, debug=CONFIG.server_debug, threaded=True)
