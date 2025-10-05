#!/usr/bin/env python3
"""
FINAL Head Pose Detection Application
=====================================
CLEAN VERSION - No complex behavioral analysis, just working detection
Key Issues Fixed:
- NoneType errors eliminated  
- Simple detection logic
- Stable streaming
- Visual feedback with confidence threshold
"""

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
    BEHAVIOR_ANALYSIS_AVAILABLE = True
    print("✅ Behavioral analysis imported successfully (with LiveMetrics)")
except ImportError as e:
    print(f"⚠️ Behavioral analysis import failed: {e}")
    BEHAVIOR_ANALYSIS_AVAILABLE = False

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
    (run / "labeled_frames").mkdir(parents=True, exist_ok=True)
    (run / "labels").mkdir(parents=True, exist_ok=True)
    return run

def save_run_config(run_dir: Path, cfg: dict):
    """Save configuration snapshot"""
    with open(run_dir / "run_config.json", "w") as fp:
        json.dump(cfg, fp, indent=2)

def save_yolo_label_txt(path_txt: Path, cls_id: int, bbox_xyxy: Tuple[float,float,float,float], 
                       img_w: int, img_h: int, conf: Optional[float] = None):
    """Save detection in YOLO format"""
    x1,y1,x2,y2 = bbox_xyxy
    bw = x2-x1; bh = y2-y1
    cx = x1 + bw/2.0; cy = y1 + bh/2.0
    nx = cx / img_w; ny = cy / img_h; nw = bw / img_w; nh = bh / img_h
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(path_txt, "w") as f:
        if conf is None:
            f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {conf:.6f}\n")

def save_labeled_frame(path_img: Path, frame: np.ndarray, max_w: int = 900):
    """Save processed frame image"""
    h, w = frame.shape[:2]
    if w > max_w and w > 0:
        scale = max_w/float(w)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path_img), frame)

# ═══════════════════════════════════════════════════════════════════════════════

class SimpleHeadPoseDetector:
    def __init__(self, model_path, video_path):
        """Initialize with ultra-simple detection approach"""
        self.model_path = model_path
        self.video_path = video_path
        self.model = None
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 10
        self.detection_count = 0
        
        # Initialize behavioral detector for trajectory-based analysis
        if BEHAVIOR_ANALYSIS_AVAILABLE:
            config = BehaviorConfig(
                min_moving_frames=3,      # Need 3+ consecutive frames to declare moving
                stop_threshold=300.0,      # 300px threshold for movement detection
                min_stationary_frames=3   # Need 5+ frames to declare stationary
            )
            self.behavior_detector = BehaviorDetector(config)
            self.live_metrics = LiveMetrics()  # Initialize LiveMetrics for comprehensive tracking
            print("✅ Behavioral detector initialized with trajectory-based movement detection + LiveMetrics")
        else:
            self.behavior_detector = None
            self.live_metrics = None
            print("⚠️ Using simple behavioral events (no trajectory analysis)")
        
        # Store behavioral events (newest first)
        self.behavioral_events = []
        self.events_lock = threading.Lock()
        
        # Detection settings
        self.CONFIDENCE_THRESHOLD = 0.2  # Very low threshold to see any detections
        
        # File organization (from video_pose_pipeline.py)
        self.output_dir = Path("../output/detections")  # Save to lizard-tracking/output/detections
        self.run_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.save_every_n = 10  # Save labeled frame every N detections
        
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
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
            print(f"✅ Video loaded: {self.total_frames} frames at {self.fps} FPS")
            return True
        except Exception as e:
            print(f"❌ Video loading failed: {e}")
            return False
    
    def add_behavioral_event(self, event_type, description):
        """Add behavioral event to the list (newest first)"""
        current_time = datetime.now().strftime("%H:%M:%S")
        video_second = round(self.current_frame_number / self.fps, 1)
        
        event = {
            "time": current_time,
            "video_second": video_second,
            "type": event_type,
            "description": description
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
        """Simplified pose detection"""
        if self.model is None:
            return frame, []
        
        try:
            # Run detection
            results = self.model(frame, verbose=False, conf=self.CONFIDENCE_THRESHOLD)
            
            if results and len(results) > 0:
                self.detection_count += 1
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
                                print(f"🔍 DEBUG: Found {len(kpts)} keypoints: {kpts}")
                                if len(kpts) >= 3:
                                    # Let's try different mappings to see which is correct
                                    kpt0 = (kpts[0][0], kpts[0][1]) if kpts[0][0] > 0 and kpts[0][1] > 0 else None
                                    kpt1 = (kpts[1][0], kpts[1][1]) if kpts[1][0] > 0 and kpts[1][1] > 0 else None
                                    kpt2 = (kpts[2][0], kpts[2][1]) if kpts[2][0] > 0 and kpts[2][1] > 0 else None
                                    
                                    print(f"🔍 DEBUG: kpt0={kpt0}, kpt1={kpt1}, kpt2={kpt2}")
                                    
                                    # Fix: Now nose is on right ear, so let's try: kpt0=left_ear, kpt1=right_ear, kpt2=nose
                                    nose = kpt2      # Third keypoint is actually the nose
                                    ear_left = kpt0  # First keypoint is left ear
                                    ear_right = kpt1 # Second keypoint is right ear
                            
                            # Draw with proper parameters
                            frame = draw_head_pose(frame, (x1, y1, x2, y2), nose, ear_left, ear_right, confidence)
                            print(f"✅ Drew keypoints: nose={nose is not None}, ears={ear_left is not None and ear_right is not None}")
                        else:
                            frame = self.draw_simple_detection(frame, results)
                            print("⚠️ No boxes found, using simple drawing")
                    except Exception as e:
                        print(f"⚠️ draw_head_pose failed: {e}, using simple drawing")
                        frame = self.draw_simple_detection(frame, results)
                else:
                    frame = self.draw_simple_detection(frame, results)
                
                # Process behavioral analysis for trajectory-based movement detection
                self.process_behavioral_detection(results, frame)
                
                # Add simple detection event occasionally
                if self.detection_count % 30 == 1:  # Every ~2 seconds at 15fps
                    self.add_behavioral_event("detection", f"Head detected (conf: {self.CONFIDENCE_THRESHOLD})")
            
            else:
                print(f"❌ Frame {self.current_frame_number}: No detections")
            
            return frame, results
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return frame, []
    
    def process_behavioral_detection(self, results, frame):
        """Process detection results for trajectory-based behavioral analysis + LiveMetrics"""
        if not self.behavior_detector or not results or len(results) == 0:
            return
        
        try:
            # Extract position from first detection
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Get center position of first detection
                box = boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Update LiveMetrics with position (tracks speed, distance, etc.)
                if self.live_metrics:
                    # Use rightmost screen position as reference point for distance calculation
                    reference_point = (800, 300)  # Rightmost edge center
                    self.live_metrics.update_position((center_x, center_y), reference_point)
                
                # Get video timestamp
                video_timestamp = self.current_frame_number / self.fps
                
                # Process frame with behavioral detector (trajectory-based analysis)
                events = self.behavior_detector.process_frame(
                    position=(center_x, center_y),
                    frame_number=self.current_frame_number
                )
                
                # Update event count in metrics
                if self.live_metrics and events:
                    self.live_metrics.events_detected += len(events)
                
                # Add behavioral events for movement detection
                for event in events:
                    if event.event_type == EventType.STOP_START:
                        position_desc = self.get_position_description(center_x, center_y)
                        self.add_behavioral_event(
                            "movement", 
                            f"moving to {position_desc} (video: {video_timestamp:.1f}s)"
                        )
                        print(f"ℹ️ Movement STARTED to ({center_x}, {center_y}) - {position_desc}")
                    elif event.event_type == EventType.STOP_END:
                        position_desc = self.get_position_description(center_x, center_y)
                        self.add_behavioral_event(
                            "stationary", 
                            f"Stationary at {position_desc} (video: {video_timestamp:.1f}s)"
                        )
                        print(f"ℹ️ Now STATIONARY at ({center_x}, {center_y}) - {position_desc}")
                        
        except Exception as e:
            print(f"❌ Behavioral detection error: {e}")
    
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
    
    def start_detection(self):
        """Start the detection process"""
        if not self.load_model() or not self.load_video():
            return False
        
        self.running = True
        print("🎬 Detection started")
        
        # Setup file organization (from video_pose_pipeline.py)
        self.setup_file_organization()
        
        # Add initial event
        self.add_behavioral_event("system", "Detection started")
        
        def detection_loop():
            frame_count = 0
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video - stop detection
                    print("📹 End of video reached")
                    self.running = False
                    break
                
                self.current_frame_number = frame_count
                frame_count += 1
                
                # Add timestamp overlay (keep original)
                timestamp = datetime.now().strftime("%H:%M:%S")
                video_time = f"Video: {self.current_frame_number/self.fps:.1f}s"
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, video_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Detect poses
                frame, results = self.detect_poses(frame)
                
                # Save detection data (like video_pose_pipeline.py)
                self.save_detection_data(results, frame_count, frame_width, frame_height, video_fps, frame)
                
                # Save metrics snapshot periodically
                self.save_metrics_snapshot(frame_count)
                
                # Store latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
            
            # Cleanup on end
            self.cleanup_files()
        
        # Start detection in background thread
        self.detection_thread = threading.Thread(target=detection_loop, daemon=True)
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
            "fps": self.fps,
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
    
    def save_detection_data(self, results, frame_count, frame_width, frame_height, video_fps, frame):
        """Save detection data like video_pose_pipeline.py - extracts nose x,y from YOLO results"""
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
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                
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
                            pose.nose = (float(kpt2[0]), float(kpt2[1]))
                        
                        # Save ear positions
                        if kpt0[0] > 0 and kpt0[1] > 0:
                            pose.ear_left = (float(kpt0[0]), float(kpt0[1]))
                        if kpt1[0] > 0 and kpt1[1] > 0:
                            pose.ear_right = (float(kpt1[0]), float(kpt1[1]))
        
        # Log detection to CSV (includes nose x,y if available)
        obs = PoseObservation(frame_count, pose)
        self.csv_writer.writerow(obs.as_row())
        self.csv_file.flush()  # Ensure data is written
        
        # Save files if detection found
        if pose is not None:
            # Save labeled frame periodically
            if self.detection_count % self.save_every_n == 0:
                frame_name = f"frame{frame_count:08d}.jpg"
                save_labeled_frame(self.run_dir / "labeled_frames" / frame_name, frame)
            
            # Save YOLO label
            label_name = f"frame{frame_count:08d}.txt"
            save_yolo_label_txt(
                self.run_dir / "labels" / label_name, 
                0, pose.bbox_xyxy, frame_width, frame_height, pose.conf
            )
    
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
        
        if self.run_dir:
            print(f"✅ Processing complete!")
            print(f"📊 Total detections: {self.detection_count}")
            print(f"📁 Results saved to: {self.run_dir}")
    
    def stop_detection(self):
        """Stop detection"""
        self.running = False
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.add_behavioral_event("system", "Detection stopped")
    
    def get_latest_frame(self):
        """Get the latest processed frame"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
    
    def get_status(self):
        """Get current detection status"""
        device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        memory_info = f"GPU: {torch.cuda.memory_allocated()/1024**2:.0f}MB" if torch.cuda.is_available() else "CPU Mode"
        
        return {
            "running": self.running,
            "device": device_info,
            "memory": memory_info,
            "fps": self.fps,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "output_dir": str(self.run_dir) if self.run_dir else "Not started yet"
        }
    
    def get_behavioral_events(self):
        """Get behavioral events (already ordered newest first)"""
        with self.events_lock:
            return self.behavioral_events.copy()

# Flask App
app = Flask(__name__)

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
            padding-bottom: 10px;
            margin-bottom: 20px;
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
        .controls {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status {
            background: #e7f5e7;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .behavioral-log {
            background: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            max-height: 300px;
            overflow-y: auto;
        }
        .event-item {
            background: white;
            margin: 5px 0;
            padding: 8px;
            border-radius: 5px;
            border-left: 3px solid #4CAF50;
            font-size: 0.9em;
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
            <h1>🦎 FINAL Head Pose Detection Interface</h1>
            <p>Real-time lizard head pose tracking with behavioral analysis</p>
        </div>
        
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="video-stream" src="/video_feed" alt="Video Stream" style="width: 100%; max-width: 800px;">
                </div>
            </div>
            
            <div class="controls-section">
                <div class="controls">
                    <h3>🎮 Controls</h3>
                    <button id="start-btn" onclick="startDetection()">▶️ Start Detection</button>
                    <button id="stop-btn" onclick="stopDetection()" class="stop-btn" disabled>⏹️ Stop Detection</button>
                </div>
                
                <div class="status">
                    <h3>📊 Status</h3>
                    <div id="status-info">
                        <p>🔄 Ready to start...</p>
                    </div>
                </div>
                
                <div class="behavioral-log">
                    <h3>🎯 Behavioral Events (Newest First)</h3>
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
                        statusDiv.innerHTML = `
                            <p>✅ Status: <strong>RUNNING</strong></p>
                            <p>🖥️ Device: ${data.device || 'Unknown'}</p>
                            <p>💾 Memory: ${data.memory || 'Unknown'}</p>
                            <p>⚡ FPS: ${data.fps}</p>
                            <p>🔍 Confidence: ${data.confidence_threshold}</p>
                            <p>📁 Events: output/behavioral_events.log</p>
                            <p>📁 Trajectory: output/trajectory.log</p>
                        `;
                    } else {
                        statusDiv.innerHTML = '<p>⏸️ Status: <strong>STOPPED</strong></p>';
                    }
                });
        }
        
        function updateEvents() {
            fetch('/api/events')
                .then(response => response.json())
                .then(data => {
                    const eventsDiv = document.getElementById('behavioral-events');
                    if (data.length === 0) {
                        eventsDiv.innerHTML = '<p>No events yet...</p>';
                        return;
                    }
                    
                    eventsDiv.innerHTML = data.map(event => `
                        <div class="event-item">
                            <div class="event-time">${event.time}</div>
                            <div class="event-video-time">Video: ${event.video_second}s</div>
                            <div><strong>${event.type}:</strong> ${event.description}</div>
                        </div>
                    `).join('');
                });
        }
        
        // Auto-update status and events
        setInterval(() => {
            if (isRunning) {
                updateStatus();
            }
            updateEvents();
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
        print("✅ Video streaming started")
        while True:
            if detector and detector.running:
                frame = detector.get_latest_frame()
                if frame is not None:
                    # Encode frame
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1/15)  # ~15 FPS
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_detection():
    global detector
    try:
        if detector and detector.running:
            return jsonify({"success": False, "message": "Already running"})
        
        # Initialize detector
        video_path = "/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/pose-head/videos/top_20250916T150021.mp4"
        model_path = "/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/autogenerate/best.pt"
        
        detector = SimpleHeadPoseDetector(model_path, video_path)
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
    print("📹 Video: top_20250916T150021.mp4")
    print("🤖 Model: best.pt")
    print("🌐 Server will be available at: http://localhost:8089")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8089, debug=False, threaded=True)