#!/usr/bin/env python3
"""
Core Frame Processor - Unified pipeline for real-time and offline processing.
Inspired by PreyTouch's ImageHandler architecture.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable
from pathlib import Path
import numpy as np
import cv2
import csv
from datetime import datetime

# Import behavioral analysis
try:
    from behavioral_analysis import BehaviorDetector, BehaviorConfig, BehaviorEvent
    BEHAVIORAL_ANALYSIS_AVAILABLE = True
except ImportError:
    BEHAVIORAL_ANALYSIS_AVAILABLE = False
    print("Warning: behavioral_analysis not available")


@dataclass
class ProcessorConfig:
    """Configuration for frame processor."""
    # Detection settings
    model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    imgsz: int = 640
    device: str = "0"
    
    # Output settings
    output_dir: Path = Path("output")
    save_trajectory: bool = True
    save_labels: bool = False
    save_frames: bool = False
    
    # Behavioral analysis settings
    enable_behavioral_analysis: bool = True
    reference_point: Optional[Tuple[float, float]] = None  # (x, y) or None for rightmost edge
    screen_location: str = "right"  # "left", "right", "top", "bottom"
    approach_threshold_px: float = 300.0
    retreat_threshold_px: float = 300.0
    stop_speed_threshold: float = 2.0
    
    # Processing settings
    process_every_n_frames: int = 1  # Process every frame by default
    fps_target: float = 30.0


@dataclass
class HeadPose:
    """Head pose detection result."""
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    nose: Optional[Tuple[float, float]] = None
    ear_left: Optional[Tuple[float, float]] = None
    ear_right: Optional[Tuple[float, float]] = None


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    frame_number: int
    timestamp: float
    original_frame: np.ndarray
    display_frame: np.ndarray  # With overlays drawn
    pose: Optional[HeadPose]
    behavioral_events: list  # List of BehaviorEvent objects
    distance_from_reference: Optional[float] = None
    speed_px_per_frame: Optional[float] = None


class YOLOPoseModel:
    """YOLO pose detection wrapper."""
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device: str = "0"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
    
    def predict(self, frame: np.ndarray) -> list[HeadPose]:
        """Run inference on frame."""
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )[0]
        
        poses = []
        if results is None or not hasattr(results, 'boxes') or len(results.boxes) == 0:
            return poses
        
        boxes = results.boxes
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            conf = float(confs[i])
            pose = HeadPose(bbox_xyxy=(x1, y1, x2, y2), conf=conf)
            
            # Extract keypoints if available
            try:
                if hasattr(results, 'keypoints') and results.keypoints is not None:
                    kpts = results.keypoints.xy.cpu().numpy()
                    if i < len(kpts) and len(kpts[i]) >= 3:
                        pose.nose = tuple(map(float, kpts[i][0]))
                        pose.ear_left = tuple(map(float, kpts[i][1]))
                        pose.ear_right = tuple(map(float, kpts[i][2]))
            except Exception:
                pass
            
            poses.append(pose)
        
        return poses


class FrameProcessor:
    """
    Unified frame processor for real-time and offline analysis.
    Handles detection, behavioral analysis, and data logging.
    """
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.frame_count = 0
        self.start_time = datetime.now()
        
        # Initialize model
        self.model = YOLOPoseModel(
            config.model_path,
            imgsz=config.imgsz,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            device=config.device
        )
        
        # Initialize behavioral analysis
        self.behavior_detector = None
        if config.enable_behavioral_analysis and BEHAVIORAL_ANALYSIS_AVAILABLE:
            self._setup_behavioral_analysis()
        
        # Initialize output paths
        self._setup_output_paths()
        
        # Initialize trajectory CSV
        if config.save_trajectory:
            self._init_trajectory_csv()
    
    def _setup_behavioral_analysis(self):
        """Initialize behavioral analysis with config."""
        behavior_config = BehaviorConfig(
            approach_threshold_px=self.config.approach_threshold_px,
            retreat_threshold_px=self.config.retreat_threshold_px,
            stop_speed_px_per_frame=self.config.stop_speed_threshold,
            detect_approach=True,
            detect_retreat=True,
            detect_stop=True,
            reference_point=self.config.reference_point,
            min_stationary_frames=5,
            min_moving_frames=3,
            hysteresis_px=10.0
        )
        self.behavior_detector = BehaviorDetector(behavior_config)
        print(f"✅ Behavioral analysis initialized (threshold={self.config.approach_threshold_px}px)")
    
    def _setup_output_paths(self):
        """Create output directories."""
        self.output_dir = Path(self.config.output_dir)
        self.trajectory_dir = self.output_dir / "trajectory"
        self.labels_dir = self.output_dir / "labels"
        self.frames_dir = self.output_dir / "frames"
        
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_labels:
            self.labels_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_trajectory_csv(self):
        """Initialize trajectory CSV file."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.trajectory_dir / f"trajectory_{timestamp}.csv"
        
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame', 'timestamp', 'elapsed_sec',
            'head_x', 'head_y', 'confidence',
            'distance_from_edge', 'speed_px_per_frame',
            'event_type', 'event_name'
        ])
        print(f"📊 Trajectory CSV: {self.csv_path}")
    
    def _calculate_reference_point(self, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate reference point based on screen location."""
        h, w = frame_shape[:2]
        
        if self.config.reference_point is not None:
            return self.config.reference_point
        
        # Calculate based on screen location
        location_map = {
            "right": (w - 1, h / 2),
            "left": (0, h / 2),
            "top": (w / 2, 0),
            "bottom": (w / 2, h - 1)
        }
        
        return location_map.get(self.config.screen_location, (w - 1, h / 2))
    
    def process_frame(self, frame: np.ndarray, frame_number: Optional[int] = None) -> FrameResult:
        """
        Process a single frame with detection and behavioral analysis.
        
        Args:
            frame: Input frame (BGR format)
            frame_number: Optional explicit frame number (otherwise auto-incremented)
        
        Returns:
            FrameResult with all processing outputs
        """
        if frame_number is None:
            frame_number = self.frame_count
            self.frame_count += 1
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Run detection
        poses = self.model.predict(frame)
        best_pose = max(poses, key=lambda p: p.conf) if poses else None
        
        # Draw overlay
        display_frame = self._draw_overlay(frame.copy(), best_pose)
        
        # Behavioral analysis
        behavioral_events = []
        distance = None
        speed = None
        
        if best_pose and self.behavior_detector:
            # Update reference point if not set
            if self.behavior_detector.config.reference_point is None:
                ref_point = self._calculate_reference_point(frame.shape)
                self.behavior_detector.config.reference_point = ref_point
            
            # Process frame
            events = self.behavior_detector.process_frame(best_pose, frame_number)
            behavioral_events = events
            
            # Get metrics
            distance = self.behavior_detector.metrics.current_distance
            speed = self.behavior_detector.metrics.current_speed
        
        # Save outputs
        if self.config.save_trajectory:
            self._write_trajectory_row(
                frame_number, elapsed, best_pose, 
                distance, speed, behavioral_events
            )
        
        if self.config.save_labels and best_pose:
            self._save_label(frame_number, best_pose, frame.shape)
        
        if self.config.save_frames and best_pose:
            self._save_frame(frame_number, display_frame)
        
        return FrameResult(
            frame_number=frame_number,
            timestamp=elapsed,
            original_frame=frame,
            display_frame=display_frame,
            pose=best_pose,
            behavioral_events=behavioral_events,
            distance_from_reference=distance,
            speed_px_per_frame=speed
        )
    
    def _draw_overlay(self, frame: np.ndarray, pose: Optional[HeadPose]) -> np.ndarray:
        """Draw detection overlay on frame."""
        if pose is None:
            return frame
        
        x1, y1, x2, y2 = map(int, pose.bbox_xyxy)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw nose
        if pose.nose:
            nose = (int(pose.nose[0]), int(pose.nose[1]))
            cv2.circle(frame, nose, 6, (0, 0, 255), -1)
        
        # Draw ears and head direction line
        if pose.ear_left and pose.ear_right and pose.nose:
            left = (int(pose.ear_left[0]), int(pose.ear_left[1]))
            right = (int(pose.ear_right[0]), int(pose.ear_right[1]))
            mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
            
            cv2.circle(frame, left, 5, (255, 0, 0), -1)
            cv2.circle(frame, right, 5, (255, 0, 0), -1)
            cv2.line(frame, (int(pose.nose[0]), int(pose.nose[1])), mid, (0, 255, 255), 2)
        
        # Draw confidence
        label = f"HEAD {pose.conf:.3f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        return frame
    
    def _write_trajectory_row(self, frame_num: int, elapsed: float, 
                              pose: Optional[HeadPose], distance: Optional[float],
                              speed: Optional[float], events: list):
        """Write trajectory data to CSV."""
        timestamp = datetime.now().isoformat()
        
        # Extract position
        head_x = head_y = conf = ""
        if pose:
            if pose.nose:
                head_x, head_y = pose.nose
            else:
                # Fallback to bbox center
                x1, y1, x2, y2 = pose.bbox_xyxy
                head_x = (x1 + x2) / 2
                head_y = (y1 + y2) / 2
            conf = pose.conf
        
        # Extract event info
        event_type = event_name = ""
        if events:
            event = events[0]  # Take first event
            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            event_name = event_type.replace('_', ' ').title()
        
        self.csv_writer.writerow([
            frame_num, timestamp, f"{elapsed:.3f}",
            head_x, head_y, conf,
            f"{distance:.2f}" if distance is not None else "",
            f"{speed:.2f}" if speed is not None else "",
            event_type, event_name
        ])
        self.csv_file.flush()
    
    def _save_label(self, frame_num: int, pose: HeadPose, frame_shape: Tuple[int, int]):
        """Save YOLO format label."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = pose.bbox_xyxy
        
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        label_file = self.labels_dir / f"frame{frame_num:08d}.txt"
        with open(label_file, 'w') as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {pose.conf:.6f}\n")
    
    def _save_frame(self, frame_num: int, frame: np.ndarray):
        """Save frame image."""
        frame_file = self.frames_dir / f"frame{frame_num:08d}.jpg"
        cv2.imwrite(str(frame_file), frame)
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            print(f"✅ Trajectory saved: {self.csv_path}")
