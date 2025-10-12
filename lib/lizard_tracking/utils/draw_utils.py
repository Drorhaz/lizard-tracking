#!/usr/bin/env python3
"""
Drawing utilities for lizard tracking visualizations
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Any

# Color definitions (BGR format)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_head_pose(frame: np.ndarray, bbox_xyxy: Tuple[float, float, float, float], 
                   nose: Optional[Tuple[float, float]] = None,
                   ear_left: Optional[Tuple[float, float]] = None,
                   ear_right: Optional[Tuple[float, float]] = None,
                   confidence: Optional[float] = None) -> np.ndarray:
    """
    Draw head pose detection overlay on frame
    
    Args:
        frame: Input frame (BGR format)
        bbox_xyxy: Bounding box coordinates [x1, y1, x2, y2]
        nose: Nose keypoint coordinates [x, y]
        ear_left: Left ear keypoint coordinates [x, y]  
        ear_right: Right ear keypoint coordinates [x, y]
        confidence: Optional confidence score to display
        
    Returns:
        Frame with drawn overlay
    """
    # Draw bounding box
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    
    # Draw keypoints
    if nose is not None:
        cv2.circle(frame, (int(nose[0]), int(nose[1])), 6, RED, -1, lineType=cv2.LINE_AA)
    
    if ear_left is not None and ear_right is not None:
        cv2.circle(frame, (int(ear_left[0]), int(ear_left[1])), 5, BLUE, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (int(ear_right[0]), int(ear_right[1])), 5, BLUE, -1, lineType=cv2.LINE_AA)
        
        # Draw line from nose to ear midpoint
        if nose is not None:
            ex = int(0.5 * (ear_left[0] + ear_right[0]))
            ey = int(0.5 * (ear_left[1] + ear_right[1]))
            cv2.line(frame, (int(nose[0]), int(nose[1])), (ex, ey), YELLOW, 2, lineType=cv2.LINE_AA)
    
    # Draw confidence text if provided
    if confidence is not None:
        bbox_txt = f"HEAD {confidence:.3f}"
        # Black outline for better readability
        cv2.putText(frame, bbox_txt, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)
        # Green text
        cv2.putText(frame, bbox_txt, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1, cv2.LINE_AA)
    
    return frame


def draw_head_pose_from_object(frame: np.ndarray, head_pose_obj: Any) -> np.ndarray:
    """
    Draw head pose from a head pose object (HeadPose, SimpleDetection, etc.)
    
    Args:
        frame: Input frame (BGR format)
        head_pose_obj: Object with bbox, nose_tip, left_ear, right_ear, confidence attributes
        
    Returns:
        Frame with drawn overlay
    """
    if head_pose_obj is None:
        return frame
        
    # Extract coordinates based on object type
    bbox = getattr(head_pose_obj, 'bbox_xyxy', None) or getattr(head_pose_obj, 'bbox', None)
    nose = getattr(head_pose_obj, 'nose_tip', None) or getattr(head_pose_obj, 'nose', None)
    ear_left = getattr(head_pose_obj, 'left_ear', None) or getattr(head_pose_obj, 'ear_left', None)
    ear_right = getattr(head_pose_obj, 'right_ear', None) or getattr(head_pose_obj, 'ear_right', None)
    confidence = getattr(head_pose_obj, 'confidence', None) or getattr(head_pose_obj, 'conf', None)
    
    if bbox is None:
        return frame
        
    return draw_head_pose(frame, bbox, nose, ear_left, ear_right, confidence)


def draw_no_detection(frame: np.ndarray, text: str = "No Detection") -> np.ndarray:
    """
    Draw "No Detection" text overlay on frame
    
    Args:
        frame: Input frame (BGR format)
        text: Text to display
        
    Returns:
        Frame with text overlay
    """
    h, w = frame.shape[:2]
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2
    
    # Black outline
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLACK, 3, cv2.LINE_AA)
    # White text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    
    return frame


def draw_processing_info(frame: np.ndarray, mode: str, fps: float, frame_count: int, 
                        detection_count: int) -> np.ndarray:
    """
    Draw processing information overlay
    
    Args:
        frame: Input frame (BGR format)
        mode: Processing mode (offline/realtime/preview)
        fps: Current FPS
        frame_count: Current frame count
        detection_count: Total detections so far
        
    Returns:
        Frame with info overlay
    """
    h, w = frame.shape[:2]
    
    # Create info text
    info_lines = [
        f"Mode: {mode.upper()}",
        f"FPS: {fps:.1f}",
        f"Frame: {frame_count}",
        f"Detections: {detection_count}"
    ]
    
    # Draw background rectangle
    bg_height = len(info_lines) * 30 + 20
    cv2.rectangle(frame, (10, 10), (250, bg_height), BLACK, -1)
    cv2.rectangle(frame, (10, 10), (250, bg_height), GREEN, 2)
    
    # Draw text lines
    for i, line in enumerate(info_lines):
        y = 35 + i * 25
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    
    return frame


def draw_trajectory_line(frame: np.ndarray, points: list, color: Tuple[int, int, int] = YELLOW,
                        thickness: int = 2) -> np.ndarray:
    """
    Draw trajectory line connecting multiple points
    
    Args:
        frame: Input frame (BGR format)
        points: List of (x, y) coordinate tuples
        color: Line color (BGR format)
        thickness: Line thickness
        
    Returns:
        Frame with trajectory line
    """
    if len(points) < 2:
        return frame
        
    for i in range(1, len(points)):
        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_behavioral_event(frame: np.ndarray, event_type: str, position: Tuple[float, float],
                         duration: float = 2.0) -> np.ndarray:
    """
    Draw behavioral event notification
    
    Args:
        frame: Input frame (BGR format)
        event_type: Type of event (approach, retreat, stop)
        position: Position where event occurred
        duration: How long to show the event (not used in single frame)
        
    Returns:
        Frame with event overlay
    """
    # Event colors
    event_colors = {
        'approach': (0, 255, 0),    # Green
        'retreat': (0, 0, 255),     # Red  
        'stop': (0, 165, 255),      # Orange
        'close_to_target': (255, 255, 0),  # Cyan
        'far_from_target': (255, 0, 255),  # Magenta
    }
    
    color = event_colors.get(event_type.lower(), WHITE)
    text = event_type.upper()
    
    # Draw circle at position
    pos = (int(position[0]), int(position[1]))
    cv2.circle(frame, pos, 20, color, 3, cv2.LINE_AA)
    
    # Draw text near position
    text_pos = (pos[0] + 25, pos[1] - 10)
    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    
    return frame