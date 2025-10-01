"""Live metrics calculation for behavioral analysis."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np
import time


@dataclass
class LiveMetrics:
    """Real-time behavioral metrics."""
    
    # Current state
    current_position: Optional[Tuple[float, float]] = None
    current_speed_px_per_frame: float = 0.0
    current_direction_deg: float = 0.0
    distance_from_reference: float = 0.0
    
    # Counters
    frames_processed: int = 0
    total_distance_traveled: float = 0.0
    events_detected: int = 0
    
    # Time tracking
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    
    # Movement history for calculations
    position_history: deque = field(default_factory=lambda: deque(maxlen=50))
    speed_history: deque = field(default_factory=lambda: deque(maxlen=20))
    
    def update_position(self, position: Tuple[float, float], reference_point: Optional[Tuple[float, float]] = None):
        """Update current position and calculate derived metrics."""
        self.last_update_time = time.time()
        self.frames_processed += 1
        
        # Update position
        prev_position = self.current_position
        self.current_position = position
        self.position_history.append(position)
        
        # Calculate speed if we have previous position
        if prev_position is not None:
            distance = np.sqrt((position[0] - prev_position[0])**2 + (position[1] - prev_position[1])**2)
            self.current_speed_px_per_frame = distance
            self.total_distance_traveled += distance
            self.speed_history.append(distance)
            
            # Calculate direction
            dx = position[0] - prev_position[0]
            dy = position[1] - prev_position[1]
            if dx != 0 or dy != 0:
                self.current_direction_deg = np.degrees(np.arctan2(dy, dx))
        
        # Calculate distance from reference point
        if reference_point is not None:
            self.distance_from_reference = np.sqrt(
                (position[0] - reference_point[0])**2 + (position[1] - reference_point[1])**2
            )
    
    def get_average_speed(self, window_frames: int = 10) -> float:
        """Get average speed over recent frames."""
        if not self.speed_history:
            return 0.0
        recent_speeds = list(self.speed_history)[-window_frames:]
        return sum(recent_speeds) / len(recent_speeds)
    
    def get_smoothed_position(self, window_frames: int = 5) -> Optional[Tuple[float, float]]:
        """Get smoothed position over recent frames."""
        if len(self.position_history) < window_frames:
            return self.current_position
        
        recent_positions = list(self.position_history)[-window_frames:]
        avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
        avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
        return (avg_x, avg_y)
    
    def get_trajectory_length(self) -> float:
        """Get total trajectory length from position history."""
        if len(self.position_history) < 2:
            return 0.0
        
        total_length = 0.0
        positions = list(self.position_history)
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_length += np.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box of trajectory (min_x, min_y, max_x, max_y)."""
        if not self.position_history:
            return None
        
        positions = list(self.position_history)
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def is_stationary(self, threshold_px: float = 5.0, window_frames: int = 10) -> bool:
        """Check if animal is stationary based on recent movement."""
        if len(self.position_history) < window_frames:
            return False
        
        recent_positions = list(self.position_history)[-window_frames:]
        if len(recent_positions) < 2:
            return False
        
        # Calculate variance in positions
        x_coords = [pos[0] for pos in recent_positions]
        y_coords = [pos[1] for pos in recent_positions]
        
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        return (x_variance + y_variance) < threshold_px**2
    
    def get_direction_stability(self, window_frames: int = 10) -> float:
        """Get direction stability (0.0 = very unstable, 1.0 = very stable)."""
        if len(self.position_history) < window_frames + 1:
            return 0.0
        
        recent_positions = list(self.position_history)[-(window_frames + 1):]
        directions = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            if dx != 0 or dy != 0:
                direction = np.degrees(np.arctan2(dy, dx))
                directions.append(direction)
        
        if len(directions) < 2:
            return 0.0
        
        # Calculate circular variance for direction stability
        angles_rad = np.radians(directions)
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
        
        return 1.0 - circular_variance  # Convert to stability score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'current_position': self.current_position,
            'current_speed_px_per_frame': self.current_speed_px_per_frame,
            'current_direction_deg': self.current_direction_deg,
            'distance_from_reference': self.distance_from_reference,
            'frames_processed': self.frames_processed,
            'total_distance_traveled': self.total_distance_traveled,
            'events_detected': self.events_detected,
            'average_speed': self.get_average_speed(),
            'trajectory_length': self.get_trajectory_length(),
            'is_stationary': self.is_stationary(),
            'direction_stability': self.get_direction_stability(),
            'elapsed_time_seconds': self.last_update_time - self.start_time,
        }
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.current_position = None
        self.current_speed_px_per_frame = 0.0
        self.current_direction_deg = 0.0
        self.distance_from_reference = 0.0
        self.frames_processed = 0
        self.total_distance_traveled = 0.0
        self.events_detected = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.position_history.clear()
        self.speed_history.clear()