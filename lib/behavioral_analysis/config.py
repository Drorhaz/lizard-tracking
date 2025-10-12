"""Configuration classes for behavioral analysis."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class BehaviorConfig:
    """Configuration for behavioral analysis parameters."""
    
    # Distance thresholds (pixels, since no calibration by default)
    close_distance_px: float = 200.0      # Pixels considered "close to screen/target"
    far_distance_px: float = 600.0        # Pixels considered "far from screen/target" 
    approach_threshold_px: float = 300.0   # Distance threshold for detecting approach
    retreat_threshold_px: float = 300.0    # Distance threshold for detecting retreat
    approach_threshold: float = 200.0     # Alias - distance to consider "approaching"
    retreat_threshold: float = 400.0      # Alias - distance to consider "retreating"
    hysteresis_px: float = 10.0           # Hysteresis band to prevent flickering events
    
    # Speed/movement thresholds  
    stop_speed_px_per_frame: float = 2.0  # Below this speed = stopped
    stop_threshold: float = 2.0           # Alias for stop_speed_px_per_frame
    velocity_window_frames: int = 6       # Number of frames for velocity calculation
    min_stationary_frames: int = 5        # Minimum frames to confirm stop state
    min_moving_frames: int = 3            # Minimum frames to confirm movement
    
    # Event detection toggles (configurable via web UI)
    detect_approach: bool = True
    detect_retreat: bool = True  
    detect_stop: bool = True
    detect_close_to_target: bool = True
    detect_far_from_target: bool = True
    detect_fast_movement: bool = True
    detect_direction_change: bool = True
    
    # Event thresholds
    fast_movement_threshold_px: float = 100.0  # Speed considered "fast"
    direction_change_angle_deg: float = 45.0   # Angle change for direction change event
    
    # Export and data saving settings
    export_csv: bool = True
    export_trajectory: bool = True
    save_labeled_frames: bool = True
    save_every_n_frames: int = 10
    
    # Reference point (for distance calculations)
    reference_point: Optional[tuple[float, float]] = None  # (x, y) reference point
    
    # Calibration (optional - for real-world measurements)
    pixels_per_cm: Optional[float] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BehaviorConfig':
        """Create config from dictionary (e.g., from web form or JSON)."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    def enable_real_world_measurements(self, pixels_per_cm: float, reference_point: tuple[float, float]):
        """Enable real-world measurements with calibration."""
        self.pixels_per_cm = pixels_per_cm
        self.reference_point = reference_point
        
        # Convert pixel thresholds to real-world if calibration available
        if pixels_per_cm > 0:
            self.close_distance_px = 5.0 * pixels_per_cm  # 5cm
            self.far_distance_px = 20.0 * pixels_per_cm   # 20cm
            self.approach_threshold_px = 2.0 * pixels_per_cm  # 2cm
            self.retreat_threshold_px = 2.0 * pixels_per_cm   # 2cm