"""Advanced configuration for trajectory-based behavioral analysis with arena mapping."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Type aliases (strings for Python 3.7 compatibility)
TargetLine = str  # 'left' | 'right' | 'top' | 'bottom'
ArenaPhase = str  # 'approaching' | 'retreating' | 'resting'
BandLabel = str   # 'near' | 'middle' | 'far'
ArenaSide = str   # 'RIGHTSIDE' | 'LEFTSIDE'
MotionType = str  # 'head-only' | 'whole-body' | 'unknown'


@dataclass
class AdvancedBehaviorConfig:
    """Configuration for advanced trajectory analysis with arena mapping."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1) TARGET LINE & ARENA ORIENTATION
    # ═══════════════════════════════════════════════════════════════════════
    target_line: TargetLine = 'right'  # 'left' | 'right' | 'top' | 'bottom'
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2) NEAR / MIDDLE / FAR BANDS (normalized distance [0,1])
    # ═══════════════════════════════════════════════════════════════════════
    near_max: float = 0.20        # distance ≤ 0.20 → near
    middle_max: float = 0.30      # 0.20 < distance ≤ 0.30 → middle (thin buffer)
                                  # distance > 0.30 → far
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3) MOTION DETECTION THRESHOLDS (normalized units)
    # ═══════════════════════════════════════════════════════════════════════
    advance_threshold: float = 0.002   # Δd < -0.002 → approaching
    retreat_threshold: float = 0.002   # Δd > +0.002 → retreating
    
    # Directional qualifiers (normalized by frame size)
    x_dir_thresh_norm: float = 0.01    # X-axis movement threshold (as fraction of width)
    y_dir_thresh_norm: float = 0.01    # Y-axis movement threshold (as fraction of height)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 4) HEAD-ONLY vs WHOLE-BODY MOVEMENT (normalized by frame diagonal)
    # ═══════════════════════════════════════════════════════════════════════
    head_only_thresh_norm: float = 0.005   # Head displacement threshold
    body_move_thresh_norm: float = 0.010   # Body displacement threshold
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5) MISSING DETECTION HANDLING
    # ═══════════════════════════════════════════════════════════════════════
    lookback_window: int = 5  # Frames to look back for valid detection
    
    # ═══════════════════════════════════════════════════════════════════════
    # 6) PLOTTING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    arrow_length_norm: float = 0.05       # Arrow length for heading visualization
    plot_colorscale: str = 'Viridis'      # Plotly colorscale for time encoding
    plot_every_n_frames: int = 5          # Subsample frames for cleaner plot
    
    # ═══════════════════════════════════════════════════════════════════════
    # 7) OUTPUT PATHS
    # ═══════════════════════════════════════════════════════════════════════
    output_trajectory_dir: str = '../output/trajectory'
    output_events_dir: str = '../output/events'
    output_plots_dir: str = '../output/plots'
    
    # ═══════════════════════════════════════════════════════════════════════
    # 8) LEGACY COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════════
    min_moving_frames: int = 3        # For compatibility with old BehaviorDetector
    stop_threshold: float = 30.0      # Legacy stop detection
    min_stationary_frames: int = 5    # Legacy stationary detection
    
    def get_band(self, dist_norm: float) -> BandLabel:
        """Get band label from normalized distance."""
        if dist_norm <= self.near_max:
            return 'near'
        elif dist_norm <= self.middle_max:
            return 'middle'
        else:
            return 'far'
    
    def get_arena_side(self, x: float, y: float, frame_width: int, frame_height: int) -> ArenaSide:
        """
        Get arena side (RIGHTSIDE/LEFTSIDE) based on target line and position.
        
        Mapping (from spec §1):
        - right line: y < H/2 → LEFTSIDE (top), y ≥ H/2 → RIGHTSIDE (bottom)
        - left line:  y < H/2 → RIGHTSIDE (top), y ≥ H/2 → LEFTSIDE (bottom)
        - top line:   x < W/2 → LEFTSIDE (left), x ≥ W/2 → RIGHTSIDE (right)
        - bottom line: x < W/2 → RIGHTSIDE (left), x ≥ W/2 → LEFTSIDE (right)
        """
        if self.target_line == 'right':
            return 'RIGHTSIDE' if y >= frame_height / 2 else 'LEFTSIDE'
        elif self.target_line == 'left':
            return 'LEFTSIDE' if y >= frame_height / 2 else 'RIGHTSIDE'
        elif self.target_line == 'top':
            return 'RIGHTSIDE' if x >= frame_width / 2 else 'LEFTSIDE'
        else:  # bottom
            return 'LEFTSIDE' if x >= frame_width / 2 else 'RIGHTSIDE'
    
    def compute_distance_to_target(self, x: float, y: float, 
                                   frame_width: int, frame_height: int) -> tuple[float, float]:
        """
        Compute distance to target line in pixels and normalized [0,1].
        
        Returns: (distance_px, distance_norm)
        """
        if self.target_line == 'right':
            d_px = (frame_width - 1) - x
            d_norm = d_px / frame_width
        elif self.target_line == 'left':
            d_px = x
            d_norm = d_px / frame_width
        elif self.target_line == 'top':
            d_px = y
            d_norm = d_px / frame_height
        else:  # bottom
            d_px = (frame_height - 1) - y
            d_norm = d_px / frame_height
        
        return (d_px, d_norm)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AdvancedBehaviorConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'target_line': self.target_line,
            'near_max': self.near_max,
            'middle_max': self.middle_max,
            'advance_threshold': self.advance_threshold,
            'retreat_threshold': self.retreat_threshold,
            'x_dir_thresh_norm': self.x_dir_thresh_norm,
            'y_dir_thresh_norm': self.y_dir_thresh_norm,
            'head_only_thresh_norm': self.head_only_thresh_norm,
            'body_move_thresh_norm': self.body_move_thresh_norm,
            'lookback_window': self.lookback_window,
            'arrow_length_norm': self.arrow_length_norm,
            'plot_colorscale': self.plot_colorscale,
            'plot_every_n_frames': self.plot_every_n_frames,
        }
