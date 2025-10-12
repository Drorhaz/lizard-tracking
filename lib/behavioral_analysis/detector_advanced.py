"""Advanced trajectory-based behavioral detector with arena mapping."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import numpy as np

# Try relative import first, fallback to absolute
try:
    from .config_advanced import AdvancedBehaviorConfig, ArenaPhase, BandLabel, ArenaSide, MotionType
except ImportError:
    from config_advanced import AdvancedBehaviorConfig, ArenaPhase, BandLabel, ArenaSide, MotionType


@dataclass
class DetectionFrame:
    """Single frame detection data."""
    frame_idx: int
    nose_x: Optional[float]
    nose_y: Optional[float]
    ear_left_x: Optional[float]
    ear_left_y: Optional[float]
    ear_right_x: Optional[float]
    ear_right_y: Optional[float]
    bbox_x1: Optional[float]
    bbox_y1: Optional[float]
    bbox_x2: Optional[float]
    bbox_y2: Optional[float]
    
    def has_valid_nose(self) -> bool:
        return self.nose_x is not None and self.nose_y is not None
    
    def has_valid_ears(self) -> bool:
        return (self.ear_left_x is not None and self.ear_left_y is not None and
                self.ear_right_x is not None and self.ear_right_y is not None)
    
    def has_valid_bbox(self) -> bool:
        return (self.bbox_x1 is not None and self.bbox_y1 is not None and
                self.bbox_x2 is not None and self.bbox_y2 is not None)
    
    def get_anchor_point(self) -> Optional[Tuple[float, float]]:
        """Get anchor point: nose preferred, fallback to ear midpoint."""
        if self.has_valid_nose():
            return (self.nose_x, self.nose_y)
        elif self.has_valid_ears():
            mid_x = (self.ear_left_x + self.ear_right_x) / 2.0
            mid_y = (self.ear_left_y + self.ear_right_y) / 2.0
            return (mid_x, mid_y)
        return None
    
    def get_body_center(self) -> Optional[Tuple[float, float]]:
        """Get body center from bbox or ear midpoint."""
        if self.has_valid_bbox():
            cx = (self.bbox_x1 + self.bbox_x2) / 2.0
            cy = (self.bbox_y1 + self.bbox_y2) / 2.0
            return (cx, cy)
        elif self.has_valid_ears():
            mid_x = (self.ear_left_x + self.ear_right_x) / 2.0
            mid_y = (self.ear_left_y + self.ear_right_y) / 2.0
            return (mid_x, mid_y)
        return None
    
    def compute_head_angle_deg(self) -> Optional[float]:
        """Compute head angle from ear midpoint to nose."""
        anchor = self.get_anchor_point()
        body_center = self.get_body_center()
        
        if anchor and body_center and self.has_valid_nose():
            dx = self.nose_x - body_center[0]
            dy = self.nose_y - body_center[1]
            if dx != 0 or dy != 0:
                angle_rad = np.arctan2(dy, dx)
                return np.degrees(angle_rad)
        return None


@dataclass
class BehavioralInstruction:
    """Single behavioral instruction/event."""
    frame_idx: int
    timestamp_ms: float
    video_seconds: float
    phase: ArenaPhase
    band: BandLabel
    arena_side: ArenaSide
    motion_type: Optional[MotionType]
    x_direction: Optional[str]  # 'leftward' | 'rightward' | None
    instruction: str  # Full formatted string
    
    def to_dict(self):
        return {
            'frame_idx': self.frame_idx,
            'timestamp_ms': self.timestamp_ms,
            'video_seconds': self.video_seconds,
            'phase': self.phase,
            'band': self.band,
            'arena_side': self.arena_side,
            'motion_type': self.motion_type,
            'x_direction': self.x_direction,
            'instruction': self.instruction,
        }


class AdvancedBehavioralDetector:
    """Advanced trajectory analyzer with arena mapping and instruction labels."""
    
    def __init__(self, config: AdvancedBehaviorConfig, frame_width: int, frame_height: int, fps: float = 30.0):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Compute thresholds in pixels
        self.x_dir_thresh_px = frame_width * config.x_dir_thresh_norm
        self.y_dir_thresh_px = frame_height * config.y_dir_thresh_norm
        
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        self.head_only_thresh_px = frame_diagonal * config.head_only_thresh_norm
        self.body_move_thresh_px = frame_diagonal * config.body_move_thresh_norm
        
        # Detection history (lookback window)
        self.detection_history: deque[DetectionFrame] = deque(maxlen=config.lookback_window + 1)
        
        # Previous valid state for delta calculations
        self.prev_valid_detection: Optional[DetectionFrame] = None
        self.prev_dist_norm: Optional[float] = None
        
        # Instructions log
        self.instructions: List[BehavioralInstruction] = []
        
        # Plot data storage
        self.plot_data: List[dict] = []
        
    def process_frame(self, frame_idx: int, 
                     nose: Optional[Tuple[float, float]],
                     ear_left: Optional[Tuple[float, float]],
                     ear_right: Optional[Tuple[float, float]],
                     bbox: Optional[Tuple[float, float, float, float]]) -> Optional[BehavioralInstruction]:
        """
        Process a single frame and generate behavioral instruction if applicable.
        
        Args:
            frame_idx: Frame number
            nose: (x, y) or None
            ear_left: (x, y) or None
            ear_right: (x, y) or None
            bbox: (x1, y1, x2, y2) or None
            
        Returns:
            BehavioralInstruction if generated, else None
        """
        # Create detection frame
        det = DetectionFrame(
            frame_idx=frame_idx,
            nose_x=nose[0] if nose else None,
            nose_y=nose[1] if nose else None,
            ear_left_x=ear_left[0] if ear_left else None,
            ear_left_y=ear_left[1] if ear_left else None,
            ear_right_x=ear_right[0] if ear_right else None,
            ear_right_y=ear_right[1] if ear_right else None,
            bbox_x1=bbox[0] if bbox else None,
            bbox_y1=bbox[1] if bbox else None,
            bbox_x2=bbox[2] if bbox else None,
            bbox_y2=bbox[3] if bbox else None,
        )
        
        self.detection_history.append(det)
        
        # Get current detection or use lookback
        current_det = self._get_valid_detection()
        if current_det is None:
            # No valid detection in lookback window
            return None
        
        # Get anchor point (nose or ear midpoint)
        anchor = current_det.get_anchor_point()
        if anchor is None:
            return None
        
        x, y = anchor
        
        # Compute distance to target line
        d_px, d_norm = self.config.compute_distance_to_target(x, y, self.frame_width, self.frame_height)
        band = self.config.get_band(d_norm)
        arena_side = self.config.get_arena_side(x, y, self.frame_width, self.frame_height)
        
        # Determine phase (approaching/retreating/resting)
        phase = self._determine_phase(d_norm)
        
        # Determine directional qualifier and motion type
        x_direction, motion_type = self._analyze_movement(current_det)
        
        # Format instruction
        instruction = self._format_instruction(phase, x_direction, band, arena_side, motion_type)
        
        # Store for plotting
        head_angle = current_det.compute_head_angle_deg()
        if frame_idx % self.config.plot_every_n_frames == 0:
            self.plot_data.append({
                'frame_idx': frame_idx,
                'x_norm': x / self.frame_width,
                'y_norm': y / self.frame_height,
                'head_angle_deg': head_angle,
                'dist_norm': d_norm,
            })
        
        # Create instruction object
        video_seconds = frame_idx / self.fps
        instr = BehavioralInstruction(
            frame_idx=frame_idx,
            timestamp_ms=video_seconds * 1000,
            video_seconds=video_seconds,
            phase=phase,
            band=band,
            arena_side=arena_side,
            motion_type=motion_type,
            x_direction=x_direction,
            instruction=instruction,
        )
        
        self.instructions.append(instr)
        
        # Update previous state
        self.prev_valid_detection = current_det
        self.prev_dist_norm = d_norm
        
        return instr
    
    def _get_valid_detection(self) -> Optional[DetectionFrame]:
        """Get most recent valid detection from lookback window."""
        for det in reversed(self.detection_history):
            if det.get_anchor_point() is not None:
                return det
        return None
    
    def _determine_phase(self, d_norm: float) -> ArenaPhase:
        """Determine approach/retreat/rest phase."""
        if self.prev_dist_norm is None:
            return 'resting'
        
        delta_d = d_norm - self.prev_dist_norm
        
        if delta_d < -self.config.advance_threshold:
            return 'approaching'
        elif delta_d > self.config.retreat_threshold:
            return 'retreating'
        else:
            return 'resting'
    
    def _analyze_movement(self, current_det: DetectionFrame) -> Tuple[Optional[str], Optional[MotionType]]:
        """
        Analyze movement to determine x-direction and motion type.
        
        Returns: (x_direction, motion_type)
            x_direction: 'leftward' | 'rightward' | None
            motion_type: 'head-only' | 'whole-body' | None
        """
        if self.prev_valid_detection is None:
            return (None, None)
        
        # Get anchor points
        curr_anchor = current_det.get_anchor_point()
        prev_anchor = self.prev_valid_detection.get_anchor_point()
        
        if curr_anchor is None or prev_anchor is None:
            return (None, None)
        
        # Compute deltas
        dx = curr_anchor[0] - prev_anchor[0]
        dy = curr_anchor[1] - prev_anchor[1]
        
        # Determine axis dominance
        x_only = abs(dx) >= self.x_dir_thresh_px and abs(dy) < self.y_dir_thresh_px
        y_only = abs(dy) >= self.y_dir_thresh_px and abs(dx) < self.x_dir_thresh_px
        both = abs(dx) >= self.x_dir_thresh_px and abs(dy) >= self.y_dir_thresh_px
        
        # X-direction (only for 'both' case with approaching/retreating)
        x_direction = None
        if both:
            x_direction = 'rightward' if dx > 0 else 'leftward'
        
        # Motion type (head-only vs whole-body)
        motion_type = self._determine_motion_type(current_det)
        
        return (x_direction, motion_type)
    
    def _determine_motion_type(self, current_det: DetectionFrame) -> Optional[MotionType]:
        """Determine if movement is head-only or whole-body."""
        if self.prev_valid_detection is None:
            return None
        
        # Get body centers
        curr_body = current_det.get_body_center()
        prev_body = self.prev_valid_detection.get_body_center()
        
        # Get nose positions
        curr_nose = (current_det.nose_x, current_det.nose_y) if current_det.has_valid_nose() else None
        prev_nose = (self.prev_valid_detection.nose_x, self.prev_valid_detection.nose_y) if self.prev_valid_detection.has_valid_nose() else None
        
        if curr_body is None or prev_body is None:
            return None
        
        # Compute displacements
        body_disp = np.sqrt((curr_body[0] - prev_body[0])**2 + (curr_body[1] - prev_body[1])**2)
        
        head_disp = 0.0
        if curr_nose and prev_nose:
            head_disp = np.sqrt((curr_nose[0] - prev_nose[0])**2 + (curr_nose[1] - prev_nose[1])**2)
        
        # Classify
        if body_disp < self.body_move_thresh_px and head_disp >= self.head_only_thresh_px:
            return 'head-only'
        elif body_disp >= self.body_move_thresh_px:
            return 'whole-body'
        else:
            return None
    
    def _format_instruction(self, phase: ArenaPhase, x_direction: Optional[str],
                           band: BandLabel, arena_side: ArenaSide,
                           motion_type: Optional[MotionType]) -> str:
        """
        Format instruction string according to spec grammar.
        
        Format: <phase>[, <x-direction>] — <band> @ <arena-side> [<motion-type>]
        """
        parts = [phase]
        
        # Add x-direction for approaching/retreating with both axes
        if x_direction and phase in ('approaching', 'retreating'):
            parts.append(f", {x_direction}")
        
        # Main part
        instruction = ''.join(parts) + f" — {band} @ {arena_side}"
        
        # Add motion type postfix
        if motion_type:
            instruction += f" [{motion_type}]"
        
        return instruction
    
    def get_instructions_csv_format(self) -> List[Tuple]:
        """Get instructions in CSV format: (ts_ms, frame_idx, instruction, meta_json)."""
        import json
        result = []
        for instr in self.instructions:
            meta = {
                'phase': instr.phase,
                'band': instr.band,
                'arena_side': instr.arena_side,
                'motion_type': instr.motion_type,
                'x_direction': instr.x_direction,
            }
            result.append((
                instr.timestamp_ms,
                instr.frame_idx,
                instr.instruction,
                json.dumps(meta)
            ))
        return result
    
    def get_plot_data(self) -> List[dict]:
        """Get data for plotting."""
        return self.plot_data
