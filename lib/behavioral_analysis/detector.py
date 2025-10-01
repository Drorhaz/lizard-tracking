"""Core behavioral detection logic."""
from __future__ import annotations
from typing import Tuple, Optional, List, Set, Union
import numpy as np
from dataclasses import dataclass
from .config import BehaviorConfig
from .events import BehaviorEvent, EventType, EventBus
from .metrics import LiveMetrics

# Optional integration with lizard_tracking library
try:
    import sys
    from pathlib import Path
    root_dir = Path(__file__).resolve().parents[2] / "lib"
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from lizard_tracking.core import HeadPose
    from lizard_tracking.ui.stream import ActivityEvent as LizardActivityEvent
    LIZARD_TRACKING_AVAILABLE = True
except ImportError:
    HeadPose = None
    LizardActivityEvent = None
    LIZARD_TRACKING_AVAILABLE = False


@dataclass
class DetectionState:
    """Internal state for behavior detection."""
    last_position: Optional[Tuple[float, float]] = None
    last_speed: float = 0.0
    stationary_count: int = 0
    moving_count: int = 0
    in_approach: bool = False
    in_retreat: bool = False
    approach_start_distance: Optional[float] = None
    retreat_start_distance: Optional[float] = None


class BehaviorDetector:
    """Main behavioral analysis detector."""
    
    def __init__(self, config: BehaviorConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.metrics = LiveMetrics()
        self.state = DetectionState()
        self._enabled_events: Set[EventType] = set()
        
        # Initialize enabled events based on config
        if config.detect_approach:
            self._enabled_events.add(EventType.APPROACH_START)
            self._enabled_events.add(EventType.APPROACH_END)
        if config.detect_retreat:
            self._enabled_events.add(EventType.RETREAT_START)
            self._enabled_events.add(EventType.RETREAT_END)
        if config.detect_stop:
            self._enabled_events.add(EventType.STOP_START)
            self._enabled_events.add(EventType.STOP_END)
    
    def process_frame(self, position, frame_number: int = 0) -> List[BehaviorEvent]:
        """Process a single frame and detect behavioral events.
        
        Args:
            position: Either (x, y) coordinates or HeadPose object from lizard_tracking
            frame_number: Current frame number
            
        Returns:
            List of detected behavioral events
        """
        # Handle HeadPose object from lizard_tracking
        if LIZARD_TRACKING_AVAILABLE and isinstance(position, HeadPose):
            pos_coords = position.center()
            # Store additional pose information in metrics
            if hasattr(self.metrics, 'current_pose_data'):
                self.metrics.current_pose_data = {
                    'confidence': position.conf,
                    'yaw_rad': position.yaw_rad,
                    'bbox': position.bbox_xyxy,
                    'nose': position.nose,
                    'ear_left': position.ear_left,
                    'ear_right': position.ear_right
                }
        else:
            pos_coords = position
        
        events = []
        
        # Update metrics
        self.metrics.update_position(pos_coords, self.config.reference_point)
        
        # Calculate current distance and speed
        current_distance = self.metrics.distance_from_reference
        current_speed = self.metrics.current_speed_px_per_frame
        
        # Detect approach/retreat events
        if self.config.reference_point is not None:
            approach_events = self._detect_approach_retreat(current_distance, frame_number)
            events.extend(approach_events)
        
        # Detect stop/movement events
        stop_events = self._detect_stop_movement(current_speed, frame_number)
        events.extend(stop_events)
        
        # Update state
        self.state.last_position = pos_coords
        self.state.last_speed = current_speed
        
        # Publish events to event bus
        for event in events:
            self.event_bus.publish(event)
            self.metrics.events_detected += 1
        
        return events
    
    def _detect_approach_retreat(self, current_distance: float, frame_number: int) -> List[BehaviorEvent]:
        """Detect approach and retreat events based on distance changes."""
        events = []
        
        # Approach detection
        if EventType.APPROACH_START in self._enabled_events:
            if not self.state.in_approach and current_distance <= self.config.approach_threshold:
                # Start approach
                self.state.in_approach = True
                self.state.approach_start_distance = current_distance
                
                event = BehaviorEvent(
                    event_type=EventType.APPROACH_START,
                    frame_number=frame_number,
                    position=self.metrics.current_position,
                    metadata={
                        'distance': current_distance,
                        'threshold': self.config.approach_threshold
                    }
                )
                events.append(event)
            
            elif self.state.in_approach and current_distance > self.config.approach_threshold + self.config.hysteresis_px:
                # End approach (with hysteresis)
                self.state.in_approach = False
                
                event = BehaviorEvent(
                    event_type=EventType.APPROACH_END,
                    frame_number=frame_number,
                    position=self.metrics.current_position,
                    metadata={
                        'distance': current_distance,
                        'threshold': self.config.approach_threshold,
                        'start_distance': self.state.approach_start_distance
                    }
                )
                events.append(event)
                self.state.approach_start_distance = None
        
        # Retreat detection
        if EventType.RETREAT_START in self._enabled_events:
            if not self.state.in_retreat and current_distance >= self.config.retreat_threshold:
                # Start retreat
                self.state.in_retreat = True
                self.state.retreat_start_distance = current_distance
                
                event = BehaviorEvent(
                    event_type=EventType.RETREAT_START,
                    frame_number=frame_number,
                    position=self.metrics.current_position,
                    metadata={
                        'distance': current_distance,
                        'threshold': self.config.retreat_threshold
                    }
                )
                events.append(event)
            
            elif self.state.in_retreat and current_distance < self.config.retreat_threshold - self.config.hysteresis_px:
                # End retreat (with hysteresis)
                self.state.in_retreat = False
                
                event = BehaviorEvent(
                    event_type=EventType.RETREAT_END,
                    frame_number=frame_number,
                    position=self.metrics.current_position,
                    metadata={
                        'distance': current_distance,
                        'threshold': self.config.retreat_threshold,
                        'start_distance': self.state.retreat_start_distance
                    }
                )
                events.append(event)
                self.state.retreat_start_distance = None
        
        return events
    
    def _detect_stop_movement(self, current_speed: float, frame_number: int) -> List[BehaviorEvent]:
        """Detect stop and movement events based on speed changes."""
        events = []
        
        if EventType.STOP_START not in self._enabled_events:
            return events
        
        # Update counters
        if current_speed < self.config.stop_threshold:
            self.state.stationary_count += 1
            self.state.moving_count = 0
        else:
            self.state.moving_count += 1
            self.state.stationary_count = 0
        
        # Detect stop start
        if (self.state.stationary_count >= self.config.min_stationary_frames and 
            self.state.stationary_count == self.config.min_stationary_frames):
            
            event = BehaviorEvent(
                event_type=EventType.STOP_START,
                frame_number=frame_number,
                position=self.metrics.current_position,
                metadata={
                    'speed': current_speed,
                    'threshold': self.config.stop_threshold,
                    'stationary_frames': self.state.stationary_count
                }
            )
            events.append(event)
        
        # Detect stop end (movement start)
        if (self.state.moving_count >= self.config.min_moving_frames and 
            self.state.moving_count == self.config.min_moving_frames):
            
            event = BehaviorEvent(
                event_type=EventType.STOP_END,
                frame_number=frame_number,
                position=self.metrics.current_position,
                metadata={
                    'speed': current_speed,
                    'threshold': self.config.stop_threshold,
                    'moving_frames': self.state.moving_count
                }
            )
            events.append(event)
        
        return events
    
    def set_reference_point(self, point: Tuple[float, float]):
        """Update the reference point for approach/retreat detection."""
        self.config.reference_point = point
    
    def enable_event_type(self, event_type: EventType):
        """Enable detection of a specific event type."""
        self._enabled_events.add(event_type)
    
    def disable_event_type(self, event_type: EventType):
        """Disable detection of a specific event type."""
        self._enabled_events.discard(event_type)
    
    def is_event_enabled(self, event_type: EventType) -> bool:
        """Check if an event type is enabled."""
        return event_type in self._enabled_events
    
    def get_current_state(self) -> dict:
        """Get current behavioral state summary."""
        return {
            'position': self.metrics.current_position,
            'speed': self.metrics.current_speed_px_per_frame,
            'distance_from_reference': self.metrics.distance_from_reference,
            'in_approach': self.state.in_approach,
            'in_retreat': self.state.in_retreat,
            'is_stationary': self.metrics.is_stationary(self.config.stop_threshold),
            'stationary_frames': self.state.stationary_count,
            'moving_frames': self.state.moving_count,
            'enabled_events': list(self._enabled_events),
        }
    
    def reset(self):
        """Reset detector state and metrics."""
        self.state = DetectionState()
        self.metrics.reset()
        self.event_bus.clear_history()
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics since last reset."""
        event_counts = {}
        for event_type in EventType:
            event_counts[event_type.value] = len(self.event_bus.get_events_by_type(event_type))
        
        return {
            'frames_processed': self.metrics.frames_processed,
            'total_events': self.metrics.events_detected,
            'event_counts': event_counts,
            'total_distance': self.metrics.total_distance_traveled,
            'average_speed': self.metrics.get_average_speed(),
            'trajectory_length': self.metrics.get_trajectory_length(),
            'direction_stability': self.metrics.get_direction_stability(),
            'elapsed_time': self.metrics.last_update_time - self.metrics.start_time,
        }