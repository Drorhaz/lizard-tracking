"""Event system for behavioral analysis."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import json


@dataclass
class BehaviorEvent:
    """Represents a detected behavioral event."""
    
    event_type: str          # "approach", "retreat", "stop", "close", "far", etc.
    timestamp: float         # Unix timestamp
    frame_number: int        # Frame number when event occurred
    confidence: float        # Confidence score (0.0 to 1.0)
    position: tuple[float, float]  # (x, y) position when event occurred
    metrics: Dict[str, Any]  # Additional metrics (speed, distance, etc.)
    metadata: Optional[Dict[str, Any]] = None  # Optional additional data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehaviorEvent':
        """Create event from dictionary."""
        return cls(**data)


class EventBus:
    """Event bus for handling and distributing behavioral events."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable[[BehaviorEvent], None]]] = {}
        self._global_listeners: List[Callable[[BehaviorEvent], None]] = []
        self._event_history: List[BehaviorEvent] = []
        self._max_history: int = 1000
    
    def subscribe(self, event_type: str, callback: Callable[[BehaviorEvent], None]):
        """Subscribe to specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable[[BehaviorEvent], None]):
        """Subscribe to all events."""
        self._global_listeners.append(callback)
    
    def emit(self, event: BehaviorEvent):
        """Emit an event to all subscribers."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify specific listeners
        if event.event_type in self._listeners:
            for callback in self._listeners[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")
        
        # Notify global listeners
        for callback in self._global_listeners:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in global event callback: {e}")
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[BehaviorEvent]:
        """Get recent events, optionally filtered by type."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
    
    def get_event_counts(self) -> Dict[str, int]:
        """Get count of each event type."""
        counts = {}
        for event in self._event_history:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
    
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()


# Default event handlers for common use cases
def print_event_handler(event: BehaviorEvent):
    """Simple event handler that prints events to console."""
    print(f"🎯 {event.event_type.upper()}: Frame {event.frame_number}, "
          f"Position: ({event.position[0]:.1f}, {event.position[1]:.1f}), "
          f"Confidence: {event.confidence:.2f}")


def csv_log_handler(csv_file_path: str):
    """Create event handler that logs events to CSV file."""
    import csv
    import os
    
    # Create CSV file with headers if it doesn't exist
    file_exists = os.path.exists(csv_file_path)
    
    def handler(event: BehaviorEvent):
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write headers
                writer.writerow(['timestamp', 'frame_number', 'event_type', 'confidence', 
                               'position_x', 'position_y', 'metrics'])
            
            # Write event data
            writer.writerow([
                event.timestamp, event.frame_number, event.event_type, event.confidence,
                event.position[0], event.position[1], json.dumps(event.metrics)
            ])
    
    return handler