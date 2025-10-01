"""
Behavioral Analysis Library

A standalone library for real-time behavioral analysis of animal movements.
Designed to work with any pose detection system or coordinate tracking data.

Features:
- Real-time behavior event detection (approach, retreat, stop, etc.)
- Live metrics calculation (speed, distance, trajectory analysis)
- Configurable thresholds and parameters
- Data export (CSV, trajectory plots, labeled frames)
- Event system for external integrations
- Trajectory reconstruction and visualization
- Space-time plotting capabilities

Usage:
    from behavioral_analysis import BehaviorDetector, BehaviorConfig
    
    config = BehaviorConfig(detect_approach=True, detect_retreat=True)
    detector = BehaviorDetector(config)
    
    # Process pose data
    events = detector.process_frame((x, y), frame_number)
    live_metrics = detector.metrics.to_dict()
"""

from .detector import BehaviorDetector
from .config import BehaviorConfig
from .events import BehaviorEvent, EventBus, EventType
from .metrics import LiveMetrics
from .export import BehaviorExporter
from .trajectory import TrajectoryAnalyzer, TrajectoryPoint

__version__ = "1.0.0"
__author__ = "Lizard Tracking Project"

__all__ = [
    "BehaviorDetector",
    "BehaviorConfig", 
    "BehaviorEvent",
    "EventBus",
    "EventType",
    "LiveMetrics",
    "BehaviorExporter",
    "TrajectoryAnalyzer",
    "TrajectoryPoint",
]