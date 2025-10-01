"""Data export utilities for behavioral analysis."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

from .events import BehaviorEvent, EventType
from .metrics import LiveMetrics


class BehaviorExporter:
    """Export behavioral analysis data in various formats."""
    
    def __init__(self, output_dir: Union[str, Path] = "output/behavioral_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_events_csv(self, events: List[BehaviorEvent], filename: Optional[str] = None) -> Path:
        """Export events to CSV format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"behavioral_events_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                'timestamp',
                'event_type', 
                'frame_number',
                'position_x',
                'position_y',
                'metadata_json'
            ])
            
            # Data rows
            for event in events:
                writer.writerow([
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.frame_number,
                    event.position[0] if event.position else None,
                    event.position[1] if event.position else None,
                    json.dumps(event.metadata) if event.metadata else '{}'
                ])
        
        return filepath
    
    def export_events_json(self, events: List[BehaviorEvent], filename: Optional[str] = None) -> Path:
        """Export events to JSON format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"behavioral_events_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        events_data = []
        for event in events:
            event_dict = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'frame_number': event.frame_number,
                'position': event.position,
                'metadata': event.metadata or {}
            }
            events_data.append(event_dict)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(events_data, jsonfile, indent=2)
        
        return filepath
    
    def export_trajectory_csv(self, metrics: LiveMetrics, filename: Optional[str] = None) -> Path:
        """Export trajectory data to CSV format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['frame', 'x', 'y'])
            
            # Data rows
            for frame, position in enumerate(metrics.position_history):
                writer.writerow([frame, position[0], position[1]])
        
        return filepath
    
    def export_metrics_json(self, metrics: LiveMetrics, filename: Optional[str] = None) -> Path:
        """Export metrics summary to JSON format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        metrics_data = metrics.to_dict()
        
        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return obj
        
        # Clean up position history (convert deque to list)
        if hasattr(metrics_data, 'position_history'):
            metrics_data['position_history'] = list(metrics.position_history)
        if hasattr(metrics_data, 'speed_history'):
            metrics_data['speed_history'] = list(metrics.speed_history)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(metrics_data, jsonfile, indent=2, default=convert_for_json)
        
        return filepath
    
    def export_session_summary(self, 
                             events: List[BehaviorEvent], 
                             metrics: LiveMetrics,
                             config_dict: Optional[Dict] = None,
                             filename: Optional[str] = None) -> Path:
        """Export comprehensive session summary."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_summary_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Event statistics
        event_stats = {}
        for event_type in EventType:
            count = sum(1 for event in events if event.event_type == event_type)
            event_stats[event_type.value] = count
        
        # Time-based analysis
        if events:
            session_start = min(event.timestamp for event in events)
            session_end = max(event.timestamp for event in events)
            session_duration = (session_end - session_start).total_seconds()
        else:
            session_start = None
            session_end = None
            session_duration = 0
        
        summary = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'session_start': session_start.isoformat() if session_start else None,
                'session_end': session_end.isoformat() if session_end else None,
                'session_duration_seconds': session_duration,
                'total_events': len(events),
                'frames_processed': metrics.frames_processed
            },
            'configuration': config_dict or {},
            'event_statistics': event_stats,
            'trajectory_metrics': metrics.to_dict(),
            'events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'frame_number': event.frame_number,
                    'position': event.position,
                    'metadata': event.metadata or {}
                }
                for event in events
            ]
        }
        
        with open(filepath, 'w') as jsonfile:
            json.dump(summary, jsonfile, indent=2, default=str)
        
        return filepath
    
    def export_for_analysis_software(self, 
                                   events: List[BehaviorEvent], 
                                   metrics: LiveMetrics,
                                   format_type: str = "ethovision") -> Path:
        """Export data in format compatible with analysis software."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "ethovision":
            return self._export_ethovision_format(events, metrics, timestamp)
        elif format_type.lower() == "boris":
            return self._export_boris_format(events, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_ethovision_format(self, events: List[BehaviorEvent], metrics: LiveMetrics, timestamp: str) -> Path:
        """Export in EthoVision-compatible format."""
        filename = f"ethovision_export_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # EthoVision header format
            writer.writerow(['Trial time', 'X center', 'Y center', 'Behavior'])
            
            # Create timeline with positions and behaviors
            position_data = list(metrics.position_history)
            event_dict = {event.frame_number: event.event_type.value for event in events}
            
            for frame, position in enumerate(position_data):
                behavior = event_dict.get(frame, '')
                time_seconds = frame / 30.0  # Assume 30 fps, could be configurable
                writer.writerow([time_seconds, position[0], position[1], behavior])
        
        return filepath
    
    def _export_boris_format(self, events: List[BehaviorEvent], timestamp: str) -> Path:
        """Export in BORIS-compatible format."""
        filename = f"boris_export_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # BORIS header format
            writer.writerow(['Time', 'Subject', 'Behavior', 'Behavioral category', 'Comment'])
            
            for event in events:
                time_seconds = event.frame_number / 30.0  # Assume 30 fps
                writer.writerow([
                    time_seconds,
                    'Animal1',
                    event.event_type.value,
                    'Movement',
                    json.dumps(event.metadata) if event.metadata else ''
                ])
        
        return filepath
    
    def export_raw_data(self, data: Any, filename: str, format_type: str = "pickle") -> Path:
        """Export raw data objects."""
        filepath = self.output_dir / filename
        
        if format_type == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format_type == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return filepath
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of exported files."""
        files = list(self.output_dir.glob("*"))
        
        return {
            'output_directory': str(self.output_dir),
            'total_files': len(files),
            'files': [
                {
                    'name': f.name,
                    'size_bytes': f.stat().st_size,
                    'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                }
                for f in files
            ]
        }