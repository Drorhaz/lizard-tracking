#!/usr/bin/env python3
"""
Enhanced pose-head integration with lib/lizard_tracking and lib/behavioral_analysis

This integration provides:
1. Professional pose detection using lib/lizard_tracking  
2. Advanced behavioral analysis using lib/behavioral_analysis
3. Clean web interface integration
4. Consolidated trajectory analysis
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

# Add lib directories to path for integration
project_root = Path(__file__).resolve().parents[2]
lib_dir = project_root / "lib"
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

# Import the excellent existing libraries
try:
    from lizard_tracking.config import PoseInferenceConfig
    from lizard_tracking.ui.stream import LivePoseProcessor, ActivityDetector, ActivityEvent
    from lizard_tracking.core import HeadPose, PoseObservation
    from lizard_tracking.models.pogona_pose import PogonaHeadPoseModel
    LIZARD_TRACKING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: lizard_tracking not available: {e}")
    LIZARD_TRACKING_AVAILABLE = False

try:
    from behavioral_analysis import (
        BehaviorDetector, BehaviorConfig, BehaviorEvent, 
        EventType, LiveMetrics, BehaviorExporter, TrajectoryAnalyzer
    )
    BEHAVIORAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: behavioral_analysis not available: {e}")
    BEHAVIORAL_ANALYSIS_AVAILABLE = False


class EnhancedPoseWebInterface:
    """
    Enhanced pose processing for web interface integrating:
    - lib/lizard_tracking: Professional pose detection + beautiful overlays  
    - lib/behavioral_analysis: Advanced behavioral event detection
    - Consolidated trajectory analysis and export
    """
    
    def __init__(self, 
                 model_weights: str,
                 imgsz: int = 640,
                 conf: float = 0.25,
                 device: str = "0"):
        
        self.model_weights = model_weights
        self.pose_processor = None
        self.behavior_detector = None
        self.exporter = None
        self.frame_count = 0
        
        # Initialize if libraries available
        if LIZARD_TRACKING_AVAILABLE:
            self._setup_pose_processing(model_weights, imgsz, conf, device)
        
        if BEHAVIORAL_ANALYSIS_AVAILABLE:
            self._setup_behavioral_analysis()
    
    def _setup_pose_processing(self, weights: str, imgsz: int, conf: float, device: str):
        """Setup pose processing using lib/lizard_tracking."""
        try:
            pose_config = PoseInferenceConfig(
                weights=weights,
                imgsz=imgsz, 
                conf=conf,
                device=device
            )
            
            # Use the excellent LivePoseProcessor with ActivityDetector
            self.pose_processor = LivePoseProcessor(
                pose_config,
                activity_detector=ActivityDetector(
                    forward_axis="y",
                    advance_threshold=8.0,
                    retreat_threshold=-8.0,
                    stop_delta=2.0,
                    stop_patience=6
                )
            )
            print("✅ Pose processing initialized with lib/lizard_tracking")
            
        except Exception as e:
            print(f"❌ Failed to initialize pose processing: {e}")
            self.pose_processor = None
    
    def _setup_behavioral_analysis(self, reference_point: Optional[tuple] = None):
        """Setup advanced behavioral analysis."""
        try:
            behavior_config = BehaviorConfig(
                detect_approach=True,
                detect_retreat=True,
                detect_stop=True,
                approach_threshold=100,     # pixels
                retreat_threshold=300,      # pixels 
                stop_threshold=5,           # pixels/frame
                reference_point=reference_point or (320, 240),  # center of frame
                hysteresis_px=10,
                min_stationary_frames=10,
                min_moving_frames=5
            )
            
            self.behavior_detector = BehaviorDetector(behavior_config)
            self.exporter = BehaviorExporter(output_dir="output/behavioral_data")
            print("✅ Behavioral analysis initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize behavioral analysis: {e}")
            self.behavior_detector = None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process frame with both pose detection and behavioral analysis.
        
        Returns enhanced result with:
        - Beautiful frame overlay (from lizard_tracking)
        - Pose detection data
        - Basic activity events (from lizard_tracking) 
        - Advanced behavioral events (from behavioral_analysis)
        - Live metrics and statistics
        """
        result = {
            'frame_number': self.frame_count,
            'original_frame': frame.copy(),
            'frame_with_overlay': frame.copy(),  # fallback
            'pose_detected': False,
            'error': None
        }
        
        try:
            # Process with lib/lizard_tracking (gets beautiful overlays automatically!)
            if self.pose_processor:
                pose_result = self.pose_processor.process_frame(frame)
                
                result.update({
                    'frame_with_overlay': pose_result.frame,  # Already has keypoints drawn!
                    'pose': pose_result.head,
                    'pose_detected': pose_result.head is not None,
                    'lizard_activity': pose_result.event.value if pose_result.event else None,
                })
                
                # Add advanced behavioral analysis if pose detected
                if pose_result.head and self.behavior_detector:
                    behavior_events = self.behavior_detector.process_frame(
                        pose_result.head, self.frame_count
                    )
                    
                    result.update({
                        'behavior_events': [e.to_dict() for e in behavior_events],
                        'live_metrics': self.behavior_detector.metrics.to_dict(),
                        'behavioral_state': self.behavior_detector.get_current_state(),
                    })
            
            self.frame_count += 1
            
        except Exception as e:
            result['error'] = str(e)
            print(f"Error processing frame {self.frame_count}: {e}")
        
        return result
    
    def configure_behavioral_analysis(self, config_dict: Dict[str, Any]):
        """Update behavioral analysis configuration from web interface."""
        if not self.behavior_detector:
            return False
        
        try:
            # Update configuration
            config = self.behavior_detector.config
            
            # Toggle event detection
            if 'detect_approach' in config_dict:
                config.detect_approach = bool(config_dict['detect_approach'])
                if config.detect_approach:
                    self.behavior_detector.enable_event_type(EventType.APPROACH_START)
                    self.behavior_detector.enable_event_type(EventType.APPROACH_END)
                else:
                    self.behavior_detector.disable_event_type(EventType.APPROACH_START)
                    self.behavior_detector.disable_event_type(EventType.APPROACH_END)
            
            if 'detect_retreat' in config_dict:
                config.detect_retreat = bool(config_dict['detect_retreat'])
                if config.detect_retreat:
                    self.behavior_detector.enable_event_type(EventType.RETREAT_START)
                    self.behavior_detector.enable_event_type(EventType.RETREAT_END)
                else:
                    self.behavior_detector.disable_event_type(EventType.RETREAT_START)
                    self.behavior_detector.disable_event_type(EventType.RETREAT_END)
            
            if 'detect_stop' in config_dict:
                config.detect_stop = bool(config_dict['detect_stop'])
                if config.detect_stop:
                    self.behavior_detector.enable_event_type(EventType.STOP_START)
                    self.behavior_detector.enable_event_type(EventType.STOP_END)
                else:
                    self.behavior_detector.disable_event_type(EventType.STOP_START)
                    self.behavior_detector.disable_event_type(EventType.STOP_END)
            
            # Update thresholds
            if 'approach_threshold' in config_dict:
                config.approach_threshold = float(config_dict['approach_threshold'])
            if 'retreat_threshold' in config_dict:
                config.retreat_threshold = float(config_dict['retreat_threshold'])
            if 'stop_threshold' in config_dict:
                config.stop_threshold = float(config_dict['stop_threshold'])
            
            # Update reference point
            if 'reference_x' in config_dict and 'reference_y' in config_dict:
                config.reference_point = (
                    float(config_dict['reference_x']),
                    float(config_dict['reference_y'])
                )
                self.behavior_detector.set_reference_point(config.reference_point)
            
            return True
            
        except Exception as e:
            print(f"Error configuring behavioral analysis: {e}")
            return False
    
    def export_session_data(self) -> Optional[Path]:
        """Export complete session data."""
        if not self.behavior_detector or not self.exporter:
            return None
        
        try:
            events = self.behavior_detector.event_bus.get_all_events()
            metrics = self.behavior_detector.metrics
            config_dict = self.behavior_detector.config.__dict__
            
            return self.exporter.export_session_summary(events, metrics, config_dict)
            
        except Exception as e:
            print(f"Error exporting session data: {e}")
            return None
    
    def get_trajectory_analysis(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive trajectory analysis."""
        if not self.behavior_detector:
            return None
        
        try:
            # Create trajectory analyzer from position history
            positions = list(self.behavior_detector.metrics.position_history)
            if len(positions) < 2:
                return None
            
            analyzer = TrajectoryAnalyzer.from_coordinates(positions)
            return analyzer.get_summary()
            
        except Exception as e:
            print(f"Error in trajectory analysis: {e}")
            return None
    
    def reset_session(self):
        """Reset all session data."""
        if self.behavior_detector:
            self.behavior_detector.reset()
        if self.pose_processor:
            self.pose_processor.activity_detector.reset()
        self.frame_count = 0


def create_web_integration_example():
    """Example showing how to integrate with Flask web interface."""
    
    return '''
    # Example Flask route integration
    
    @app.route('/api/configure_behavior', methods=['POST'])
    def configure_behavior():
        """Configure behavioral analysis from web form."""
        config = request.get_json()
        success = enhanced_processor.configure_behavioral_analysis(config)
        return jsonify({'success': success})
    
    @app.route('/api/current_metrics')
    def get_current_metrics():
        """Get live behavioral metrics."""
        if enhanced_processor.behavior_detector:
            metrics = enhanced_processor.behavior_detector.metrics.to_dict()
            state = enhanced_processor.behavior_detector.get_current_state()
            return jsonify({'metrics': metrics, 'state': state})
        return jsonify({'error': 'Behavioral analysis not available'})
    
    @app.route('/api/export_session')
    def export_session():
        """Export session data."""
        filepath = enhanced_processor.export_session_data()
        if filepath:
            return jsonify({'export_path': str(filepath)})
        return jsonify({'error': 'Export failed'})
    
    @app.route('/api/trajectory_analysis')
    def get_trajectory_analysis():
        """Get trajectory analysis."""
        analysis = enhanced_processor.get_trajectory_analysis()
        if analysis:
            return jsonify(analysis)
        return jsonify({'error': 'Not enough trajectory data'})
    '''


if __name__ == "__main__":
    print("🔧 Enhanced Pose-Head Integration")
    print("=" * 50)
    
    print("This integration consolidates:")
    print("✅ lib/lizard_tracking - Professional pose detection + overlays")
    print("✅ lib/behavioral_analysis - Advanced behavioral event detection")
    print("✅ Trajectory analysis and data export")
    print("✅ Web interface ready architecture")
    
    print(f"\n📚 Libraries available:")
    print(f"   lizard_tracking: {LIZARD_TRACKING_AVAILABLE}")
    print(f"   behavioral_analysis: {BEHAVIORAL_ANALYSIS_AVAILABLE}")
    
    if LIZARD_TRACKING_AVAILABLE and BEHAVIORAL_ANALYSIS_AVAILABLE:
        print("\n🚀 All systems ready for enhanced pose-head integration!")
    else:
        print("\n⚠️  Some libraries not available - check imports")