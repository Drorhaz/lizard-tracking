#!/usr/bin/env python3
"""
Enhanced Pose Processing Demo

Demonstrates integration between lib/lizard_tracking and lib/behavioral_analysis
for comprehensive real-time behavioral monitoring with beautiful frame overlays.
"""
from pathlib import Path
import sys

# Add lib to path
ROOT_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT_DIR / "lib"
sys.path.insert(0, str(LIB_DIR))

from lizard_tracking.config import PoseInferenceConfig
from lizard_tracking.ui.stream import LivePoseProcessor, ActivityDetector
from behavioral_analysis import BehaviorDetector, BehaviorConfig
import cv2
import json


class EnhancedPoseProcessor:
    """
    Integration of lizard_tracking (pose detection + drawing) 
    with behavioral_analysis (advanced behavioral events).
    
    This gives you:
    - Professional frame overlays with keypoints (from lizard_tracking)
    - Advanced behavioral event detection (from behavioral_analysis)
    - Real-time metrics and data export
    """
    
    def __init__(self, pose_config: PoseInferenceConfig, behavior_config: BehaviorConfig):
        # Use the excellent LivePoseProcessor for pose + drawing
        self.pose_processor = LivePoseProcessor(
            pose_config, 
            activity_detector=ActivityDetector()
        )
        
        # Add our advanced behavioral analysis
        self.behavior_detector = BehaviorDetector(behavior_config)
        
    def process_frame(self, frame, frame_number=0):
        """Process frame with both pose detection and behavioral analysis."""
        
        # Get pose detection with beautiful overlay drawing
        pose_result = self.pose_processor.process_frame(frame)
        
        enhanced_result = {
            'frame_with_overlay': pose_result.frame,  # Already has keypoints drawn!
            'pose': pose_result.head,
            'lizard_activity': pose_result.event,  # Basic activity from lizard_tracking
        }
        
        # Add advanced behavioral analysis if pose detected
        if pose_result.head:
            # Process HeadPose object directly (enhanced detector supports this!)
            behavior_events = self.behavior_detector.process_frame(
                pose_result.head, frame_number
            )
            
            enhanced_result.update({
                'behavior_events': [e.to_dict() for e in behavior_events],
                'live_metrics': self.behavior_detector.metrics.to_dict(),
                'behavioral_state': self.behavior_detector.get_current_state()
            })
        
        return enhanced_result


def demo_configuration():
    """Example configuration for enhanced processing."""
    
    # Pose detection config (using existing excellent implementation)
    pose_config = PoseInferenceConfig(
        weights="output/models/head_pose/best.pt",  # Your trained model
        imgsz=640,
        conf=0.25,
        device=0
    )
    
    # Advanced behavioral analysis config
    behavior_config = BehaviorConfig(
        detect_approach=True,
        detect_retreat=True,
        detect_stop=True,
        approach_threshold=100,     # pixels from reference point
        retreat_threshold=300,      # pixels from reference point  
        stop_threshold=5,           # pixels/frame speed threshold
        reference_point=(320, 240), # center of frame
        hysteresis_px=10,           # prevent false triggering
        min_stationary_frames=10,   # frames to confirm stop
        min_moving_frames=5         # frames to confirm movement
    )
    
    return pose_config, behavior_config


def web_interface_integration_example():
    """
    Example of how this would integrate with a web interface.
    
    This gives you everything you wanted:
    - Beautiful frame overlays with keypoints
    - Real-time behavioral event detection  
    - Configurable parameters via web form
    - Live metrics display
    - Data export capabilities
    """
    
    return '''
    <!-- Web Interface Controls -->
    <div class="behavior-controls">
        <h3>Behavioral Analysis</h3>
        
        <!-- Event Detection Toggles -->
        <label><input type="checkbox" id="detect_approach" checked> Detect Approach</label>
        <label><input type="checkbox" id="detect_retreat" checked> Detect Retreat</label>  
        <label><input type="checkbox" id="detect_stop" checked> Detect Stop/Movement</label>
        
        <!-- Threshold Controls -->
        <div class="thresholds">
            <label>Approach Threshold: <input type="range" id="approach_threshold" min="50" max="200" value="100"> px</label>
            <label>Retreat Threshold: <input type="range" id="retreat_threshold" min="200" max="500" value="300"> px</label>
            <label>Stop Threshold: <input type="range" id="stop_threshold" min="1" max="20" value="5"> px/frame</label>
        </div>
        
        <!-- Reference Point -->
        <div class="reference-point">
            <label>Reference Point: <input type="text" id="ref_point" value="320,240"> (x,y)</label>
        </div>
    </div>
    
    <!-- Live Metrics Display -->
    <div class="live-metrics">
        <h3>Live Metrics</h3>
        <div id="current-speed">Speed: -- px/frame</div>
        <div id="distance-from-ref">Distance from Reference: -- px</div>
        <div id="total-distance">Total Distance: -- px</div>
        <div id="events-detected">Events Detected: --</div>
    </div>
    
    <!-- Recent Events -->
    <div class="recent-events">
        <h3>Recent Events</h3>
        <ul id="event-list"></ul>
    </div>
    '''


if __name__ == "__main__":
    print("Enhanced Pose Processing Integration Demo")
    print("=" * 50)
    
    print("✅ This integration provides:")
    print("   - Professional pose detection with frame overlays (lizard_tracking)")
    print("   - Advanced behavioral event detection (behavioral_analysis)")  
    print("   - Real-time metrics and data export")
    print("   - Web interface ready architecture")
    
    print("\n🎯 Key Benefits:")
    print("   - Uses existing excellent LivePoseProcessor")
    print("   - Beautiful keypoint visualization automatically")
    print("   - No code duplication")
    print("   - Modular, configurable design")
    
    print("\n🧹 Cleanup Recommendations:")
    print("   - Remove tools/pose_head_pipeline.py (524 lines of duplication)")
    print("   - Keep tools/run_pose_stream.py (modern, uses lib properly)")
    print("   - Integrate behavioral_analysis with web interface")
    print("   - Consolidate trajectory tools")
    
    pose_config, behavior_config = demo_configuration()
    print(f"\n📋 Example Configuration:")
    print(f"   Pose: {pose_config.weights}, conf={pose_config.conf}")
    print(f"   Behavior: approach={behavior_config.approach_threshold}px, retreat={behavior_config.retreat_threshold}px")