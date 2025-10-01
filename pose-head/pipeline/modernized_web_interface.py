"""
Modernized pose-head web interface using consolidated lib/lizard_tracking 
and lib/behavioral_analysis libraries.

This replaces the duplicated YOLO implementation with professional libraries.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

# Add lib to path  
project_root = Path(__file__).resolve().parents[2]
lib_dir = project_root / "lib"
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

# Import enhanced integration
from enhanced_integration import EnhancedPoseWebInterface


class ModernizedWebInterface:
    """
    Clean, modern web interface using consolidated libraries.
    
    Benefits over legacy implementation:
    - Uses lib/lizard_tracking for professional pose detection + overlays
    - Adds lib/behavioral_analysis for advanced behavioral events  
    - Eliminates code duplication
    - Provides better web controls for behavioral analysis
    - Real-time metrics and data export
    """
    
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        
        self.enhanced_processor = None
        self.current_frame = None
        self._frame_lock = threading.Lock()
        self._processing_active = False
        self._processing_thread = None
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes with enhanced functionality."""
        
        @self.app.route('/')
        def index():
            """Main interface with behavioral analysis controls."""
            videos = self._get_available_videos()
            return render_template('index.html', videos=videos)
        
        @self.app.route('/video')
        def video_stream():
            """Video stream with pose overlays."""
            return Response(self._generate_video_stream(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/start_enhanced', methods=['POST'])
        def start_enhanced_processing():
            """Start processing with enhanced libraries."""
            config = request.get_json()
            
            # Initialize enhanced processor
            model_weights = config.get('model_weights', 'output/models/head_pose/best.pt')
            
            self.enhanced_processor = EnhancedPoseWebInterface(
                model_weights=model_weights,
                imgsz=config.get('imgsz', 640),
                conf=config.get('conf', 0.25),
                device=config.get('device', '0')
            )
            
            # Configure behavioral analysis if requested
            if config.get('enable_behavioral_analysis', False):
                behavior_config = {
                    'detect_approach': config.get('detect_approach', True),
                    'detect_retreat': config.get('detect_retreat', True),
                    'detect_stop': config.get('detect_stop', True),
                    'approach_threshold': config.get('approach_threshold', 100),
                    'retreat_threshold': config.get('retreat_threshold', 300),
                    'stop_threshold': config.get('stop_threshold', 5),
                    'reference_x': config.get('reference_x', 320),
                    'reference_y': config.get('reference_y', 240),
                }
                self.enhanced_processor.configure_behavioral_analysis(behavior_config)
            
            # Start video processing
            success = self._start_video_processing(config.get('video_path'))
            
            return jsonify({
                'success': success,
                'lizard_tracking_available': self.enhanced_processor.pose_processor is not None,
                'behavioral_analysis_available': self.enhanced_processor.behavior_detector is not None
            })
        
        @self.app.route('/api/stop_processing', methods=['POST'])
        def stop_processing():
            """Stop current processing."""
            self._processing_active = False
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=2.0)
            return jsonify({'success': True})
        
        @self.app.route('/api/behavioral_config', methods=['POST'])
        def update_behavioral_config():
            """Update behavioral analysis configuration."""
            if not self.enhanced_processor:
                return jsonify({'success': False, 'error': 'Processor not initialized'})
            
            config = request.get_json()
            success = self.enhanced_processor.configure_behavioral_analysis(config)
            return jsonify({'success': success})
        
        @self.app.route('/api/live_metrics')
        def get_live_metrics():
            """Get current behavioral metrics."""
            if not self.enhanced_processor or not self.enhanced_processor.behavior_detector:
                return jsonify({'available': False})
            
            metrics = self.enhanced_processor.behavior_detector.metrics.to_dict()
            state = self.enhanced_processor.behavior_detector.get_current_state()
            
            return jsonify({
                'available': True,
                'metrics': metrics,
                'state': state,
                'frame_count': self.enhanced_processor.frame_count
            })
        
        @self.app.route('/api/recent_events')
        def get_recent_events():
            """Get recent behavioral events."""
            if not self.enhanced_processor or not self.enhanced_processor.behavior_detector:
                return jsonify({'events': []})
            
            # Get last 10 events
            all_events = self.enhanced_processor.behavior_detector.event_bus.get_all_events()
            recent_events = [e.to_dict() for e in all_events[-10:]]
            
            return jsonify({'events': recent_events})
        
        @self.app.route('/api/export_data', methods=['POST'])
        def export_session_data():
            """Export session data."""
            if not self.enhanced_processor:
                return jsonify({'success': False, 'error': 'No active session'})
            
            export_path = self.enhanced_processor.export_session_data()
            if export_path:
                return jsonify({'success': True, 'export_path': str(export_path)})
            else:
                return jsonify({'success': False, 'error': 'Export failed'})
        
        @self.app.route('/api/trajectory_analysis')
        def get_trajectory_analysis():
            """Get trajectory analysis."""
            if not self.enhanced_processor:
                return jsonify({'available': False})
            
            analysis = self.enhanced_processor.get_trajectory_analysis()
            if analysis:
                return jsonify({'available': True, 'analysis': analysis})
            else:
                return jsonify({'available': False, 'error': 'Insufficient data'})
        
        @self.app.route('/api/reset_session', methods=['POST'])
        def reset_session():
            """Reset current session."""
            if self.enhanced_processor:
                self.enhanced_processor.reset_session()
                return jsonify({'success': True})
            return jsonify({'success': False})
        
        # Add compatibility routes for original interface
        @self.app.route('/api/videos')
        def list_videos():
            """List available videos (compatibility with original interface)."""
            videos = self._get_available_videos()
            return jsonify(videos)
        
        @self.app.route('/api/start', methods=['POST'])
        def start_pipeline():
            """Start pipeline (compatibility route that delegates to enhanced processing)."""
            config = request.json
            # Convert old config format to new enhanced format
            enhanced_config = {
                'video_path': config.get('video_path'),
                'model_weights': config.get('model_weights', 'output/models/head_pose/best.pt'),
                'imgsz': config.get('imgsz', 640),
                'conf': config.get('conf', 0.25),
                'device': config.get('device', '0'),
                'enable_behavioral_analysis': True,  # Enable by default
                'detect_approach': True,
                'detect_retreat': True,
                'detect_stop': True,
            }
            
            # Initialize enhanced processor
            model_weights = enhanced_config.get('model_weights')
            self.enhanced_processor = EnhancedPoseWebInterface(
                model_weights=model_weights,
                imgsz=enhanced_config.get('imgsz', 640),
                conf=enhanced_config.get('conf', 0.25),
                device=enhanced_config.get('device', '0')
            )
            
            # Configure behavioral analysis
            if enhanced_config.get('enable_behavioral_analysis', False):
                behavior_config = {
                    'detect_approach': enhanced_config.get('detect_approach', True),
                    'detect_retreat': enhanced_config.get('detect_retreat', True),
                    'detect_stop': enhanced_config.get('detect_stop', True),
                    'approach_threshold': enhanced_config.get('approach_threshold', 100),
                    'retreat_threshold': enhanced_config.get('retreat_threshold', 300),
                    'stop_threshold': enhanced_config.get('stop_threshold', 5),
                    'reference_x': enhanced_config.get('reference_x', 320),
                    'reference_y': enhanced_config.get('reference_y', 240),
                }
                self.enhanced_processor.configure_behavioral_analysis(behavior_config)
            
            # Start video processing
            success = self._start_video_processing(enhanced_config.get('video_path'))
            
            return jsonify({
                'success': success,
                'job_id': 'enhanced_local',
                'immediate_start': True
            })
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_pipeline():
            """Stop pipeline (compatibility route)."""
            self._processing_active = False
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=2.0)
            return jsonify({'success': True})
    
    def _get_available_videos(self):
        """Scan for available video files (using working logic from original)."""
        videos = []
        seen_paths = set()  # Track unique video paths to avoid duplicates
        print("🔍 Starting video scan...")
        
        # Get project root and build relative paths
        print(f"📁 Project root: {project_root}")
        
        # Look in common video directories - use relative paths from project root
        video_dirs = [
            project_root / "videos",  # Main project videos
            project_root / "scripts",  # Scripts folder
            project_root / "dataset" / "videos",  # Dataset videos
            project_root / "data" / "videos",  # Data videos
            project_root / "pose-head" / "videos",  # Pose-head videos
            Path("videos"),  # Local relative videos
            Path("../videos"),  # Parent directory videos
            Path("../../videos"),  # Grandparent directory videos
            Path("scripts"),  # Local scripts folder
            Path("../scripts"),  # Parent scripts
            Path("../../scripts"),  # Grandparent scripts
        ]
        
        for video_dir in video_dirs:
            abs_path = video_dir.resolve() if video_dir.exists() else video_dir
            print(f"📂 Checking: {video_dir} -> {abs_path}")
            
            if video_dir.exists():
                print(f"✅ Directory exists: {video_dir}")
                found_in_dir = 0
                for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                    for video_file in video_dir.glob(ext):
                        video_path = str(video_file.absolute())
                        if video_path not in seen_paths:  # Only add unique paths
                            seen_paths.add(video_path)
                            videos.append({
                                'name': video_file.name,
                                'path': video_path
                            })
                            print(f"🎬 Found video: {video_file.name}")
                            found_in_dir += 1
                if found_in_dir == 0:
                    print(f"   (No video files in {video_dir})")
            else:
                print(f"❌ Directory not found: {video_dir}")
        
        print(f"🎬 Total videos found: {len(videos)}")
        if len(videos) == 0:
            print("⚠️  No videos found! Please check that video files exist in:")
            print(f"   - {project_root}/videos/ directory")
            print(f"   - {project_root}/scripts/ directory") 
            print("   - videos/ relative to current directory")
        
        return videos
    
    def _start_video_processing(self, video_path: str) -> bool:
        """Start video processing in background thread."""
        if not self.enhanced_processor:
            return False
        
        def process_video():
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open video: {video_path}")
                    return
                
                self._processing_active = True
                print(f"Starting enhanced processing for: {video_path}")
                
                while self._processing_active and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame with enhanced libraries
                    result = self.enhanced_processor.process_frame(frame)
                    
                    # Update current frame for streaming
                    with self._frame_lock:
                        self.current_frame = result['frame_with_overlay']
                    
                    # Log events
                    if result.get('behavior_events'):
                        for event in result['behavior_events']:
                            print(f"Behavioral event: {event['event_type']} at frame {result['frame_number']}")
                    
                    # Control playback speed
                    time.sleep(1/30)  # 30 FPS
                
                cap.release()
                print("Video processing completed")
                
            except Exception as e:
                print(f"Error in video processing: {e}")
            finally:
                self._processing_active = False
        
        self._processing_thread = threading.Thread(target=process_video, daemon=True)
        self._processing_thread.start()
        return True
    
    def _generate_video_stream(self):
        """Generate video stream for web interface."""
        while True:
            with self._frame_lock:
                frame = self.current_frame
            
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1/30)  # 30 FPS
    
    def run(self, debug=False):
        """Run the modernized web interface."""
        print(f"🚀 Starting Modernized Pose-Head Web Interface")
        print(f"📍 URL: http://{self.host}:{self.port}")
        print(f"✨ Features:")
        print(f"   - Professional pose detection (lib/lizard_tracking)")
        print(f"   - Advanced behavioral analysis (lib/behavioral_analysis)")
        print(f"   - Real-time metrics and event detection")
        print(f"   - Data export and trajectory analysis")
        
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


if __name__ == "__main__":
    interface = ModernizedWebInterface()
    interface.run(debug=True)