#!/usr/bin/env python3
"""
Web Video Streaming Module
Extracted from pose-head system for reusable video streaming functionality
"""
import cv2
import threading
import time
from typing import Optional, Dict, Any, Generator
from pathlib import Path

class WebVideoStreamer:
    """
    Reusable web video streaming interface extracted from pose-head.
    
    This class manages:
    - Video frame streaming for web display
    - JPEG encoding and buffering
    - Thread-safe frame updates
    - Flask/web integration
    - Video timestamp overlay
    """
    
    def __init__(self, show_timestamp=True, timestamp_position=(10, 30), timestamp_color=(255, 255, 255)):
        self._jpeg = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0
        self._start_time = time.time()
        self.show_timestamp = show_timestamp
        self.timestamp_position = timestamp_position
        self.timestamp_color = timestamp_color
        
    def update(self, bgr_frame):
        """
        Update video stream with new frame.
        
        Args:
            bgr_frame: Frame image in BGR format (OpenCV format)
        """
        if bgr_frame is None:
            return
        
        # Add timestamp overlay if enabled
        if self.show_timestamp:
            frame_with_timestamp = bgr_frame.copy()
            elapsed_seconds = time.time() - self._start_time
            timestamp_text = f"{elapsed_seconds:.1f}s"
            
            import cv2
            cv2.putText(frame_with_timestamp, timestamp_text, self.timestamp_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.timestamp_color, 2)
            bgr_frame = frame_with_timestamp
            
        # Encode frame for streaming (copied from pose-head)
        ok, buf = cv2.imencode('.jpg', bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self._frame_lock:
                self._jpeg = buf.tobytes()
                self._frame_count += 1

    def get_frame(self) -> Optional[bytes]:
        """Get the current frame as JPEG bytes for streaming"""
        with self._frame_lock:
            return self._jpeg

    def clear_frame(self):
        """Clear the current frame (stop streaming)"""
        with self._frame_lock:
            self._jpeg = None
        print("🔄 Video stream cleared")
    
    def configure_timestamp(self, show=True, position=(10, 30), color=(255, 255, 255)):
        """
        Configure timestamp overlay settings.
        
        Args:
            show: Whether to show timestamp overlay
            position: (x, y) position for timestamp text
            color: (B, G, R) color for timestamp text in BGR format
        """
        self.show_timestamp = show
        self.timestamp_position = position
        self.timestamp_color = color
        print(f"📊 Timestamp overlay: {'enabled' if show else 'disabled'} at {position}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'frame_count': self._frame_count,
            'elapsed_time': elapsed,
            'average_fps': fps,
            'has_frame': self._jpeg is not None,
            'timestamp_enabled': self.show_timestamp
        }
    
    def generate_stream(self) -> Generator[bytes, None, None]:
        """
        Generate video stream for Flask Response (copied from pose-head approach).
        
        Yields:
            MJPEG stream bytes with proper HTTP headers
        """
        while True:
            frame = self.get_frame()
            
            if frame is None:
                time.sleep(0.1)  # Sleep if no frame available
                continue
                
            # MJPEG streaming format (from pose-head)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.1)  # 10 FPS max streaming rate (from pose-head)

class FlaskVideoIntegration:
    """
    Flask integration for web video streaming.
    Provides easy Flask route setup.
    """
    
    def __init__(self, streamer: WebVideoStreamer):
        self.streamer = streamer
    
    def video_feed_response(self):
        """
        Create Flask Response for video feed route.
        
        Usage in Flask app:
            from flask import Response
            
            @app.route('/video_feed')
            def video_feed():
                return integration.video_feed_response()
        """
        try:
            from flask import Response
            return Response(self.streamer.generate_stream(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        except ImportError:
            raise ImportError("Flask is required for FlaskVideoIntegration")
    
    def status_response(self):
        """
        Create Flask JSON response for status endpoint.
        
        Usage in Flask app:
            from flask import jsonify
            
            @app.route('/api/status')
            def status():
                return integration.status_response()
        """
        try:
            from flask import jsonify
            stats = self.streamer.get_stats()
            return jsonify({
                'running': stats['has_frame'],
                'frame_count': stats['frame_count'],
                'fps': round(stats['average_fps'], 1),
                'elapsed': round(stats['elapsed_time'], 1)
            })
        except ImportError:
            raise ImportError("Flask is required for FlaskVideoIntegration")

class SimpleVideoPlayer:
    """
    Simple video player that feeds frames to WebVideoStreamer.
    Extracted and simplified from pose-head approach.
    """
    
    def __init__(self, video_path: str, target_fps: float = 10.0, show_timestamp: bool = True):
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.cap = None
        self.running = False
        self.thread = None
        self.streamer = WebVideoStreamer(show_timestamp=show_timestamp)
        
    def start(self) -> bool:
        """Start video playback"""
        if self.running:
            return True
            
        # Open video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"❌ Could not open video: {self.video_path}")
            return False
        
        # Get video properties
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Opened video: {self.video_path}")
        print(f"📊 Video info: {total_frames} frames at {original_fps:.1f} FPS")
        print(f"🎬 Streaming at target FPS: {self.target_fps}")
        
        self.running = True
        
        # Start playback thread
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()
        
        return True
    
    def stop(self):
        """Stop video playback"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        self.streamer.clear_frame()
        print("🛑 Video playback stopped")
    
    def _playback_loop(self):
        """Main playback loop (runs in background thread)"""
        frame_interval = 1.0 / self.target_fps
        
        while self.running and self.cap:
            ret, frame = self.cap.read()
            
            if ret:
                # Feed frame to streamer
                self.streamer.update(frame)
            else:
                # End of video - restart (loop)
                print("🔄 Restarting video loop...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            time.sleep(frame_interval)
    
    def get_flask_integration(self) -> FlaskVideoIntegration:
        """Get Flask integration object"""
        return FlaskVideoIntegration(self.streamer)

# Convenience function for easy usage
def create_web_video_player(video_path: str, target_fps: float = 10.0, show_timestamp: bool = True) -> SimpleVideoPlayer:
    """
    Create a simple web video player.
    
    Args:
        video_path: Path to video file
        target_fps: Target streaming FPS (default: 10.0)
        show_timestamp: Whether to show elapsed time overlay (default: True)
    
    Returns:
        SimpleVideoPlayer instance ready to start
    
    Example:
        player = create_web_video_player("video.mp4", fps=15, show_timestamp=True)
        player.start()
        
        # Configure timestamp appearance
        player.streamer.configure_timestamp(show=True, position=(20, 40), color=(0, 255, 0))
        
        # In Flask app:
        integration = player.get_flask_integration()
        
        @app.route('/video_feed')
        def video_feed():
            return integration.video_feed_response()
    """
    return SimpleVideoPlayer(video_path, target_fps, show_timestamp)

# Global instance approach (similar to pose-head shared interface)
_global_streamer: Optional[WebVideoStreamer] = None
_global_lock = threading.Lock()

def get_global_streamer() -> WebVideoStreamer:
    """Get global video streamer instance (singleton)"""
    global _global_streamer
    
    if _global_streamer is None:
        with _global_lock:
            if _global_streamer is None:
                _global_streamer = WebVideoStreamer()
    
    return _global_streamer

def update_global_stream(bgr_frame):
    """Update global video stream with new frame"""
    streamer = get_global_streamer()
    streamer.update(bgr_frame)

def get_global_flask_integration() -> FlaskVideoIntegration:
    """Get Flask integration for global streamer"""
    streamer = get_global_streamer()
    return FlaskVideoIntegration(streamer)