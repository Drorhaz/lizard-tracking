#!/usr/bin/env python3
"""
Video Streaming Library
Extracted from pose-head for reusable video streaming functionality
"""
from __future__ import annotations
import cv2
import numpy as np
import threading
import time
import io
from typing import Optional, Callable, Any, Dict, List, Tuple
from pathlib import Path
import json


class VideoStream:
    """
    Real-time video streaming with frame processing and web interface support.
    
    Features:
    - Live video capture from camera or file
    - Frame processing pipeline with callbacks
    - JPEG compression for web streaming
    - Thread-safe frame updates
    - FPS control and monitoring
    """
    
    def __init__(self, source: Optional[str] = None, fps: float = 30.0):
        """
        Initialize video stream.
        
        Args:
            source: Video source (camera index, file path, or None for no source)
            fps: Target frames per second
        """
        self.source = source
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading and synchronization
        self._frame_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._jpeg_frame: Optional[bytes] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Frame processing
        self._frame_processors: List[Callable[[np.ndarray, int], np.ndarray]] = []
        self._frame_callbacks: List[Callable[[np.ndarray, int], None]] = []
        
        # Statistics
        self._frame_count = 0
        self._start_time = 0
        self._actual_fps = 0.0
        self._last_fps_update = 0
        
        # Configuration
        self.jpeg_quality = 85
        self.auto_resize = True
        self.max_width = 1920
        self.max_height = 1080
    
    def add_frame_processor(self, processor: Callable[[np.ndarray, int], np.ndarray]):
        """
        Add a frame processor function.
        
        Args:
            processor: Function that takes (frame, frame_number) and returns processed frame
        """
        self._frame_processors.append(processor)
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, int], None]):
        """
        Add a frame callback function.
        
        Args:
            callback: Function called with (frame, frame_number) for each frame
        """
        self._frame_callbacks.append(callback)
    
    def start(self) -> bool:
        """
        Start the video stream.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            print("⚠️  Video stream already running")
            return True
        
        # Initialize video capture if source provided
        if self.source is not None:
            if isinstance(self.source, (int, str)):
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    print(f"❌ Failed to open video source: {self.source}")
                    return False
                
                # Set camera properties for live sources
                if isinstance(self.source, int):
                    self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                
                print(f"✅ Opened video source: {self.source}")
            else:
                print(f"❌ Invalid video source type: {type(self.source)}")
                return False
        
        # Start processing thread
        self._running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        
        print(f"🎬 Video stream started (target FPS: {self.target_fps})")
        return True
    
    def stop(self):
        """Stop the video stream."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self._frame_lock:
            self._current_frame = None
            self._jpeg_frame = None
        
        print("🛑 Video stream stopped")
    
    def update_frame(self, frame: np.ndarray):
        """
        Manually update the current frame (for external frame sources).
        
        Args:
            frame: New frame to display
        """
        if not self._running:
            return
        
        # Process frame
        processed_frame = self._process_frame(frame, self._frame_count)
        
        # Update frame data
        with self._frame_lock:
            self._current_frame = processed_frame.copy()
            # Encode JPEG for streaming
            self._encode_jpeg(processed_frame)
        
        # Call callbacks
        for callback in self._frame_callbacks:
            try:
                callback(processed_frame, self._frame_count)
            except Exception as e:
                print(f"⚠️  Error in frame callback: {e}")
        
        self._frame_count += 1
        self._update_fps()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame as numpy array."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def get_jpeg_frame(self) -> Optional[bytes]:
        """Get the current frame as JPEG bytes for web streaming."""
        with self._frame_lock:
            return self._jpeg_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        return {
            'running': self._running,
            'frame_count': self._frame_count,
            'elapsed_time': elapsed,
            'target_fps': self.target_fps,
            'actual_fps': self._actual_fps,
            'source': str(self.source) if self.source is not None else 'manual',
            'has_current_frame': self._current_frame is not None
        }
    
    def _stream_loop(self):
        """Main streaming loop (runs in thread)."""
        frame_interval = 1.0 / self.target_fps
        
        while self._running:
            loop_start = time.time()
            
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.update_frame(frame)
                else:
                    print("⚠️  Failed to read frame from video source")
                    break
            else:
                # No video source, just sleep
                time.sleep(frame_interval)
                continue
            
            # FPS control
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process frame through all registered processors."""
        processed = frame.copy()
        
        # Auto-resize if enabled
        if self.auto_resize:
            h, w = processed.shape[:2]
            if w > self.max_width or h > self.max_height:
                scale = min(self.max_width / w, self.max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h))
        
        # Apply frame processors
        for processor in self._frame_processors:
            try:
                processed = processor(processed, frame_number)
            except Exception as e:
                print(f"⚠️  Error in frame processor: {e}")
        
        return processed
    
    def _encode_jpeg(self, frame: np.ndarray):
        """Encode frame as JPEG for streaming."""
        try:
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                self._jpeg_frame = buf.tobytes()
        except Exception as e:
            print(f"⚠️  Error encoding JPEG: {e}")
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self._last_fps_update >= 1.0:  # Update every second
            elapsed = current_time - self._start_time
            if elapsed > 0:
                self._actual_fps = self._frame_count / elapsed
            self._last_fps_update = current_time


class FlaskVideoStreamer:
    """
    Flask integration for video streaming.
    Provides routes for video feeds and status endpoints.
    """
    
    def __init__(self, video_stream: VideoStream):
        self.video_stream = video_stream
    
    def generate_frames(self):
        """
        Generator function for Flask video streaming.
        Yields MJPEG frames for web display.
        """
        while True:
            frame_data = self.video_stream.get_jpeg_frame()
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                # No frame available, yield empty frame or wait
                time.sleep(0.033)  # ~30 FPS polling
    
    def setup_routes(self, app):
        """
        Setup Flask routes for video streaming.
        
        Args:
            app: Flask application instance
        """
        from flask import Response, jsonify
        
        @app.route('/video_feed')
        def video_feed():
            """Video streaming route."""
            return Response(
                self.generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @app.route('/api/stream/status')
        def stream_status():
            """Get streaming status."""
            return jsonify(self.video_stream.get_stats())
        
        @app.route('/api/stream/start', methods=['POST'])
        def start_stream():
            """Start video stream."""
            success = self.video_stream.start()
            return jsonify({'success': success})
        
        @app.route('/api/stream/stop', methods=['POST'])
        def stop_stream():
            """Stop video stream."""
            self.video_stream.stop()
            return jsonify({'success': True})


class VideoPlayer:
    """
    Simple video player for testing and demonstration.
    """
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.stream = VideoStream(str(video_path))
        self.window_name = f"Video Player - {self.video_path.name}"
    
    def play(self, show_fps: bool = True):
        """
        Play video with OpenCV window display.
        
        Args:
            show_fps: Whether to show FPS counter
        """
        if not self.video_path.exists():
            print(f"❌ Video file not found: {self.video_path}")
            return
        
        # Add FPS display processor if requested
        if show_fps:
            self.stream.add_frame_processor(self._add_fps_overlay)
        
        # Start stream
        if not self.stream.start():
            print("❌ Failed to start video stream")
            return
        
        print(f"🎬 Playing: {self.video_path}")
        print("Press 'q' to quit, 'p' to pause, 's' for stats")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    frame = self.stream.get_current_frame()
                    if frame is not None:
                        cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    stats = self.stream.get_stats()
                    print(f"📊 Stats: {json.dumps(stats, indent=2)}")
                
                # Check if stream ended
                if not self.stream._running:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.stream.stop()
            cv2.destroyAllWindows()
    
    def _add_fps_overlay(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Add FPS overlay to frame."""
        stats = self.stream.get_stats()
        fps_text = f"FPS: {stats['actual_fps']:.1f} | Frame: {frame_number}"
        
        # Add text with background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(fps_text, font, scale, thickness)
        
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (text_w + 20, text_h + 20), (0, 0, 0), -1)
        # Text
        cv2.putText(frame, fps_text, (15, text_h + 15), font, scale, (0, 255, 0), thickness)
        
        return frame


# Convenience functions
def stream_video(video_path: str) -> VideoPlayer:
    """
    Create a video player for the given video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        VideoPlayer instance
    """
    return VideoPlayer(video_path)


def create_camera_stream(camera_index: int = 0, fps: float = 30.0) -> VideoStream:
    """
    Create a video stream from camera.
    
    Args:
        camera_index: Camera device index (usually 0)
        fps: Target frames per second
        
    Returns:
        VideoStream instance
    """
    return VideoStream(camera_index, fps)


def create_file_stream(video_path: str, fps: float = 30.0) -> VideoStream:
    """
    Create a video stream from file.
    
    Args:
        video_path: Path to video file
        fps: Target frames per second
        
    Returns:
        VideoStream instance
    """
    return VideoStream(video_path, fps)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("🎬 Video Streaming Library Test")
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        player = stream_video(video_path)
        player.play()
    else:
        print("Usage: python video_stream.py <video_path>")
        print("Example: python video_stream.py test_video.mp4")
        
        # Test with camera if available
        print("\n🎥 Testing camera stream...")
        camera_stream = create_camera_stream(0)
        if camera_stream.start():
            print("✅ Camera stream started")
            time.sleep(2)
            stats = camera_stream.get_stats()
            print(f"📊 Camera stats: {json.dumps(stats, indent=2)}")
            camera_stream.stop()
        else:
            print("❌ No camera available")