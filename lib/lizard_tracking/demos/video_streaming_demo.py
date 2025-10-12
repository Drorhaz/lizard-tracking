#!/usr/bin/env python3
"""
Simple Video Streaming Web Page
Uses the extracted web_video_streaming module to stream video at 30 FPS
Enhanced with start/pause/resume/restart functionality
"""
import sys
import os
from pathlib import Path

# Add lib to path (adjust path since we're in lib/demos)
lib_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, lib_path)

try:
    from flask import Flask, render_template_string
    from lizard_tracking.utils.web_video_streaming import create_web_video_player
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're using the correct conda environment")
    sys.exit(1)

# Video configuration
VIDEO_PATH = "/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/pose-head/videos/top_20250916T150021.mp4"
TARGET_FPS = 10

# Create Flask app
app = Flask(__name__)

# Global video player
video_player = None

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🦎 Lizard Video Stream - 30 FPS</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        
        .video-info {
            font-size: 0.9em;
            opacity: 0.7;
            background: rgba(0,0,0,0.3);
            padding: 10px 20px;
            border-radius: 20px;
            margin-bottom: 20px;
        }
        
        .video-container {
            background: rgba(0,0,0,0.4);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        
        .video-stream {
            max-width: 900px;
            width: 100%;
            height: auto;
            border-radius: 10px;
            border: 3px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            background: #000;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .btn {
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            text-decoration: none;
            transition: all 0.3s ease;
            display: inline-block;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .btn.success {
            background: linear-gradient(45deg, #27ae60, #229954);
        }
        
        .btn.info {
            background: linear-gradient(45deg, #3498db, #2980b9);
        }
        
        .status-bar {
            background: rgba(0,0,0,0.3);
            padding: 15px 25px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            min-width: 300px;
        }
        
        .status-item {
            display: inline-block;
            margin: 0 15px;
            font-size: 0.9em;
        }
        
        .status-value {
            font-weight: bold;
            color: #f1c40f;
        }
        
        .loading {
            color: #f39c12;
        }
        
        .running {
            color: #2ecc71;
        }
        
        .paused {
            color: #f39c12;
        }
        
        .stopped {
            color: #e74c3c;
        }
        
        .footer {
            margin-top: 30px;
            text-align: center;
            opacity: 0.6;
            font-size: 0.8em;
        }
        
        @media (max-width: 768px) {
            .video-stream {
                max-width: 100%;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🦎 Lizard Video Stream</h1>
        <div class="subtitle">High-Quality Video Streaming at 30 FPS</div>
        <div class="video-info">
            📁 {{ video_path }}<br>
            🎬 Target FPS: {{ target_fps }} | 📦 Using extracted web_video_streaming module<br>
            ⏱️ Video timestamp overlay enabled (configurable position and color)
        </div>
    </div>
    
    <div class="video-container">
        <img id="videoStream" 
             src="/video_feed" 
             class="video-stream" 
             alt="Loading video stream..."
             onload="updateStatus('running')"
             onerror="updateStatus('error')">
        
        <div class="controls">
            <button class="btn success" onclick="startStream()">▶️ Start Stream</button>
            <button class="btn" onclick="pauseStream()">⏸️ Pause</button>
            <button class="btn info" onclick="restartStream()">🔄 Restart</button>
            <button class="btn info" onclick="refreshStats()">📊 Refresh Stats</button>
        </div>
    </div>
    
    <div class="status-bar">
        <div class="status-item">
            Status: <span id="streamStatus" class="status-value loading">Loading...</span>
        </div>
        <div class="status-item">
            Frames: <span id="frameCount" class="status-value">--</span>
        </div>
        <div class="status-item">
            FPS: <span id="currentFps" class="status-value">--</span>
        </div>
        <div class="status-item">
            Time: <span id="elapsedTime" class="status-value">--</span>s
        </div>
    </div>
    
    <div class="footer">
        <p>🔬 Powered by extracted web_video_streaming module from pose-head</p>
        <p>📍 Module: lib/lizard_tracking/utils/web_video_streaming.py</p>
        <p>🎮 Controls: Start → Pause → Resume | Restart from beginning</p>
    </div>
    
    <script>
        let updateInterval = null;
        
        function startStream() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('✅ Stream started/resumed successfully');
                        document.getElementById('videoStream').src = '/video_feed?' + Date.now();
                        startStatusUpdates();
                    } else {
                        console.error('❌ Failed to start stream:', data.error);
                        updateStatus('error');
                    }
                })
                .catch(err => {
                    console.error('❌ Start request failed:', err);
                    updateStatus('error');
                });
        }
        
        function pauseStream() {
            fetch('/api/pause', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('⏸️ Stream paused successfully');
                        updateStatus('paused');
                    } else {
                        console.error('❌ Failed to pause stream');
                    }
                });
        }
        
        function restartStream() {
            fetch('/api/restart', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('🔄 Stream restarted successfully');
                        document.getElementById('videoStream').src = '/video_feed?' + Date.now();
                        updateStatus('running');
                        startStatusUpdates();
                    } else {
                        console.error('❌ Failed to restart stream:', data.error);
                    }
                })
                .catch(err => {
                    console.error('❌ Restart request failed:', err);
                    updateStatus('error');
                });
        }
        
        function refreshStats() {
            updateStats();
            console.log('📊 Stream stats refreshed');
        }
        
        function updateStatus(status) {
            const statusEl = document.getElementById('streamStatus');
            statusEl.className = 'status-value ' + status;
            
            switch(status) {
                case 'running':
                    statusEl.textContent = 'Running ✅';
                    break;
                case 'paused':
                    statusEl.textContent = 'Paused ⏸️';
                    break;
                case 'stopped':
                    statusEl.textContent = 'Stopped ⏹️';
                    break;
                case 'error':
                    statusEl.textContent = 'Error ❌';
                    break;
                default:
                    statusEl.textContent = 'Loading...';
            }
        }
        
        function updateStats() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('frameCount').textContent = data.frame_count || '--';
                    document.getElementById('currentFps').textContent = data.fps ? data.fps.toFixed(1) : '--';
                    document.getElementById('elapsedTime').textContent = data.elapsed ? data.elapsed.toFixed(1) : '--';
                    
                    if (data.running) {
                        updateStatus('running');
                    } else if (data.frame_count > 0) {
                        updateStatus('paused');
                    }
                })
                .catch(err => {
                    console.error('Stats update failed:', err);
                });
        }
        
        function startStatusUpdates() {
            if (updateInterval) clearInterval(updateInterval);
            updateInterval = setInterval(updateStats, 500);
        }
        
        function stopStatusUpdates() {
            if (updateInterval) {
                clearInterval(updateInterval);
                updateInterval = null;
            }
        }
        
        // Page load - wait for user to start
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🚀 Page loaded, waiting for user to start video...');
            startStatusUpdates();
        });
        
        console.log('🦎 Lizard Video Streaming Interface Ready');
    </script>
</body>
</html>
"""

def init_video_player():
    """Initialize the video player"""
    global video_player
    
    try:
        # Check if video file exists
        if not Path(VIDEO_PATH).exists():
            print(f"❌ Video file not found: {VIDEO_PATH}")
            return False
        
        print(f"📹 Initializing video player...")
        print(f"   Video path: {VIDEO_PATH}")
        print(f"   Target FPS: {TARGET_FPS}")
        
        # Create video player with our extracted module (with timestamp overlay)
        video_player = create_web_video_player(VIDEO_PATH, target_fps=TARGET_FPS, show_timestamp=True)
        
        # Configure timestamp appearance (top-left, white text)
        video_player.streamer.configure_timestamp(
            show=True, 
            position=(15, 35),  # Top-left corner with some padding
            color=(255, 255, 255)  # White text (BGR format)
        )
        
        if video_player.start():
            print(f"✅ Video player started successfully at {TARGET_FPS} FPS")
            return True
        else:
            print(f"❌ Failed to start video player")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing video player: {e}")
        return False

@app.route('/')
def index():
    """Main page with video stream"""
    return render_template_string(
        HTML_TEMPLATE, 
        video_path=VIDEO_PATH,
        target_fps=TARGET_FPS
    )

@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    global video_player
    
    if video_player is None:
        init_video_player()
    
    if video_player is None:
        # Return error response
        from flask import Response
        def error_stream():
            import cv2
            import numpy as np
            import time
            
            while True:
                # Create error frame
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, 'Click "Start Stream" to begin', (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f'File: {Path(VIDEO_PATH).name}', (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(img, 'Pause/Resume and Restart available', (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', img)
                frame_data = buffer.tobytes()
                
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
                time.sleep(1)
        
        return Response(error_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Use our extracted module
    integration = video_player.get_flask_integration()
    return integration.video_feed_response()

@app.route('/api/status')
def api_status():
    """Status API endpoint"""
    global video_player
    
    if video_player is None:
        return {
            'running': False,
            'frame_count': 0,
            'fps': 0.0,
            'elapsed': 0.0,
            'error': 'Video player not initialized'
        }
    
    # Use our extracted module
    integration = video_player.get_flask_integration()
    return integration.status_response()

@app.route('/api/start', methods=['POST'])
def start_stream():
    """Start or resume video streaming"""
    global video_player
    
    try:
        # If player doesn't exist, create it
        if video_player is None:
            if not init_video_player():
                return {
                    'success': False,
                    'error': 'Failed to initialize video player'
                }
        # If player exists but stopped, restart it
        elif not video_player.running:
            if not video_player.start():
                return {
                    'success': False,
                    'error': 'Failed to resume video player'
                }
        
        return {
            'success': True,
            'message': f'Video streaming active at {TARGET_FPS} FPS',
            'fps': TARGET_FPS,
            'video_path': str(Path(VIDEO_PATH).name)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error starting stream: {str(e)}'
        }

@app.route('/api/pause', methods=['POST'])
def pause_stream():
    """Pause video streaming (can be resumed)"""
    global video_player
    
    try:
        if video_player is not None:
            video_player.stop()  # This pauses the video
        
        return {
            'success': True,
            'message': 'Video streaming paused (can be resumed with Start)'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error pausing stream: {str(e)}'
        }

@app.route('/api/restart', methods=['POST'])
def restart_stream():
    """Restart video streaming from beginning"""
    global video_player
    
    try:
        # Stop existing player
        if video_player is not None:
            video_player.stop()
            video_player = None
        
        # Create new player (this restarts from beginning)
        if not init_video_player():
            return {
                'success': False,
                'error': 'Failed to restart video player'
            }
        
        return {
            'success': True,
            'message': f'Video restarted from beginning at {TARGET_FPS} FPS',
            'fps': TARGET_FPS,
            'video_path': str(Path(VIDEO_PATH).name)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error restarting stream: {str(e)}'
        }

def main():
    """Main function"""
    print("🚀 Starting Enhanced Lizard Video Streaming Server")
    print("=" * 50)
    print(f"📹 Video: {Path(VIDEO_PATH).name}")
    print(f"🎬 Target FPS: {TARGET_FPS}")
    print(f"📦 Using: web_video_streaming module")
    print()
    print("🌐 Server will start on:")
    print("   http://localhost:8080")
    print()
    print("✨ Features:")
    print("   - Manual start/pause/resume video streaming")
    print("   - Restart video from beginning") 
    print("   - Real-time FPS monitoring")
    print("   - Responsive design")
    print("   - 30 FPS smooth playback")
    print("   - Video timestamp overlay (configurable)")
    print()
    print("🎮 Controls:")
    print("   ▶️ Start Stream - Begin or resume video")
    print("   ⏸️ Pause - Pause video (keeps position)")
    print("   🔄 Restart - Start from beginning")
    print("   📊 Refresh Stats - Update statistics")
    print()
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if video_player:
            video_player.stop()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        if video_player:
            video_player.stop()

if __name__ == "__main__":
    main()