#!/usr/bin/env python3
"""
HPC Web Interface for Pose Pipeline
Provides a web UI to submit GPU jobs and view real-time inference results
"""
from __future__ import annotations
import io, threading, time, os, subprocess, json, uuid
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import cv2, numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from pipeline.shared_web_interface import SharedWebInterface

# HTML Template with Play Button and Controls
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <title>HPC Pose Pipeline</title>
    <style>
        body { 
            background: #1a1a1a; 
            margin: 0; 
            font-family: Arial, sans-serif; 
            color: #fff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .controls {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .video-container {
            text-align: center;
            background: #000;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .video-stream {
            max-width: 95vw;
            max-height: 60vh;
            border-radius: 8px;
        }
        .status {
            background: #333;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .btn {
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .btn:hover { background: #45a049; }
        .btn:disabled { background: #666; cursor: not-allowed; }
        .btn-stop { background: #f44336; }
        .btn-stop:hover { background: #da190b; }
        .form-group {
            margin: 10px 0;
        }
        .form-group label {
            display: inline-block;
            width: 150px;
            margin-right: 10px;
        }
        .form-group input, .form-group select {
            padding: 8px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #444;
            color: #fff;
            width: 200px;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #444;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            width: 0%;
            transition: width 0.3s ease;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 14px;
            color: #aaa;
        }
        .log {
            background: #000;
            color: #0f0;
            padding: 15px;
            border-radius: 8px;
            height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦎 HPC Pose Pipeline</h1>
            <p>Real-time lizard pose estimation on GPU cluster</p>
        </div>

        <div class="controls">
            <h3>Pipeline Controls</h3>
            <div class="form-group">
                <label>Video File:</label>
                <select id="videoSelect">
                    <option value="">Select video...</option>
                </select>
            </div>
            <div class="form-group">
                <label>Execution Mode:</label>
                <select id="executionMode" onchange="toggleExecutionOptions()">
                    <option value="local">Local (CPU)</option>
                    <option value="hpc">HPC Cluster (GPU)</option>
                </select>
            </div>

            <div class="form-group" id="timeLimitGroup" style="display:none;">
                <label>Time Limit:</label>
                <input type="text" id="timeLimit" value="02:00:00" placeholder="HH:MM:SS">
            </div>
            <div class="form-group">
                <label>Confidence Threshold:</label>
                <input type="number" id="confThresh" value="0.25" min="0.1" max="1.0" step="0.05">
            </div>
            <div class="form-group">
                <label>Image Size:</label>
                <input type="number" id="imgSize" value="960" min="320" max="1920" step="32">
            </div>
            
            <div style="margin-top: 20px;">
                <button class="btn" id="playBtn" onclick="startPipeline()">
                    ▶️ Start Pipeline
                </button>
                <button class="btn btn-stop hidden" id="stopBtn" onclick="stopPipeline()">
                    ⏹️ Stop Pipeline
                </button>
            </div>
        </div>

        <div class="status">
            <h3>Pipeline Status</h3>
            <div id="statusText">Ready to start</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div id="progressText">0% complete</div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="fpsValue">0.0</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="detectionRate">0.0%</div>
                <div class="stat-label">Detection Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="processedFrames">0</div>
                <div class="stat-label">Processed Frames</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="jobId">-</div>
                <div class="stat-label">SLURM Job ID</div>
            </div>
        </div>

        <div class="video-container">
            <h3>Live Pose Detection</h3>
            <img src="/video" class="video-stream" alt="Waiting for video stream...">
        </div>

        <div class="status">
            <h3>Pipeline Log</h3>
            <div class="log" id="logOutput">Waiting for pipeline to start...\n</div>
        </div>
    </div>

    <script>
        let jobId = null;
        let statusInterval = null;

        // Toggle execution mode options
        function toggleExecutionOptions() {
            const mode = document.getElementById('executionMode').value;
            const timeLimitGroup = document.getElementById('timeLimitGroup');
            const playBtn = document.getElementById('playBtn');
            
            if (mode === 'hpc') {
                timeLimitGroup.style.display = 'block';
                playBtn.innerHTML = '▶️ Start GPU Pipeline';
            } else {
                timeLimitGroup.style.display = 'none';
                playBtn.innerHTML = '▶️ Start Local Pipeline';
            }
        }

        // Load available videos on page load
        fetch('/api/videos')
            .then(r => r.json())
            .then(videos => {
                const select = document.getElementById('videoSelect');
                videos.forEach(video => {
                    const option = document.createElement('option');
                    option.value = video.path;
                    option.textContent = video.name;
                    select.appendChild(option);
                });
            });

        function startPipeline() {
            const executionMode = document.getElementById('executionMode').value;
            const config = {
                execution_mode: executionMode,
                video_path: document.getElementById('videoSelect').value,
                partition: 'gpu',  // Always use gpu partition
                time_limit: executionMode === 'hpc' ? document.getElementById('timeLimit').value : null,
                conf_thresh: parseFloat(document.getElementById('confThresh').value),
                img_size: parseInt(document.getElementById('imgSize').value)
            };

            if (!config.video_path) {
                alert('Please select a video file');
                return;
            }

            fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    jobId = data.job_id;
                    document.getElementById('jobId').textContent = jobId;
                    document.getElementById('playBtn').classList.add('hidden');
                    document.getElementById('stopBtn').classList.remove('hidden');
                    
                    if (data.immediate_start) {
                        // CPU mode: start video immediately
                        document.getElementById('statusText').textContent = 'Pipeline running locally - streaming video...';
                        startStatusUpdates();
                    } else {
                        // GPU mode: wait for job to start
                        document.getElementById('statusText').textContent = 'Job submitted to GPU cluster - waiting for start...';
                        startStatusUpdates();
                    }
                } else {
                    alert('Failed to start pipeline: ' + data.error);
                }
            });
        }

        function stopPipeline() {
            if (!jobId) return;
            
            fetch('/api/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({job_id: jobId})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('playBtn').classList.remove('hidden');
                document.getElementById('stopBtn').classList.add('hidden');
                document.getElementById('statusText').textContent = 'Pipeline stopped';
                stopStatusUpdates();
            });
        }

        function startStatusUpdates() {
            statusInterval = setInterval(updateStatus, 1000);
        }

        function stopStatusUpdates() {
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
        }

        function updateStatus() {
            if (!jobId) return;

            fetch('/api/status/' + jobId)
                .then(r => r.json())
                .then(data => {
                    document.getElementById('statusText').textContent = data.status;
                    document.getElementById('fpsValue').textContent = data.fps.toFixed(1);
                    document.getElementById('detectionRate').textContent = data.detection_rate.toFixed(1) + '%';
                    document.getElementById('processedFrames').textContent = data.processed_frames;
                    
                    if (data.progress !== undefined) {
                        document.getElementById('progressFill').style.width = data.progress + '%';
                        document.getElementById('progressText').textContent = data.progress.toFixed(1) + '% complete';
                    }

                    // Update log
                    if (data.log_lines) {
                        const logDiv = document.getElementById('logOutput');
                        data.log_lines.forEach(line => {
                            logDiv.innerHTML += line + '\\n';
                        });
                        logDiv.scrollTop = logDiv.scrollHeight;
                    }

                    // Check if job is finished
                    if (data.status.includes('Completed') || data.status.includes('Failed')) {
                        stopStatusUpdates();
                        document.getElementById('playBtn').classList.remove('hidden');
                        document.getElementById('stopBtn').classList.add('hidden');
                    }
                });
        }
    </script>
</body>
</html>
"""

class HPCWebInterface:
    """Web interface for submitting and monitoring HPC GPU jobs"""
    
    _instance = None  # Class variable to store the running instance
    
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._lock = threading.Lock()
        self._th = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Use shared web interface for video streaming
        self.shared_web = SharedWebInterface.get_instance()
        
        # Store instance for pipeline connection
        HPCWebInterface._instance = self
        
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
            
        @self.app.route('/video')
        def video():
            return Response(self._gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')
            
        @self.app.route('/api/videos')
        def list_videos():
            videos = self._get_available_videos()
            return jsonify(videos)
            
        @self.app.route('/api/start', methods=['POST'])
        def start_pipeline():
            config = request.json
            execution_mode = config.get('execution_mode', 'local')
            
            if execution_mode == 'local':
                # CPU mode: run locally and start video immediately
                success = self._start_local_pipeline(config)
                if success:
                    return jsonify({'success': True, 'job_id': 'local', 'immediate_start': True})
                else:
                    return jsonify({'success': False, 'error': 'Failed to start local pipeline'})
            else:
                # GPU mode: submit SLURM job and wait for it to start
                job_id = self._submit_hpc_job(config)
                if job_id:
                    return jsonify({'success': True, 'job_id': job_id, 'immediate_start': False})
                else:
                    return jsonify({'success': False, 'error': 'Failed to submit GPU job'})
                
        @self.app.route('/api/stop', methods=['POST'])
        def stop_pipeline():
            job_id = request.json.get('job_id')
            success = self._cancel_job(job_id)
            return jsonify({'success': success})
            
        @self.app.route('/api/status/<job_id>')
        def get_status(job_id):
            status = self._get_job_status(job_id)
            return jsonify(status)
    
    def _get_available_videos(self):
        """Scan for available video files"""
        videos = []
        video_dirs = [
            Path("../scripts"),  # Look in scripts folder as requested
            Path("../../scripts"),
            Path("scripts"),
            Path("../videos"),  # Keep existing video directories as fallback
            Path("../../videos"),
            Path("videos")
        ]
        
        for video_dir in video_dirs:
            if video_dir.exists():
                for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                    for video_file in video_dir.glob(ext):
                        videos.append({
                            'name': video_file.name,
                            'path': str(video_file.absolute())
                        })
        
        return videos
    
    def _start_local_pipeline(self, config):
        """Start pipeline locally for CPU mode - no job submission needed"""
        try:
            # Import required modules for pose detection
            import sys
            sys.path.append('/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/pose-head')
            from src.lizard_tracking.models import YOLOPoseModel
            from src.draw import draw_overlay
            
            def run_local_inference():
                """Run local inference with real-time streaming"""
                try:
                    print(f"▶ Starting local inference for {config['video_path']}")
                    
                    # Initialize video capture
                    cap = cv2.VideoCapture(config['video_path'])
                    if not cap.isOpened():
                        print(f"❌ Could not open video: {config['video_path']}")
                        return
                    
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"📹 Video: {total_frames} frames @ {fps:.1f} FPS")
                    
                    # Initialize YOLO model
                    model_dir = Path("output/models/head_pose")
                    model_paths = list(model_dir.glob("**/best*.pt")) + list(model_dir.glob("**/*.pt"))
                    if not model_paths:
                        print(f"❌ No model found in {model_dir}")
                        return
                    
                    model_path = model_paths[0]
                    print(f"🤖 Loading model: {model_path}")
                    
                    model = YOLOPoseModel(
                        model_path,
                        imgsz=config['img_size'],
                        conf=config['conf_thresh'],
                        iou=0.5
                    )
                    
                    frame_count = 0
                    detections = 0
                    
                    # Process video frames
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("📹 End of video reached")
                            break
                        
                        frame_count += 1
                        
                        # Run inference
                        results = model.predict(frame)
                        
                        # Find best detection
                        best_detection = None
                        best_conf = 0
                        
                        if results and len(results) > 0:
                            for result in results:
                                if hasattr(result, 'keypoints') and result.keypoints is not None:
                                    kpts = result.keypoints.data  # Shape: [N, num_keypoints, 3]
                                    boxes = result.boxes.data if result.boxes is not None else None
                                    
                                    if kpts.shape[0] > 0 and boxes is not None:
                                        # Get confidence from bounding box
                                        conf = float(boxes[0, 4])  # confidence score
                                        if conf > best_conf:
                                            best_conf = conf
                                            best_detection = {
                                                'bbox': boxes[0, :4].cpu().numpy(),  # x1, y1, x2, y2
                                                'conf': conf,
                                                'keypoints': kpts[0].cpu().numpy()  # [num_keypoints, 3]
                                            }
                        
                        # Draw overlay on frame
                        if best_detection:
                            detections += 1
                            frame_with_overlay = draw_overlay(frame, best_detection)
                        else:
                            frame_with_overlay = frame.copy()
                        
                        # Update web stream with processed frame
                        self.shared_web.update(frame_with_overlay)
                        
                        # Update job statistics
                        if 'local' in self.jobs:
                            job = self.jobs['local']
                            job['processed_frames'] = frame_count
                            job['fps'] = fps
                            job['detection_rate'] = (detections / frame_count) * 100
                            job['progress'] = (frame_count / total_frames) * 100
                        
                        # Small delay to control playback speed
                        time.sleep(1.0 / fps if fps > 0 else 0.033)  # Match video FPS
                    
                    cap.release()
                    print(f"✅ Local inference completed: {frame_count} frames, {detections} detections")
                    
                    # Mark job as completed
                    if 'local' in self.jobs:
                        self.jobs['local']['status'] = 'Completed successfully'
                    
                except Exception as e:
                    print(f"❌ Error in local inference: {e}")
                    if 'local' in self.jobs:
                        self.jobs['local']['status'] = f'Failed: {str(e)}'
            
            # Start inference in background thread
            inference_thread = threading.Thread(target=run_local_inference, daemon=True)
            inference_thread.start()
            
            # Store local job info
            self.jobs['local'] = {
                'thread': inference_thread,
                'config': config,
                'status': 'Running locally',
                'progress': 0.0,
                'fps': 0.0,
                'detection_rate': 0.0,
                'processed_frames': 0,
                'log_lines': [],
                'start_time': time.time(),
                'execution_mode': 'local'
            }
            
            return True
            
        except Exception as e:
            print(f"Error starting local pipeline: {e}")
            return False
    
    def _submit_hpc_job(self, config):
        """Submit SLURM job to GPU partition"""
        job_id = str(uuid.uuid4())[:8]
        
        # Create SLURM script
        slurm_script = self._create_slurm_script(config, job_id)
        script_path = Path(f"/tmp/pose_job_{job_id}.sbatch")
        
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        
        try:
            # Submit job
            result = subprocess.run(['sbatch', str(script_path)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract SLURM job ID from output
                slurm_job_id = result.stdout.strip().split()[-1]
                
                self.jobs[job_id] = {
                    'slurm_id': slurm_job_id,
                    'config': config,
                    'status': 'Submitted to SLURM',
                    'progress': 0.0,
                    'fps': 0.0,
                    'detection_rate': 0.0,
                    'processed_frames': 0,
                    'log_lines': [],
                    'start_time': time.time(),
                    'execution_mode': 'hpc'
                }
                
                return job_id
            else:
                print(f"Failed to submit job: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error submitting job: {e}")
            return None
    
    def _create_slurm_script(self, config, job_id):
        """Generate SLURM batch script for GPU job"""
        return f"""#!/bin/bash
#SBATCH --job-name=pose_pipeline_{job_id}
#SBATCH --partition={config['partition']}
#SBATCH --time={config['time_limit']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/tmp/pose_job_{job_id}.out
#SBATCH --error=/tmp/pose_job_{job_id}.err

# Load environment
source /scratch200/bareketd1/LizardPose/bin/activate

# Set environment variables
export VIDEO_PATH="{config['video_path']}"
export MODE="LABELS_ONLY"
export CONF_THRESH="{config['conf_thresh']}"
export IMG_SIZE="{config['img_size']}"
export OUTPUT_DIR="output/gpu_runs"
export WEB_PREVIEW="true"
export WEB_HOST="0.0.0.0"
export WEB_PORT="8765"

# Change to pipeline directory
cd /a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/pose-head

# Run pipeline
python pipeline/video_pose_pipeline.py

echo "Pipeline completed"
"""
    
    def _cancel_job(self, job_id):
        """Cancel running job - either local process or SLURM job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        try:
            if job.get('execution_mode') == 'local':
                # For local jobs, we can't cleanly kill threads, just mark as cancelled
                job['status'] = 'Cancelled'
                return True
            else:
                # Cancel SLURM job
                slurm_id = job['slurm_id']
                subprocess.run(['scancel', slurm_id], check=True)
                job['status'] = 'Cancelled'
                return True
        except subprocess.CalledProcessError:
            return False
        except Exception as e:
            print(f"Error cancelling job: {e}")
            return False
    
    def _get_job_status(self, job_id):
        """Get current job status and statistics"""
        if job_id not in self.jobs:
            return {'status': 'Job not found', 'progress': 0, 'fps': 0, 'detection_rate': 0, 'processed_frames': 0}
        
        job = self.jobs[job_id]
        
        # Handle local jobs differently from SLURM jobs
        if job.get('execution_mode') == 'local':
            # For local jobs, check thread status
            thread = job.get('thread')
            if thread and thread.is_alive():
                job['status'] = 'Running locally'
            elif thread and not thread.is_alive():
                # Thread finished, check if status was updated
                if job['status'] == 'Running locally':
                    job['status'] = 'Completed successfully'
            
        else:
            # Handle SLURM jobs
            try:
                result = subprocess.run(['squeue', '-j', job['slurm_id'], '-h', '-o', '%T'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    slurm_status = result.stdout.strip()
                    job['status'] = f"SLURM: {slurm_status}"
                else:
                    # Job might be completed, check logs
                    job['status'] = "Checking completion..."
                    
            except Exception:
                job['status'] = "Status unknown"
        
        # Read job output for progress updates
        if job.get('execution_mode') != 'local':
            # For SLURM jobs, read log file
            log_file = Path(f"/tmp/pose_job_{job_id}.out")
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        
                    # Parse progress from tqdm output or other metrics
                    for line in lines[-10:]:  # Check last 10 lines
                        if '%|' in line:  # tqdm progress bar
                            # Extract progress percentage
                            try:
                                progress_part = line.split('%|')[0].split()[-1]
                                job['progress'] = float(progress_part.replace('%', ''))
                            except:
                                pass
                        
                        if 'fps' in line.lower():
                            # Extract FPS
                            try:
                                words = line.split()
                                for i, word in enumerate(words):
                                    if 'fps' in word.lower() and i > 0:
                                        job['fps'] = float(words[i-1].replace('(', '').replace(',', ''))
                                        break
                            except:
                                pass
                        
                        if 'detection rate' in line.lower():
                            # Extract detection rate
                            try:
                                parts = line.split('detection rate')[0].split()
                                job['detection_rate'] = float(parts[-1].replace('(', '').replace('%', ''))
                            except:
                                pass
                    
                    # Add new log lines
                    current_lines = len(job.get('log_lines', []))
                    if len(lines) > current_lines:
                        job['log_lines'] = lines
                        
                except Exception as e:
                    print(f"Error reading log file: {e}")
        
        return {
            'status': job['status'],
            'progress': job.get('progress', 0),
            'fps': job.get('fps', 0),
            'detection_rate': job.get('detection_rate', 0),
            'processed_frames': job.get('processed_frames', 0),
            'log_lines': job.get('log_lines', [])[-5:] if job.get('log_lines') else []  # Last 5 lines
        }
    
    def update(self, bgr):
        """Update video stream (called from pipeline) - delegate to shared interface"""
        self.shared_web.update(bgr)
    
    def _gen_video(self):
        """Generate video stream using shared interface"""
        while True:
            frame = self.shared_web.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.03)
    
    def start(self):
        """Start the web server"""
        if self._th and self._th.is_alive():
            return
            
        def run():
            self.app.run(host=self.host, port=self.port, debug=False, 
                        use_reloader=False, threaded=True)
        
        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()
        print(f"HPC Web Interface started at http://{self.host}:{self.port}/")

if __name__ == "__main__":
    # Standalone mode for testing
    interface = HPCWebInterface()
    interface.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")