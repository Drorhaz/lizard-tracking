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
from flask import Flask, Response, render_template, request, jsonify
from pipeline.shared_web_interface import SharedWebInterface

class HPCWebInterface:
    """Web interface for submitting and monitoring HPC GPU jobs"""
    
    _instance = None  # Class variable to store the running instance
    
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        # Configure Flask app with proper template and static directories
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self._lock = threading.Lock()
        self._th = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._stop_flags: Dict[str, threading.Event] = {}  # Stop flags for jobs
        
        # Use shared web interface for video streaming
        self.shared_web = SharedWebInterface.get_instance()
        
        # Store instance for pipeline connection
        HPCWebInterface._instance = self
        
        self.setup_routes()
        
    def _get_project_root(self):
        """Find the lizard-tracking project root directory"""
        # Start from current file location and walk up to find project root
        current_path = Path(__file__).parent.absolute()
        
        # Look for project indicators (pyproject.toml, requirements.txt, etc.)
        while current_path.parent != current_path:  # Not at filesystem root
            if any((current_path / indicator).exists() for indicator in 
                   ['pyproject.toml', 'requirements.txt', '.git', 'README.md']):
                # Double-check this looks like our project
                if (current_path / 'pose-head').exists() or current_path.name == 'lizard-tracking':
                    return current_path
            current_path = current_path.parent
        
        # Fallback: assume we're in pose-head/pipeline and go up two levels
        fallback = Path(__file__).parent.parent.parent
        print(f"⚠️  Using fallback project root: {fallback}")
        return fallback
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            # Get videos and pass them directly to template
            videos = self._get_available_videos()
            return render_template('index.html', videos=videos)
            
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
            
            if execution_mode in ['local', 'local_gpu']:
                # Local CPU or GPU mode: run locally and start video immediately
                success = self._start_local_pipeline(config)
                if success:
                    return jsonify({'success': True, 'job_id': execution_mode, 'immediate_start': True})
                else:
                    return jsonify({'success': False, 'error': f'Failed to start {execution_mode} pipeline'})
            else:
                # HPC mode: submit SLURM job and wait for it to start
                job_id = self._submit_hpc_job(config)
                if job_id:
                    return jsonify({'success': True, 'job_id': job_id, 'immediate_start': False})
                else:
                    return jsonify({'success': False, 'error': 'Failed to submit GPU job'})
                
        @self.app.route('/api/stop', methods=['POST'])
        def stop_pipeline():
            job_id = request.json.get('job_id')
            success = self._stop_pipeline(job_id)
            return jsonify({'success': success})
            
        @self.app.route('/api/status/<job_id>')
        def get_status(job_id):
            status = self._get_job_status(job_id)
            return jsonify(status)
            
        @self.app.route('/api/check_saved_labels', methods=['POST'])
        def check_saved_labels():
            """Check if saved labels exist for a video"""
            video_name = request.json.get('video_name')
            project_root = self._get_project_root()
            output_base = project_root / "output" / "detections"
            
            found = False
            if output_base.exists():
                # Search for directories that start with the video name
                for run_dir in output_base.iterdir():
                    if run_dir.is_dir() and run_dir.name.startswith(video_name):
                        labeled_frames_dir = run_dir / "labeled_frames"
                        if labeled_frames_dir.exists():
                            frame_files = list(labeled_frames_dir.glob("frame*.jpg"))
                            if frame_files:
                                found = True
                                break
            
            return jsonify({'found': found})
            
        @self.app.route('/api/completed_runs')
        def list_completed_runs():
            """List available completed runs with labeled frames"""
            runs = []
            project_root = self._get_project_root()
            output_base = project_root / "output" / "detections"
            if output_base.exists():
                run_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                
                for run_dir in run_dirs[:10]:  # Show last 10 runs
                    labeled_frames_dir = run_dir / "labeled_frames"
                    if labeled_frames_dir.exists():
                        frame_files = list(labeled_frames_dir.glob("frame*.jpg"))
                        if frame_files:
                            runs.append({
                                'name': run_dir.name,
                                'path': str(run_dir),
                                'frame_count': len(frame_files),
                                'created': run_dir.stat().st_mtime
                            })
            return jsonify(runs)
    
    def _get_available_videos(self):
        """Scan for available video files"""
        videos = []
        seen_paths = set()  # Track unique video paths to avoid duplicates
        print("🔍 Starting video scan...")
        
        # Get project root and build relative paths
        project_root = self._get_project_root()
        print(f"📁 Project root: {project_root}")
        
        # Look in common video directories - use relative paths from project root
        video_dirs = [
            project_root / "videos",  # Main project videos
            project_root / "pose-head" / "videos",  # Pose-head videos directory
            project_root / "scripts",  # Scripts folder
            project_root / "dataset" / "videos",  # Dataset videos
            project_root / "data" / "videos",  # Data videos
            Path("videos"),  # Local relative videos
            Path("../videos"),  # Parent directory videos
            Path("../../videos"),  # Grandparent directory videos
            Path("scripts"),  # Local scripts folder
            Path("../scripts"),  # Parent scripts
            Path("../../scripts"),  # Grandparent scripts
        ]
        
        for video_dir in video_dirs:
            abs_path = video_dir.resolve() if video_dir.exists() else video_dir
            print(f"� Checking: {video_dir} -> {abs_path}")
            
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
    
    def _start_local_pipeline(self, config):
        """Start pipeline locally - handles both live detection and offline label loading"""
        try:
            # Get execution mode to determine device
            execution_mode = config.get('execution_mode', 'local')
            
            # Create stop flag for this job (use execution_mode for unique ID)
            stop_flag = threading.Event()
            self._stop_flags[execution_mode] = stop_flag
            
            detection_mode = config.get('detection_mode', 'live')
            
            if detection_mode == 'offline':
                # Offline mode: Load and display saved labels
                return self._start_offline_playback(config, stop_flag, execution_mode)
            else:
                # Live mode: Run inference
                return self._start_live_inference(config, stop_flag, execution_mode)
                
        except Exception as e:
            print(f"❌ Error starting local pipeline: {e}")
            return False
    
    def _start_offline_playback(self, config, stop_flag, execution_mode='local'):
        """Play back saved labels from previous runs"""
        def run_offline_playback():
            try:
                video_name = Path(config['video_path']).stem
                print(f"📁 Loading saved labels for video: {video_name}")
                
                # Find the most recent run with labeled frames for this video
                project_root = self._get_project_root()
                output_base = project_root / "output" / "detections"
                labeled_frames_dir = None
                
                if output_base.exists():
                    # Find directories starting with video name
                    matching_dirs = []
                    for run_dir in output_base.iterdir():
                        if run_dir.is_dir() and run_dir.name.startswith(video_name):
                            frames_dir = run_dir / "labeled_frames"
                            if frames_dir.exists():
                                frame_files = list(frames_dir.glob("frame*.jpg"))
                                if frame_files:
                                    matching_dirs.append((run_dir.stat().st_mtime, frames_dir))
                    
                    if matching_dirs:
                        # Use the most recent one
                        matching_dirs.sort(reverse=True)
                        labeled_frames_dir = matching_dirs[0][1]
                        print(f"✅ Found labeled frames in: {labeled_frames_dir}")
                
                if not labeled_frames_dir:
                    print(f"❌ No saved labels found for video: {video_name}")
                    return
                
                # Get video properties for timing
                cap = cv2.VideoCapture(config['video_path'])
                video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
                cap.release()
                
                # Get all frame files, sorted by frame number
                frame_files = []
                for frame_file in labeled_frames_dir.glob("frame*.jpg"):
                    try:
                        # Extract frame number from filename like "frame00000010.jpg"
                        frame_num_str = frame_file.stem.replace('frame', '')
                        frame_num = int(frame_num_str)
                        frame_files.append((frame_num, frame_file))
                    except ValueError:
                        continue
                
                # Sort by actual frame number
                frame_files.sort(key=lambda x: x[0])
                print(f"📹 Playing {len(frame_files)} saved frames at {video_fps:.1f} FPS")
                
                frame_count = 0
                total_detections = 0
                
                for frame_num, frame_file in frame_files:
                    if stop_flag.is_set():
                        break
                    
                    # Load and display frame
                    frame = cv2.imread(str(frame_file))
                    if frame is not None:
                        self.shared_web.update(frame)
                        frame_count += 1
                        total_detections += 1  # Assume saved frames have detections
                        
                        # Update statistics
                        with self._lock:
                            if 'local' in self.jobs:
                                job = self.jobs['local']
                                job['processed_frames'] = frame_count
                                job['fps'] = video_fps
                                job['detection_rate'] = 100.0  # All saved frames have detections
                                job['progress'] = (frame_count / len(frame_files)) * 100
                                job['total_detections'] = total_detections
                    
                    # Control playback speed - much faster than original since these are subsampled frames
                    time.sleep(1.0 / min(video_fps * 2, 60))  # 2x speed, max 60 FPS
                
                print(f"✅ Offline playback completed: {frame_count} frames displayed")
                
                # Mark job as completed
                with self._lock:
                    if 'local' in self.jobs:
                        self.jobs['local']['status'] = 'Offline playback completed'
                        
            except Exception as e:
                print(f"❌ Error in offline playback: {e}")
                with self._lock:
                    if 'local' in self.jobs:
                        self.jobs['local']['status'] = f'Failed: {str(e)}'
        
        # Start playback in background thread
        playback_thread = threading.Thread(target=run_offline_playback, daemon=True)
        playback_thread.start()
        
        # Store job info
        self.jobs['local'] = {
            'thread': playback_thread,
            'config': config,
            'status': 'Loading saved labels...',
            'progress': 0.0,
            'fps': 0.0,
            'detection_rate': 0.0,
            'processed_frames': 0,
            'total_detections': 0,
            'log_lines': [],
            'start_time': time.time(),
            'execution_mode': 'local',
            'detection_mode': 'offline'
        }
        
        return True
    
    def _start_live_inference(self, config, stop_flag, execution_mode='local'):
        """Run live inference with smooth video playback"""
        def run_live_inference():
            try:
                print(f"▶ Starting live inference for {config['video_path']} on {execution_mode}")
                
                # Initialize video capture
                cap = cv2.VideoCapture(config['video_path'])
                if not cap.isOpened():
                    print(f"❌ Could not open video: {config['video_path']}")
                    return
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                # Use original video FPS for smooth playback
                target_fps = video_fps if video_fps > 0 else 30.0
                frame_time = 1.0 / target_fps
                
                print(f"📹 Video: {total_frames} frames @ {video_fps:.1f} FPS -> playing at {target_fps:.1f} FPS")
                
                # Initialize YOLO model only if needed
                model = None
                if config.get('detection_mode', 'live') == 'live':
                    try:
                        print(f"🤖 Loading YOLO model...")
                        from ultralytics import YOLO
                        
                        # Use model path from config or find default in project root
                        if 'model_path' in config:
                            model_path = config['model_path']
                        else:
                            project_root = self._get_project_root()
                            # Look for common model files in project root
                            for model_name in ['yolo11s-pose.pt', 'yolo11n-pose.pt', 'best.pt']:
                                model_path = project_root / model_name
                                if model_path.exists():
                                    break
                            else:
                                # Fallback to relative path
                                model_path = 'yolo11s-pose.pt'
                        
                        model = YOLO(str(model_path))
                        
                        # Set device based on execution mode
                        if execution_mode == 'local_gpu':
                            print("🔥 Using local GPU for inference")
                            device = 'cuda:0' if hasattr(model, 'device') else 0
                        else:
                            print("💻 Using CPU for inference")
                            device = 'cpu'
                        
                        # Set model to the appropriate device
                        model.to(device)
                        print(f"✅ Model loaded on {device}")
                        
                    except Exception as e:
                        print(f"❌ Failed to load model: {e}")
                        cap.release()
                        return
                model = None
                if config.get('detection_mode', 'live') == 'live':
                    import sys
                    # Add project paths to Python path for imports
                    project_root = self._get_project_root()
                    pose_head_path = project_root / "pose-head"
                    if pose_head_path.exists():
                        sys.path.append(str(pose_head_path))
                    from pipeline.video_pose_pipeline import YOLOPoseModel, draw_overlay
                    
                    model_dir = Path("../output/models/head_pose")
                    model_paths = list(model_dir.glob("**/best*.pt")) + list(model_dir.glob("**/*.pt"))
                    if not model_paths:
                        print(f"❌ No model found in {model_dir}")
                        return
                    
                    model_path = model_paths[0]
                    print(f"🤖 Loading model: {model_path}")
                    
                    model = YOLOPoseModel(
                        model_path,
                        imgsz=config.get('img_size', 640),
                        conf=config.get('conf_thresh', 0.1),
                        iou=0.5
                    )
                
                frame_count = 0
                detections = 0
                start_time = time.time()
                
                # Process video frames with precise timing
                while not stop_flag.is_set():
                    frame_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        print("📹 End of video reached")
                        break
                    
                    frame_count += 1
                    
                    # Always display the frame, regardless of detection
                    display_frame = frame.copy()
                    
                    # Run inference if in live mode
                    if model is not None:
                        poses = model.predict(frame)
                        
                        # Find best pose
                        best_pose = None
                        if poses:
                            valid_poses = [p for p in poses if p.conf > config.get('conf_thresh', 0.1)]
                            if valid_poses:
                                best_pose = max(valid_poses, key=lambda p: p.conf)
                                detections += 1
                        
                        # Draw overlay if detection found
                        if best_pose:
                            display_frame = draw_overlay(frame, best_pose)
                    
                    # Always update the stream - smooth video playback
                    self.shared_web.update(display_frame)
                    
                    # Update statistics with thread safety
                    with self._lock:
                        if 'local' in self.jobs:
                            job = self.jobs['local']
                            job['processed_frames'] = frame_count
                            job['fps'] = target_fps
                            job['detection_rate'] = (detections / frame_count) * 100 if frame_count > 0 else 0
                            job['progress'] = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                            job['total_detections'] = detections
                            
                            # Add timing information
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                actual_fps = frame_count / elapsed
                                job['log_lines'].append(f"Frame {frame_count}/{total_frames} - {actual_fps:.1f} FPS - {detections} detections")
                                # Keep only last 10 log lines
                                job['log_lines'] = job['log_lines'][-10:]
                    
                    # Precise frame timing for smooth playback
                    frame_end = time.time()
                    processing_time = frame_end - frame_start
                    sleep_time = max(0, frame_time - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                cap.release()
                
                if stop_flag.is_set():
                    print(f"🛑 Live inference stopped: {frame_count} frames, {detections} detections")
                else:
                    print(f"✅ Live inference completed: {frame_count} frames, {detections} detections")
                
                # Mark job as completed
                with self._lock:
                    if 'local' in self.jobs:
                        if stop_flag.is_set():
                            self.jobs['local']['status'] = 'Stopped by user'
                        else:
                            self.jobs['local']['status'] = 'Completed successfully'
                        
            except Exception as e:
                print(f"❌ Error in live inference: {e}")
                with self._lock:
                    if 'local' in self.jobs:
                        self.jobs['local']['status'] = f'Failed: {str(e)}'
        
        # Start inference in background thread
        inference_thread = threading.Thread(target=run_live_inference, daemon=True)
        inference_thread.start()
        
        # Store job info
        self.jobs['local'] = {
            'thread': inference_thread,
            'config': config,
            'status': 'Running live inference...',
            'progress': 0.0,
            'fps': 0.0,
            'detection_rate': 0.0,
            'processed_frames': 0,
            'total_detections': 0,
            'log_lines': [],
            'start_time': time.time(),
            'execution_mode': 'local',
            'detection_mode': 'live'
        }
        
        return True
    
    def _submit_hpc_job(self, config):
        """Submit SLURM job using existing working submit_labels_gpu.sh script"""
        try:
            # Set environment variables for the job
            env = os.environ.copy()
            
            # Base configuration
            env.update({
                'VIDEO_PATH': config['video_path'],
                'CONF_THRESH': str(config.get('conf_thresh', 0.1)),
                'IMG_SIZE': str(config.get('img_size', 640)),
                'WEB_PREVIEW': 'true',  # Enable web streaming
                'WEB_HOST': '0.0.0.0',
                'WEB_PORT': '8765',
                'PREVIEW': 'false',  # Disable local preview
                'HPC_MODE': 'true',   # Signal that we're running on HPC
                'SAVE_LABELS': str(config.get('save_labels', True)).lower()
            })
            
            # Set mode based on detection type
            detection_mode = config.get('detection_mode', 'live')
            if detection_mode == 'offline':
                env['MODE'] = 'LOAD_LABELS'  # New mode for loading saved labels
                print(f"🏃 HPC mode: Loading saved labels for video")
            else:
                env['MODE'] = 'INFER_LIVE'  # Live inference
                print(f"🤖 HPC mode: Live inference with GPU")
            
            # Add timing parameters for smooth playback
            env['FPS_LIMIT'] = '30'  # Standard FPS for display
            env['SMOOTH_PLAYBACK'] = 'true'  # Enable smooth frame timing
            
            # Change to the correct directory and run the existing script
            script_path = Path("hpc/submit_labels_gpu.sh").resolve()
            if not script_path.exists():
                script_path = Path("../hpc/submit_labels_gpu.sh").resolve()
            
            if not script_path.exists():
                print(f"❌ Could not find submit_labels_gpu.sh script in hpc/ directory")
                return None
            
            print(f"🚀 Using existing SLURM script: {script_path}")
            print(f"📊 Config: {detection_mode} mode, conf={config.get('conf_thresh', 0.1)}, size={config.get('img_size', 640)}")
            
            # Submit job using the existing working script
            result = subprocess.run(['bash', str(script_path)], 
                                  capture_output=True, text=True, env=env,
                                  cwd=script_path.parent.parent)  # Run from pose-head directory
            
            if result.returncode == 0:
                # Extract SLURM job ID from output - look for "Submitted batch job XXXXX"
                output_lines = result.stdout.strip().split('\n')
                slurm_job_id = None
                
                for line in output_lines:
                    if 'Submitted batch job' in line:
                        slurm_job_id = line.split()[-1]
                        break
                
                if slurm_job_id:
                    print(f"✅ SLURM job submitted: {slurm_job_id}")
                    
                    # Use the SLURM job ID as our job ID (not UUID)
                    job_id = slurm_job_id
                    
                    self.jobs[job_id] = {
                        'slurm_id': slurm_job_id,
                        'config': config,
                        'status': 'Submitted to SLURM',
                        'progress': 0.0,
                        'fps': 30.0,  # Set expected FPS
                        'detection_rate': 0.0,
                        'processed_frames': 0,
                        'total_detections': 0,
                        'log_lines': [],
                        'start_time': time.time(),
                        'execution_mode': 'hpc',
                        'stream_delay': 10  # 10 second buffer for processing
                    }
                    
                    # Start a delayed video stream checker
                    self._start_hpc_stream_monitor(job_id, slurm_job_id)
                    
                    return job_id
                else:
                    print(f"❌ Could not extract SLURM job ID from output: {result.stdout}")
                    return None
            else:
                print(f"❌ Failed to submit job: {result.stderr}")
                print(f"❌ stdout: {result.stdout}")
                return None
                
        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            return None
    
    def _start_hpc_stream_monitor(self, job_id, slurm_job_id):
        """Start monitoring HPC job and begin video streaming after delay"""
        def monitor_hpc_job():
            try:
                print(f"🕒 Starting HPC stream monitor for job {slurm_job_id} (10s delay)")
                
                # Wait for the job to start processing (10 second buffer)
                time.sleep(10)
                
                # Update status
                with self._lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]['status'] = 'GPU processing started - buffering...'
                
                # Monitor for output and start streaming
                # For now, we'll just update the status and let the existing status monitoring handle the rest
                print(f"✅ HPC job {slurm_job_id} should be processing, ready for streaming")
                
            except Exception as e:
                print(f"❌ Error in HPC stream monitor: {e}")
                with self._lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]['status'] = f'Monitor error: {str(e)}'
        
        # Start monitor in background thread
        monitor_thread = threading.Thread(target=monitor_hpc_job, daemon=True)
        monitor_thread.start()
    

    def _stop_pipeline(self, job_id):
        """Stop pipeline - handles both local processes and SLURM jobs properly"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        try:
            if job.get('execution_mode') in ['local', 'local_gpu']:
                print(f"🛑 Stopping {job.get('execution_mode')} pipeline {job_id}...")
                
                # Set stop flag to break the processing loop
                if job_id in self._stop_flags:
                    self._stop_flags[job_id].set()
                    print(f"🛑 Stop signal sent to {job.get('execution_mode')} pipeline {job_id}")
                
                # Clear the video stream immediately
                self.shared_web.clear_frame()
                
                # Update job status immediately
                with self._lock:
                    job['status'] = 'Stopped by user'
                    job['progress'] = 0.0
                    job['fps'] = 0.0
                    job['detection_rate'] = 0.0
                    job['processed_frames'] = 0
                    job['total_detections'] = 0
                
                # Clean up after a short delay
                def cleanup_job():
                    time.sleep(1)
                    with self._lock:
                        if job_id in self.jobs:
                            del self.jobs[job_id]
                        if job_id in self._stop_flags:
                            del self._stop_flags[job_id]
                    print(f"✅ Local pipeline {job_id} cleaned up")
                
                threading.Thread(target=cleanup_job, daemon=True).start()
                return True
            else:
                # Cancel SLURM job - job_id is now the SLURM job ID directly
                slurm_id = job_id  # Since we're using SLURM ID as our job ID
                result = subprocess.run(['scancel', slurm_id], capture_output=True, text=True)
                if result.returncode == 0:
                    job['status'] = 'Cancelled'
                    print(f"✅ SLURM job {slurm_id} cancelled")
                    return True
                else:
                    print(f"❌ Failed to cancel SLURM job {slurm_id}: {result.stderr}")
                    return False
        except Exception as e:
            print(f"❌ Error stopping pipeline: {e}")
            return False
    
    def _get_job_status(self, job_id):
        """Get current job status and statistics"""
        if job_id not in self.jobs:
            return {'status': 'Job not found', 'progress': 0, 'fps': 0, 'detection_rate': 0, 'processed_frames': 0}
        
        # Thread-safe access to job data
        with self._lock:
            job = self.jobs[job_id].copy()  # Make a copy to avoid race conditions
        
        # Handle local jobs differently from SLURM jobs
        if job.get('execution_mode') in ['local', 'local_gpu']:
            # For local jobs, check thread status
            thread = job.get('thread')
            if thread and thread.is_alive():
                if job_id in self._stop_flags and self._stop_flags[job_id].is_set():
                    job['status'] = 'Stopping...'
                else:
                    execution_mode = job.get('execution_mode', 'local')
                    device_label = 'GPU' if execution_mode == 'local_gpu' else 'CPU'
                    job['status'] = f'Running locally on {device_label}'
            elif thread and not thread.is_alive():
                # Thread finished, check if status was updated
                if 'Running locally' in job['status']:
                    job['status'] = 'Completed successfully'
            
        else:
            # Handle SLURM jobs - job_id is now the SLURM job ID directly
            slurm_job_id = job_id
            try:
                result = subprocess.run(['squeue', '-j', slurm_job_id, '-h', '-o', '%T'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    slurm_status = result.stdout.strip()
                    job['status'] = f"SLURM: {slurm_status}"
                else:
                    # Job might be completed, check logs
                    job['status'] = "Checking completion..."
                    
            except Exception:
                job['status'] = "Status unknown"
        
        # Read job output for progress updates (only for SLURM jobs)
        if job.get('execution_mode') not in ['local', 'local_gpu']:
            # For SLURM jobs, try multiple log file locations
            user = os.environ.get('USER', os.environ.get('LOGNAME', 'user'))
            slurm_job_id = job_id  # job_id is now the SLURM job ID
            
            # Try multiple possible log locations
            possible_log_paths = [
                Path(f"/scratch200/{user}/logs/ph/lb/pose-labels-{slurm_job_id}.out"),  # HPC location
                Path(f"/tmp/pose_job_{job_id}.out"),  # Temp location
                Path.home() / "logs" / f"pose-labels-{slurm_job_id}.out",  # Home logs
                Path(f"pose_job_{job_id}.out"),  # Current directory
            ]
            
            log_file = None
            for possible_path in possible_log_paths:
                if possible_path.exists():
                    log_file = possible_path
                    break
            
            if log_file:
                print(f"🔍 Reading log file: {log_file}")  # Debug info
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        
                    print(f"📄 Found log file with {len(lines)} lines")  # Debug info
                    
                    # Parse progress from output
                    for line in lines[-20:]:  # Check last 20 lines for more context
                        line = line.strip()
                        
                        # Look for frame processing info
                        if 'frame' in line.lower() and 'fps' in line.lower():
                            try:
                                # Extract FPS and frame info
                                words = line.split()
                                for i, word in enumerate(words):
                                    if 'fps' in word.lower() and i > 0:
                                        job['fps'] = 30.0  # Force 30 FPS as requested
                                        break
                            except:
                                pass
                        
                        # Look for detection information
                        if 'detection' in line.lower() and '%' in line:
                            try:
                                # Extract detection rate
                                if 'detection rate' in line.lower():
                                    parts = line.split('%')[0].split()
                                    job['detection_rate'] = float(parts[-1])
                            except:
                                pass
                        
                        # Look for progress information
                        if '%|' in line or 'progress' in line.lower():
                            try:
                                # Extract progress percentage
                                if '%|' in line:
                                    progress_part = line.split('%|')[0].split()[-1]
                                    job['progress'] = float(progress_part.replace('%', ''))
                                elif '%' in line and 'complete' in line.lower():
                                    progress_part = line.split('%')[0].split()[-1]
                                    job['progress'] = float(progress_part)
                            except:
                                pass
                    
                    # Add new log lines
                    current_lines = len(job.get('log_lines', []))
                    if len(lines) > current_lines:
                        job['log_lines'] = lines
                        
                except Exception as e:
                    print(f"Error reading log file {log_file}: {e}")
            else:
                print(f"⚠️ Log file not found: {log_file}")  # Debug info
        
        return {
            'status': job['status'],
            'progress': job.get('progress', 0),
            'fps': job.get('fps', 0),
            'detection_rate': job.get('detection_rate', 0),
            'processed_frames': job.get('processed_frames', 0),
            'total_detections': job.get('total_detections', 0),
            'log_lines': job.get('log_lines', [])[-5:] if job.get('log_lines') else []  # Last 5 lines
        }
    
    def update(self, bgr):
        """Update video stream (called from pipeline) - delegate to shared interface"""
        self.shared_web.update(bgr)
    
    def _gen_video(self):
        """Generate video stream using shared interface or HPC saved frames"""
        last_frame_name = None
        start_time = time.time()
        hpc_frames_found = False
        current_frame_data = None
        
        while True:
            frame = None
            
            # Check if we have any HPC job (active or completed) and look for saved labeled frames
            if self.jobs:
                for job_id, job in self.jobs.items():
                    # Look for frames from any HPC job, not just running ones
                    if job.get('execution_mode') == 'hpc':
                        # Look for the latest output directory for this job
                        # Extract stats and create output directory structure
                        project_root = self._get_project_root()
                        output_base = project_root / "output" / "detections"
                        if output_base.exists():
                            # Find the most recent run directory
                            run_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()], 
                                            key=lambda x: x.stat().st_mtime, reverse=True)
                            
                            for run_dir in run_dirs:  # Try multiple recent runs
                                labeled_frames_dir = run_dir / "labeled_frames"
                                
                                if labeled_frames_dir.exists():
                                    # Get all frame files, sorted by frame number (not alphabetically)
                                    frame_files = []
                                    for frame_file in labeled_frames_dir.glob("frame*.jpg"):
                                        try:
                                            # Extract frame number from filename like "frame00000010.jpg"
                                            frame_num_str = frame_file.stem.replace('frame', '')
                                            frame_num = int(frame_num_str)
                                            frame_files.append((frame_num, frame_file))
                                        except ValueError:
                                            continue
                                    
                                    # Sort by actual frame number
                                    frame_files.sort(key=lambda x: x[0])
                                    
                                    if frame_files:
                                        hpc_frames_found = True
                                        
                                        # Calculate which frame to show based on elapsed time
                                        # Assume original video was ~10 FPS (since frames are saved every 10th)
                                        # And we want to play back at ~2 FPS for comfortable viewing
                                        current_time = time.time()
                                        elapsed = current_time - start_time
                                        
                                        # Show each saved frame for 0.5 seconds (2 FPS)
                                        frame_index = int(elapsed * 2) % len(frame_files)
                                        frame_num, current_frame_file = frame_files[frame_index]
                                        
                                        # Only load new frame if it's different
                                        if current_frame_file.name != last_frame_name:
                                            try:
                                                with open(current_frame_file, 'rb') as f:
                                                    current_frame_data = f.read()
                                                last_frame_name = current_frame_file.name
                                                print(f"📹 [HPC] Playing frame {frame_index+1}/{len(frame_files)}: {current_frame_file.name} (orig frame {frame_num})")
                                            except Exception as e:
                                                print(f"⚠️ Error reading saved frame {current_frame_file}: {e}")
                                                continue
                                        
                                        # Use the current frame data
                                        frame = current_frame_data
                                        break  # Found frames, use them
                                
                                if frame:
                                    break  # Got a frame, stop searching
                            
                            if frame:
                                break  # Got a frame from this job, stop checking other jobs
            
            # Fallback to shared interface frame (local mode) only if no HPC frames found
            if frame is None and not hpc_frames_found:
                frame = self.shared_web.get_frame()
            
            if frame is None:
                time.sleep(0.1)  # Sleep if no frame available
                continue
                
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.1)  # 10 FPS max streaming rate
    
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