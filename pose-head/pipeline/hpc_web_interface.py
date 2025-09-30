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
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
            
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
            success = self._stop_pipeline(job_id)
            return jsonify({'success': success})
            
        @self.app.route('/api/status/<job_id>')
        def get_status(job_id):
            status = self._get_job_status(job_id)
            return jsonify(status)
            
        @self.app.route('/api/completed_runs')
        def list_completed_runs():
            """List available completed runs with labeled frames"""
            runs = []
            output_base = Path("/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/output/detections")
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
            from pipeline.video_pose_pipeline import YOLOPoseModel, draw_overlay
            
            # Create stop flag for this job
            stop_flag = threading.Event()
            self._stop_flags['local'] = stop_flag
            
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
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = 30.0  # Force 30 FPS as requested
                    print(f"📹 Video: {total_frames} frames @ {video_fps:.1f} FPS (playing at {fps:.1f} FPS)")
                    
                    # Initialize YOLO model
                    model_dir = Path("../output/models/head_pose")
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
                    while not stop_flag.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            print("📹 End of video reached")
                            break
                        
                        frame_count += 1
                        
                        # Run inference
                        poses = model.predict(frame)
                        
                        # Find best pose with more lenient criteria
                        best_pose = None
                        if poses:
                            # Debug: show all pose confidences
                            if frame_count % 50 == 0:
                                conf_list = [f"{p.conf:.3f}" for p in poses]
                                print(f"🔍 Frame {frame_count}: Found {len(poses)} poses with confidences: {conf_list}")
                            
                            # Accept any pose above a lower threshold
                            valid_poses = [p for p in poses if p.conf > 0.1]  # Even lower threshold
                            if valid_poses:
                                best_pose = max(valid_poses, key=lambda p: p.conf)
                                
                                # Additional debug for good detections
                                if frame_count % 100 == 0:
                                    print(f"✅ Frame {frame_count}: Best pose conf: {best_pose.conf:.3f}, threshold: {config['conf_thresh']}")
                        
                        # Draw overlay on frame
                        if best_pose:
                            detections += 1
                            frame_with_overlay = draw_overlay(frame, best_pose)
                        else:
                            frame_with_overlay = frame.copy()
                            # Debug info for no detections
                            if frame_count % 100 == 0:
                                print(f"⚠️ Frame {frame_count}: No poses detected above 0.1 confidence")
                        
                        # Update web stream with processed frame
                        self.shared_web.update(frame_with_overlay)
                        
                        # Update job statistics WITH thread safety
                        with self._lock:
                            if 'local' in self.jobs:
                                job = self.jobs['local']
                                job['processed_frames'] = frame_count
                                job['fps'] = fps
                                job['detection_rate'] = (detections / frame_count) * 100 if frame_count > 0 else 0
                                job['progress'] = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                                job['total_detections'] = detections
                                
                                # Debug print every 50 frames
                                if frame_count % 50 == 0:
                                    print(f"📊 Stats: {frame_count}/{total_frames} frames, {detections} detections ({job['detection_rate']:.1f}%)")
                        
                        # Small delay to control playback speed
                        time.sleep(1.0 / fps if fps > 0 else 0.033)  # Match video FPS
                    
                    cap.release()
                    
                    if stop_flag.is_set():
                        print(f"🛑 Local inference stopped by user: {frame_count} frames, {detections} detections")
                    else:
                        print(f"✅ Local inference completed: {frame_count} frames, {detections} detections")
                    
                    # Mark job as completed
                    with self._lock:
                        if 'local' in self.jobs:
                            if stop_flag.is_set():
                                self.jobs['local']['status'] = 'Stopped by user'
                            else:
                                self.jobs['local']['status'] = 'Completed successfully'
                    
                except Exception as e:
                    print(f"❌ Error in local inference: {e}")
                    with self._lock:
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
                'total_detections': 0,
                'log_lines': [],
                'start_time': time.time(),
                'execution_mode': 'local'
            }
            
            return True
            
        except Exception as e:
            print(f"Error starting local pipeline: {e}")
            return False
    
    def _submit_hpc_job(self, config):
        """Submit SLURM job using existing working submit_labels_gpu.sh script"""
        try:
            # Set environment variables for the job
            env = os.environ.copy()
            env.update({
                'VIDEO_PATH': config['video_path'],
                'MODE': 'INFER_LIVE',  # Use INFER_LIVE for streaming support
                'CONF_THRESH': str(config['conf_thresh']),
                'IMG_SIZE': str(config['img_size']),
                'WEB_PREVIEW': 'true',  # Enable web streaming
                'WEB_HOST': '0.0.0.0',
                'WEB_PORT': '8765',
                'PREVIEW': 'false',  # Disable local preview
                'HPC_MODE': 'true',   # Signal that we're running on HPC
                'FPS_LIMIT': '30'     # Force 30 FPS
            })
            
            # Change to the correct directory and run the existing script
            script_path = Path("../hpc/submit_labels_gpu.sh").resolve()
            if not script_path.exists():
                script_path = Path("hpc/submit_labels_gpu.sh").resolve()
            
            if not script_path.exists():
                print(f"❌ Could not find submit_labels_gpu.sh script")
                return None
            
            print(f"🚀 Using existing SLURM script: {script_path}")
            
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
            if job.get('execution_mode') == 'local':
                print(f"🛑 Stopping local pipeline {job_id}...")
                
                # Set stop flag to break the processing loop
                if job_id in self._stop_flags:
                    self._stop_flags[job_id].set()
                    print(f"🛑 Stop signal sent to local pipeline {job_id}")
                
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
        if job.get('execution_mode') == 'local':
            # For local jobs, check thread status
            thread = job.get('thread')
            if thread and thread.is_alive():
                if job_id in self._stop_flags and self._stop_flags[job_id].is_set():
                    job['status'] = 'Stopping...'
                else:
                    job['status'] = 'Running locally'
            elif thread and not thread.is_alive():
                # Thread finished, check if status was updated
                if job['status'] == 'Running locally':
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
        if job.get('execution_mode') != 'local':
            # For SLURM jobs, read log file from the correct location
            user = os.environ.get('USER', 'bareketd1')
            slurm_job_id = job_id  # job_id is now the SLURM job ID
            log_file = Path(f"/scratch200/{user}/logs/ph/lb/pose-labels-{slurm_job_id}.out")
            
            # Also try the old location as fallback
            if not log_file.exists():
                log_file = Path(f"/tmp/pose_job_{job_id}.out")
            
            print(f"🔍 Looking for log file: {log_file}")  # Debug info
            
            if log_file.exists():
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
                        output_base = Path("/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking/output/detections")
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