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
    
    // Immediately update UI
    document.getElementById('statusText').textContent = 'Stopping pipeline...';
    document.getElementById('stopBtn').disabled = true;
    
    fetch('/api/stop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({job_id: jobId})
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            // Reset UI immediately
            document.getElementById('playBtn').classList.remove('hidden');
            document.getElementById('stopBtn').classList.add('hidden');
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('statusText').textContent = 'Pipeline stopped';
            
            // Reset all stats
            document.getElementById('fpsValue').textContent = '0.0';
            document.getElementById('detectionRate').textContent = '0.0%';
            document.getElementById('processedFrames').textContent = '0';
            document.getElementById('totalDetections').textContent = '0';
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressText').textContent = '0% complete';
            
            stopStatusUpdates();
            jobId = null;
        } else {
            document.getElementById('statusText').textContent = 'Failed to stop pipeline';
            document.getElementById('stopBtn').disabled = false;
        }
    })
    .catch(error => {
        console.error('Error stopping pipeline:', error);
        document.getElementById('statusText').textContent = 'Error stopping pipeline';
        document.getElementById('stopBtn').disabled = false;
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
            document.getElementById('totalDetections').textContent = data.total_detections || 0;
            
            if (data.progress !== undefined) {
                document.getElementById('progressFill').style.width = data.progress + '%';
                document.getElementById('progressText').textContent = data.progress.toFixed(1) + '% complete';
            }

            // Update log
            if (data.log_lines) {
                const logDiv = document.getElementById('logOutput');
                data.log_lines.forEach(line => {
                    logDiv.innerHTML += line + '\n';
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

function viewCompletedRuns() {
    // Simple way to view completed runs - just refresh the video stream
    // The backend will automatically show the most recent completed run
    document.getElementById('statusText').textContent = 'Viewing most recent completed run...';
    const videoImg = document.querySelector('.video-stream');
    if (videoImg) {
        // Force refresh the video stream
        const currentSrc = videoImg.src;
        videoImg.src = '';
        setTimeout(() => {
            videoImg.src = currentSrc + '?t=' + Date.now();
        }, 100);
    }
}