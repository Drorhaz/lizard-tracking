let jobId = null;
let statusInterval = null;

// Toggle execution mode options
function toggleExecutionOptions() {
    const mode = document.getElementById('executionMode').value;
    const timeLimitGroup = document.getElementById('timeLimitGroup');
    const playBtn = document.getElementById('playBtn');
    
    if (mode === 'hpc') {
        timeLimitGroup.style.display = 'block';
        playBtn.innerHTML = '▶️ Start GPU Cluster Pipeline';
    } else if (mode === 'local_gpu') {
        timeLimitGroup.style.display = 'none';
        playBtn.innerHTML = '▶️ Start Local GPU Pipeline';
    } else {
        timeLimitGroup.style.display = 'none';
        playBtn.innerHTML = '▶️ Start Local CPU Pipeline';
    }
}

// Toggle detection mode (live vs offline)
function toggleDetectionMode() {
    const isLive = document.getElementById('liveMode').checked;
    const saveLabelsGroup = document.getElementById('saveLabels').parentElement;
    const confThreshGroup = document.getElementById('confThresh').parentElement;
    const imgSizeGroup = document.getElementById('imgSize').parentElement;
    
    console.log('Toggle detection mode:', isLive ? 'Live' : 'Offline');
    
    if (isLive) {
        // Live mode: show detection settings
        if (saveLabelsGroup) saveLabelsGroup.style.display = 'block';
        if (confThreshGroup) confThreshGroup.style.display = 'block';
        if (imgSizeGroup) imgSizeGroup.style.display = 'block';
        console.log('Showing live mode controls');
    } else {
        // Offline mode: hide detection settings (just load saved labels)
        if (saveLabelsGroup) saveLabelsGroup.style.display = 'none';
        if (confThreshGroup) confThreshGroup.style.display = 'none';
        if (imgSizeGroup) imgSizeGroup.style.display = 'none';
        console.log('Hiding live mode controls');
    }
}

// Initialize toggle functionality when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing controls...');
    
    // Add click handler to the glass toggle
    const glassToggle = document.querySelector('.glass-toggle-switch');
    if (glassToggle) {
        glassToggle.addEventListener('click', function(e) {
            console.log('Glass toggle clicked');
            const liveRadio = document.getElementById('liveMode');
            const offlineRadio = document.getElementById('offlineMode');
            
            // Toggle between modes
            if (liveRadio.checked) {
                offlineRadio.checked = true;
                console.log('Switched to offline mode');
            } else {
                liveRadio.checked = true;
                console.log('Switched to live mode');
            }
            
            // Update UI based on new selection
            toggleDetectionMode();
        });
        console.log('Glass toggle handler attached');
    } else {
        console.error('Glass toggle not found!');
    }
    
    // Also listen to radio button changes
    const liveRadio = document.getElementById('liveMode');
    const offlineRadio = document.getElementById('offlineMode');
    if (liveRadio) {
        liveRadio.addEventListener('change', toggleDetectionMode);
    }
    if (offlineRadio) {
        offlineRadio.addEventListener('change', toggleDetectionMode);
    }
    
    // Trigger on initial load
    toggleDetectionMode();
    
    // Log how many videos are available
    const videoSelect = document.getElementById('videoSelect');
    if (videoSelect) {
        const videoCount = videoSelect.options.length - 1; // Subtract 1 for "Select video..." option
        console.log(`Videos loaded from template: ${videoCount} videos available`);
    }
});

function startPipeline() {
    console.log('🚀 Starting pipeline...');
    
    try {
        const executionMode = document.getElementById('executionMode').value;
        const detectionModeElement = document.querySelector('input[name="detectionMode"]:checked');
        
        if (!detectionModeElement) {
            console.error('❌ No detection mode selected');
            alert('Error: No detection mode selected');
            return;
        }
        
        const detectionMode = detectionModeElement.value;
        const videoPath = document.getElementById('videoSelect').value;
        
        const config = {
            execution_mode: executionMode,
            detection_mode: detectionMode,
            video_path: videoPath,
            partition: executionMode === 'hpc' ? 'gpu' : 'local',
            time_limit: executionMode === 'hpc' ? document.getElementById('timeLimit').value : null,
            save_labels: document.getElementById('saveLabels').checked,
            conf_thresh: parseFloat(document.getElementById('confThresh').value),
            img_size: parseInt(document.getElementById('imgSize').value)
        };

        console.log('📋 Pipeline config:', config);

        if (!config.video_path) {
            alert('Please select a video file');
            console.error('❌ No video selected');
            return;
        }
        
        // Disable start button and show loading
        const playBtn = document.getElementById('playBtn');
        if (playBtn) {
            playBtn.disabled = true;
            playBtn.innerHTML = '⏳ Starting...';
        }
        
        console.log('📡 Sending request to /api/start...');
        
        // Send the start request
        fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        })
        .then(response => {
            console.log('📥 Response received:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Start response:', data);
            if (data.success) {
                jobId = data.job_id;
                console.log(`🎯 Job started with ID: ${jobId}`);
                
                // Update UI
                document.getElementById('statusText').textContent = 'Pipeline starting...';
                document.getElementById('jobId').textContent = jobId;
                
                // Show stop button, hide start button
                document.getElementById('playBtn').style.display = 'none';
                document.getElementById('stopBtn').classList.remove('hidden');
                
                // Start status polling
                startStatusPolling();
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        })
        .catch(error => {
            console.error('❌ Failed to start pipeline:', error);
            alert(`Failed to start pipeline: ${error.message}`);
            
            // Re-enable start button
            if (playBtn) {
                playBtn.disabled = false;
                playBtn.innerHTML = '▶️ Start Pipeline';
            }
        });
        
    } catch (error) {
        console.error('❌ Error in startPipeline:', error);
        alert(`Error: ${error.message}`);
    }
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
            document.getElementById('playBtn').style.display = 'inline-block';
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
            
            stopStatusPolling();
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

function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(updateStatus, 1000);
}

function stopStatusPolling() {
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
                stopStatusPolling();
                document.getElementById('playBtn').style.display = 'inline-block';
                document.getElementById('stopBtn').classList.add('hidden');
            }
        })
        .catch(error => {
            console.error('Error updating status:', error);
        });
}