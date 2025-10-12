#!/bin/bash
"""
Launch HPC Web Interface with proper environment
"""

# Activate the LizardPose environment
echo "🔧 Activating LizardPose environment..."
conda activate /scratch200/$USER/LizardPose

# Check if activation worked
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate LizardPose environment"
    exit 1
fi

echo "✅ Environment activated successfully"

# Launch the web interface
echo "🚀 Starting HPC Pose Pipeline Web Interface..."
cd /a/home/cc/students/neurosci/$USER/sandbox/lizard-tracking/pose-head
python launch_hpc_web.py