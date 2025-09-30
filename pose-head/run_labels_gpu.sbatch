#!/bin/bash
#SBATCH --job-name=pose-labels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# Load your environment (edit these lines for your cluster)
# Example: conda
# module load cuda/12.0  # if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate LizardPose

# Project root = directory of this sbatch file's parent
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

# Export config vars so the python process sees them
set -a
if [ -f "config/.env" ]; then
  source config/.env
fi
# Force headless & labels-only unless overridden in .env
MODE="${MODE:-LABELS_ONLY}"
PREVIEW="${PREVIEW:-false}"
WEB_PREVIEW="${WEB_PREVIEW:-false}"
set +a

echo "Working dir: $(pwd)"
echo "Video: ${VIDEO_PATH:-<unset>}"
echo "Device: ${MODEL_DEVICE:-cuda:0}"
python -m pip install --user ultralytics opencv-python-headless numpy python-dotenv tqdm >/dev/null 2>&1 || true

python pipeline/video_pose_pipeline.py
