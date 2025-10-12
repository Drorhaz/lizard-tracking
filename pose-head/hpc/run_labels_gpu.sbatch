#!/bin/bash
#SBATCH --job-name=pose-labels
#SBATCH --account=sheinmark-users
#SBATCH --partition=gpu-general
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G   
#SBATCH --time=1:00:00
#SBATCH --output=/scratch200/%u/logs/ph/lb/%x-%j.out
#SBATCH --error=/scratch200/%u/logs/ph/lb/%x-%j.err


set -euo pipefail
mkdir -p logs

# Load your environment (edit these lines for your cluster)
# Example: conda
# module load cuda/12.0  # if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate LizardPose

# Project root = directory of this sbatch file's parent

BASE_PATH="/a/home/cc/students/neurosci/$USER/sandbox/lizard-tracking"
cd "$BASE_PATH" || exit
echo "[INFO] changed directory to: $BASE_PATH"


#------------------------------------ activate conda env ------------------------------------#
CONDA_ENV="LizardPose"
module load miniconda/miniconda3-4.7.12-environmentally
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "/scratch200/$USER/$CONDA_ENV"
echo "[INFO] activated conda env: $CONDA_ENV"

#------------------------------------ log info ------------------------------------#
date

echo "[INFO] node(s): $(scontrol show hostnames "$SLURM_NODELIST" | xargs echo)"
echo "[INFO] JOB_ID: ${SLURM_JOB_ID}"



# Export config vars so the python process sees them

# Force headless & labels-only unless overridden in .env
MODE="${MODE:-LABELS_ONLY}"
PREVIEW="${PREVIEW:-false}"
WEB_PREVIEW="${WEB_PREVIEW:-false}"
# set -a
# source "$BASE_PATH/pose-head/config/.env"
# set +a

echo "Working dir: $(pwd)"
echo "Video: ${VIDEO_PATH:-<unset>}"
echo "Device: ${MODEL_DEVICE:-cuda:0}"
python -m pip install --user ultralytics opencv-python-headless numpy python-dotenv tqdm >/dev/null 2>&1 || true

#------------------------------------ run script ------------------------------------#


PATH_TO_SCRIPT="$BASE_PATH/pose-head/pipeline/video_pose_pipeline.py" 
echo "[INFO] running script: $PATH_TO_SCRIPT"

python "$PATH_TO_SCRIPT" --conf "$BASE_PATH/pose-head/config/.env"

#------------------------------------ log info ------------------------------------#
echo "[INFO] job finished"
date

