#!/usr/bin/env bash
#SBATCH --job-name=pogona_optuna
#SBATCH --account=sheinmark-users
#SBATCH --partition=gpu-general
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G   
#SBATCH --time=8:00:00
#SBATCH --output=/scratch200/%u/logs/ph/compare/%x-%j.out
#SBATCH --error=/scratch200/%u/logs/ph/compare/%x-%j.err



BASE_PATH="$PWD/sandbox/lizard-tracking"
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



#------------------------------------ run script ------------------------------------#


PATH_TO_SCRIPT="$BASE_PATH/src/scripts/compare_runs_full.py" 
echo "[INFO] running script: $PATH_TO_SCRIPT"


# --- make sure CUDA is visible to the job ---
unset CUDA_VISIBLE_DEVICES                       # remove any "cpu-only" setting
# Prefer SLURM-provided variables, which list the allocated GPU ids
if [[ -n "$SLURM_JOB_GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
elif [[ -n "$SLURM_STEP_GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# (optional, but often required on clusters)
module -q load CUDA || true

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi | head -n 20 || true


srun --gres=gpu:1 --export=ALL python "$PATH_TO_SCRIPT"

#------------------------------------ log info ------------------------------------#
echo "[INFO] job finished"
date
