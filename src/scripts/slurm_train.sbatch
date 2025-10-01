#!/usr/bin/env bash
#SBATCH --job-name=pogona_optuna
#SBATCH --account=sheinmark-users
#SBATCH --partition=gpu-general
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G   
#SBATCH --time=12:00:00
#SBATCH --output=/scratch200/%u/logs/ph/%x-%j.out
#SBATCH --error=/scratch200/%u/logs/ph/%x-%j.err



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


PATH_TO_SCRIPT="$BASE_PATH/src/scripts/pogona_pipeline_cfg.py" #pogona_pipeline_cfg_optuna
echo "[INFO] running script: $PATH_TO_SCRIPT"

python "$PATH_TO_SCRIPT"

#------------------------------------ log info ------------------------------------#
echo "[INFO] job finished"
date
