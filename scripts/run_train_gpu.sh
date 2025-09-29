#!/usr/bin/env bash
set -euo pipefail
CONDA_ENV="LizardPose"
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
python scripts/pogona_pipeline_cfg_optuna.py
