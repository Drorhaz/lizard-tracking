#!/usr/bin/env bash
# One-liner entry point: "./hpc/submit_labels_gpu.sh"
set -euo pipefail
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$here/.."
sbatch hpc/run_labels_gpu.sbatch
echo "Submitted. Check 'squeue -u $USER' and logs/ for progress."
