#!/usr/bin/env bash
set -euo pipefail
WEIGHTS=${1:-runs/pose/pogona_head_pose/weights/best.pt}
DATA=${2:-data/pogona_head_pose.yaml}
DEVICE=${DEVICE:-0}

exec python -m lizard_tracking.cli validate "$WEIGHTS" --data "$DATA" --device "$DEVICE"
