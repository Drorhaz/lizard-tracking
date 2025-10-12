#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-data/pogona_head_pose.yaml}
EPOCHS=${EPOCHS:-150}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-16}
DEVICE=${DEVICE:-0}
MODEL=${MODEL:-yolo11s-pose.pt}
RUN_NAME=${RUN_NAME:-pogona_head_pose}
PROJECT=${PROJECT:-runs/pose}
LR0=${LR0:-0.01}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
PAT=${PAT:-}
EXTRA=${EXTRA:-}

CMD=(python -m lizard_tracking.cli train
  --data "$DATA"
  --model "$MODEL"
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --device "$DEVICE"
  --lr0 "$LR0"
  --weight-decay "$WEIGHT_DECAY"
  --project "$PROJECT"
  --run-name "$RUN_NAME"
)

if [[ -n "$PAT" ]]; then
  CMD+=(--patience "$PAT")
fi
if [[ -n "$EXTRA" ]]; then
  CMD+=(--extra "$EXTRA")
fi

exec "${CMD[@]}"
