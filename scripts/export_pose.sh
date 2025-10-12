#!/usr/bin/env bash
set -euo pipefail
WEIGHTS=${1:-runs/pose/pogona_head_pose/weights/best.pt}
FORMATS=${FORMATS:-onnx,torchscript}
python - <<'PY'
import os
import sys
from lizard_tracking.config import PoseTrainingConfig
from lizard_tracking.pipelines import PoseTrainer

weights = sys.argv[1]
formats = sys.argv[2].split(',') if sys.argv[2] else ['onnx']
trainer = PoseTrainer(PoseTrainingConfig(model=weights))
for fmt in formats:
    print(f"[EXPORT] {fmt}")
    trainer.export(weights, fmt=fmt)
print("Export complete.")
PY
"$WEIGHTS" "$FORMATS"
