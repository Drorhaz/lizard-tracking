from pathlib import Path
import sys

# Add lib to path
ROOT_DIR = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import numpy as np
import pytest

from lizard_tracking.core import HeadPose
from lizard_tracking.models.pogona_pose import PogonaHeadPoseModel


def test_inference_smoke():
    weights = Path("runs/pose/pogona_head_pose/weights/best.pt")
    if not weights.exists():
        pytest.skip("pose weights not present; skipping inference smoke test")

    model = PogonaHeadPoseModel(str(weights), conf=0.2)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    preds = model.predict(img)
    assert isinstance(preds, list)
    if preds:
        head: HeadPose = preds[0]
        assert isinstance(head.bbox_xyxy, tuple) and len(head.bbox_xyxy) == 4
        assert head.nose and head.ear_left and head.ear_right
