import math
import numpy as np
from typing import Optional, Tuple

def rad_to_deg(a: Optional[float]) -> Optional[float]:
    return None if a is None else a * 180.0 / math.pi

def deg_to_rad(d: Optional[float]) -> Optional[float]:
    return None if d is None else d * math.pi / 180.0

def angle_from_delta(dx: float, dy: float) -> Optional[float]:
    if dx == 0 and dy == 0:
        return None
    return math.atan2(dy, dx)

def center_of_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def moving_average(points: list[Tuple[float, float]], window: int = 3) -> Tuple[float, float]:
    if len(points) == 0:
        return (0.0, 0.0)
    arr = np.array(points[-window:])
    return float(arr[:,0].mean()), float(arr[:,1].mean())