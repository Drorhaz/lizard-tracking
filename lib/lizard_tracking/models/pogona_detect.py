from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

@dataclass
class HeadDet:
    bbox_xyxy: Tuple[float,float,float,float]
    conf: float

class PogonaHeadDetect:
    def __init__(self, weights: str, imgsz=640, conf=0.25):
        self.model = YOLO(weights)
        self.imgsz = imgsz; self.conf = conf
    def predict(self, image_bgr) -> List[HeadDet]:
        res = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        if res.boxes is None: return out
        for b,c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append(HeadDet(tuple(map(float,b)), float(c)))
        return out
