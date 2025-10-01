"""Pose training wrapper around the Ultralytics YOLO API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import re

from ultralytics import YOLO

from ..config import PoseTrainingConfig


class PoseTrainer:
    """High-level wrapper around Ultralytics YOLO pose training."""

    def __init__(self, config: PoseTrainingConfig):
        self.config = config
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.config.model)
        return self._model

    def train(self, overrides: Optional[Dict[str, Any]] = None):
        args = self.config.as_ultralytics_kwargs()
        if overrides:
            args.update(overrides)

        run_dir = self.config.run_directory
        run_dir.mkdir(parents=True, exist_ok=True)

        return self.model.train(
            task="pose",
            data=self.config.data_yaml,
            project=self.config.project,
            name=self.config.run_name,
            resume=self.config.resume,
            **args,
        )

    def best_checkpoint(self) -> Path:
        """Find the best checkpoint based on mAP50-95 metrics across all runs."""
        run_dir = Path(self.config.project)
        
        # List of possible mAP columns, in order of preference
        map_candidates = [
            "metrics/mAP50-95(P)",  # Pose mAP 0.5:0.95 (preferred)
            "metrics/pose/mAP50-95",
            "metrics/pose/mAP@0.5:0.95",
            "keypoints/mAP50-95",
            "keypoints/mAP@0.5:0.95",
            "metrics/mAP50-95",
            "metrics/mAP@0.5:0.95",
            "mAP50-95",
            "map50-95",
        ]
        
        best_checkpoint = None
        best_map = -1.0
        best_run_name = None
        
        # Look through all run directories
        for folder in run_dir.glob("*"):
            if not folder.is_dir():
                continue
                
            results_csv = folder / "results.csv"
            checkpoint = folder / "weights" / "best.pt"
            
            if not results_csv.exists() or not checkpoint.exists():
                continue
                
            try:
                # Read the results CSV
                df = pd.read_csv(results_csv)
                
                # Find the best mAP column available
                map_col = None
                for candidate in map_candidates:
                    if candidate in df.columns:
                        map_col = candidate
                        break
                
                # If no exact match, try regex patterns
                if map_col is None:
                    for col in df.columns:
                        # Look for pose-specific mAP columns first
                        if re.search(r"mAP\s*50-?95.*\(P\)", col, flags=re.I):
                            map_col = col
                            break
                        elif re.search(r"pose.*mAP.*50-?95", col, flags=re.I):
                            map_col = col
                            break
                    
                    # Fall back to any mAP50-95 style column
                    if map_col is None:
                        for col in df.columns:
                            if re.search(r"mAP\s*50-?95", col, flags=re.I) or re.search(r"mAP@?0\.5:?0\.95", col, flags=re.I):
                                map_col = col
                                break
                
                if map_col is None or map_col not in df.columns:
                    print(f"[WARN] No mAP column found in {results_csv}")
                    continue
                
                # Get the maximum mAP value from this run
                max_map = df[map_col].max()
                
                if pd.isna(max_map):
                    continue
                    
                # Update best if this run is better
                if max_map > best_map:
                    best_map = max_map
                    best_checkpoint = checkpoint
                    best_run_name = folder.name
                    
            except Exception as e:
                print(f"[WARN] Error processing {results_csv}: {e}")
                continue
        
        # Fall back to the original behavior if no metrics found
        if best_checkpoint is None:
            # Default to the configured run directory
            path = self.config.run_directory / "weights" / "best.pt"
            if path.exists():
                return path
                
            # Or find any best.pt file
            for folder in run_dir.glob("*"):
                candidate = folder / "weights" / "best.pt"
                if candidate.exists():
                    return candidate
            
            raise FileNotFoundError(
                "best.pt not found; run training or pass an explicit weights path"
            )
        
        print(f"[INFO] Best checkpoint: {best_run_name} (mAP50-95: {best_map:.4f})")
        return best_checkpoint

    def validate(self, weights: Optional[str] = None):
        weights_path = Path(weights) if weights else self.best_checkpoint()
        model = YOLO(str(weights_path))
        return model.val(task="pose", data=self.config.data_yaml)

    def export(self, weights: Optional[str] = None, fmt: str = "onnx", **kwargs):
        weights_path = Path(weights) if weights else self.best_checkpoint()
        model = YOLO(str(weights_path))
        return model.export(format=fmt, **kwargs)
