"""Compatibility shims forwarding to the reorganised pipelines package."""
from ..pipelines import FrameResult, PoseTrainer, VideoTracker

__all__ = ["FrameResult", "PoseTrainer", "VideoTracker"]
