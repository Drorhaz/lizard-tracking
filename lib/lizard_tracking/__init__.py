"""Lizard tracking library for pose estimation and behavioral analysis."""

from . import config
from . import pipelines
from . import core
from . import models
from . import ui
from . import utils

# Import commonly used classes for easy access
from .config import PoseTrainingConfig, PoseInferenceConfig, VideoTrackingConfig
from .pipelines import PoseTrainer, VideoTracker, FrameResult
# Import draw utilities  
from .utils.draw_utils import (
    draw_head_pose, 
    draw_head_pose_from_object, 
    draw_no_detection,
    draw_behavioral_event,
    draw_processing_info,
    draw_trajectory_line
)

# Import video streaming utilities
from .utils.video_stream import (
    VideoStream,
    VideoPlayer,
    FlaskVideoStreamer,
    create_camera_stream,
    create_file_stream,
    stream_video
)

__all__ = [
    'config',
    'pipelines', 
    'core',
    'models',
    'ui',
    'utils',
    'PoseTrainingConfig',
    'PoseInferenceConfig', 
    'VideoTrackingConfig',
    'PoseTrainer',
    'VideoTracker',
    'FrameResult',
    'draw_head_pose',
    'draw_head_pose_from_object', 
    'draw_no_detection',
    'VideoStream',
    'VideoPlayer',
    'FlaskVideoStreamer',
    'create_camera_stream',
    'create_file_stream',
    'stream_video'
]
