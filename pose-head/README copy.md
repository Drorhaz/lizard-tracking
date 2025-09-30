# Video Pose Pipeline (config-first)

- No CLI args; everything comes from `config/.env` (copy `.env.example`).
- Modes:
  - `INFER_LIVE`: run YOLO on the video, overlay best detection, save per-frame YOLO labels
  - `LABELS_ONLY`: run YOLO + save labels/frames, **no preview window**
  - `PLAYBACK_CACHE`: draw overlays from saved labels (skip inference)

## Layout
video-pose-pipeline/
├─ pipeline/
│  └─ video_pose_pipeline.py
├─ config/
│  └─ .env.example
└─ README.md

## Run
1) `cp config/.env.example config/.env` and edit VIDEO_PATH etc.
2) `pip install ultralytics opencv-python-headless numpy python-dotenv`
3) `python pipeline/video_pose_pipeline.py`

Outputs: `OUTPUT_DIR/<video-stem>-<timestamp>/{detections.csv, labeled_frames/, labels/, run_config.json}`
