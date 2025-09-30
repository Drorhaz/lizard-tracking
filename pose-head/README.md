# Video Pose Pipeline — GPU/CPU configurable + HPC submit

- Device selection via env (`MODEL_DEVICE=cuda:0` or `cpu`) and optional FP16 (`MODEL_HALF=true`).
- Progress bar with `tqdm` (frames processed, ETA).
- Headless-safe (no cv2.imshow by default).
- **One-line Slurm submit**: `./hpc/submit_labels_gpu.sh`

## Layout
video-pose-pipeline-hpc/
├─ pipeline/
│  ├─ video_pose_pipeline.py
│  └─ web_preview.py
├─ config/
│  └─ .env.example
├─ hpc/
│  ├─ run_labels_gpu.sbatch
│  └─ submit_labels_gpu.sh
└─ README.md

## Configure
```bash
cp config/.env.example config/.env
# edit config/.env (VIDEO_PATH, MODEL_PATH/MODEL_DIR, MODEL_DEVICE, etc.)
```

## Local run
```bash
pip install ultralytics opencv-python-headless numpy python-dotenv tqdm
python pipeline/video_pose_pipeline.py
```

## HPC Slurm (one-liner from VSCode/SSH)
```bash
./hpc/submit_labels_gpu.sh
# or: sbatch hpc/run_labels_gpu.sbatch
```
- The sbatch script sources `config/.env`, sets `MODE=LABELS_ONLY`, and disables previews by default.
- Edit module/conda lines in `hpc/run_labels_gpu.sbatch` for your cluster.

### Tips
- CPU run? set `MODEL_DEVICE=cpu` and `IMG_SIZE=512` or `640`, maybe `PROCESS_EVERY_N=2`.
- GPU run? set `MODEL_DEVICE=cuda:0` and `MODEL_HALF=true` for speed-up.
- Turn off image saving (`LABEL_EVERY_N=0`) if disk is slow; labels `.txt` still saved every detection.
