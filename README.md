# Real-Time Lizard Tracking — Head Detection

This repo contains the code and assets for **real-time lizard head detection** using YOLO (v11 small) and the project workflow described in the initial report.

## ⚠️ Why a sanity run first (Mac users)
On macOS, the **MPS (Apple GPU) backend** can cause training to silently fail (losses/metrics = 0). To avoid this:
- First run a **short CPU sanity training** (`tools/train_sanity.py`) for 2–3 epochs to confirm that learning works.
- Then run the **full training** (`train_full.py`) on a CUDA GPU (Colab/Kaggle/remote workstation).

---

## Quick Start

### 0) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate  # or use your preferred env
pip install -r requirements.txt
```

### 1) Dataset (YOLO format)
```
dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
```
> The `dataset/` folder is **gitignored**. Keep your data local or mount in Colab/remote.

### 2) Sanity training (CPU on Mac)
Run a tiny training pass to verify learning works:
```bash
python tools/train_sanity.py
```
**Expected:** non-zero losses; validation metrics > 0 in `runs/detect/.../results.csv`.

### 3) Full training (CUDA / Colab / remote)
Once sanity passes, run the long training on a CUDA GPU:
```bash
python train_full.py --device 0 --epochs 60 --batch auto
```
This will create a `runs/detect/...` folder with `weights/best.pt` and training plots.

### 4) Inference & FPS check
(Optional) After training, run inference + latency profiling:
```bash
python tools/bench_infer.py --weights runs/detect/<RUN>/weights/best.pt --images dataset/images/val
```

---

## Repository Map (aligned with project phases)
- **configs/** – Ultralytics data YAML (Phase 1)
- **dataset/** – YOLO images/labels (local only, ignored by Git)
- **tools/** – Sanity train, verify dataset, benchmark inference (T1/T3)
- **reports/** – CSVs/plots/summary (Phase 7)
- **demos/** – Demo video & slides (Phase 7)
- **docs/** – ADRs & notes (e.g., `adr-001-heading.md` for T5)
- **models/** – Optional model exports (prefer Git LFS)
- **runs/** – YOLO outputs (auto, ignored)

---

## Known issues
- **MPS training on Mac:** may yield all zeros → always run `tools/train_sanity.py` on CPU first.
- **Large files:** Don’t commit datasets or weights (`*.pt`); use Git LFS if needed.

---

## License
Add your chosen license here.
