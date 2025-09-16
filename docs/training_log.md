# Training Log - Pogona Head Detection

## Project Overview
- **Model**: YOLOv11s (9.4M parameters)
- **Task**: Lizard head detection
- **Dataset**: 2,974 labeled images (thermal + visible light)
- **Classes**: 1 (pogona_head)

## Training History

### 🚨 **Critical Discovery: MPS Backend Issue (Sep 16, 2025)**

#### Initial MPS Training (FAILED)
- **Run**: `runs/detect/train3/`
- **Hardware**: Mac M1 Pro with MPS acceleration
- **Duration**: 42.84 minutes (10 epochs)
- **Result**: **TRAINING FAILURE** - Silent learning failure

**Symptoms:**
- All losses stayed at 0 throughout training
- All metrics (mAP@0.5, Recall, Precision) remained at 0
- Training appeared to complete successfully
- Model was saved but learned nothing

**Configuration:**
```yaml
epochs: 10
batch: 8
device: mps
imgsz: 640
optimizer: AdamW
lr0: 0.001
```

#### ✅ **CPU Sanity Check (SUCCESSFUL)**
- **Run**: `runs/detect/train4/`
- **Hardware**: Mac M1 Pro (CPU only)
- **Duration**: 49.68 minutes (3 epochs)
- **Result**: **EXCELLENT PERFORMANCE**

**Final Metrics (Epoch 3):**
- **mAP@0.5**: 0.983 (98.3%)
- **Recall**: 0.953 (95.3%)  
- **Precision**: 0.958 (95.8%)
- **mAP@0.5-0.95**: 0.674 (67.4%)

**Loss Progression:**
| Epoch | Box Loss | Cls Loss | DFL Loss |
|-------|----------|----------|----------|
| 1     | 1.415    | 1.357    | 1.140    |
| 2     | 1.246    | 0.777    | 1.073    |
| 3     | 1.167    | 0.664    | 1.028    |

**Configuration:**
```yaml
epochs: 3
batch: 8  
device: cpu
imgsz: 640
optimizer: AdamW
lr0: 0.001
workers: 2
seed: 42
```

## 🔍 **Analysis & Findings**

### MPS Backend Bug Confirmed
The dramatic difference between MPS and CPU results confirms a known PyTorch MPS backend issue with YOLO training:

| Backend | Losses     | Final mAP@0.5 | Learning Status |
|---------|------------|---------------|-----------------|
| MPS     | All zeros  | 0.000         | ❌ BROKEN       |
| CPU     | Decreasing | 0.983         | ✅ WORKING      |

### Dataset Validation
The CPU training proves our dataset and configuration are excellent:
- **98.3% mAP@0.5 in just 3 epochs** indicates high-quality labels
- Consistent loss decrease shows proper learning
- Both thermal and visible images are properly handled

## 📋 **Recommendations**

### For Production Training:
1. **Never use MPS** for YOLO training on macOS
2. **Use CUDA GPU** (Google Colab, Kaggle, or NVIDIA cloud)
3. **Target 50-80 epochs** for full training
4. **Expected performance**: >95% mAP@0.5 based on sanity results

### For Development:
1. **Always run sanity check first** using `train_sanity.py`
2. **Use CPU for quick validation** (3-5 epochs)
3. **Verify non-zero losses** before full training

## 🚀 **Next Steps**
- [ ] Upload dataset to Google Colab
- [ ] Run full 50+ epoch training on CUDA
- [ ] Compare final results with CPU sanity check baseline
- [ ] Document production model performance

## 📁 **File Locations**
- Sanity check results: `runs/detect/train4/`
- Failed MPS training: `runs/detect/train3/`
- Training scripts: `train.py`, `train_sanity.py`
- Dataset: `dataset/labeled/`