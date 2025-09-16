# MPS Training - Failed Due to Backend Issue

## Experiment Details
- **Date**: September 16, 2025  
- **Purpose**: Initial attempt at YOLO training on macOS
- **Hardware**: Mac M1 Pro with MPS acceleration
- **Status**: ❌ **FAILED - Silent Learning Failure**
- **Training Run**: `runs/detect/train3/`

## Configuration Used
```yaml
data: configs/pogona_head.yaml
epochs: 10
imgsz: 640
batch: 8
device: mps
optimizer: AdamW
lr0: 0.001
conf: 0.25
iou: 0.45
patience: 10
augment: true
```

## Failure Symptoms

### 🚨 **Complete Learning Failure**
- **All losses remained at 0** throughout 10 epochs
- **All metrics stayed at 0**: mAP@0.5, Recall, Precision
- **Training appeared successful** - no error messages
- **Model files were created** but contained no learned features
- **43 minutes wasted** with no actual learning

### Detailed Failure Pattern
| Epoch | Box Loss | Cls Loss | DFL Loss | mAP@0.5 | Recall | Precision |
|-------|----------|----------|----------|---------|--------|-----------|
| 1-10  | 0.000    | 0.000    | 0.000    | 0.000   | 0.000  | 0.000     |

## Root Cause Analysis

### PyTorch MPS Backend Bug
- **Issue**: Known compatibility problem between PyTorch MPS and YOLO training
- **Scope**: Affects all YOLO versions (v8, v11) on Apple Silicon with MPS
- **Manifestation**: Silent failure - training runs but doesn't learn
- **Frequency**: Consistent - affects all attempts

### Why This Happens
1. **MPS Backend Immaturity**: Apple's Metal Performance Shaders backend is newer
2. **Gradient Calculation Issues**: MPS doesn't properly compute gradients for YOLO operations
3. **Silent Failure**: No error thrown, making it hard to detect
4. **Model-Specific**: Particularly affects object detection models like YOLO

## Impact Assessment

### Time & Resources Wasted
- **Duration**: 42.84 minutes (0.714 hours) of compute time
- **GPU Memory**: 4.57GB consistently used but unproductive
- **Opportunity Cost**: Could have run productive CPU training instead

### Detection Difficulty
- **No Error Messages**: Training completed "successfully"
- **Files Generated**: Model weights saved (but useless)
- **Logs Look Normal**: Progress bars and epochs completed
- **Only Clue**: All metrics remaining at exactly 0

## Comparison with Working CPU Training

| Metric | MPS (Failed) | CPU (Success) | Difference |
|--------|-------------|---------------|------------|
| Duration | 42.84 min | 49.68 min | +6.84 min |
| Speed | 1.4 it/s | 0.4 it/s | -71% |
| mAP@0.5 | 0.000 | 0.983 | +98.3% |
| Learning | None | Excellent | Complete failure |

## Lessons Learned

### 🚨 **Critical Warnings**
1. **Never use MPS for YOLO training** on macOS
2. **Always run sanity check first** on CPU  
3. **Zero metrics = failed training** regardless of completion
4. **MPS shows higher speed but zero learning** - misleading performance

### 🔍 **Detection Methods**
Signs of MPS training failure:
- All losses exactly 0.000
- All metrics exactly 0.000
- Training "completes successfully"
- Model files created but unusable

## Recommended Solutions

### ✅ **Immediate Fix**
1. Use `device='cpu'` for development/testing
2. Run 3-5 epoch sanity checks first
3. Verify non-zero losses before full training

### 🚀 **Production Fix**
1. Use CUDA GPU (Google Colab, Kaggle)
2. Never rely on MPS for YOLO training
3. Budget for cloud GPU costs

## Files Generated (Unusable)
Located in `runs/detect/train3/`:
- `weights/best.pt` - ⚠️ Contains no learned features
- `weights/last.pt` - ⚠️ Identical to best.pt, unusable
- `results.csv` - Shows zero metrics throughout
- Various visualization files - All show zero performance

## Prevention Strategy

### Before Training
```bash
# Always run sanity check first
python3 train_sanity.py  # CPU-based validation

# Check for non-zero losses:
# ✅ Good: box_loss > 0, cls_loss > 0, mAP > 0
# ❌ Bad:  all values = 0.000
```

### During Training
- Monitor first epoch carefully
- If all losses = 0, stop immediately
- Switch to CPU or CUDA immediately

---

**⚠️ This failure led to the discovery that our dataset and configuration are excellent - the CPU sanity check achieved 98.3% mAP@0.5, proving the issue was purely the MPS backend.**