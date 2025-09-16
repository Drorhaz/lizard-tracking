# Inference Performance Testing

This directory contains comprehensive inference performance testing for the baseline lizard detection model.

## 📁 Directory Structure
```
tests/inference/
├── README.md                 # This file
├── test_inference.py         # Performance testing script
├── RESULTS.md               # Detailed test results (Sep 16, 2025)
└── inference_demo/          # Visual demo results with bounding boxes
    ├── predict/             # Demo result 1
    ├── predict2/            # Demo result 2
    └── predict3/            # Demo result 3
```

## 🚀 **Quick Start**

### Run Inference Performance Test
```bash
# From project root
cd tests/inference
python3 test_inference.py
```

### Test Your Own Images
```python
from ultralytics import YOLO

# Load baseline model
model = YOLO('../../experiments/sanity_check_cpu/best.pt')

# Run inference with timing
import time
start = time.time()
results = model('path/to/your/image.jpg')
inference_time = (time.time() - start) * 1000  # ms

print(f"Inference time: {inference_time:.1f}ms")
print(f"Detections: {len(results[0].boxes) if results[0].boxes else 0}")
```

## 📊 **Latest Results Summary**

### Performance Grades
- **Speed**: 🚀 **Excellent** (62.9ms avg, 15.9 fps)
- **Detection Rate**: ⚠️ **Moderate** (75% images detected)  
- **Confidence**: 👍 **Good** (68.6% average)
- **Overall**: **B+ - Production Ready**

### Key Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| Average Inference Time | 62.9 ms | Real-time capable |
| Theoretical FPS | 15.9 fps | Good for video |
| Detection Rate | 75% (15/20 images) | Room for improvement |
| Average Confidence | 68.6% | Reliable predictions |

## 🧪 **Test Details**

### Test Configuration
- **Model**: YOLOv11s baseline (98.3% training mAP@0.5)
- **Hardware**: Mac M1 Pro (CPU only)
- **Dataset**: 20 validation images (thermal + visible)
- **Test Date**: September 16, 2025

### What the Test Measures
1. **Inference Speed**: Time per image (ms) and theoretical FPS
2. **Detection Performance**: Success rate and accuracy
3. **Confidence Analysis**: Prediction reliability scores
4. **Consistency**: Performance variance across images
5. **Multi-Modal**: Performance on thermal vs visible images

## 🎯 **Production Assessment**

### ✅ **Ready for Production**
The baseline model is approved for production deployment in:
- Real-time lizard tracking applications
- Video processing systems (15+ fps capability)
- Batch image analysis workflows
- CPU-only edge deployments
- Mixed thermal/visible camera environments

### 🚀 **Future Optimizations**
- GPU acceleration → <20ms inference
- Model optimization (TensorRT/ONNX)
- Additional training data → higher detection rates
- Confidence threshold tuning

## 📈 **How to Interpret Results**

### Speed Grades
- 🚀 **Excellent**: <100ms (real-time capable)
- ⚡ **Good**: 100-200ms (suitable for most applications)  
- 🚶 **Acceptable**: 200-500ms (batch processing)
- 🐌 **Slow**: >500ms (optimization needed)

### Detection Rate Grades
- 🎯 **Excellent**: >90% images with detections
- ✅ **Good**: 80-90% detection rate
- ⚠️ **Moderate**: 60-80% detection rate
- ❌ **Poor**: <60% detection rate

### Confidence Grades
- 💪 **High**: >80% average confidence
- 👍 **Good**: 60-80% average confidence
- ⚠️ **Moderate**: 40-60% average confidence
- ❌ **Low**: <40% average confidence

## 🔄 **Running New Tests**

### Test New Models
```python
# Edit test_inference.py line 19:
model_path = "path/to/your/new/model.pt"

# Run test
python3 test_inference.py
```

### Test on Different Hardware
The script automatically detects hardware and adjusts expectations accordingly.

### Custom Test Parameters
Modify these variables in `test_inference.py`:
- `image_files[:20]` → Change number of test images
- Test different confidence thresholds
- Add custom image directories

## 📊 **Baseline Comparison**

Use these baseline results to compare future model improvements:
- **Speed baseline**: 62.9ms average inference time
- **Accuracy baseline**: 75% detection rate  
- **Confidence baseline**: 68.6% average confidence
- **Overall baseline**: B+ grade, production-ready

---
*For detailed results and analysis, see [RESULTS.md](RESULTS.md)*