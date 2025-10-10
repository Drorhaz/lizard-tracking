#!/bin/bash

# ===================================================================    python -c "
from ultralytics import YOLO
import torch
import os

print('🔧 Training configuration:')
print(f'   Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')
print(f'   CUDA available: {torch.cuda.is_available()}')

# Load model
model = YOLO('${BASE_MODEL}')"# Enhanced YOLO Pose Training with Advanced Augmentation
# ============================================================================

set -e

# Configuration
PROJECT_ROOT="/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking"
CONDA_ENV="/scratch200/bareketd1/LizardPose"
DATASET_DIR="${PROJECT_ROOT}/dataset"
OUTPUT_DIR="${PROJECT_ROOT}/output/models/enhanced_pose"

# Training parameters
EPOCHS=150
BATCH_SIZE=16
BASE_MODEL="yolo11n-pose.pt"
PATIENCE=25

echo "🚀 Starting Enhanced YOLO Pose Model Training"
echo "=============================================="
echo "Project: ${PROJECT_ROOT}"
echo "Dataset: ${DATASET_DIR}" 
echo "Output: ${OUTPUT_DIR}"
echo "Base model: ${BASE_MODEL}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment..."
source activate ${CONDA_ENV}
cd ${PROJECT_ROOT}

# Count dataset
TOTAL_IMAGES=$(find ${DATASET_DIR}/images -name "*.jpg" | wc -l)
TOTAL_LABELS=$(find ${DATASET_DIR}/labels -name "*.txt" | wc -l)

echo "📊 Dataset Statistics:"
echo "   Images: ${TOTAL_IMAGES}"
echo "   Labels: ${TOTAL_LABELS}"
echo "   Coverage: $(( TOTAL_LABELS * 100 / TOTAL_IMAGES ))%"

# Create enhanced dataset YAML
echo "📝 Creating enhanced training configuration..."
mkdir -p ${OUTPUT_DIR}

cat > "${OUTPUT_DIR}/enhanced_pose.yaml" << EOF
# Enhanced pose dataset configuration
path: ${DATASET_DIR}
train: images
val: images

# Classes
names:
  0: lizard_head

# Keypoints (nose, ear_left, ear_right)
kpt_shape: [3, 2]

# Enhanced training augmentations
hsv_h: 0.015      # Hue augmentation (fraction)
hsv_s: 0.7        # Saturation augmentation (fraction)
hsv_v: 0.4        # Value augmentation (fraction)
degrees: 15.0     # Rotation degrees
translate: 0.1    # Translation (fraction)
scale: 0.5        # Scale augmentation (fraction)
shear: 0.0        # Shear (degrees)
perspective: 0.0  # Perspective (probability)
flipud: 0.0       # Vertical flip probability
fliplr: 0.5       # Horizontal flip probability
mosaic: 1.0       # Mosaic probability
mixup: 0.2        # Mixup probability
copy_paste: 0.1   # Copy-paste probability
EOF

echo "✅ Configuration created"

# Step 1: Train enhanced model
echo "🧠 Starting enhanced training..."

python -c "
from ultralytics import YOLO
import torch

print('🔧 Training configuration:')
print(f'   Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')
print(f'   CUDA available: {torch.cuda.is_available()}')

# Load model
model = YOLO('${BASE_MODEL}')

# Enhanced training with better hyperparameters
results = model.train(
    data='${OUTPUT_DIR}/enhanced_pose.yaml',
    epochs=${EPOCHS},
    batch=${BATCH_SIZE},
    imgsz=640,
    device='auto',
    
    # Learning rate schedule
    lr0=0.01,
    lrf=0.001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Advanced training settings
    patience=${PATIENCE},
    save_period=10,
    
    # Validation settings
    val=True,
    plots=True,
    save_json=True,
    
    # Output settings
    project='${OUTPUT_DIR}',
    name='enhanced_training',
    exist_ok=True,
    
    # Loss weights (optimized for pose)
    box=7.5,
    cls=0.5,
    pose=12.0,
    kobj=1.0,
    
    # Advanced augmentations
    copy_paste=0.1,
    mixup=0.2,
    mosaic=1.0,
    
    # Multi-scale training
    rect=False,
    
    # Optimizer
    optimizer='AdamW',
    cos_lr=True
)

print('✅ Training completed!')
print(f'📊 Best mAP: {results.best_fitness:.4f}')
print(f'📁 Model saved to: {results.save_dir}')
"

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 2: Find and evaluate best model
echo "📊 Evaluating trained model..."

BEST_MODEL=$(find ${OUTPUT_DIR} -name "best.pt" -type f | head -1)

if [ -n "$BEST_MODEL" ]; then
    echo "📈 Found best model: $BEST_MODEL"
    
    # Run comprehensive validation
    python -c "
from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO('${BEST_MODEL}')

# Validation metrics
print('📊 Running validation...')
val_results = model.val(
    data='${OUTPUT_DIR}/enhanced_pose.yaml',
    split='val',
    plots=True,
    save_json=True,
    conf=0.25,
    iou=0.7
)

print('📈 Validation Results:')
print(f'   Box mAP50: {val_results.box.map50:.4f}')
print(f'   Box mAP50-95: {val_results.box.map:.4f}')
if hasattr(val_results, 'pose') and val_results.pose:
    print(f'   Pose mAP50: {val_results.pose.map50:.4f}')
    print(f'   Pose mAP50-95: {val_results.pose.map:.4f}')

# Performance benchmarking
print('⚡ Running performance benchmark...')
test_video = '${PROJECT_ROOT}/arena_mock_app/videos/top_20250916T150021.mp4'

if os.path.exists(test_video):
    cap = cv2.VideoCapture(test_video)
    times = []
    detections_count = []
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        results = model(frame, conf=0.3, verbose=False)
        end_time = time.time()
        
        times.append(end_time - start_time)
        detections_count.append(len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0)
    
    cap.release()
    
    if times:
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        detection_rate = sum(1 for x in detections_count if x > 0) / len(detections_count)
        
        print(f'⚡ Performance Metrics:')
        print(f'   Average inference time: {avg_time*1000:.2f}ms')
        print(f'   Average FPS: {fps:.1f}')
        print(f'   Detection rate: {detection_rate:.1%}')
        print(f'   Total detections: {sum(detections_count)}')
    
    # Export optimized model
    print('🚀 Exporting optimized model...')
    try:
        model.export(format='onnx', optimize=True, simplify=True)
        print('✅ ONNX model exported successfully')
    except Exception as e:
        print(f'⚠️ ONNX export failed: {e}')

else:
    print('⚠️ Test video not found, skipping performance test')
"

    # Model comparison
    echo ""
    echo "🎯 Comparing with original model..."
    
    ORIGINAL_MODEL="${PROJECT_ROOT}/output/models/head_pose/best.pt"
    
    if [ -f "$ORIGINAL_MODEL" ]; then
        python -c "
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load models
print('🔍 Loading models for comparison...')
original_model = YOLO('${ORIGINAL_MODEL}')
enhanced_model = YOLO('${BEST_MODEL}')

test_video = '${PROJECT_ROOT}/arena_mock_app/videos/top_20250916T150021.mp4'

if os.path.exists(test_video):
    cap = cv2.VideoCapture(test_video)
    
    original_detections = []
    enhanced_detections = []
    original_times = []
    enhanced_times = []
    
    print('📊 Running comparison on test video...')
    
    for i in range(50):  # Test 50 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Test original model
        start_time = time.time()
        orig_results = original_model(frame, conf=0.3, verbose=False)
        original_times.append(time.time() - start_time)
        orig_count = len(orig_results[0].boxes) if len(orig_results) > 0 and orig_results[0].boxes is not None else 0
        original_detections.append(orig_count)
        
        # Test enhanced model
        start_time = time.time()
        enh_results = enhanced_model(frame, conf=0.3, verbose=False)
        enhanced_times.append(time.time() - start_time)
        enh_count = len(enh_results[0].boxes) if len(enh_results) > 0 and enh_results[0].boxes is not None else 0
        enhanced_detections.append(enh_count)
    
    cap.release()
    
    # Calculate metrics
    orig_rate = sum(1 for x in original_detections if x > 0) / len(original_detections)
    enh_rate = sum(1 for x in enhanced_detections if x > 0) / len(enhanced_detections)
    orig_fps = 1.0 / np.mean(original_times)
    enh_fps = 1.0 / np.mean(enhanced_times)
    
    print(f'📊 Model Comparison Results:')
    print(f'   Original Model:')
    print(f'     Detection rate: {orig_rate:.1%}')
    print(f'     Average FPS: {orig_fps:.1f}')
    print(f'     Total detections: {sum(original_detections)}')
    print(f'   Enhanced Model:')
    print(f'     Detection rate: {enh_rate:.1%}')
    print(f'     Average FPS: {enh_fps:.1f}')
    print(f'     Total detections: {sum(enhanced_detections)}')
    print(f'   Improvement:')
    print(f'     Detection rate: {(enh_rate - orig_rate)*100:+.1f} percentage points')
    print(f'     FPS change: {enh_fps - orig_fps:+.1f}')
else:
    print('⚠️ Test video not found for comparison')
"
    else
        echo "⚠️ Original model not found for comparison"
    fi

else
    echo "❌ No trained model found!"
    exit 1
fi

echo ""
echo "🎉 ENHANCED TRAINING COMPLETED!"
echo "=============================================="
echo "📁 Enhanced model: $BEST_MODEL"
echo "📊 Training logs: ${OUTPUT_DIR}"
echo ""
echo "🔗 To use the enhanced model, update your .env file:"
echo "   MODEL_PATH=${BEST_MODEL}"
echo ""
echo "✨ Enhanced model improvements:"
echo "   ✓ Advanced data augmentation"
echo "   ✓ Optimized hyperparameters"
echo "   ✓ Better loss weighting for pose detection"
echo "   ✓ Longer training with patience"
echo "   ✓ AdamW optimizer with cosine learning rate"
echo ""
echo "📈 Expected improvements:"
echo "   • Higher detection rate"
echo "   • Better pose accuracy"
echo "   • More robust to lighting changes"
echo "   • Improved generalization"