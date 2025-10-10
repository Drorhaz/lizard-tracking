#!/bin/bash

# ============================================================================
# Train YOLO Pose Model with Embedding Enhancement
# ============================================================================

set -e

# Configuration
PROJECT_ROOT="/a/home/cc/students/neurosci/bareketd1/sandbox/lizard-tracking"
CONDA_ENV="/scratch200/bareketd1/LizardPose"
DATASET_DIR="${PROJECT_ROOT}/dataset"
OUTPUT_DIR="${PROJECT_ROOT}/output/models/embedding_pose"
MINED_DATA_DIR="${PROJECT_ROOT}/output/mined_data"

# Training parameters
EPOCHS=100
BATCH_SIZE=8
EMBEDDING_DIM=64
CONTRASTIVE_WEIGHT=0.1
TRIPLET_WEIGHT=0.05
BASE_MODEL="yolo11n-pose.pt"

echo "🚀 Starting YOLO Pose Model Training with Embeddings"
echo "=============================================="
echo "Project: ${PROJECT_ROOT}"
echo "Dataset: ${DATASET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Embedding dim: ${EMBEDDING_DIM}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment..."
source activate ${CONDA_ENV}
cd ${PROJECT_ROOT}

# Step 1: Mine contrastive pairs from existing data
echo "⛏️ Step 1: Mining contrastive pairs..."
python tools/mine_pose_data.py \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${MINED_DATA_DIR}" \
    --sequence_length 30 \
    --temporal_gap 5 \
    --min_confidence 0.3

if [ $? -ne 0 ]; then
    echo "❌ Data mining failed!"
    exit 1
fi

echo "✅ Data mining completed"

# Step 2: Create training dataset YAML
echo "📝 Step 2: Creating training configuration..."
cat > "${MINED_DATA_DIR}/pogona_embedding.yaml" << EOF
# Pose dataset with contrastive learning
path: ${MINED_DATA_DIR}
train: images
val: images

# Classes
names:
  0: lizard_head

# Keypoints (nose, ear_left, ear_right)
kpt_shape: [3, 2]

# Contrastive learning configuration
train_pairs: train_pairs.json
val_pairs: val_pairs.json
contrastive_weight: ${CONTRASTIVE_WEIGHT}
triplet_weight: ${TRIPLET_WEIGHT}
embedding_dim: ${EMBEDDING_DIM}
EOF

# Step 3: Start training with embedding enhancement
echo "🧠 Step 3: Training pose model with embeddings..."

python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))

from lizard_tracking.training.contrastive_trainer import train_embedding_pose_model

# Train the model
trainer = train_embedding_pose_model(
    data_yaml='${MINED_DATA_DIR}/pogona_embedding.yaml',
    model_cfg='${BASE_MODEL}',
    epochs=${EPOCHS},
    batch_size=${BATCH_SIZE},
    embedding_dim=${EMBEDDING_DIM},
    contrastive_weight=${CONTRASTIVE_WEIGHT},
    triplet_weight=${TRIPLET_WEIGHT},
    project='${OUTPUT_DIR}',
    name='head_pose_embedding',
    device='auto',
    patience=20,
    save_period=10,
    plots=True,
    val=True
)

print('✅ Training completed successfully!')
print(f'📊 Best model saved to: {trainer.best}')
print(f'📈 Training logs: {trainer.save_dir}')
"

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 4: Evaluate the trained model
echo "📊 Step 4: Evaluating trained model..."

BEST_MODEL=$(find ${OUTPUT_DIR} -name "best.pt" -type f | head -1)

if [ -n "$BEST_MODEL" ]; then
    echo "📈 Found best model: $BEST_MODEL"
    
    # Run validation
    python -c "
from ultralytics import YOLO

model = YOLO('${BEST_MODEL}')
results = model.val(
    data='${MINED_DATA_DIR}/pogona_embedding.yaml',
    split='val',
    plots=True,
    save_json=True
)

print('📊 Validation Results:')
print(f'   mAP50: {results.box.map50:.4f}')
print(f'   mAP50-95: {results.box.map:.4f}')
if hasattr(results, 'pose'):
    print(f'   Pose mAP50: {results.pose.map50:.4f}')
    print(f'   Pose mAP50-95: {results.pose.map:.4f}')
"

    # Test inference speed
    echo "⚡ Testing inference speed..."
    python -c "
import time
import cv2
from ultralytics import YOLO

model = YOLO('${BEST_MODEL}')

# Test video path
test_video = '${PROJECT_ROOT}/arena_mock_app/videos/top_20250916T150021.mp4'

cap = cv2.VideoCapture(test_video)
times = []

for i in range(100):  # Test 100 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()
    results = model(frame, conf=0.3, verbose=False)
    end_time = time.time()
    
    times.append(end_time - start_time)

cap.release()

if times:
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f'⚡ Inference Performance:')
    print(f'   Average time: {avg_time*1000:.2f}ms')
    print(f'   Average FPS: {fps:.1f}')
else:
    print('⚠️ No frames processed for speed test')
"

    echo ""
    echo "🎯 Step 5: Model Comparison"
    echo "Comparing embedding model vs original model..."
    
    ORIGINAL_MODEL="${PROJECT_ROOT}/output/models/head_pose/best.pt"
    
    if [ -f "$ORIGINAL_MODEL" ]; then
        python -c "
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load models
original_model = YOLO('${ORIGINAL_MODEL}')
embedding_model = YOLO('${BEST_MODEL}')

test_video = '${PROJECT_ROOT}/arena_mock_app/videos/top_20250916T150021.mp4'
cap = cv2.VideoCapture(test_video)

original_detections = []
embedding_detections = []
frame_count = 0

print('🔍 Comparing models on test video...')

while frame_count < 50:  # Test first 50 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # Original model
    orig_results = original_model(frame, conf=0.3, verbose=False)
    orig_count = len(orig_results[0].boxes) if len(orig_results) > 0 else 0
    original_detections.append(orig_count)
    
    # Embedding model  
    emb_results = embedding_model(frame, conf=0.3, verbose=False)
    emb_count = len(emb_results[0].boxes) if len(emb_results) > 0 else 0
    embedding_detections.append(emb_count)
    
    frame_count += 1

cap.release()

# Calculate detection rates
orig_rate = sum(1 for x in original_detections if x > 0) / len(original_detections)
emb_rate = sum(1 for x in embedding_detections if x > 0) / len(embedding_detections)

print(f'📊 Detection Rate Comparison:')
print(f'   Original model: {orig_rate:.1%} ({sum(original_detections)} total detections)')
print(f'   Embedding model: {emb_rate:.1%} ({sum(embedding_detections)} total detections)')
print(f'   Improvement: {(emb_rate - orig_rate)*100:+.1f} percentage points')
"
    else
        echo "⚠️ Original model not found for comparison"
    fi

else
    echo "❌ No trained model found!"
    exit 1
fi

echo ""
echo "🎉 TRAINING PIPELINE COMPLETED!"
echo "=============================================="
echo "📁 Model location: $BEST_MODEL"
echo "📊 Training logs: ${OUTPUT_DIR}"
echo "🔗 To use the new model, update MODEL_PATH in your .env file:"
echo "   MODEL_PATH=${BEST_MODEL}"
echo ""
echo "🧠 The embedding-enhanced model should provide:"
echo "   ✓ Better temporal consistency"
echo "   ✓ Improved gap filling"
echo "   ✓ More robust pose tracking"
echo ""
echo "💡 Next steps:"
echo "   1. Update your app configuration"
echo "   2. Test with live video stream"
echo "   3. Monitor detection improvements"