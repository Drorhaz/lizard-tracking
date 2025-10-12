#!/usr/bin/env python3
"""Enhanced YOLO pose training script."""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def train_enhanced_pose():
    """Train enhanced YOLO pose model."""
    
    # Get script directory and project root using relative paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    # Configuration - use embedding dataset
    dataset_dir = project_root / "dataset" / "embedding_dataset"
    output_dir = project_root / "output/models/enhanced_pose"
    
    # Training parameters
    epochs = 150
    batch_size = 16
    base_model = "yolo11n-pose.pt"
    patience = 25
    
    print("🚀 Starting Enhanced YOLO Pose Model Training")
    print("=" * 50)
    print(f"Project: {project_root}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Check if embedding dataset exists
    if not dataset_dir.exists():
        print(f"❌ Embedding dataset not found at: {dataset_dir}")
        print("   Please run convert_labels.py first to create the embedding dataset")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count dataset
    total_images = len(list((dataset_dir / "images").rglob("*.jpg")))  # Recursive search
    total_labels = len(list((dataset_dir / "labels").rglob("*.txt")))  # Recursive search
    
    print("📊 Dataset Statistics:")
    print(f"   Images: {total_images}")
    print(f"   Labels: {total_labels}")
    if total_images > 0:
        print(f"   Coverage: {total_labels * 100 // total_images}%")
    else:
        print("   Coverage: No images found!")
        sys.exit(1)
    print()
    yaml_content = f"""# Enhanced pose dataset configuration
path: {dataset_dir}
train: images
val: images

# Classes
names:
  0: lizard_head

# Keypoints (nose, ear_left, ear_right)
kpt_shape: [3, 2]

# Keypoint flip indices for horizontal flip augmentation
# For horizontal flip: nose stays same (0), left ear (1) ↔ right ear (2)
flip_idx: [0, 2, 1]

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
"""
    
    yaml_path = output_dir / "enhanced_pose.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("✅ Configuration created")
    print()
    
    # Training configuration
    print("🔧 Training configuration:")
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"   Device: {device_name}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print()
    
    # Load and train model
    print("🧠 Starting enhanced training...")
    model = YOLO(base_model)
    
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
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
            patience=patience,
            save_period=10,
            
            # Validation settings
            val=True,
            plots=True,
            save_json=True,
            
            # Output settings
            project=str(output_dir),
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
        
        print("✅ Training completed!")
        print(f"📊 Best fitness: {getattr(results, 'best_fitness', 'N/A')}")
        print(f"📁 Model saved to: {results.save_dir}")
        
        # Find best model
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"🎯 Best model: {best_model_path}")
            
            # Quick validation
            print("📊 Running quick validation...")
            model_best = YOLO(str(best_model_path))
            val_results = model_best.val(data=str(yaml_path), plots=False, verbose=False)
            
            print("📈 Validation Results:")
            print(f"   Box mAP50: {val_results.box.map50:.4f}")
            print(f"   Box mAP50-95: {val_results.box.map:.4f}")
            
            if hasattr(val_results, 'pose') and val_results.pose:
                print(f"   Pose mAP50: {val_results.pose.map50:.4f}")
                print(f"   Pose mAP50-95: {val_results.pose.map:.4f}")
            
            print()
            print("🎉 ENHANCED TRAINING COMPLETED!")
            print("=" * 50)
            print(f"📁 Enhanced model: {best_model_path}")
            print()
            print("🔗 To use the enhanced model, update your .env file:")
            print(f"   MODEL_PATH={best_model_path}")
            print()
            print("✨ Enhanced model improvements:")
            print("   ✓ Advanced data augmentation")
            print("   ✓ Optimized hyperparameters")
            print("   ✓ Better loss weighting for pose detection")
            print("   ✓ Longer training with patience")
            print("   ✓ AdamW optimizer with cosine learning rate")
            
            return str(best_model_path)
        else:
            print("⚠️ Best model not found!")
            return None
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


if __name__ == "__main__":
    train_enhanced_pose()