#!/usr/bin/env python3
"""
Test script to verify model loading works
"""
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from pipeline.video_pose_pipeline import YOLOPoseModel
    
    # Check model file
    model_dir = Path("../output/models/head_pose")
    model_paths = list(model_dir.glob("**/best*.pt")) + list(model_dir.glob("**/*.pt"))
    
    if not model_paths:
        print(f"❌ No model found in {model_dir}")
        print(f"Directory exists: {model_dir.exists()}")
        if model_dir.exists():
            print(f"Contents: {list(model_dir.glob('*'))}")
    else:
        model_path = model_paths[0]
        print(f"✅ Found model: {model_path}")
        
        # Try to load model
        print("🤖 Loading model...")
        model = YOLOPoseModel(model_path, conf=0.1)
        print("✅ Model loaded successfully!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()