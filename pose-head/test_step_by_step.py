#!/usr/bin/env python3
"""
Test HPC Web Interface imports step by step
"""
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    print("1. Testing basic imports...")
    import cv2
    import numpy as np
    from flask import Flask
    print("✅ Basic imports OK")
    
    print("2. Testing pipeline imports...")
    from pipeline.shared_web_interface import SharedWebInterface
    print("✅ SharedWebInterface OK")
    
    from pipeline.video_pose_pipeline import YOLOPoseModel, draw_overlay
    print("✅ YOLOPoseModel and draw_overlay OK")
    
    print("3. Testing HPCWebInterface import...")
    from pipeline.hpc_web_interface import HPCWebInterface
    print("✅ HPCWebInterface import OK")
    
    print("4. Testing HPCWebInterface initialization...")
    interface = HPCWebInterface(host='127.0.0.1', port=8767)
    print("✅ HPCWebInterface initialization OK")
    
    print("5. Testing video discovery...")
    videos = interface._get_available_videos()
    print(f"✅ Found {len(videos)} videos")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()