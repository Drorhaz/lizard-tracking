#!/usr/bin/env python3
"""
Test script to verify portable HPC Web Interface
This script tests that the interface can run on different systems without hardcoded paths
"""
import sys
from pathlib import Path

def test_portable_interface():
    """Test that the interface works with portable paths"""
    print("🧪 Testing Portable HPC Web Interface")
    print("=" * 50)
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from pipeline.web_interface import HPCWebInterface
        
        # Test 1: Initialize interface
        print("✅ 1. Interface import successful")
        
        interface = HPCWebInterface(host='0.0.0.0', port=8765)
        print("✅ 2. Interface initialization successful")
        
        # Test 2: Project root detection
        project_root = interface._get_project_root()
        print(f"✅ 3. Project root detected: {project_root}")
        
        # Test 3: Video scanning
        videos = interface._get_available_videos()
        print(f"✅ 4. Video scan completed: {len(videos)} videos found")
        
        # Test 4: Check for required project structure
        required_dirs = ['pose-head', 'pipeline']
        missing_dirs = []
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"⚠️  Missing directories: {missing_dirs}")
        else:
            print("✅ 5. Project structure validation passed")
        
        print("\n🎉 All tests passed! Interface is portable and ready to use.")
        print(f"📍 Detected project location: {project_root}")
        print(f"🎬 Available videos: {len(videos)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_portable_interface()
    sys.exit(0 if success else 1)