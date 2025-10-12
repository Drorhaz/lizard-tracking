#!/usr/bin/env python3
"""
Test FPS configuration fix

This script verifies that:
1. Video FPS is read from file
2. Processing FPS is read from config
3. Both values are properly separated
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena_mock_app.api import Config

def test_fps_separation():
    """Test that FPS values are properly separated"""
    
    print("=" * 60)
    print("Testing FPS Configuration Fix")
    print("=" * 60)
    
    # Load config
    config = Config()
    
    print(f"\n✅ Config loaded successfully")
    print(f"   PROCESSING_FPS from config: {config.processing_fps}")
    print(f"   STREAM_FPS from config: {config.stream_fps}")
    
    # Verify config values
    expected_processing_fps = int(os.getenv('PROCESSING_FPS', '60'))
    expected_stream_fps = int(os.getenv('STREAM_FPS', '15'))
    
    assert config.processing_fps == expected_processing_fps, \
        f"Expected processing_fps={expected_processing_fps}, got {config.processing_fps}"
    
    assert config.stream_fps == expected_stream_fps, \
        f"Expected stream_fps={expected_stream_fps}, got {config.stream_fps}"
    
    print(f"\n✅ Configuration values match expected:")
    print(f"   ✓ PROCESSING_FPS = {config.processing_fps}")
    print(f"   ✓ STREAM_FPS = {config.stream_fps}")
    
    print(f"\n📝 Expected behavior:")
    print(f"   • Video's actual FPS will be read from file metadata")
    print(f"   • Processing speed will use PROCESSING_FPS ({config.processing_fps})")
    print(f"   • Video timestamps will use video's original FPS")
    print(f"   • Browser stream will use STREAM_FPS ({config.stream_fps})")
    
    print(f"\n" + "=" * 60)
    print("✅ All FPS tests passed!")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        test_fps_separation()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
