#!/usr/bin/env python3
"""
Launch Lizard Pose Web Interface
Unified web interface for lizard pose detection with multiple execution modes:
- Local CPU processing
- Local GPU processing  
- HPC cluster GPU jobs
"""
import sys
from pathlib import Path

# Add current directory to path so we can import from pipeline/
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pipeline.web_interface import HPCWebInterface

def main():
    print("🚀 Starting Lizard Pose Web Interface")
    print("=" * 50)
    
    # Start the web interface
    interface = HPCWebInterface(host='0.0.0.0', port=8765)
    interface.start()
    
    print(f"🌐 Web interface available at: http://0.0.0.0:8765/")
    print("📝 Features:")
    print("  - Local CPU/GPU processing")
    print("  - HPC cluster GPU jobs")
    print("  - Real-time pose detection monitoring")
    print("  - Live video stream from inference")
    print("  - Job status and performance metrics")
    print("\n💡 Usage:")
    print("  1. Select a video file from the dropdown")
    print("  2. Choose execution mode: Local CPU, Local GPU, or HPC Cluster")
    print("  3. Choose detection mode: Live inference or Offline playback")
    print("  4. Click 'Start Pipeline' to begin processing")
    print("  5. Monitor progress and view live results")
    print("\n🔗 Access from browser: http://<your-ip>:8765/")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down web interface...")

if __name__ == "__main__":
    main()