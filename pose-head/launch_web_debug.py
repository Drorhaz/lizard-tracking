#!/usr/bin/env python3
"""
Launch Lizard Pose Web Interface with debugging
Debug version with enhanced error reporting and verbose logging
"""
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    try:
        print("🚀 Starting Lizard Pose Web Interface")
        print("=" * 50)
        
        from pipeline.web_interface import HPCWebInterface
        
        # Start the web interface
        print("📱 Initializing interface...")
        interface = HPCWebInterface(host='0.0.0.0', port=8765)
        
        print("🌐 Starting server...")
        interface.start()
        
        print(f"✅ Web interface available at: http://0.0.0.0:8765/")
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
        
        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutting down web interface...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()