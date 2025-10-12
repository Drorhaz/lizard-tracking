#!/usr/bin/env python3
"""
Shared Web Interface Module
Provides a singleton web interface that can be shared between HPC interface and pipeline
"""
import cv2
import threading
import time
from typing import Optional

class SharedWebInterface:
    """Singleton web interface for video streaming"""
    
    _instance: Optional['SharedWebInterface'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._jpeg = None
        self._frame_lock = threading.Lock()
        
    def update(self, bgr):
        """Update video stream with new frame (with pose overlays)"""
        ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self._frame_lock:
                self._jpeg = buf.tobytes()
    
    def get_frame(self):
        """Get latest frame for streaming"""
        with self._frame_lock:
            return self._jpeg
    
    def clear_frame(self):
        """Clear the current frame (stop streaming)"""
        with self._frame_lock:
            self._jpeg = None
            print("🔄 Video stream cleared")
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        return cls()

# Global function for pipeline to use
def update_web_frame(frame):
    """Update web interface with new frame"""
    web = SharedWebInterface.get_instance()
    web.update(frame)