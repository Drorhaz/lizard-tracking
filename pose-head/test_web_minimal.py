#!/usr/bin/env python3
"""
Minimal test of web interface startup
"""
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    print("Testing imports...")
    from flask import Flask
    print("✅ Flask imported")
    
    from pipeline.shared_web_interface import SharedWebInterface
    print("✅ SharedWebInterface imported")
    
    # Test basic Flask app
    app = Flask(__name__)
    @app.route('/')
    def hello():
        return "Hello World"
    
    print("✅ Basic Flask app created")
    print("🚀 Starting minimal server...")
    
    app.run(host='0.0.0.0', port=8766, debug=False)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()