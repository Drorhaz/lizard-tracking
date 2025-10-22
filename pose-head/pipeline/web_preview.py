#!/usr/bin/env python3
from __future__ import annotations
import io, threading, time
from typing import Optional
import cv2, numpy as np
from flask import Flask, Response, render_template_string
HTML = """<!doctype html><html><head><meta charset='utf-8'><title>Pose Overlay</title>
<style>body{background:#111;margin:0}img{max-width:98vw;max-height:95vh;display:block;margin:1vh auto;border-radius:10px}</style>
</head><body><img src="/video"></body></html>"""
class WebPreview:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host=host; self.port=port
        self.app=Flask(__name__)
        self._jpeg=None; self._lock=threading.Lock(); self._th=None
        @self.app.route('/')
        def idx(): return render_template_string(HTML)
        @self.app.route('/video')
        def video():
            return Response(self._gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    def update(self, bgr):
        ok,buf=cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY),85])
        if ok:
            with self._lock: self._jpeg=buf.tobytes()
    def _gen(self):
        while True:
            with self._lock: j=self._jpeg
            if j is None: time.sleep(0.03); continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+j+b'\r\n'
            time.sleep(0.03)
    def start(self):
        if self._th and self._th.is_alive(): return
        def run(): self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)
        self._th=threading.Thread(target=run, daemon=True); self._th.start()
