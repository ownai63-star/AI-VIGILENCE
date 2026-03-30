import cv2
import threading
import time
import os

# Force OpenCV to use UDP and drop delay for RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

class CameraHandler:
    def __init__(self, camera_id, source):
        self.camera_id = camera_id
        self.source = source
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Force low-latency and no buffering (crucial for 60-90 FPS feel)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        fails = 0
        while self.running:
            # Capture frame; use grab/retrieve for better performance on some streams
            if not self.cap.grab():
                time.sleep(0.1)
                fails += 1
                if fails > 100:
                    # Reconnect logic for RTSP
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fails = 0
                continue
            
            ret, frame = self.cap.retrieve()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
            fails = 0

    def get_frame(self):
        with self.lock:
            return self.frame if self.frame is not None else None

    def get_frame_with_id(self):
        with self.lock:
            return (self.frame, self.frame_id) if self.frame is not None else (None, 0)

    def stop(self):
        self.running = False
        self.cap.release()

from typing import Dict, Any

class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, Any] = {}

    def add_camera(self, camera_id, source):
        if camera_id not in self.cameras:
            handler = CameraHandler(camera_id, source)
            self.cameras[camera_id] = handler
            return True
        return False

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            self.cameras.pop(camera_id, None)
            return True
        return False

    def get_camera_frame(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame()
        return None
        
    def get_camera_frame_with_id(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame_with_id()
        return None, 0

    def get_active_cameras(self):
        return list(self.cameras.keys())
