import cv2
import threading
import time
import os
import sys

# Force OpenCV to use UDP and drop delay for RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

# Suppress OpenCV GUI warnings on headless Linux
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("DISPLAY", ":0")

# Common RTSP stream paths to try when a bare IP is given (no path after port)
RTSP_PROBE_PATHS = [
    "",                                      # bare — try as-is first
    "/axis-media/media.amp",                 # Axis
    "/Streaming/Channels/101",               # Hikvision main
    "/Streaming/Channels/1",                 # Hikvision alt
    "/cam/realmonitor?channel=1&subtype=0",  # Dahua
    "/stream1",                              # Generic
    "/live/ch00_0",                          # Generic
    "/h264/ch1/main/av_stream",              # Generic Hikvision
    "/onvif-media/media.amp",                # ONVIF
]

def probe_rtsp_url(url: str) -> str:
    """
    If the RTSP URL has no path (just host:port), try common stream paths
    and return the first one that opens successfully. Returns original if none work.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    # If there's already a meaningful path, don't probe
    if parsed.path and parsed.path not in ("", "/"):
        return url

    base = url.rstrip("/")
    for path in RTSP_PROBE_PATHS:
        candidate = base + path
        cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        opened = cap.isOpened()
        cap.release()
        if opened:
            print(f"[CameraProbe] Found working path: {candidate}")
            return candidate

    print(f"[CameraProbe] No working path found for {url}, using as-is")
    return url

class CameraHandler:
    def __init__(self, camera_id, source):
        self.camera_id = camera_id
        self.source = source
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Force low-latency and no buffering (crucial for smooth live streaming)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Request 30 FPS from camera
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()
        # Use higher priority thread for video capture
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Capture frames as fast as possible for smooth video."""
        fails = 0
        while self.running:
            # Read frame - don't wait, keep reading for latest frame
            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails > 100:
                    # Reconnect logic for RTSP
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    fails = 0
                continue
            
            # Always store latest frame - no delay
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
            # For bare RTSP URLs, auto-discover the correct stream path
            if isinstance(source, str) and source.startswith("rtsp://"):
                source = probe_rtsp_url(source)
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
