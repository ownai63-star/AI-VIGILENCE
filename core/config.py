"""
Core Configuration for AI Vigilance.
Stores shared constants, intervals, and processing limits.
"""
import os

# ---------------------------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = "snapshots"
DATASET_DIR = "dataset"
RECORDING_DIR = "recordings"

# Ensure directories exist
for d in [SNAPSHOT_DIR, DATASET_DIR, RECORDING_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# AI PERFORMANCE SETTINGS (Optimized for CPU at 60 FPS)
# ---------------------------------------------------------------------------
DETECTION_INTERVAL = 20      # YOLO Person detection (3x per sec at 60fps)
FACE_DETECTION_INTERVAL = 40  # MTCNN localized face finding
RECOGNITION_INTERVAL = 60     # FaceNet full biometric verification

# Image Sizes for AI processing (High-speed downscaling)
AI_INPUT_SIZE = 416           # Resolution used for YOLO detection

# ---------------------------------------------------------------------------
# TIMEOUTS AND CACHES
# ---------------------------------------------------------------------------
RECOGNITION_CACHE_FRAMES = 60  # Cache valid for 60 frames (~1 second)
FACE_CACHE_FRAMES = 60         # Increased for 60fps support
STALE_TRACK_TIMEOUT = 180      # Frames to wait before deleting track (3s)

# ---------------------------------------------------------------------------
# VIDEO RECORDING SETTINGS
# ---------------------------------------------------------------------------
RECORDING_FPS = 60.0           # Framerate for saved manual recordings
RECORDING_CODEC = 'VP80'       # WebM compatible codec
