"""
Shared Global State for AI Vigilance.
Stores shared dictionaries and locks for multi-threaded access.
"""
import threading
from typing import Dict, Any

# ---------------------------------------------------------------------------
# GLOBAL SHARED MEMORY
# ---------------------------------------------------------------------------

# camera_results: Dict[str, Any]
# key: camera_id, value: {"rendered_frame": np.array, "frame_id": int, "recognized_name": str}
camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()  # Protects rendered frames for broadcasting

# camera_writers: Dict[str, Any]
# key: camera_id, value: {"writer": cv2.VideoWriter, "db_id": int}
camera_writers: Dict[str, Any] = {}
writer_lock = threading.Lock()   # Protects VideoWriter objects during recording

# active_search: Tracking specific individuals
active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()
