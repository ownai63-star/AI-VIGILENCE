"""
AI VIGILANCE PROCESSING ENGINE
Responsible for real-time person detection, tracking, and recognition across 
multiple cameras in dedicated threads.
"""
import cv2
import time
import threading
from typing import Dict, Any

# CONFIGS & SHARED STATE
from core.config import (
    DETECTION_INTERVAL, FACE_DETECTION_INTERVAL, RECOGNITION_INTERVAL,
    AI_INPUT_SIZE, STALE_TRACK_TIMEOUT, FACE_CACHE_FRAMES
)
from core.state import camera_results, results_lock, camera_writers, writer_lock

# AI UTILS
from database.db_manager import DatabaseManager
from utils.detector import PersonDetector
from utils.tracker import ObjectTracker
from utils.recognizer import FaceRecognizer
from cameras.camera_manager import CameraManager

# Singleton AI Model and Management instances shared across threads
db_manager = DatabaseManager()      # Database (SQLite)
detector = PersonDetector()          # YOLOv8 Person detection
recognizer = FaceRecognizer()        # FaceNet Facial Identification
camera_manager = CameraManager()    # Global Camera Handler
recognizer.load_known_faces(db_manager) 

def start_camera_engine(camera_id: str, camera_manager: CameraManager):
    """
    Launches a dedicated background thread for a specific camera feed.
    """
    thread = threading.Thread(target=process_camera, args=(camera_id, camera_manager), daemon=True)
    thread.start()
    return thread

def process_camera(camera_id: str, camera_manager: CameraManager):
    """
    Main background thread for AI processing on a specific camera.
    - YOLOv8 Detection (Person)
    - DeepSORT Tracking (IDs)
    - FaceNet Recognition (MTCNN + InceptionResnet)
    """
    print(f"[CoreAI] Engine launched for: {camera_id}")
    tracker: Any = ObjectTracker()
    last_frame_id: int = -1
    frame_count: int = 0
    
    # Tracking and cache state
    face_bbox_cache: Dict[Any, tuple] = {}
    track_states: Dict[Any, dict] = {}
    unknown_snapped = set()
    last_detections = []

    while True:
        # 1. Acquire latest frame
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.001) # Avoid 100% CPU usage on loop spinning
            continue
            
        last_frame_id = frame_id
        frame_count += 1

        try:
            h, w = frame.shape[:2]  # Typical 1080p frame
            
            run_detection = (frame_count % DETECTION_INTERVAL == 1)
            run_face_detection = (frame_count % FACE_DETECTION_INTERVAL == 1)
            run_recognition = (frame_count % RECOGNITION_INTERVAL == 1)

            # ---------------------------------------------------------------
            # STAGE 1: PERSON DETECTION (YOLO) & TRACKING (DeepSORT)
            # ---------------------------------------------------------------
            current_detections = []
            if run_detection:
                ai_frame = cv2.resize(frame, (AI_INPUT_SIZE, AI_INPUT_SIZE))
                detections = detector.detect(ai_frame, min_box_size=5)
                
                for det in detections:
                    box, conf, cls = det
                    x, y, wb, hb = box
                    # Map from 416p back to original resolution (e.g. 1080p)
                    scaled_box = [x * (w / AI_INPUT_SIZE), y * (h / AI_INPUT_SIZE), wb * (w / AI_INPUT_SIZE), hb * (h / AI_INPUT_SIZE)]
                    current_detections.append((scaled_box, conf, cls))
            
            # CRITICAL: Always update tracker (even with empty list) to predict next positions
            # This ensures smooth boxes at 60-90 FPS even if YOLO runs every 20 frames.
            tracks = tracker.update(current_detections, frame)
            
            # Maintain active track states for rendering and face-finding
            current_ids = set()
            for t in tracks:
                tid = t["id"]
                current_ids.add(tid)
                new_bbox = t["bbox"] # Current (predicted) box
                
                # Active mark and state refresh
                track_states[tid] = {
                    "bbox": new_bbox, 
                    "last_update": frame_count, 
                    "active": True
                }
            
            # CRITICAL: Mark tracks as inactive if they are no longer being tracked in this frame
            for tid in track_states:
                if tid not in current_ids:
                    track_states[tid]["active"] = False
            
            # Cleanup stale tracks (not seen for a while)
            stale_ids = [tid for tid, s in track_states.items() if (frame_count - s["last_update"]) > STALE_TRACK_TIMEOUT]
            for tid in stale_ids:
                del track_states[tid]
                face_bbox_cache.pop(tid, None)
                unknown_snapped.discard(tid)

            # ---------------------------------------------------------------
            # STAGE 2: FACE FINDING
            # ---------------------------------------------------------------
            if run_face_detection:
                for tid, state in list(track_states.items()):
                    if not state["active"]: continue
                    
                    bbox = state["bbox"]
                    bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    try:
                        crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                        if crop.size > 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            with recognizer.ai_lock:
                                boxes, _ = recognizer.mtcnn.detect(crop_rgb)
                            
                            if boxes is not None:
                                fx1, fy1, fx2, fy2 = boxes[0]
                                face_box = [max(0, bx1+int(fx1)), max(0, by1+int(fy1)), min(w, bx1+int(fx2)), min(h, by1+int(fy2))]
                                face_bbox_cache[tid] = (face_box, frame_count)
                    except Exception: pass
            
            # ---------------------------------------------------------------
            # STAGE 3: RECOGNITION
            # ---------------------------------------------------------------
            recognized_person = None
            if run_recognition:
                for tid, state in track_states.items():
                    if not state["active"] or (tid not in face_bbox_cache): continue
                    
                    face_box, _ = face_bbox_cache[tid]
                    name, conf = recognizer.recognize(frame, face_box)
                    
                    if name != "Unknown" and conf > 0.35:
                        recognized_person = name
                    else:
                        if tid not in unknown_snapped:
                            unknown_snapped.add(tid)
                            ts = int(time.time()); snap = f"snapshots/unknown_{camera_id}_{tid}_{ts}.jpg"
                            cv2.imwrite(snap, frame)
                            db_manager.log_detection(None, camera_id, snap)

            # ---------------------------------------------------------------
            # STAGE 4: RENDERING & BUFFERING
            # ---------------------------------------------------------------
            record_frame = frame.copy()
            for tid, s in track_states.items():
                if not s["active"]: continue
                b = s["bbox"]
                cv2.rectangle(record_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(record_frame, f"Person {tid}", (int(b[0]), int(b[1]-10)), 0, 0.6, (0, 255, 0), 2)

            with results_lock:
                camera_results[camera_id] = {
                    "rendered_frame": record_frame, 
                    "frame_id": frame_id,
                    "recognized_name": recognized_person if recognized_person else camera_results.get(camera_id, {}).get("recognized_name")
                }

            with writer_lock:
                writer_data = camera_writers.get(camera_id)
                if writer_data:
                    writer_data["writer"].write(record_frame)

        except Exception as e:
            print(f"[ErrorAI] {camera_id}: {e}")
