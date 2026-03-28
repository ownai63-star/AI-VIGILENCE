import cv2
import numpy as np
import os
import shutil
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from database.db_manager import DatabaseManager
from utils.detector import PersonDetector
from utils.tracker import ObjectTracker
from utils.recognizer import FaceRecognizer
from cameras.camera_manager import CameraManager
import threading
import time
import urllib.parse
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_rtsp_url(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url
    url_str = str(url)
    last_at = url_str.rfind("@")
    if last_at == -1:
        return url_str
    auth_part = url_str[7:last_at]
    if ":" in auth_part:
        user, pwd = auth_part.split(":", 1)
        safe_pwd = urllib.parse.quote(pwd)
        return f"rtsp://{user}:{safe_pwd}{url_str[last_at:]}"
    return url_str

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")
# Configure Jinja2 templates with no cache to avoid errors
from jinja2 import FileSystemLoader, Environment as JinjaEnvironment
template_env = JinjaEnvironment(loader=FileSystemLoader("templates"), auto_reload=True)
templates = Jinja2Templates(env=template_env)

db_manager = DatabaseManager()
detector = PersonDetector()
recognizer = FaceRecognizer()
camera_manager = CameraManager()
recognizer.load_known_faces(db_manager)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Per-camera: latest tracks for video overlay
camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()  # Single shared lock for camera_results

# Recording state
camera_writers: Dict[str, Any] = {}
writer_lock = threading.Lock()
occupancy_last_count: Dict[str, int] = {}

# Active search mission — set by /api/start_search, cleared by /api/stop_search
# {person_id, name, encoding, found_track_ids: set, running: bool}
active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Passive camera processing — ONLY detection + tracking, NO recognition
# ---------------------------------------------------------------------------

def process_camera(camera_id: str):
    """Background thread per camera: detection + tracking + face recognition.
    
    Strategy:
    - Heavy processing (detection, face detection, recognition) runs every N frames
    - DeepSORT tracker predicts positions between detection frames
    - Rendering happens every frame at 60 FPS with interpolated positions
    """
    print(f"[process_camera] Thread started for: {camera_id}")
    tracker: Any = ObjectTracker()
    last_frame_id: int = -1
    frame_count: int = 0
    
    # Processing intervals (heavier CPU utilization per request)
    DETECTION_INTERVAL = 1       # Run YOLO detection every frame
    RECOGNITION_INTERVAL = 30     # Run face recognition periodically
    
    # Recognition cache: track_id -> (name, confidence, frame_number)
    RECOGNITION_CACHE_FRAMES = 60  # Cache valid for 60 frames (~1 second at 60 FPS)
    recognition_cache: Dict[Any, tuple] = {}  # track_id -> (name, conf, frame_num)
    
    # Unique people seen on this camera (cumulative unique track IDs)
    seen_track_ids = set()
    
    # Track state for smooth rendering
    # track_id -> {"bbox": [x1,y1,x2,y2], "velocity": [vx,vy,vx,vy], "last_update": frame_num}
    track_states: Dict[Any, dict] = {}
    
    # Last detection results
    last_detections = []
    last_detection_frame = 0

    while True:
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None or frame_id == last_frame_id:
            continue  # No sleep - keep looping for max FPS
        last_frame_id = frame_id
        frame_count += 1

        try:
            h, w = frame.shape[:2]
            
            # Determine which heavy operations to run this frame
            run_detection = (frame_count % DETECTION_INTERVAL == 0)
            run_recognition = (frame_count % RECOGNITION_INTERVAL == 0)

            if run_detection:
                # Heavy: YOLO person detection + DeepSORT tracking update
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)
                last_detections = tracks
                last_detection_frame = frame_count
                
                # Update track states with new detections
                current_ids = set()
                for t in tracks:
                    track_id = t["id"]
                    current_ids.add(track_id)
                    if track_id not in seen_track_ids:
                        seen_track_ids.add(track_id)
                    new_bbox = t["bbox"]
                    
                    if track_id in track_states:
                        old_bbox = track_states[track_id]["bbox"]
                        # Calculate velocity for prediction
                        velocity = [
                            new_bbox[0] - old_bbox[0],
                            new_bbox[1] - old_bbox[1],
                            new_bbox[2] - old_bbox[2],
                            new_bbox[3] - old_bbox[3]
                        ]
                        track_states[track_id] = {
                            "bbox": new_bbox,
                            "velocity": velocity,
                            "last_update": frame_count,
                            "active": True
                        }
                    else:
                        track_states[track_id] = {
                            "bbox": new_bbox,
                            "velocity": [0, 0, 0, 0],
                            "last_update": frame_count,
                            "active": True
                        }
                
                # Mark tracks not detected as inactive
                for tid in track_states:
                    if tid not in current_ids:
                        track_states[tid]["active"] = False
                
                # Remove very stale tracks (not updated for 90 frames)
                stale_ids = [tid for tid, state in track_states.items() 
                            if not state["active"] and (frame_count - state["last_update"]) > 90]
                for tid in stale_ids:
                    del track_states[tid]
                    recognition_cache.pop(tid, None)
            
            # Predict positions for all tracks using velocity (Kalman filter-like)
            frames_since_detection = frame_count - last_detection_frame
            if frames_since_detection > 0:
                for tid, state in track_states.items():
                    if state["active"]:
                        # Predict new position based on velocity
                        v = state["velocity"]
                        old_bbox = state["bbox"]
                        # Apply velocity with damping to reduce drift
                        damping = 0.95 ** frames_since_detection
                        predicted_bbox = [
                            old_bbox[0] + v[0] * damping,
                            old_bbox[1] + v[1] * damping,
                            old_bbox[2] + v[2] * damping,
                            old_bbox[3] + v[3] * damping
                        ]
                        state["predicted_bbox"] = predicted_bbox
                    else:
                        state["predicted_bbox"] = state["bbox"]
            else:
                for state in track_states.values():
                    state["predicted_bbox"] = state["bbox"]

            # No face detection stage — remove face tracking entirely

            # Face recognition (periodic)
            if run_recognition:
                for tid, state in track_states.items():
                    if not state["active"]:
                        continue
                    # Check recognition cache staleness
                    cached = recognition_cache.get(tid)
                    if cached:
                        cached_name, cached_conf, cached_frame = cached
                        cache_age = frame_count - cached_frame
                        if cache_age < RECOGNITION_CACHE_FRAMES:
                            continue
                    # Approximate face region from the body bbox (upper portion)
                    bb = state.get("predicted_bbox", state["bbox"])
                    bx1, by1, bx2, by2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w, bx2), min(h, by2)
                    bw = max(0, bx2 - bx1)
                    bh = max(0, by2 - by1)
                    if bw < 20 or bh < 20:
                        continue
                    fx1 = bx1 + int(0.15 * bw)
                    fx2 = bx2 - int(0.15 * bw)
                    fy1 = by1
                    fy2 = by1 + int(0.45 * bh)
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2, fy2 = min(w, fx2), min(h, fy2)
                    if fx2 - fx1 < 20 or fy2 - fy1 < 20:
                        continue
                    try:
                        name, conf = recognizer.recognize(frame, [fx1, fy1, fx2, fy2])
                        if name != "Unknown" and conf > 0.35:
                            recognition_cache[tid] = (name, conf, frame_count)
                        elif tid in recognition_cache:
                            pass
                    except Exception:
                        pass

            # Build render data from track states
            processed = []
            for tid, state in track_states.items():
                bbox = state["predicted_bbox"]
                bx1 = max(0, int(bbox[0]))
                by1 = max(0, int(bbox[1]))
                bx2 = min(w - 1, int(bbox[2]))
                by2 = min(h - 1, int(bbox[3]))
                
                box_w = bx2 - bx1
                box_h = by2 - by1
                if box_w < 20 or box_h < 20:
                    continue
                if box_w > w * 0.95 or box_h > h * 0.95:
                    continue

                # Get recognition result
                name, conf = "Unknown", 0.0
                if tid in recognition_cache:
                    cached_name, cached_conf, cached_frame = recognition_cache[tid]
                    if (frame_count - cached_frame) < RECOGNITION_CACHE_FRAMES:
                        name, conf = cached_name, cached_conf

                processed.append({
                    "id": tid,
                    "bbox": [bx1, by1, bx2, by2],
                    "name": name,
                    "confidence": conf
                })

            # Active search check
            with active_search_lock:
                search = dict(active_search)

            if search.get("running") and run_recognition:
                for t in processed:
                    track_key = (camera_id, t["id"])
                    if track_key not in search.get("found_track_ids", set()):
                        if t["name"] == "Unknown":
                            bx1, by1, bx2, by2 = t["bbox"]
                            bw = max(0, bx2 - bx1)
                            bh = max(0, by2 - by1)
                            fx1 = bx1 + int(0.15 * bw)
                            fx2 = bx2 - int(0.15 * bw)
                            fy1 = by1
                            fy2 = by1 + int(0.45 * bh)
                            fx1, fy1 = max(0, fx1), max(0, fy1)
                            fx2, fy2 = min(frame.shape[1]-1, fx2), min(frame.shape[0]-1, fy2)
                            face_box_guess = [fx1, fy1, fx2, fy2]
                            threading.Thread(
                                target=recognition_worker,
                                args=(frame.copy(), face_box_guess, t["id"], camera_id, recognition_cache),
                                daemon=True
                            ).start()

            # Render at full rate - every frame gets overlay
            record_frame = frame.copy()
            people_count = 0
            for t in processed:
                bx1, by1, bx2, by2 = t["bbox"]
                name = str(t["name"])
                conf = float(t["confidence"])
                tid = str(t["id"])

                if name != "Unknown":
                    body_color = (0, 255, 0)  # Green
                    label = f"{name} ({conf:.2f})"
                else:
                    body_color = (0, 165, 255)  # Orange
                    label = f"Person {tid}"
                people_count += 1
                # Thicker lines for better visibility
                cv2.rectangle(record_frame, (bx1, by1), (bx2, by2), body_color, 3)
                cv2.putText(record_frame, label, (bx1, max(by1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, body_color, 2)
            if occupancy_last_count.get(camera_id) != people_count:
                occupancy_last_count[camera_id] = people_count
                try:
                    db_manager.log_occupancy(camera_id, people_count)
                except Exception:
                    pass
            cv2.putText(record_frame, f"Count: {people_count}  Total: {len(seen_track_ids)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Output at full frame rate
            with results_lock:
                camera_results[camera_id] = {"rendered_frame": record_frame, "frame_id": frame_id, "tracks": processed}

            # Write to recording
            with writer_lock:
                writer_data = camera_writers.get(camera_id)
                if writer_data and writer_data.get("writer"):
                    writer_data["writer"].write(record_frame)

        except Exception as e:
            print(f"[process_camera:{camera_id}] {e}")
            import traceback; traceback.print_exc()


def recognition_worker(frame, face_bbox, track_id, camera_id, recognition_cache):
    """
    Background recognition for active search using exact face bbox.
    Updates the recognition cache when a match is found.
    """
    with active_search_lock:
        if not active_search.get("running"):
            return
        target_encoding = active_search.get("encoding")
        target_name = active_search.get("name")
        target_person_id = active_search.get("person_id")
        found_ids = active_search.get("found_track_ids", set())
        track_key = (camera_id, track_id)
        if track_key in found_ids:
            return

    # Run face recognition on the face box
    name, confidence = recognizer.recognize(frame, face_bbox)

    if name == target_name and confidence > 0.35:  # Lower threshold for better detection
        with active_search_lock:
            if not active_search.get("running"):
                return
            if track_key in active_search.get("found_track_ids", set()):
                return
            active_search["found_track_ids"].add(track_key)

        # Update recognition cache so it appears on live feed immediately
        # Format: (name, confidence, frame_number, miss_count)
        recognition_cache[track_id] = (target_name, confidence, 0, 0)

        # Take ONE snapshot
        timestamp = int(time.time())
        snap_name = f"snap_{camera_id}_{track_id}_{timestamp}.jpg"
        snap_path = f"snapshots/{snap_name}"
        cv2.imwrite(snap_path, frame)
        db_manager.log_detection(target_person_id, camera_id, snap_path)
        print(f"[ActiveSearch] Found {target_name} on {camera_id} — snapshot saved")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Paginated feeds
    try:
        page = int(request.query_params.get("page", "1"))
        per_page = int(request.query_params.get("per_page", "4"))
    except Exception:
        page, per_page = 1, 4
    cameras_all = camera_manager.get_active_cameras()
    total = len(cameras_all)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    cameras = cameras_all[start:end]
    return templates.TemplateResponse("index.html", {"request": request, "cameras": cameras, "page": page, "total_pages": total_pages, "per_page": per_page, "total": total})


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    return templates.TemplateResponse("recordings.html", {"request": request})

@app.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    return templates.TemplateResponse("people.html", {"request": request})

@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    return templates.TemplateResponse("cameras.html", {"request": request})
@app.post("/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    img_dir = f"dataset/{name}"
    os.makedirs(img_dir, exist_ok=True)
    file_path = f"{img_dir}/{file.filename}"
    with open(file_path, "wb") as buf:
        buf.write(await file.read())

    image = cv2.imread(file_path)
    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        db_manager.register_person(name, file_path, encoding.tobytes())
        recognizer.load_known_faces(db_manager)
        return {"status": "success", "message": f"{name} registered."}
    return {"status": "error", "message": "No face detected in the image."}


@app.post("/add_camera")
async def add_camera(camera_id: str = Form(...), camera_type: str = Form(...), source: str = Form(...)):
    parsed = source
    if camera_type == "webcam":
        try:
            parsed = int(source)
        except ValueError:
            pass
    elif camera_type == "rtsp":
        parsed = sanitize_rtsp_url(source)
    elif camera_type == "droidcam":
        if not source.startswith("http"):
            parsed = f"http://{source}:4747/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "ipwebcam":
        if not source.startswith("http"):
            parsed = f"http://{source}:8080/video" if ":" not in source else f"http://{source}/video"

    if camera_manager.add_camera(camera_id, parsed):
        threading.Thread(target=process_camera, args=(camera_id,), daemon=True).start()
        return {"status": "success"}
    return {"status": "error", "message": "Camera already exists or could not connect."}


@app.post("/delete_camera")
async def delete_camera(camera_id: str = Form(...)):
    if camera_manager.remove_camera(camera_id):
        camera_results.pop(camera_id, None)
        return {"status": "success"}
    return {"status": "error"}


@app.get("/api/cameras")
async def api_cameras():
    return camera_manager.get_active_cameras()

@app.get("/api/occupancy")
async def api_occupancy(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    rows = db_manager.search_occupancy(camera_id, start_time, end_time)
    return [{"id": r[0], "camera_id": r[1], "timestamp": r[2], "count": r[3]} for r in rows]

# ---------------------------------------------------------------------------
# Recording API
# ---------------------------------------------------------------------------
@app.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    with writer_lock:
        if camera_id in camera_writers:
            # Stop recording
            writer_data = camera_writers.pop(camera_id)
            writer_data["writer"].release()
            db_manager.end_recording(writer_data["db_id"])
            return {"status": "success", "recording": False}
        else:
            # Start recording
            with results_lock:
                data = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")
            if frame is None:
                return {"status": "error", "message": "Camera offline or warming up"}
                
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            timestamp = int(time.time())
            filename = f"rec_{camera_id}_{timestamp}.webm"
            file_path = f"recordings/{filename}"
            writer = cv2.VideoWriter(file_path, fourcc, 20.0, (w, h))
            
            db_id = db_manager.start_recording(camera_id, file_path)
            camera_writers[camera_id] = {"writer": writer, "db_id": db_id}
            return {"status": "success", "recording": True}

@app.get("/api/recording_status")
async def get_recording_status():
    with writer_lock:
        return {"active_recordings": list(camera_writers.keys())}


# ---------------------------------------------------------------------------
# Active Search API
# ---------------------------------------------------------------------------

@app.post("/api/start_search")
async def start_search(name: str = Form(...)):
    """Start an active face-search mission for the given person."""
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"'{name}' is not registered."}

    encoding = np.frombuffer(target[3], dtype=np.float32)
    with active_search_lock:
        active_search.clear()
        active_search.update({
            "running": True,
            "person_id": target[0],
            "name": target[1],
            "encoding": encoding,
            "found_track_ids": set()
        })
    print(f"[ActiveSearch] Mission started for: {target[1]}")
    return {
        "status": "success",
        "message": f"Searching for {target[1]}",
        "name": target[1],
        "image_path": target[2]  # registered photo from dataset/
    }


@app.post("/api/stop_search")
async def stop_search():
    """Stop the active search mission."""
    with active_search_lock:
        active_search.clear()
    print("[ActiveSearch] Mission stopped.")
    return {"status": "success"}


@app.get("/api/active_search")
async def get_active_search():
    """Return current active search target (if any)."""
    with active_search_lock:
        name = active_search.get("name")
    return {"active": name is not None, "name": name}


# ---------------------------------------------------------------------------
# History Search API
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def api_search(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_detections(name, start_time, end_time)
    return [{"id": r[0], "person_name": r[5] or "Unknown", "camera_id": r[2], "timestamp": r[3], "image_path": r[4]} for r in results]

@app.post("/api/search_by_image")
async def search_by_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    encoding = recognizer.get_encoding(image)
    if encoding is None:
        return []

    best_person_id = None
    min_dist = 1.0
    for p in db_manager.get_registered_persons():
        if p[3] is not None:
            db_enc = np.frombuffer(p[3], dtype=np.float32)
            dist = float(np.linalg.norm(db_enc - encoding))
            if dist < min_dist:
                min_dist = dist
                best_person_id = p[0]

    if best_person_id is None:
        return []

    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT d.*, rp.name FROM detections d
                          LEFT JOIN registered_persons rp ON d.person_id = rp.id
                          WHERE d.person_id = ? ORDER BY d.timestamp DESC''', (best_person_id,))
        results = cursor.fetchall()
    return [{"id": r[0], "person_name": r[5] or "Unknown", "camera_id": r[2], "timestamp": r[3], "image_path": r[4]} for r in results]


@app.post("/clear_history")
async def clear_history():
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections")
            conn.commit()
    except Exception as e:
        print(f"DB clear error: {e}")

    snaps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    deleted: int = 0
    if os.path.isdir(snaps_dir):
        for fname in os.listdir(snaps_dir):
            try:
                os.remove(os.path.join(snaps_dir, fname))
                deleted = int(deleted) + 1
            except Exception:
                pass

    print(f"Cleared {deleted} snapshots.")
    return {"status": "success", "message": f"Cleared {deleted} records"}

@app.get("/api/recordings")
async def api_recordings(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_recordings(camera_id, start_time, end_time)
    return [{"id": r[0], "camera_id": r[1], "start_time": r[2], "end_time": r[3], "file_path": r[4]} for r in results]

@app.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: int):
    rec = db_manager.get_recording(record_id)
    if rec:
        try:
            os.remove(rec[4])
        except Exception:
            pass
        db_manager.delete_recording(record_id)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Video Person Search API
# ---------------------------------------------------------------------------

import json
from fastapi import BackgroundTasks

# Store video search progress
video_search_progress: Dict[str, Any] = {}
video_search_lock = threading.Lock()

@app.get("/api/persons")
async def api_persons():
    """Get all registered persons for dropdown selection."""
    persons = db_manager.get_registered_persons()
    return [{"id": p[0], "name": p[1], "image_path": p[2]} for p in persons]


def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 10) -> list:
    """
    Scan a video file for ALL occurrences of a person with the target face encoding.
    Detects every face in each frame and matches against the target person.
    Groups continuous appearances into flagged segments with start/end timestamps.
    Returns list of detection segments where the person appears.
    """
    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoScan] ERROR: Could not open video {video_path}")
        return results
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Track continuous appearances
    current_segment = None
    last_match_frame = -1
    min_segment_gap = int(fps * 2)  # 2 seconds gap to create new segment
    
    # Lower threshold for better detection (same as live recognition)
    DISTANCE_THRESHOLD = 1.15
    
    print(f"[VideoScan] Starting scan of {video_path}")
    print(f"[VideoScan] Total frames: {total_frames}, FPS: {fps}, Sample interval: {sample_interval}")
    
    matches_found = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame for efficiency
        if frame_count % sample_interval == 0:
            try:
                # Detect ALL faces in frame using full frame (not just body crop)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with recognizer.ai_lock:
                    boxes, probs = recognizer.mtcnn.detect(frame_rgb)
                
                match_found = False
                best_confidence = 0.0
                best_distance = 999.0
                
                if boxes is not None and len(boxes) > 0:
                    # Check EACH face in the frame against target
                    for i, box in enumerate(boxes):
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        
                        # Ensure valid box
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(frame.shape[1], fx2), min(frame.shape[0], fy2)
                        
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fw < 30 or fh < 30:  # Skip very small faces
                            continue
                        
                        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
                        
                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (160, 160))
                            face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(recognizer.device)
                            face_tensor = (face_tensor - 127.5) / 128.0
                            
                            with recognizer.ai_lock:
                                with torch.no_grad():
                                    embedding = recognizer.resnet(face_tensor).cpu().numpy()[0]
                            
                            # Compare with target
                            distance = float(np.linalg.norm(target_encoding - embedding))
                            confidence = 1 - (distance / 2.0)
                            
                            if distance < DISTANCE_THRESHOLD:  # Match found
                                match_found = True
                                matches_found += 1
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_distance = distance
                                if frame_count % 100 == 0:  # Log every 100th match frame
                                    print(f"[VideoScan] Match at frame {frame_count}, dist: {distance:.3f}, conf: {confidence:.2f}")
                
                # Handle segment tracking
                if match_found:
                    timestamp_sec = frame_count / fps
                    
                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        # Start new segment
                        if current_segment is not None:
                            results.append(current_segment)
                        current_segment = {
                            "start_seconds": timestamp_sec,
                            "start_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "end_seconds": timestamp_sec,
                            "end_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "confidence": best_confidence,
                            "start_frame": frame_count,
                            "end_frame": frame_count
                        }
                        print(f"[VideoScan] New segment started at {current_segment['start_timestamp']}")
                    else:
                        # Extend current segment
                        current_segment["end_seconds"] = timestamp_sec
                        current_segment["end_timestamp"] = f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}"
                        current_segment["end_frame"] = frame_count
                        if best_confidence > current_segment["confidence"]:
                            current_segment["confidence"] = best_confidence
                    
                    last_match_frame = frame_count
                    
            except Exception as e:
                print(f"[VideoScan] Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
        
        frame_count += 1
        
        # Progress update every 500 frames
        if frame_count % 500 == 0 and total_frames > 0:
            progress = (frame_count / total_frames) * 100
            print(f"[VideoScan] Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Don't forget the last segment
    if current_segment is not None:
        results.append(current_segment)
    
    cap.release()
    print(f"[VideoScan] Scan complete. Found {len(results)} segments, {matches_found} total matches")
    return results


@app.post("/api/search_video_by_name")
async def search_video_by_name(request: Request):
    """Search for a person by name across selected videos."""
    data = await request.json()
    name = data.get("name")
    video_ids = data.get("video_ids", [])
    
    if not name or not video_ids:
        return {"status": "error", "message": "Name and video IDs required"}
    
    # Get person's encoding
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"Person '{name}' not found"}
    
    target_encoding = np.frombuffer(target[3], dtype=np.float32)
    
    # Search each video
    all_results = []
    total_segments = 0
    for vid_id in video_ids:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            total_segments += len(segments)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": name
                })
    
    # Sort by start time
    all_results.sort(key=lambda x: x["start_seconds"])
    
    return {
        "status": "success", 
        "results": all_results,
        "total_segments": total_segments,
        "videos_searched": len(video_ids)
    }


@app.post("/api/search_video_by_image")
async def search_video_by_image(file: UploadFile = File(...), video_ids: str = Form(...)):
    """Search for a person using an uploaded image across selected videos."""
    video_ids_list = json.loads(video_ids)
    
    if not video_ids_list:
        return {"status": "error", "message": "Video IDs required"}
    
    # Get encoding from uploaded image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"status": "error", "message": "No face detected in uploaded image"}
    
    # Search each video
    all_results = []
    total_segments = 0
    for vid_id in video_ids_list:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            total_segments += len(segments)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": "Unknown (from image)"
                })
    
    # Sort by start time
    all_results.sort(key=lambda x: x["start_seconds"])
    
    return {
        "status": "success", 
        "results": all_results,
        "total_segments": total_segments,
        "videos_searched": len(video_ids_list)
    }


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

def gen_frames(camera_id: str):
    last_sent_id = -1
    while True:
        with results_lock:
            data = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
            frame_id = data.get("frame_id", -1)
        if frame is None or frame_id == last_sent_id:
            continue
            
        last_sent_id = frame_id

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
