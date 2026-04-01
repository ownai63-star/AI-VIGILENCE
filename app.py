import cv2
import numpy as np
import os
import shutil
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from database.db_manager import DatabaseManager
from utils.detector import PersonDetector
from utils.tracker import ObjectTracker
from utils.recognizer import FaceRecognizer
from cameras.camera_manager import CameraManager
import threading
import time
from typing import Dict, Any, Optional

# Security setup
security = HTTPBasic(auto_error=False)

# Simple admin credentials (in production, use database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Session storage (in production, use proper session management)
authenticated_sessions: set = set()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials."""
    if credentials:
        is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if is_correct_username and is_correct_password:
            return credentials.username
    return None

def require_auth(request: Request):
    """Check if user is authenticated via session cookie."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        return True
    return False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_rtsp_url(url: str) -> str:
    """Percent-encode special characters in the password portion of an RTSP URL.
    Handles passwords containing multiple '@' signs by using rfind to locate the
    last '@' as the user:pass / host boundary.
    """
    if not isinstance(url, str):
        return url
    url = url.strip()
    if not url.startswith("rtsp://"):
        return url

    # Everything after rtsp://
    rest = url[7:]
    last_at = rest.rfind("@")
    if last_at == -1:
        return url  # No auth in URL

    auth_part = rest[:last_at]       # e.g. "test:dei@12@12"
    host_part = rest[last_at + 1:]   # e.g. "10.7.16.48:554"

    colon = auth_part.find(":")
    if colon == -1:
        return url  # No password, nothing to encode

    user = auth_part[:colon]
    pwd  = auth_part[colon + 1:]     # e.g. "dei@12@12"

    # Encode only '@' in the password — FFmpeg requires this
    safe_pwd = pwd.replace("@", "%40")

    return f"rtsp://{user}:{safe_pwd}@{host_part}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reload all saved cameras from the database on startup."""
    print("[Startup] Loading persistent cameras from database...")
    cameras = db_manager.get_cameras()
    for cam_id, source in cameras:
        # Handle webcam IDs stored as strings
        parsed_source = source
        if str(source).isdigit():
            parsed_source = int(source)
        
        if camera_manager.add_camera(cam_id, parsed_source):
             threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
             print(f"[Startup] Restored camera: {cam_id}")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

# Configure Jinja2 templates with cache disabled to avoid unhashable type error
templates = Jinja2Templates(directory="templates")
templates.env.cache_size = 0

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

# Per-camera: recognized persons info
camera_recognized_persons: Dict[str, Dict[int, str]] = {}
recognized_lock = threading.Lock()

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
    Process exactly 2 FPS for high accuracy with reduced system load.
    """
    print(f"[Camera:{camera_id}] Processing thread started (2 FPS mode)")
    
    # Wait for camera to be ready
    warmup_frames = 0
    while warmup_frames < 5:
        frame, _ = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None:
            warmup_frames += 1
        time.sleep(0.1)
    print(f"[Camera:{camera_id}] Camera ready - Processing at 2 FPS")
    
    # Improved tracker: immediate tracking (n_init=1), quick recovery, low IoU threshold
    tracker: ObjectTracker = ObjectTracker(max_age=3, n_init=1, iou_threshold=0.25)
    last_frame_id: int = -1
    frame_count: int = 0
    
    # Process exactly 2 frames per second
    FRAME_INTERVAL: float = 0.5  # 500ms = 2 FPS
    
    # Recognition runs on EVERY frame (2 FPS) for maximum accuracy
    
    # Recognition cache: track_id -> (name, confidence, frame_number)
    RECOGNITION_CACHE_FRAMES: int = 4  # Cache valid for 4 frames (~2 seconds at 2 FPS)
    recognition_cache: Dict[Any, tuple] = {}
    
    # Track IDs currently in frame (to prevent double counting)
    current_frame_track_ids: set = set()
    
    # Face encoding cache for deduplication: track_id -> encoding
    face_encoding_cache: Dict[int, np.ndarray] = {}
    # Track merge map: old_id -> new_id (for deduplication)
    track_merge_map: Dict[int, int] = {}
    last_process_time: float = 0

    while True:
        # Wait for next 2 FPS interval
        current_time = time.time()
        elapsed = current_time - last_process_time
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)
        
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None:
            continue
            
        # Get latest frame (may skip some camera frames to maintain 2 FPS)
        last_frame_id = frame_id
        frame_count += 1
        last_process_time = time.time()

        try:
            h, w = frame.shape[:2]
            
            # Run detection on EVERY frame (2 FPS) for high accuracy
            detections = detector.detect(frame)
            
            # Run recognition on EVERY frame (2 FPS) - no skip
            
            # Update tracker
            tracks = tracker.update(detections, frame)
            
            # Build current frame track IDs for anti-double-counting
            new_track_ids = set(t["id"] for t in tracks)
            
            # Log count on every frame at 2 FPS
            if len(new_track_ids) != len(current_frame_track_ids):
                print(f"[Camera:{camera_id}] Persons: {len(tracks)}")
            current_frame_track_ids = new_track_ids

            # Build processed tracks with cached recognition
            processed = []
            for t in tracks:
                track_id = t["id"]
                bbox = t["bbox"]
                
                # Check recognition cache
                name, conf = "Unknown", 0.0
                if track_id in recognition_cache:
                    cached_name, cached_conf, cached_frame = recognition_cache[track_id]
                    if (frame_count - cached_frame) < RECOGNITION_CACHE_FRAMES:
                        name, conf = cached_name, cached_conf

                processed.append({
                    "id": track_id,
                    "bbox": bbox,
                    "name": name,
                    "confidence": conf,
                    "stable": True
                })

            # Non-Maximum Suppression (Overlapping Box Kill)
            # If two boxes overlap > 70%, remove the newer one to fix 'Double Boxes'
            final_processed = []
            processed = sorted(processed, key=lambda x: x["id"])
            for i, p1 in enumerate(processed):
                keep = True
                for j, p2 in enumerate(final_processed):
                    # Simple IOU check
                    box1, box2 = p1["bbox"], p2["bbox"]
                    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
                    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
                    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.7:
                        keep = False
                        break
                if keep:
                    final_processed.append(p1)
            processed = final_processed

            # Face recognition logic - runs on EVERY frame (2 FPS)
            for t in processed:
                tid = t["id"]
                # Skip if cache is still valid
                if tid in recognition_cache and (frame_count - recognition_cache[tid][2]) < RECOGNITION_CACHE_FRAMES:
                    continue
                
                # Offload heavy biometric check to background thread
                bx1, by1, bx2, by2 = int(t["bbox"][0]), int(t["bbox"][1]), int(t["bbox"][2]), int(t["bbox"][3])
                bw, bh = max(0, bx2 - bx1), max(0, by2 - by1)
                if bw < 30 or bh < 30: continue
                
                face_box = [bx1 + int(0.15 * bw), by1, bx2 - int(0.15 * bw), by1 + int(0.45 * bh)]
                threading.Thread(
                    target=self_recognition_worker,
                    args=(frame.copy(), face_box, tid, recognition_cache, frame_count, face_encoding_cache, track_merge_map),
                    daemon=True
                ).start()

            # Active search check - also runs every frame
            with active_search_lock:
                search = dict(active_search)

            if search.get("running"):
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
            people_count = len(processed)
            
            # Generate distinct colors for each person ID
            def get_person_color(pid):
                # Use HSV color space for distinct colors
                hue = (pid * 137) % 180  # Golden angle approximation for good distribution
                import cv2
                hsv = np.uint8([[[hue, 255, 255]]])
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                return tuple(int(c) for c in rgb)
            
            for t in processed:
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                name = str(t["name"])
                conf = float(t["confidence"])
                tid = int(t["id"])

                if name != "Unknown":
                    body_color = (0, 255, 0)  # Green for recognized
                    label = f"{name}"
                else:
                    # Distinct color per person ID
                    body_color = get_person_color(tid)
                    # Only show ID for crowd management (no "Person" prefix to save space)
                    label = f"#{tid}"
                
                # Ensure box is within frame bounds
                h, w = record_frame.shape[:2]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w-1, bx2), min(h-1, by2)
                
                # Draw bounding box with thickness based on confidence
                thickness = 2 if name == "Unknown" else 3
                cv2.rectangle(record_frame, (bx1, by1), (bx2, by2), body_color, thickness)
                
                # Draw label background for readability
                label_y = max(by1 - 5, 20)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(record_frame, (bx1, label_y - text_h - 4), (bx1 + text_w, label_y + 4), (0, 0, 0), -1)
                cv2.putText(record_frame, label, (bx1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_color, 2)
            
            # Log occupancy if changed
            if occupancy_last_count.get(camera_id) != people_count:
                occupancy_last_count[camera_id] = people_count
                try:
                    db_manager.log_occupancy(camera_id, people_count)
                    
                    # Save snapshot with bounding boxes when count changes
                    if people_count > 0:
                        snapshot_dir = f"snapshots/{camera_id}"
                        os.makedirs(snapshot_dir, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        snapshot_path = f"{snapshot_dir}/snapshot_{timestamp}.jpg"
                        
                        # Save bbox data as JSON
                        import json
                        bbox_data = json.dumps([{
                            "id": t["id"],
                            "bbox": t["bbox"],
                            "name": t["name"]
                        } for t in processed])
                        
                        cv2.imwrite(snapshot_path, record_frame)
                        db_manager.log_detection_snapshot(camera_id, people_count, snapshot_path, bbox_data)
                except Exception as e:
                    print(f"[Camera:{camera_id}] Snapshot error: {e}")
            
            # Display count - only currently detected persons
            count_text = f"Persons: {people_count}"
            cv2.putText(record_frame, count_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Output at full frame rate
            with results_lock:
                camera_results[camera_id] = {"rendered_frame": record_frame, "frame_id": frame_id, "tracks": processed}
            
            # Store recognized persons for API
            with recognized_lock:
                recognized_dict = {}
                for t in processed:
                    if t["name"] != "Unknown":
                        recognized_dict[t["id"]] = t["name"]
                camera_recognized_persons[camera_id] = recognized_dict

            # Write to recording
            with writer_lock:
                writer_data = camera_writers.get(camera_id)
                if writer_data and writer_data.get("writer"):
                    writer_data["writer"].write(record_frame)
            
            # No frame rate limiting - run as fast as possible for smooth video

        except Exception as e:
            print(f"[Camera:{camera_id}] Error: {e}")
            import traceback; traceback.print_exc()


def self_recognition_worker(frame, face_box, track_id, recognition_cache, frame_count, face_encoding_cache, track_merge_map):
    """Background task for periodic biometric verification with deduplication."""
    try:
        name, conf, face_encoding = recognizer.recognize_with_encoding(frame, face_box)
        
        # Store face encoding for deduplication
        if face_encoding is not None:
            face_encoding_cache[track_id] = face_encoding
            
            # Check for duplicate tracks (same person, different track ID)
            for other_id, other_encoding in face_encoding_cache.items():
                if other_id != track_id:
                    # Compare face encodings
                    distance = np.linalg.norm(face_encoding - other_encoding)
                    if distance < 0.6:  # Same person
                        # Merge tracks - use lower ID
                        if track_id < other_id:
                            track_merge_map[other_id] = track_id
                        else:
                            track_merge_map[track_id] = other_id
                        break
        
        if name != "Unknown" and conf > 0.40:  # Higher confidence threshold
            recognition_cache[track_id] = (name, conf, frame_count)
    except Exception:
        pass


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
    # Check authentication
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    # New dynamic camera grid - cameras loaded via JavaScript
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})

@app.post("/api/login")
async def api_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        import uuid
        session_token = str(uuid.uuid4())
        authenticated_sessions.add(session_token)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session", value=session_token, httponly=True)
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        authenticated_sessions.discard(session_token)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session")
    return response


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "search.html", {})

@app.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "recordings.html", {})

@app.get("/detection_logs", response_class=HTMLResponse)
async def detection_logs_page(request: Request, camera_id: Optional[str] = None):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "detection_logs.html", {"camera_id": camera_id})

@app.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "people.html", {})

@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "cameras.html", {})
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
    elif camera_type == "mjpeg":
        # Direct MJPEG HTTP stream — pass as-is
        parsed = source.strip()

    if camera_manager.add_camera(camera_id, parsed):
        db_manager.add_camera_to_db(camera_id, parsed)
        threading.Thread(target=process_camera, args=(camera_id,), daemon=True).start()
        return {"status": "success"}
    return {"status": "error", "message": "Camera already exists or could not connect."}


@app.post("/delete_camera")
async def delete_camera(camera_id: str = Form(...)):
    if camera_manager.remove_camera(camera_id):
        db_manager.remove_camera_from_db(camera_id)
        camera_results.pop(camera_id, None)
        return {"status": "success"}
    return {"status": "error"}


@app.get("/api/cameras")
async def api_cameras():
    """Get all active cameras with their source info."""
    cameras = []
    for cam_id in camera_manager.get_active_cameras():
        # Get camera source from database
        cam_info = {"id": cam_id, "source": "Unknown"}
        try:
            db_cams = db_manager.get_cameras()
            for db_cam in db_cams:
                if db_cam[0] == cam_id:
                    cam_info["source"] = db_cam[1] if len(db_cam) > 1 else "Local"
                    break
        except:
            pass
        cameras.append(cam_info)
    return cameras

@app.get("/api/recognized/{camera_id}")
async def api_recognized_persons(camera_id: str):
    """Get recognized persons for a specific camera."""
    with recognized_lock:
        persons = camera_recognized_persons.get(camera_id, {})
        return [{"track_id": tid, "name": name} for tid, name in persons.items()]

@app.get("/api/occupancy")
async def api_occupancy(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    """Get occupancy data - either current counts or historical."""
    # If no time range specified, return current live counts
    if not start_time and not end_time:
        results = []
        for cam_id in camera_manager.get_active_cameras():
            if camera_id and cam_id != camera_id:
                continue
            count = occupancy_last_count.get(cam_id, 0)
            results.append({
                "id": cam_id,
                "camera_id": cam_id,
                "timestamp": int(time.time()),
                "count": count
            })
        return results
    
    # Historical data query
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
            timestamp = int(time.time())
            # Try VP80/WebM first (Windows), fall back to mp4v/MP4 (Linux headless)
            fourcc_vp80 = cv2.VideoWriter_fourcc(*'VP80')
            test_path = f"recordings/rec_{camera_id}_{timestamp}.webm"
            test_writer = cv2.VideoWriter(test_path, fourcc_vp80, 20.0, (w, h))
            if test_writer.isOpened():
                fourcc = fourcc_vp80
                filename = f"rec_{camera_id}_{timestamp}.webm"
                file_path = test_path
                writer = test_writer
            else:
                test_writer.release()
                try: os.remove(test_path)
                except Exception: pass
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                filename = f"rec_{camera_id}_{timestamp}.mp4"
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
# Camera Recording Settings API
# ---------------------------------------------------------------------------

@app.get("/api/camera_settings/{camera_id}")
async def get_camera_settings(camera_id: str):
    """Get recording settings for a camera."""
    db_setting = db_manager.get_camera_recording_setting(camera_id)
    # Also check if actually recording
    with writer_lock:
        actually_recording = camera_id in camera_writers
    return {"camera_id": camera_id, "recording_enabled": bool(db_setting), "actually_recording": actually_recording}

@app.post("/api/camera_settings/{camera_id}")
async def set_camera_settings(camera_id: str, enabled: bool = Form(...)):
    """Set recording settings for a camera and start/stop actual recording."""
    # Save setting to database
    db_manager.set_camera_recording(camera_id, enabled)
    
    # Actually start/stop the recording
    with writer_lock:
        if enabled:
            # Start recording if not already recording
            if camera_id not in camera_writers:
                # Get frame dimensions from camera results
                with results_lock:
                    data = camera_results.get(camera_id, {})
                    frame = data.get("rendered_frame")
                    if frame is None:
                        return {"status": "error", "message": "Camera not streaming"}
                    h, w = frame.shape[:2]
                
                # Setup video writer
                os.makedirs("recordings", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = f"recordings/{camera_id}_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(file_path, fourcc, 2.0, (w, h))  # 2 FPS to match processing
                
                if writer.isOpened():
                    db_id = db_manager.start_recording(camera_id, file_path)
                    camera_writers[camera_id] = {"writer": writer, "db_id": db_id}
                    print(f"[Recording] Started recording {camera_id} to {file_path}")
                else:
                    return {"status": "error", "message": "Failed to start video writer"}
        else:
            # Stop recording if currently recording
            if camera_id in camera_writers:
                writer_data = camera_writers.pop(camera_id)
                writer_data["writer"].release()
                db_manager.end_recording(writer_data["db_id"])
                print(f"[Recording] Stopped recording {camera_id}")
    
    return {"status": "success", "camera_id": camera_id, "recording_enabled": enabled}

# ---------------------------------------------------------------------------
# Detection Snapshots API
# ---------------------------------------------------------------------------

@app.get("/api/detection_snapshots")
async def get_detection_snapshots(camera_id: Optional[str] = None, limit: int = 100):
    """Get detection snapshots with bounding boxes."""
    snapshots = db_manager.get_detection_snapshots(camera_id=camera_id, limit=limit)
    return [
        {
            "id": s[0],
            "camera_id": s[1],
            "timestamp": s[2],
            "person_count": s[3],
            "snapshot_path": s[4],
            "bbox_data": s[5]
        }
        for s in snapshots
    ]

@app.get("/api/snapshot/{snapshot_id}")
async def get_snapshot(snapshot_id: int):
    """Get a specific snapshot with bounding box data."""
    snapshot = db_manager.get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {
        "id": snapshot[0],
        "camera_id": snapshot[1],
        "timestamp": snapshot[2],
        "person_count": snapshot[3],
        "snapshot_path": snapshot[4],
        "bbox_data": snapshot[5]
    }

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
    """Generate MJPEG stream at 2 FPS matching processing rate."""
    import cv2
    import time
    
    last_sent_id = -1
    last_send_time = 0
    FRAME_INTERVAL = 0.5  # 2 FPS to match processing
    
    while True:
        with results_lock:
            data = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
            frame_id = data.get("frame_id", -1)
        
        # Skip if no frame
        if frame is None:
            time.sleep(0.05)
            continue
        
        # Rate limit to 2 FPS
        current_time = time.time()
        if current_time - last_send_time < FRAME_INTERVAL:
            time.sleep(0.05)
            continue
        
        # Send latest frame even if not new (maintains 2 FPS stream)
        last_sent_id = frame_id
        last_send_time = current_time

        # Resize for streaming
        h, w = frame.shape[:2]
        target_w = 1280
        if w > target_w:
            scale = target_w / w
            target_h = int(h * scale)
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # JPEG encoding
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 75,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
        ]
        
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
               b"\r\n" + frame_bytes + b"\r\n")


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
