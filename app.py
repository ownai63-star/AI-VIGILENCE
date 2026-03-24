import cv2
import numpy as np
import os
import json
import base64
from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
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
from typing import Dict, Any

def sanitize_rtsp_url(url):
    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url
    # Find the last @ separating auth from host
    last_at = url.rfind("@")
    if last_at == -1:
        return url
    
    auth_part = str(url)[7:last_at] # type: ignore
    if ":" in auth_part:
        user, pwd = auth_part.split(":", 1)
        safe_pwd = urllib.parse.quote(pwd)
        return f"rtsp://{user}:{safe_pwd}{url[last_at:]}"
    return url

# Ensure necessary directories exist
os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
templates = Jinja2Templates(directory="templates")

# Initialize managers
db_manager = DatabaseManager()
detector = PersonDetector()
# tracker must be initialized per-camera, not globally!
recognizer = FaceRecognizer()
camera_manager = CameraManager()

# Load known faces from DB
recognizer.load_known_faces(db_manager)

# Global dictionary to keep track of processed data for display
camera_results: Dict[str, Any] = {} # {camera_id: {tracks: [], last_frame: frame}}
track_cache: Dict[tuple, Any] = {} # {(camera_id, track_id): {"name": str, "conf": float, "last_reid": float}}
last_logged: Dict[tuple, Any] = {} # {(camera_id, track_id): {"snapshot": str, "name": str}}

def process_camera(camera_id):
    """
    Background thread to process frames for a specific camera.
    """
    # Each camera needs its own independent stateful tracker!
    tracker = ObjectTracker()
    last_processed_id = -1
    while True:
        # Get frame with ID to avoid duplicate processing
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None or frame_id == last_processed_id:
            time.sleep(0.01)
            continue
            
        last_processed_id = frame_id
        
        try:
            # 1. Detection
            detections = detector.detect(frame)
            
            # 2. Tracking
            tracks = tracker.update(detections, frame)
            
            # 3. Recognition & Logging
            processed_tracks = []
            now = time.time()
            
            for track in tracks:
                bbox = [int(v) for v in track['bbox']]  # [x1, y1, x2, y2]
                track_id = track['id']
                cache_key = (camera_id, track_id)
                
                # Initialize cache for new track
                if cache_key not in track_cache:
                    track_cache[cache_key] = {"name": "Unknown", "conf": 0.0, "last_reid": 0}

                cached = track_cache[cache_key]
                
                # Throttle recognition to prevent massive CPU spikes and thread the execution so DeepSORT physics tracker isn't blocked
                if cached["name"] == "Unknown" and (now - float(cached["last_reid"])) > 2.0:
                    cached["last_reid"] = now
                    
                    def do_recognize(f, b, c_key, t_reid, cam_id):
                        n, c = recognizer.recognize(f, b)
                        if n != "Unknown":
                            with threading.Lock():
                                track_cache[c_key]["name"] = n
                                track_cache[c_key]["conf"] = c
                                
                            # Retroactively update database if it was already logged as Unknown
                            if c_key in last_logged:
                                prev_log = last_logged[c_key]
                                if prev_log["name"] == "Unknown":
                                    person_id = None
                                    for p in db_manager.get_registered_persons():
                                        if p[1] == n:
                                            person_id = p[0]
                                            break
                                    db_manager.update_detection_person(cam_id, prev_log["snapshot"], person_id)
                                    prev_log["name"] = n
                                    print(f"Async updated track {t_reid} to {n} on {cam_id}")

                    # run in background
                    threading.Thread(target=do_recognize, args=(frame.copy(), bbox.copy(), cache_key, track_id, camera_id), daemon=True).start()
                
                track_data = {
                    'id': track_id,
                    'bbox': bbox,
                    'name': cached["name"],
                    'confidence': cached["conf"]
                }
                processed_tracks.append(track_data)
                
                # Take EXACTLY ONE snapshot per tracking session
                if cache_key not in last_logged:
                    timestamp = int(now)
                    snapshot_name = f"snap_{camera_id}_{track_id}_{timestamp}.jpg"
                    snapshot_path = f"snapshots/{snapshot_name}"
                    
                    cv2.imwrite(snapshot_path, frame)
                    
                    # Convert cached name to DB ID
                    person_id = None
                    if cached["name"] != "Unknown":
                        for p in db_manager.get_registered_persons():
                            if p[1] == cached["name"]:
                                person_id = p[0]
                                break
                    
                    db_manager.log_detection(person_id, camera_id, snapshot_path)
                    last_logged[cache_key] = {"snapshot": snapshot_path, "name": cached["name"]}
                    print(f"Logged new track {track_id} on {camera_id}")

            with threading.Lock():
                camera_results[camera_id] = {
                    'tracks': processed_tracks,
                    'last_frame': frame.copy()
                }
        except Exception as e:
            print(f"Error in process_camera({camera_id}): {e}")
            import traceback
            traceback.print_exc()

        
        time.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cameras = camera_manager.get_active_cameras()
    return templates.TemplateResponse("index.html", {"request": request, "cameras": cameras})

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    # Save image
    img_dir = f"dataset/{name}"
    os.makedirs(img_dir, exist_ok=True)
    file_path = f"{img_dir}/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Get encoding
    image = cv2.imread(file_path)
    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        db_manager.register_person(name, file_path, encoding.tobytes())
        recognizer.load_known_faces(db_manager) # Reload
        return {"status": "success", "message": f"Person {name} registered."}
    return {"status": "error", "message": "Face not detected in image."}

@app.post("/add_camera")
async def add_camera(camera_id: str = Form(...), camera_type: str = Form(...), source: str = Form(...)):
    # Local Webcams use integers for cv2.VideoCapture
    parsed_source = source
    if camera_type == 'webcam':
        try:
            parsed_source = int(source)
        except ValueError:
            pass # allow fallback to string if they didn't enter a number
    elif camera_type == 'rtsp':
        parsed_source = sanitize_rtsp_url(source)
    elif camera_type == 'droidcam':
        if not source.startswith("http"):
            if ":" not in source:
                parsed_source = f"http://{source}:4747/video"
            else:
                parsed_source = f"http://{source}/video"
    elif camera_type == 'ipwebcam':
        if not source.startswith("http"):
            if ":" not in source:
                parsed_source = f"http://{source}:8080/video"
            else:
                parsed_source = f"http://{source}/video"
            
    if camera_manager.add_camera(camera_id, parsed_source):
        # Start background processing thread
        t = threading.Thread(target=process_camera, args=(camera_id,), daemon=True)
        t.start()
        return {"status": "success"}
    return {"status": "error"}

@app.post("/delete_camera")
async def delete_camera(camera_id: str = Form(...)):
    if camera_manager.remove_camera(camera_id):
        with threading.Lock():
            camera_results.pop(camera_id, None)
        return {"status": "success"}
    return {"status": "error"}

@app.get("/api/cameras")
async def api_cameras():
    return camera_manager.get_active_cameras()

def gen_frames(camera_id):
    while True:
        # Get raw low-latency real-time frame
        frame = camera_manager.get_camera_frame(camera_id)
        if frame is None:
            time.sleep(0.1)
            continue

        # Quickly overlay latest available AI tracks (doesn't wait for background AI loop to finish this precise frame)
        with threading.Lock():
            if camera_id in camera_results:
                data = camera_results[camera_id]
                tracks = data.get('tracks', [])
                
                for track in tracks:
                    bbox = track['bbox']
                    name = str(track['name'])
                    conf = float(track['confidence'])
                    tid = str(track['id'])
                    
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    label = f"{name} ({conf:.2f})" if name != "Unknown" else f"Person {tid}"
                    
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2) # type: ignore
                    cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # type: ignore
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.02) # Yield CPU to prevent starving threads, bounds roughly to 50fps max
        # time.sleep(0.04) # ~25 FPS

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

from typing import Optional

@app.get("/api/search")
async def api_search(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_detections(name, start_time, end_time)
    formatted = []
    for r in results:
        formatted.append({
            "id": r[0],
            "person_name": r[5] or "Unknown",
            "camera_id": r[2],
            "timestamp": r[3],
            "image_path": r[4]
        })
    return formatted

import shutil

@app.post("/clear_history")
async def clear_history():
    # Clear database detections table
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM detections')
            conn.commit()
    except Exception as e:
        print(f"Error clearing DB: {e}")

    # Delete snapshot files one-by-one (Windows-safe: skip locked files)
    snaps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    deleted = 0
    failed = 0
    if os.path.isdir(snaps_dir):
        for fname in os.listdir(snaps_dir):
            fpath = os.path.join(snaps_dir, fname)
            try:
                os.remove(fpath)
                deleted += 1
            except Exception:
                failed += 1  # File is locked, skip it
    
    # Clear in-memory state so new detections start fresh
    global last_logged, track_cache
    last_logged.clear()
    track_cache.clear()
    
    print(f"Clear history: deleted {deleted} files, skipped {failed} locked files")
    return {"status": "success", "message": f"Cleared {deleted} records"}

@app.post("/api/search_by_image")
async def search_by_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    encoding = recognizer.get_encoding(image)
    if encoding is None:
        return [] # No face found
        
    # Find matching person
    best_person_id = None
    min_dist = 1.0
    for p in db_manager.get_registered_persons():
        if p[3] is not None:
            db_enc = np.frombuffer(p[3], dtype=np.float32)
            dist = np.linalg.norm(db_enc - encoding)
            if dist < min_dist:
                min_dist = dist
                best_person_id = p[0]
                
    if best_person_id is None:
        return []
        
    # Fetch their detections
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.*, rp.name 
            FROM detections d 
            LEFT JOIN registered_persons rp ON d.person_id = rp.id
            WHERE d.person_id = ?
            ORDER BY d.timestamp DESC
        ''', (best_person_id,))
        results = cursor.fetchall()
        
    formatted = []
    for r in results:
        formatted.append({
            "id": r[0],
            "person_name": r[5] or "Unknown",
            "camera_id": r[2],
            "timestamp": r[3],
            "image_path": r[4]
        })
    return formatted

if __name__ == "__main__":
    import uvicorn
    # Default: Add laptop webcam as Camera 0
    camera_manager.add_camera("Webcam", 0)
    threading.Thread(target=process_camera, args=("Webcam",), daemon=True).start()
    
    # Default: Add the provided RTSP Camera
    rtsp_url = sanitize_rtsp_url("rtsp://test:dei@12@12@10.7.16.48:554")
    camera_manager.add_camera("RTSP_Cam", rtsp_url)
    threading.Thread(target=process_camera, args=("RTSP_Cam",), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
