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
    last_at = url.rfind("@")
    if last_at == -1:
        return url
    auth_part = url[7:last_at]
    if ":" in auth_part:
        user, pwd = auth_part.split(":", 1)
        safe_pwd = urllib.parse.quote(pwd)
        return f"rtsp://{user}:{safe_pwd}{url[last_at:]}"
    return url

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
templates = Jinja2Templates(directory="templates")

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

# Active search mission — set by /api/start_search, cleared by /api/stop_search
# {person_id, name, encoding, found_track_ids: set, running: bool}
active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Passive camera processing — ONLY detection + tracking, NO recognition
# ---------------------------------------------------------------------------

def process_camera(camera_id: str):
    """Background thread per camera: detection + tracking + recording."""
    print(f"[process_camera] Thread started for: {camera_id}")
    tracker: Any = ObjectTracker()
    last_frame_id: int = -1
    frame_count: int = 0
    track_labels: Dict[Any, tuple] = {}
    # Cache face bboxes so we don't run MTCNN every single frame
    face_bbox_cache: Dict[Any, Any] = {}
    
    processed = []  # Keep track of last known detections for skipped frames

    while True:
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.01)
            continue
        last_frame_id = frame_id
        frame_count += 1

        try:
            h, w = frame.shape[:2]

            # Run detection + tracking only every 5 frames
            if frame_count % 5 == 1:
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)

            # Prune stale labels and face cache
                active_ids = {t["id"] for t in tracks}
                track_labels   = {k: v for k, v in track_labels.items()   if k in active_ids}
                face_bbox_cache = {k: v for k, v in face_bbox_cache.items() if k in active_ids}

                new_processed = []
                run_face = (frame_count % 10 == 1)  # refresh face bbox every 10 frames (6 times a sec)

                for t in tracks:
                    # Clamp body bbox strictly to frame
                    bx1 = max(0, int(t["bbox"][0]))
                    by1 = max(0, int(t["bbox"][1]))
                    bx2 = min(w - 1, int(t["bbox"][2]))
                    by2 = min(h - 1, int(t["bbox"][3]))

                    # Skip degenerate or full-frame boxes
                    box_w = bx2 - bx1
                    box_h = by2 - by1
                    if box_w < 20 or box_h < 20:
                        continue
                    if box_w > w * 0.95 or box_h > h * 0.95:
                        continue

                    bbox = [bx1, by1, bx2, by2]

                    # Run MTCNN face detection periodically
                    if run_face:
                        face_bbox = None
                        try:
                            crop = frame[by1:by2, bx1:bx2]
                            if crop.size > 0:
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                boxes, _ = recognizer.mtcnn.detect(crop_rgb)
                                if isinstance(boxes, np.ndarray) and len(boxes) > 0:
                                    fx1, fy1, fx2, fy2 = boxes[0]
                                    face_bbox = [
                                        max(0,     bx1 + int(fx1)),
                                        max(0,     by1 + int(fy1)),
                                        min(w - 1, bx1 + int(fx2)),
                                        min(h - 1, by1 + int(fy2))
                                    ]
                                    # Sanity: face must be smaller than body box
                                    fw = face_bbox[2] - face_bbox[0]
                                    fh = face_bbox[3] - face_bbox[1]
                                    if fw < 10 or fh < 10 or fw > box_w or fh > box_h:
                                        face_bbox = None
                        except Exception:
                            face_bbox = None
                        face_bbox_cache[t["id"]] = face_bbox
                    else:
                        face_bbox = face_bbox_cache.get(t["id"])

                    name, conf = track_labels.get(t["id"], ("Unknown", 0.0))
                    new_processed.append({
                        "id": t["id"],
                        "bbox": bbox,
                        "face_bbox": face_bbox,
                        "name": name,
                        "confidence": conf
                    })

            processed = new_processed

            with active_search_lock:
                search = dict(active_search)

            if search.get("running") and run_face:
                for t in processed:
                    track_key = (camera_id, t["id"])
                    if track_key not in search.get("found_track_ids", set()):
                        if t.get("face_bbox") is not None:
                            threading.Thread(
                                target=recognition_worker,
                                args=(frame.copy(), t["face_bbox"], t["id"], camera_id, track_labels),
                                daemon=True
                            ).start()

            # Overlay rendering MUST happen for every frame to provide perfectly synced stream
            record_frame = frame.copy()
            for t in processed:
                bx1, by1, bx2, by2 = t["bbox"]
                face_bbox = t.get("face_bbox")
                name = str(t["name"])
                conf = float(t["confidence"])
                tid = str(t["id"])

                body_color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                label = f"{name} ({conf:.2f})" if name != "Unknown" else f"Person {tid}"
                cv2.rectangle(record_frame, (bx1, by1), (bx2, by2), body_color, 2)
                cv2.putText(record_frame, label, (bx1, max(by1 - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, body_color, 2)

                if face_bbox:
                    cv2.rectangle(record_frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 255, 0), 2)
                    cv2.putText(record_frame, "face", (face_bbox[0], max(face_bbox[1] - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Output perfectly synchronized frame to the live feed
            with results_lock:
                camera_results[camera_id] = {"rendered_frame": record_frame, "frame_id": frame_id, "tracks": processed}

        except Exception as e:
            print(f"[process_camera:{camera_id}] {e}")
            import traceback; traceback.print_exc()

        # Removed forced 0.01s sleep to allow maximum unrestricted 60FPS loop speed


def recognition_worker(frame, face_bbox, track_id, camera_id, track_labels):
    """
    Background recognition for active search using exact face bbox.
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

    # Run face recognition on the perfectly snapped face box
    name, confidence = recognizer.recognize(frame, face_bbox)

    if name == target_name and confidence > 0.4:
        with active_search_lock:
            if not active_search.get("running"):
                return
            if track_key in active_search.get("found_track_ids", set()):
                return
            active_search["found_track_ids"].add(track_key)

        # Persist the label so it survives frame resets
        track_labels[track_id] = (target_name, confidence)

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
    cameras = camera_manager.get_active_cameras()
    return templates.TemplateResponse("index.html", {"request": request, "cameras": cameras})


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

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
                deleted += 1
            except Exception:
                pass

    print(f"Cleared {deleted} snapshots.")
    return {"status": "success", "message": f"Cleared {deleted} records"}


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
            time.sleep(0.005)  # 5ms check loop
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
    camera_manager.add_camera("Webcam", 0)
    threading.Thread(target=process_camera, args=("Webcam",), daemon=True).start()

    rtsp_url = sanitize_rtsp_url("rtsp://test:dei@12@12@10.7.16.48:554")
    camera_manager.add_camera("RTSP_Cam", rtsp_url)
    threading.Thread(target=process_camera, args=("RTSP_Cam",), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
