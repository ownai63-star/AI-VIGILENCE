# AI Vigilance: Smart Multi-Camera Surveillance System

A high-performance, production-ready real-time surveillance dashboard. Built specifically for complex multi-camera tracking, AI Vigilance utilizes a multi-threaded architecture combined with low-latency network protocols to track and recognize individuals at up to 60FPS.

## 🚀 Features

- **Massive Concurrency**: Supports multiple simultaneous feeds including RTSP IP Cameras, mobile DroidCams, IP Webcam apps, and local physical Webcams.
- **Zero-Latency Feed Integration**: Explicitly utilizes UDP-forced network bindings to completely bypass video streaming buffers.
- **Anti-Ghost Tracking**: Dynamically purges tracking artifacts (ghost boxes) caused by Kalman Filter over-prediction when users exit the frame.
- **Micro-Targeted Auto-Search**: Initiate "Active Search Missions" that leverage background threads to immediately scan networks of cameras directly for a specific individual, triggering snapshots when matching biometric traits.

---

## 🧠 The AI Stack: What We Use & Why

### 1. YOLOv8 (You Only Look Once) - *Ultralytics*
- **What it does**: Primary object detection running physically on raw camera frames.
- **Why we use it**: It is the industry standard for real-time bounding box detection. We specifically restricted the model to the "Person" class to save massive computational power, avoiding analyzing the frame for irrelevant objects.

### 2. DeepSORT (Deep Simple Online and Realtime Tracking)
- **What it does**: Assigns unique mathematical IDs to people (e.g. Person 1, Person 2) and tracks their physical trajectory as they move across the video stream.
- **Why we use it**: YOLO alone cannot remember people between frames. DeepSORT solves this.
- **Our Custom Optimization**: Native DeepSORT tends to predict boxes into empty space when someone walks away. We heavily modified the integration to instantly drop tracking anchors if YOLO loses strict physical visibility of the target, eliminating "ghost boxes."

### 3. MTCNN (Multi-task Cascaded Convolutional Networks)
- **What it does**: A high-efficiency AI layer dedicated distinctly to cropping human faces.
- **Why we use it**: Running large facial recognition models over entire 1080p rooms is impossible for real-time systems. MTCNN efficiently checks only inside the boundaries of YOLO boxes, finding the precise pixel layout of the face.

### 4. FaceNet (InceptionResnetV1) via *PyTorch*
- **What it does**: Converts the tiny physical face crop from MTCNN into a 512-dimensional biometric mathematical vector (an "embedding" or "array").
- **Why we use it**: It calculates mathematical distance. When you register a photo of a person, FaceNet stores their 512D array in the database. During an "Active Search", FaceNet mathematically measures live faces against that array. If the distance is close, it announces a highly accurate match!
- **Our Custom Optimization**: To protect multi-thread scaling across 5+ cameras, we enveloped FaceNet calls in a strict native Python `Threading.Lock()`, allowing the PyTorch model to sequentially analyze crowds of people dynamically under heavy computational loads without throttling.

---

## 💻 The Backend Stack

- **FastAPI & Uvicorn**: The asynchronous Python backend. Highly robust and handles streaming thousands of MJPEG camera frames simultaneously to the dashboard without interrupting background AI calculations.
- **OpenCV (`cv2`)**: Connects to RTSP and local feeds. We applied powerful `FFMPEG` flags (`rtsp_transport;udp|fflags;nobuffer|flags;low_delay`) to rip out stream delays and ensure 100% real-time synchronized FPS.
- **SQLite3 Database**: A built-in local database that securely stores registered faces, search histories, and physical logs of every detected match locally on your machine.
- **Jinja2 + Vanilla CSS**: Dashboard templates optimized out-of-the-box for high-performance viewing and integrated with Search Engine Optimization (SEO) Open Graph tags and robot-blocking security on private logs.

## 🛠️ Setup Instructions

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Linux / Linux VM (Headless)

Run the automated setup script (installs system libs, venv, and CPU-only PyTorch):

```bash
chmod +x setup_linux.sh && ./setup_linux.sh
source .venv/bin/activate
python app.py
```

Or manually:

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg

# Create venv and install CPU-only torch (avoids ~2GB CUDA download)
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python app.py
```

> Note: `opencv-python-headless` is used in `requirements.txt` — this works on VMs without a display server. Physical webcams (index `0`, `1`, etc.) may not be available on VMs; use RTSP, DroidCam, or IP Webcam sources instead.

### 3. Access the Dashboard
Navigate to `http://localhost:8000` (or `http://<vm-ip>:8000` from another machine).

## 📁 Repository Structure
- `app.py`: Main FastAPI concurrency router and AI Thread manager.
- `cameras/`: Real-time streaming integration protocols.
- `database/`: SQLite3 manager and database tables.
- `utils/`: Core Logic scripts handling detection, tracking, tracking-snaps, and PyTorch facial locks.
- `dataset/`: Storage for biometric registration references.
- `snapshots/`: Triggered output log imagery.
