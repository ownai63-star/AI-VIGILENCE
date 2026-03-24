# AI Vigilance: Multi-Camera Person Detection & Recognition

A production-ready surveillance system using YOLOv8, DeepSORT, and FaceNet.

## Features
- **Multi-Camera Support**: RTSP streams, Webcams, Mobile cameras.
- **Real-time Tracking**: DeepSORT ensures unique IDs persist across frames.
- **Face Recognition**: FaceNet (InceptionResnetV1) identifies registered individuals.
- **Automated Logging**: Saves snapshots and logs detections to a searchable SQLite database.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```
   *By default, the application will attempt to open your laptop webcam (Camera 0).*

3. **Access the Dashboard**:
   Open a browser and navigate to `http://localhost:8000`.

## Key Endpoints
- `/`: Real-time monitoring dashboard.
- `/register`: Add a person to the database for recognition.
- `/search`: Search historical detections by name and time.
- `/video_feed/{camera_id}`: MJPEG stream for external clients.

## Project Structure
- `app.py`: Main FastAPI backend.
- `cameras/`: Camera management and multi-threading.
- `models/`: AI model wrappers.
- `database/`: SQLite schema and data manager.
- `utils/`: Face recognition, detection, and tracking logic.
- `dataset/`: Storage for registered face images.
- `snapshots/`: Real-time detection event images.
