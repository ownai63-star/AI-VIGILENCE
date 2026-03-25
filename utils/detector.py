import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # This will download the model automatically on first run
        self.model = YOLO(model_path)
        self.classes = [0]  # COCO class for person is 0

    def detect(self, frame):
        # Default conf to 0.5 and default imgsz to prevent false positive tracking
        results = self.model.predict(frame, classes=self.classes, conf=0.5, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # [x1, y1, x2, y2], confidence, class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        return detections
