import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # This will download the model automatically on first run
        self.model = YOLO(model_path)
        self.classes = [0]  # COCO class for person is 0

    def detect(self, frame):
        # Slightly increased conf to 0.45 + n_init=3 will eliminate almost all false positives
        results = self.model.predict(frame, classes=self.classes, conf=0.45, imgsz=320, verbose=False)
        detections = []
        h, w = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # [x1, y1, x2, y2], confidence, class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                bw, bh = x2-x1, y2-y1
                # Filter out boxes that are TOO small or have weird aspect ratios for a person
                # Minimum height for a person at 1080p should be at least ~50px to be relevant
                if bh < 40 or bw < 10:
                    continue
                # Person aspect ratio (h/w) is usually between 1.5 and 5.0
                if bh / bw < 1.1 or bh / bw > 6.0:
                    continue
                    
                detections.append(([x1, y1, bw, bh], conf, 'person'))
        return detections
