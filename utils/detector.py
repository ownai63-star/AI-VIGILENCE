import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Try to use YOLO if available, fallback to OpenCV DNN
        self.use_yolo = False
        self.use_opencv_dnn = False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.classes = [0]  # COCO class for person is 0
            self.use_yolo = True
            print("[PersonDetector] Using YOLOv8 for person detection")
        except Exception as e:
            print(f"[PersonDetector] YOLO not available: {e}")
            print("[PersonDetector] Falling back to OpenCV HOG+SVM detector")
            self._init_opencv_detector()

    def _init_opencv_detector(self):
        """Initialize OpenCV's HOG+SVM person detector as fallback"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.use_opencv_dnn = True

    def detect(self, frame):
        detections = []
        
        if self.use_yolo:
            try:
                return self._detect_yolo(frame)
            except Exception as e:
                print(f"[PersonDetector] YOLO detection failed: {e}, switching to fallback")
                self.use_yolo = False
                self._init_opencv_detector()
        
        if self.use_opencv_dnn:
            return self._detect_opencv(frame)
            
        return detections

    def _detect_yolo(self, frame):
        """YOLOv8 detection"""
        results = self.model.predict(frame, classes=self.classes, conf=0.45, imgsz=320, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                bw, bh = x2-x1, y2-y1
                if bh < 40 or bw < 10:
                    continue
                if bh / bw < 1.1 or bh / bw > 6.0:
                    continue
                    
                detections.append(([x1, y1, bw, bh], conf, 'person'))
        return detections

    def _detect_opencv(self, frame):
        """OpenCV HOG+SVM detection as fallback"""
        detections = []
        h, w = frame.shape[:2]
        
        # Resize large frames for faster processing
        scale = 1.0
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small_frame = frame
        
        # Detect people using HOG+SVM
        rects, weights = self.hog.detectMultiScale(
            small_frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
            useMeanshiftGrouping=False
        )
        
        for i, (x, y, w_rect, h_rect) in enumerate(rects):
            conf = float(weights[i]) if i < len(weights) else 0.5
            
            # Scale back to original frame size
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                w_rect = int(w_rect / scale)
                h_rect = int(h_rect / scale)
            
            # Filter by size and aspect ratio
            if h_rect < 40 or w_rect < 10:
                continue
            if h_rect / w_rect < 1.1 or h_rect / w_rect > 6.0:
                continue
            
            detections.append(([x, y, w_rect, h_rect], conf, 'person'))
        
        return detections
