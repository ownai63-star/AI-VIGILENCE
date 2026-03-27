import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading

class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # MTCNN for face detection within the bbox just to be sure
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        # Resnet for generating embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.ai_lock = threading.Lock()
        
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self, db_manager):
        persons = db_manager.get_registered_persons()
        self.known_face_encodings = []
        self.known_face_names = []
        for person in persons:
            if person[3] is not None:
                # encoding is stored as blob of floats
                encoding = np.frombuffer(person[3], dtype=np.float32)
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person[1])

    def recognize(self, frame, face_bbox):
        """
        frame: full color frame
        face_bbox: [x1, y1, x2, y2] tight face bounding box
        """
        if not face_bbox:
            return "Unknown", 0.0

        fx1, fy1, fx2, fy2 = face_bbox
        face_crop = frame[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]
        
        if face_crop.size > 0:
            # Convert to RGB for Resnet processing
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(self.device)
            face_tensor = (face_tensor - 127.5) / 128.0
            
            with self.ai_lock:
                with torch.no_grad():
                    embedding = self.resnet(face_tensor).cpu().numpy()[0]
            
            if self.known_face_encodings:
                distances = np.linalg.norm(self.known_face_encodings - embedding, axis=1)
                min_idx = np.argmin(distances)
                # Lowered threshold to 1.15 for better side/profile face detection
                # Original was 1.0, increased to catch more angles
                if distances[min_idx] < 1.15:
                    name = self.known_face_names[min_idx]
                    confidence = 1 - (distances[min_idx] / 2.0)
                    return name, float(confidence)
                
        return "Unknown", 0.0

    def get_encoding(self, image):
        """
        Get encoding for registration. Image should be BGR array (from cv2.imread)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.ai_lock:
            boxes, _ = self.mtcnn.detect(image_rgb)
        
        if boxes is not None and hasattr(boxes, "__len__") and len(list(boxes)) > 0:
            fx1, fy1, fx2, fy2 = [int(b) for b in boxes[0]]
            face_crop = image_rgb[max(0, fy1):max(0, fy2), max(0, fx1):max(0, fx2)]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(self.device)
                face_tensor = (face_tensor - 127.5) / 128.0
                
                with self.ai_lock:
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor).cpu().numpy()[0]
                return embedding
        return None
