import numpy as np

class ObjectTracker:
    """IoU-based tracker optimized for all person tracking scenarios."""
    def __init__(self, max_age=3, n_init=1, iou_threshold=0.25):
        # max_age=3: Quick recovery from occlusion, but not too long to ghost
        self.max_age = max_age
        self.n_init = n_init  # Only 1 hit needed to start tracking
        self.iou_threshold = iou_threshold  # Lower threshold for better matching
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        # Face recognition cache: track_id -> face_encoding
        self.face_encodings = {}
        # Track merge history to prevent double counting same person
        self.merged_tracks = {}

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _compute_center_distance(self, box1, box2):
        """Compute distance between box centers."""
        c1_x = (box1[0] + box1[2]) / 2
        c1_y = (box1[1] + box1[3]) / 2
        c2_x = (box2[0] + box2[2]) / 2
        c2_y = (box2[1] + box2[3]) / 2
        return np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)

    def _compute_box_size(self, box):
        """Compute box area."""
        return (box[2] - box[0]) * (box[3] - box[1])

    def update(self, detections, frame=None):
        """
        Update tracker with new detections.
        Optimized for fast movement, occlusion, and various poses.
        """
        self.frame_count += 1
        
        # Convert detections from [x, y, w, h] to [x1, y1, x2, y2]
        det_boxes = []
        for det in detections:
            bbox, conf, label = det
            x, y, w, h = bbox
            det_boxes.append({
                'bbox': [float(x), float(y), float(x + w), float(y + h)],
                'conf': float(conf),
                'label': label,
                'matched': False
            })
        
        # Hungarian algorithm style matching - find best matches
        matched_track_indices = set()
        matched_det_indices = set()
        
        # First pass: match by IoU (primary matching)
        for det_idx, det in enumerate(det_boxes):
            if det_idx in matched_det_indices:
                continue
                
            best_iou = self.iou_threshold
            best_track_idx = -1
            
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_track_indices:
                    continue
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                # Update existing track
                track = self.tracks[best_track_idx]
                track['bbox'] = det['bbox']
                track['conf'] = det['conf']
                track['age'] = 0
                track['hits'] += 1
                track['last_seen'] = self.frame_count
                matched_track_indices.add(best_track_idx)
                matched_det_indices.add(det_idx)
                det['matched'] = True
        
        # Second pass: match by center distance for fast moving objects
        for det_idx, det in enumerate(det_boxes):
            if det_idx in matched_det_indices:
                continue
                
            best_dist = 2000  # pixels - larger for fast movement
            best_track_idx = -1
            
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_track_indices:
                    continue
                # Only match if track has been seen recently
                if self.frame_count - track['last_seen'] > 1:
                    continue
                dist = self._compute_center_distance(det['bbox'], track['bbox'])
                # Allow larger movement for distant/small objects
                max_dist = 300 if track.get('small', False) else 200
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                track = self.tracks[best_track_idx]
                track['bbox'] = det['bbox']
                track['conf'] = det['conf']
                track['age'] = 0
                track['hits'] += 1
                track['last_seen'] = self.frame_count
                matched_track_indices.add(best_track_idx)
                matched_det_indices.add(det_idx)
                det['matched'] = True
        
        # Age unmatched tracks
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_track_indices:
                track['age'] += 1
        
        # Create new tracks for unmatched detections
        for det in det_boxes:
            if not det['matched']:
                new_track = {
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'conf': det['conf'],
                    'label': det['label'],
                    'age': 0,
                    'hits': 1,
                    'last_seen': self.frame_count,
                    'created_at': self.frame_count,
                    'small': self._compute_box_size(det['bbox']) < 5000  # Mark small objects
                }
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        
        # Return currently matched tracks (age == 0)
        active_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.n_init and track['age'] == 0:
                active_tracks.append({
                    'id': track['id'],
                    'bbox': track['bbox']
                })
        
        return active_tracks
    
    def get_active_count(self):
        """Get count of currently detected persons."""
        return len([t for t in self.tracks if t['hits'] >= self.n_init and t['age'] == 0])
    
    def get_total_unique_count(self):
        """Get total unique persons seen (cumulative count)."""
        return self.next_id - 1
