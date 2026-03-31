from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=30, n_init=1):
        # n_init=1: confirm a track after just 1 detection (important when YOLO runs every 3 frames)
        # max_age=30: keep a lost track for 30 frames (~1 sec) before deleting it
        # max_iou_distance=0.7: more lenient IoU matching for occluded/moving persons
        # embedder=None: disable deep embeddings (rely on IoU matching only) - avoids PyTorch issues
        self.tracker = DeepSort(
            max_age=max_age, 
            n_init=n_init, 
            max_iou_distance=0.7, 
            embedder=None,
            bgr=True
        )

    def update(self, detections, frame=None):
        """
        detections: list of ([x1, y1, w, h], confidence, label)
        """
        # Update deep_sort using internal 'mobilenet' embedder
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        active_results = []
        for track in tracks:
            # 1. Filter out unconfirmed tracks (prevents flicker/false positives)
            if not track.is_confirmed():
                continue
            
            # 2. Filter out 'ghost' tracks that haven't been seen by YOLO recently
            # With DETECTION_INTERVAL=3, a live track should have time_since_update <= 9
            if track.time_since_update > 9:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb() # Smooth Kalman predicted box
            
            active_results.append({
                "id": track_id,
                "bbox": [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            })
            
        return active_results
